// bench_helpers.hpp - Shared key/value/touch/populate/Zipf/label helpers for bench_kv.cpp.
// Pure standard-library code plus core/env.hpp; no backend or Google Benchmark
// dependency, so every tool links it header-only.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace bench_help {

inline constexpr std::size_t kInlineValueBytes = 8;
inline constexpr std::size_t kIndexKeyBytes = 16;
// For a non-8B value size, 1-in-kMixEveryNth keys get an 8-byte (inline-eligible) value instead.
inline constexpr std::size_t kMixEveryNth = 5;
inline constexpr uint64_t kBenchmarkSeed = 42;

// ikey() is the formatting helper behind make_key(): single canonical "k%015zx"
// spelling shared by every bulk_load/scan/get/update/remove call site.
inline auto ikey(std::size_t index) -> std::string {
  std::array<char, 32> key_buffer{};
  std::snprintf(key_buffer.data(), key_buffer.size(), "k%015zx", index);
  return key_buffer.data();
}

// Canonical benchmark key for `index`; keep call sites on make_key() so the
// spelling stays single-sourced if the format ever changes.
inline auto make_key(std::size_t index) -> std::string { return ikey(index); }

inline auto get_value_size_for_key(std::size_t index, std::size_t target_size) -> std::size_t {
  if (target_size == kInlineValueBytes) {
    return kInlineValueBytes;
  }
  if (index % kMixEveryNth == 0) {
    return kInlineValueBytes;
  }
  return target_size;
}

// Pseudo-random bytes (xorshift64 seeded from `index`) rather than a repeated character: repeated
// bytes are trivially compressible and skew any benchmark involving a compression layer (e.g.
// zswap), while xorshift64 output stays deterministic across repeated populate() calls.
inline auto make_value_for_key(std::size_t index, std::size_t target_size) -> std::string {
  const std::size_t size = get_value_size_for_key(index, target_size);
  std::string value(size, '\0');
  uint64_t state = static_cast<uint64_t>(index) * 0x9E3779B97F4A7C15ULL + 1;
  for (std::size_t offset = 0; offset < size; offset += sizeof(uint64_t)) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    std::memcpy(value.data() + offset, &state, std::min(sizeof(uint64_t), size - offset));
  }
  return value;
}

// Folds every byte of `value` into a checksum so DoNotOptimize() forces the whole span to be
// read (paged in from swap if needed). Touching only pointer/length would fault just the first
// page and silently skip large values' real cost.
inline auto touch_bytes(std::span<const std::byte> value) noexcept -> uint64_t {
  uint64_t checksum = 0;
  for (std::byte b : value) {
    checksum = (checksum * 131) + static_cast<uint8_t>(b);
  }
  return checksum;
}

struct PopulateOptions {
  size_t key_count;
  size_t value_size = kInlineValueBytes;
};

// Fixed-seed permutation of [0, n): shuffled insert order over the same logical key set.
// Shared by populate(shuffled) below and shuffled-master construction, so every shuffled
// build of the same corpus is byte-identical.
inline auto shuffled_order(std::size_t n) -> std::vector<std::size_t> {
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::mt19937_64 rng(kBenchmarkSeed);
  std::shuffle(order.begin(), order.end(), rng);
  return order;
}

// Populates via the uniform bulk_load() entry point every backend implements. `shuffled`
// inserts the same logical key set in fixed-seed-shuffled order, so the T2 layout ends up
// genuinely scattered relative to key order instead of key-ordered (bulk_load appends in call
// order). Deterministic, so repeated builds of the same master are byte-identical.
template <typename Store>
inline void populate(Store &store, PopulateOptions options, bool shuffled = false) {
  if (!shuffled) {
    store.bulk_load(
        options.key_count,
        [](std::size_t index) { return make_key(index); },
        [value_size = options.value_size](std::size_t index) { return make_value_for_key(index, value_size); });
    return;
  }
  const auto order = std::make_shared<std::vector<std::size_t>>(shuffled_order(options.key_count));
  store.bulk_load(
      options.key_count,
      [order](std::size_t i) { return make_key((*order)[i]); },
      [order, value_size = options.value_size](std::size_t i) { return make_value_for_key((*order)[i], value_size); });
}

// ---- Zipf distribution -------------------------------------------------------

class ZipfDistribution {
 public:
  struct Params {
    std::size_t item_count;
    double alpha;
  };

  explicit ZipfDistribution(Params params)
      : item_count_(params.item_count),
        alpha_(params.alpha),
        h_integral_x1_(h_integral(kZipfPivot) - 1.0),
        h_integral_inf_(h_integral(static_cast<double>(params.item_count) + kHalfStep)),
        s_(1.0 - h_integral_inv(h_integral(kZipfPivot) - 1.0)) {}

  auto operator()(std::mt19937_64 &rng) -> std::size_t {
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    while (true) {
      double uniform_sample = h_integral_inf_ + uni(rng) * (h_integral_x1_ - h_integral_inf_);
      double sample_position = h_integral_inv(uniform_sample);
      std::size_t rank = static_cast<std::size_t>(std::llround(sample_position));
      if (rank < 1) rank = 1;
      if (rank > item_count_) rank = item_count_;
      if (rank - sample_position <= s_ ||
          uniform_sample >= h_integral(static_cast<double>(rank) + kHalfStep) - h(rank)) {
        return rank - 1;
      }
    }
  }

 private:
  static constexpr double kZipfPivot = 1.5;
  static constexpr double kHalfStep = 0.5;

  std::size_t item_count_;
  double alpha_, h_integral_x1_, h_integral_inf_, s_;
  [[nodiscard]] auto h(double val) const -> double { return std::exp(-alpha_ * std::log(val)); }
  [[nodiscard]] auto h_integral(double val) const -> double {
    return (alpha_ == 1.0) ? std::log(val) : std::exp((1.0 - alpha_) * std::log(val)) / (1.0 - alpha_);
  }
  [[nodiscard]] auto h_integral_inv(double val) const -> double {
    return (alpha_ == 1.0) ? std::exp(val) : std::exp(std::log(val * (1.0 - alpha_)) / (1.0 - alpha_));
  }
};

// ---- Benchmark naming vocabulary ---------------------------------------------

// Rival backends report a bare name (no "VMemKV/" prefix); only vmemkv configurations use it,
// so a new rival needs no string-matching change here.
inline auto store_label(std::string_view store_name) -> std::string {
  constexpr std::string_view kVMemKVPrefix = "VMemKV/";
  return store_name.rfind(kVMemKVPrefix, 0) == 0 ? "VMemKV" : std::string(store_name);
}

inline auto variant_label(std::string_view store_name) -> std::string {
  constexpr std::string_view kPrefix = "VMemKV/";
  if (store_name.rfind(kPrefix, 0) != 0) {
    return std::string(store_name);
  }
  std::string variant = std::string(store_name.substr(kPrefix.size()));
  std::replace(variant.begin(), variant.end(), '/', '-');
  return variant;
}

inline auto value_label(size_t value_size) -> std::string {
  if (value_size == kInlineValueBytes) {
    return "8B";
  }
  if (value_size == 1024ULL) {
    return "1KB(20% 8B)";
  }
  if (value_size == 64ULL * 1024ULL) {
    return "64KB(20% 8B)";
  }
  return std::to_string(value_size) + "B(20% 8B)";
}

inline auto benchmark_name(std::string_view store_name,
                           std::string_view op,
                           std::optional<std::string_view> mode = std::nullopt,
                           std::optional<std::string_view> dist = std::nullopt,
                           std::optional<std::string_view> value = std::nullopt) -> std::string {
  std::string name;
  name.reserve(128);
  name += "Store=";
  name += store_label(store_name);
  name += "/Variant=";
  name += variant_label(store_name);
  name += "/Op=";
  name += op;
  if (mode.has_value()) {
    name += "/Mode=";
    name += *mode;
  }
  if (dist.has_value()) {
    name += "/Dist=";
    name += *dist;
  }
  if (value.has_value()) {
    name += "/Value=";
    name += *value;
  }
  return name;
}

struct BenchmarkMetadata {
  std::size_t corpus_keys;
  std::size_t corpus_bytes;
};

inline auto make_metadata(std::size_t corpus_keys, std::size_t value_bytes) -> BenchmarkMetadata {
  return BenchmarkMetadata{
      corpus_keys,
      corpus_keys * (kIndexKeyBytes + value_bytes),
  };
}

}  // namespace bench_help
