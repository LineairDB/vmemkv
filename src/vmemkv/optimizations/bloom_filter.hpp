#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace t1_detail {

inline constexpr std::size_t kBloomFilterWordBits = 64;
inline constexpr std::size_t kBloomBitsPerEntry = 8;
inline constexpr uint64_t kMixShift = 33;
inline constexpr uint64_t kMixMul1 = 0xff51afd7ed558ccdULL;
inline constexpr uint64_t kMixMul2 = 0xc4ceb9fe1a85ec53ULL;

class BloomFilter {
 public:
  explicit BloomFilter(size_t entry_count = 0) { reset(entry_count); }

  void reset(size_t entry_count) {
    if (entry_count == 0) {
      words_.clear();
      mask_bits_ = 0;
      return;
    }
    const size_t bits = std::bit_ceil(std::max<size_t>(kBloomFilterWordBits, entry_count * kBloomBitsPerEntry));
    words_.assign(bits / kBloomFilterWordBits, 0);
    mask_bits_ = bits - 1;
  }

  void add(uint64_t hash) noexcept {
    if (words_.empty()) {
      return;
    }
    set(bit_index(hash));
    set(bit_index(mix(hash)));
    set(bit_index(mix(mix(hash))));
  }

  [[nodiscard]] auto maybe_contains(uint64_t hash) const noexcept -> bool {
    if (words_.empty()) {
      return true;
    }
    return test(bit_index(hash)) && test(bit_index(mix(hash))) && test(bit_index(mix(mix(hash))));
  }

 private:
  static auto mix(uint64_t hash_value) noexcept -> uint64_t {
    hash_value ^= hash_value >> kMixShift;
    hash_value *= kMixMul1;
    hash_value ^= hash_value >> kMixShift;
    hash_value *= kMixMul2;
    hash_value ^= hash_value >> kMixShift;
    return hash_value;
  }

  [[nodiscard]] auto bit_index(uint64_t hash) const noexcept -> size_t {
    return static_cast<size_t>(hash) & mask_bits_;
  }

  void set(size_t index) noexcept {
    words_[index / kBloomFilterWordBits] |= (uint64_t{1} << (index % kBloomFilterWordBits));
  }

  [[nodiscard]] auto test(size_t index) const noexcept -> bool {
    return (words_[index / kBloomFilterWordBits] & (uint64_t{1} << (index % kBloomFilterWordBits))) != 0;
  }

  std::vector<uint64_t> words_;
  size_t mask_bits_ = 0;
};

}  // namespace t1_detail
