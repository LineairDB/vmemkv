// utils.hpp - Internal mathematical and layout helpers for vmemkv.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <span>
#include <system_error>

namespace vmemkv {

inline constexpr uint64_t kDefaultAlignmentBytes = 8;

struct EmptyOption {};
static_assert(sizeof(EmptyOption) <= 1, "EmptyOption must be a zero-sized or minimal size empty helper");

constexpr auto align_up(uint64_t size, uint64_t alignment = kDefaultAlignmentBytes) noexcept -> uint64_t {
  return (size + alignment - 1) & ~(alignment - 1);
}

// Aligned byte length of one record: header plus key plus value, rounded up.
constexpr auto record_aligned_len(uint64_t header_bytes,
                                  uint64_t key_bytes,
                                  uint64_t value_bytes,
                                  uint64_t alignment = kDefaultAlignmentBytes) noexcept -> uint64_t {
  return align_up(header_bytes + key_bytes + value_bytes, alignment);
}

// FNV-1a64: fast non-cryptographic checksum for on-disk formats (torn writes / bit rot).
// Not for security use.
inline constexpr uint64_t kFnvOffsetBasis64 = 14695981039346656037ULL;
inline constexpr uint64_t kFnvPrime64 = 1099511628211ULL;

// NOLINTBEGIN(readability-magic-numbers,readability-identifier-length)
inline auto fnv1a64_update(uint64_t hash, const void *data, size_t size) noexcept -> uint64_t {
  const auto *bytes = static_cast<const unsigned char *>(data);
  size_t idx = 0;
  for (; idx + 8 <= size; idx += 8) {
    hash = (hash ^ bytes[idx]) * kFnvPrime64;
    hash = (hash ^ bytes[idx + 1]) * kFnvPrime64;
    hash = (hash ^ bytes[idx + 2]) * kFnvPrime64;
    hash = (hash ^ bytes[idx + 3]) * kFnvPrime64;
    hash = (hash ^ bytes[idx + 4]) * kFnvPrime64;
    hash = (hash ^ bytes[idx + 5]) * kFnvPrime64;
    hash = (hash ^ bytes[idx + 6]) * kFnvPrime64;
    hash = (hash ^ bytes[idx + 7]) * kFnvPrime64;
  }
  for (; idx < size; ++idx) {
    hash ^= bytes[idx];
    hash *= kFnvPrime64;
  }
  return hash;
}
// NOLINTEND(readability-magic-numbers,readability-identifier-length)

// Checksum over a header with its `checksum` field zeroed (taken by value, so writers with an
// already-zero field and readers discarding the real value share this call). Duck-typed on a
// `.checksum` member; no coupling between formats.
template <typename Header>
inline auto checksum_header(Header header) noexcept -> uint64_t {
  header.checksum = 0;
  return fnv1a64_update(kFnvOffsetBasis64, &header, sizeof(header));
}

// Folding one FNV-1a64 range at a time.
inline auto fnv1a64_range(uint64_t hash, std::span<const std::byte> data) noexcept -> uint64_t {
  return fnv1a64_update(hash, data.data(), data.size());
}

// Incremental FNV-1a64 accumulator over sequentially written bytes.
struct FoldingChecksum {
  uint64_t state = kFnvOffsetBasis64;

  void add(const void *data, size_t size) noexcept { state = fnv1a64_update(state, data, size); }
  void add(std::span<const std::byte> data) noexcept { state = fnv1a64_range(state, data); }
  template <typename Header>
  void add_header(Header header) noexcept {
    header.checksum = 0;
    state = fnv1a64_update(state, &header, sizeof(header));
  }
  [[nodiscard]] auto value() const noexcept -> uint64_t { return state; }
};

// Best-effort file removal; failures ignored (a leftover is harmless clutter, never an error).
inline void remove_quiet(const std::filesystem::path &path) noexcept {
  std::error_code ignored;
  std::filesystem::remove(path, ignored);
}

}  // namespace vmemkv
