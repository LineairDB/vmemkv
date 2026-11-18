// t1_codec.hpp - T1 key-prefix codec and scan-range helpers for T1Index.
// See docs/specification/low_level_design.md 7.5.
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

namespace t1_detail {
inline constexpr std::size_t kStoreKeyBytes = 16;
// Tier 1's internal 16-byte ordered key prefix.
using StoreKey = std::array<std::byte, kStoreKeyBytes>;
inline constexpr std::size_t kInlineValueByteCount = 8;
inline constexpr std::size_t kPrefixBytes = kStoreKeyBytes;
inline constexpr uint64_t kInlineFlagMask = 1ULL << 63;
inline constexpr uint64_t kMetadataShiftBits = 60;
inline constexpr uint64_t kInlineSizeMask = 7ULL;
inline constexpr uint64_t kCleanHashMask = ~(15ULL << 60);
inline constexpr std::byte kMaxKeyFillByte{0xFF};

inline auto prefix_from_bytes(std::span<const std::byte> key) noexcept -> StoreKey {
  StoreKey out{};
  const std::size_t byte_count = std::min(key.size(), kPrefixBytes);
  if (byte_count != 0) {
    std::copy_n(key.data(), byte_count, out.data());
  }
  return out;
}

// Converts the high 8 bytes of an ordered key prefix to big-endian uint64_t for numeric comparisons.
inline auto prefix_high_u64(StoreKey prefix) noexcept -> uint64_t {
  uint64_t val = 0;
  std::memcpy(&val, prefix.data(), sizeof(val));
  return std::byteswap(val);
}

// Converts the low 8 bytes of an ordered key prefix to big-endian uint64_t for numeric comparisons.
inline auto prefix_low_u64(StoreKey prefix) noexcept -> uint64_t {
  uint64_t val = 0;
  std::memcpy(&val, prefix.data() + sizeof(val), sizeof(val));
  return std::byteswap(val);
}

// Masks out the upper 4 bits (reserved for inline metadata) so hash comparisons
// elsewhere never need to mask again.
inline auto hash_full_key(std::span<const std::byte> key) noexcept -> uint64_t {
  constexpr uint64_t basis = 14695981039346656037ULL;
  constexpr uint64_t prime = 1099511628211ULL;
  uint64_t hash = basis;
  for (const std::byte byte_value : key) {
    hash ^= static_cast<uint8_t>(byte_value);
    hash *= prime;
  }
  return hash & kCleanHashMask;
}

inline auto is_inline(uint64_t hash) noexcept -> bool { return (hash & kInlineFlagMask) != 0; }

inline auto decode_size(uint64_t hash) noexcept -> std::size_t {
  uint64_t size_code = (hash >> kMetadataShiftBits) & kInlineSizeMask;
  return (size_code == 0) ? kInlineValueByteCount : static_cast<std::size_t>(size_code);
}

inline auto embed_metadata(uint64_t hash, uint8_t size) noexcept -> uint64_t {
  return (hash & kCleanHashMask) | kInlineFlagMask |
         (static_cast<uint64_t>(size & kInlineSizeMask) << kMetadataShiftBits);
}

// Range/bounds disjointness shared by scan()'s fast path and per-region skip.
inline auto range_disjoint(const StoreKey &region_min,
                           const StoreKey &region_max,
                           const StoreKey &lower,
                           const StoreKey &upper) noexcept -> bool {
  return !(region_min < upper) || region_max < lower;
}

// Logical entry identity: key prefix plus clean hash (ignoring transient inline metadata bits).
inline auto entry_identity_equal(const StoreKey &a_key,
                                 uint64_t a_hash,
                                 const StoreKey &b_key,
                                 uint64_t b_hash) noexcept -> bool {
  return a_key == b_key && (a_hash & kCleanHashMask) == (b_hash & kCleanHashMask);
}

inline auto entry_less(const StoreKey &a_key,
                       uint64_t a_hash,
                       const StoreKey &b_key,
                       uint64_t b_hash) noexcept -> bool {
  if (a_key != b_key) {
    return a_key < b_key;
  }
  return (a_hash & kCleanHashMask) < (b_hash & kCleanHashMask);
}

}  // namespace t1_detail

namespace vmemkv::t2_detail {

// T1 payload codec shared across T1 sharding, T2 ownership, compaction, and read paths:
// `offset | (block_count << kPayloadSizeShift)`, 16-byte-granular size hint.
inline constexpr uint64_t kPayloadSizeShift = 48;
inline constexpr uint64_t kPayloadOffsetMask = (1ULL << kPayloadSizeShift) - 1;
inline constexpr uint64_t kRecordBlockAlignment = 16;
inline constexpr uint64_t kMaxBlockCount = 1ULL << 16;
// Margin making the 16-byte-granular block hint cover the true 8-byte-aligned record length.
inline constexpr uint64_t kImmutableSizeHintMargin = 16;

inline auto decode_offset(uint64_t payload_bits) noexcept -> uint64_t { return payload_bits & kPayloadOffsetMask; }

inline auto payload_hint_bytes(uint64_t payload_bits, uint64_t margin = 0) noexcept -> uint64_t {
  return ((payload_bits >> kPayloadSizeShift) * kRecordBlockAlignment) + margin;
}

inline auto encode_offset_payload(uint64_t offset, uint64_t block_count) noexcept -> uint64_t {
  return offset | (block_count << kPayloadSizeShift);
}

}  // namespace vmemkv::t2_detail
