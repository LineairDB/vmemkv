// sharded_maintenance.hpp - Maintenance helpers for ShardedT1Index.
// Routing stays in sharded_t1_index.hpp; this header holds the pure maintenance thresholds.
// See docs/specification/t1_sharding_design.md.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vmemkv/config.hpp>
#include <vmemkv/t1_index/t1_codec.hpp>

namespace vmemkv::sharded_detail {

inline constexpr size_t kPercentScale = 100;

// Pure split-size threshold shared by run_maintenance().
inline auto split_threshold(size_t target_shard_size, uint32_t percent) noexcept -> size_t {
  return (target_shard_size * percent) / kPercentScale;
}

// Extracts the aligned T2 byte footprint encoded in a T1 offset payload (0 for tombstones or inline entries).
inline auto payload_t2_bytes(uint64_t payload_bits, bool is_inline) noexcept -> uint64_t {
  if (payload_bits == STORE_NOT_FOUND || is_inline) {
    return 0;
  }
  return vmemkv::t2_detail::payload_hint_bytes(payload_bits);
}

template <typename EntryRange>
inline auto sum_entries_t2_bytes(const EntryRange &entries) noexcept -> uint64_t {
  uint64_t total = 0;
  for (const auto &entry : entries) {
    total += payload_t2_bytes(entry.payload_bits, t1_detail::is_inline(entry.hash));
  }
  return total;
}

// MemTable entries allowed before maintenance (50% of MemTable capacity).
inline auto maintenance_soft_threshold(size_t memtable_cap) noexcept -> size_t {
  constexpr size_t kSoftThresholdPercent = 50;
  return (memtable_cap * kSoftThresholdPercent) / kPercentScale;
}

}  // namespace vmemkv::sharded_detail
