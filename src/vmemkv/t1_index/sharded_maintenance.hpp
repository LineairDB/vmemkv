// sharded_maintenance.hpp - Maintenance helpers for ShardedT1Index.
// Routing stays in sharded_t1_index.hpp; this header holds the pure maintenance
// thresholds plus the reorganize-until-captured retry used by checkpoint_all_shards(),
// continue_split(), and run_maintenance(). See docs/specification/t1_sharding_design.md.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <vmemkv/config.hpp>
#include <vmemkv/core/spin_backoff.hpp>
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

// Append entries allowed before maintenance (50% of append capacity).
inline auto maintenance_soft_threshold(size_t append_cap) noexcept -> size_t {
  constexpr size_t kSoftThresholdPercent = 50;
  return (append_cap * kSoftThresholdPercent) / kPercentScale;
}

// Retries reorganize() until `on_merged` fires. A racing concurrent reorganize() makes a call
// a silent no-op (reorg_in_progress_ CAS) -- indistinguishable from an empty shard by output
// alone, hence the retry.
template <typename Shard, typename OffsetMapper, typename OnMerged>
inline void reorganize_until_captured(Shard &shard,
                                      OffsetMapper offset_mapper,
                                      OnMerged on_merged,
                                      bool parallel_sort) {
  bool captured = false;
  SpinBackoff backoff;
  const auto start = std::chrono::steady_clock::now();
  auto last_warn = start;
  while (!captured) {
    shard.reorganize(
        offset_mapper,
        [&](std::span<const typename Shard::EntrySnapshot> merged) {
          on_merged(merged);
          captured = true;
        },
        parallel_sort);
    if (!captured) {
      backoff.wait();
      const auto now = std::chrono::steady_clock::now();
      if (now - last_warn >= std::chrono::seconds(2)) {
        last_warn = now;
        const auto total_sec = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
        std::cerr << "[reorganize_until_captured] RETRYING for " << total_sec << "s (failed to win reorg lock)\n";
      }
    }
  }
}

}  // namespace vmemkv::sharded_detail
