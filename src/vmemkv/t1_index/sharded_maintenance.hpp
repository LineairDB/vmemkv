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
#include <vmemkv/core/spin_backoff.hpp>

namespace vmemkv::sharded_detail {

inline constexpr size_t kPercentScale = 100;

// Pure split-size threshold shared by run_maintenance().
inline auto split_threshold(size_t target_shard_size, uint32_t percent) noexcept -> size_t {
  return (target_shard_size * percent) / kPercentScale;
}

// Append entries allowed before maintenance: half capacity normally, L2-capped while a scan
// is active. Pure function, unit-testable at production capacities. `append_slot_bytes`
// is the caller shard's per-slot footprint (Shard::append_slot_bytes()).
inline auto maintenance_soft_threshold(size_t append_cap, bool scan_active, size_t append_slot_bytes) -> size_t {
  constexpr size_t kSoftThresholdPercent = 50;
  constexpr size_t kL2CacheSizeBytes = size_t{1024} * 1024;
  const size_t normal = (append_cap * kSoftThresholdPercent) / kPercentScale;
  if (!scan_active) {
    return normal;
  }
  return std::min(kL2CacheSizeBytes / append_slot_bytes, normal);
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
