#pragma once
#include <cstdint>
namespace vmemkv {
// VMemKVStatistics: point-in-time counters snapshot by get_statistics().
// Units: *_us microseconds, *_ns nanoseconds, *_bytes bytes, bare counts otherwise.
// Keep in sync with kVMemKVStatisticsFieldCount in src/vmemkv/config.hpp.
struct VMemKVStatistics {
  uint64_t t1_split_count = 0;                             // shard splits performed, count
  uint64_t t1_last_split_pause_us = 0;                     // last split writer-pause duration, us
  uint64_t t1_last_split_pause_end_ns = 0;                 // last split pause-end timestamp, ns
  uint64_t checkpoint_count = 0;                           // checkpoints completed, count
  uint64_t last_checkpoint_duration_us = 0;                // last checkpoint total duration, us
  uint64_t last_checkpoint_msync_duration_us = 0;          // last checkpoint msync phase, us
  uint64_t last_checkpoint_t1_reorganize_duration_us = 0;  // last checkpoint T1 reorganize phase, us
  uint64_t last_checkpoint_t1_merge_duration_us = 0;       // last checkpoint T1 merge phase, us
  uint64_t last_checkpoint_t1_write_duration_us = 0;       // last checkpoint T1 write phase, us
  uint64_t last_checkpoint_stop_writers_duration_us = 0;   // last checkpoint writer-stop phase, us
  uint64_t last_checkpoint_barrier_drain_duration_us = 0;  // last checkpoint barrier-drain phase, us
  uint64_t last_checkpoint_wal_rotate_duration_us = 0;     // last checkpoint WAL-rotate phase, us
  uint64_t last_checkpoint_wal_rotate_leader_wait_us = 0;  // last checkpoint WAL-rotate leader wait, us
  uint64_t last_checkpoint_bytes_synced = 0;               // last checkpoint range synced, bytes
  uint64_t last_checkpoint_corpus_bytes = 0;               // last checkpoint corpus size, bytes
  uint64_t total_reorganize_wait_duration_us = 0;          // cumulative reorganize wait, us
  uint64_t bulk_load_throttle_events = 0;                  // bulk-load throttle events, count
  uint64_t compaction_cycle_count = 0;                     // dirty-shard repack cycles, count
  uint64_t t2_dirty_shard_count = 0;                       // T1 shards awaiting T2 repack, count
  int64_t append_region_live_count = 0;                    // T1 append-region live entries, count
  int64_t append_region_peak_count = 0;                    // T1 append-region peak entries, count
};
}  // namespace vmemkv
