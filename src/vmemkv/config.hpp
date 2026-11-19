// config.hpp - VMemKV engine defaults and runtime StoreOptions.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vmemkv/stats.hpp>

namespace vmemkv {

inline constexpr uint64_t STORE_NOT_FOUND = ~0ULL;  // Tombstone payload; never a valid offset.

struct T1Config {
  static constexpr size_t kMemTableCapacityLog2 = 15;
  static constexpr size_t kMemTableCapacityEntries = size_t{1} << kMemTableCapacityLog2;

  // Fallback per-shard live-entry target for inline-only shards (splits past
  // kShardSplitThresholdPercent% of this target). T2-backed shards split at T2Config::kSegmentBytes.
  static constexpr size_t kShardTargetSizeEntries = size_t{1} << 20;
  static constexpr size_t kShardSplitThresholdPercent = 200;

  // Target max bytes per T1 checkpoint chunk file (merge threshold = split / 2).
  static constexpr uint64_t kT1ChunkSplitBytes = 32ULL << 20;
};

static_assert(T1Config::kMemTableCapacityLog2 > 0 &&
                  T1Config::kMemTableCapacityLog2 < std::numeric_limits<size_t>::digits,
              "kMemTableCapacityLog2 must be in (0, bitwidth(size_t))");

struct T2Config {
  static constexpr size_t kDefaultCapacityBytes = 1ULL << 40;
  static constexpr uint64_t kSegmentBytes = 8ULL << 20;  // 8 MiB segment size for hole-punch accounting.
  static constexpr uint64_t kCompactionMaxMoveBytesPerCycle = kSegmentBytes;
  static constexpr uint64_t kPunchBacklogCheckpointCapSegments = 1058;
  static constexpr uint64_t kMsyncChunkBytes = 64ULL << 20;
};

struct WalConfig {
  static constexpr size_t kMaxBytesSinceCheckpoint = 64ULL << 20;  // Checkpoint WAL threshold (64 MiB).
};

// Immutable-region T2 read policy (docs/specification/low_level_design.md 7.5):
// - Adaptive: selects per-record across random_mmap, sequential_mmap, normal_mmap, and pread.
// - RandomOnly / SeqOnly / NormalOnly / PreadOnly: pins every immutable read to one handle.
enum class ReadPolicy : std::uint8_t {
  Adaptive,
  RandomOnly,
  SeqOnly,
  NormalOnly,
  PreadOnly,
};

struct StoreOptions {
  size_t memtable_capacity_entries = T1Config::kMemTableCapacityEntries;
  size_t shard_target_size_entries = T1Config::kShardTargetSizeEntries;
  uint64_t t1_chunk_split_bytes = T1Config::kT1ChunkSplitBytes;
  uint64_t segment_bytes = T2Config::kSegmentBytes;
  uint64_t punch_backlog_checkpoint_cap_segments = T2Config::kPunchBacklogCheckpointCapSegments;
  size_t wal_max_bytes_since_checkpoint = WalConfig::kMaxBytesSinceCheckpoint;
  ReadPolicy read_policy = ReadPolicy::Adaptive;
};

// X-macro: the checkpoint phase-breakdown fields of VMemKVStatistics below, and the
// matching atomics CheckpointCoordinator publishes them from. One list drives field
// declaration, snapshot, and publish, so the three can never drift apart.
#define VMEMKV_CHECKPOINT_STATS_FIELDS(X)      \
  X(last_checkpoint_duration_us)               \
  X(last_checkpoint_msync_duration_us)         \
  X(last_checkpoint_t1_reorganize_duration_us) \
  X(last_checkpoint_t1_merge_duration_us)      \
  X(last_checkpoint_t1_write_duration_us)      \
  X(last_checkpoint_stop_writers_duration_us)  \
  X(last_checkpoint_barrier_drain_duration_us) \
  X(last_checkpoint_wal_rotate_duration_us)    \
  X(last_checkpoint_wal_rotate_leader_wait_us) \
  X(last_checkpoint_bytes_synced)              \
  X(last_checkpoint_corpus_bytes)

#define VMEMKV_DECLARE_STAT_FIELD(name) uint64_t name = 0;

// Single sync point for VMemKVStatistics (include/vmemkv/stats.hpp) and
// VMEMKV_CHECKPOINT_STATS_FIELDS above: every field is 8 bytes, so the total size pins the
// field count. Update the count when adding a field to either side.
inline constexpr std::size_t kVMemKVStatisticsFieldCount = 21;
static_assert(sizeof(VMemKVStatistics) == kVMemKVStatisticsFieldCount * sizeof(uint64_t),
              "update VMemKVStatistics and VMEMKV_CHECKPOINT_STATS_FIELDS together");

}  // namespace vmemkv
