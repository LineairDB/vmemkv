// config.hpp - VMemKV configuration facets and read-policy tags.
//
// `vmemkv::Config<ReadPolicies...>` is the source of truth for T1/T2 settings
// and base-region read-policy selection.
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vmemkv/stats.hpp>

namespace vmemkv {

inline constexpr uint64_t STORE_NOT_FOUND = ~0ULL;  // Tombstone payload; never a valid offset.

// Bloom filtering and T1 inline-value storage are enabled in every configuration.
// Read-path policies for the base/tail split (docs/specification/low_level_design.md
// 7.5). Default: quadfecta -- size- and reader-dependent choice among the four
// T2ReadHandles (random_mmap, sequential_mmap, normal_mmap, pread_fd).
// ReadPolicyRandomOnly: every base-region read goes through the primary mapping.
// ReadPolicySeqOnly: every base-region read goes through the MADV_SEQUENTIAL mapping.
// ReadPolicyNormalOnly: every base-region read goes through the plain mapping.
// ReadPolicyPreadOnly: every base-region read goes through pread (no mmap readahead).
struct ReadPolicyRandomOnly {};
struct ReadPolicySeqOnly {};
struct ReadPolicyNormalOnly {};
struct ReadPolicyPreadOnly {};

// ─── Facet configs ─────────────────────────────────────────────────────────
// Single-responsibility value groups. Config<...> below aggregates them.
// Flat names re-exported for ConfigT::X uses.
struct T1Config {
  // Append-region capacity (ablation knob). Power of two for cache-friendly masking. Every
  // AppendRegion mmaps capacity * sizeof(AppendSlot) unconditionally and open-addressing drives
  // it near-full resident even at low load -- so this trades RSS against burst headroom before
  // a reorganize() drains it.
  static constexpr size_t kAppendCapacityLog2 = 14;
  static constexpr size_t kAppendCapacityEntries = size_t{1} << kAppendCapacityLog2;

  // Per-shard live-entry target for ShardedT1Index: maintenance splits past
  // kShardSplitThresholdPercent% of it. A shard at the split threshold spans about one
  // T2 segment (kSegmentBytes) of 1KB values, so a shard repack stays within a segment.
  // See docs/specification/t1_sharding_design.md for the relation.
  static constexpr size_t kShardTargetSizeEntries = size_t{1} << 12;
  static constexpr size_t kShardSplitThresholdPercent = 200;
};

struct T2Config {
  // Default Tier 2 file capacity: 1 TiB.
  static constexpr size_t kDefaultCapacityBytes = 1ULL << 40;
  // Fixed 8MiB segments for hole-punch accounting.
  static constexpr uint64_t kSegmentBytes = 8ULL << 20;  // 8 MiB. Fixed, not tuned.
  // Maximum T2 bytes relocated per compaction cycle (8 MiB segment budget).
  static constexpr uint64_t kCompactionMaxMoveBytesPerCycle = kSegmentBytes;
  // Emergency cap on punch backlog segments to prevent runaway disk usage.
  static constexpr uint64_t kPunchBacklogCheckpointCapSegments = 1058;
  // Chunk size for subdivided msync in checkpoint coordinator to avoid long foreground stalls.
  static constexpr uint64_t kMsyncChunkBytes = 64ULL << 20;  // 64 MiB.
};

struct WalConfig {
  // Checkpoint trigger: past this many WAL bytes since the last checkpoint, a checkpoint fires
  // regardless of append-region occupancy, bounding replay time.
  // See docs/specification/low_level_design.md 4.4.
  static constexpr size_t kMaxBytesSinceCheckpoint = 64ULL << 20;  // 64 MiB.
};

// Read-path policy facet (see the tags above for the policy meanings).
template <bool RandomOnly, bool SeqOnly, bool NormalOnly, bool PreadOnly>
struct ReadPolicy {
  static constexpr bool kRandomOnly = RandomOnly;
  static constexpr bool kSeqOnly = SeqOnly;
  static constexpr bool kNormalOnly = NormalOnly;
  static constexpr bool kPreadOnly = PreadOnly;
};
// ─── Unified System Config Template ─────────────────────────────────────────
// Aggregates the facet configs above.
// Flat names re-exported for ConfigT::X uses.
template <typename... ReadPolicies>
struct Config : T1Config,
                T2Config,
                WalConfig,
                ReadPolicy<(std::is_same_v<ReadPolicyRandomOnly, ReadPolicies> || ...),
                           (std::is_same_v<ReadPolicySeqOnly, ReadPolicies> || ...),
                           (std::is_same_v<ReadPolicyNormalOnly, ReadPolicies> || ...),
                           (std::is_same_v<ReadPolicyPreadOnly, ReadPolicies> || ...)> {
  static constexpr bool UseBloomFilter = true;
  static constexpr bool UseT1InlineValue = true;
  static constexpr bool UseReadPolicyRandomOnly = (std::is_same_v<ReadPolicyRandomOnly, ReadPolicies> || ...);
  static constexpr bool UseReadPolicySeqOnly = (std::is_same_v<ReadPolicySeqOnly, ReadPolicies> || ...);
  static constexpr bool UseReadPolicyNormalOnly = (std::is_same_v<ReadPolicyNormalOnly, ReadPolicies> || ...);
  static constexpr bool UseReadPolicyPreadOnly = (std::is_same_v<ReadPolicyPreadOnly, ReadPolicies> || ...);

  static constexpr size_t kBitsPerByte = 8;

  static_assert(kAppendCapacityLog2 > 0 && kAppendCapacityLog2 < (sizeof(size_t) * kBitsPerByte),
                "kAppendCapacityLog2 must be in (0, bitwidth(size_t))");
  // Number of pinned read-path policies: at most one may be set.
  static constexpr int kActiveReadPolicies =
      static_cast<int>(UseReadPolicyRandomOnly) + static_cast<int>(UseReadPolicySeqOnly) +
      static_cast<int>(UseReadPolicyNormalOnly) + static_cast<int>(UseReadPolicyPreadOnly);
  static_assert(kActiveReadPolicies <= 1, "ReadPolicy* tags are mutually exclusive");
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
