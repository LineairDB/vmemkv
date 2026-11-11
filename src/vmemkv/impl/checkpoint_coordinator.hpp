// checkpoint_coordinator.hpp - Checkpoint/reorganize orchestration for VMemKVImpl.
//
// Owns the reorg single-flight state and every checkpoint-cycle phase. Durabilization is always
// in place (msync; T2's MAP_SHARED live mmap is the durable copy, so no relocation).
#pragma once

#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <vector>
#include <vmemkv/checkpoint/checkpoint.hpp>
#include <vmemkv/config.hpp>
#include <vmemkv/core/background_poll.hpp>
#include <vmemkv/core/env.hpp>
#include <vmemkv/core/single_flight.hpp>
#include <vmemkv/impl/compaction.hpp>
#include <vmemkv/impl/hooks.hpp>
#include <vmemkv/impl/t2_ownership.hpp>
#include <vmemkv/t1_index/sharded_t1_index.hpp>
#include <vmemkv/t2_flat_file/t2_flat_file.hpp>
#include <vmemkv/wal/wal.hpp>

namespace vmemkv {

// T1Only: in-memory merge, zero I/O. Checkpoint: durabilizes T2's tail in place and commits
// a manifest (docs/specification/low_level_design.md 5.2, 5.5). Re-exported as VMemKVImpl's nested ReorgMode.
enum class ReorgMode : std::uint8_t { T1Only, Checkpoint };

// Single-flight and trigger state for reorganize/checkpoint cycles.
struct ReorgTrigger {
  std::atomic<bool> requested{false};
  std::atomic<bool> running{false};
  // Strides wal_over_threshold() sampling below (an fstat(), too dear per write).
  mutable std::atomic<uint64_t> wal_check_counter{0};
};

// Cycle accounting: published stats for get_statistics().
struct ReorgStats {
  mutable std::atomic<uint64_t> checkpoint_count{0};
  mutable std::atomic<uint64_t> total_wait_us{0};
#define VMEMKV_DECLARE_STAT_ATOMIC(name) mutable std::atomic<uint64_t> name##_{0};
  VMEMKV_CHECKPOINT_STATS_FIELDS(VMEMKV_DECLARE_STAT_ATOMIC)
#undef VMEMKV_DECLARE_STAT_ATOMIC
};

struct ReorgState {
  ReorgTrigger trigger;
  ReorgStats stats;
};

// Phase timings for one cycle. Field names match VMEMKV_CHECKPOINT_STATS_FIELDS so
// publish/snapshot stay uniform. Single-flight only, so plain values suffice.
struct CheckpointTimings {
  VMEMKV_CHECKPOINT_STATS_FIELDS(VMEMKV_DECLARE_STAT_FIELD)
};

inline void publish_checkpoint_stats(ReorgStats &stats, const CheckpointTimings &timings) {
#define VMEMKV_PUBLISH_ONE(name) stats.name##_.store(timings.name, std::memory_order_relaxed);
  VMEMKV_CHECKPOINT_STATS_FIELDS(VMEMKV_PUBLISH_ONE)
#undef VMEMKV_PUBLISH_ONE
}

inline void snapshot_checkpoint_stats(const ReorgStats &stats, VMemKVStatistics &out) {
#define VMEMKV_SNAPSHOT_ONE(name) out.name = stats.name##_.load(std::memory_order_relaxed);
  VMEMKV_CHECKPOINT_STATS_FIELDS(VMEMKV_SNAPSHOT_ONE)
#undef VMEMKV_SNAPSHOT_ONE
}

namespace checkpoint_detail {

inline std::atomic<const char *> g_checkpoint_debug_phase{"idle"};

inline auto auto_reorg_suppressed() noexcept -> bool {
  // Empty counts as unset: getenv_view() returns empty for unset or empty alike.
  return !getenv_view("VMEMKV_SUPPRESS_AUTO_REORG").empty();
}

inline auto wal_over_threshold(const Wal &wal) -> bool {
  return wal.size_bytes() >= WalConfig::kMaxBytesSinceCheckpoint;
}

template <typename ConfigT>
inline auto punch_backlog_over_cap(const T2Ownership<ConfigT> &own) -> bool {
  return own.punch_backlog_segments() >= ConfigT::kPunchBacklogCheckpointCapSegments;
}

template <typename ConfigT>
inline auto checkpoint_due(const Wal &wal, const T2Ownership<ConfigT> &own) -> bool {
  return wal_over_threshold(wal) || punch_backlog_over_cap<ConfigT>(own);
}

template <typename ConfigT>
inline void maybe_reorganize_if_needed(ReorgTrigger &trigger, const Wal &wal, const T2Ownership<ConfigT> &own) {
  // Sampled once every kWalCheckStride writes: the delay past the byte threshold is bounded by
  // kWalCheckStride writes' worth of bytes, a rounding error against the threshold itself.
  constexpr uint64_t kWalCheckStride = 64;
  if (trigger.wal_check_counter.fetch_add(1, std::memory_order_relaxed) % kWalCheckStride == 0) {
    if (wal_over_threshold(wal) || punch_backlog_over_cap<ConfigT>(own)) {
      trigger.requested.store(true, std::memory_order_release);
    }
  }
}

// Bounded poll, not atomic::wait(): no timed overload to bound a missed wakeup. Callers only
// arrive with a cycle already in flight, so the wait is already bounded by a reorganize's own
// duration. Serializes explicit reorganize()/checkpoint() callers only; writes never call this.
inline void wait_until_reorg_not_running(const ReorgState &state) {
  constexpr auto kIdlePollInterval = kDefaultPoll10ms;
  const auto wait_start = std::chrono::steady_clock::now();
  while (state.trigger.running.load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(kIdlePollInterval);
  }
  // Summed explicit-caller wait (not the checkpoint's own overlapped duration).
  const auto waited_us =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - wait_start).count();
  state.stats.total_wait_us.fetch_add(static_cast<uint64_t>(waited_us), std::memory_order_relaxed);
}

struct CheckpointTarget {
  uint64_t old_base_boundary = 0;
  uint64_t target = 0;
};

// Briefly stops new appends only (in-place updates keep running). Every in-flight append
// holds its handle across reserve-then-write, so post-wait `target` is a fully-written
// frontier, never merely reserved. Writers resume before return -- msync() runs concurrently
// with new writes.
template <typename PreStopHook>
auto capture_checkpoint_target(T2FlatFile &tier2,
                               PreStopHook pre_stop_hook,
                               std::chrono::nanoseconds &stop_phase) -> CheckpointTarget {
  CheckpointTarget out;
  out.old_base_boundary = tier2.get_memory()->base_boundary.load(std::memory_order_acquire);
  const vmemkv::T2Memory *mem_to_drain = tier2.get_memory();
  pre_stop_hook();
  const auto stop_start = std::chrono::steady_clock::now();
  tier2.stop_writers_and_wait(mem_to_drain);
  stop_phase = std::chrono::steady_clock::now() - stop_start;
  {
    WriterResumeGuard resume_guard{&tier2};
    out.target = tier2.get_memory()->bytes_used.load(std::memory_order_acquire);
  }
  return out;
}

// msync()'s addr must be page-aligned; old_base_boundary is only block-aligned. Rounding
// down re-syncs a small already-durable overlap -- harmless, msync() is idempotent.
// Synchronizes in chunks (ConfigT::kMsyncChunkBytes) and yields between chunks to avoid
// monopolizing kernel writeback and stalling concurrent foreground operations.
template <typename ConfigT = Config<>>
inline void msync_target_range(const T2Memory *mem, uint64_t old_base_boundary, uint64_t target) {
  if (target <= old_base_boundary) {
    return;
  }
  const uint64_t page_size = t2_detail::os_page_size();
  const uint64_t aligned_start = old_base_boundary - (old_base_boundary % page_size);
  const uint64_t chunk_bytes = ConfigT::kMsyncChunkBytes;
  for (uint64_t cur = aligned_start; cur < target;) {
    const uint64_t cur_len = std::min(chunk_bytes, target - cur);
    const int msync_rc = ::msync(mem->reads.random_mmap + cur, cur_len, MS_SYNC);
    if (msync_rc != 0) {
      throw std::system_error(errno, std::generic_category(), "msync t2 checkpoint");
    }
    cur += cur_len;
    if (cur < target) {
      std::this_thread::yield();
    }
  }
}

// The only T1 publish per cycle. Checkpoint never relocates, so no payload_bits changes and
// the offset mapper is a no-op. Streams shards ascending without materializing the corpus;
// the trailer writer finishes once boundaries are known.
//
// Only offsets below `target` are persisted: post-target appends are still in flight
// (writers resumed right after the capture) and may be torn or absent on disk, so
// naming them here would reference undurable bytes. They stay WAL-covered instead:
// every post-target T1 change comes from a WAL-logged foreground op (relocations
// only rewrite the same version elsewhere), and logical replay rebuilds them at
// new offsets. Tombstones and T1-inline values carry no T2 offset and persist as-is.
template <typename ConfigT>
auto publish_t1_checkpoint(ShardedT1Index<ConfigT> &tier1,
                           const std::filesystem::path &t2_path,
                           uint64_t target,
                           std::chrono::nanoseconds &t1_write_phase) -> std::chrono::nanoseconds {
  using EntrySnapshot = typename ShardedT1Index<ConfigT>::EntrySnapshot;
  auto offset_mapper_fn = [](std::span<EntrySnapshot> /*merged*/) {};
  vmemkv::ShardedT1CheckpointWriter t1_chk_writer(vmemkv::derive_t1_chk_path(t2_path));
  auto chk_writer_fn = [&](std::span<const EntrySnapshot> merged) {
    const auto write_start = std::chrono::steady_clock::now();
    size_t filtered_count = 0;
    for (const auto &entry : merged) {
      if (entry.payload_bits != vmemkv::STORE_NOT_FOUND && !t1_detail::is_inline(entry.hash) &&
          (entry.payload_bits & t2_detail::kPayloadOffsetMask) >= target) {
        continue;
      }
      ++filtered_count;
    }
    t1_chk_writer.begin_shard(filtered_count);

    constexpr size_t kChunkSize = 2048;
    std::array<vmemkv::T1ChkEntry, kChunkSize> chunk{};
    size_t chunk_len = 0;
    for (const auto &entry : merged) {
      if (entry.payload_bits != vmemkv::STORE_NOT_FOUND && !t1_detail::is_inline(entry.hash) &&
          (entry.payload_bits & t2_detail::kPayloadOffsetMask) >= target) {
        continue;
      }
      chunk[chunk_len++] = vmemkv::T1ChkEntry{entry.key, entry.hash, entry.payload_bits};
      if (chunk_len == kChunkSize) {
        t1_chk_writer.write_shard_chunk(chunk.data(), chunk_len);
        chunk_len = 0;
      }
    }
    if (chunk_len > 0) {
      t1_chk_writer.write_shard_chunk(chunk.data(), chunk_len);
    }
    t1_write_phase += std::chrono::steady_clock::now() - write_start;
  };
  const auto start = std::chrono::steady_clock::now();
  const auto t1_boundaries = tier1.checkpoint_all_shards(offset_mapper_fn, chk_writer_fn);
  const auto elapsed = std::chrono::steady_clock::now() - start;
  t1_chk_writer.finish(t1_boundaries);
  return elapsed;
}

inline auto to_us(std::chrono::nanoseconds duration) -> uint64_t {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(duration).count());
}

// One cycle's phase durations plus its byte range, feeding publish_cycle_stats() below.
struct CheckpointPhases {
  std::chrono::nanoseconds stop{};
  std::chrono::nanoseconds barrier{};
  std::chrono::nanoseconds msync{};
  std::chrono::nanoseconds t1{};
  std::chrono::nanoseconds t1_write{};
  std::chrono::nanoseconds rotate{};
  std::chrono::nanoseconds total{};
  uint64_t old_base_boundary = 0;
  uint64_t target = 0;
};

// Published after checkpoint_count, so a poller waking on it always sees this cycle's own
// numbers, never a torn mix with the next cycle's.
inline void publish_cycle_stats(ReorgStats &stats, uint64_t rotate_leader_wait_us, const CheckpointPhases &phases) {
  stats.checkpoint_count.fetch_add(1, std::memory_order_relaxed);
  CheckpointTimings timings;
  timings.last_checkpoint_duration_us = to_us(phases.total);
  timings.last_checkpoint_msync_duration_us = to_us(phases.msync);
  timings.last_checkpoint_t1_reorganize_duration_us = to_us(phases.t1);
  timings.last_checkpoint_t1_write_duration_us = to_us(phases.t1_write);
  timings.last_checkpoint_t1_merge_duration_us =
      to_us(phases.t1 > phases.t1_write ? phases.t1 - phases.t1_write : std::chrono::nanoseconds{0});
  timings.last_checkpoint_stop_writers_duration_us = to_us(phases.stop);
  timings.last_checkpoint_barrier_drain_duration_us = to_us(phases.barrier);
  timings.last_checkpoint_wal_rotate_duration_us = to_us(phases.rotate);
  timings.last_checkpoint_wal_rotate_leader_wait_us = rotate_leader_wait_us;
  timings.last_checkpoint_bytes_synced = phases.target - phases.old_base_boundary;
  timings.last_checkpoint_corpus_bytes = phases.target;
  publish_checkpoint_stats(stats, timings);
}

// Claims the target range before msync() reads it, then drains in-flight in-place writes
// against the old watermark. Without the claim, a racing update could be msync()'d mid-write
// (torn on disk) or land after base_boundary publishes past it.
template <typename ConfigT>
auto claim_and_drain_watermark(T2Ownership<ConfigT> &own, uint64_t target) -> std::chrono::nanoseconds {
  own.claim_watermark(target);
  const auto barrier_start = std::chrono::steady_clock::now();
  own.drain_in_place(target);
  return std::chrono::steady_clock::now() - barrier_start;
}

// Publishes base_boundary, then rotates the WAL. No checkpoint_lsn needed (docs/specification/low_level_design.md
// 5.5): rotation only retires the previous cycle's generation, already covered by this cycle's
// manifest by construction.
inline auto publish_boundary_and_rotate(T2FlatFile &tier2, Wal &wal, uint64_t target) -> std::chrono::nanoseconds {
  tier2.get_memory()->base_boundary.store(target, std::memory_order_release);
  const auto rotate_start = std::chrono::steady_clock::now();
  wal.rotate_segment();
  return std::chrono::steady_clock::now() - rotate_start;
}

// Durabilizes T2's live tail grown since the last cycle, advances base_boundary in place, and
// persists a manifest-committed checkpoint. Correctness argument:
// docs/specification/why_vmemkv_does_not_need_undo_log.md; contract: docs/specification/low_level_design.md
// 4.3/5.3.
//
// Capture -> claim/drain -> msync -> T1 publish -> boundary publish/rotate -> stats.
template <typename ConfigT, typename PreStopHook, typename PreFinishHook>
  requires CheckpointHook<PreStopHook> && CheckpointHook<PreFinishHook>
void checkpoint_internal(ReorgState &state,
                         ShardedT1Index<ConfigT> &tier1,
                         T2FlatFile &tier2,
                         Wal &wal,
                         T2Ownership<ConfigT> &own,
                         const std::filesystem::path &t2_path,
                         PreStopHook pre_stop_hook,
                         PreFinishHook pre_finish_hook) {
  const auto fn_start = std::chrono::steady_clock::now();
  const uint64_t checkpoint_lsn = wal.next_lsn() - 1;

  g_checkpoint_debug_phase.store("checkpoint: capture_checkpoint_target", std::memory_order_release);
  std::chrono::nanoseconds stop_phase{0};
  const CheckpointTarget checkpoint_target = capture_checkpoint_target(tier2, pre_stop_hook, stop_phase);

  g_checkpoint_debug_phase.store("checkpoint: claim_and_drain_watermark", std::memory_order_release);
  const auto barrier_phase = claim_and_drain_watermark(own, checkpoint_target.target);

  std::chrono::nanoseconds msync_phase{0};
  std::chrono::nanoseconds t1_phase{0};
  std::chrono::nanoseconds t1_write_phase{0};
  try {
    const vmemkv::T2Memory *mem = tier2.get_memory();
    const auto msync_start = std::chrono::steady_clock::now();
    g_checkpoint_debug_phase.store("checkpoint: msync_target_range", std::memory_order_release);
    msync_target_range<ConfigT>(mem, checkpoint_target.old_base_boundary, checkpoint_target.target);
    msync_phase = std::chrono::steady_clock::now() - msync_start;

    pre_finish_hook();

    // The manifest frontier is read after the T1 merge snapshot above, so every offset the
    // snapshot names precedes it; the sparse clone copy bounded by it never truncates
    // file-referenced bytes.
    // The manifest frontier is the capture target, not the post-snapshot size:
    // post-target appends are WAL-covered only and must not be adopted as
    // durable on restart (see publish_t1_checkpoint's filter).
    g_checkpoint_debug_phase.store("checkpoint: publish_t1_checkpoint", std::memory_order_release);
    t1_phase = publish_t1_checkpoint(tier1, t2_path, checkpoint_target.target, t1_write_phase);
    g_checkpoint_debug_phase.store("checkpoint: write_manifest", std::memory_order_release);
    vmemkv::write_manifest(vmemkv::derive_manifest_path(t2_path), checkpoint_lsn, checkpoint_target.target);
  } catch (...) {
    // Nothing durable happened yet (base_boundary hasn't advanced) -- un-claim so in-place
    // updates aren't blocked forever for an unpublished cycle.
    own.revert_watermark(checkpoint_target.old_base_boundary);
    g_checkpoint_debug_phase.store("checkpoint: aborted", std::memory_order_release);
    throw;
  }

  g_checkpoint_debug_phase.store("checkpoint: publish_boundary_and_rotate", std::memory_order_release);
  const auto rotate_phase = publish_boundary_and_rotate(tier2, wal, checkpoint_target.target);
  g_checkpoint_debug_phase.store("idle", std::memory_order_release);

  CheckpointPhases phases;
  phases.stop = stop_phase;
  phases.barrier = barrier_phase;
  phases.msync = msync_phase;
  phases.t1 = t1_phase;
  phases.t1_write = t1_write_phase;
  phases.rotate = rotate_phase;
  phases.total = std::chrono::steady_clock::now() - fn_start;
  phases.old_base_boundary = checkpoint_target.old_base_boundary;
  phases.target = checkpoint_target.target;
  publish_cycle_stats(state.stats, wal.last_rotate_leader_wait_us(), phases);
}

// Merges T1 and, for Checkpoint, durabilizes T2 + commits a manifest. Runs under the running
// CAS guard.
template <typename ConfigT, typename PreStopHook, typename PreFinishHook>
  requires CheckpointHook<PreStopHook> && CheckpointHook<PreFinishHook>
void reorganize_internal(ReorgState &state,
                         ShardedT1Index<ConfigT> &tier1,
                         T2FlatFile &tier2,
                         Wal &wal,
                         T2Ownership<ConfigT> &own,
                         const std::filesystem::path &t2_path,
                         [[maybe_unused]] bool recovering,
                         ReorgMode mode,
                         PreStopHook pre_stop_hook,
                         PreFinishHook pre_finish_hook) {
  // Checkpointing during WAL replay is unsafe (recovery is inside replay()'s callback, and a
  // cycle expects sole writer status). Recovery only passes T1Only -- throw, don't silently
  // override, so a future caller bug surfaces.
  if (recovering && mode != ReorgMode::T1Only) {
    throw std::logic_error("checkpoint during recovery");
  }

  switch (mode) {
    case ReorgMode::Checkpoint:
      checkpoint_internal(state, tier1, tier2, wal, own, t2_path, pre_stop_hook, pre_finish_hook);
      // Punch eligibility trails checkpoint coverage: a segment evacuated at count
      // N is punchable only after a checkpoint completing past N. Recorded here,
      // the single choke point for every Checkpoint-mode cycle (foreground,
      // background worker, and tests), so the ownership gate cannot miss one.
      own.note_checkpoint_completed(state.stats.checkpoint_count.load(std::memory_order_relaxed));
      break;
    case ReorgMode::T1Only: {
      // Zero-I/O T1 merge with untouched payloads. checkpoint_all_shards() with no-op writers
      // gives the same "every shard merged before return" contract.
      g_checkpoint_debug_phase.store("reorganize: T1Only checkpoint_all_shards", std::memory_order_release);
      using EntrySnapshot = typename ShardedT1Index<ConfigT>::EntrySnapshot;
      tier1.checkpoint_all_shards([](std::span<EntrySnapshot> /*merged*/) {},
                                  [](std::span<const EntrySnapshot> /*merged*/) {});
      g_checkpoint_debug_phase.store("idle", std::memory_order_release);
      break;
    }
  }
}

// Shared wait/CAS/run/retry loop for reorganize()/checkpoint(). Checkpoint always forces a
// completed cycle (it needs a durabilized manifest, not just an empty append region); T1Only
// skips on empty.
//
// Runs (or waits out) exactly one cycle, then returns -- never re-checks for emptiness after.
// Under sustained writes the region is never momentarily empty, so re-checking could win the
// CAS against itself indefinitely and never return.
template <typename ConfigT, typename PreStopHook, typename PreFinishHook>
  requires CheckpointHook<PreStopHook> && CheckpointHook<PreFinishHook>
void run_reorganize(ReorgState &state,
                    ShardedT1Index<ConfigT> &tier1,
                    T2FlatFile &tier2,
                    Wal &wal,
                    T2Ownership<ConfigT> &own,
                    const std::filesystem::path &t2_path,
                    bool recovering,
                    ReorgMode mode,
                    PreStopHook pre_stop_hook,
                    PreFinishHook pre_finish_hook) {
  const bool force_run = (mode == ReorgMode::Checkpoint);
  const uint64_t start_chk_count = state.stats.checkpoint_count.load(std::memory_order_acquire);

  while (true) {
    wait_until_reorg_not_running(state);

    // If another concurrent cycle (e.g. background worker) completed a checkpoint
    // while we waited, our durability guarantee is satisfied: avoid back-to-back duplicate cycles.
    if (force_run && state.stats.checkpoint_count.load(std::memory_order_acquire) > start_chk_count) {
      return;
    }

    if (!force_run && tier1.append_size() == 0) {
      return;
    }

    if (auto guard = SingleFlightGuard::try_acquire(state.trigger.running); guard.holds) {
      reorganize_internal(state, tier1, tier2, wal, own, t2_path, recovering, mode, pre_stop_hook, pre_finish_hook);
      return;
    }
    // Lost the race to a concurrent cycle. Loop back and wait -- needed for force_run: the
    // winner's cycle might not be ours (a plain reorganize() beating our checkpoint()).
  }
}

}  // namespace checkpoint_detail
}  // namespace vmemkv
