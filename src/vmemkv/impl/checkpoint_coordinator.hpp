// checkpoint_coordinator.hpp - Checkpoint/reorganize orchestration for VMemKVImpl.
//
// Owns the reorg single-flight state and every checkpoint-cycle phase. Durabilization is always
// in place (msync; T2's MAP_SHARED live mmap is the durable copy, so no relocation).
#pragma once

#include <sys/mman.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>
#include <vmemkv/checkpoint/checkpoint.hpp>
#include <vmemkv/config.hpp>
#include <vmemkv/core/background_poll.hpp>
#include <vmemkv/core/env.hpp>
#include <vmemkv/core/paths.hpp>
#include <vmemkv/core/single_flight.hpp>
#include <vmemkv/core/utils.hpp>
#include <vmemkv/impl/hooks.hpp>
#include <vmemkv/impl/t2_ownership.hpp>
#include <vmemkv/stats.hpp>
#include <vmemkv/t1_index/sharded_t1_index.hpp>
#include <vmemkv/t1_index/t1_codec.hpp>
#include <vmemkv/t2_flat_file/t2_flat_file.hpp>
#include <vmemkv/wal/wal.hpp>

namespace vmemkv {

// Cycle accounting and single-flight state for checkpoint cycles.
struct CheckpointStats {
  mutable std::atomic<uint64_t> checkpoint_count{0};
  mutable std::atomic<uint64_t> total_wait_us{0};
#define VMEMKV_DECLARE_STAT_ATOMIC(name) mutable std::atomic<uint64_t> name##_{0};
  VMEMKV_CHECKPOINT_STATS_FIELDS(VMEMKV_DECLARE_STAT_ATOMIC)
#undef VMEMKV_DECLARE_STAT_ATOMIC
};

struct CheckpointState {
  std::atomic<bool> running{false};
  std::vector<T1ChunkMeta> t1_chunks;
  uint64_t next_t1_chunk_id{1};
  CheckpointStats stats;
};

// Phase timings for one cycle. Field names match VMEMKV_CHECKPOINT_STATS_FIELDS so
// publish/snapshot stay uniform. Single-flight only, so plain values suffice.
struct CheckpointTimings {
  VMEMKV_CHECKPOINT_STATS_FIELDS(VMEMKV_DECLARE_STAT_FIELD)
};

inline void publish_checkpoint_stats(CheckpointStats &stats, const CheckpointTimings &timings) {
#define VMEMKV_PUBLISH_ONE(name) stats.name##_.store(timings.name, std::memory_order_relaxed);
  VMEMKV_CHECKPOINT_STATS_FIELDS(VMEMKV_PUBLISH_ONE)
#undef VMEMKV_PUBLISH_ONE
}

inline void snapshot_checkpoint_stats(const CheckpointStats &stats, VMemKVStatistics &out) {
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

inline auto wal_over_threshold(const Wal &wal, size_t max_bytes = WalConfig::kMaxBytesSinceCheckpoint) -> bool {
  return wal.size_bytes() >= max_bytes;
}

inline auto punch_backlog_over_cap(const T2Ownership &own,
                                   uint64_t cap = T2Config::kPunchBacklogCheckpointCapSegments) -> bool {
  return own.punch_backlog_segments() >= cap;
}

inline auto checkpoint_due(const Wal &wal, const T2Ownership &own, const StoreOptions &options = {}) -> bool {
  return wal_over_threshold(wal, options.wal_max_bytes_since_checkpoint) ||
         punch_backlog_over_cap(own, options.punch_backlog_checkpoint_cap_segments);
}

// Bounded poll, not atomic::wait(): no timed overload to bound a missed wakeup. Callers only
// arrive with a cycle already in flight, so the wait is already bounded by a cycle's own
// duration. Serializes explicit reorganize()/checkpoint() callers only; writes never call this.
inline void wait_until_checkpoint_not_running(const CheckpointState &state) {
  constexpr auto kIdlePollInterval = kDefaultPoll10ms;
  const auto wait_start = std::chrono::steady_clock::now();
  while (state.running.load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(kIdlePollInterval);
  }
  // Summed explicit-caller wait (not the checkpoint's own overlapped duration).
  const auto waited_us =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - wait_start).count();
  state.stats.total_wait_us.fetch_add(static_cast<uint64_t>(waited_us), std::memory_order_relaxed);
}

struct CheckpointTarget {
  uint64_t old_read_only_offset = 0;
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
  out.old_read_only_offset = tier2.get_memory()->read_only_offset.load(std::memory_order_acquire);
  const vmemkv::T2Memory *mem_to_drain = tier2.get_memory();
  pre_stop_hook();
  const auto stop_start = std::chrono::steady_clock::now();
  tier2.stop_writers_and_wait(mem_to_drain);
  stop_phase = std::chrono::steady_clock::now() - stop_start;
  {
    WriterResumeGuard resume_guard{&tier2};
    out.target = tier2.get_memory()->tail_offset.load(std::memory_order_acquire);
  }
  return out;
}

// msync()'s addr must be page-aligned; old_read_only_offset is only block-aligned. Rounding
// down re-syncs a small already-durable overlap -- harmless, msync() is idempotent.
// Synchronizes in chunks (`T2Config::kMsyncChunkBytes`) and yields between chunks to avoid
// monopolizing kernel writeback and stalling concurrent foreground operations.
inline void msync_target_range(const T2Memory *mem, uint64_t old_read_only_offset, uint64_t target) {
  if (target <= old_read_only_offset) {
    return;
  }
  const uint64_t page_size = t2_detail::os_page_size();
  const uint64_t aligned_start = old_read_only_offset - (old_read_only_offset % page_size);
  for (uint64_t cur = aligned_start; cur < target;) {
    const uint64_t cur_len = std::min(T2Config::kMsyncChunkBytes, target - cur);
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
// Streams entries (< target) for one shard via add_shard(). Returns true if any non-inline
// T2 offset >= target was filtered out.
template <typename EntrySnapshot>
auto stream_filtered_shard(std::span<const EntrySnapshot> entries,
                           uint64_t target,
                           vmemkv::ShardedT1CheckpointWriter &writer) -> bool {
  auto is_post_target = [target](const EntrySnapshot &entry) noexcept {
    return entry.payload_bits != vmemkv::STORE_NOT_FOUND && !t1_detail::is_inline(entry.hash) &&
           t2_detail::decode_offset(entry.payload_bits) >= target;
  };
  if (std::ranges::none_of(entries, is_post_target)) {
    writer.add_shard(entries);
    return false;
  }
  std::vector<EntrySnapshot> filtered;
  filtered.reserve(entries.size());
  for (const auto &entry : entries) {
    if (!is_post_target(entry)) {
      filtered.push_back(entry);
    }
  }
  writer.add_shard(std::span<const EntrySnapshot>(filtered));
  return true;
}

struct ShardChunkGroup {
  size_t shard_begin = 0;
  size_t shard_end = 0;
  const vmemkv::T1ChunkMeta *prev_chunk = nullptr;
  bool dirty = false;
};

template <typename CheckpointContext>
auto map_prev_chunks_to_shards(std::span<const vmemkv::T1ChunkMeta> prev_chunks,
                               const CheckpointContext &context) -> std::vector<ShardChunkGroup> {
  const size_t total = context.shard_count();
  const auto bounds = context.boundaries();
  if (prev_chunks.empty()) {
    return {{0, total, nullptr, true}};
  }

  std::vector<ShardChunkGroup> groups;
  groups.reserve(prev_chunks.size());
  size_t cursor = 0;
  for (const auto &prev : prev_chunks) {
    const size_t begin = cursor;
    if (prev.has_upper_bound == 0) {
      cursor = total;
    } else {
      while (cursor + 1 < total && !(prev.upper_bound < bounds[cursor])) {
        ++cursor;
      }
      if (cursor == begin || cursor >= total || !(bounds[cursor - 1] == prev.upper_bound)) {
        return {{0, total, nullptr, true}};
      }
    }
    bool dirty = false;
    for (size_t shard_idx = begin; shard_idx < cursor; ++shard_idx) {
      if (context.shard_is_dirty(shard_idx)) {
        dirty = true;
        break;
      }
    }
    groups.push_back({begin, cursor, &prev, dirty});
  }
  return cursor == total ? groups : std::vector<ShardChunkGroup>{{0, total, nullptr, true}};
}

inline auto should_coalesce_with_next(uint64_t cur_bytes,
                                      bool from_open_writer,
                                      const ShardChunkGroup &next,
                                      uint64_t split_bytes) noexcept -> bool {
  const uint64_t merge_bytes = split_bytes / 2;
  const uint64_t tiny_bytes = std::max<uint64_t>(1, split_bytes / 8);
  if (next.prev_chunk == nullptr || next.dirty) {
    return cur_bytes < (from_open_writer ? merge_bytes : tiny_bytes);
  }
  const uint64_t combined = cur_bytes + next.prev_chunk->approx_bytes;
  return combined <= merge_bytes || (cur_bytes < tiny_bytes && combined <= split_bytes);
}

struct T1ChunkEmitter {
  const std::filesystem::path &t1_dir;
  uint64_t target;
  uint64_t split_bytes;
  uint64_t next_chunk_id;
  std::chrono::nanoseconds &t1_write_phase;
  std::unique_ptr<vmemkv::ShardedT1CheckpointWriter> writer;
  uint64_t active_chunk_id = 0;
  std::vector<vmemkv::T1ChkKeyPrefix> internal_boundaries;
  std::vector<vmemkv::T1ChunkMeta> staged_chunks;
  std::vector<uint64_t> written_chunk_ids;
  std::vector<size_t> persisted_shards;

  void open_if_needed() {
    if (writer != nullptr) {
      return;
    }
    active_chunk_id = next_chunk_id++;
    written_chunk_ids.push_back(active_chunk_id);
    writer = std::make_unique<vmemkv::ShardedT1CheckpointWriter>(vmemkv::derive_t1_chunk_path(t1_dir, active_chunk_id));
    internal_boundaries.clear();
  }

  void close_chunk(bool has_upper, const vmemkv::T1ChkKeyPrefix &upper) {
    const auto write_start = std::chrono::steady_clock::now();
    writer->finish(internal_boundaries);
    const uint64_t bytes = writer->bytes_written();
    t1_write_phase += std::chrono::steady_clock::now() - write_start;
    staged_chunks.push_back(
        {active_chunk_id, bytes, has_upper ? upper : vmemkv::T1ChkKeyPrefix{}, static_cast<uint8_t>(has_upper), {}});
    writer.reset();
    internal_boundaries.clear();
    active_chunk_id = 0;
  }
};

template <typename EntrySnapshot, typename CheckpointContext>
void emit_shard_group(const ShardChunkGroup &group,
                      const ShardChunkGroup *next_group,
                      CheckpointContext &context,
                      std::vector<EntrySnapshot> &dumped,
                      T1ChunkEmitter &emit) {
  for (size_t shard_idx = group.shard_begin; shard_idx < group.shard_end; ++shard_idx) {
    emit.open_if_needed();
    if (emit.writer->shard_count() > 0) {
      emit.internal_boundaries.push_back(context.boundaries()[shard_idx - 1]);
    }
    bool skipped_post_target = false;
    context.extract_shard(shard_idx, dumped, [&](std::span<const EntrySnapshot> entries) {
      const auto write_start = std::chrono::steady_clock::now();
      skipped_post_target = stream_filtered_shard(entries, emit.target, *emit.writer);
      emit.t1_write_phase += std::chrono::steady_clock::now() - write_start;
    });
    if (!skipped_post_target) {
      emit.persisted_shards.push_back(shard_idx);
    }
    if (shard_idx + 1 < context.shard_count() && emit.writer->bytes_written() >= emit.split_bytes) {
      emit.close_chunk(true, context.boundaries()[shard_idx]);
    }
  }
  if (emit.writer == nullptr) {
    return;
  }
  if (next_group != nullptr &&
      should_coalesce_with_next(emit.writer->bytes_written(), true, *next_group, emit.split_bytes)) {
    return;
  }
  const bool has_upper = (group.shard_end < context.shard_count());
  emit.close_chunk(has_upper, has_upper ? context.boundaries()[group.shard_end - 1] : vmemkv::T1ChkKeyPrefix{});
}

// Publishes only dirty/coalesced T1 chunks to `<t2_path>.t1chk/` and commits the atomic
// manifest; rolls back newly written chunk files on error without mutating committed state.
inline auto publish_t1_checkpoint_and_manifest(CheckpointState &state,
                                               ShardedT1Index &tier1,
                                               const std::filesystem::path &t2_path,
                                               uint64_t checkpoint_lsn,
                                               uint64_t target,
                                               std::chrono::nanoseconds &t1_write_phase,
                                               uint64_t t1_chunk_split_bytes = T1Config::kT1ChunkSplitBytes)
    -> std::chrono::nanoseconds {
  using EntrySnapshot = ShardedT1Index::EntrySnapshot;
  const std::filesystem::path t1_dir = vmemkv::derive_t1_chk_path(t2_path);
  std::filesystem::create_directories(t1_dir);

  const auto start = std::chrono::steady_clock::now();
  std::chrono::nanoseconds t1_elapsed{0};

  tier1.with_checkpoint_shards([&](auto &context) {
    T1ChunkEmitter emit{
        t1_dir, target, t1_chunk_split_bytes, state.next_t1_chunk_id, t1_write_phase, {}, 0, {}, {}, {}, {}};
    try {
      const auto groups = map_prev_chunks_to_shards(state.t1_chunks, context);
      emit.staged_chunks.reserve(groups.size());
      std::vector<EntrySnapshot> dumped;

      for (size_t g_idx = 0; g_idx < groups.size(); ++g_idx) {
        const ShardChunkGroup &group = groups[g_idx];
        const ShardChunkGroup *next_group = (g_idx + 1 < groups.size()) ? &groups[g_idx + 1] : nullptr;
        const bool coalesce_clean =
            group.prev_chunk != nullptr && next_group != nullptr &&
            should_coalesce_with_next(group.prev_chunk->approx_bytes, false, *next_group, t1_chunk_split_bytes);
        if (emit.writer == nullptr && !group.dirty && !coalesce_clean) {
          emit.staged_chunks.push_back(*group.prev_chunk);
          continue;
        }
        emit_shard_group<EntrySnapshot>(group, next_group, context, dumped, emit);
      }

      if (!emit.written_chunk_ids.empty()) {
        const auto sync_start = std::chrono::steady_clock::now();
        vmemkv::fsync_directory(t1_dir);
        t1_write_phase += std::chrono::steady_clock::now() - sync_start;
      }
      t1_elapsed = std::chrono::steady_clock::now() - start;

      g_checkpoint_debug_phase.store("checkpoint: write_manifest", std::memory_order_release);
      vmemkv::write_manifest(
          vmemkv::derive_manifest_path(t2_path), checkpoint_lsn, target, emit.staged_chunks, emit.next_chunk_id);

      state.t1_chunks = std::move(emit.staged_chunks);
      state.next_t1_chunk_id = emit.next_chunk_id;
      for (const size_t shard_idx : emit.persisted_shards) {
        context.mark_persisted(shard_idx);
      }
      if (!emit.written_chunk_ids.empty()) {
        vmemkv::cleanup_stale_t1_chunks(t1_dir, state.t1_chunks);
      }
    } catch (...) {
      for (const uint64_t new_id : emit.written_chunk_ids) {
        remove_quiet(vmemkv::derive_t1_chunk_path(t1_dir, new_id));
      }
      throw;
    }
  });
  return t1_elapsed;
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
  uint64_t old_read_only_offset = 0;
  uint64_t target = 0;
};

// Published after checkpoint_count, so a poller waking on it always sees this cycle's own
// numbers, never a torn mix with the next cycle's.
inline void publish_cycle_stats(CheckpointStats &stats,
                                uint64_t rotate_leader_wait_us,
                                const CheckpointPhases &phases) {
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
  timings.last_checkpoint_bytes_synced = phases.target - phases.old_read_only_offset;
  timings.last_checkpoint_corpus_bytes = phases.target;
  publish_checkpoint_stats(stats, timings);
}

// Claims the target range before msync() reads it, then drains in-flight in-place writes
// against the old fuzzy watermark. Without the claim, a racing update could be msync()'d mid-write
// (torn on disk) or land after read_only_offset publishes past it.
inline auto claim_and_drain_watermark(T2Ownership &own, uint64_t target) -> std::chrono::nanoseconds {
  own.claim_watermark(target);
  const auto barrier_start = std::chrono::steady_clock::now();
  own.drain_in_place(target);
  return std::chrono::steady_clock::now() - barrier_start;
}

// Publishes read_only_offset, then rotates the WAL. No checkpoint_lsn needed (docs/specification/low_level_design.md
// 5.5): rotation only retires the previous cycle's generation, already covered by this cycle's
// manifest by construction.
inline auto publish_boundary_and_rotate(T2FlatFile &tier2, Wal &wal, uint64_t target) -> std::chrono::nanoseconds {
  tier2.get_memory()->read_only_offset.store(target, std::memory_order_release);
  const auto rotate_start = std::chrono::steady_clock::now();
  wal.rotate_segment();
  return std::chrono::steady_clock::now() - rotate_start;
}

// Durabilizes T2's live mutable region grown since the last cycle, advances read_only_offset in
// place, and persists a manifest-committed checkpoint.
template <typename PreStopHook, typename PreFinishHook>
  requires CheckpointHook<PreStopHook> && CheckpointHook<PreFinishHook>
void checkpoint_internal(CheckpointState &state,
                         ShardedT1Index &tier1,
                         T2FlatFile &tier2,
                         Wal &wal,
                         T2Ownership &own,
                         const std::filesystem::path &t2_path,
                         const StoreOptions &options,
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
    msync_target_range(mem, checkpoint_target.old_read_only_offset, checkpoint_target.target);
    msync_phase = std::chrono::steady_clock::now() - msync_start;

    pre_finish_hook();

    g_checkpoint_debug_phase.store("checkpoint: publish_t1_checkpoint", std::memory_order_release);
    t1_phase = publish_t1_checkpoint_and_manifest(
        state, tier1, t2_path, checkpoint_lsn, checkpoint_target.target, t1_write_phase, options.t1_chunk_split_bytes);
  } catch (...) {
    // Nothing durable happened yet (read_only_offset hasn't advanced) -- un-claim so in-place
    // updates aren't blocked forever for an unpublished cycle.
    own.revert_watermark(checkpoint_target.old_read_only_offset);
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
  phases.old_read_only_offset = checkpoint_target.old_read_only_offset;
  phases.target = checkpoint_target.target;
  publish_cycle_stats(state.stats, wal.last_rotate_leader_wait_us(), phases);
  // Punch eligibility trails checkpoint coverage: a segment evacuated at count N is punchable only
  // after a checkpoint completing past N. Recorded here at the single choke point for every cycle.
  own.note_checkpoint_completed(state.stats.checkpoint_count.load(std::memory_order_relaxed));
}

// Shared wait/CAS/coalesce loop for foreground checkpoint(). Coalesces if another concurrent
// cycle completed a checkpoint while we waited.
template <typename PreStopHook, typename PreFinishHook>
  requires CheckpointHook<PreStopHook> && CheckpointHook<PreFinishHook>
void run_checkpoint(CheckpointState &state,
                    ShardedT1Index &tier1,
                    T2FlatFile &tier2,
                    Wal &wal,
                    T2Ownership &own,
                    const std::filesystem::path &t2_path,
                    const StoreOptions &options,
                    PreStopHook pre_stop_hook,
                    PreFinishHook pre_finish_hook) {
  const uint64_t start_chk_count = state.stats.checkpoint_count.load(std::memory_order_acquire);

  while (true) {
    wait_until_checkpoint_not_running(state);

    if (state.stats.checkpoint_count.load(std::memory_order_acquire) > start_chk_count) {
      return;
    }

    if (auto guard = SingleFlightGuard::try_acquire(state.running); guard.holds) {
      checkpoint_internal(state, tier1, tier2, wal, own, t2_path, options, pre_stop_hook, pre_finish_hook);
      return;
    }
  }
}

}  // namespace checkpoint_detail
}  // namespace vmemkv
