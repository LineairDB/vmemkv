// compaction.hpp - Unified T1-shard repack (4KiB page locality + T2 segment defragmentation)
// and segment hole-punch.
//
// Each T1 shard is sized to at most one T2 segment (8MiB). When a shard is queued dirty (at
// insert time or at startup) or overlaps a frozen garbage segment (< 50% live bytes),
// repack_dirty_shard() rewrites the shard's live entries contiguously in key order to the T2
// tail_offset frontier. Fully evacuated segments below read_only_offset are hole-punched after
// two-checkpoint coverage.
#pragma once

#include <fcntl.h>
#include <linux/magic.h>
#include <sys/statfs.h>
#include <sys/vfs.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <optional>
#include <span>
#include <vector>
#include <vmemkv/config.hpp>
#include <vmemkv/core/env.hpp>
#include <vmemkv/core/paths.hpp>
#include <vmemkv/core/single_flight.hpp>
#include <vmemkv/core/utils.hpp>
#include <vmemkv/impl/read_path.hpp>
#include <vmemkv/impl/t2_ownership.hpp>
#include <vmemkv/t1_index/sharded_t1_index.hpp>
#include <vmemkv/t1_index/t1_codec.hpp>
#include <vmemkv/t1_index/t1_index.hpp>
#include <vmemkv/t2_flat_file/t2_flat_file.hpp>

namespace vmemkv {

inline auto auto_compact_suppressed() noexcept -> bool {
  // Empty counts as unset: getenv_view() returns empty for unset or empty alike.
  // Tests set this (see test_main.cpp) so only the compaction tests run the worker.
  return !getenv_view("VMEMKV_SUPPRESS_AUTO_COMPACT").empty();
}

struct CompactionState {
  std::atomic<bool> auto_compact_enabled{true};
  std::atomic<bool> running{false};
  std::atomic<uint64_t> cycle_count{0};
  std::atomic<bool> punch_probed{false};
  std::atomic<bool> punch_supported{true};
  std::atomic<uint64_t> probe_segment_visits{0};
  std::atomic<uint64_t> probe_relocated{0};
  std::atomic<uint64_t> probe_stale{0};
  std::atomic<uint64_t> probe_moved_bytes{0};
  std::atomic<uint64_t> probe_sort_candidates{0};
  std::atomic<uint64_t> probe_punched_segments{0};
  std::atomic<uint64_t> probe_punched_bytes{0};
  std::atomic<uint64_t> probe_relocate_ns{0};
  std::atomic<uint64_t> probe_validate_ns{0};
  std::atomic<uint64_t> probe_movedata_ns{0};
  std::atomic<uint64_t> probe_publish_ns{0};
  std::atomic<uint64_t> probe_cycle_ns{0};
  std::atomic<uint64_t> probe_punch_ns{0};
  std::atomic<uint64_t> probe_collect_ns{0};
  std::atomic<uint64_t> probe_prefetch_ns{0};
  std::atomic<uint64_t> probe_dirty_pass_ns{0};
  std::atomic<uint64_t> probe_garbage_pass_ns{0};
  std::atomic<uint64_t> probe_cycles_ran{0};
};

namespace compaction_detail {

inline auto segment_probe_enabled() noexcept -> bool {
  static const bool enabled = !getenv_view("VMEMKV_SEGMENT_PROBE").empty();
  return enabled;
}

// Probes hole-punch support once per process (fstatfs magic; tmpfs cannot punch).
inline auto punch_supported(CompactionState &state, const std::filesystem::path &t2_path) -> bool {
  if (!state.punch_probed.load(std::memory_order_relaxed)) {
    struct statfs sfs {};
    bool supported = true;
    if (::statfs(derive_t2_chk_path(t2_path).c_str(), &sfs) == 0) {
      supported = (sfs.f_type != TMPFS_MAGIC);
    }
    state.punch_supported.store(supported, std::memory_order_relaxed);
    state.punch_probed.store(true, std::memory_order_relaxed);
  }
  return state.punch_supported.load(std::memory_order_relaxed);
}

// Punches fully-evacuated 8MiB segments below read_only_offset (seg_live == 0).
// Boundary-straddling records credit both segments in T2Ownership, so seg_live == 0
// guarantees no live record starts in or overhangs into the segment. Stateless across
// restarts: T2Ownership rebuilds seg_live from the checkpoint + WAL on startup.
//
// Punch additionally requires checkpoint coverage of the evacuation: a candidate
// becomes eligible only after two checkpoints complete past its evacuation epoch.
// One is not enough: an evacuation that finishes after a checkpoint's writer-stop
// resumes can miss that checkpoint's T1 dump (writers run concurrently with the
// dump, and only the WAL covers the gap -- which relocations deliberately lack),
// so only the second subsequent checkpoint is guaranteed to hold the
// post-relocation pointers while its msync covers the relocated bytes. Until then
// the pre-relocation copies stay readable through the checkpointed T1, which is
// what makes relocation records unnecessary in the WAL.
inline auto punch_evacuated_segments(CompactionState &state,
                                     const T2Ownership &own,
                                     const T2FlatFile &tier2,
                                     const std::filesystem::path &t2_path) -> uint64_t {
  if (!punch_supported(state, t2_path)) {
    return 0;
  }
  const uint64_t segment_bytes = own.segment_bytes();
  const vmemkv::T2Memory *mem = tier2.get_memory();
  const uint64_t read_only_offset = mem->read_only_offset.load(std::memory_order_acquire);
  const uint64_t covered_through = own.last_checkpoint_count();
  const std::vector<T2Ownership::PunchCandidate> candidates = own.take_punch_candidates();
  // Deferred, not dropped: above-read-only cohorts wait for checkpoint coverage, and
  // epoch-covered ones wait for the segment to empty. A live reading here means
  // a concurrent append landed after evacuation (immutable-region bytes never gain
  // live records); dropping the candidacy is safe because emptying re-queues it
  // through the zero-crossing path.
  std::vector<T2Ownership::PunchCandidate> deferred;
  uint64_t punched = 0;
  for (size_t idx = 0; idx < candidates.size(); ++idx) {
    const auto &candidate = candidates[idx];
    const uint64_t seg = candidate.segment;
    if (candidate.evacuated_epoch + 1 >= covered_through || (seg + 1) * segment_bytes > read_only_offset) {
      deferred.push_back(candidate);
      continue;
    }
    if (own.seg_live(seg) != 0) {
      own.retire_punch_candidates(1);
      continue;
    }
    switch (tier2.punch_if_occupied(seg * segment_bytes, segment_bytes)) {
      case vmemkv::T2FlatFile::PunchOutcome::Punched:
        ++punched;
        own.retire_punch_candidates(1);
        if (segment_probe_enabled()) {
          state.probe_punched_segments.fetch_add(1, std::memory_order_relaxed);
          state.probe_punched_bytes.fetch_add(segment_bytes, std::memory_order_relaxed);
        }
        break;
      case vmemkv::T2FlatFile::PunchOutcome::AlreadyHollow:
        own.retire_punch_candidates(1);
        break;
      case vmemkv::T2FlatFile::PunchOutcome::Failed:
        state.punch_supported.store(false, std::memory_order_relaxed);
        own.retire_punch_candidates(candidates.size() - idx);
        own.requeue_deferred_punch_candidates(deferred);
        return punched;
    }
  }
  own.requeue_deferred_punch_candidates(deferred);
  return punched;
}

// Builds the full-key-ordered list of live T2 offsets for one shard's EntrySnapshot list.
// `entries` is already sorted by 16-byte key prefix; T2 is read only when consecutive entries
// share an identical 16-byte prefix.
template <typename EntrySnapshot>
auto collect_shard_payloads(std::span<const EntrySnapshot> entries,
                            const vmemkv::T2Memory *mem) -> std::vector<uint64_t> {
  std::vector<uint64_t> payloads;
  payloads.reserve(entries.size());
  for (size_t i = 0; i < entries.size();) {
    if (entries[i].payload_bits == vmemkv::STORE_NOT_FOUND || t1_detail::is_inline(entries[i].hash)) {
      ++i;
      continue;
    }
    const size_t group_start = payloads.size();
    const auto &prefix = entries[i].key;
    payloads.push_back(entries[i].payload_bits);
    ++i;
    while (i < entries.size() && entries[i].key == prefix) {
      if (entries[i].payload_bits != vmemkv::STORE_NOT_FOUND && !t1_detail::is_inline(entries[i].hash)) {
        payloads.push_back(entries[i].payload_bits);
      }
      ++i;
    }
    if (payloads.size() - group_start <= 1) {
      continue;
    }
    auto group_begin = payloads.begin() + static_cast<std::ptrdiff_t>(group_start);
    std::sort(group_begin, payloads.end(), [&](uint64_t lhs, uint64_t rhs) {
      const uint64_t loff = t2_detail::decode_offset(lhs);
      const uint64_t roff = t2_detail::decode_offset(rhs);
      const std::span<const std::byte> lkey = T2FlatFile::at(loff, mem).key;
      const std::span<const std::byte> rkey = T2FlatFile::at(roff, mem).key;
      if (const auto ord = std::lexicographical_compare_three_way(lkey.begin(), lkey.end(), rkey.begin(), rkey.end());
          ord != 0) {
        return ord < 0;
      }
      return loff > roff;
    });
    payloads.erase(std::unique(group_begin,
                               payloads.end(),
                               [&](uint64_t lhs, uint64_t rhs) {
                                 return std::ranges::equal(T2FlatFile::at(t2_detail::decode_offset(lhs), mem).key,
                                                           T2FlatFile::at(t2_detail::decode_offset(rhs), mem).key);
                               }),
                   payloads.end());
  }
  return payloads;
}

inline auto payloads_broken_page_locality(std::span<const uint64_t> payloads) -> bool {
  for (size_t i = 1; i < payloads.size(); ++i) {
    if (!t2_detail::are_t2_payloads_page_local(payloads[i - 1], payloads[i])) {
      return true;
    }
  }
  return false;
}

template <typename EntrySnapshot>
auto shard_pages_local(std::span<const EntrySnapshot> entries, const vmemkv::T2Memory *mem) -> bool {
  return !payloads_broken_page_locality(collect_shard_payloads(entries, mem));
}

// Relocates one live record (by current T2 offset) to the tail_offset frontier.
// Returns the moved record's aligned byte size, or 0 when the entry moved concurrently,
// vanished, or went inline. When probe is non-null and the segment probe is enabled,
// accumulates validation (stripe lock + T1 recheck), data movement (T2 append),
// and publish (T1 put) phase timings. No WAL record: the same version is rewritten
// elsewhere, and the punch gate keeps the pre-relocation copies readable through
// the checkpointed T1 until checkpoint coverage.
template <typename LockStripe>
auto relocate_one(ShardedT1Index &tier1,
                  T2FlatFile &tier2,
                  T2Ownership &own,
                  LockStripe &&lock_stripe,
                  uint64_t target_payload,
                  const vmemkv::T2Memory *mem,
                  ReadPolicy read_policy = ReadPolicy::Adaptive,
                  CompactionState *probe = nullptr) -> uint64_t {
  const bool measure = probe != nullptr && segment_probe_enabled();
  const uint64_t old_off = t2_detail::decode_offset(target_payload);

  // Reuse the immutable-region read path (mmap for small/resident records, bounded pread() for
  // non-resident multi-page records below read_only_offset) with no stripe lock held across I/O;
  // records still in the mutable region fall back to T2FlatFile::at().
  thread_local static std::vector<std::byte> tl_compact_cold_buf;
  const auto read_start = measure ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  const std::optional<T2RecordView> preloaded =
      try_read_immutable_record(mem, target_payload, ImmutableReader::kGet, &tl_compact_cold_buf, read_policy);
  const auto read_end = measure ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

  const auto validate_start = measure ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  const std::span<const std::byte> key = preloaded.has_value() ? preloaded->key : T2FlatFile::at(old_off, mem).key;

  std::lock_guard<std::mutex> key_lock(lock_stripe(key));
  const auto cur = tier1.get_with_hash(key);
  if (cur.payload_bits == vmemkv::STORE_NOT_FOUND || t1_detail::is_inline(cur.raw_hash) ||
      t2_detail::decode_offset(cur.payload_bits) != old_off) {
    return 0;
  }
  const auto validate_end = measure ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  const std::span<const std::byte> value_span =
      preloaded.has_value() ? preloaded->value : T2FlatFile::at(old_off, mem).value;
  T2FlatFile::T2MemoryHandle handle = tier2.acquire_write_handle();
  const uint64_t offset = vmemkv::T2FlatFile::append_default(handle, key, value_span);
  const auto movedata_end = measure ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  const uint64_t block_count = t2_detail::checked_block_count(key.size(), value_span.size());
  const uint64_t encoded = t2_detail::encode_offset_payload(offset, block_count);
  tier1.put(key, encoded, false, 0, /*mark_t2_dirty=*/false);
  own.note_replace(cur.payload_bits, cur.raw_hash, /*new_is_offset=*/true, encoded, /*allow_mark_dirty=*/false);
  if (measure) {
    const auto publish_end = std::chrono::steady_clock::now();
    probe->probe_validate_ns.fetch_add(
        static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(validate_end - validate_start).count()),
        std::memory_order_relaxed);
    probe->probe_movedata_ns.fetch_add(
        static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(read_end - read_start).count() +
            std::chrono::duration_cast<std::chrono::nanoseconds>(movedata_end - validate_end).count()),
        std::memory_order_relaxed);
    probe->probe_publish_ns.fetch_add(
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(publish_end - movedata_end).count()),
        std::memory_order_relaxed);
  }
  return vmemkv::align_up(sizeof(ValueRecordHeader) + key.size() + value_span.size());
}

struct CompactionProgress {
  uint64_t moved_bytes = 0;
  bool did_relocate = false;
};

inline auto entries_overlap_garbage_segment(std::span<const T1Index::EntrySnapshot> entries,
                                            const T2Ownership &own,
                                            // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                            uint64_t base,
                                            uint64_t used) -> bool {
  return std::ranges::any_of(entries, [&](const T1Index::EntrySnapshot &entry) {
    return entry.payload_bits != vmemkv::STORE_NOT_FOUND && !t1_detail::is_inline(entry.hash) &&
           own.payload_overlaps_garbage_segment(entry.payload_bits, base, used);
  });
}

inline auto payloads_overlap_garbage_segment(std::span<const uint64_t> payloads,
                                             const T2Ownership &own,
                                             // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                             uint64_t base,
                                             uint64_t used) -> bool {
  return std::ranges::any_of(
      payloads, [&](uint64_t payload) { return own.payload_overlaps_garbage_segment(payload, base, used); });
}

// Batches, page-aligns, sorts, and merges T2 byte ranges for a chunk of payload handles,
// then issues MADV_WILLNEED so cold NVMe pages fault in concurrently rather than synchronously
// inside relocate_one().
inline void prefetch_payload_batch(std::span<const uint64_t> batch,
                                   const vmemkv::T2Memory *mem,
                                   uint64_t upper_bound) noexcept {
  if (batch.empty() || mem == nullptr || mem->reads.random_mmap() == nullptr) {
    return;
  }
  const uint64_t limit = std::min(upper_bound, mem->reads.capacity());
  const uint64_t page_size = t2_detail::os_page_size();
  const uint64_t page_mask = page_size - 1;
  struct PageRange {
    uint64_t start;
    uint64_t end;
  };
  std::vector<PageRange> ranges;
  ranges.reserve(batch.size());
  for (uint64_t payload : batch) {
    const uint64_t off = t2_detail::decode_offset(payload);
    if (off >= limit) {
      continue;
    }
    const uint64_t hint = t2_detail::payload_hint_bytes(payload, t2_detail::kImmutableSizeHintMargin);
    const uint64_t raw_end = off + std::min<uint64_t>(hint, limit - off);
    const uint64_t pg_start = off & ~page_mask;
    const uint64_t pg_end = std::min<uint64_t>((raw_end + page_mask) & ~page_mask, mem->reads.capacity());
    if (pg_end > pg_start) {
      ranges.push_back({pg_start, pg_end});
    }
  }
  if (ranges.empty()) {
    return;
  }
  std::sort(ranges.begin(), ranges.end(), [](const PageRange &lhs, const PageRange &rhs) {
    return lhs.start < rhs.start || (lhs.start == rhs.start && lhs.end < rhs.end);
  });
  uint64_t cur_start = ranges[0].start;
  uint64_t cur_end = ranges[0].end;
  for (size_t i = 1; i < ranges.size(); ++i) {
    if (ranges[i].start <= cur_end) {
      cur_end = std::max(cur_end, ranges[i].end);
      continue;
    }
    mem->reads.prefetch(cur_start, cur_end - cur_start);
    cur_start = ranges[i].start;
    cur_end = ranges[i].end;
  }
  mem->reads.prefetch(cur_start, cur_end - cur_start);
}

inline constexpr size_t kPrefetchBatchSize = 256;

inline void prefetch_target_chunk(CompactionState &state,
                                  std::span<const uint64_t> targets,
                                  size_t offset,
                                  const vmemkv::T2Memory *mem,
                                  uint64_t upper_bound,
                                  bool measure) noexcept {
  if (offset >= targets.size()) {
    return;
  }
  const auto prefetch_start = measure ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  const size_t len = std::min(kPrefetchBatchSize, targets.size() - offset);
  prefetch_payload_batch(targets.subspan(offset, len), mem, upper_bound);
  if (measure) {
    state.probe_prefetch_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                                std::chrono::steady_clock::now() - prefetch_start)
                                                                .count()),
                                      std::memory_order_relaxed);
  }
}

template <typename LockStripe>
void relocate_targets_pipelined(CompactionState &state,
                                std::span<const uint64_t> targets,
                                ShardedT1Index &tier1,
                                T2FlatFile &tier2,
                                T2Ownership &own,
                                LockStripe &&lock_stripe,
                                const vmemkv::T2Memory *mem,
                                uint64_t upper_bound,
                                ReadPolicy read_policy,
                                CompactionProgress &progress,
                                bool measure) {
  prefetch_target_chunk(state, targets, 0, mem, upper_bound, measure);
  for (size_t batch_start = 0; batch_start < targets.size(); batch_start += kPrefetchBatchSize) {
    prefetch_target_chunk(state, targets, batch_start + kPrefetchBatchSize, mem, upper_bound, measure);
    const size_t batch_end = std::min(batch_start + kPrefetchBatchSize, targets.size());
    for (size_t i = batch_start; i < batch_end; ++i) {
      const auto relocation_start =
          measure ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
      const uint64_t moved = relocate_one(tier1, tier2, own, lock_stripe, targets[i], mem, read_policy, &state);
      if (moved != 0) {
        progress.moved_bytes += moved;
        progress.did_relocate = true;
      }
      if (measure) {
        (moved == 0 ? state.probe_stale : state.probe_relocated).fetch_add(1, std::memory_order_relaxed);
        state.probe_moved_bytes.fetch_add(moved, std::memory_order_relaxed);
        state.probe_relocate_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                                    std::chrono::steady_clock::now() - relocation_start)
                                                                    .count()),
                                          std::memory_order_relaxed);
      }
    }
  }
}

template <typename LockStripe>
void repack_dirty_shard(CompactionState &state,
                        std::span<const T1Index::EntrySnapshot> entries,
                        ShardedT1Index &tier1,
                        T2FlatFile &tier2,
                        T2Ownership &own,
                        LockStripe &&lock_stripe,
                        const vmemkv::T2Memory *mem,
                        ReadPolicy read_policy,
                        CompactionProgress &progress,
                        bool check_page_locality = true) {
  const uint64_t read_only = mem->read_only_offset.load(std::memory_order_acquire);
  const uint64_t tail = mem->tail_offset.load(std::memory_order_acquire);
  if (!check_page_locality && !entries_overlap_garbage_segment(entries, own, read_only, tail)) {
    return;
  }

  const bool measure = segment_probe_enabled();
  const auto collect_start = measure ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  const std::vector<uint64_t> payloads = collect_shard_payloads(entries, mem);
  if (measure) {
    state.probe_collect_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                               std::chrono::steady_clock::now() - collect_start)
                                                               .count()),
                                     std::memory_order_relaxed);
  }
  const size_t count = payloads.size();
  if (count == 0) {
    return;
  }

  // Unified shard repack gate:
  //   1. When visiting a dirty shard with broken 4KiB page locality (scan locality repair),
  //      repack the entire shard in key order.
  //   2. When a shard is already page-local and only overlaps a frozen garbage segment
  //      (segment defragmentation), relocate the entries overlapping garbage segments in key
  //      order without rewriting healthy segments.
  const bool repack_entire_shard = check_page_locality && payloads_broken_page_locality(payloads);
  if (!repack_entire_shard && !payloads_overlap_garbage_segment(payloads, own, read_only, tail)) {
    return;
  }

  if (measure) {
    state.probe_segment_visits.fetch_add(1, std::memory_order_relaxed);
    state.probe_sort_candidates.fetch_add(count, std::memory_order_relaxed);
  }

  std::vector<uint64_t> filtered_targets;
  std::span<const uint64_t> targets = payloads;
  if (!repack_entire_shard) {
    filtered_targets.reserve(count);
    for (uint64_t payload : payloads) {
      if (own.payload_overlaps_garbage_segment(payload, read_only, tail)) {
        filtered_targets.push_back(payload);
      }
    }
    targets = filtered_targets;
  }
  if (!targets.empty()) {
    relocate_targets_pipelined(
        state, targets, tier1, tier2, own, lock_stripe, mem, tail, read_policy, progress, measure);
  }
}

// Runs one compaction cycle: punches evacuated segments and repacks shards that either
// have broken 4KiB page locality or overlap frozen garbage segments.
template <typename LockStripe>
auto compact_cycle(CompactionState &state,
                   ShardedT1Index &tier1,
                   T2FlatFile &tier2,
                   T2Ownership &own,
                   const std::filesystem::path &t2_path,
                   LockStripe &&lock_stripe,
                   const StoreOptions &options = {}) -> bool {
  auto running_guard = SingleFlightGuard::try_acquire(state.running);
  if (!running_guard.holds) {
    return false;
  }

  const bool measure = segment_probe_enabled();
  const auto cycle_start = measure ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  uint64_t punched = punch_evacuated_segments(state, own, tier2, t2_path);
  const auto punch_end = measure ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

  const vmemkv::T2Memory *mem = tier2.get_memory();
  const uint64_t sync_begin = mem != nullptr ? mem->tail_offset.load(std::memory_order_acquire) : 0;
  CompactionProgress progress;

  using EntrySnapshot = T1Index::EntrySnapshot;
  const bool had_dirty_shard = tier1.has_dirty_shard();
  if (had_dirty_shard) {
    tier1.for_each_dirty_shard([&](std::span<const EntrySnapshot> entries) -> bool {
      repack_dirty_shard(state, entries, tier1, tier2, own, lock_stripe, mem, options.read_policy, progress);
      return progress.moved_bytes < T2Config::kCompactionMaxMoveBytesPerCycle;
    });
  }
  const auto dirty_end = measure ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

  if (progress.moved_bytes < T2Config::kCompactionMaxMoveBytesPerCycle) {
    const uint64_t read_only = mem->read_only_offset.load(std::memory_order_acquire);
    const uint64_t tail = mem->tail_offset.load(std::memory_order_acquire);
    if (own.has_garbage_segment(read_only, tail)) {
      const bool completed_all_shards =
          tier1.for_each_shard_for_compaction([&](std::span<const EntrySnapshot> entries) -> bool {
            repack_dirty_shard(state,
                               entries,
                               tier1,
                               tier2,
                               own,
                               lock_stripe,
                               mem,
                               options.read_policy,
                               progress,
                               /*check_page_locality=*/false);
            return progress.moved_bytes < T2Config::kCompactionMaxMoveBytesPerCycle;
          });
      if (completed_all_shards) {
        own.clear_garbage_segments(read_only, tail);
      }
    }
  }
  // Flush this cycle's relocated T2 range to disk without waiting for background writeback
  // timers, keeping dirty page-cache buildup bounded and naturally pacing compaction to
  // storage write bandwidth. Crash durability stays governed by the WAL and checkpoint msync().
  if (progress.did_relocate && mem != nullptr && mem->reads.pread_fd() >= 0) {
    const uint64_t sync_end = mem->tail_offset.load(std::memory_order_acquire);
    if (sync_end > sync_begin) {
      (void)::sync_file_range(mem->reads.pread_fd(),
                              static_cast<off_t>(sync_begin),
                              static_cast<off_t>(sync_end - sync_begin),
                              SYNC_FILE_RANGE_WAIT_BEFORE | SYNC_FILE_RANGE_WRITE | SYNC_FILE_RANGE_WAIT_AFTER);
    }
  }
  if (measure) {
    const auto cycle_end = std::chrono::steady_clock::now();
    auto to_ns = [](auto begin, auto end) -> uint64_t {
      return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
    };
    state.probe_cycles_ran.fetch_add(1, std::memory_order_relaxed);
    state.probe_punch_ns.fetch_add(to_ns(cycle_start, punch_end), std::memory_order_relaxed);
    state.probe_dirty_pass_ns.fetch_add(to_ns(punch_end, dirty_end), std::memory_order_relaxed);
    state.probe_garbage_pass_ns.fetch_add(to_ns(dirty_end, cycle_end), std::memory_order_relaxed);
    state.probe_cycle_ns.fetch_add(to_ns(cycle_start, cycle_end), std::memory_order_relaxed);
  }

  if (progress.did_relocate || punched > 0) {
    state.cycle_count.fetch_add(1, std::memory_order_relaxed);
  }
  return had_dirty_shard || progress.did_relocate || punched > 0;
}

}  // namespace compaction_detail
}  // namespace vmemkv
