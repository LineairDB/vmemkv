// compaction.hpp - T2 segment evacuation, dirty-shard order repair, and segment hole-punch.
//
// Replacements queue their old T2 segments and keys for evacuation. The worker relocates live
// values from one queued segment and current values of its replaced keys in key order. Fully
// evacuated segments below base_boundary are hole-punched. Dirty shards without queued garbage
// retain the key-order repair path (see docs/specification/t2_compaction_design.md).
#pragma once

#include <linux/magic.h>
#include <sys/vfs.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <vmemkv/api/utils.hpp>
#include <vmemkv/checkpoint/paths.hpp>
#include <vmemkv/core/background_poll.hpp>
#include <vmemkv/core/bytes.hpp>
#include <vmemkv/core/env.hpp>
#include <vmemkv/core/single_flight.hpp>
#include <vmemkv/impl/t2_ownership.hpp>
#include <vmemkv/t1_index/t1_codec.hpp>
#include <vmemkv/t2_flat_file/t2_flat_file.hpp>

namespace vmemkv {

inline auto auto_compact_suppressed() noexcept -> bool {
  // Empty counts as unset: getenv_view() returns empty for unset or empty alike.
  // Tests set this (see test_main.cpp) so only the compaction tests run the worker.
  return !getenv_view("VMEMKV_SUPPRESS_AUTO_COMPACT").empty();
}

struct CompactionState {
  std::atomic<bool> running{false};
  std::atomic<uint64_t> cycle_count{0};
  std::atomic<bool> punch_probed{false};
  std::atomic<bool> punch_supported{true};
  std::atomic<uint64_t> probe_segment_visits{0};
  std::atomic<uint64_t> probe_source_candidates{0};
  std::atomic<uint64_t> probe_tracked_keys{0};
  std::atomic<uint64_t> probe_tracked_live{0};
  std::atomic<uint64_t> probe_candidates_deduplicated{0};
  std::atomic<uint64_t> probe_relocated{0};
  std::atomic<uint64_t> probe_stale{0};
  std::atomic<uint64_t> probe_moved_bytes{0};
  std::atomic<uint64_t> probe_collect_ns{0};
  std::atomic<uint64_t> probe_scan_ns{0};
  std::atomic<uint64_t> probe_scan_entries{0};
  std::atomic<uint64_t> probe_scan_matched{0};
  std::atomic<uint64_t> probe_copy_bytes{0};
  std::atomic<uint64_t> probe_lookup_ns{0};
  std::atomic<uint64_t> probe_lookup_stale{0};
  std::atomic<uint64_t> probe_sort_ns{0};
  std::atomic<uint64_t> probe_sort_candidates{0};
  std::atomic<uint64_t> probe_t2_key_reads{0};
  std::atomic<uint64_t> probe_punched_segments{0};
  std::atomic<uint64_t> probe_punched_bytes{0};
  std::atomic<uint64_t> probe_relocate_ns{0};
  std::atomic<uint64_t> probe_validate_ns{0};
  std::atomic<uint64_t> probe_movedata_ns{0};
  std::atomic<uint64_t> probe_publish_ns{0};
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

// Punches fully-evacuated 8MiB segments below base_boundary (seg_live == 0).
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
template <typename ConfigT>
auto punch_evacuated_segments(CompactionState &state,
                              const T2Ownership<ConfigT> &own,
                              const T2FlatFile &tier2,
                              const std::filesystem::path &t2_path) -> uint64_t {
  if (!punch_supported(state, t2_path)) {
    return 0;
  }
  const vmemkv::T2Memory *mem = tier2.get_memory();
  const uint64_t base = mem->base_boundary.load(std::memory_order_acquire);
  const uint64_t covered_through = own.last_checkpoint_count();
  const std::vector<typename T2Ownership<ConfigT>::PunchCandidate> candidates = own.take_punch_candidates();
  // Deferred, not dropped: below-base cohorts wait for checkpoint coverage, and
  // epoch-covered ones wait for the segment to empty. A live reading here means
  // a concurrent append landed after evacuation (below-base bytes never gain
  // live records); dropping the candidacy is safe because emptying re-queues it
  // through the zero-crossing path.
  std::vector<typename T2Ownership<ConfigT>::PunchCandidate> deferred;
  uint64_t punched = 0;
  for (const auto &candidate : candidates) {
    const uint64_t seg = candidate.segment;
    if (candidate.evacuated_epoch + 1 >= covered_through || (seg + 1) * ConfigT::kSegmentBytes > base) {
      deferred.push_back(candidate);
      continue;
    }
    if (own.seg_live(seg) != 0) {
      continue;
    }
    switch (tier2.punch_if_occupied(seg * ConfigT::kSegmentBytes, ConfigT::kSegmentBytes)) {
      case vmemkv::T2FlatFile::PunchOutcome::Punched:
        ++punched;
        if (segment_probe_enabled()) {
          state.probe_punched_segments.fetch_add(1, std::memory_order_relaxed);
          state.probe_punched_bytes.fetch_add(ConfigT::kSegmentBytes, std::memory_order_relaxed);
        }
        break;
      case vmemkv::T2FlatFile::PunchOutcome::AlreadyHollow:
        break;
      case vmemkv::T2FlatFile::PunchOutcome::Failed:
        state.punch_supported.store(false, std::memory_order_relaxed);
        own.requeue_punch_candidates(deferred);
        return punched;
    }
  }
  own.requeue_punch_candidates(deferred);
  return punched;
}

// Builds the full-key-ordered list of live T2 offsets for one shard's EntrySnapshot list.
// `entries` is already sorted by 16-byte key prefix; T2 is read only when consecutive entries
// share an identical 16-byte prefix.
template <typename EntrySnapshot>
auto collect_shard_payloads(std::span<const EntrySnapshot> entries,
                            const vmemkv::T2Memory *mem) -> std::vector<uint64_t> {
  struct Item {
    t1_detail::StoreKey prefix{};
    uint64_t payload = 0;
  };
  std::vector<Item> live;
  live.reserve(entries.size());
  for (const auto &entry : entries) {
    if (entry.payload_bits == vmemkv::STORE_NOT_FOUND || t1_detail::is_inline(entry.hash)) {
      continue;
    }
    live.push_back(Item{entry.key, entry.payload_bits});
  }
  std::vector<uint64_t> payloads;
  payloads.reserve(live.size());
  for (size_t lo = 0; lo < live.size();) {
    size_t group_end = lo + 1;
    while (group_end < live.size() && live[group_end].prefix == live[lo].prefix) {
      ++group_end;
    }
    if (group_end - lo == 1) {
      payloads.push_back(live[lo].payload);
      lo = group_end;
      continue;
    }
    std::vector<uint64_t> group(group_end - lo);
    for (size_t i = lo; i < group_end; ++i) {
      group[i - lo] = live[i].payload;
    }
    std::sort(group.begin(), group.end(), [&](uint64_t lhs, uint64_t rhs) {
      const uint64_t loff = lhs & t2_detail::kPayloadOffsetMask;
      const uint64_t roff = rhs & t2_detail::kPayloadOffsetMask;
      const std::span<const std::byte> lkey = T2FlatFile::at(loff, mem).key;
      const std::span<const std::byte> rkey = T2FlatFile::at(roff, mem).key;
      if (const auto ord = std::lexicographical_compare_three_way(lkey.begin(), lkey.end(), rkey.begin(), rkey.end());
          ord != 0) {
        return ord < 0;
      }
      return loff > roff;
    });
    for (size_t i = 0; i < group.size(); ++i) {
      const uint64_t curr_off = group[i] & t2_detail::kPayloadOffsetMask;
      if (i == 0) {
        payloads.push_back(group[i]);
        continue;
      }
      const uint64_t prev_off = group[i - 1] & t2_detail::kPayloadOffsetMask;
      if (std::lexicographical_compare_three_way(T2FlatFile::at(prev_off, mem).key.begin(),
                                                 T2FlatFile::at(prev_off, mem).key.end(),
                                                 T2FlatFile::at(curr_off, mem).key.begin(),
                                                 T2FlatFile::at(curr_off, mem).key.end()) != 0) {
        payloads.push_back(group[i]);
      }
    }
    lo = group_end;
  }
  return payloads;
}

struct SegmentRelocationCandidate {
  uint64_t offset = 0;
  t1_detail::StoreKey prefix{};
  // Full key, present from the start for replaced keys and materialized from T2
  // on demand for source entries whose prefix collides with another candidate.
  std::vector<std::byte> full_key;
  bool has_full_key = false;
};

struct SegmentSourceIndex {
  std::unordered_map<uint64_t, std::vector<SegmentRelocationCandidate>> by_segment;
  uint64_t entries = 0;
  uint64_t matched = 0;
  uint64_t copy_bytes = 0;
};

// Single T1 scan serving every queued dirty segment. Keys are captured as
// 16-byte T1 prefixes; full T2 keys are read only for prefix-colliding groups
// during ordering. A straddling record is offered to each dirty segment it
// overlaps; relocate_one() revalidates the offset under the key lock, so a
// record moved for an earlier segment is skipped as stale for a later one.
template <typename ConfigT, typename T1>
auto build_source_index(T1 &tier1, const std::unordered_set<uint64_t> &dirty_set, bool measure) -> SegmentSourceIndex {
  SegmentSourceIndex index;
  std::array<std::byte, t1_detail::kPrefixBytes> lower{};
  std::array<std::byte, t1_detail::kPrefixBytes> upper{};
  upper.fill(std::byte{std::numeric_limits<unsigned char>::max()});
  tier1.scan(std::span<const std::byte>(lower),
             std::span<const std::byte>(upper),
             [&](std::span<const std::byte> index_key, uint64_t payload, uint64_t hash) {
               if (measure) {
                 ++index.entries;
               }
               if (payload == vmemkv::STORE_NOT_FOUND || t1_detail::is_inline(hash)) {
                 return;
               }
               const uint64_t off = payload & t2_detail::kPayloadOffsetMask;
               const uint64_t size_hint = t2_detail::payload_hint_bytes(payload, t2_detail::kBaseSizeHintMargin);
               const uint64_t first = T2Ownership<ConfigT>::seg_index(off);
               const uint64_t last = T2Ownership<ConfigT>::seg_index(off + size_hint - 1);
               const bool overlaps = dirty_set.contains(first) || (last != first && dirty_set.contains(last));
               if (!overlaps) {
                 return;
               }
               if (measure) {
                 ++index.matched;
                 index.copy_bytes += t1_detail::kPrefixBytes;
               }
               SegmentRelocationCandidate candidate;
               candidate.offset = off;
               std::copy_n(index_key.begin(), t1_detail::kPrefixBytes, candidate.prefix.begin());
               if (dirty_set.contains(first)) {
                 index.by_segment[first].push_back(candidate);
               }
               if (last != first && dirty_set.contains(last)) {
                 index.by_segment[last].push_back(candidate);
               }
             });
  return index;
}

struct T2MaterializationStats {
  uint64_t reads = 0;
  uint64_t bytes = 0;
};

// Orders candidates by 16-byte T1 prefix; only equal-prefix runs read T2 full
// keys, so distinct prefixes never touch T2. Same pattern as
// collect_shard_offsets() groups. Returns the deduplicated count removed and
// reports T2 full-key reads through the stats struct.
inline auto sort_dedup_candidates(std::vector<SegmentRelocationCandidate> &candidates,
                                  const vmemkv::T2Memory *mem,
                                  bool measure,
                                  T2MaterializationStats &t2_stats) -> uint64_t {
  std::sort(
      candidates.begin(), candidates.end(), [](const auto &lhs, const auto &rhs) { return lhs.prefix < rhs.prefix; });
  std::vector<SegmentRelocationCandidate> ordered;
  ordered.reserve(candidates.size());
  uint64_t removed = 0;
  for (size_t lo = 0; lo < candidates.size();) {
    size_t group_end = lo + 1;
    while (group_end < candidates.size() && candidates[group_end].prefix == candidates[lo].prefix) {
      ++group_end;
    }
    if (group_end - lo == 1) {
      ordered.push_back(std::move(candidates[lo]));
      lo = group_end;
      continue;
    }
    for (size_t i = lo; i < group_end; ++i) {
      if (!candidates[i].has_full_key) {
        const std::span<const std::byte> key = T2FlatFile::at(candidates[i].offset, mem).key;
        candidates[i].full_key.assign(key.begin(), key.end());
        candidates[i].has_full_key = true;
        if (measure) {
          ++t2_stats.reads;
          t2_stats.bytes += key.size();
        }
      }
    }
    std::sort(candidates.begin() + static_cast<std::ptrdiff_t>(lo),
              candidates.begin() + static_cast<std::ptrdiff_t>(group_end),
              [](const auto &lhs, const auto &rhs) {
                const int key_order = vmemkv::bytes_compare_3way(lhs.full_key, rhs.full_key);
                return key_order == 0 ? lhs.offset > rhs.offset : key_order < 0;
              });
    for (size_t i = lo; i < group_end; ++i) {
      if (!ordered.empty() && ordered.back().has_full_key &&
          vmemkv::bytes_equal(candidates[i].full_key, ordered.back().full_key)) {
        ++removed;
        continue;
      }
      ordered.push_back(std::move(candidates[i]));
    }
    lo = group_end;
  }
  candidates = std::move(ordered);
  return removed;
}

template <typename ConfigT, typename T1>
auto prepare_indexed_candidates(CompactionState &state,
                                [[maybe_unused]] T1 &tier1,
                                SegmentSourceIndex &source_index,
                                typename T2Ownership<ConfigT>::DirtySegment &work,
                                const vmemkv::T2Memory *mem,
                                bool measure) -> std::vector<SegmentRelocationCandidate> {
  std::vector<SegmentRelocationCandidate> candidates;
  auto source_it = source_index.by_segment.find(work.segment);
  if (source_it != source_index.by_segment.end()) {
    candidates = std::move(source_it->second);
  }
  if (measure) {
    state.probe_sort_candidates.fetch_add(candidates.size(), std::memory_order_relaxed);
    state.probe_segment_visits.fetch_add(1, std::memory_order_relaxed);
  }
  const auto sort_start = measure ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  T2MaterializationStats materialization;
  const uint64_t removed = sort_dedup_candidates(candidates, mem, measure, materialization);
  if (measure) {
    const auto sort_end = std::chrono::steady_clock::now();
    const uint64_t sort_ns =
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(sort_end - sort_start).count());
    state.probe_sort_ns.fetch_add(sort_ns, std::memory_order_relaxed);
    state.probe_collect_ns.fetch_add(sort_ns, std::memory_order_relaxed);
    state.probe_candidates_deduplicated.fetch_add(removed, std::memory_order_relaxed);
    state.probe_t2_key_reads.fetch_add(materialization.reads, std::memory_order_relaxed);
    state.probe_copy_bytes.fetch_add(materialization.bytes, std::memory_order_relaxed);
  }
  return candidates;
}

// Relocates one live record (by current T2 offset) to the append frontier.
// Returns the moved record's aligned byte size, or 0 when the entry moved concurrently,
// vanished, or went inline. When probe is non-null and the segment probe is enabled,
// accumulates validation (stripe lock + T1 recheck), data movement (T2 append),
// and publish (T1 put) phase timings. No WAL record: the same version is rewritten
// elsewhere, and the punch gate keeps the pre-relocation copies readable through
// the checkpointed T1 until checkpoint coverage.
template <typename ConfigT, typename T1, typename LockStripe, typename MaybeReorg>
auto relocate_one(T1 &tier1,
                  T2FlatFile &tier2,
                  T2Ownership<ConfigT> &own,
                  LockStripe &&lock_stripe,
                  MaybeReorg &&maybe_reorg,
                  uint64_t old_off,
                  const vmemkv::T2Memory *mem,
                  CompactionState *probe = nullptr) -> uint64_t {
  const bool measure = probe != nullptr && segment_probe_enabled();
  const auto validate_start = measure ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  const std::span<const std::byte> key = T2FlatFile::at(old_off, mem).key;

  std::lock_guard<std::mutex> key_lock(lock_stripe(key));
  const auto cur = tier1.get_with_hash(key);
  if (cur.payload_bits == vmemkv::STORE_NOT_FOUND || t1_detail::is_inline(cur.raw_hash) ||
      (cur.payload_bits & t2_detail::kPayloadOffsetMask) != old_off) {
    return 0;
  }
  const auto validate_end = measure ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  const std::span<const std::byte> value_span = T2FlatFile::at(old_off, mem).value;
  T2FlatFile::T2MemoryHandle handle = tier2.acquire_write_handle();
  const uint64_t offset = vmemkv::T2FlatFile::append_default(handle, key, value_span);
  const auto movedata_end = measure ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  const uint64_t block_count = t2_detail::checked_block_count(key.size(), value_span.size());
  const uint64_t encoded = t2_detail::encode_offset_payload(offset, block_count);
  for (;;) {
    if (tier1.put(key, encoded, false, 0, /*mark_t2_dirty=*/false) == T1::PutResult::Applied) {
      break;
    }
    maybe_reorg();
  }
  own.note_replace(cur.payload_bits,
                   cur.raw_hash,
                   /*new_is_offset=*/true,
                   encoded,
                   /*mark_old_segments_dirty=*/false);
  if (measure) {
    const auto publish_end = std::chrono::steady_clock::now();
    probe->probe_validate_ns.fetch_add(
        static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(validate_end - validate_start).count()),
        std::memory_order_relaxed);
    probe->probe_movedata_ns.fetch_add(
        static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(movedata_end - validate_end).count()),
        std::memory_order_relaxed);
    probe->probe_publish_ns.fetch_add(
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(publish_end - movedata_end).count()),
        std::memory_order_relaxed);
  }
  return vmemkv::align_up(sizeof(ValueRecordHeader) + key.size() + value_span.size());
}

template <typename ConfigT, typename T1, typename LockStripe, typename MaybeReorg>
auto relocate_candidates(CompactionState &state,
                         T1 &tier1,
                         T2FlatFile &tier2,
                         T2Ownership<ConfigT> &own,
                         LockStripe &&lock_stripe,
                         MaybeReorg &&maybe_reorg,
                         const std::vector<SegmentRelocationCandidate> &candidates,
                         const vmemkv::T2Memory *mem) -> uint64_t {
  uint64_t moved_bytes = 0;
  const bool measure = segment_probe_enabled();
  for (const SegmentRelocationCandidate &candidate : candidates) {
    const auto relocation_start = measure ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const uint64_t moved =
        relocate_one<ConfigT>(tier1, tier2, own, lock_stripe, maybe_reorg, candidate.offset, mem, &state);
    moved_bytes += moved;
    if (measure) {
      (moved == 0 ? state.probe_stale : state.probe_relocated).fetch_add(1, std::memory_order_relaxed);
      state.probe_moved_bytes.fetch_add(moved, std::memory_order_relaxed);
      state.probe_relocate_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                                  std::chrono::steady_clock::now() - relocation_start)
                                                                  .count()),
                                        std::memory_order_relaxed);
    }
  }
  return moved_bytes;
}

// Evacuates every queued dirty segment sharing one T1 scan. Replaced keys are
// resolved per segment through T1; relocate_one() revalidates each offset under
// the key lock, so candidates that moved for an earlier segment are skipped as
// stale for a later one.
template <typename ConfigT, typename T1, typename LockStripe, typename MaybeReorg>
auto evacuate_dirty_segments(CompactionState &state,
                             T1 &tier1,
                             T2FlatFile &tier2,
                             T2Ownership<ConfigT> &own,
                             LockStripe &&lock_stripe,
                             MaybeReorg &&maybe_reorg,
                             std::vector<typename T2Ownership<ConfigT>::DirtySegment> &works,
                             const vmemkv::T2Memory *mem) -> uint64_t {
  uint64_t moved_bytes = 0;
  const bool measure = segment_probe_enabled();
  std::unordered_set<uint64_t> dirty_set;
  dirty_set.reserve(works.size() * 2);
  for (const auto &work : works) {
    dirty_set.insert(work.segment);
  }
  SegmentSourceIndex source_index;
  try {
    const auto scan_start = measure ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    source_index = build_source_index<ConfigT>(tier1, dirty_set, measure);
    if (measure) {
      const auto scan_end = std::chrono::steady_clock::now();
      const uint64_t scan_ns =
          static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(scan_end - scan_start).count());
      state.probe_scan_ns.fetch_add(scan_ns, std::memory_order_relaxed);
      state.probe_collect_ns.fetch_add(scan_ns, std::memory_order_relaxed);
      state.probe_scan_entries.fetch_add(source_index.entries, std::memory_order_relaxed);
      state.probe_scan_matched.fetch_add(source_index.matched, std::memory_order_relaxed);
      state.probe_copy_bytes.fetch_add(source_index.copy_bytes, std::memory_order_relaxed);
      state.probe_source_candidates.fetch_add(source_index.matched, std::memory_order_relaxed);
    }
  } catch (...) {
    for (auto &work : works) {
      own.requeue_dirty_segment(std::move(work));
    }
    throw;
  }
  for (size_t work_index = 0; work_index < works.size(); ++work_index) {
    auto &work = works[work_index];
    try {
      auto candidates = prepare_indexed_candidates<ConfigT>(state, tier1, source_index, work, mem, measure);
      moved_bytes += relocate_candidates<ConfigT>(state, tier1, tier2, own, lock_stripe, maybe_reorg, candidates, mem);
    } catch (...) {
      for (size_t rest = work_index; rest < works.size(); ++rest) {
        own.requeue_dirty_segment(std::move(works[rest]));
      }
      throw;
    }
    const uint64_t frozen_segs = mem->base_boundary.load(std::memory_order_acquire) / ConfigT::kSegmentBytes;
    if (work.segment < frozen_segs && own.seg_live(work.segment) != 0) {
      own.queue_dirty_segment(work.segment);
    }
  }
  return moved_bytes;
}

struct CompactionProgress {
  uint64_t moved_bytes = 0;
  bool did_relocate = false;
};

template <typename ConfigT, typename T1, typename LockStripe, typename MaybeReorg, typename EntrySnapshot>
void repack_dirty_shard(std::span<const EntrySnapshot> entries,
                        T1 &tier1,
                        T2FlatFile &tier2,
                        T2Ownership<ConfigT> &own,
                        LockStripe &&lock_stripe,
                        MaybeReorg &&maybe_reorg,
                        const vmemkv::T2Memory *mem,
                        CompactionProgress &progress) {
  const std::vector<uint64_t> payloads = collect_shard_payloads(entries, mem);
  const size_t count = payloads.size();
  if (count < 2) {
    return;
  }
  auto local = [&](size_t lhs, size_t rhs) -> bool {
    return t2_detail::are_t2_payloads_page_local(payloads[lhs], payloads[rhs]);
  };
  auto raw_keep = [&](size_t idx) -> bool {
    if (idx == 0) {
      return local(0, 1);
    }
    if (idx + 1 == count) {
      return local(count - 2, count - 1);
    }
    return local(idx - 1, idx) && local(idx, idx + 1);
  };
  for (size_t i = 0; i < count; ++i) {
    const bool keep = raw_keep(i) && ((i > 0 && raw_keep(i - 1)) || (i + 1 < count && raw_keep(i + 1)));
    if (keep) {
      continue;
    }
    const uint64_t off = payloads[i] & t2_detail::kPayloadOffsetMask;
    const uint64_t moved = relocate_one<ConfigT>(tier1, tier2, own, lock_stripe, maybe_reorg, off, mem);
    if (moved != 0) {
      progress.moved_bytes += moved;
      progress.did_relocate = true;
    }
  }
}

// Runs one compaction cycle: punches evacuated segments, evacuates dirty T2 segments
// sharing one T1 scan, or repairs dirty-shard offset order when no segment evacuation
// is pending. Returns whether the cycle processed dirty state or punched a segment.
template <typename ConfigT, typename T1, typename LockStripe, typename MaybeReorg>
auto compact_cycle(CompactionState &state,
                   T1 &tier1,
                   T2FlatFile &tier2,
                   T2Ownership<ConfigT> &own,
                   const std::filesystem::path &t2_path,
                   LockStripe &&lock_stripe,
                   MaybeReorg &&maybe_reorg) -> bool {
  auto running_guard = SingleFlightGuard::try_acquire(state.running);
  if (!running_guard.holds) {
    return false;
  }

  uint64_t punched = punch_evacuated_segments<ConfigT>(state, own, tier2, t2_path);

  const vmemkv::T2Memory *mem = tier2.get_memory();
  CompactionProgress progress;

  if (auto works = own.take_all_dirty_segments(); !works.empty()) {
    progress.moved_bytes =
        evacuate_dirty_segments<ConfigT>(state, tier1, tier2, own, lock_stripe, maybe_reorg, works, mem);
    progress.did_relocate = progress.moved_bytes != 0;
    if (progress.did_relocate || punched > 0) {
      state.cycle_count.fetch_add(1, std::memory_order_relaxed);
    }
    return true;
  }

  using EntrySnapshot = typename T1::EntrySnapshot;
  const bool had_dirty_shard = tier1.has_dirty_shard();
  if (had_dirty_shard) {
    tier1.for_each_dirty_shard([&](std::span<const EntrySnapshot> entries) -> bool {
      repack_dirty_shard<ConfigT>(entries, tier1, tier2, own, lock_stripe, maybe_reorg, mem, progress);
      return progress.moved_bytes < ConfigT::kCompactionMaxMoveBytesPerCycle;
    });
  }

  if (progress.moved_bytes < ConfigT::kCompactionMaxMoveBytesPerCycle) {
    const uint64_t base = mem->base_boundary.load(std::memory_order_acquire);
    uint64_t planned_bytes = progress.moved_bytes;
    while (planned_bytes < ConfigT::kCompactionMaxMoveBytesPerCycle) {
      const uint64_t cand_seg = own.find_garbage_segment(base);
      if (cand_seg == T2Ownership<ConfigT>::kInvalidSegment) {
        break;
      }
      own.queue_record_segment(cand_seg);
      planned_bytes += own.seg_live(cand_seg);
    }
    auto batch_work = own.take_all_dirty_segments();
    if (!batch_work.empty()) {
      const uint64_t seg_moved =
          evacuate_dirty_segments<ConfigT>(state, tier1, tier2, own, lock_stripe, maybe_reorg, batch_work, mem);
      progress.moved_bytes += seg_moved;
      progress.did_relocate = progress.did_relocate || (seg_moved != 0);
    }
  }

  if (progress.did_relocate || punched > 0) {
    state.cycle_count.fetch_add(1, std::memory_order_relaxed);
  }
  return had_dirty_shard || progress.did_relocate || punched > 0;
}

}  // namespace compaction_detail
}  // namespace vmemkv
