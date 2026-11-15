// t2_ownership.hpp - T2-side ownership state: segment live-byte accounting, the checkpoint
// capture watermark, and the in-place-update barrier. Together they decide whether a byte
// range is safe to durabilize, punch, or rewrite in place.
#pragma once

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>
#include <vmemkv/api/utils.hpp>
#include <vmemkv/core/env.hpp>
#include <vmemkv/impl/hooks.hpp>
#include <vmemkv/t1_index/t1_index.hpp>

namespace vmemkv {

namespace t2_detail {
// T1 payload codec shared by read/write paths: `offset |
// (block_count << kPayloadSizeShift)`, 16-byte-granular size hint.
inline constexpr uint64_t kPayloadSizeShift = 48;
inline constexpr uint64_t kPayloadOffsetMask = (1ULL << kPayloadSizeShift) - 1;
inline constexpr uint64_t kRecordBlockAlignment = 16;
inline constexpr uint64_t kMaxBlockCount = 1ULL << 16;

// Block count for a fresh record, throwing when it doesn't fit the codec's 16 bits (checked
// pre-append: wrapping would corrupt try_read_base_record()'s embedded size hint).
inline auto checked_block_count(uint64_t key_bytes, uint64_t value_bytes) -> uint64_t {
  const uint64_t aligned = align_up(sizeof(ValueRecordHeader) + key_bytes + value_bytes);
  const uint64_t block_count = aligned / kRecordBlockAlignment;
  if (block_count >= kMaxBlockCount) {
    throw std::runtime_error("Record size exceeds 1.04MB limit");
  }
  return block_count;
}

// Encodes a fresh-append payload from a T2 offset and a checked block count.
inline auto encode_offset_payload(uint64_t offset, uint64_t block_count) noexcept -> uint64_t {
  return offset | (block_count << kPayloadSizeShift);
}

// Margin making the embedded block-count hint cover the true aligned record length on the
// base-region read path.
inline constexpr uint64_t kBaseSizeHintMargin = 16;

// Embedded block-count size hint for a T1 payload, plus a caller-chosen margin. The hint is
// 16-byte-granular while records are only 8-byte aligned, so with zero margin it sits at or
// under the true length; with margin it covers the true aligned length for sizing
// base-region reads, whether resident or pread.
inline auto payload_hint_bytes(uint64_t payload_bits, uint64_t margin) noexcept -> uint64_t {
  return ((payload_bits >> kPayloadSizeShift) * kRecordBlockAlignment) + margin;
}

// OS page size for mincore()/msync() range alignment and the small/large read threshold.
inline auto os_page_size() noexcept -> uint64_t {
  static const auto kCached = static_cast<uint64_t>(::sysconf(_SC_PAGESIZE));
  return kCached;
}

inline constexpr uint64_t kOsPageShift = 12;

// True when two T1-adjacent T2 payloads occupy the same 4KiB page or immediately consecutive
// forward pages, so relocating them cannot reduce page-fault transitions.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
inline auto are_t2_payloads_page_local(uint64_t prev_payload, uint64_t curr_payload) noexcept -> bool {
  const uint64_t prev_off = prev_payload & kPayloadOffsetMask;
  const uint64_t prev_bytes =
      std::max<uint64_t>(kRecordBlockAlignment, (prev_payload >> kPayloadSizeShift) * kRecordBlockAlignment);
  const uint64_t curr_off = curr_payload & kPayloadOffsetMask;
  const uint64_t curr_bytes =
      std::max<uint64_t>(kRecordBlockAlignment, (curr_payload >> kPayloadSizeShift) * kRecordBlockAlignment);
  const uint64_t prev_start_page = prev_off >> kOsPageShift;
  const uint64_t prev_end_page = (prev_off + prev_bytes - 1) >> kOsPageShift;
  const uint64_t curr_start_page = curr_off >> kOsPageShift;
  const uint64_t curr_end_page = (curr_off + curr_bytes - 1) >> kOsPageShift;
  if (curr_start_page <= prev_end_page && prev_start_page <= curr_end_page) {
    return true;
  }
  return curr_off > prev_off && curr_start_page == prev_end_page + 1;
}
}  // namespace t2_detail

// Per-segment live bytes (size-hint sums) plus the checkpoint capture watermark and
// in-place-update barrier. Atomics: same-segment entries on different stripes update
// concurrently. A boundary-straddling record credits both segments it touches, so
// seg_live(seg) == 0 guarantees no live record starts in or overhangs into `seg`.
template <typename ConfigT>
class T2Ownership {
 public:
  static constexpr uint64_t kInvalidSegment = std::numeric_limits<uint64_t>::max();

  // A segment with no live records, tagged with the completed-checkpoint count at
  // the moment it became empty. Punching is safe only after a checkpoint completed
  // that covers the evacuation (durable tail plus a T1 dump holding the
  // post-relocation pointers); the tag orders candidates against that condition.
  struct PunchCandidate {
    uint64_t segment = 0;
    uint64_t evacuated_epoch = 0;
  };

  explicit T2Ownership(size_t segment_count = 0)
      : segment_count_(segment_count),
        seg_live_(segment_count),
        punch_pending_(segment_count),
        punch_nodes_(segment_count),
        garbage_pending_(segment_count),
        garbage_nodes_(segment_count),
        dirty_flag_(segment_count) {
    for (auto &counter : seg_live_) {
      counter.store(0, std::memory_order_relaxed);
    }
    for (size_t i = 0; i < segment_count; ++i) {
      punch_pending_[i].store(false, std::memory_order_relaxed);
      punch_nodes_[i].evacuated_epoch = 0;
      punch_nodes_[i].next.store(kInvalidSegment, std::memory_order_relaxed);
      garbage_pending_[i].store(false, std::memory_order_relaxed);
      garbage_nodes_[i].next.store(kInvalidSegment, std::memory_order_relaxed);
      dirty_flag_[i].store(false, std::memory_order_relaxed);
    }
  }

  static auto seg_index(uint64_t masked_offset) noexcept -> uint64_t { return masked_offset / ConfigT::kSegmentBytes; }

  // Adjusts accounting for one T1 payload replacement. Callers hold the key's stripe lock
  // (or run single-threaded: WAL replay / checkpoint load), so the old classification can't
  // race the put. Tombstones and inlines address no T2 segment and contribute nothing.
  // Compaction relocations pass allow_mark_dirty = false so moving a shard that straddles a
  // segment boundary does not mark a neighboring healthy segment dirty.
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  void note_replace(
      uint64_t old_bits, uint64_t old_hash, bool new_is_offset, uint64_t new_bits, bool allow_mark_dirty = true) {
    if (old_bits != vmemkv::STORE_NOT_FOUND && !t1_detail::is_inline(old_hash)) {
      adjust_live(old_bits, /*add=*/false, allow_mark_dirty);
    }
    if (new_is_offset) {
      adjust_live(new_bits, /*add=*/true, allow_mark_dirty);
    }
  }

  auto seg_live(uint64_t seg) const noexcept -> uint64_t {
    assert(seg < seg_live_.size());
    return seg_live_[seg].load(std::memory_order_relaxed);
  }

  void mark_segment_dirty(uint64_t seg) const noexcept {
    if (seg >= segment_count_) {
      return;
    }
    dirty_flag_[seg].store(true, std::memory_order_relaxed);
    if (garbage_pending_[seg].load(std::memory_order_relaxed)) {
      return;
    }
    if (garbage_pending_[seg].exchange(true, std::memory_order_acq_rel)) {
      return;
    }
    uint64_t head = garbage_head_.load(std::memory_order_relaxed);
    do {
      garbage_nodes_[seg].next.store(head, std::memory_order_relaxed);
    } while (!garbage_head_.compare_exchange_weak(head, seg, std::memory_order_release, std::memory_order_relaxed));
  }

  // Exclusive upper bound on segment indices eligible for garbage evacuation: a segment must
  // sit below base_boundary and precede the active append segment containing bytes_used.
  [[nodiscard]] auto evac_limit(
      // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
      uint64_t base_boundary,
      uint64_t bytes_used = std::numeric_limits<uint64_t>::max()) const noexcept -> uint64_t {
    if (base_boundary == 0) {
      return 0;
    }
    const uint64_t frozen_segs = (base_boundary + ConfigT::kSegmentBytes - 1) / ConfigT::kSegmentBytes;
    const uint64_t current_seg = bytes_used / ConfigT::kSegmentBytes;
    return std::min<uint64_t>({frozen_segs, current_seg, static_cast<uint64_t>(segment_count_)});
  }

  // True when `seg` is a frozen segment marked dirty with remaining live bytes.
  [[nodiscard]] auto is_garbage_segment(
      // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
      uint64_t seg,
      uint64_t base_boundary,
      uint64_t bytes_used = std::numeric_limits<uint64_t>::max()) const noexcept -> bool {
    if (seg >= evac_limit(base_boundary, bytes_used)) {
      return false;
    }
    return dirty_flag_[seg].load(std::memory_order_relaxed) && !punch_pending_[seg].load(std::memory_order_relaxed) &&
           seg_live_[seg].load(std::memory_order_relaxed) > 0;
  }

  // True when `payload_bits` touches any frozen garbage segment.
  [[nodiscard]] auto payload_overlaps_garbage_segment(
      // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
      uint64_t payload_bits,
      uint64_t base_boundary,
      uint64_t bytes_used = std::numeric_limits<uint64_t>::max()) const noexcept -> bool {
    const uint64_t off = payload_bits & t2_detail::kPayloadOffsetMask;
    const uint64_t hint = hint_bytes(payload_bits);
    const uint64_t first = seg_index(off);
    const uint64_t last = seg_index(off + hint - 1);
    return is_garbage_segment(first, base_boundary, bytes_used) ||
           (last != first && is_garbage_segment(last, base_boundary, bytes_used));
  }

  [[nodiscard]] auto has_garbage_segment(
      // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
      uint64_t base_boundary,
      uint64_t bytes_used = std::numeric_limits<uint64_t>::max()) const noexcept -> bool {
    for (uint64_t curr = garbage_head_.load(std::memory_order_acquire); curr != kInvalidSegment;
         curr = garbage_nodes_[curr].next.load(std::memory_order_relaxed)) {
      if (is_garbage_segment(curr, base_boundary, bytes_used)) {
        return true;
      }
    }
    return false;
  }

  [[nodiscard]] auto garbage_segment_count(
      // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
      uint64_t base_boundary,
      uint64_t bytes_used = std::numeric_limits<uint64_t>::max()) const noexcept -> size_t {
    size_t count = 0;
    for (uint64_t curr = garbage_head_.load(std::memory_order_acquire); curr != kInvalidSegment;
         curr = garbage_nodes_[curr].next.load(std::memory_order_relaxed)) {
      if (is_garbage_segment(curr, base_boundary, bytes_used)) {
        ++count;
      }
    }
    return count;
  }

  void clear_garbage_segments(
      // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
      uint64_t base_boundary,
      uint64_t bytes_used = std::numeric_limits<uint64_t>::max()) const noexcept {
    const uint64_t limit = evac_limit(base_boundary, bytes_used);
    uint64_t curr = garbage_head_.exchange(kInvalidSegment, std::memory_order_acquire);
    while (curr != kInvalidSegment) {
      const uint64_t next = garbage_nodes_[curr].next.load(std::memory_order_relaxed);
      if (curr < limit) {
        dirty_flag_[curr].store(false, std::memory_order_relaxed);
        garbage_pending_[curr].store(false, std::memory_order_release);
      } else if (dirty_flag_[curr].load(std::memory_order_relaxed) &&
                 !punch_pending_[curr].load(std::memory_order_relaxed) &&
                 seg_live_[curr].load(std::memory_order_relaxed) > 0) {
        uint64_t head = garbage_head_.load(std::memory_order_relaxed);
        do {
          garbage_nodes_[curr].next.store(head, std::memory_order_relaxed);
        } while (
            !garbage_head_.compare_exchange_weak(head, curr, std::memory_order_release, std::memory_order_relaxed));
      } else {
        garbage_pending_[curr].store(false, std::memory_order_release);
      }
      curr = next;
    }
  }

  // Queues every fully-frozen segment with no live record during checkpoint adoption.
  void queue_empty_segments(uint64_t frozen_boundary) const {
    const uint64_t frozen_segments = frozen_boundary / ConfigT::kSegmentBytes;
    for (uint64_t seg = 0; seg < std::min<uint64_t>(frozen_segments, seg_live_.size()); ++seg) {
      if (seg_live_[seg].load(std::memory_order_relaxed) == 0) {
        queue_punch_candidate(seg);
      }
    }
  }

  [[nodiscard]] auto has_punch_candidates() const noexcept -> bool {
    return punch_head_.load(std::memory_order_relaxed) != kInvalidSegment;
  }

  auto take_punch_candidates() const -> std::vector<PunchCandidate> {
    uint64_t curr = punch_head_.exchange(kInvalidSegment, std::memory_order_acquire);
    std::vector<PunchCandidate> out;
    while (curr != kInvalidSegment) {
      const uint64_t epoch = punch_nodes_[curr].evacuated_epoch;
      out.push_back(PunchCandidate{curr, epoch});
      punch_pending_[curr].store(false, std::memory_order_release);
      const uint64_t next = punch_nodes_[curr].next.load(std::memory_order_relaxed);
      punch_nodes_[curr].next.store(kInvalidSegment, std::memory_order_relaxed);
      curr = next;
    }
    punch_candidate_count_.fetch_sub(out.size(), std::memory_order_relaxed);
    return out;
  }

  // Requeues with epochs preserved: re-tagging with the current count would defer
  // eligibility every time a checkpoint completes between cycles.
  void requeue_punch_candidates(std::span<const PunchCandidate> candidates) const {
    for (const PunchCandidate &candidate : candidates) {
      queue_punch_candidate_with_epoch(candidate.segment, candidate.evacuated_epoch);
    }
  }

  // Checkpoint's write-side barrier: offsets < watermark redirect out-of-place instead of
  // racing the durabilizing read. Set to `target` once per cycle (msync covers the range in
  // one shot, so the claim must too); reverted on abort. Conservative: at least as far as
  // genuinely read this cycle, never exactly so.
  //
  // seq_cst: paired with the barrier's seq_cst registration/scan (two separate atomics touched
  // by both sides -- see ThreadReferenceTracker::acquire()).
  void claim_watermark(uint64_t target) noexcept { watermark_.store(target, std::memory_order_seq_cst); }

  void revert_watermark(uint64_t old_boundary) noexcept { watermark_.store(old_boundary, std::memory_order_seq_cst); }

  auto load_watermark() const noexcept -> uint64_t { return watermark_.load(std::memory_order_seq_cst); }

  [[nodiscard]] auto enter_in_place(uint64_t offset) const noexcept -> InPlaceUpdateBarrier::Guard {
    return barrier_.enter(offset);
  }

  // Drains in-place writes that passed allow_in_place against the pre-claim watermark and are
  // still physically writing -- else msync() or the boundary publish could catch them mid-write.
  void drain_in_place(uint64_t boundary) const noexcept { barrier_.wait_until_retired(boundary); }

  // Completed-checkpoint count mirror, set by the store after each Checkpoint-mode
  // cycle. Punch candidates queue with the count observed at evacuation time.
  void note_checkpoint_completed(uint64_t checkpoint_count) const noexcept {
    last_checkpoint_count_.store(checkpoint_count, std::memory_order_relaxed);
  }

  auto last_checkpoint_count() const noexcept -> uint64_t {
    return last_checkpoint_count_.load(std::memory_order_relaxed);
  }

  // Segments waiting for a coverage-granting checkpoint. Drives the checkpoint
  // trigger alongside the WAL-size trigger: without relocation WAL records, the
  // WAL alone no longer paces punch drainage.
  auto punch_backlog_segments() const noexcept -> uint64_t {
    return punch_candidate_count_.load(std::memory_order_relaxed);
  }

  static auto hint_bytes(uint64_t payload_bits) noexcept -> uint64_t {
    return t2_detail::payload_hint_bytes(payload_bits, t2_detail::kBaseSizeHintMargin);
  }

  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  void adjust_live(uint64_t payload_bits, bool add, bool allow_mark_dirty = true) {
    const uint64_t off = payload_bits & t2_detail::kPayloadOffsetMask;
    const uint64_t hint = hint_bytes(payload_bits);
    const uint64_t seg0 = seg_index(off);
    const uint64_t seg1 = seg_index(off + hint - 1);
    if (add) {
      if (seg0 < seg_live_.size()) {
        seg_live_[seg0].fetch_add(hint, std::memory_order_relaxed);
      }
      if (seg1 != seg0 && seg1 < seg_live_.size()) {
        seg_live_[seg1].fetch_add(hint, std::memory_order_relaxed);
      }
    } else {
      subtract_segment(seg0, hint, allow_mark_dirty);
      if (seg1 != seg0) {
        subtract_segment(seg1, hint, allow_mark_dirty);
      }
    }
  }

  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  void subtract_segment(uint64_t seg, uint64_t hint, bool allow_mark_dirty = true) {
    if (seg >= seg_live_.size()) {
      return;
    }
    const uint64_t previous = seg_live_[seg].fetch_sub(hint, std::memory_order_relaxed);
    if (previous == hint) {
      dirty_flag_[seg].store(false, std::memory_order_relaxed);
      queue_punch_candidate(seg);
    } else if (allow_mark_dirty && (previous - hint) * 2 < ConfigT::kSegmentBytes) {
      mark_segment_dirty(seg);
    }
  }

  void queue_punch_candidate(uint64_t seg) const {
    queue_punch_candidate_with_epoch(seg, last_checkpoint_count_.load(std::memory_order_relaxed));
  }

 private:
  struct GarbageNode {
    std::atomic<uint64_t> next{kInvalidSegment};
  };

  struct PunchNode {
    uint64_t evacuated_epoch{0};
    std::atomic<uint64_t> next{kInvalidSegment};
  };

  void push_punch(uint64_t seg, uint64_t epoch) const noexcept {
    punch_nodes_[seg].evacuated_epoch = epoch;
    uint64_t head = punch_head_.load(std::memory_order_relaxed);
    do {
      punch_nodes_[seg].next.store(head, std::memory_order_relaxed);
    } while (!punch_head_.compare_exchange_weak(head, seg, std::memory_order_release, std::memory_order_relaxed));
    punch_candidate_count_.fetch_add(1, std::memory_order_relaxed);
  }

  void queue_punch_candidate_with_epoch(uint64_t seg, uint64_t epoch) const noexcept {
    if (seg >= segment_count_) {
      return;
    }
    if (punch_pending_[seg].load(std::memory_order_relaxed)) {
      return;
    }
    if (punch_pending_[seg].exchange(true, std::memory_order_acq_rel)) {
      return;
    }
    push_punch(seg, epoch);
  }

  size_t segment_count_{0};
  std::vector<std::atomic<uint64_t>> seg_live_;
  std::atomic<uint64_t> watermark_{0};
  InPlaceUpdateBarrier barrier_;

  mutable std::vector<std::atomic<bool>> punch_pending_;
  mutable std::vector<PunchNode> punch_nodes_;
  mutable std::atomic<uint64_t> punch_head_{kInvalidSegment};
  mutable std::atomic<uint64_t> punch_candidate_count_{0};

  mutable std::vector<std::atomic<bool>> garbage_pending_;
  mutable std::vector<GarbageNode> garbage_nodes_;
  mutable std::atomic<uint64_t> garbage_head_{kInvalidSegment};
  mutable std::vector<std::atomic<bool>> dirty_flag_;
  mutable std::atomic<uint64_t> last_checkpoint_count_{0};
};

}  // namespace vmemkv
