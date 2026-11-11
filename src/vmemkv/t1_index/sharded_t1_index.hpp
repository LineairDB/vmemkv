// sharded_t1_index.hpp -- Range-sharded routing over K T1Index instances.
// See docs/specification/t1_sharding_design.md for the full design. T1Index is the per-shard leaf.
// Maintenance helpers live in sharded_maintenance.hpp.
#pragma once

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stop_token>
#include <thread>
#include <utility>
#include <vector>
#include <vmemkv/config.hpp>
#include <vmemkv/core/background_poll.hpp>
#include <vmemkv/core/reference_tracker.hpp>
#include <vmemkv/core/single_flight.hpp>
#include <vmemkv/core/spin_backoff.hpp>
#include <vmemkv/scan_control.hpp>
#include <vmemkv/t1_index/sharded_maintenance.hpp>
#include <vmemkv/t1_index/t1_index.hpp>

namespace vmemkv {

namespace t1_debug {
inline std::atomic<size_t> g_checkpoint_current_shard{0};
inline std::atomic<size_t> g_checkpoint_total_shards{0};
}  // namespace t1_debug

template <typename Config = vmemkv::Config<>>
class ShardedT1Index {
 public:
  using Key = t1_detail::StoreKey;
  using Payload = uint64_t;
  using Shard = T1Index<Config>;
  using PutResult = typename Shard::PutResult;
  using LookupResult = typename Shard::LookupResult;
  using EntrySnapshot = typename Shard::EntrySnapshot;

  // `target_shard_size`: the one knob callers tune -- a shard's background maintenance splits
  // it past Config::kShardSplitThresholdPercent% of this target
  // (see docs/specification/t1_sharding_design.md).
  // `worker_threads`: background maintenance pool size; defaults to a quarter of hardware
  // concurrency (cores stay reserved for foreground traffic). 0 defers startup (see
  // start_workers()) for callers needing single-threaded setup first (e.g. checkpoint load).
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  explicit ShardedT1Index(size_t append_cap = Config::kAppendCapacityEntries,
                          size_t target_shard_size = Config::kShardTargetSizeEntries,
                          size_t worker_threads = default_worker_count())
      : append_cap_(append_cap), target_shard_size_(target_shard_size) {
    auto *slot = new ShardSlot(std::make_unique<Shard>(append_cap));
    auto *dir = new Directory();
    dir->shards.push_back(slot);
    directory_.store(dir, std::memory_order_release);
    start_workers(worker_threads);
  }

  // Spawns `count` more workers. Single-threaded-caller only (construction with
  // worker_threads=0, then this, exactly once) -- workers_ itself isn't synchronized.
  void start_workers(size_t count = default_worker_count()) {
    workers_.reserve(workers_.size() + count);
    for (size_t i = 0; i < count; ++i) {
      workers_.emplace_back([this](std::stop_token stop_token) { worker_loop(stop_token); });
    }
  }

  // Stops/joins workers before tearing down shards (a worker could be mid-reorganize()).
  // request_stop() only flags; this blocks until in-flight maintenance finishes naturally.
  ~ShardedT1Index() {
    for (auto &worker : workers_) {
      worker.request_stop();
    }
    workers_.clear();  // Each jthread's destructor joins.

    Directory *dir = directory_.load(std::memory_order_relaxed);
    for (ShardSlot *slot : dir->shards) {
      delete slot;
    }
    delete dir;
  }

  ShardedT1Index(const ShardedT1Index &) = delete;
  auto operator=(const ShardedT1Index &) -> ShardedT1Index & = delete;

  auto get(std::span<const std::byte> key) const -> Payload { return get_with_hash(key).payload_bits; }

  auto get_with_hash(std::span<const std::byte> key) const -> LookupResult {
    const Key prefix = t1_detail::prefix_from_bytes(key);
    return with_routing_guard([&]() -> LookupResult {
      ShardSlot *slot = resolve(prefix);
      return slot->index->get_with_hash(key);
    });
  }

  // Requests maintenance past the soft threshold; on a full append region retries with
  // backoff (maintenance drains room, or a split routes to a fresh shard). Backpressure stays
  // scoped to this shard's key range. After applying, re-checks superseded: a split may have
  // retired the slot mid-write, in which case re-resolve and re-apply (idempotent upsert) so
  // every returned put is already in a live shard -- and never overwrites a newer write.
  auto put(std::span<const std::byte> key,
           Payload value,
           bool is_inline = false,
           uint8_t inline_size = 0,
           bool mark_t2_dirty = true) -> PutResult {
    const Key prefix = t1_detail::prefix_from_bytes(key);
    SpinBackoff backoff;
    for (;;) {
      const std::optional<PutResult> attempted = with_routing_guard([&]() -> std::optional<PutResult> {
        ShardSlot *slot = resolve_for_write(prefix);
        if (slot == nullptr) {
          return std::nullopt;
        }
        request_maintenance_if_needed(slot);
        const PutResult result = slot->index->put(key, value, is_inline, inline_size);
        if (result == PutResult::AppendRegionFull || slot->superseded.load(std::memory_order_acquire) != nullptr) {
          return std::nullopt;
        }
        if (mark_t2_dirty && value != STORE_NOT_FOUND && !is_inline) {
          slot->t2_dirty.store(true, std::memory_order_relaxed);
          const uint64_t prefix_hi = t1_detail::prefix_high_u64(prefix);
          const uint64_t prefix_lo = t1_detail::prefix_low_u64(prefix);
          const uint64_t prev_hi = slot->last_t2_prefix_hi.load(std::memory_order_relaxed);
          const uint64_t prev_lo = slot->last_t2_prefix_lo.load(std::memory_order_relaxed);
          if (slot->has_last_t2_prefix.load(std::memory_order_relaxed) &&
              (prefix_hi < prev_hi || (prefix_hi == prev_hi && prefix_lo < prev_lo))) {
            slot->t2_inversion.store(true, std::memory_order_relaxed);
          }
          slot->last_t2_prefix_hi.store(prefix_hi, std::memory_order_relaxed);
          slot->last_t2_prefix_lo.store(prefix_lo, std::memory_order_relaxed);
          slot->has_last_t2_prefix.store(true, std::memory_order_relaxed);
        }
        if (value == STORE_NOT_FOUND) {
          slot->tombstones_since_maintenance.fetch_add(1, std::memory_order_relaxed);
          // Re-check post-increment: the final delete of a quiet workload must still trip the
          // trigger without another write after it.
          request_maintenance_if_needed(slot);
        }
        return result;
      });
      if (attempted.has_value()) {
        return *attempted;
      }
      backoff.wait();
    }
  }

  // Set while a scan is in flight (cleared after). While set, the soft threshold drops to
  // keep append regions L2-sized, so the scan's linear pass stays cache-resident.
  void set_scan_active(bool active) const noexcept { scan_active_.store(active, std::memory_order_relaxed); }

  // Append entries allowed before maintenance: half capacity normally, L2-capped while a scan
  // is active. Pure function, unit-testable at production capacities.
  static constexpr size_t kPercentScale = sharded_detail::kPercentScale;
  static auto maintenance_soft_threshold(size_t append_cap, bool scan_active) -> size_t {
    return sharded_detail::maintenance_soft_threshold(append_cap, scan_active, Shard::append_slot_bytes());
  }

  // Range scan over [lo_bytes, hi_bytes). Structural completeness + per-key freshness only;
  // cross-key real-time ordering is deliberately not promised
  // (see docs/specification/t1_sharding_design.md).
  template <typename Callback>
  auto scan(std::span<const std::byte> lo_bytes,
            std::span<const std::byte> hi_bytes,
            Callback callback) const -> size_t {
    const Key lower = t1_detail::prefix_from_bytes(lo_bytes);
    const Key upper = t1_detail::prefix_from_bytes(hi_bytes);
    bool stopped = false;
    auto scan_callback = [&callback, &stopped](auto &&...args) -> ScanControl {
      const ScanControl control = scan_detail::invoke_callback(callback, std::forward<decltype(args)>(args)...);
      stopped = control == ScanControl::Stop;
      return control;
    };
    return with_routing_guard([&]() -> size_t {
      Directory *dir = directory_.load(std::memory_order_acquire);
      if (dir->shards.size() == 1) {
        if (std::optional<size_t> fast = scan_single_shard(dir, lo_bytes, hi_bytes, scan_callback)) {
          return *fast;
        }
      }
      return scan_multi_shard(dir, lower, upper, lo_bytes, hi_bytes, scan_callback, stopped);
    });
  }

  // Must run under with_routing_guard(): an unguarded read of directory_ could be freed by a
  // concurrent split's publish+drain before use.
  [[nodiscard]] auto shard_count() const -> size_t {
    return with_routing_guard([&]() -> size_t { return directory_.load(std::memory_order_acquire)->shards.size(); });
  }

  // Completed splits over this store's lifetime (maintenance no-ops don't count).
  [[nodiscard]] auto total_splits() const -> uint64_t { return total_splits_.load(std::memory_order_relaxed); }

  // Last completed split's writer-visible pause: Closing-unresolvable start to Split-publish
  // end. Not the whole continue_split() duration (drain/redistribution run unblocked after).
  // Last-only, not aggregated (diagnostic, matches last_checkpoint_*).
  [[nodiscard]] auto last_split_pause_us() const -> uint64_t {
    return last_split_pause_us_.load(std::memory_order_relaxed);
  }

  // steady_clock timestamp of that same pause end (process-local clock only): lets an
  // in-process caller place the pause on its own timeline instead of inferring it from a lagging
  // total_splits() observation.
  [[nodiscard]] auto last_split_pause_end_ns() const -> uint64_t {
    return last_split_pause_end_ns_.load(std::memory_order_relaxed);
  }

  // Sums one per-shard counter under a single routing guard (directory_ must not be read
  // unguarded).
  template <typename Total, typename Counter>
  auto reduce_shards(Counter counter) const -> Total {
    return with_routing_guard([&]() -> Total {
      Total total = 0;
      for (ShardSlot *slot : directory_.load(std::memory_order_acquire)->shards) {
        total += (slot->index.get()->*counter)();
      }
      return total;
    });
  }

  // Sum of every shard's current append occupancy: "anything left to merge" for a T1-only
  // reorganize skip-if-empty check.
  [[nodiscard]] auto append_size() const -> size_t { return reduce_shards<size_t>(&Shard::append_size); }

  // Whole-store aggregates of the per-shard RSS/churn counters. Live shards only: a retired
  // shard's peak drops out (monitoring use; not a precise lifetime maximum).
  [[nodiscard]] auto append_region_live_count() const -> int64_t {
    return reduce_shards<int64_t>(&Shard::append_region_live_count);
  }

  // Approximate: sums each *currently live* shard's own peak, so a shard retired by a split after
  // reaching a high peak doesn't contribute it here. Acceptable for its monitoring use (see
  // T1Index::append_region_live_count()'s comment) -- not a precise store-lifetime maximum.
  [[nodiscard]] auto append_region_peak_count() const -> int64_t {
    return reduce_shards<int64_t>(&Shard::append_region_peak_count);
  }

  // Test/maintenance hook: full Closing -> re-reorganize -> Split on `key`'s shard. No-op when
  // already splitting (single-flight). The second reorganize() is a correctness requirement
  // (see docs/specification/t1_sharding_design.md "Splitting"), not an optimization.
  //
  // Resolve + Closing CAS happen inside one guard: the CAS dereferences the slot, and a gap
  // would let a concurrent split free it first. continue_split() then runs unguarded on the
  // won exclusive ownership.
  void split_shard_containing(std::span<const std::byte> key) {
    const Key prefix = t1_detail::prefix_from_bytes(key);
    bool claimed = false;
    ShardSlot *target = with_routing_guard([&]() -> ShardSlot * {
      ShardSlot *slot = resolve(prefix);
      if (splits_paused_.load(std::memory_order_acquire) != 0 || slot->split_paused.load(std::memory_order_seq_cst)) {
        return slot;  // checkpoint_all_shards() in progress -- see its own comment. No-op.
      }
      Split *expected = nullptr;
      claimed = slot->superseded.compare_exchange_strong(expected, kClosingSentinel, std::memory_order_seq_cst);
      if (claimed && slot->split_paused.load(std::memory_order_seq_cst)) {
        slot->superseded.store(nullptr, std::memory_order_release);
        claimed = false;
      }
      return slot;
    });
    if (claimed) {
      continue_split(target);
    }
  }

  // Captures every shard's live entries (merging first, unless append is empty -- then the
  // sorted region serializes as-is, byte-identical to a merge at O(sorted) cost) and invokes
  // offset_mapper then per_shard_writer per shard, ascending -- mirroring reorganize()'s
  // contract once per shard, so a stateful offset_mapper works unmodified. Returns boundary
  // keys for load_from_checkpoint().
  //
  // Splits pause for the whole call (directory stays stable); in-flight splits still complete,
  // so touched shards are held by outside_refs, not one long guard. Only the directory snapshot
  // is taken guarded. Non-splitting maintenance still runs concurrently: reorg_in_progress_
  // single-flights it against this loop's own reorganize(), and the loser skips its callback --
  // hence reorganize_until_captured() retries until the callback fires.
  template <typename OffsetMapper, typename PerShardWriter>
  auto checkpoint_all_shards(OffsetMapper offset_mapper, PerShardWriter per_shard_writer) -> std::vector<Key> {
    SplitPause pause(splits_paused_);
    wait_until_no_split();

    std::vector<ShardSlot *> snapshot;
    std::vector<Key> boundaries;
    with_routing_guard([&] {
      Directory *dir = directory_.load(std::memory_order_acquire);
      boundaries = dir->boundaries;
      snapshot = dir->shards;
      for (ShardSlot *slot : snapshot) {
        slot->outside_refs.fetch_add(1, std::memory_order_relaxed);
      }
    });
    struct RefRelease {
      std::vector<ShardSlot *> *slots;
      ~RefRelease() {
        for (ShardSlot *slot : *slots) {
          slot->outside_refs.fetch_sub(1, std::memory_order_relaxed);
        }
      }
    } ref_release{&snapshot};

    const size_t total_shards = snapshot.size();
    t1_debug::g_checkpoint_total_shards.store(total_shards, std::memory_order_release);
    std::vector<EntrySnapshot> dumped;
    size_t shard_idx = 0;
    for (ShardSlot *slot : snapshot) {
      ++shard_idx;
      t1_debug::g_checkpoint_current_shard.store(shard_idx, std::memory_order_release);
      const auto shard_start = std::chrono::steady_clock::now();
      // Merge-free shortcut: an empty active append means the merge output is the sorted
      // region as-is (same order, same tombstone-skipping) -- but only when no merge is
      // in flight (an in-flight reorganize() has frozen the previous active into immutable,
      // which a sorted-only dump would silently drop). Concurrent writes stay consistent for
      // the same reason a merge snapshot does: they are WAL-covered past this cycle's
      // checkpoint_lsn and converge via replay. Tombstone pressure is not reset here --
      // only a real merge carries tombstones away.
      if (slot->index->append_size() == 0 && !slot->index->has_immutable()) {
        slot->index->dump_sorted_region(dumped);
        offset_mapper(std::span<EntrySnapshot>(dumped));
        per_shard_writer(dumped);
      } else {
        reorganize_until_captured(*slot->index, offset_mapper, per_shard_writer, /*parallel_sort=*/true);
        slot->tombstones_since_maintenance.store(0, std::memory_order_relaxed);
      }
      const auto shard_elapsed = std::chrono::steady_clock::now() - shard_start;
      if (shard_elapsed > std::chrono::seconds(1)) {
        std::cerr << "[checkpoint_all_shards] Shard " << shard_idx << "/" << total_shards << " took "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(shard_elapsed).count() << "ms\n";
      }
      // Same re-drive as for_each_dirty_shard: checkpoint merges drain appends without
      // splitting, so a big shard must re-enter maintenance here. Approximate post-merge
      // size is enough as a gate; maintenance re-merges and confirms the exact size.
      if (slot->index->approx_entry_count() >=
          split_threshold(target_shard_size_, Config::kShardSplitThresholdPercent)) {
        queue_.push_if_live(slot);
      }
    }
    t1_debug::g_checkpoint_current_shard.store(0, std::memory_order_release);
    return boundaries;
  }

  // Recovery-only: replaces this fresh, still-unshared instance's single shard with a
  // directory rebuilt from checkpoint data. No synchronization; call before publishing this
  // instance to any thread (a fresh instance's queue is always empty, so no worker can act).
  void load_from_checkpoint(std::span<const Key> boundaries,
                            std::span<const std::vector<EntrySnapshot>> per_shard_entries) {
    auto *new_dir = new Directory();
    new_dir->boundaries.assign(boundaries.begin(), boundaries.end());
    new_dir->shards.reserve(per_shard_entries.size());
    for (const auto &entries : per_shard_entries) {
      auto shard = std::make_unique<Shard>(append_cap_);
      shard->load_sorted_region_from_checkpoint(entries);
      new_dir->shards.push_back(new ShardSlot(std::move(shard)));
    }

    Directory *old_dir = directory_.load(std::memory_order_relaxed);
    for (ShardSlot *slot : old_dir->shards) {
      delete slot;
    }
    delete old_dir;
    directory_.store(new_dir, std::memory_order_relaxed);
  }

  // Recovery-only T2 maintenance state initialization. `shard_index` addresses the loaded
  // checkpoint directory and must be called before the instance is published to workers.
  void mark_t2_shard_state(size_t shard_index, bool dirty) {
    Directory *dir = directory_.load(std::memory_order_relaxed);
    assert(shard_index < dir->shards.size());
    dir->shards[shard_index]->t2_dirty.store(dirty, std::memory_order_relaxed);
  }

  [[nodiscard]] auto dirty_shard_count() const -> size_t {
    return with_routing_guard([&]() -> size_t {
      size_t count = 0;
      for (ShardSlot *slot : directory_.load(std::memory_order_acquire)->shards) {
        count += static_cast<size_t>(slot->t2_dirty.load(std::memory_order_relaxed));
      }
      return count;
    });
  }

  // True when any live shard has uncompacted T2 writes or dead T2 records.
  [[nodiscard]] auto has_dirty_shard() const -> bool {
    return with_routing_guard([&]() -> bool {
      Directory *dir = directory_.load(std::memory_order_acquire);
      return std::ranges::any_of(dir->shards,
                                 [](const ShardSlot *slot) { return slot->t2_dirty.load(std::memory_order_relaxed); });
    });
  }

  // Visits dirty shards in ascending key-range order. For each dirty shard, atomically clears
  // its dirty/garbage flags and captures live EntrySnapshots. The visitor receives both flags.
  // Returning false or throwing restores the dirty state, leaving the shard eligible for a
  // later cycle.
  // Spins until no shard is mid-split (Closing or Split published but not yet drained).
  // Splits are paused above us, so this only waits out splits that won Closing earlier.
  void wait_until_no_split() const {
    SpinBackoff split_backoff;
    for (;;) {
      const bool split_active = with_routing_guard([&] {
        Directory *dir = directory_.load(std::memory_order_acquire);
        return std::ranges::any_of(dir->shards, [](const ShardSlot *slot) {
          return slot->superseded.load(std::memory_order_acquire) != nullptr;
        });
      });
      if (!split_active) {
        break;
      }
      split_backoff.wait();
    }
  }

  void drain_maintenance() {
    const size_t threshold = split_threshold(target_shard_size_, Config::kShardSplitThresholdPercent);
    with_routing_guard([&] {
      Directory *dir = directory_.load(std::memory_order_acquire);
      for (ShardSlot *slot : dir->shards) {
        if (slot->index->approx_entry_count() >= threshold || slot->index->append_size() > 0) {
          queue_.push_if_live(slot);
        }
      }
    });
    SpinBackoff backoff;
    for (;;) {
      if (run_maintenance()) {
        continue;
      }
      wait_until_no_split();
      if (queue_.empty() && active_maintenance_.load(std::memory_order_acquire) == 0) {
        break;
      }
      backoff.wait();
    }
  }

  template <typename Visitor>
  void for_each_dirty_shard(Visitor &&visitor) {
    std::vector<ShardSlot *> snapshot;
    with_routing_guard([&] {
      Directory *dir = directory_.load(std::memory_order_acquire);
      snapshot.reserve(dir->shards.size());
      for (ShardSlot *slot : dir->shards) {
        if (slot->t2_dirty.load(std::memory_order_relaxed) &&
            slot->superseded.load(std::memory_order_acquire) == nullptr) {
          slot->outside_refs.fetch_add(1, std::memory_order_relaxed);
          snapshot.push_back(slot);
        }
      }
    });

    std::vector<EntrySnapshot> entries;
    size_t next = 0;
    try {
      for (; next < snapshot.size(); ++next) {
        if (visit_dirty_shard(snapshot[next], entries, visitor) == DirtyShardVisit::kStop) {
          ++next;
          break;
        }
      }
    } catch (...) {
      // The throwing visit already released its own pin; only later slots still hold theirs.
      release_snapshot_tail(snapshot, next + 1);
      throw;
    }
    // The unvisited tail holds snapshot pins until released here.
    release_snapshot_tail(snapshot, next);
  }

 private:
  struct ShardSlot;

  // Published once, atomically, on Closing -> Split. Keys < boundary route to `low`.
  struct Split {
    Key boundary{};
    ShardSlot *low = nullptr;
    ShardSlot *high = nullptr;
  };

  // Split pause is a refcount, not a flag: checkpoint_all_shards() and
  // for_each_dirty_shard() pause independently and may overlap; whoever exits
  // last resumes. (A shared bool would let the first exiter re-enable splits
  // while the other pauser is still inside.)
  struct SplitPause {
    std::atomic<int> *count = nullptr;
    explicit SplitPause(std::atomic<int> &guarded) noexcept : count(&guarded) {
      count->fetch_add(1, std::memory_order_release);
    }
    ~SplitPause() {
      if (count != nullptr) {
        count->fetch_sub(1, std::memory_order_release);
      }
    }
    SplitPause(const SplitPause &) = delete;
    auto operator=(const SplitPause &) -> SplitPause & = delete;
  };

  // Never dereferenced, only compared: "splitting, writes wait, children undecided".
  static Split *const kClosingSentinel;

  struct ShardSlot {
    explicit ShardSlot(std::unique_ptr<Shard> shard) : index(std::move(shard)) {}
    std::unique_ptr<Shard> index;
    // null: normal. kClosingSentinel: splitting (writes wait; reads use `index` directly).
    // Else a real Split*: redirect.
    mutable std::atomic<Split *> superseded{nullptr};
    // One outstanding maintenance request per shard. Cleared at run_maintenance()'s start, not
    // end: a trip during the run re-enqueues instead of dropping.
    std::atomic<bool> maintenance_pending{false};
    // Tombstone puts since the last reorganize() (which merges them away). Delete-pressure
    // trigger; relaxed heuristic -- a racing reset only delays the next trigger one round.
    std::atomic<uint64_t> tombstones_since_maintenance{0};
    // T2 compaction flags: t2_dirty marks uncompacted T2 appends; t2_inversion marks out-of-order
    // key appends within this shard.
    std::atomic<bool> t2_dirty{false};
    std::atomic<bool> t2_inversion{false};
    std::atomic<uint64_t> last_t2_prefix_hi{0};
    std::atomic<uint64_t> last_t2_prefix_lo{0};
    std::atomic<bool> has_last_t2_prefix{false};
    std::atomic<bool> split_paused{false};
    // Hazard refs for unguarded worker/checkpoint touches; continue_split() waits for zero.
    std::atomic<int> outside_refs{0};
  };

  // Outcome of one dirty-shard visit: keep scanning shards or stop at this one.
  enum class DirtyShardVisit : std::uint8_t { kContinue, kStop };

  // Captures one dirty shard and releases its snapshot pin on every return path.
  template <typename Visitor>
  auto visit_dirty_shard(ShardSlot *slot, std::vector<EntrySnapshot> &entries, Visitor &&visitor) -> DirtyShardVisit {
    slot->split_paused.store(true, std::memory_order_seq_cst);
    auto finish_visit = [&](bool check_split) {
      slot->split_paused.store(false, std::memory_order_release);
      if (check_split && entries.size() >= split_threshold(target_shard_size_, Config::kShardSplitThresholdPercent)) {
        queue_.push_if_live(slot);
      }
      slot->outside_refs.fetch_sub(1, std::memory_order_release);
    };
    if (slot->superseded.load(std::memory_order_seq_cst) != nullptr) {
      finish_visit(false);
      return DirtyShardVisit::kContinue;
    }
    const bool had_dirty = slot->t2_dirty.exchange(false, std::memory_order_acq_rel);
    const bool had_inversion = slot->t2_inversion.exchange(false, std::memory_order_acq_rel);
    if (!had_dirty && !had_inversion) {
      finish_visit(false);
      return DirtyShardVisit::kContinue;
    }
    auto restore = [&](bool check_split) {
      slot->t2_dirty.store(true, std::memory_order_release);
      if (had_inversion) {
        slot->t2_inversion.store(true, std::memory_order_release);
      }
      finish_visit(check_split);
    };
    try {
      if (slot->index->append_size() == 0 && !slot->index->has_immutable()) {
        slot->index->dump_sorted_region(entries);
      } else {
        if (!had_inversion) {
          finish_visit(false);
          return DirtyShardVisit::kContinue;
        }
        reorganize_until_captured(
            *slot->index,
            [](std::span<EntrySnapshot> /*merged*/) {},
            [&](std::span<const EntrySnapshot> merged) { entries.assign(merged.begin(), merged.end()); },
            /*parallel_sort=*/false);
        slot->tombstones_since_maintenance.store(0, std::memory_order_relaxed);
      }
      if (!visitor(std::span<const EntrySnapshot>(entries))) {
        restore(true);
        return DirtyShardVisit::kStop;
      }
    } catch (...) {
      restore(false);
      throw;
    }
    finish_visit(true);
    return DirtyShardVisit::kContinue;
  }

  // Releases snapshot pins for shards that do not complete a visitor call.
  static void release_snapshot_tail(const std::vector<ShardSlot *> &snapshot, size_t first) {
    for (size_t index = first; index < snapshot.size(); ++index) {
      snapshot[index]->outside_refs.fetch_sub(1, std::memory_order_release);
    }
  }

  // Pure split-size threshold shared by run_maintenance().
  static auto split_threshold(size_t target_shard_size, uint32_t percent) noexcept -> size_t {
    return sharded_detail::split_threshold(target_shard_size, percent);
  }

  // Queue ownership shared by request_maintenance_if_needed()/run_maintenance()/continue_split().
  struct ShardQueue {
    std::mutex mutex;
    std::deque<ShardSlot *> queue;
    // Enqueues slot if live: CASes pending false->true, then under lock re-checks superseded
    // (same mutex continue_split() purges under, making push vs purge race-free).
    void push_if_live(ShardSlot *slot) {
      bool expected = false;
      if (!slot->maintenance_pending.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        return;
      }
      const std::lock_guard<std::mutex> lock(mutex);
      if (slot->superseded.load(std::memory_order_acquire) != nullptr) {
        return;
      }
      queue.push_back(slot);
    }
    // Pops one entry (or nullptr) and clears its pending flag for the next round.
    auto pop() -> ShardSlot * {
      const std::lock_guard<std::mutex> lock(mutex);
      if (queue.empty()) {
        return nullptr;
      }
      ShardSlot *slot = queue.front();
      queue.pop_front();
      slot->maintenance_pending.store(false, std::memory_order_release);
      return slot;
    }
    void purge(ShardSlot *target) {
      const std::lock_guard<std::mutex> lock(mutex);
      queue.erase(std::remove(queue.begin(), queue.end(), target), queue.end());
    }
    [[nodiscard]] auto empty() -> bool {
      const std::lock_guard<std::mutex> lock(mutex);
      return queue.empty();
    }
  };

  // Immutable once published; replaced wholesale (RCU) per split. shards[i] owns
  // [boundaries[i-1], boundaries[i]).
  struct Directory {
    std::vector<Key> boundaries;
    std::vector<ShardSlot *> shards;

    [[nodiscard]] auto shard_for(const Key &key) const -> ShardSlot * {
      const auto upper = std::upper_bound(boundaries.begin(), boundaries.end(), key);
      return shards[static_cast<size_t>(upper - boundaries.begin())];
    }
  };

  // Single-shard fast path: no boundary search, leaf collection, or per-call allocation (a split
  // in progress reads through Closing). Nullopt when superseded by a completed Split.
  template <typename Callback>
  auto scan_single_shard(Directory *dir,
                         std::span<const std::byte> lo_bytes,
                         std::span<const std::byte> hi_bytes,
                         Callback callback) const -> std::optional<size_t> {
    ShardSlot *slot = dir->shards[0];
    Split *split = slot->superseded.load(std::memory_order_acquire);
    if (split != nullptr && split != kClosingSentinel) {
      return std::nullopt;
    }
    return slot->index->scan(lo_bytes, hi_bytes, callback);
  }

  template <typename Callback>
  auto scan_multi_shard(Directory *dir,
                        const Key &lower,
                        const Key &upper,
                        std::span<const std::byte> lo_bytes,
                        std::span<const std::byte> hi_bytes,
                        Callback callback,
                        bool &stopped) const -> size_t {
    std::vector<ShardSlot *> leaves;
    ShardSetView{dir}.for_each_leaf_in_range(lower, upper, [&](ShardSlot *leaf) { leaves.push_back(leaf); });
    size_t total = 0;
    for (ShardSlot *leaf : leaves) {
      total += leaf->index->scan(lo_bytes, hi_bytes, callback);
      if (stopped) {
        break;
      }
    }
    return total;
  }

  // Reads use a Closing slot directly; writes return no slot and retry after releasing their
  // routing guard. Single-shard shortcut: routing is trivial below the first split threshold,
  // though the epoch guard is never skipped.
  enum class ResolveMode : std::uint8_t { Read, Write };

  [[nodiscard]] auto resolve_impl(const Key &key, ResolveMode mode) const -> ShardSlot * {
    Directory *dir = directory_.load(std::memory_order_acquire);
    ShardSlot *slot = dir->shards.size() == 1 ? dir->shards[0] : dir->shard_for(key);
    for (;;) {
      Split *split = slot->superseded.load(std::memory_order_acquire);
      if (split == nullptr) {
        return slot;
      }
      if (split == kClosingSentinel) {
        if (mode == ResolveMode::Read) {
          return slot;
        }
        return nullptr;
      }
      slot = (key < split->boundary) ? split->low : split->high;
    }
  }

  [[nodiscard]] auto resolve(const Key &key) const -> ShardSlot * { return resolve_impl(key, ResolveMode::Read); }

  [[nodiscard]] auto resolve_for_write(const Key &key) const -> ShardSlot * {
    return resolve_impl(key, ResolveMode::Write);
  }

  // View over one Directory snapshot: initial routing plus leaf fan-out in range.
  struct ShardSetView {
    Directory *dir;
    void for_each_leaf_in_range(const Key &lower, const Key &upper, auto &&visit) const {
      const size_t start_idx =
          dir->boundaries.empty()
              ? 0
              : static_cast<size_t>(std::upper_bound(dir->boundaries.begin(), dir->boundaries.end(), lower) -
                                    dir->boundaries.begin());
      for (size_t i = start_idx; i < dir->shards.size(); ++i) {
        if (i > 0 && !(dir->boundaries[i - 1] < upper)) {
          break;
        }
        collect_into(dir->shards[i], lower, upper, visit);
      }
    }
    static void collect_into(ShardSlot *slot, const Key &lower, const Key &upper, auto &&visit) {
      Split *split = slot->superseded.load(std::memory_order_acquire);
      if (split == nullptr || split == kClosingSentinel) {
        visit(slot);  // Closing reads through the splitting slot directly.
        return;
      }
      if (lower < split->boundary) {
        collect_into(split->low, lower, upper, visit);
      }
      if (split->boundary < upper) {
        collect_into(split->high, lower, upper, visit);
      }
    }
  };

  template <typename Func>
  auto with_routing_guard(Func &&func) const noexcept(noexcept(func())) -> decltype(func()) {
    typename ThreadReferenceTracker<uint64_t>::Guard handle(routing_epochs_,
                                                            routing_epoch_.load(std::memory_order_relaxed));
    return func();
  }

  // Retries reorganize() until `on_merged` fires (see sharded_maintenance.hpp).
  template <typename OffsetMapper, typename OnMerged>
  static void reorganize_until_captured(Shard &shard,
                                        OffsetMapper offset_mapper,
                                        OnMerged on_merged,
                                        bool parallel_sort) {
    sharded_detail::reorganize_until_captured(shard, offset_mapper, on_merged, parallel_sort);
  }

  // Post-Closing-CAS split body. Unguarded by design: winning the CAS made `target` this
  // thread's exclusive property, and guarding this slow part would deadlock its own
  // wait_until_epoch() against the waited-on guard. The CAS itself must be won guarded
  // (split_shard_containing()/run_maintenance()): it dereferences `target`, and a concurrent
  // split completing in a resolve-to-CAS gap would leave it touching freed memory.
  void continue_split(ShardSlot *target) {
    // Pause-clock start: writers spin on Closing from the run_maintenance() CAS just before
    // this call (close enough; nothing runs between). Stopped at the Split publish below --
    // the drain/redistribution after runs unblocked, so full-call duration would overstate it.
    const auto split_pause_start = std::chrono::steady_clock::now();

    // Purge `target`'s queue entry under push_if_live()'s mutex: no worker dequeues it after
    // this point. Already-dequeued workers hold outside_refs, drained before `delete target`.
    queue_.purge(target);

    // Second reorganize() after Closing is visible: captures "no new writer resolves target"
    // state (in-flight put()s drain via T1Index's own epoch mechanism). A writer that resolved
    // just before Closing can still land after this snapshot -- closed by put()'s own
    // superseded re-check forwarding to the live shard (see put()).
    //
    // Retried: a duplicate queued run_maintenance() for this shard can be mid-reorganize() now,
    // making this call a no-op via the reorg_in_progress_ CAS. Without retry, lost races would
    // abort every split attempt under write pressure (resetting superseded to null makes them
    // look like "too small"), growing the shard without bound.
    std::vector<EntrySnapshot> merged_entries;
    reorganize_until_captured(
        *target->index,
        [](std::span<EntrySnapshot> /*merged*/) {},
        [&](std::span<const EntrySnapshot> merged) { merged_entries.assign(merged.begin(), merged.end()); },
        /*parallel_sort=*/false);  // Many shards' reorganize() run concurrently; see
                                   // T1Index::reorganize()'s own doc comment on this parameter.

    constexpr size_t kMinSplitEntries = 2;
    if (merged_entries.size() < kMinSplitEntries) {
      // Too small (callback fired: real state, not a lost race) -- resume normal operation.
      // Reset superseded *before* maintenance_pending: request_maintenance_if_needed() CASes the
      // flag then checks superseded, so the reverse order could strand the flag true forever on
      // a shard about to read normal again. Clearing the flag at all is still needed: the purged
      // (never dequeued) entry's flag would otherwise never clear.
      target->superseded.store(nullptr, std::memory_order_release);
      target->maintenance_pending.store(false, std::memory_order_relaxed);
      target->tombstones_since_maintenance.store(0, std::memory_order_relaxed);
      return;
    }

    const size_t mid = merged_entries.size() / 2;
    const Key boundary = merged_entries[mid].key;
    auto low_shard = std::make_unique<Shard>(target->index->append_capacity());
    auto high_shard = std::make_unique<Shard>(target->index->append_capacity());
    low_shard->load_sorted_region_from_checkpoint(std::span<const EntrySnapshot>(merged_entries.data(), mid));
    high_shard->load_sorted_region_from_checkpoint(
        std::span<const EntrySnapshot>(merged_entries.data() + mid, merged_entries.size() - mid));

    auto *low_slot = new ShardSlot(std::move(low_shard));
    auto *high_slot = new ShardSlot(std::move(high_shard));
    const bool dirty = target->t2_dirty.load(std::memory_order_relaxed);
    const bool inversion = target->t2_inversion.load(std::memory_order_relaxed);
    low_slot->t2_dirty.store(dirty, std::memory_order_relaxed);
    high_slot->t2_dirty.store(dirty, std::memory_order_relaxed);
    low_slot->t2_inversion.store(inversion, std::memory_order_relaxed);
    high_slot->t2_inversion.store(inversion, std::memory_order_relaxed);
    low_slot->outside_refs.store(1, std::memory_order_relaxed);
    high_slot->outside_refs.store(1, std::memory_order_relaxed);

    Directory *old_dir = publish_directory_after_split(target, low_slot, high_slot, boundary);

    auto *split_info = new Split{boundary, low_slot, high_slot};
    // Closing -> Split: spinning writers wake and redirect.
    target->superseded.store(split_info, std::memory_order_release);

    const auto split_pause_end = std::chrono::steady_clock::now();
    last_split_pause_us_.store(
        static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(split_pause_end - split_pause_start).count()),
        std::memory_order_relaxed);
    // Process-local steady_clock timestamp (comparable only to same-process now() calls):
    // total_splits_ increments only after the drain/redistribution below, so observing it would
    // place the pause late.
    last_split_pause_end_ns_.store(static_cast<uint64_t>(split_pause_end.time_since_epoch().count()),
                                   std::memory_order_relaxed);

    const size_t threshold = split_threshold(target_shard_size_, Config::kShardSplitThresholdPercent);
    if (mid >= threshold && merged_entries.front().key < merged_entries[mid - 1].key) {
      queue_.push_if_live(low_slot);
    }
    if (merged_entries.size() - mid >= threshold && merged_entries[mid].key < merged_entries.back().key) {
      queue_.push_if_live(high_slot);
    }
    low_slot->outside_refs.fetch_sub(1, std::memory_order_release);
    high_slot->outside_refs.fetch_sub(1, std::memory_order_release);

    // Drain routing-guarded calls still referencing `target`/`old_dir`, then free both. No
    // bulk redistribution of leftovers: post-snapshot writes re-apply themselves via put()'s
    // superseded forwarding (never overwriting a newer completed write, unlike a bulk copy
    // here). Drain after the Split publish (not before the snapshot): writers spin on Closing
    // until this function releases them, so draining first would self-deadlock.
    const uint64_t bumped = routing_epoch_.fetch_add(1, std::memory_order_acq_rel) + 1;
    routing_epochs_.wait_until_epoch(bumped);

    // Drain unguarded maintenance/checkpoint users holding outside_refs before freeing.
    {
      SpinBackoff backoff;
      while (target->outside_refs.load(std::memory_order_acquire) != 0) {
        backoff.wait();
      }
    }

    delete old_dir;
    delete target;
    total_splits_.fetch_add(1, std::memory_order_relaxed);
  }

  // Swaps `target` for {low_slot, high_slot} at `boundary` via CAS retry (concurrent splits
  // of *other* shards retry independently; `target` is single-flight by its Closing CAS).
  // Returns the retired Directory* for the caller to free after its epoch drain.
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  auto publish_directory_after_split(ShardSlot *target,
                                     ShardSlot *low_slot,
                                     ShardSlot *high_slot,
                                     const Key &boundary) -> Directory * {
    return with_routing_guard([&]() -> Directory * {
      for (;;) {
        Directory *old_dir = directory_.load(std::memory_order_acquire);
        const auto target_it = std::find(old_dir->shards.begin(), old_dir->shards.end(), target);
        assert(target_it != old_dir->shards.end());
        const auto idx = static_cast<size_t>(target_it - old_dir->shards.begin());

        auto *new_dir = new Directory(*old_dir);
        new_dir->shards[idx] = low_slot;
        new_dir->shards.insert(new_dir->shards.begin() + static_cast<std::ptrdiff_t>(idx) + 1, high_slot);
        new_dir->boundaries.insert(new_dir->boundaries.begin() + static_cast<std::ptrdiff_t>(idx), boundary);

        if (directory_.compare_exchange_weak(old_dir, new_dir, std::memory_order_acq_rel, std::memory_order_relaxed)) {
          return old_dir;
        }
        delete new_dir;  // Lost the race to a concurrent split of a different shard; retry.
      }
    });
  }

  static auto default_worker_count() -> size_t {
    const unsigned int hardware_threads = std::thread::hardware_concurrency();
    return std::max<size_t>(1, (hardware_threads == 0 ? 4 : hardware_threads) / 4);
  }

  // Enqueues past the soft threshold (scan-aware) or a shard's worth of tombstones -- deletes
  // only retarget T1 offsets, so delete-heavy workloads need the tombstone half to ever fire.
  // Push-vs-purge race-freedom comes from push_if_live()'s mutex-serialized superseded
  // re-check: ordered-before-purge gets cleaned; ordered-after sees set superseded and backs
  // out. No post-return queue entry for a splitting shard, ever.
  void request_maintenance_if_needed(ShardSlot *slot) {
    const size_t size = slot->index->append_size();
    const size_t cap = slot->index->append_capacity();
    const bool over_soft = size >= maintenance_soft_threshold(cap, scan_active_.load(std::memory_order_relaxed));
    const bool delete_heavy = slot->tombstones_since_maintenance.load(std::memory_order_relaxed) >= target_shard_size_;
    if (!over_soft && !delete_heavy) {
      return;
    }
    queue_.push_if_live(slot);
  }

  // Fixed-interval poll, not a condvar: bounds shutdown latency with no wakeup signal, and the
  // interval is noise against a multi-millisecond maintenance cycle.
  void worker_loop(const std::stop_token &stop_token) {
    constexpr auto kIdlePollInterval = kDefaultPoll10ms;
    poll_worker_until_stop(stop_token, kIdlePollInterval, [&] { return run_maintenance(); });
  }

  // Dequeue-only entry point (must itself run guarded -- see run_maintenance()).
  auto pop_queue() -> ShardSlot * { return queue_.pop(); }

  // Split gate shared by every reorganize completion site (maintenance worker,
  // dirty-shard visits, checkpoints): split-worthiness is a property of the merged size,
  // not of the path that merged. Returns true when this call won exclusive split ownership
  // (caller must then drop its own outside_refs pin before continue_split(), which drains
  // pins). The caller must already pin `slot` (routing guard or outside_refs); no extra
  // guard is taken here.
  [[nodiscard]] auto try_claim_split(ShardSlot *slot, size_t merged_size) -> bool {
    if (merged_size < split_threshold(target_shard_size_, Config::kShardSplitThresholdPercent)) {
      return false;
    }
    if (splits_paused_.load(std::memory_order_acquire) != 0 || slot->split_paused.load(std::memory_order_seq_cst)) {
      return false;
    }
    if (slot->superseded.load(std::memory_order_acquire) != nullptr) {
      return false;
    }
    Split *expected = nullptr;
    if (!slot->superseded.compare_exchange_strong(expected, kClosingSentinel, std::memory_order_seq_cst)) {
      return false;
    }
    if (slot->split_paused.load(std::memory_order_seq_cst)) {
      slot->superseded.store(nullptr, std::memory_order_release);
      return false;
    }
    return true;
  }

  // One cycle: dequeue, reorganize, split when big enough. False only on an empty queue (tells
  // worker_loop() to poll-sleep). Only dequeue runs guarded; the reorganize() between runs
  // unguarded, kept alive by outside_refs (taken guarded). Duplicate processing by multiple
  // workers is harmless (reorg_in_progress_ CAS + single-flight Closing CAS). A big shard
  // seen while splits are paused is re-queued, never dropped: pauses are bounded, and every
  // merge completion site re-drives this queue, so the split check cannot starve.
  auto run_maintenance() -> bool {
    struct ActiveGuard {
      std::atomic<size_t> *counter;
      explicit ActiveGuard(std::atomic<size_t> &guarded) noexcept : counter(&guarded) {
        counter->fetch_add(1, std::memory_order_acq_rel);
      }
      ~ActiveGuard() { counter->fetch_sub(1, std::memory_order_release); }
      ActiveGuard(const ActiveGuard &) = delete;
      auto operator=(const ActiveGuard &) -> ActiveGuard & = delete;
    } active_guard(active_maintenance_);
    struct Claimed {
      ShardSlot *slot = nullptr;
      bool held = false;
    };
    Claimed claimed = with_routing_guard([&]() -> Claimed {
      ShardSlot *slot = pop_queue();
      if (slot == nullptr) {
        return {};
      }
      if (slot->superseded.load(std::memory_order_acquire) != nullptr) {
        return {slot, false};
      }
      slot->outside_refs.fetch_add(1, std::memory_order_relaxed);
      return {slot, true};
    });
    if (claimed.slot == nullptr) {
      return false;
    }
    if (!claimed.held) {
      return true;
    }
    ShardSlot *slot = claimed.slot;

    // Split-worthiness gate: while splits are paused and there is nothing to drain,
    // re-drive later instead of burning a full merge every poll. Draining itself must
    // never wait on the pause: a full append blocks writers, and writers keep the
    // compaction worker (the pause holder) busy -- waiting would deadlock bulk loads.
    const bool pause_active =
        splits_paused_.load(std::memory_order_acquire) != 0 || slot->split_paused.load(std::memory_order_acquire);
    if (pause_active &&
        slot->index->append_size() <
            maintenance_soft_threshold(slot->index->append_capacity(), scan_active_.load(std::memory_order_relaxed))) {
      slot->outside_refs.fetch_sub(1, std::memory_order_relaxed);
      queue_.push_if_live(slot);
      return false;
    }

    std::vector<EntrySnapshot> merged_entries;
    bool reorganized = false;
    slot->index->reorganize([](std::span<EntrySnapshot> /*merged*/) {},
                            [&](std::span<const EntrySnapshot> merged) {
                              merged_entries.assign(merged.begin(), merged.end());
                              reorganized = true;
                            },
                            /*parallel_sort=*/false);
    if (reorganized) {
      slot->tombstones_since_maintenance.store(0, std::memory_order_relaxed);
    }

    const bool won = try_claim_split(slot, merged_entries.size());
    if (!won && merged_entries.size() >= split_threshold(target_shard_size_, Config::kShardSplitThresholdPercent) &&
        slot->superseded.load(std::memory_order_acquire) == nullptr) {
      // Lost to a pause (or a lost reorganize race another round will retry): re-drive
      // instead of dropping the signal.
      queue_.push_if_live(slot);
    }
    slot->outside_refs.fetch_sub(1, std::memory_order_relaxed);
    if (won) {
      continue_split(slot);
    }
    return true;
  }

  // See set_scan_active().
  mutable std::atomic<bool> scan_active_{false};
  mutable ThreadReferenceTracker<uint64_t> routing_epochs_;
  std::atomic<uint64_t> routing_epoch_{1};
  std::atomic<uint64_t> total_splits_{0};
  std::atomic<uint64_t> last_split_pause_us_{0};
  std::atomic<uint64_t> last_split_pause_end_ns_{0};
  std::atomic<Directory *> directory_{nullptr};
  size_t append_cap_;
  size_t target_shard_size_;
  ShardQueue queue_;
  std::vector<std::jthread> workers_;
  // Set for checkpoint_all_shards()' duration: directory stays stable (splits initiate only
  // at the Closing CAS, which checks this); in-flight splits are unaffected.
  std::atomic<int> splits_paused_{0};
  std::atomic<size_t> active_maintenance_{0};
};

template <typename Config>
typename ShardedT1Index<Config>::Split *const ShardedT1Index<Config>::kClosingSentinel =
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    reinterpret_cast<typename ShardedT1Index<Config>::Split *>(static_cast<uintptr_t>(1));

}  // namespace vmemkv
