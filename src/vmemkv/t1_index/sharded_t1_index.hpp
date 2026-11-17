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
    auto *slot = new ShardSlot(std::make_unique<Shard>(append_cap, &epoch_domain_));
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
      return slot->index->get_with_hash(key, kCallerHeldEpoch);
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
        const PutResult result = slot->index->put(key, value, is_inline, inline_size, kCallerHeldEpoch);
        if (result == PutResult::AppendRegionFull || slot->superseded.load(std::memory_order_acquire) != nullptr) {
          return std::nullopt;
        }
        note_put_applied(slot, value, is_inline, mark_t2_dirty);
        return result;
      });
      if (attempted.has_value()) {
        return *attempted;
      }
      backoff.wait();
    }
  }

  // Append entries allowed before maintenance: half capacity. Pure function, unit-testable.
  static constexpr size_t kPercentScale = sharded_detail::kPercentScale;
  static auto maintenance_soft_threshold(size_t append_cap) noexcept -> size_t {
    return sharded_detail::maintenance_soft_threshold(append_cap);
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

  // Sums one per-shard value under a single routing guard (directory_ must not be read
  // unguarded).
  template <typename Total, typename Fn>
  auto reduce_shards(Fn &&func) const -> Total {
    return with_routing_guard([&]() -> Total {
      Total total = 0;
      for (ShardSlot *slot : directory_.load(std::memory_order_acquire)->shards) {
        total += func(*slot->index);
      }
      return total;
    });
  }

  // Sum of every shard's current append occupancy: "anything left to merge" for a T1-only
  // reorganize skip-if-empty check.
  [[nodiscard]] auto append_size() const -> size_t {
    return reduce_shards<size_t>([](const Shard &shard) noexcept { return shard.append_size(kCallerHeldEpoch); });
  }

  // Whole-store aggregates of the per-shard RSS/churn counters. Live shards only: a retired
  // shard's peak drops out (monitoring use; not a precise lifetime maximum).
  [[nodiscard]] auto append_region_live_count() const -> int64_t {
    return reduce_shards<int64_t>([](const Shard &shard) noexcept { return shard.append_region_live_count(); });
  }

  // Approximate: sums each *currently live* shard's own peak, so a shard retired by a split after
  // reaching a high peak doesn't contribute it here. Acceptable for its monitoring use (see
  // T1Index::append_region_live_count()'s comment) -- not a precise store-lifetime maximum.
  [[nodiscard]] auto append_region_peak_count() const -> int64_t {
    return reduce_shards<int64_t>([](const Shard &shard) noexcept { return shard.append_region_peak_count(); });
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
    std::vector<Key> out_boundaries;
    with_checkpoint_shards(
        [&](auto &context) {
          const auto bounds = context.boundaries();
          out_boundaries.assign(bounds.begin(), bounds.end());
          std::vector<EntrySnapshot> dumped;
          for (size_t idx = 0; idx < context.shard_count(); ++idx) {
            t1_debug::g_checkpoint_current_shard.store(idx + 1, std::memory_order_release);
            context.extract_shard(idx, dumped, offset_mapper, per_shard_writer);
          }
          t1_debug::g_checkpoint_current_shard.store(0, std::memory_order_release);
        },
        /*advance_epoch=*/false);
    return out_boundaries;
  }

  struct CheckpointContext;

  // Captures a split-paused snapshot of live shards. When `advance_epoch` is true, advances
  // checkpoint_write_epoch_ and drains in-flight routing-guarded writes before invoking `func`.
  template <typename Func>
  void with_checkpoint_shards(Func &&func, bool advance_epoch = true) {
    SplitPause pause(splits_paused_);
    wait_until_no_split();

    uint64_t active_epoch = checkpoint_write_epoch_.load(std::memory_order_acquire);
    if (advance_epoch) {
      active_epoch = checkpoint_write_epoch_.fetch_add(1, std::memory_order_acq_rel);
      const uint64_t bumped_routing = epoch_domain_.epoch.fetch_add(1, std::memory_order_acq_rel) + 1;
      epoch_domain_.tracker.wait_until_epoch(bumped_routing);
    }

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

    t1_debug::g_checkpoint_total_shards.store(snapshot.size(), std::memory_order_release);
    CheckpointContext context{*this, snapshot, boundaries, active_epoch};
    func(context);
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
      auto shard = std::make_unique<Shard>(append_cap_, &epoch_domain_);
      shard->load_sorted_region_from_checkpoint(entries);
      auto *slot = new ShardSlot(std::move(shard));
      slot->approx_t2_bytes.store(sharded_detail::sum_entries_t2_bytes(std::span<const EntrySnapshot>(entries)),
                                  std::memory_order_relaxed);
      new_dir->shards.push_back(slot);
    }

    queue_.clear();
    dirty_queue_.clear();
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
    if (dirty) {
      dirty_queue_.push_if_live(dir->shards[shard_index]);
    }
  }

  [[nodiscard]] auto dirty_shard_count() const -> size_t {
    return dirty_queue_.size() + active_dirty_visits_.load(std::memory_order_acquire);
  }

  // True when any live shard has uncompacted T2 writes or dead T2 records.
  [[nodiscard]] auto has_dirty_shard() const -> bool { return dirty_shard_count() != 0; }

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
    with_routing_guard([&] {
      Directory *dir = directory_.load(std::memory_order_acquire);
      for (ShardSlot *slot : dir->shards) {
        if (should_split_shard(slot->index->approx_entry_count(kCallerHeldEpoch),
                               slot->approx_t2_bytes.load(std::memory_order_relaxed)) ||
            slot->index->append_size(kCallerHeldEpoch) > 0) {
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
    std::vector<EntrySnapshot> entries;
    const size_t max_visits = dirty_queue_.size();
    for (size_t visited = 0; visited < max_visits; ++visited) {
      ShardSlot *slot =
          with_routing_guard([&]() -> ShardSlot * { return dirty_queue_.pop_and_pin(active_dirty_visits_); });
      if (slot == nullptr) {
        break;
      }
      if (visit_dirty_shard(slot, entries, visitor, /*from_dirty_queue=*/true, nullptr) == DirtyShardVisit::kStop) {
        break;
      }
    }
  }

  template <typename Visitor>
  auto for_each_shard_for_compaction(Visitor &&visitor) -> bool {
    std::vector<ShardSlot *> snapshot;
    with_routing_guard([&] {
      Directory *dir = directory_.load(std::memory_order_acquire);
      snapshot.reserve(dir->shards.size());
      for (ShardSlot *slot : dir->shards) {
        if (slot->superseded.load(std::memory_order_acquire) == nullptr) {
          slot->outside_refs.fetch_add(1, std::memory_order_relaxed);
          snapshot.push_back(slot);
        }
      }
    });
    if (snapshot.empty()) {
      return true;
    }
    const size_t start = compaction_cursor_.load(std::memory_order_relaxed) % snapshot.size();
    if (start != 0) {
      std::rotate(snapshot.begin(), snapshot.begin() + static_cast<std::ptrdiff_t>(start), snapshot.end());
    }

    std::vector<EntrySnapshot> entries;
    size_t next = 0;
    bool all_visited = true;
    try {
      for (; next < snapshot.size(); ++next) {
        if (visit_dirty_shard(snapshot[next], entries, visitor, /*from_dirty_queue=*/false, &all_visited) ==
            DirtyShardVisit::kStop) {
          if (next + 1 < snapshot.size()) {
            all_visited = false;
          }
          ++next;
          break;
        }
      }
    } catch (...) {
      compaction_cursor_.store((start + next + 1) % snapshot.size(), std::memory_order_relaxed);
      // The throwing visit already released its own pin; only later slots still hold theirs.
      release_snapshot_tail(snapshot, next + 1);
      throw;
    }
    compaction_cursor_.store((start + next) % snapshot.size(), std::memory_order_relaxed);
    // The unvisited tail holds snapshot pins until released here.
    release_snapshot_tail(snapshot, next);
    return all_visited;
  }

 private:
  struct ShardSlot;

  // Published once, atomically, on Closing -> Split. Keys < boundary route to `low`.
  struct Split {
    Key boundary{};
    ShardSlot *low = nullptr;
    ShardSlot *high = nullptr;
  };

  // Split pause is a refcount, not a flag: checkpoint_all_shards() pauses splits
  // for its duration and resumes when the last guard exits.
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
    // Approximate live T2 bytes referenced by this shard (sum of payload size hints).
    std::atomic<uint64_t> approx_t2_bytes{0};
    // True while this shard is queued for T2 compaction.
    std::atomic<bool> t2_dirty{false};
    std::atomic<bool> split_paused{false};
    // Checkpoint epoch tracking: foreground puts stamp checkpoint_write_epoch_; checkpoint
    // advances persisted_epoch after durable commit when no post-target entries were skipped.
    std::atomic<uint64_t> dirty_epoch{0};
    uint64_t persisted_epoch{0};
    // Hazard refs for unguarded worker/checkpoint touches; continue_split() waits for zero.
    std::atomic<int> outside_refs{0};
  };

 public:
  struct CheckpointContext {
    ShardedT1Index &owner;
    std::span<ShardSlot *const> shards;
    std::span<const Key> dir_boundaries;
    uint64_t snapshot_epoch;

    [[nodiscard]] auto shard_count() const noexcept -> size_t { return shards.size(); }
    [[nodiscard]] auto boundaries() const noexcept -> std::span<const Key> { return dir_boundaries; }
    [[nodiscard]] auto shard_is_dirty(size_t index) const noexcept -> bool {
      return shards[index]->dirty_epoch.load(std::memory_order_acquire) > shards[index]->persisted_epoch;
    }
    template <typename OffsetMapper, typename OnEntries>
    void extract_shard(size_t index,
                       std::vector<EntrySnapshot> &dumped,
                       OffsetMapper &&offset_mapper,
                       OnEntries &&on_entries) const {
      owner.extract_shard_for_checkpoint(shards[index],
                                         index + 1,
                                         shards.size(),
                                         dumped,
                                         std::forward<OffsetMapper>(offset_mapper),
                                         std::forward<OnEntries>(on_entries));
    }
    void mark_persisted(size_t index) const noexcept { shards[index]->persisted_epoch = snapshot_epoch; }
  };

 private:
  template <typename OffsetMapper, typename PerShardWriter>
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  void extract_shard_for_checkpoint(ShardSlot *slot,
                                    size_t shard_idx,
                                    size_t total_shards,
                                    std::vector<EntrySnapshot> &dumped,
                                    OffsetMapper &&offset_mapper,
                                    PerShardWriter &&per_shard_writer) {
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
      slot->approx_t2_bytes.store(sharded_detail::sum_entries_t2_bytes(std::span<const EntrySnapshot>(dumped)),
                                  std::memory_order_relaxed);
      per_shard_writer(dumped);
    } else {
      reorganize_until_captured(
          *slot->index,
          offset_mapper,
          [&](std::span<const EntrySnapshot> merged) {
            slot->approx_t2_bytes.store(sharded_detail::sum_entries_t2_bytes(merged), std::memory_order_relaxed);
            per_shard_writer(merged);
          },
          /*parallel_sort=*/true);
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
    if (should_split_shard(slot->index->approx_entry_count(), slot->approx_t2_bytes.load(std::memory_order_relaxed))) {
      queue_.push_if_live(slot);
    }
  }

  // Outcome of one dirty-shard visit: keep scanning shards or stop at this one.
  enum class DirtyShardVisit : std::uint8_t { kContinue, kStop };

  // Captures one shard and releases its snapshot pin on every return path.
  template <typename Visitor>
  auto visit_dirty_shard(ShardSlot *slot,
                         std::vector<EntrySnapshot> &entries,
                         Visitor &&visitor,
                         bool from_dirty_queue = true,
                         bool *out_all_visited = nullptr) -> DirtyShardVisit {
    if (!from_dirty_queue) {
      slot->split_paused.store(true, std::memory_order_seq_cst);
    }
    auto finish_visit = [&](bool check_split) {
      slot->split_paused.store(false, std::memory_order_release);
      if (check_split) {
        const uint64_t merged_t2_bytes = sharded_detail::sum_entries_t2_bytes(std::span<const EntrySnapshot>(entries));
        slot->approx_t2_bytes.store(merged_t2_bytes, std::memory_order_relaxed);
        if (should_split_shard(entries.size(), merged_t2_bytes)) {
          queue_.push_if_live(slot);
        }
      }
      slot->outside_refs.fetch_sub(1, std::memory_order_release);
      if (from_dirty_queue) {
        active_dirty_visits_.fetch_sub(1, std::memory_order_release);
      }
    };
    if (!from_dirty_queue && slot->superseded.load(std::memory_order_seq_cst) != nullptr) {
      if (out_all_visited != nullptr) {
        *out_all_visited = false;
      }
      finish_visit(false);
      return DirtyShardVisit::kContinue;
    }
    auto restore = [&](bool check_split) {
      if (from_dirty_queue) {
        dirty_queue_.push_if_live(slot);
      }
      finish_visit(check_split);
    };
    try {
      if (slot->index->append_size() == 0 && !slot->index->has_immutable()) {
        slot->index->dump_sorted_region(entries);
      } else {
        reorganize_until_captured(
            *slot->index,
            [](std::span<EntrySnapshot> /*merged*/) {},
            [&](std::span<const EntrySnapshot> merged) { entries.assign(merged.begin(), merged.end()); },
            /*parallel_sort=*/false);
        slot->tombstones_since_maintenance.store(0, std::memory_order_relaxed);
      }
      if (!visitor(std::span<const EntrySnapshot>(entries))) {
        finish_visit(true);
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

  // True when a shard has reached either the entry-count split threshold or one T2 segment worth
  // of referenced payload bytes. A shard needs at least 2 entries to split.
  [[nodiscard]] auto should_split_shard(size_t entry_count, uint64_t t2_bytes) const noexcept -> bool {
    if (entry_count < 2) {
      return false;
    }
    return entry_count >= split_threshold(target_shard_size_, Config::kShardSplitThresholdPercent) ||
           t2_bytes >= Config::kSegmentBytes;
  }

  // Queue ownership shared by maintenance and dirty-shard compaction queues.
  template <std::atomic<bool> ShardSlot::*PendingFlag>
  struct ShardQueue {
    mutable std::mutex mutex;
    std::deque<ShardSlot *> queue;
    // Enqueues slot if live: CASes pending false->true, then under lock re-checks superseded
    // (same mutex continue_split() purges under, making push vs purge race-free).
    void push_if_live(ShardSlot *slot) {
      if ((slot->*PendingFlag).load(std::memory_order_relaxed)) {
        return;
      }
      bool expected = false;
      if (!(slot->*PendingFlag).compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
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
      (slot->*PendingFlag).store(false, std::memory_order_release);
      return slot;
    }
    // Pops one non-splitting shard, pausing splits on it before clearing its pending flag so
    // continue_split() never misses a dirty flag on a concurrently splitting shard.
    auto pop_and_pin(std::atomic<size_t> &active_visits) -> ShardSlot * {
      const std::lock_guard<std::mutex> lock(mutex);
      while (!queue.empty()) {
        ShardSlot *slot = queue.front();
        queue.pop_front();
        slot->split_paused.store(true, std::memory_order_seq_cst);
        if (slot->superseded.load(std::memory_order_seq_cst) != nullptr) {
          slot->split_paused.store(false, std::memory_order_release);
          continue;
        }
        (slot->*PendingFlag).store(false, std::memory_order_release);
        slot->outside_refs.fetch_add(1, std::memory_order_relaxed);
        active_visits.fetch_add(1, std::memory_order_acq_rel);
        return slot;
      }
      return nullptr;
    }
    void purge(ShardSlot *target) {
      const std::lock_guard<std::mutex> lock(mutex);
      queue.erase(std::remove(queue.begin(), queue.end(), target), queue.end());
    }
    void clear() {
      const std::lock_guard<std::mutex> lock(mutex);
      queue.clear();
    }
    [[nodiscard]] auto empty() const -> bool {
      const std::lock_guard<std::mutex> lock(mutex);
      return queue.empty();
    }
    [[nodiscard]] auto size() const -> size_t {
      const std::lock_guard<std::mutex> lock(mutex);
      return queue.size();
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
    return slot->index->scan(lo_bytes, hi_bytes, callback, kCallerHeldEpoch);
  }

  template <typename Callback>
  auto scan_multi_shard(Directory *dir,
                        const Key &lower,
                        const Key &upper,
                        std::span<const std::byte> lo_bytes,
                        std::span<const std::byte> hi_bytes,
                        Callback callback,
                        bool &stopped) const -> size_t {
    size_t total = 0;
    ShardSetView{dir}.for_each_leaf_in_range(lower, upper, [&](ShardSlot *leaf) -> bool {
      total += leaf->index->scan(lo_bytes, hi_bytes, callback, kCallerHeldEpoch);
      return !stopped;
    });
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
        if (!collect_into(dir->shards[i], lower, upper, visit)) {
          break;
        }
      }
    }
    static auto collect_into(ShardSlot *slot, const Key &lower, const Key &upper, auto &&visit) -> bool {
      Split *split = slot->superseded.load(std::memory_order_acquire);
      if (split == nullptr || split == kClosingSentinel) {
        return visit(slot);  // Closing reads through the splitting slot directly.
      }
      if (lower < split->boundary && !collect_into(split->low, lower, upper, visit)) {
        return false;
      }
      if (split->boundary < upper && !collect_into(split->high, lower, upper, visit)) {
        return false;
      }
      return true;
    }
  };

  template <typename Func>
  auto with_routing_guard(Func &&func) const noexcept(noexcept(func())) -> decltype(func()) {
    assert(!epoch_domain_.tracker.is_held_by_current_thread());
    typename ThreadReferenceTracker<uint64_t>::Guard handle(epoch_domain_.tracker,
                                                            epoch_domain_.epoch.load(std::memory_order_relaxed));
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
    const bool was_dirty = target->t2_dirty.load(std::memory_order_acquire);
    queue_.purge(target);
    dirty_queue_.purge(target);

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
    if (merged_entries.size() < kMinSplitEntries || !(merged_entries.front().key < merged_entries.back().key)) {
      // Too small (callback fired: real state, not a lost race) -- resume normal operation.
      // Reset superseded *before* maintenance_pending: request_maintenance_if_needed() CASes the
      // flag then checks superseded, so the reverse order could strand the flag true forever on
      // a shard about to read normal again. Clearing the flag at all is still needed: the purged
      // (never dequeued) entry's flag would otherwise never clear.
      target->superseded.store(nullptr, std::memory_order_release);
      target->maintenance_pending.store(false, std::memory_order_relaxed);
      target->t2_dirty.store(false, std::memory_order_relaxed);
      if (was_dirty) {
        dirty_queue_.push_if_live(target);
      }
      target->tombstones_since_maintenance.store(0, std::memory_order_relaxed);
      return;
    }

    size_t mid = merged_entries.size() / 2;
    if (!(merged_entries[mid - 1].key < merged_entries[mid].key)) {
      size_t right = mid + 1;
      while (right < merged_entries.size() && !(merged_entries[right - 1].key < merged_entries[right].key)) {
        ++right;
      }
      size_t left = mid - 1;
      while (left > 0 && !(merged_entries[left - 1].key < merged_entries[left].key)) {
        --left;
      }
      if (right < merged_entries.size() && (left == 0 || (right - mid) <= (mid - left))) {
        mid = right;
      } else {
        mid = left;
      }
    }
    const Key boundary = merged_entries[mid].key;
    const std::span<const EntrySnapshot> low_entries(merged_entries.data(), mid);
    const std::span<const EntrySnapshot> high_entries(merged_entries.data() + mid, merged_entries.size() - mid);
    const uint64_t low_t2_bytes = sharded_detail::sum_entries_t2_bytes(low_entries);
    const uint64_t high_t2_bytes = sharded_detail::sum_entries_t2_bytes(high_entries);
    auto low_shard = std::make_unique<Shard>(target->index->append_capacity(), &epoch_domain_);
    auto high_shard = std::make_unique<Shard>(target->index->append_capacity(), &epoch_domain_);
    low_shard->load_sorted_region_from_checkpoint(low_entries);
    high_shard->load_sorted_region_from_checkpoint(high_entries);

    auto *low_slot = new ShardSlot(std::move(low_shard));
    auto *high_slot = new ShardSlot(std::move(high_shard));
    const uint64_t split_dirty_epoch = checkpoint_write_epoch_.load(std::memory_order_acquire);
    low_slot->dirty_epoch.store(split_dirty_epoch, std::memory_order_relaxed);
    high_slot->dirty_epoch.store(split_dirty_epoch, std::memory_order_relaxed);
    low_slot->approx_t2_bytes.store(low_t2_bytes, std::memory_order_relaxed);
    high_slot->approx_t2_bytes.store(high_t2_bytes, std::memory_order_relaxed);
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

    // Drain routing-guarded calls still referencing `target`/`old_dir` before sampling final
    // `t2_dirty` state, queuing children, and retiring `target`. Any writer that observed
    // `superseded == nullptr` before Closing may still be executing `note_put_applied(target)`
    // inside its routing guard; waiting here guarantees its dirty flag is never missed.
    // Waiting after Split publish avoids deadlocking writers that spin outside routing guards.
    const uint64_t bumped = epoch_domain_.epoch.fetch_add(1, std::memory_order_acq_rel) + 1;
    epoch_domain_.tracker.wait_until_epoch(bumped);

    const bool dirty = was_dirty || target->t2_dirty.load(std::memory_order_acquire);
    if (dirty) {
      dirty_queue_.push_if_live(low_slot);
      dirty_queue_.push_if_live(high_slot);
    }
    if (should_split_shard(mid, low_t2_bytes) && merged_entries.front().key < merged_entries[mid - 1].key) {
      queue_.push_if_live(low_slot);
    }
    if (should_split_shard(merged_entries.size() - mid, high_t2_bytes) &&
        merged_entries[mid].key < merged_entries.back().key) {
      queue_.push_if_live(high_slot);
    }
    low_slot->outside_refs.fetch_sub(1, std::memory_order_release);
    high_slot->outside_refs.fetch_sub(1, std::memory_order_release);

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

  void mark_checkpoint_dirty(ShardSlot *slot) noexcept {
    const uint64_t cur = checkpoint_write_epoch_.load(std::memory_order_acquire);
    if (slot->dirty_epoch.load(std::memory_order_relaxed) != cur) {
      slot->dirty_epoch.store(cur, std::memory_order_release);
    }
  }

  void note_put_applied(ShardSlot *slot, Payload value, bool is_inline, bool mark_t2_dirty) {
    mark_checkpoint_dirty(slot);
    if (mark_t2_dirty && value != STORE_NOT_FOUND && !is_inline) {
      dirty_queue_.push_if_live(slot);
      const uint64_t rec_bytes = sharded_detail::payload_t2_bytes(value, is_inline);
      if (rec_bytes > 0) {
        const uint64_t prev_bytes = slot->approx_t2_bytes.fetch_add(rec_bytes, std::memory_order_relaxed);
        if (prev_bytes + rec_bytes >= Config::kSegmentBytes) {
          request_maintenance_if_needed(slot);
        }
      }
    }
    if (value == STORE_NOT_FOUND) {
      slot->tombstones_since_maintenance.fetch_add(1, std::memory_order_relaxed);
      // Re-check post-increment: the final delete of a quiet workload must still trip the
      // trigger without another write after it.
      request_maintenance_if_needed(slot);
    }
  }

  // Enqueues past the soft threshold or a shard's worth of tombstones -- deletes
  // only retarget T1 offsets, so delete-heavy workloads need the tombstone half to ever fire.
  // Push-vs-purge race-freedom comes from push_if_live()'s mutex-serialized superseded
  // re-check: ordered-before-purge gets cleaned; ordered-after sees set superseded and backs
  // out. No post-return queue entry for a splitting shard, ever.
  void request_maintenance_if_needed(ShardSlot *slot) {
    const size_t size = slot->index->append_size(kCallerHeldEpoch);
    const size_t cap = slot->index->append_capacity();
    const bool over_soft = size >= maintenance_soft_threshold(cap);
    const bool delete_heavy = slot->tombstones_since_maintenance.load(std::memory_order_relaxed) >= target_shard_size_;
    const bool over_t2_bytes =
        size > 0 && slot->approx_t2_bytes.load(std::memory_order_relaxed) >= Config::kSegmentBytes;
    if (!over_soft && !delete_heavy && !over_t2_bytes) {
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
  [[nodiscard]] auto try_claim_split(ShardSlot *slot,
                                     std::span<const EntrySnapshot> merged_entries,
                                     uint64_t merged_t2_bytes) -> bool {
    if (!should_split_shard(merged_entries.size(), merged_t2_bytes) ||
        !(merged_entries.front().key < merged_entries.back().key)) {
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
    if (pause_active && slot->index->append_size() < maintenance_soft_threshold(slot->index->append_capacity())) {
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
    uint64_t merged_t2_bytes = 0;
    if (reorganized) {
      merged_t2_bytes = sharded_detail::sum_entries_t2_bytes(merged_entries);
      slot->approx_t2_bytes.store(merged_t2_bytes, std::memory_order_relaxed);
      slot->tombstones_since_maintenance.store(0, std::memory_order_relaxed);
    }

    const bool won = try_claim_split(slot, merged_entries, merged_t2_bytes);
    if (!won && should_split_shard(merged_entries.size(), merged_t2_bytes) &&
        merged_entries.front().key < merged_entries.back().key &&
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

  // Declared before directory_ so epoch_domain_ outlives every Shard torn down after workers join.
  mutable EpochDomain epoch_domain_;
  std::atomic<uint64_t> total_splits_{0};
  std::atomic<uint64_t> last_split_pause_us_{0};
  std::atomic<uint64_t> last_split_pause_end_ns_{0};
  std::atomic<Directory *> directory_{nullptr};
  size_t append_cap_;
  size_t target_shard_size_;
  ShardQueue<&ShardSlot::maintenance_pending> queue_;
  ShardQueue<&ShardSlot::t2_dirty> dirty_queue_;
  std::atomic<size_t> active_dirty_visits_{0};
  std::atomic<size_t> compaction_cursor_{0};
  std::vector<std::jthread> workers_;
  // Set for checkpoint_all_shards()' duration: directory stays stable (splits initiate only
  // at the Closing CAS, which checks this); in-flight splits are unaffected.
  std::atomic<int> splits_paused_{0};
  std::atomic<size_t> active_maintenance_{0};
  std::atomic<uint64_t> checkpoint_write_epoch_{1};
};

template <typename Config>
typename ShardedT1Index<Config>::Split *const ShardedT1Index<Config>::kClosingSentinel =
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    reinterpret_cast<typename ShardedT1Index<Config>::Split *>(static_cast<uintptr_t>(1));

}  // namespace vmemkv
