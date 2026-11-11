// t1_index.hpp — Fixed-size two-region in-memory index with lock-free readers.
// Codec helpers live in t1_codec.hpp, merge helpers in
// t1_merge.hpp (see docs/specification/low_level_design.md 7.5).
#pragma once

#include <sys/mman.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <execution>
#include <memory>
#include <memory_resource>
#include <mutex>
#include <span>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
#include <vmemkv/api/utils.hpp>
#include <vmemkv/config.hpp>
#include <vmemkv/core/bytes.hpp>
#include <vmemkv/core/lock_free_hash_table.hpp>
#include <vmemkv/core/mmap.hpp>
#include <vmemkv/core/reference_tracker.hpp>
#include <vmemkv/core/spin_backoff.hpp>
#include <vmemkv/optimizations/bloom_filter.hpp>
#include <vmemkv/scan_control.hpp>
#include <vmemkv/t1_index/t1_codec.hpp>
#include <vmemkv/t1_index/t1_merge.hpp>

namespace vmemkv {

// ─── T1Index Structure Overview ─────────────────────────────────────────────
//
//               +-----------------------------+
//               |  sorted_region_ (raw ptr)   |
//               +--------------+--------------+
//                              |
//                              v
//             +----------------------------------+
//             | SortedSlot0 | SortedSlot1 | ...  |
//             +----------------------------------+
//                    O(log N) Binary Search
//                    (Cold / Read-Only Region)
//
//               +-----------------------------+
//               |   append_active_ (raw ptr)  |
//               +--------------+--------------+
//                              |
//                              v
//             +----------------------------------+
//             | AppendSlot0 | AppendSlot1 | ...  |
//             +----------------------------------+
//                    O(1) Lock-free Hash Table
//                    (Hot / Active Write Region)
//
//               +-----------------------------+
//               | append_immutable_ (raw ptr) | -- (Only during reorganize)
//               +-----------------------------+
//
// ─── Read Concurrency Protocol (EBR & Double-Buffering) ──────────────────────
// 1. Enter epoch: register current reorg_epoch_ in thread-local slot (T1ReadHandle).
// 2. Search append_active_. If found, return payload.
// 3. Search append_immutable_ (if exists/not null). If found, return payload.
// 4. Search sorted_region_. If found, return payload.
// 5. Exit epoch: clear thread-local slot registration upon handle destruction.
//
template <typename Config = vmemkv::Config<>>
class T1Index {
 public:
  using Key = t1_detail::StoreKey;
  using Payload = uint64_t;

  // RAII epoch registration for readers (lets reorganize() know these buffers are in use).
  using T1ReadHandle = typename ThreadReferenceTracker<uint64_t>::Guard;

  // `append_cap`: lock-free append-region (LSM-style MemTable) capacity, not whole-KVS capacity.
  // Defaults to Config::kAppendCapacityEntries; sharded T1 passes a per-shard override.
  explicit T1Index(size_t append_cap = Config::kAppendCapacityEntries) : append_cap_(append_cap) {
    auto *active_region = new AppendRegion(append_cap_);
    note_append_region_created();
    auto *active_index = new AppendIndex(append_cap_);
    append_active_.store(new AppendGeneration{active_region, active_index}, std::memory_order_release);
    sorted_region_.store(new SortedRegion(), std::memory_order_release);
  }

  ~T1Index() noexcept {
    if (AppendGeneration *active = append_active_.load(std::memory_order_relaxed); active != nullptr) {
      note_append_region_destroyed();
    }
    if (AppendGeneration *imm = append_immutable_.get(); imm != nullptr) {
      note_append_region_destroyed();
    }
    delete_generation(append_active_.load(std::memory_order_relaxed));
    delete_generation(append_immutable_.get());
    delete sorted_region_.load(std::memory_order_relaxed);
  }

  T1Index(const T1Index &) = delete;
  auto operator=(const T1Index &) -> T1Index & = delete;

  // raw_hash's upper bits carry inline metadata; returning it saves a re-hash/re-lookup.
  struct LookupResult {
    Payload payload_bits;
    uint64_t raw_hash;
  };

  enum class PutResult : uint8_t {
    Applied,
    AppendRegionFull,
  };

  // A live entry as reorganize() merges it. Exposed so a checkpoint writer can serialize the
  // sorted region without T1Index knowing about files.
  struct EntrySnapshot {
    Key key{};
    Payload payload_bits{STORE_NOT_FOUND};
    uint64_t hash{0};
  };

  struct NoOpChkWriter {
    void operator()(std::span<const EntrySnapshot> /*merged*/) const noexcept {}
  };

  auto get_with_hash(std::span<const std::byte> key) const -> LookupResult {
    const auto [prefix, hash] = prepare_key_and_hash(key);
    return with_epoch_guard([&]() -> LookupResult { return lookup_by_prefix_hash(prefix, hash); });
  }

 private:
  // Lookup body for get_with_hash() -- call inside with_epoch_guard(). The (hash, payload)
  // pair is read via load_slot_consistent(): independent loads could feed a torn pair onward.
  auto lookup_by_prefix_hash(const t1_detail::StoreKey &prefix, uint64_t hash) const -> LookupResult {
    const AppendGeneration *active_gen = append_active_.load(std::memory_order_acquire);
    const AppendRegion *active = active_gen->region;

    // 1. check active append region
    if (const AppendSlot *slot = active->find_with_index(*active_gen->index, prefix, hash)) {
      const auto [slot_hash, slot_payload] = load_slot_consistent(*slot);
      return LookupResult{slot_payload, slot_hash};
    }

    // 2. check immutable append region if exists
    if (const AppendGeneration *imm_gen = append_immutable_.get()) {
      const AppendRegion *imm = imm_gen->region;
      if (const AppendSlot *slot = imm->find_with_index(*imm_gen->index, prefix, hash)) {
        const auto [slot_hash, slot_payload] = load_slot_consistent(*slot);
        return LookupResult{slot_payload, slot_hash};
      }
    }

    // 3. check sorted region (loaded after immutable so a racing reorganize() that drained
    // immutable into next_sorted is visible here)
    const SortedRegion *sorted = sorted_region_.load(std::memory_order_acquire);
    if (const SortedSlot *slot = find_sorted(*sorted, prefix, hash)) {
      const auto [slot_hash, slot_payload] = load_slot_consistent(*slot);
      return LookupResult{slot_payload, slot_hash};
    }

    return LookupResult{STORE_NOT_FOUND, 0};
  }

 public:
  auto get(std::span<const std::byte> key) const -> Payload { return get_with_hash(key).payload_bits; }

  [[nodiscard]] auto append_size() const noexcept -> size_t {
    return with_epoch_guard([&]() noexcept { return append_active_.load(std::memory_order_acquire)->region->size(); });
  }

  // Approximate total entry count (sorted plus active append, including dead entries).
  // Cheap enough to poll: two atomic loads under an epoch guard, no merge. Over-approximates
  // live entries, so it gates expensive work, never correctness.
  [[nodiscard]] auto approx_entry_count() const noexcept -> size_t {
    return with_epoch_guard([&]() noexcept {
      return sorted_region_.load(std::memory_order_acquire)->size +
             append_active_.load(std::memory_order_acquire)->region->size();
    });
  }

  // Fixed for this instance's lifetime.
  [[nodiscard]] auto append_capacity() const noexcept -> size_t { return append_cap_; }

  // True while a reorganize() has frozen the previous active region and not yet
  // published its replacement: a snapshot that reads only the sorted region is
  // incomplete until the merge lands.
  [[nodiscard]] auto has_immutable() const noexcept -> bool { return append_immutable_.get() != nullptr; }

  // Streams every live sorted-region entry in key order without merging. Behind
  // checkpoint_all_shards()'s clean-shard shortcut (identical output to a merge when both
  // the active append and the frozen immutable are empty; callers check has_immutable()).
  // Epoch-guarded; tombstones skipped as a merge would skip them.
  void dump_sorted_region(std::vector<EntrySnapshot> &out) const {
    with_epoch_guard([&] {
      out.clear();
      const SortedRegion *sorted = sorted_region_.load(std::memory_order_acquire);
      Key max_key;
      max_key.fill(t1_detail::kMaxKeyFillByte);
      walk_sorted_region(sorted, Key{}, max_key, [&](const SortedSlot &slot, uint64_t hash, Payload payload) {
        out.push_back(EntrySnapshot{slot.key, payload, hash});
      });
    });
  }

  // One append slot's bytes, for cache-derived entry budgets without depending on layout.
  static auto append_slot_bytes() noexcept -> size_t;

  // Resident AppendRegion instances (active + not-yet-reclaimed retiring generations) and the
  // lifetime peak. Each carries an append_cap_ * sizeof(AppendSlot) mmap footprint that stays
  // near-fully resident from light occupancy, so this pile-up directly bounds T1's RSS churn.
  [[nodiscard]] auto append_region_live_count() const noexcept -> int64_t {
    return append_region_live_count_.load(std::memory_order_relaxed);
  }
  [[nodiscard]] auto append_region_peak_count() const noexcept -> int64_t {
    return append_region_peak_count_.load(std::memory_order_relaxed);
  }

  auto put(std::span<const std::byte> key,
           Payload value,
           bool is_inline = false,
           uint8_t inline_size = 0) -> PutResult {
    const auto [prefix, hash] = prepare_key_and_hash(key);
    const uint64_t stored_hash = is_inline ? t1_detail::embed_metadata(hash, inline_size) : hash;
    return put_with_stored_hash(prefix, hash, stored_hash, value);
  }

 private:
  // Insert body for put(), after (prefix, clean hash, stored_hash) resolution.
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  auto put_with_stored_hash(const Key &prefix, uint64_t hash, uint64_t stored_hash, Payload value) -> PutResult {
    return with_epoch_guard([&]() -> PutResult {
      // Retries: the in-place write can race a displacing concurrent insert (publish_slot()'s
      // on_displaced). same_slot_as() after writing tells real displacement (retire ours, retry)
      // from delete/reinsert; the frozen-region case below falls through to insert-new.
      // Terminates: bounded by resolve()'s immutable-region bypass window.
      while (true) {
        ResolvedSlot slot = resolve(prefix, hash);
        if (slot.found()) {
          // begin/end_write() bracket the store with the slot seqlock: otherwise a reader could
          // pair a post-try_store() payload with a pre-store_hash() hash.
          slot.begin_write();
          Payload current = slot.load();
          if (!slot.try_store(current, value)) {
            slot.end_write();
            continue;  // Stale snapshot (a benign concurrent update raced us) -- re-resolve.
          }
          ResolvedSlot after = resolve(prefix, hash);
          if (after.same_slot_as(slot)) {
            slot.store_hash(stored_hash);
            slot.end_write();
            return PutResult::Applied;
          }
          if (after.found()) {
            // A different slot answers now: genuine hash-collision displacement (on_displaced).
            // Ours is an orphaned duplicate (index can't find it; walks still can) -- best-effort
            // retire before retrying.
            Payload ours = value;
            std::ignore = slot.try_store(ours, STORE_NOT_FOUND);
            slot.end_write();
            continue;
          }
          // Nothing found: only a concurrent reorganize() freezing our region explains it (sorted
          // slots can't be displaced; immutable hits never report found). Our slot still holds
          // `value` correctly -- leave it, fall through to insert-new (a safety-net copy in the
          // fresh active region, covering a merge that already read our old value).
          slot.end_write();
          continue;
        }

        AppendGeneration *active_gen = append_active_.load(std::memory_order_acquire);
        AppendRegion *active = active_gen->region;
        const size_t index = active->reserve();
        if (index >= append_cap_) {
          return PutResult::AppendRegionFull;
        }

        active->publish(index, prefix, stored_hash, value);
        publish_append_index(*active_gen->index, *active, index, prefix, hash);
        return PutResult::Applied;
      }
    });
  }

 public:
  // Range scan over [lo_bytes, hi_bytes): consistent snapshot of both regions, deduped.
  // `Callback`: `(key, payload, hash) -> void | ScanControl`.
  template <typename Callback>
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters,readability-function-cognitive-complexity)
  auto scan(std::span<const std::byte> lo_bytes,
            std::span<const std::byte> hi_bytes,
            Callback callback) const -> size_t {
    const t1_detail::StoreKey lower_bound = t1_detail::prefix_from_bytes(lo_bytes);
    const t1_detail::StoreKey upper_bound = t1_detail::prefix_from_bytes(hi_bytes);
    // NOLINTNEXTLINE(readability-function-cognitive-complexity)
    struct ScanCandidate {
      Key key;
      Payload payload_bits;
      uint64_t hash;
      int gen;  // 0: sorted, 1: imm, 2: active (newest)
    };

    // Use a 16KB stack buffer (~500 candidates): this function never heap-allocates.
    constexpr std::size_t kStackBufferBytes = 16384;
    std::array<std::byte, kStackBufferBytes> stack_buf;
    std::pmr::monotonic_buffer_resource mem_res(stack_buf.data(), stack_buf.size());
    std::pmr::vector<ScanCandidate> candidates(&mem_res);

    // Fast path: when active and immutable append regions are disjoint from the range,
    // stream sorted entries directly to callback without candidates buffering, sorting, or dedup.
    const auto fast_res = with_epoch_guard([&]() -> std::optional<size_t> {
      const SortedRegion *sorted = sorted_region_.load(std::memory_order_acquire);
      const AppendRegion *active = append_active_.load(std::memory_order_acquire)->region;
      const AppendGeneration *imm_gen = append_immutable_.get();
      const AppendRegion *imm = imm_gen != nullptr ? imm_gen->region : nullptr;

      auto region_disjoint = [&](const AppendRegion *region) {
        if (region == nullptr || region->size() == 0) {
          return true;
        }
        Key region_min{};
        Key region_max{};
        if (!region->bounds(region_min, region_max)) {
          return true;
        }
        return t1_detail::range_disjoint(region_min, region_max, lower_bound, upper_bound);
      };

      if (region_disjoint(active) && region_disjoint(imm)) {
        size_t match_count = 0;
        walk_sorted_region(
            sorted, lower_bound, upper_bound, [&](const SortedSlot &slot, uint64_t hash, Payload payload) {
              ++match_count;
              return scan_detail::invoke_callback(callback, std::span<const std::byte>(slot.key), payload, hash);
            });
        return match_count;
      }
      return std::nullopt;
    });

    if (fast_res.has_value()) {
      return *fast_res;
    }

    // Step 1: Extract candidate entries under epoch guard (fast, microseconds, in-memory index only).
    // NOLINTNEXTLINE(readability-function-cognitive-complexity)
    with_epoch_guard([&] {
      const SortedRegion *sorted = sorted_region_.load(std::memory_order_acquire);
      const AppendRegion *active = append_active_.load(std::memory_order_acquire)->region;
      const AppendGeneration *imm_gen = append_immutable_.get();
      const AppendRegion *imm = imm_gen != nullptr ? imm_gen->region : nullptr;

      // Extract from sorted_region using binary search (O(log S))
      walk_sorted_region(sorted, lower_bound, upper_bound, [&](const SortedSlot &slot, uint64_t hash, Payload payload) {
        candidates.push_back({slot.key, payload, hash, 0});
      });

      // Extract from append active/immutable regions
      auto extract_append = [&](const AppendRegion *region, int gen) {
        if (!region) {
          return;
        }
        size_t count = region->size();
        if (count == 0) {
          return;
        }
        // Skip this region's O(count) scan when the range can't intersect its bounds.
        {
          Key region_min{};
          Key region_max{};
          if (region->bounds(region_min, region_max)) {
            if (t1_detail::range_disjoint(region_min, region_max, lower_bound, upper_bound)) {
              return;
            }
          }
        }
        for (size_t i = 0; i < count; ++i) {
          const AppendSlot &slot = region->data()[i];
          if (!slot.published.load(std::memory_order_acquire)) {
            continue;
          }
          // Paired read; see load_slot_consistent().
          const auto [slot_hash, value] = load_slot_consistent(slot);
          if (value == vmemkv::STORE_NOT_FOUND) {
            continue;
          }
          if (!(slot.key < lower_bound) && (slot.key < upper_bound)) {
            candidates.push_back({slot.key, value, slot_hash, gen});
          }
        }
      };

      const size_t sorted_count = candidates.size();

      extract_append(active, 2);
      extract_append(imm, 1);

      if (candidates.size() > sorted_count) {
        // Sort candidates by Key ASC, then Generation DESC (newest first).
        auto candidate_less = [](const ScanCandidate &lhs, const ScanCandidate &rhs) {
          if (lhs.key != rhs.key) {
            return lhs.key < rhs.key;
          }
          return lhs.gen > rhs.gen;
        };
        std::sort(candidates.begin() + static_cast<std::ptrdiff_t>(sorted_count), candidates.end(), candidate_less);

        std::pmr::vector<ScanCandidate> merged(&mem_res);
        merged.reserve(candidates.size());
        size_t sorted_idx = 0;
        size_t append_idx = sorted_count;
        while (sorted_idx < sorted_count && append_idx < candidates.size()) {
          if (candidate_less(candidates[append_idx], candidates[sorted_idx])) {
            merged.push_back(candidates[append_idx++]);
          } else {
            merged.push_back(candidates[sorted_idx++]);
          }
        }
        while (sorted_idx < sorted_count) {
          merged.push_back(candidates[sorted_idx++]);
        }
        while (append_idx < candidates.size()) {
          merged.push_back(candidates[append_idx++]);
        }
        candidates.swap(merged);
      }
    });

    if (candidates.empty()) {
      return 0;
    }

    // Step 2: Deduplicate (keep only newest generation) and run callbacks OUTSIDE epoch guard.
    // Callbacks may trigger page faults / swap I/O in LTM; releasing the epoch guard first prevents
    // blocking concurrent maintenance / checkpoint epoch waits.
    size_t match_count = 0;
    bool first = true;
    Key prev_key{};

    for (const auto &cand : candidates) {
      if (first || cand.key != prev_key) {
        first = false;
        prev_key = cand.key;
        if (cand.payload_bits != vmemkv::STORE_NOT_FOUND) {
          const ScanControl control = scan_detail::invoke_callback(
              callback, std::span<const std::byte>(cand.key), cand.payload_bits, cand.hash);
          ++match_count;
          if (control == ScanControl::Stop) {
            break;
          }
        }
      }
    }

    return match_count;
  }

  // Merges append_active_ into sorted_region_; lock-free readers may run concurrently.
  // - OffsetMapper: `void(span<EntrySnapshot> merged)` once, with the full deduped key-ordered
  //   live set. May rewrite `.payload_bits` in place (e.g. relocate a T2 offset), in any order --
  //   but must NOT reorder or resize `merged` (chk_writer and the SortedRegion built next both
  //   require key order). Unpublished to other threads at this point, so in-place mutation is
  //   safe. Entries it needn't touch (inline, T1-only reorg) must stay untouched.
  // - ChkWriter: once with the finalized entries, for checkpoint serialization without T1Index
  //   knowing about files. No-op default.
  // `parallel_sort`: true matches single-T1 behavior (one reorganize in flight: fan out). A
  // caller running many concurrent reorganize() calls (sharded background pool, one per shard)
  // passes false: N parallel sorts oversubscribe cores (pool size * fan-out).
  template <typename OffsetMapper, typename ChkWriter = NoOpChkWriter>
  void reorganize(OffsetMapper offset_mapper, ChkWriter chk_writer = ChkWriter{}, bool parallel_sort = true) {
    bool expected = false;
    if (!reorg_in_progress_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
      return;
    }

    // 1. Prepare new active buffers
    auto *next_active_region = new AppendRegion(append_cap_);
    note_append_region_created();
    auto *next_active_index = new AppendIndex(append_cap_);
    auto *next_active_gen = new AppendGeneration{next_active_region, next_active_index};

    // 2. Freeze current active to immutable, publish a fresh active. Single atomic stores, so
    // a reader never sees a region paired with the wrong generation's index.
    AppendGeneration *old_active_gen = append_active_.load(std::memory_order_acquire);
    append_immutable_.freeze(old_active_gen);
    append_active_.store(next_active_gen, std::memory_order_release);

    // Advance epoch and wait out writers on the old region. After this: no new reserve() on
    // it, and every reserved slot is fully published -- straggler and post-freeze races closed.
    uint64_t freeze_epoch = reorg_epoch_.fetch_add(1, std::memory_order_acq_rel) + 1;
    active_epochs_.wait_until_epoch(freeze_epoch);

    // 3. Rebuild sorted region from sorted_region and append_immutable
    const auto sorted = sorted_region_.load(std::memory_order_acquire);
    // Freeze before the merge reads a slot: until the replacement publishes, `sorted` is frozen
    // for writers (an in-place write here would be a silent Lost Update -- see FreezableRegion).
    sorted_write_frozen_.freeze(sorted);
    AppendRegion *old_active = old_active_gen->region;
    const size_t imm_n = old_active->size();

    // Extract append immutable entries (unsorted)
    std::vector<EntrySnapshot> imm_entries;
    imm_entries.reserve(imm_n);
    old_active->collect_live_entries(imm_entries, imm_n);

    // Ordered by *clean* hash, not raw: raw embeds transient inline metadata that shifts
    // independently of identity, which dedup relies on not happening.
    auto comp = [](const EntrySnapshot &lhs, const EntrySnapshot &rhs) {
      return t1_detail::entry_less(lhs.key, lhs.hash, rhs.key, rhs.hash);
    };

    // Sort the smaller append-immutable side (parallel unless the caller opts out, see above).
    if (parallel_sort) {
      std::sort(std::execution::par, imm_entries.begin(), imm_entries.end(), comp);
    } else {
      std::sort(imm_entries.begin(), imm_entries.end(), comp);
    }

    // Merge sorted_region with imm_entries in one pass (no redundant O(sorted) copy).
    // Hand-written loop, not std::merge: the sorted side needs live-filtering interleaved.
    std::vector<EntrySnapshot> merged;
    merged.reserve(sorted->size + imm_entries.size());
    {
      size_t sorted_idx = 0;
      size_t imm_idx = 0;
      auto skip_dead = [&]() {
        while (sorted_idx < sorted->size &&
               !is_live(sorted->slots[sorted_idx].payload_bits.load(std::memory_order_relaxed))) {
          ++sorted_idx;
        }
      };
      skip_dead();
      // Consistent (hash, payload) pair; a torn pair could misread inline as a T2 offset.
      while (sorted_idx < sorted->size && imm_idx < imm_entries.size()) {
        const SortedSlot &sorted_slot = sorted->slots[sorted_idx];
        const EntrySnapshot &imm_entry = imm_entries[imm_idx];
        const auto [s_hash, s_payload] = load_slot_consistent(sorted_slot);
        if (t1_detail::entry_identity_equal(sorted_slot.key, s_hash, imm_entry.key, imm_entry.hash)) {
          // Same entry both sides (updated since last reorg) -- imm (newer) wins. Clean-hash
          // match: an inline<->non-inline transition changes raw hash without changing identity,
          // and a raw match would leave a permanent sorted-region duplicate.
          merged.push_back(imm_entry);
          ++sorted_idx;
          ++imm_idx;
          skip_dead();
        } else if (t1_detail::entry_less(sorted_slot.key, s_hash, imm_entry.key, imm_entry.hash)) {
          merged.push_back(EntrySnapshot{sorted_slot.key, s_payload, s_hash});
          ++sorted_idx;
          skip_dead();
        } else {
          merged.push_back(imm_entry);
          ++imm_idx;
        }
      }
      while (sorted_idx < sorted->size) {
        const SortedSlot &sorted_slot = sorted->slots[sorted_idx];
        const auto [s_hash, s_payload] = load_slot_consistent(sorted_slot);
        merged.push_back(EntrySnapshot{sorted_slot.key, s_payload, s_hash});
        ++sorted_idx;
        skip_dead();
      }
      while (imm_idx < imm_entries.size()) {
        merged.push_back(imm_entries[imm_idx]);
        ++imm_idx;
      }
    }

    assert_no_duplicates(merged);

    offset_mapper(std::span<EntrySnapshot>(merged));

    auto *next_sorted = new SortedRegion(merged);

    // Unfreeze right after publish: resolve() sees the complete unfrozen region; bypass over.
    sorted_region_.store(next_sorted, std::memory_order_release);
    sorted_write_frozen_.unfreeze();
    append_immutable_.unfreeze();

    chk_writer(std::span<const EntrySnapshot>(merged));

    // 4. Retire old buffers: advance epoch, wait out readers, then delete.
    uint64_t target_epoch = reorg_epoch_.fetch_add(1, std::memory_order_acq_rel) + 1;
    active_epochs_.wait_until_epoch(target_epoch);

    delete old_active_gen->region;
    note_append_region_destroyed();
    delete old_active_gen->index;
    delete old_active_gen;
    delete sorted;

    reorg_in_progress_.store(false, std::memory_order_release);
  }

  // Installs a pre-built sorted region (from a T1 checkpoint file) as the entire live state.
  // No synchronization or retirement -- construction-time only. `entries` must be key-ascending.
  void load_sorted_region_from_checkpoint(std::span<const EntrySnapshot> entries) {
    // Build before teardown so a throwing constructor leaves the original pointer intact.
    auto *next_sorted = new SortedRegion(entries);
    delete sorted_region_.load(std::memory_order_relaxed);
    sorted_region_.store(next_sorted, std::memory_order_relaxed);
  }

 private:
  // ─── Private Type Definitions ──────────────────────────────────────────

  // Guards a region reorganize() replaces wholesale with a snapshot (immutable append region;
  // sorted region while the merge loop reads it). Hazard: between snapshot and publish, an
  // in-place write on the pre-snapshot region succeeds but never reaches the replacement -- a
  // silent Lost Update. While frozen, resolve() treats write-path hits as not-found, forcing
  // put() into the live append region. Any future snapshot-then-swap region must route its
  // write lookup through this type. Reads are unaffected: frozen bytes stay the live view
  // until the replacement publishes.
  template <typename RegionPtr>
  class FreezableRegion {
   public:
    void freeze(RegionPtr region) noexcept { frozen_.store(region, std::memory_order_release); }
    void unfreeze() noexcept { frozen_.store(nullptr, std::memory_order_release); }

    // Caller runs its own region-specific find() on the result; this type owns only the
    // freeze/unfreeze lifecycle (append and sorted finds have unrelated shapes).
    [[nodiscard]] auto get() const noexcept -> RegionPtr { return frozen_.load(std::memory_order_acquire); }

   private:
    std::atomic<RegionPtr> frozen_{nullptr};
  };

  // Shared slot body: key plus seqlock-guarded (hash, payload_bits). Atomics: put()'s in-place
  // path mutates a live published slot concurrently with readers and the merge loop (a plain
  // field would be a data race). mutable: writable through const-slot handles. Version starts
  // even; only the in-place path brackets it odd->even.
  struct SlotBase {
    Key key{};
    mutable std::atomic<uint64_t> hash{0};
    mutable std::atomic<Payload> payload_bits{STORE_NOT_FOUND};
    mutable std::atomic<uint64_t> version{0};

    auto clean_hash() const noexcept -> uint64_t {
      return hash.load(std::memory_order_acquire) & t1_detail::kCleanHashMask;
    }
  };

  struct SortedSlot : SlotBase {};

  struct AppendSlot : SlotBase {
    std::atomic<bool> published{false};
    // Version unneeded for the initial publish() (published's release already makes
    // hash/payload visible together); only for later in-place updates.
  };

  struct SortedRegion {
    size_t size = 0;
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    std::unique_ptr<SortedSlot[]> slots;
    [[no_unique_address]] std::conditional_t<Config::UseBloomFilter, t1_detail::BloomFilter, EmptyOption> bloom;

    SortedRegion() = default;
    explicit SortedRegion(std::span<const EntrySnapshot> entries)
        // NOLINTNEXTLINE(modernize-avoid-c-arrays)
        : size(entries.size()), slots(size == 0 ? nullptr : std::make_unique<SortedSlot[]>(size)) {
      if constexpr (Config::UseBloomFilter) {
        bloom.reset(entries.size());
      }
      for (size_t i = 0; i < size; ++i) {
        slots[i].key = entries[i].key;
        slots[i].hash.store(entries[i].hash, std::memory_order_relaxed);
        slots[i].payload_bits.store(entries[i].payload_bits, std::memory_order_relaxed);
        if constexpr (Config::UseBloomFilter) {
          bloom.add(entries[i].hash & t1_detail::kCleanHashMask);
        }
      }
    }
  };

  // Walks the sorted region within [lower_bound, upper_bound), calling sink(slot, hash,
  // payload) per live match. One implementation for scan()'s fast and slow paths; the sink
  // decides per match. Templated (not std::function) so sinks inline.
  template <typename Sink>
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  static auto walk_sorted_region(const SortedRegion *sorted, Key lower_bound, Key upper_bound, Sink &&sink)
      -> ScanControl {
    if (sorted == nullptr || sorted->size == 0) {
      return ScanControl::Continue;
    }
    const SortedSlot *slots_begin = sorted->slots.get();
    const SortedSlot *slots_end = slots_begin + sorted->size;
    const SortedSlot *it_start = std::lower_bound(
        slots_begin, slots_end, lower_bound, [](const SortedSlot &slot, const t1_detail::StoreKey &bound) {
          return slot.key < bound;
        });
    for (const SortedSlot *it = it_start; it < slots_end; ++it) {
      if (!(it->key < upper_bound)) {
        break;
      }
      const auto [hash, payload] = load_slot_consistent(*it);
      if (payload != vmemkv::STORE_NOT_FOUND) {
        if (scan_detail::invoke_callback(sink, *it, hash, payload) == ScanControl::Stop) {
          return ScanControl::Stop;
        }
      }
    }
    return ScanControl::Continue;
  }

  using AppendIndex = LockFreeHashTable<Key, AppendSlot>;

  struct ResolvedSlot {
    AppendSlot *append = nullptr;
    const SortedSlot *sorted = nullptr;

    [[nodiscard]] auto found() const noexcept -> bool { return append != nullptr || sorted != nullptr; }

    // Which physical slot, not its contents: lets put()'s retry tell an ordinary delete
    // (same slot) from a concurrent displacement (different slot).
    [[nodiscard]] auto same_slot_as(const ResolvedSlot &other) const noexcept -> bool {
      return append == other.append && sorted == other.sorted;
    }

    [[nodiscard]] auto load() const noexcept -> Payload {
      if (append != nullptr) {
        return append->payload_bits.load(std::memory_order_acquire);
      }
      return sorted->payload_bits.load(std::memory_order_acquire);
    }

    // CAS, not a plain store: a store could overwrite a concurrent displacement's tombstone
    // (on_displaced), resurrecting an orphaned slot that walks would carry forward as a
    // permanent duplicate. On failure the caller must not retry blindly (see put()).
    [[nodiscard]] auto try_store(Payload &expected_current, Payload value) const noexcept -> bool {
      if (append != nullptr) {
        return append->payload_bits.compare_exchange_strong(
            expected_current, value, std::memory_order_release, std::memory_order_acquire);
      }
      return sorted->payload_bits.compare_exchange_strong(
          expected_current, value, std::memory_order_release, std::memory_order_acquire);
    }

    void store_hash(uint64_t hash) const noexcept {
      if (append != nullptr) {
        append->hash.store(hash, std::memory_order_release);
      } else {
        sorted->hash.store(hash, std::memory_order_release);
      }
    }

    [[nodiscard]] auto version_ref() const noexcept -> std::atomic<uint64_t> & {
      return append != nullptr ? append->version : sorted->version;
    }

    // Brackets put()'s in-place update with the seqlock. Safe unconditionally: put() holds the
    // per-key stripe lock, so only readers interleave -- and a version bump on a failed CAS just
    // costs a reader one retry.
    void begin_write() const noexcept { version_ref().fetch_add(1, std::memory_order_release); }
    void end_write() const noexcept { version_ref().fetch_add(1, std::memory_order_release); }
  };

  // Seqlock reader for raw slot handles (merge loop, collect_live_entries): consistent
  // (hash, payload) pair across a racing begin/end_write() bracket.
  template <typename SlotT>
  [[nodiscard]] static auto load_slot_consistent(const SlotT &slot) noexcept -> std::pair<uint64_t, Payload> {
    SpinBackoff backoff;
    while (true) {
      const uint64_t version_before = slot.version.load(std::memory_order_acquire);
      if (version_before % 2 != 0) {
        backoff.wait();
        continue;
      }
      const uint64_t hash = slot.hash.load(std::memory_order_acquire);
      const Payload payload = slot.payload_bits.load(std::memory_order_acquire);
      std::atomic_thread_fence(std::memory_order_acquire);
      const uint64_t version_after = slot.version.load(std::memory_order_acquire);
      if (version_before == version_after) {
        return {hash, payload};
      }
    }
  }

  class AppendRegion {
   public:
    // Unpublished slots stay all-zero like MAP_ANONYMOUS memory: never faulted or zeroed upfront.
    explicit AppendRegion(size_t capacity)
        : capacity_(capacity),
          mapping_(mmap_anon_or_throw(capacity_ * sizeof(AppendSlot)), capacity_ * sizeof(AppendSlot)),
          slots_(static_cast<AppendSlot *>(mapping_.get())) {}

    AppendRegion(const AppendRegion &) = delete;
    auto operator=(const AppendRegion &) -> AppendRegion & = delete;

    auto data() noexcept -> AppendSlot * { return slots_; }
    [[nodiscard]] auto data() const noexcept -> const AppendSlot * { return slots_; }

    [[nodiscard]] auto size() const noexcept -> size_t { return tail_.load(std::memory_order_acquire); }

    auto reserve() noexcept -> size_t {
      size_t index = tail_.load(std::memory_order_relaxed);
      while (true) {
        if (index >= capacity_) {
          return capacity_;
        }
        if (tail_.compare_exchange_weak(index, index + 1, std::memory_order_acq_rel, std::memory_order_relaxed)) {
          return index;
        }
      }
    }

    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    void publish(size_t index, Key key, uint64_t hash, Payload value) noexcept {
      slots_[index].key = key;
      slots_[index].hash.store(hash, std::memory_order_relaxed);
      slots_[index].payload_bits.store(value, std::memory_order_relaxed);
      // Bounds before the published flag: scan() may skip this region on bounds(), so the
      // reverse order could hide an already-visible key as "out of range".
      update_bounds(key);
      slots_[index].published.store(true, std::memory_order_release);
    }

    // Conservative [min, max] envelope letting scan() skip this region's O(size()) walk on a
    // disjoint range. Key isn't lock-free atomic without -mcx16, so a dedicated spinlock guards
    // just this pair, off the lock-free insert path.
    void update_bounds(Key key) noexcept {
      SpinBackoff backoff;
      while (bounds_lock_.test_and_set(std::memory_order_acquire)) {
        backoff.wait();
      }
      if (!has_bounds_) {
        bounds_min_ = key;
        bounds_max_ = key;
        has_bounds_ = true;
      } else {
        if (key < bounds_min_) {
          bounds_min_ = key;
        }
        if (bounds_max_ < key) {
          bounds_max_ = key;
        }
      }
      bounds_lock_.clear(std::memory_order_release);
    }

    // Returns false when nothing was ever published (caller falls through to a normal scan).
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    [[nodiscard]] auto bounds(Key &out_min, Key &out_max) const noexcept -> bool {
      SpinBackoff backoff;
      while (bounds_lock_.test_and_set(std::memory_order_acquire)) {
        backoff.wait();
      }
      const bool present = has_bounds_;
      if (present) {
        out_min = bounds_min_;
        out_max = bounds_max_;
      }
      bounds_lock_.clear(std::memory_order_release);
      return present;
    }

    auto find_with_index(const AppendIndex &index, Key key, uint64_t hash) noexcept -> AppendSlot * {
      return const_cast<AppendSlot *>(static_cast<const AppendRegion *>(this)->find_with_index(index, key, hash));
    }

    [[nodiscard]] auto find_with_index(const AppendIndex &index,
                                       Key key,
                                       uint64_t hash) const noexcept -> const AppendSlot * {
      const auto slot_plus_one =
          index.find_slot_index(key, hash, [&](size_t slot_index) -> const AppendSlot & { return slots_[slot_index]; });
      if (slot_plus_one == AppendIndex::kNotFound) {
        return nullptr;
      }
      const AppendSlot &slot = slots_[static_cast<size_t>(slot_plus_one - 1)];
      if (!slot.published.load(std::memory_order_acquire)) {
        return nullptr;
      }
      return (slot.key == key && slot.clean_hash() == hash) ? &slot : nullptr;
    }

    void collect_live_entries(std::vector<EntrySnapshot> &out, size_t count) const {
      for (size_t i = 0; i < count; ++i) {
        const AppendSlot &slot = slots_[i];
        if (!slot.published.load(std::memory_order_acquire)) {
          continue;
        }
        // Consistent pair: every entry feeds the offset_mapper, so a torn pair could misdirect
        // a T2 read.
        const auto [hash, value] = load_slot_consistent(slot);
        if (!is_live(value)) {
          continue;
        }
        out.push_back(EntrySnapshot{slot.key, value, hash});
      }
    }

   private:
    size_t capacity_;
    MmapGuard mapping_;  // Owns the slots below; unmaps on destruction.
    AppendSlot *slots_;
    std::atomic<size_t> tail_{0};

    // See update_bounds()/bounds() above. mutable: bounds() is logically const (just a read),
    // but must briefly take the lock like any other reader.
    mutable std::atomic_flag bounds_lock_;
    bool has_bounds_ = false;
    Key bounds_min_{};
    Key bounds_max_{};
  };

  // Bundles a region with its hash index as one atomically-swappable unit: two independent
  // atomics could be read as a torn pair (region with the wrong generation's index).
  struct AppendGeneration {
    AppendRegion *region;
    AppendIndex *index;
  };

  static void delete_generation(AppendGeneration *generation) noexcept {
    if (generation == nullptr) {
      return;
    }
    delete generation->region;
    delete generation->index;
    delete generation;
  }

  void note_append_region_created() noexcept {
    int64_t live = append_region_live_count_.fetch_add(1, std::memory_order_relaxed) + 1;
    int64_t peak = append_region_peak_count_.load(std::memory_order_relaxed);
    while (live > peak && !append_region_peak_count_.compare_exchange_weak(peak, live, std::memory_order_relaxed)) {
    }
  }
  void note_append_region_destroyed() noexcept { append_region_live_count_.fetch_sub(1, std::memory_order_relaxed); }

  // ─── Helper Methods: Key and Lookup ────────────────────────────────────
  auto prepare_key_and_hash(std::span<const std::byte> key) const noexcept -> std::pair<t1_detail::StoreKey, uint64_t> {
    const t1_detail::StoreKey prefix = t1_detail::prefix_from_bytes(key);
    return {prefix, t1_detail::hash_full_key(key)};
  }

  static auto is_live(Payload value) noexcept -> bool { return value != STORE_NOT_FOUND; }

  auto find_sorted(const SortedRegion &sorted, Key key, uint64_t hash) const noexcept -> const SortedSlot * {
    if constexpr (Config::UseBloomFilter) {
      if (!sorted.bloom.maybe_contains(hash)) {
        return nullptr;
      }
    }

    const size_t slot_count = sorted.size;
    if (slot_count == 0) {
      return nullptr;
    }

    const std::span<const SortedSlot> slots_span(sorted.slots.get(), sorted.size);
    auto found_it = std::ranges::lower_bound(slots_span, key, {}, &SortedSlot::key);
    const SortedSlot *end = sorted.slots.get() + sorted.size;
    const SortedSlot *found = (found_it != slots_span.end()) ? &(*found_it) : end;

    while (found != end && found->key == key) {
      if (found->clean_hash() == hash) {
        return found;
      }
      ++found;
    }
    return nullptr;
  }

  // Runs `func` epoch-registered: every method dereferencing the guarded regions must go
  // through this, or reorganize() could free buffers still in use. `func` must never block on
  // reorganize() completing (deadlock against wait_until_epoch()).
  template <typename Func>
  auto with_epoch_guard(Func &&func) const noexcept(noexcept(func())) -> decltype(func()) {
    T1ReadHandle handle(active_epochs_, reorg_epoch_.load(std::memory_order_relaxed));
    return func();
  }

  auto resolve(Key key, uint64_t hash) noexcept -> ResolvedSlot {
    AppendGeneration *active_gen = append_active_.load(std::memory_order_acquire);
    if (AppendSlot *slot = active_gen->region->find_with_index(*active_gen->index, key, hash)) {
      return ResolvedSlot{slot, nullptr};
    }

    // A frozen-region hit means an in-place update would be a Lost Update this cycle (see
    // FreezableRegion) -- bypass to a fresh active entry instead. Safe: T2 never relocates, so
    // the bypass offset stays valid; a later reorganize() dedups by key + clean hash.
    if (AppendGeneration *imm_gen = append_immutable_.get()) {
      if (imm_gen->region->find_with_index(*imm_gen->index, key, hash) != nullptr) {
        return ResolvedSlot{};
      }
    }

    if (const SortedRegion *frozen_sorted = sorted_write_frozen_.get()) {
      // While frozen, sorted_region_ points at this same region, so no second check below.
      if (find_sorted(*frozen_sorted, key, hash) != nullptr) {
        return ResolvedSlot{};
      }
    } else {
      const auto sorted = sorted_region_.load(std::memory_order_acquire);
      if (const SortedSlot *slot = find_sorted(*sorted, key, hash)) {
        return ResolvedSlot{nullptr, slot};
      }
    }

    return ResolvedSlot{};
  }

  void publish_append_index(
      AppendIndex &idx, const AppendRegion &region, size_t index, Key key, uint64_t hash) noexcept {
    const auto slot_plus_one = static_cast<typename AppendIndex::slot_index_type>(index + 1);
    idx.publish_slot(
        key,
        slot_plus_one,
        hash,
        [&](size_t slot_index) -> const AppendSlot & { return region.data()[slot_index]; },
        [](const AppendSlot &displaced) {
          // Same-key insert-race loser, unreachable via the index now. Tombstone it so walks
          // (which don't consult the index) don't carry it forward as a duplicate.
          displaced.payload_bits.store(vmemkv::STORE_NOT_FOUND, std::memory_order_release);
        });
  }

  static void assert_no_duplicates(const std::vector<EntrySnapshot> &entries) {
    const auto duplicate =
        std::adjacent_find(entries.begin(), entries.end(), [](const EntrySnapshot &lhs, const EntrySnapshot &rhs) {
          return t1_detail::entry_identity_equal(lhs.key, lhs.hash, rhs.key, rhs.hash);
        });
    assert(duplicate == entries.end() &&
           "T1Index invariant violated: duplicate key prefix + clean hash detected in merge output");
    (void)duplicate;
  }

  // ─── Member Variables ──────────────────────────────────────────────────
  const size_t append_cap_;
  mutable std::atomic<bool> reorg_in_progress_{false};
  std::atomic<uint64_t> reorg_epoch_{1};

  std::atomic<const SortedRegion *> sorted_region_{nullptr};
  std::atomic<AppendGeneration *> append_active_{nullptr};
  FreezableRegion<AppendGeneration *> append_immutable_;
  std::atomic<int64_t> append_region_live_count_{0};
  std::atomic<int64_t> append_region_peak_count_{0};
  // Frozen for the merge loop's duration (same region sorted_region_ points at); closes the
  // sorted-region Lost Update counterpart append_immutable_ already guards.
  FreezableRegion<const SortedRegion *> sorted_write_frozen_;

  mutable ThreadReferenceTracker<uint64_t> active_epochs_;
};

template <typename Config>
auto T1Index<Config>::append_slot_bytes() noexcept -> size_t {
  return sizeof(AppendSlot);
}

}  // namespace vmemkv
