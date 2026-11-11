// lock_free_hash_table.hpp -- Fixed-capacity open-addressing hash table.
//
// Append-only publication: writers publish a slot index only after the slot is fully visible;
// readers look up lock-free via atomic bucket loads; clearing requires external synchronization
// guaranteeing no concurrent readers/writers still depend on the previous contents.
//
// Used by T1 for append-region indexing (key -> append slot index): inserts are O(1) expected,
// deletes don't physically remove buckets, and reorganize clears the whole table.

#pragma once

#include <sys/mman.h>

#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <system_error>
#include <vmemkv/core/mmap.hpp>

template <typename Key, typename Slot>
class LockFreeHashTable {
 public:
  using slot_index_type = uint32_t;
  static constexpr slot_index_type kEmpty = 0;
  static constexpr slot_index_type kNotFound = 0;

  // Bucket{}'s zero-initialized atomic (kEmpty) matches the bit pattern MAP_ANONYMOUS memory
  // already reads as, so buckets_ can come from a raw mmap instead of a constructed std::array:
  // untouched pages cost no resident memory instead of faulting in the whole table up front
  // regardless of how many entries this generation actually holds.
  explicit LockFreeHashTable(size_t slot_capacity)
      : bucket_count_(std::bit_ceil(slot_capacity * 2)),
        mapping_(vmemkv::mmap_anon_or_throw(bucket_count_ * sizeof(Bucket)), bucket_count_ * sizeof(Bucket)),
        buckets_(static_cast<Bucket *>(mapping_.get())) {}

  LockFreeHashTable(const LockFreeHashTable &) = delete;
  auto operator=(const LockFreeHashTable &) -> LockFreeHashTable & = delete;

  // Caller-provided hash, e.g. hash(full_key) for T1 append indexing.
  template <typename SlotAccessor>
  auto find_slot_index(const Key &key, uint64_t hash, SlotAccessor &&slot_at) const noexcept -> slot_index_type {
    size_t pos = bucket_index(hash);
    for (size_t probe = 0; probe < bucket_count_; ++probe) {
      const slot_index_type observed = buckets_[pos].slot_plus_one.load(std::memory_order_acquire);
      if (observed == kEmpty) {
        return kNotFound;
      }

      const Slot &slot = slot_at(static_cast<size_t>(observed - 1));
      if (slot.published.load(std::memory_order_acquire) && slot.clean_hash() == hash && slot.key == key) {
        return observed;
      }
      pos = next_pos(pos);
    }
    return kNotFound;
  }

  // `on_displaced` is invoked with the slot that publish_slot() bumps out when two callers race
  // to insert the same key (routine in T1: resolve()'s immutable-region bypass makes every
  // concurrent put() for a key already in append_immutable_ take the "not found, insert new"
  // path). Without this callback, the loser's slot stays published and reachable by
  // collect_live_entries()/scan()'s raw slot-array walks even though this index can no longer
  // find it -- an orphaned-but-live duplicate that would keep accumulating and get carried into
  // every future reorganize() and T2 checkpoint.
  template <typename SlotAccessor, typename OnDisplaced>
  auto publish_slot(const Key &key,
                    slot_index_type slot_plus_one,  // NOLINT(bugprone-easily-swappable-parameters)
                    uint64_t hash,
                    SlotAccessor &&slot_at,
                    OnDisplaced &&on_displaced) noexcept -> bool {
    size_t pos = bucket_index(hash);
    for (size_t probe = 0; probe < bucket_count_; ++probe) {
      // Retries by re-reading `pos` (not advancing) on CAS failure: a plain store() here is
      // unsound under >2-way contention -- if B and C both see A's slot and both decide to
      // displace it, a plain store from whichever runs second would silently clobber the
      // other's already-published slot without ever calling on_displaced on it, leaking it as
      // an orphaned-but-live duplicate. The CAS only ever displaces whatever occupant is
      // actually present when it wins, so every slot pushed out of a bucket is retired by
      // whoever actually pushed it out.
      slot_index_type expected = buckets_[pos].slot_plus_one.load(std::memory_order_acquire);
      while (true) {
        if (expected == kEmpty) {
          if (buckets_[pos].slot_plus_one.compare_exchange_weak(
                  expected, slot_plus_one, std::memory_order_release, std::memory_order_acquire)) {
            return true;
          }
          continue;  // expected now holds whatever's actually there now; re-examine it below.
        }

        const Slot &slot = slot_at(static_cast<size_t>(expected - 1));
        if (slot.published.load(std::memory_order_acquire) && slot.clean_hash() == hash && slot.key == key) {
          const slot_index_type displaced_index = expected;
          if (buckets_[pos].slot_plus_one.compare_exchange_weak(
                  expected, slot_plus_one, std::memory_order_release, std::memory_order_acquire)) {
            on_displaced(slot_at(static_cast<size_t>(displaced_index - 1)));
            return true;
          }
          continue;  // Someone else raced us here too; expected was refreshed, loop re-examines it.
        }

        break;  // Occupied by an unrelated key -- probe the next bucket.
      }
      pos = next_pos(pos);
    }
    return false;
  }

 private:
  struct Bucket {
    std::atomic<slot_index_type> slot_plus_one{kEmpty};
  };

  [[nodiscard]] auto bucket_index(uint64_t hash) const noexcept -> size_t {
    return static_cast<size_t>(hash) & (bucket_count_ - 1);
  }

  [[nodiscard]] auto next_pos(size_t pos) const noexcept -> size_t { return (pos + 1) & (bucket_count_ - 1); }

  size_t bucket_count_;
  vmemkv::MmapGuard mapping_;  // Owns the buckets below; unmaps on destruction.
  Bucket *buckets_;
};
