// write_path.hpp - Write-path helpers for VMemKVImpl: stripe discipline and WAL batching.
//
// Stripe protocol (type-enforced by with_key_stripe() + StripeResult): the key's stripe lock
// is held only for the T1/T2 mutation and the WAL reservation (which fixes LSN order among
// same-key writers). The multi-millisecond await_durable() wait and the reorg check run
// after the lock is released -- holding the stripe through either would needlessly serialize
// other writers on the same stripe, and holding a T2 write handle into
// maybe_reorganize_if_needed() can deadlock against a concurrent reorganize cycle.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <optional>
#include <span>
#include <vector>
#include <vmemkv/api/utils.hpp>
#include <vmemkv/core/bytes.hpp>
#include <vmemkv/impl/read_path.hpp>
#include <vmemkv/impl/t2_ownership.hpp>
#include <vmemkv/t1_index/t1_index.hpp>
#include <vmemkv/t2_flat_file/t2_flat_file.hpp>
#include <vmemkv/wal/wal.hpp>

namespace vmemkv {

// What a with_key_stripe() callback returns: applied or not, the WAL record reserved under
// the stripe (awaited after release), and whether to reorg-check after release.
struct StripeResult {
  bool applied = false;
  Wal::PendingRecord *pending = nullptr;
  bool need_reorg_check = false;
};

namespace write_path_debug {
inline std::atomic<uint64_t> g_stripe_lock_ns{0};
inline std::atomic<uint64_t> g_mutation_ns{0};
inline std::atomic<uint64_t> g_await_durable_ns{0};
inline std::atomic<uint64_t> g_sample_count{0};

inline void reset() noexcept {
  g_stripe_lock_ns.store(0, std::memory_order_relaxed);
  g_mutation_ns.store(0, std::memory_order_relaxed);
  g_await_durable_ns.store(0, std::memory_order_relaxed);
  g_sample_count.store(0, std::memory_order_relaxed);
}
}  // namespace write_path_debug

// Holds the stripe lock for the callback, then awaits the WAL record and reorg-checks with
// the lock released. Returns whether the mutation applied.
template <typename LockStripe, typename AwaitDurable, typename CheckReorg, typename Fn>
auto with_key_stripe(LockStripe &&lock_stripe,
                     AwaitDurable &&await_durable,
                     CheckReorg &&check_reorg,
                     std::span<const std::byte> key,
                     Fn &&func) -> bool {
  constexpr uint64_t kSampleMask = 0xFF;
  thread_local static uint64_t tl_counter = 0;
  const bool sample = ((++tl_counter) & kSampleMask) == 0;
  std::chrono::steady_clock::time_point sample_start;
  std::chrono::steady_clock::time_point sample_locked;
  std::chrono::steady_clock::time_point sample_mutated;
  if (sample) {
    sample_start = std::chrono::steady_clock::now();
  }

  StripeResult result;
  {
    std::lock_guard<std::mutex> key_lock(lock_stripe(key));
    if (sample) {
      sample_locked = std::chrono::steady_clock::now();
    }
    result = func();
  }
  if (sample) {
    sample_mutated = std::chrono::steady_clock::now();
  }

  if (result.pending != nullptr) {
    await_durable(result.pending);
  }

  if (sample) {
    const auto sample_finished = std::chrono::steady_clock::now();
    write_path_debug::g_stripe_lock_ns.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(sample_locked - sample_start).count(),
        std::memory_order_relaxed);
    write_path_debug::g_mutation_ns.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(sample_mutated - sample_locked).count(),
        std::memory_order_relaxed);
    write_path_debug::g_await_durable_ns.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(sample_finished - sample_mutated).count(),
        std::memory_order_relaxed);
    write_path_debug::g_sample_count.fetch_add(1, std::memory_order_relaxed);
  }

  if (result.need_reorg_check) {
    check_reorg();
  }
  return result.applied;
}

// update_impl()'s outcomes for a non-inline entry: Aborted (key vanished or in-place failed),
// FellThrough (caller falls back to write_entry_lockfree()), Applied (in-place done, `pending`
// carries the WAL record).
enum class InPlaceOutcome : std::uint8_t { Aborted, FellThrough, Applied };
struct InPlaceUpdateResult {
  InPlaceOutcome outcome;
  Wal::PendingRecord *pending = nullptr;
};

// T1 inlining for a (key, value) pair. Keys only (T1 stores just a 16-byte prefix), and
// never keys ending in 0x00: with no T2 record, scan_impl() recovers the length via
// inline_key_len()'s trailing-zero trim, unambiguous only when trailing zeros are padding.
template <typename ConfigT>
auto try_make_inline_payload(std::span<const std::byte> full_key,
                             std::span<const std::byte> value,
                             uint8_t &out_size) noexcept -> std::optional<uint64_t> {
  if constexpr (ConfigT::UseT1InlineValue) {
    if (full_key.size() <= t1_detail::kPrefixBytes && (full_key.empty() || full_key.back() != std::byte{0})) {
      if (!value.empty() && value.size() <= t1_detail::kInlineValueByteCount) {
        uint64_t payload = 0;
        std::memcpy(&payload, value.data(), value.size());
        // A full-width all-1-bits value equals the STORE_NOT_FOUND sentinel: inlining it would
        // read as permanently absent. Only possible at full width (shorter values leave upper
        // bytes zeroed). Falls through to the T2 path, whose offset payload can't collide.
        if (payload == vmemkv::STORE_NOT_FOUND) {
          return std::nullopt;
        }
        out_size = static_cast<uint8_t>(value.size());
        return payload;
      }
    }
  }
  return std::nullopt;
}

// Appends a (key, value) pair to T2 and publishes it in T1, retrying across AppendRegionFull.
// The WAL reservation stays the caller's job, after the mutation applies.
template <typename ConfigT, typename T1, typename MaybeReorg>
auto write_entry_lockfree(T1 &tier1,
                          T2FlatFile &tier2,
                          T2Ownership<ConfigT> &own,
                          MaybeReorg &&maybe_reorg,
                          std::span<const std::byte> full_key,
                          std::span<const std::byte> value,
                          std::optional<typename T1::LookupResult> prev) -> bool {
  // Shared publish tail for the inline and append paths: resolve the previous entry for
  // segment accounting (callers that already looked it up pass it in; single-threaded WAL
  // replay and bulk load fetch it here), then T1-put the new payload.
  auto publish = [&](uint64_t new_payload, bool new_is_offset, bool new_is_inline, uint8_t inline_size) -> bool {
    typename T1::LookupResult old = prev.has_value() ? *prev : tier1.get_with_hash(full_key);
    if (tier1.put(full_key, new_payload, new_is_inline, inline_size, /*mark_t2_dirty=*/true) !=
        T1::PutResult::Applied) {
      return false;
    }
    own.note_replace(old.payload_bits, old.raw_hash, new_is_offset, new_payload, true);
    return true;
  };
  while (true) {
    uint8_t inline_size = 0;
    if (const auto inline_payload = try_make_inline_payload<ConfigT>(full_key, value, inline_size)) {
      if (publish(*inline_payload, /*new_is_offset=*/false, /*new_is_inline=*/true, inline_size)) {
        return true;
      }
      // AppendRegionFull: retry from the top after maybe_reorg().
      maybe_reorg();
      continue;
    }

    // Block count checked pre-append (not just asserted post-append); see
    // checked_block_count().
    const uint64_t block_count = t2_detail::checked_block_count(full_key.size(), value.size());

    // acquire_write_handle() (not plain get_memory()) defers while checkpoint_internal() holds
    // new writers stopped -- and `mem` stays alive across the T2 append *and* the T1 publish
    // below, not released in between, closing checkpoint_internal()'s residual-window race.
    {
      T2FlatFile::T2MemoryHandle mem = tier2.acquire_write_handle();
      uint64_t offset = vmemkv::T2FlatFile::append_default(mem, full_key, value);
      uint64_t encoded_payload = t2_detail::encode_offset_payload(offset, block_count);
      if (publish(encoded_payload, /*new_is_offset=*/true, /*new_is_inline=*/false, 0)) {
        return true;
      }
    }
    // Released strictly before maybe_reorg(): that call can block on a concurrent reorganize()
    // that is itself waiting for this handle to drain -- holding it longer deadlocks the two.
    // AppendRegionFull: the appended T2 record becomes unreachable garbage for a future
    // reorganize() to reclaim -- retry from the top.
    maybe_reorg();
  }
}

// update_impl()'s in-place decision for a non-inline entry: applies it or conclusively falls
// through to write_entry_lockfree().
template <typename ConfigT, typename T1>
auto try_in_place_update(T1 &tier1,
                         T2FlatFile &tier2,
                         T2Ownership<ConfigT> &own,
                         Wal &wal,
                         std::span<const std::byte> full_key,
                         std::span<const std::byte> value) -> InPlaceUpdateResult {
  const auto res = tier1.get_with_hash(full_key);
  if (res.payload_bits == vmemkv::STORE_NOT_FOUND) {
    return {InPlaceOutcome::Aborted};
  }
  if (t1_detail::is_inline(res.raw_hash)) {
    return {InPlaceOutcome::FellThrough};
  }

  const T2Memory *mem = tier2.get_memory();
  const uint64_t offset = res.payload_bits & t2_detail::kPayloadOffsetMask;
  const uint64_t base_boundary = mem->base_boundary.load(std::memory_order_acquire);

  // Fast-path immutable base region:
  // Base-region records cannot be mutated in place and must redirect to out-of-place append.
  // Because T1 only stores a 16-byte prefix plus a 60-bit hash, we must verify that the full key
  // actually matches before allowing the out-of-place update. A hash collision on a nonexistent
  // key must abort (return false), never insert a new record or overwrite T1.
  // Base records are immutable, so try_read_base_record() verifies the key seqlock-free.
  if (offset < base_boundary) {
    if (full_key.size() <= t1_detail::kPrefixBytes) {
      return {InPlaceOutcome::FellThrough};
    }
    thread_local static std::vector<std::byte> tl_base_buf;
    if (const auto base_record =
            vmemkv::try_read_base_record<ConfigT>(mem, res.payload_bits, BaseReader::kGet, &tl_base_buf);
        base_record.has_value()) {
      if (!vmemkv::bytes_equal(base_record->key, full_key)) {
        return {InPlaceOutcome::Aborted};
      }
      return {InPlaceOutcome::FellThrough};
    }
  }

  // Mutable tail record (or base record whose direct read fell back to the tail path):
  // Read key/alloc_len fresh under the seqlock: an unprotected t2_.at() isn't safe here.
  bool key_matches = false;
  uint32_t alloc_len = 0;
  read_t2_record_seqlock([&]() -> T2RecordView { return T2FlatFile::at(offset, mem); },
                         [&](const T2RecordView &record) -> bool {
                           key_matches = vmemkv::bytes_equal(record.key, full_key);
                           alloc_len = record.header->alloc_len;
                           return true;
                         });

  if (!key_matches) {
    return {InPlaceOutcome::Aborted};
  }

  // If claimed by an in-flight checkpoint or in the base region, redirect out-of-place.
  if (offset < base_boundary || offset < own.load_watermark()) {
    return {InPlaceOutcome::FellThrough};
  }
  if (value.size() > alloc_len) {
    return {InPlaceOutcome::FellThrough};
  }

  // Registered *before* writing, held through the write.
  const auto write_guard = own.enter_in_place(offset);

  if (!T2FlatFile::update_value_at(offset, value, mem)) {
    return {InPlaceOutcome::Aborted};
  }
  return {InPlaceOutcome::Applied, wal.reserve_update(full_key, value)};
}

}  // namespace vmemkv
