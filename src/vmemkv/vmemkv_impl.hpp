// vmemkv_impl.hpp - vmemkv coordinator: routes reads/writes across T1Index and T2FlatFile.
//
// The coordination logic itself lives in src/vmemkv/impl/: hooks.hpp (checkpoint seams and the
// in-place-update barrier), t2_ownership.hpp (T2 payload codec, capture watermark,
// barrier), read_path.hpp (base-region reads, seqlock-protected record reads), write_path.hpp
// (inline payloads, lock-free appends, in-place updates, stripe protocol, WAL batching),
// checkpoint_coordinator.hpp (checkpoint/reorganize orchestration),
// and recovery.hpp (checkpoint adopt
// helpers). Poll intervals live in core/background_poll.hpp. This class owns lifetimes (T1/T2/WAL,
// stripes, background workers) and delegates every operation to those units.
//
// ─── CONCURRENCY SPECIFICATION & MATRIX ──────────────────────────────────────
//
// The vmemkv architecture enforces thread-safety at the VMemKVImpl level,
// coordinating and routing operations across two underlying structural layers:
// 1. T1Index (In-memory Index)
// 2. T2FlatFile (Binary Disk Log File)
//
// +--------------------+-------------------+---------------------------------------------------+
// | Component          | Read Operations   | Write Operations (Insert/Update/Delete)           |
// +--------------------+-------------------+---------------------------------------------------+
// | vmemkv::T1Index    | Thread-Safe       | Thread-Unsafe (relies on VMemKVImpl serialization)|
// | vmemkv::T2FlatFile  | Thread-Safe       | Thread-Unsafe (relies on VMemKVImpl serialization)|
// | vmemkv::VMemKVStore | Thread-Safe       | Thread-Safe (fully serialized/coordinated)        |
// +--------------------+-------------------+---------------------------------------------------+
//
// Below is the Concurrency Matrix detailing synchronization across macro operations:
//
// +--------------------------+------------+--------------------+---------------------+-----------------------------+-----------------------------+
// | Operation A \ Operation B | Get / Scan | Write (Same Key)   | Write (Diff Key)    | Reorganize (T1-only)        |
// Checkpoint                  |
// +--------------------------+------------+--------------------+---------------------+-----------------------------+-----------------------------+
// | Get / Scan               | Concurrent | Concurrent         | Concurrent          | Concurrent (retry/snapshot) |
// Concurrent                  | | Write (Same Key)         | Concurrent | Serial (Stripes)   | Concurrent          |
// Concurrent                  | Concurrent (brief append stall) | | Write (Diff Key)         | Concurrent | Concurrent
// | Concurrent (atomic) | Concurrent                  | Concurrent (brief append stall) | | Reorganize (T1-only)     |
// Concurrent | Concurrent         | Concurrent          | Single-flight (trigger.running) | Single-flight
// (trigger.running) | | Checkpoint               | Concurrent | Concurrent         | Concurrent          |
// Single-flight (trigger.running) | Single-flight (trigger.running) |
// +--------------------------+------------+--------------------+---------------------+-----------------------------+-----------------------------+
//
// Reorg/checkpoint single-flight is shared (checkpoint_coordinator.hpp). T1 background
// maintenance runs on a sharded worker pool with per-shard Closing/Split protocol
// (docs/specification/t1_sharding_design.md); explicit reorganize()/checkpoint() wait via
// wait_until_reorg_not_running().

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>
#include <vmemkv/checkpoint/checkpoint.hpp>
#include <vmemkv/config.hpp>
#include <vmemkv/core/background_poll.hpp>
#include <vmemkv/core/bytes.hpp>
#include <vmemkv/core/cgroup_memory_throttle.hpp>
#include <vmemkv/core/reference_tracker.hpp>
#include <vmemkv/core/single_flight.hpp>
#include <vmemkv/core/swap_check.hpp>
#include <vmemkv/impl/checkpoint_coordinator.hpp>
#include <vmemkv/impl/compaction.hpp>
#include <vmemkv/impl/hooks.hpp>
#include <vmemkv/impl/read_path.hpp>
#include <vmemkv/impl/recovery.hpp>
#include <vmemkv/impl/t2_ownership.hpp>
#include <vmemkv/impl/write_path.hpp>
#include <vmemkv/scan_control.hpp>
#include <vmemkv/t1_index/sharded_t1_index.hpp>
#include <vmemkv/t2_flat_file/t2_flat_file.hpp>
#include <vmemkv/wal/wal.hpp>

struct alignas(vmemkv::kCacheLineSize) AlignedMutex {
  std::mutex mu;
};

namespace vmemkv {

// ─── VMemKVImpl Layer Overview ─────────────────────────────────────────────
// StoreAdapter -> VMemKVImpl -> { ShardedT1Index (key -> payload), T2FlatFile (values) }
// * get: T1 payload; inline decodes, offset resolves T2.
// * insert: T2 append -> T1 publish -> WAL reserve, awaited post-stripe.
// * reorganize: T1-only merge, zero I/O.
// * checkpoint: in-place msync + manifest commit.
//
template <typename ConfigT = vmemkv::Config<>>
class VMemKVImpl {
 public:
  static constexpr bool kIsEnabled = true;
  using ConfigType = ConfigT;

  static auto name() -> std::string {
    std::vector<std::string> parts;
    if constexpr (ConfigT::UseBloomFilter) {
      parts.emplace_back("Bloom");
    }
    if constexpr (ConfigT::UseT1InlineValue) {
      parts.emplace_back("T1InlineValue");
    }
    if constexpr (ConfigT::UseReadPolicyRandomOnly) {
      parts.emplace_back("ReadRandom");
    }
    if constexpr (ConfigT::UseReadPolicySeqOnly) {
      parts.emplace_back("ReadSeq");
    }
    if constexpr (ConfigT::UseReadPolicyNormalOnly) {
      parts.emplace_back("ReadNormal");
    }
    if constexpr (ConfigT::UseReadPolicyPreadOnly) {
      parts.emplace_back("ReadPread");
    }

    std::string result = "VMemKV";
    for (const auto &part : parts) {
      result += "/" + part;
    }
    return result;
  }

  using T1IndexT = vmemkv::ShardedT1Index<ConfigT>;

  // T1 payload codec, re-exported as ImplT::X names (tests use ImplT::kBlockAlignment et al.).
  // Canonical values live in t1_codec.hpp's vmemkv::t2_detail namespace.
  static constexpr uint64_t kSizeEmbeddingShift = t2_detail::kPayloadSizeShift;
  static constexpr uint64_t kOffsetMask = t2_detail::kPayloadOffsetMask;
  static constexpr uint64_t kBlockAlignment = t2_detail::kRecordBlockAlignment;
  static_assert(kSizeEmbeddingShift == t2_detail::kPayloadSizeShift);
  static_assert(kOffsetMask == t2_detail::kPayloadOffsetMask);
  static_assert(kBlockAlignment == t2_detail::kRecordBlockAlignment);

  // Construction phases, in order (each depends on the previous):
  // 1. validate_swap: fail fast before any mmap/recovery work.
  // 2. load_checkpoint_if_present: single-threaded adopt (t1_ starts with 0 workers so its
  //    shard can be wholesale-replaced; workers start right after, before replay, because
  //    replay put()s like live traffic and needs drainable MemTables).
  // 3. recover_from_wal, then clear recovering_.
  // 4. spawn_background_workers, last: workers must never observe recovering_ state.
  VMemKVImpl(const std::filesystem::path &t2_path, uint64_t t2_bytes_capacity)
      : t1_(ConfigT::kMemTableCapacityEntries, ConfigT::kShardTargetSizeEntries, /*worker_threads=*/0),
        t2_(t2_path, t2_bytes_capacity, adopted_t2_bytes_used(t2_path)),
        wal_(vmemkv::derive_wal_path(t2_path)),
        t2_own_((t2_bytes_capacity + ConfigT::kSegmentBytes - 1) / ConfigT::kSegmentBytes) {
    vmemkv::validate_swap_for_ltm();
    recovering_ = true;
    load_checkpoint_if_present(t2_path);
    t1_.start_workers();
    recover_from_wal();
    recovering_ = false;
    t2_own_.queue_empty_segments(t2_.get_memory()->read_only_offset.load(std::memory_order_acquire));
    spawn_background_workers();
  }

  VMemKVImpl(const VMemKVImpl &) = delete;
  auto operator=(const VMemKVImpl &) -> VMemKVImpl & = delete;
  VMemKVImpl(VMemKVImpl &&) = delete;
  auto operator=(VMemKVImpl &&) -> VMemKVImpl & = delete;

  // Durabilizes T2's mutable region and commits a manifest. Exposed for tests injecting hooks.
  // Throws std::system_error on I/O failure (msync/open/fstat); std::logic_error during recovery.
  template <typename PreStopHook = NoOpPreStopHook, typename PreFinishHook = NoOpPreFinishHook>
    requires CheckpointHook<PreStopHook> && CheckpointHook<PreFinishHook>
  void checkpoint_internal(PreStopHook pre_stop_hook = PreStopHook{}, PreFinishHook pre_finish_hook = PreFinishHook{}) {
    if (recovering_) {
      throw std::logic_error("checkpoint during recovery");
    }
    checkpoint_detail::checkpoint_internal(
        checkpoint_, t1_, t2_, wal_, t2_own_, t2_path(), pre_stop_hook, pre_finish_hook);
  }

  // T1-only in-memory merge, zero I/O. Safe anytime. Serialized with checkpoint() via
  // checkpoint_.running only to keep shard state untouched during a T1 checkpoint dump.
  void reorganize() {
    while (t1_.has_pending_memtables()) {
      checkpoint_detail::wait_until_checkpoint_not_running(checkpoint_);
      if (!t1_.has_pending_memtables()) {
        return;
      }
      if (auto guard = SingleFlightGuard::try_acquire(checkpoint_.running); guard.holds) {
        t1_.reorganize();
        return;
      }
    }
  }

  // Durabilizes T2's mutable region in place and commits the T1 checkpoint/manifest.
  // Throws std::system_error on I/O failure (msync/open/fstat); std::logic_error during recovery.
  void checkpoint() {
    if (recovering_) {
      throw std::logic_error("checkpoint during recovery");
    }
    checkpoint_detail::run_checkpoint(
        checkpoint_, t1_, t2_, wal_, t2_own_, t2_path(), NoOpPreStopHook{}, NoOpPreFinishHook{});
  }

  // Drains in-flight background compaction, dirty segments, and dirty shards until clean.
  void drain_compaction() {
    t1_.drain_maintenance();
    SpinBackoff backoff;
    for (;;) {
      if (!has_garbage_segments() && !t1_.has_dirty_shard() && !compact_.running.load(std::memory_order_acquire)) {
        break;
      }
      if (!run_compaction_cycle()) {
        backoff.wait();
      }
    }
    t1_.drain_maintenance();
  }

  // T1/T2 accessors, mainly for testing.
  auto t1() noexcept -> T1IndexT & { return t1_; }
  auto t1() const noexcept -> const T1IndexT & { return t1_; }
  auto t2() noexcept -> vmemkv::T2FlatFile & { return t2_; }
  auto t2() const noexcept -> const vmemkv::T2FlatFile & { return t2_; }
  auto compaction_probe_state() const noexcept -> const vmemkv::CompactionState & { return compact_; }
  void set_t1_chunk_split_bytes_for_test(uint64_t split_bytes) noexcept {
    checkpoint_.t1_chunk_split_bytes = split_bytes;
  }
  auto t1_checkpoint_chunks() const -> std::vector<vmemkv::T1ChunkMeta> { return checkpoint_.t1_chunks; }
  auto pending_segment_count() const -> size_t {
    const T2Memory *mem = t2_.get_memory();
    const uint64_t read_only = mem != nullptr ? mem->read_only_offset.load(std::memory_order_acquire) : 0;
    const uint64_t tail = mem != nullptr ? mem->tail_offset.load(std::memory_order_acquire) : 0;
    return t2_own_.garbage_segment_count(read_only, tail);
  }
  auto has_garbage_segments() const -> bool {
    const T2Memory *mem = t2_.get_memory();
    const uint64_t read_only = mem != nullptr ? mem->read_only_offset.load(std::memory_order_acquire) : 0;
    const uint64_t tail = mem != nullptr ? mem->tail_offset.load(std::memory_order_acquire) : 0;
    return t2_own_.has_garbage_segment(read_only, tail);
  }
  auto has_compaction_work() const -> bool {
    return t2_own_.has_punch_candidates() || t1_.has_dirty_shard() || has_garbage_segments();
  }
  auto punch_backlog_segments() const -> size_t { return t2_own_.punch_backlog_segments(); }
  auto compaction_running() const noexcept -> bool { return compact_.running.load(std::memory_order_acquire); }

  auto get_statistics() const noexcept -> vmemkv::VMemKVStatistics {
    vmemkv::VMemKVStatistics stats;
    stats.t1_split_count = t1_.total_splits();
    stats.t1_last_split_pause_us = t1_.last_split_pause_us();
    stats.t1_last_split_pause_end_ns = t1_.last_split_pause_end_ns();
    stats.checkpoint_count = checkpoint_.stats.checkpoint_count.load(std::memory_order_relaxed);
    vmemkv::snapshot_checkpoint_stats(checkpoint_.stats, stats);
    stats.total_reorganize_wait_duration_us = checkpoint_.stats.total_wait_us.load(std::memory_order_relaxed);
    stats.memtable_live_count = t1_.memtable_live_count();
    stats.memtable_peak_count = t1_.memtable_peak_count();
    stats.bulk_load_throttle_events = bulk_load_throttle_.throttle_events();
    stats.compaction_cycle_count = compact_.cycle_count.load(std::memory_order_relaxed);
    stats.t2_dirty_shard_count = t1_.dirty_shard_count();
    return stats;
  }

  // ─── Low-level byte-span APIs (called by StoreAdapter) ───────────────────────

  // Inserts a new key-value pair. Ordering: T2 write strictly precedes the T1 write (no
  // dangling offsets for concurrent readers); WAL append comes last, only after the mutation
  // succeeds (logging first would persist a never-happened write; replay would re-throw on
  // every restart, bricking the store -- unlogged failure is safe since T1/T2 rebuild from
  // the WAL). WAL wait runs after the stripe releases.
  auto insert_impl(std::span<const std::byte> full_key, std::span<const std::byte> value) -> bool {
    return vmemkv::with_key_stripe(
        lock_stripe_fn(),
        [this](Wal::PendingRecord *pending) { wal_.await_durable(pending); },
        full_key,
        [&]() -> vmemkv::StripeResult {
          const auto old = t1_.get_with_hash(full_key);
          if (old.payload_bits != vmemkv::STORE_NOT_FOUND) {
            return {};
          }
          vmemkv::write_entry_lockfree<ConfigT>(t1_, t2_, t2_own_, full_key, value, old);
          return {true, wal_.reserve_insert(full_key, value)};
        });
  }

  // Retrieves a value and invokes callback with its raw bytes. Lock-free.
  // A T1 lookup names a T2 offset guarded by one of two mechanisms: below read_only_offset the
  // offset is immutable forever (try_read_immutable_record() reads seqlock-free); at or above it,
  // read_t2_record_seqlock() re-checks a version around the read and retries torn observations.
  // get_memory() needs no handle: T2 never remaps to another T2Memory in a store's lifetime.
  template <typename Callback>
  auto get_impl(std::span<const std::byte> full_key, Callback callback) const -> bool {
    const auto res = t1_.get_with_hash(full_key);
    if (res.payload_bits == vmemkv::STORE_NOT_FOUND) {
      return false;
    }

    if (vmemkv::try_read_inline_value<ConfigT>(
            res.payload_bits, res.raw_hash, [&](std::span<const std::byte> value) { callback(value); })) {
      return true;
    }

    const T2Memory *mem = t2_.get_memory();
    const uint64_t offset = res.payload_bits & kOffsetMask;

    // Scratch for both paths below, used sequentially (never overlapping): the immutable region's
    // pread() destination, then the seqlock path's owned copy. thread_local/static: per-thread
    // reuse, no per-call allocation.
    // Ownership: tl_get_buf/tl_scan_* live here (get/scan callback buffers); read_path.hpp
    // owns tl_pread_scratch. No sharing across units.
    thread_local static std::vector<std::byte> tl_get_buf;
    if (const auto imm_record =
            vmemkv::try_read_immutable_record<ConfigT>(mem, res.payload_bits, ImmutableReader::kGet, &tl_get_buf);
        imm_record.has_value()) {
      if (vmemkv::bytes_equal(imm_record->key, full_key)) {
        callback(imm_record->value);
        return true;
      }
      // Defensive mismatch (bounds check already guards garbage): fall through to the
      // always-correct seqlock path instead of trusting this read.
    }

    // Torn-read fix: copy_func only *copies* into an owned buffer and returns -- never calls
    // back from inside. The version check bounds only the copy window, not what a callback
    // might do inside it; a concurrent in-place update could tear the caller's bytes mid-read.
    bool key_matches = vmemkv::read_t2_record_seqlock([&]() -> T2RecordView { return T2FlatFile::at(offset, mem); },
                                                      [&](const T2RecordView &record) -> bool {
                                                        if (!vmemkv::bytes_equal(record.key, full_key)) {
                                                          return false;
                                                        }
                                                        tl_get_buf.assign(record.value.begin(), record.value.end());
                                                        return true;
                                                      });

    if (key_matches) {
      callback(std::span<const std::byte>(tl_get_buf));
    }
    return key_matches;
  }

  // Updates an existing key: in-place on T2 when the allocation fits, else append + T1
  // repoint. WAL append only after the mutation applies. Stripe-guarded; WAL wait runs after
  // release.
  auto update_impl(std::span<const std::byte> full_key, std::span<const std::byte> value) -> bool {
    return vmemkv::with_key_stripe(
        lock_stripe_fn(),
        [this](Wal::PendingRecord *pending) { wal_.await_durable(pending); },
        full_key,
        [&]() -> vmemkv::StripeResult {
          // Ungated lookup: when the *current* value is inline this call never touches T2
          // (inline-ness of the new value re-decides fresh), so in-place has nothing to protect.
          const auto quick = t1_.get_with_hash(full_key);
          if (quick.payload_bits == vmemkv::STORE_NOT_FOUND) {
            return {};
          }
          if (t1_detail::is_inline(quick.raw_hash)) {
            vmemkv::write_entry_lockfree<ConfigT>(t1_, t2_, t2_own_, full_key, value, quick);
            return {true, wal_.reserve_update(full_key, value)};
          }
          const vmemkv::InPlaceUpdateResult result =
              vmemkv::try_in_place_update<ConfigT>(t2_, t2_own_, wal_, full_key, value, quick.payload_bits);
          if (result.outcome == vmemkv::InPlaceOutcome::Aborted) {
            return {};
          }
          if (result.outcome == vmemkv::InPlaceOutcome::Applied) {
            return {true, result.pending};
          }
          vmemkv::write_entry_lockfree<ConfigT>(t1_, t2_, t2_own_, full_key, value, quick);
          return {true, wal_.reserve_update(full_key, value)};
        });
  }

  // Logically removes a key (T1 tombstone; space reclaims at reorganize). Stripe-guarded;
  // WAL wait runs after release.
  auto remove_impl(std::span<const std::byte> full_key) -> bool {
    return vmemkv::with_key_stripe(
        lock_stripe_fn(),
        [this](Wal::PendingRecord *pending) { wal_.await_durable(pending); },
        full_key,
        [&]() -> vmemkv::StripeResult {
          const auto old = t1_.get_with_hash(full_key);
          if (old.payload_bits == vmemkv::STORE_NOT_FOUND) {
            return {};
          }
          if (t1_.put(full_key, vmemkv::STORE_NOT_FOUND) == T1IndexT::PutResult::Applied) {
            t2_own_.note_replace(old.payload_bits, old.raw_hash, /*new_is_offset=*/false, 0);
            return {true, wal_.reserve_delete(full_key)};
          }
          return {};
        });
  }

  // Bulk-loads `count` entries with no WAL (faster than insert_impl(); a post-crash loss
  // stands unless the caller checkpoints after). Ordinary T1-only reorganizes still trigger
  // past the soft threshold. Paces against cgroup memory pressure (no-op without a limit).
  // Not safe concurrently with other writers.
  template <typename KeyFn, typename ValueFn>
  void bulk_load_impl(std::size_t count, KeyFn &&make_key, ValueFn &&make_value) {
    for (std::size_t index = 0; index < count; ++index) {
      const std::string key = make_key(index);
      const std::string value = make_value(index);
      vmemkv::write_entry_lockfree<ConfigT>(
          t1_,
          t2_,
          t2_own_,
          std::span<const std::byte>(reinterpret_cast<const std::byte *>(key.data()), key.size()),
          std::span<const std::byte>(reinterpret_cast<const std::byte *>(value.data()), value.size()),
          std::nullopt);
      bulk_load_throttle_.maybe_throttle(index);
    }
  }

  // Range scan invoking callback(key, val) per live match. Concurrently readable: t1_.scan()
  // holds one self-consistent epoch-guarded snapshot for the whole call, including callbacks.
  template <typename Callback>
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  auto scan_impl(std::span<const std::byte> lower_bound,
                 std::span<const std::byte> upper_bound,
                 Callback callback) const -> size_t {
    size_t total_count = 0;

    // Bounds are constant per call: check once whether prefix containment already implies
    // full-key containment, skipping the per-entry recheck when it does.
    const bool bounds_clean = vmemkv::scan_bounds_clean(lower_bound, upper_bound);

    t1_.scan(
        lower_bound,
        upper_bound,
        // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
        [&](std::span<const std::byte> index_key, uint64_t payload, uint64_t hash) -> ScanControl {
          if (payload == vmemkv::STORE_NOT_FOUND) {
            return ScanControl::Continue;
          }

          ScanControl inline_control = ScanControl::Continue;
          if (vmemkv::try_read_inline_value<ConfigT>(payload, hash, [&](std::span<const std::byte> value) {
                const size_t key_len = vmemkv::inline_key_len(index_key);
                if (!bounds_clean) {
                  const std::span<const std::byte> key_view(index_key.data(), key_len);
                  if (!vmemkv::bytes_in_range(key_view, lower_bound, upper_bound)) {
                    return;
                  }
                }
                inline_control = scan_detail::invoke_callback(
                    callback, std::span<const std::byte>(index_key.data(), key_len), value);
                ++total_count;
              })) {
            return inline_control;
          }

          const T2Memory *mem = t2_.get_memory();

          // Immutable-region fast path, no seqlock (immutable bytes: live spans straight in).
          if (const auto imm_record =
                  vmemkv::try_read_immutable_record<ConfigT>(mem, payload, vmemkv::ImmutableReader::kScan, nullptr);
              imm_record.has_value()) {
            if (!vmemkv::bytes_in_range(imm_record->key, lower_bound, upper_bound)) {
              return ScanControl::Continue;
            }
            ++total_count;
            return scan_detail::invoke_callback(callback, imm_record->key, imm_record->value);
          }

          // t2_.at() inside read_t2_record_seqlock() (as AtFunc), matching get_impl().
          //
          // Torn-read fix: copy into an owned buffer and return, never callback from inside
          // (see get_impl()). thread_local/static: concurrent-safe, no per-call reallocation.
          // Ownership: see tl_get_buf above; read_path.hpp owns tl_pread_scratch.
          thread_local static std::vector<std::byte> tl_scan_key_buf;
          thread_local static std::vector<std::byte> tl_scan_value_buf;
          const bool in_range = vmemkv::read_t2_record_seqlock(
              [&]() -> T2RecordView { return T2FlatFile::at(payload & kOffsetMask, mem); },
              [&](const T2RecordView &record) -> bool {
                if (!vmemkv::bytes_in_range(record.key, lower_bound, upper_bound)) {
                  return false;
                }
                tl_scan_key_buf.assign(record.key.begin(), record.key.end());
                tl_scan_value_buf.assign(record.value.begin(), record.value.end());
                return true;
              });
          if (in_range) {
            ++total_count;
            return scan_detail::invoke_callback(
                callback, std::span<const std::byte>(tl_scan_key_buf), std::span<const std::byte>(tl_scan_value_buf));
          }
          return ScanControl::Continue;
        });

    return total_count;
  }

 private:
  auto run_compaction_cycle() -> bool {
    return compaction_detail::compact_cycle<ConfigT>(compact_, t1_, t2_, t2_own_, t2_path(), lock_stripe_fn());
  }

  auto lock_stripe_fn() {
    return [this](std::span<const std::byte> key) -> std::mutex & { return key_mutex(key); };
  }

  auto t2_path() const -> std::filesystem::path { return t2_.path(); }

  // Phase 4 of construction: start the unified T2 maintenance worker after recovery completes.
  void spawn_background_workers() { t2_worker_ = std::jthread(&VMemKVImpl::t2_maintenance_loop, this); }

  // Manifest-named live T2 bytes for T2FlatFile's adopt-constructor. Static: runs in the
  // constructor's initializer list before t2_ exists. Delegates to recovery_detail.
  static auto adopted_t2_bytes_used(const std::filesystem::path &t2_path) -> std::optional<uint64_t> {
    return recovery_detail::adopted_t2_bytes_used(t2_path);
  }

  // Fast-boot (docs/specification/low_level_design.md 5.3): with a committed manifest, adopt its T1 sorted region
  // to match the T2 data the constructor already adopted. No manifest: untouched (full replay
  // from LSN 1 is already correct).
  //
  // Failures past a valid manifest propagate (no silent fresh-start fallback): with rotation the
  // WAL holds only the post-checkpoint tail, so the manifest is the only route to everything
  // before checkpoint_lsn -- falling back would quietly lose records.
  void load_checkpoint_if_present(const std::filesystem::path &t2_path) {
    const auto manifest = vmemkv::read_manifest(vmemkv::derive_manifest_path(t2_path));
    if (!manifest.has_value()) {
      return;
    }

    const std::filesystem::path t1_chk_dir = vmemkv::derive_t1_chk_path(t2_path);
    if (manifest->t1_chunks.empty() || !std::filesystem::is_directory(t1_chk_dir)) {
      throw std::runtime_error("sharded t1 checkpoint directory missing or empty for valid manifest");
    }
    vmemkv::cleanup_stale_t1_chunks(t1_chk_dir, manifest->t1_chunks);

    // O(N) memcpy-shaped conversion (on-disk order -> EntrySnapshot order), no hashing or
    // per-key insertion -- what makes fast boot fast (docs/specification/low_level_design.md 5.4), done once per
    // shard. Runs before t1_.start_workers() (see the constructor's own comment), so this is the
    // single-threaded window load_from_checkpoint()'s own contract requires.
    using EntrySnapshot = typename T1IndexT::EntrySnapshot;
    using Key = typename T1IndexT::Key;
    std::vector<std::vector<EntrySnapshot>> per_shard_entries;
    std::vector<Key> boundaries;
    std::vector<bool> shard_local;

    for (const auto &chunk_meta : manifest->t1_chunks) {
      vmemkv::ShardedT1CheckpointFile chunk_file(vmemkv::derive_t1_chunk_path(t1_chk_dir, chunk_meta.chunk_id));
      const auto chunk_bounds = chunk_file.boundaries();
      boundaries.insert(boundaries.end(), chunk_bounds.begin(), chunk_bounds.end());
      if (chunk_meta.has_upper_bound != 0) {
        boundaries.push_back(chunk_meta.upper_bound);
      }

      for (size_t shard_index = 0; shard_index < chunk_file.shard_count(); ++shard_index) {
        std::vector<EntrySnapshot> entries;
        const auto on_disk_entries = chunk_file.shard_entries(shard_index);
        entries.reserve(on_disk_entries.size());
        for (const auto &on_disk : on_disk_entries) {
          const EntrySnapshot entry{on_disk.key_prefix, on_disk.payload_bits, on_disk.hash};
          entries.push_back(entry);
          if (entry.payload_bits != vmemkv::STORE_NOT_FOUND && !t1_detail::is_inline(entry.hash)) {
            t2_own_.note_replace(vmemkv::STORE_NOT_FOUND, 0, /*new_is_offset=*/true, entry.payload_bits);
          }
        }
        shard_local.push_back(compaction_detail::shard_pages_local<EntrySnapshot>(entries, t2_.get_memory()));
        per_shard_entries.push_back(std::move(entries));
      }
    }

    t1_.load_from_checkpoint(boundaries, per_shard_entries);
    checkpoint_.t1_chunks = manifest->t1_chunks;
    checkpoint_.next_t1_chunk_id = manifest->next_t1_chunk_id;
    for (size_t shard_index = 0; shard_index < per_shard_entries.size(); ++shard_index) {
      t1_.mark_t2_shard_state(shard_index, !shard_local[shard_index]);
    }
    const uint64_t used = manifest->t2_bytes_used;
    const uint64_t used_segs = (used + ConfigT::kSegmentBytes - 1) / ConfigT::kSegmentBytes;
    for (uint64_t seg = 0; seg < used_segs; ++seg) {
      const uint64_t seg_used = std::min(ConfigT::kSegmentBytes, used - seg * ConfigT::kSegmentBytes);
      if (t2_own_.seg_live(seg) * 2 < seg_used && t2_own_.seg_live(seg) > 0) {
        t2_own_.mark_segment_dirty(seg);
      }
    }
  }

  // Replays wal_ into T1 (and via the append path, T2) -- full history or post-checkpoint
  // tail, transparently. Runs before t2_worker_ starts (constructor order), so no
  // VMemKVImpl-level thread can touch T2 maintenance state yet. t1_'s own pool already runs
  // (started post-checkpoint-load, pre-replay): replay leans on it for AppendRegionFull
  // drainage like live traffic does.
  void recover_from_wal() {
    wal_.replay([&](vmemkv::WalRecordType type,
                    std::span<const std::byte> key,
                    std::span<const std::byte> value,
                    uint64_t /*lsn*/) {
      switch (type) {
        case vmemkv::WalRecordType::Insert:
        case vmemkv::WalRecordType::Update:
          // put() overwrites in place: Insert and Update replay identically, last-writer-wins.
          vmemkv::write_entry_lockfree<ConfigT>(t1_, t2_, t2_own_, key, value, std::nullopt);
          break;
        case vmemkv::WalRecordType::Delete: {
          const auto old = t1_.get_with_hash(key);
          if (t1_.put(key, vmemkv::STORE_NOT_FOUND) == T1IndexT::PutResult::Applied) {
            t2_own_.note_replace(old.payload_bits, old.raw_hash, /*new_is_offset=*/false, 0);
          }
          break;
        }
      }
    });
  }

  // get_impl()/scan_impl()'s inline stack buffers are sized by T1Index's own cap -- the single
  // source of truth, so T1Index can never accept a payload wider than these buffers.
  static_assert(t1_detail::kInlineValueByteCount == sizeof(uint64_t));
  auto maybe_run_auto_checkpoint() -> bool {
    if (checkpoint_detail::auto_reorg_suppressed() || !checkpoint_detail::checkpoint_due<ConfigT>(wal_, t2_own_)) {
      return false;
    }
    auto guard = SingleFlightGuard::try_acquire(checkpoint_.running);
    if (!guard.holds) {
      return false;
    }
    try {
      checkpoint_internal();
      // NOLINTNEXTLINE(bugprone-empty-catch)
    } catch (...) {
      // Background loop; retry next poll without terminating worker.
    }
    return true;
  }

  void maybe_run_auto_compaction(std::chrono::steady_clock::time_point &next_compact_at, uint64_t &observed_chk_count) {
    const uint64_t chk_now = checkpoint_.stats.checkpoint_count.load(std::memory_order_acquire);
    if (chk_now != observed_chk_count) {
      observed_chk_count = chk_now;
      next_compact_at = std::chrono::steady_clock::now();
    }
    const auto now = std::chrono::steady_clock::now();
    if (vmemkv::auto_compact_suppressed() || now < next_compact_at || !has_compaction_work()) {
      return;
    }
    try {
      run_compaction_cycle();
      // NOLINTNEXTLINE(bugprone-empty-catch)
    } catch (...) {
      // Background loop; retry next cadence without terminating worker.
    }
    next_compact_at = has_garbage_segments() ? std::chrono::steady_clock::now()
                                             : std::chrono::steady_clock::now() + kCompactionPoll1s;
  }

  // Unified T2 storage maintenance loop: checkpoints (10ms poll when WAL >= 64MiB or punch
  // backlog >= cap) plus compaction and hole punching (1s idle cadence; zero wait while
  // frozen garbage segments remain or right after any checkpoint advances).
  void t2_maintenance_loop(std::stop_token stop_token) {
    auto next_compact_at = std::chrono::steady_clock::now();
    uint64_t observed_chk_count = checkpoint_.stats.checkpoint_count.load(std::memory_order_acquire);
    poll_worker_until_stop(stop_token, kDefaultPoll10ms, [&] {
      if (recovering_) {
        return false;
      }
      maybe_run_auto_checkpoint();
      if (stop_token.stop_requested()) {
        return true;
      }
      maybe_run_auto_compaction(next_compact_at, observed_chk_count);
      return has_garbage_segments() ||
             (!checkpoint_detail::auto_reorg_suppressed() && checkpoint_detail::checkpoint_due<ConfigT>(wal_, t2_own_));
    });
  }

  // 256 key-hash stripes: no contention without per-key lock allocation. Cacheline-aligned
  // against false sharing.
  static constexpr size_t kKeyStripeCount = 256;
  mutable std::array<AlignedMutex, kKeyStripeCount> write_stripes_;

  auto key_mutex(std::span<const std::byte> key) const noexcept -> std::mutex & { return stripe_state(key).mu; }

  auto stripe_state(std::span<const std::byte> key) const noexcept -> AlignedMutex & {
    const uint64_t hash = t1_detail::hash_full_key(key);
    return write_stripes_[hash & (kKeyStripeCount - 1)];
  }

  T1IndexT t1_;
  vmemkv::T2FlatFile t2_;
  vmemkv::Wal wal_;
  // Paces bulk_load_impl() against cgroup memory pressure (count survives here for stats).
  vmemkv::CgroupMemoryThrottle bulk_load_throttle_;
  // T2 checkpoint fuzzy watermark and in-place-update barrier.
  T2Ownership<ConfigT> t2_own_;
  // Dirty-shard repack state and delayed segment-punch request.
  mutable CompactionState compact_;
  // Checkpoint single-flight state and phase stats.
  CheckpointState checkpoint_;

  bool recovering_ = false;  // True only during the constructor's initial WAL replay.

  // Declared last: destruction joins the worker before its state. jthread joins on destruct;
  // loop polls stop_token, bounding shutdown latency to one interval.
  std::jthread t2_worker_;
};

using vmemkv = VMemKVImpl<vmemkv::Config<>>;

}  // namespace vmemkv
