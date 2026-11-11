// leanstore_store.hpp — Thin LeanStore rival wrapper exposing byte-span APIs.
//
// LeanStore (pointer-swizzling buffer-pool engine, pinned by commit in CMakeLists) speaks a
// typed, transactional API: every op runs inside a TX on one of its own worker threads, so this
// wrapper dispatches each call via CRManager::scheduleJobSync() and retries aborts internally.
//
// Durability: WAL on with per-group fdatasync, plus every mutating call waits for a covering
// group flush (via the worker's WAL cursor) -- as close to vmemkv's per-write fsync and
// RocksDB's sync=true as LeanStore allows.
//
// Value ceiling: BTreeVI lengths are u16 (65535 max); the 64KB corpus is excluded for this
// backend (see the benchmark matrix).
//
// Threading: concurrent writers serialize only on real conflicts (abort + internal retry).
// Worker-slot dispatch is per-slot mutexed: same-slot callers serialize there instead of
// overwriting each other's job.

#pragma once

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>
#include <vmemkv/scan_control.hpp>

#ifdef ENABLE_LEANSTORE
#include <leanstore/KVInterface.hpp>
#include <leanstore/LeanStore.hpp>
#include <leanstore/concurrency-recovery/Worker.hpp>
#include <leanstore/storage/buffer-manager/DTRegistry.hpp>
#include <leanstore/utils/JumpMU.hpp>
#endif

#include <vmemkv/core/env.hpp>
#include <vmemkv/core/io.hpp>
#include <vmemkv/core/spin_backoff.hpp>

#include "rival_common.hpp"
#include "rival_store_disabled_stub.hpp"

namespace vmemkv::rivals {}  // namespace vmemkv::rivals

class LeanStoreStore {
 public:
  static constexpr bool kIsEnabled =
#ifdef ENABLE_LEANSTORE
      true;
#else
      false;
#endif
  static constexpr bool kIsRival = true;
  static void checkpoint() noexcept {}

#ifdef ENABLE_LEANSTORE
  // Worker threads per instance; sized to cover the harness's maximum benchmark thread count.
  static constexpr uint64_t kWorkerThreads = 32;
  // BTreeVI key/value lengths are u16; values above this cannot be stored.
  static constexpr std::size_t kMaxValueBytes = 65535;

  static auto checked_len(std::size_t n) -> ::u16 {
    if (n > kMaxValueBytes) {
      throw std::runtime_error("LeanStore cannot store values over 65535 bytes (64KB corpus excluded)");
    }
    return static_cast<::u16>(n);
  }

  // BTreeVI orders keys by raw memcmp, mirroring LMDBStore::compare_to_bound exactly.
  static auto compare_to_bound(const ::u8 *key, ::u16 key_len, std::span<const std::byte> upper_bound) noexcept -> int {
    return vmemkv::bytes_compare_3way(std::span<const std::byte>(reinterpret_cast<const std::byte *>(key), key_len),
                                      upper_bound);
  }

  // Aborts the current worker TX on ABORT_TX; abortTX() longjmps, so this stays trivial.
  static void throw_on_abort(leanstore::OP_RESULT res) {
    if (res == leanstore::OP_RESULT::ABORT_TX) {
      leanstore::cr::Worker::my().abortTX();
    }
  }

  // Exact-match existence probe with optional value capture. Scan-based rather than
  // lookup()-based (see get_impl): only the scan path skips removed entries. ABORT_TX
  // funnels through throw_on_abort() for the caller's retry loop.
  template <typename ValueCb>
  static void scan_exact(leanstore::KVInterface &btree,
                         std::span<const std::byte> key,
                         ValueCb &&on_value,
                         bool &found) {
    ::u8 *k = const_cast<::u8 *>(reinterpret_cast<const ::u8 *>(key.data()));
    const ::u16 klen = checked_len(key.size());
    const auto res = btree.scanAsc(
        k,
        klen,
        [&](const ::u8 *sk, ::u16 sklen, const ::u8 *sv, ::u16 svlen) {
          if (sklen != klen || std::memcmp(sk, key.data(), klen) != 0) {
            return false;
          }
          on_value(sv, svlen);
          found = true;
          return false;
        },
        [] {});
    throw_on_abort(res);
  }

  // Exact-match liveness probe used by insert/update/remove.
  static auto require_live(leanstore::KVInterface &btree, std::span<const std::byte> key) -> bool {
    bool seen = false;
    scan_exact(btree, key, [](const ::u8 *, ::u16) {}, seen);
    return seen;
  }

  // Opens a fresh database at a unique subpath, mirroring LMDBStore's per-instance uniquing.
  explicit LeanStoreStore(std::string path) {
    const std::string stem = vmemkv::rivals::make_unique_instance_path(std::move(path), ".leanstore");
    open(stem, OpenMode::Fresh);
  }

  ~LeanStoreStore() {
    close();
    std::error_code ignored;
    std::filesystem::remove(ssd_path_, ignored);
    std::filesystem::remove(json_path_, ignored);
  }

  LeanStoreStore(const LeanStoreStore &) = delete;
  auto operator=(const LeanStoreStore &) -> LeanStoreStore & = delete;

  // Shared tag selecting the clone-from-master constructor (see rival_common.hpp).
  using CloneFromMasterTag = vmemkv::rivals::CloneFromMasterTag;

  // Builds `master_path` once via bulk_load, then clones it with a plain file copy (never a
  // hardlink: pages are written in place).
  template <typename KeyFn, typename ValueFn>
  LeanStoreStore(CloneFromMasterTag /*tag*/,
                 const std::string &master_path,
                 std::size_t key_count,
                 KeyFn &&make_key,
                 ValueFn &&make_value) {
    ensure_master_built(master_path, key_count, std::forward<KeyFn>(make_key), std::forward<ValueFn>(make_value));
    // Fixed path, not an ever-incrementing counter: every clone of the same master is
    // interchangeable, and a fixed name keeps at most one clone generation on disk.
    const std::string stem = master_path + "_clone.leanstore";
    clone_from(master_path, stem);
    open(stem, OpenMode::Recover);
  }

  // No-op: LeanStore reclaims via its buffer manager/page recycling internally; there is no
  // manual reorganize/compaction concept during normal operation.
  void reorganize() {}

  // ─── Low-level byte-span APIs (called by StoreAdapter) ───────────────────────

  template <typename Callback>
  auto get_impl(std::span<const std::byte> key, Callback callback) const -> bool {
    // Liveness via scan_exact plus value via lookup, both inside one TX. The value is copied
    // into heap state before commit.
    struct GetState {
      std::vector<std::byte> value;
      bool found = false;
    };
    auto state = run_on_worker<GetState>(
        [&](leanstore::KVInterface &btree) {
          GetState out;
          bool live = false;
          scan_exact(btree, key, [&](const ::u8 *, ::u16) {}, live);
          if (!live) {
            return out;
          }
          ::u8 *k = const_cast<::u8 *>(reinterpret_cast<const ::u8 *>(key.data()));
          const auto res = btree.lookup(k, checked_len(key.size()), [&](const ::u8 *payload, ::u16 len) {
            out.value.assign(reinterpret_cast<const std::byte *>(payload),
                             reinterpret_cast<const std::byte *>(payload) + len);
          });
          throw_on_abort(res);
          // A concurrent remover may have deleted the key between the probe and the lookup;
          // linearize the delete first.
          out.found = (res == leanstore::OP_RESULT::OK);
          return out;
        },
        /*read_only=*/true);
    if (state.found) {
      callback(std::span<const std::byte>(state.value.data(), state.value.size()));
    }
    return state.found;
  }

  auto insert_impl(std::span<const std::byte> key, std::span<const std::byte> value) -> bool {
    checked_len(value.size());
    // BTreeVI::insert on an existing key hits an unimplemented path, so existence is checked
    // first within the same TX: a concurrent inserter wins the race as ABORT_TX and this call
    // retries, observing the key the second time.
    return run_on_worker<bool>([&](leanstore::KVInterface &btree) {
      ::u8 *k = const_cast<::u8 *>(reinterpret_cast<const ::u8 *>(key.data()));
      ::u8 *v = const_cast<::u8 *>(reinterpret_cast<const ::u8 *>(value.data()));
      if (require_live(btree, key)) {
        return false;
      }
      const auto res = btree.insert(k, checked_len(key.size()), v, checked_len(value.size()));
      throw_on_abort(res);
      return res == leanstore::OP_RESULT::OK;
    });
  }

  auto update_impl(std::span<const std::byte> key, std::span<const std::byte> value) -> bool {
    checked_len(value.size());
    // Same-size in-place update: remove + insert is unimplemented in this backend version
    // ("Implement inserts after remove cases" TODO in BTreeVI), so size-changing updates have
    // no correct path. The harness always updates with the key's own index-derived size, hence
    // same-size; anything else fails fast instead of hitting ensure(false) inside the engine.
    return run_on_worker<bool>([&](leanstore::KVInterface &btree) {
      ::u8 *k = const_cast<::u8 *>(reinterpret_cast<const ::u8 *>(key.data()));
      // Liveness via scan (lookup is tombstone-blind); the exact length via lookup (the scan
      // fast path overstates chained value lengths). Both in one TX.
      ::u16 old_len = 0;
      if (!require_live(btree, key)) {
        return false;
      }
      const auto lres = btree.lookup(k, checked_len(key.size()), [&](const ::u8 *, ::u16 len) { old_len = len; });
      throw_on_abort(lres);
      if (lres != leanstore::OP_RESULT::OK) {
        return false;
      }
      if (old_len != value.size()) {
        throw std::runtime_error("LeanStore update with changed value size is unsupported by the engine");
      }
      ::u8 desc_buf[sizeof(leanstore::UpdateSameSizeInPlaceDescriptor) +
                    sizeof(leanstore::UpdateSameSizeInPlaceDescriptor::Slot)];
      auto &desc = *reinterpret_cast<leanstore::UpdateSameSizeInPlaceDescriptor *>(desc_buf);
      desc.count = 1;
      desc.slots[0].offset = 0;
      desc.slots[0].length = checked_len(value.size());
      const auto res = btree.updateSameSizeInPlace(
          k,
          checked_len(key.size()),
          [&](::u8 *payload, ::u16) { std::memcpy(payload, value.data(), value.size()); },
          desc);
      throw_on_abort(res);
      return res == leanstore::OP_RESULT::OK;
    });
  }

  auto remove_impl(std::span<const std::byte> key) -> bool {
    // Existence is checked first within the same TX: removing an already-removed key hits an
    // engine ensure() instead of returning NOT_FOUND.
    return run_on_worker<bool>([&](leanstore::KVInterface &btree) {
      ::u8 *k = const_cast<::u8 *>(reinterpret_cast<const ::u8 *>(key.data()));
      if (!require_live(btree, key)) {
        return false;
      }
      const auto res = btree.remove(k, checked_len(key.size()));
      throw_on_abort(res);
      return res == leanstore::OP_RESULT::OK;
    });
  }

  template <typename KeyFn, typename ValueFn>
  void bulk_load_impl(std::size_t key_count, KeyFn &&make_key, ValueFn &&make_value) {
    if (key_count == 0) {
      return;
    }
    // Bulk load goes through regular transactions with a commit every batch; single-threaded
    // ascending order, probing before each insert.
    constexpr std::size_t kBatchKeys = 10000;
    std::atomic<bool> failed{false};
    for (std::size_t base = 0; base < key_count && !failed.load(std::memory_order_relaxed);) {
      const std::size_t chunk_end = std::min(key_count, base + kBatchKeys);
      run_on_worker<int>(
          [&](leanstore::KVInterface &btree) {
            for (std::size_t index = base; index < chunk_end; ++index) {
              const std::string key = make_key(index);
              const std::string value = make_value(index);
              if (value.size() > kMaxValueBytes) {
                failed.store(true, std::memory_order_relaxed);
                return 0;
              }
              bool present = false;
              scan_exact(
                  btree,
                  std::span<const std::byte>(reinterpret_cast<const std::byte *>(key.data()), key.size()),
                  [&](const ::u8 *, ::u16) {},
                  present);
              if (present) {
                continue;
              }
              const auto res = btree.insert(reinterpret_cast<::u8 *>(const_cast<char *>(key.data())),
                                            checked_len(key.size()),
                                            reinterpret_cast<::u8 *>(const_cast<char *>(value.data())),
                                            checked_len(value.size()));
              throw_on_abort(res);
            }
            return 0;
          },
          /*read_only=*/false,
          /*slot_hint=*/0);
      base = chunk_end;
    }
    if (failed.load()) {
      throw std::runtime_error("LeanStore bulk load failed (value over u16 ceiling or backend error)");
    }
  }

  template <typename Cb>
  [[nodiscard]] auto scan_impl(std::span<const std::byte> lower_bound,
                               std::span<const std::byte> upper_bound,
                               Cb callback) const -> size_t {
    // Values resolve through lookup, not the scan callback (see get_impl). Two phases in one
    // TX: collect in-range keys first, then look each up. The lookup must not nest inside the
    // scan callback (latch-order risk across leaves); sequential phases hold no iterator
    // latch while looking up.
    struct ScanState {
      std::vector<std::string> keys;
      size_t count = 0;
    };
    auto state = run_on_worker<ScanState>(
        [&](leanstore::KVInterface &btree) {
          ScanState out;
          ::u8 *lo = const_cast<::u8 *>(reinterpret_cast<const ::u8 *>(lower_bound.data()));
          const auto res = btree.scanAsc(
              lo,
              checked_len(lower_bound.size()),
              [&](const ::u8 *sk, ::u16 sklen, const ::u8 *, ::u16) {
                if (compare_to_bound(sk, sklen, upper_bound) >= 0) {
                  return false;
                }
                out.keys.emplace_back(reinterpret_cast<const char *>(sk), sklen);
                return true;
              },
              [] {});
          throw_on_abort(res);
          std::vector<std::byte> value_buf;
          for (const std::string &key : out.keys) {
            value_buf.clear();
            bool live = false;
            ::u8 *lk = const_cast<::u8 *>(reinterpret_cast<const ::u8 *>(key.data()));
            const auto lres = btree.lookup(lk, checked_len(key.size()), [&](const ::u8 *payload, ::u16 len) {
              value_buf.assign(reinterpret_cast<const std::byte *>(payload),
                               reinterpret_cast<const std::byte *>(payload) + len);
              live = true;
            });
            throw_on_abort(lres);
            if (!live) {
              continue;
            }
            const vmemkv::ScanControl control = vmemkv::scan_detail::invoke_callback(
                callback,
                std::span<const std::byte>(reinterpret_cast<const std::byte *>(key.data()), key.size()),
                std::span<const std::byte>(value_buf.data(), value_buf.size()));
            ++out.count;
            if (control == vmemkv::ScanControl::Stop) {
              break;
            }
          }
          return out;
        },
        /*read_only=*/true);
    return state.count;
  }

 private:
  // ─── Instance lifecycle ───

  static auto dram_gib() -> double {
    if (const std::string_view env = vmemkv::getenv_view("LEANSTORE_DRAM_GIB"); !env.empty()) {
      double value = 0;
      if (std::from_chars(env.data(), env.data() + env.size(), value).ec == std::errc{}) {
        return value;
      }
      return std::strtod(std::string(env).c_str(), nullptr);
    }
    // Mirror the harness's memory posture: constrained under LTM, generous in-memory.
    // VMEMKV_BENCH_LTM=1 is set by the benchmark harness for LTM scenarios.
    if (!vmemkv::getenv_view("VMEMKV_BENCH_LTM").empty()) {
      return 1.0;
    }
    return 16.0;
  }

  static auto worker_threads() -> uint32_t {
    // Clamped to the slot-mutex array bound; fewer workers only (bisect/debug knob).
    return getenv_u32(
        "LEANSTORE_WORKER_THREADS", static_cast<uint32_t>(kWorkerThreads), static_cast<uint32_t>(kWorkerThreads));
  }

  static auto getenv_u32(const char *name, uint32_t dflt, uint32_t clamp_max) -> uint32_t {
    // min() in uint64 before narrowing: huge inputs clamp, never truncate.
    return static_cast<uint32_t>(std::min<uint64_t>(vmemkv::getenv_uint64(name, dflt), clamp_max));
  }

  // LeanStore derives recover/persist from non-default file paths at construction
  // (recover_file != default forces recover), so every flag is reset explicitly on each open:
  // Benchmark and test runs construct instances sequentially while sharing these globals.
  static void reset_flags(const std::string &ssd) {
    FLAGS_ssd_path = ssd;
    FLAGS_trunc = false;
    FLAGS_persist = false;
    FLAGS_persist_file = "./leanstore.json";
    FLAGS_recover = false;
    FLAGS_recover_file = "./leanstore.json";
    FLAGS_wal = true;
    FLAGS_wal_fsync = true;
    FLAGS_wal_pwrite = true;
    // WAL region grows downward from this offset; corpora reach ~17GB, past the 10GiB
    // default, which would overwrite data pages. Files stay sparse: unwritten ranges
    // between the data image and this offset consume no blocks.
    FLAGS_wal_offset_gib = 64;
    FLAGS_worker_threads = worker_threads();
    FLAGS_dram_gib = dram_gib();
    FLAGS_pin_threads = false;
    FLAGS_cpu_counters = false;
    // Unset env keeps the current value.
    FLAGS_pp_threads = getenv_u32("LEANSTORE_PP_THREADS", FLAGS_pp_threads, UINT32_MAX);
  }

  // Blocks until every worker has executed one job, proving all workers are past startup.
  void warmup_workers() {
    for (uint64_t t = 0; t < worker_threads(); ++t) {
      db_->getCRManager().scheduleJobSync(t, [] {});
    }
  }

  // Drops all datastructure-instance registrations. LeanStore never unregisters an
  // instance, so every destroyed instance leaves a dangling entry behind. A later instance
  // recovering from a clone of the same master re-registers the same persisted id, and the
  // registry's insert keeps the first (dangling) entry, so recovery would deserialize into
  // freed memory. At most one instance is alive at any open: benchmark cells serialize
  // through a single active holder and tests construct stores sequentially. Type
  // registrations are untouched.
  static void prune_stale_registrations() {
    std::lock_guard guard(leanstore::storage::DTRegistry::global_dt_registry.mutex);
    leanstore::storage::DTRegistry::global_dt_registry.dt_instances_ht.clear();
  }

  enum class OpenMode { Fresh, Recover };

  void open(const std::string &stem, OpenMode mode) {
    prune_stale_registrations();
    ssd_path_ = stem;
    json_path_ = stem + ".json";
    if (mode == OpenMode::Fresh) {
      std::error_code ignored;
      std::filesystem::remove(ssd_path_, ignored);
      std::filesystem::remove(json_path_, ignored);
    }
    reset_flags(ssd_path_);
    if (mode == OpenMode::Fresh) {
      FLAGS_trunc = true;
    } else {
      FLAGS_recover = true;
      FLAGS_recover_file = json_path_;
    }
    db_ = std::make_unique<leanstore::LeanStore>();
    if (mode == OpenMode::Fresh) {
      // Table registration runs on a worker: B-tree creation requires worker-local state.
      db_->getCRManager().scheduleJobSync(
          0, [&] { table_ = &db_->registerBTreeVI("kv", {.enable_wal = true, .use_bulk_insert = false}); });
    } else {
      db_->getCRManager().scheduleJobSync(0, [&] { table_ = &db_->retrieveBTreeVI("kv"); });
    }
    warmup_workers();
  }

  void close() {
    table_ = nullptr;
    db_.reset();
  }

  // Builds a fresh master at a ".building" sibling path, closed (persist flushes all buffer
  // frames, so the file copy below is self-consistent) before anything copies it, so a crash
  // mid-build never leaves a partial master for a later call to trust. WAL is disabled for
  // the build itself, like the other engines' non-durable bulk loaders.
  template <typename KeyFn, typename ValueFn>
  static void ensure_master_built(const std::string &master_path,
                                  std::size_t key_count,
                                  KeyFn &&make_key,
                                  ValueFn &&make_value) {
    const std::string master_json = master_path + ".json";
    if (std::filesystem::exists(master_path) && std::filesystem::exists(master_json)) {
      return;
    }
    prune_stale_registrations();
    const std::string building_ssd = vmemkv::rivals::building_path(master_path);
    const std::string building_json = building_ssd + ".json";
    std::error_code ignored;
    std::filesystem::remove(building_ssd, ignored);
    std::filesystem::remove(building_json, ignored);

    LeanStoreStore tmp;
    tmp.ssd_path_ = building_ssd;
    tmp.json_path_ = building_json;
    reset_flags(building_ssd);
    FLAGS_trunc = true;
    FLAGS_persist = true;
    FLAGS_persist_file = building_json;
    tmp.db_ = std::make_unique<leanstore::LeanStore>();
    tmp.db_->getCRManager().scheduleJobSync(
        0, [&] { tmp.table_ = &tmp.db_->registerBTreeVI("kv", {.enable_wal = false, .use_bulk_insert = false}); });
    tmp.warmup_workers();
    tmp.bulk_load_impl(key_count, std::forward<KeyFn>(make_key), std::forward<ValueFn>(make_value));
    tmp.close();
    prune_stale_registrations();

    std::filesystem::remove(master_path, ignored);
    std::filesystem::remove(master_json, ignored);
    vmemkv::rivals::atomic_rename(building_ssd, master_path, "LeanStore master build rename failed");
    vmemkv::rivals::atomic_rename(building_json, master_json, "LeanStore master json rename failed");
  }

  // Copies a database image preserving holes: with the WAL region parked high above
  // the data image the files are sparse, and a plain copy would materialize tens of
  // gigabytes of zeros per clone. Falls back to a full copy where SEEK_DATA is unsupported.
  static void copy_sparse(const std::string &source, const std::string &dest) {
    vmemkv::FdGuard src{::open(source.c_str(), O_RDONLY)};
    if (src.fd < 0) {
      throw std::runtime_error("LeanStore clone open failed: " + source);
    }
    const off_t total = ::lseek(src.fd, 0, SEEK_END);
    if (total < 0) {
      throw std::runtime_error("LeanStore clone stat failed: " + source);
    }
    vmemkv::FdGuard dst{::open(dest.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666)};
    if (dst.fd < 0) {
      throw std::runtime_error("LeanStore clone create failed: " + dest);
    }
    std::string scratch(1 << 20, '\0');
    const off_t probe = ::lseek(src.fd, 0, SEEK_DATA);
    if (probe < 0 && errno == EINVAL) {
      copy_range(src.fd, dst.fd, 0, total, scratch);
    } else {
      for (off_t off = 0; off < total;) {
        const off_t data_at = (off == 0 && probe >= 0) ? probe : ::lseek(src.fd, off, SEEK_DATA);
        if (data_at < 0) {
          if (errno == ENXIO) {
            break;
          }
          throw std::runtime_error("LeanStore clone seek failed: " + source);
        }
        off_t hole_at = ::lseek(src.fd, data_at, SEEK_HOLE);
        if (hole_at < 0) {
          throw std::runtime_error("LeanStore clone seek failed: " + source);
        }
        copy_range(src.fd, dst.fd, data_at, std::min(hole_at, total) - data_at, scratch);
        off = std::min(hole_at, total);
      }
    }
    if (::ftruncate(dst.fd, total) != 0) {
      throw std::runtime_error("LeanStore clone truncate failed: " + dest);
    }
  }

  static void copy_range(int src_fd, int dest_fd, off_t offset, off_t count, std::string &buf) {
    while (count > 0) {
      const std::size_t chunk = static_cast<std::size_t>(std::min<off_t>(count, static_cast<off_t>(buf.size())));
      const ssize_t got = ::pread(src_fd, buf.data(), chunk, offset);
      if (got <= 0) {
        throw std::runtime_error("LeanStore clone read failed");
      }
      ssize_t put = 0;
      while (put < got) {
        const ssize_t wrote = ::pwrite(dest_fd, buf.data() + put, static_cast<std::size_t>(got - put), offset + put);
        if (wrote <= 0) {
          throw std::runtime_error("LeanStore clone write failed");
        }
        put += wrote;
      }
      offset += got;
      count -= got;
    }
  }

  static void clone_from(const std::string &source_stem, const std::string &dest_stem) {
    std::error_code ignored;
    std::filesystem::remove(dest_stem, ignored);
    std::filesystem::remove(dest_stem + ".json", ignored);
    copy_sparse(source_stem, dest_stem);
    // Make the clone durable before any timed run starts from it: data and WAL share one
    // file here, so the first timed fdatasync would otherwise flush this whole just-copied
    // image (tens of GB) inside the measurement window.
    vmemkv::FdGuard fd{::open(dest_stem.c_str(), O_RDONLY)};
    if (fd.fd < 0) {
      throw std::runtime_error("LeanStore clone sync open failed: " + dest_stem);
    }
    if (::fsync(fd.fd) != 0) {
      throw std::runtime_error("LeanStore clone sync failed: " + dest_stem);
    }
    std::filesystem::copy_file(source_stem + ".json", dest_stem + ".json", ignored);
    if (ignored) {
      throw std::runtime_error("LeanStore clone json copy failed: " + ignored.message());
    }
  }

  // ─── Worker-dispatched transactions ───

  auto worker_slot() const -> uint64_t {
    return std::hash<std::thread::id>{}(std::this_thread::get_id()) % worker_threads();
  }

  // Blocks until a group-commit round has flushed everything this worker published,
  // witnessed through its WAL consumption cursor: every committer round writes out all
  // published bytes, fdatasyncs, and only then records the new cursor, so observing the
  // cursor reach our post-commit end offset proves a covering fdatasync completed. Only
  // meaningful after a mutating transaction ran on that worker; read-only transactions
  // enqueue nothing and skip this. Bounded: a committer that never advances means the
  // engine, not the benchmark, is wedged -- fail loudly instead of billing an infinite hang.
  void await_group_durable(uint64_t slot_hint) const {
    leanstore::cr::Worker &worker = *db_->getCRManager().workers[slot_hint];
    // Frozen from here on: slot_mutex_ serializes all callers sharing this slot, so no
    // other transaction can publish on this worker meanwhile.
    const uint64_t committed_end = worker.logging.wt_to_lw.getSync().wal_written_offset;
    const uint64_t op_seq = op_seq_.fetch_add(1, std::memory_order_relaxed);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (true) {
      const uint64_t gct = worker.logging.wal_gct_cursor.load(std::memory_order_acquire);
      if (gct == committed_end) {
        return;
      }
      if (std::chrono::steady_clock::now() > deadline) {
        std::fprintf(stderr,
                     "await-timeout op=%llu slot=%lu end=%lu gct=%lu workers=%u\n",
                     (unsigned long long)op_seq,
                     (unsigned long)slot_hint,
                     (unsigned long)committed_end,
                     (unsigned long)gct,
                     (unsigned)db_->getCRManager().workers_count);
        throw std::runtime_error("LeanStore group-durability wait timed out");
      }
      // Plain yield, not SpinBackoff: the watched event (group-committer round, tens of
      // microseconds) is far shorter than any sleep quantum, so sleeping would dominate the wait.
      std::this_thread::yield();
    }
  }

  // Runs `body` inside a single TX on a LeanStore worker and returns its value, retrying
  // internally on abort. ABORT_TX from any backend call must funnel through abortTX() (which
  // longjmps into the catch below). Results travel in heap state, never in setjmp-crossing
  // locals.
  template <typename T, typename Body>
  auto run_on_worker(Body &&body, bool read_only, uint64_t slot_hint) const -> T {
    struct Slot {
      T value{};
      bool done = false;
    };
    auto state = std::make_unique<Slot>();
    Slot *s = state.get();
    std::lock_guard<std::mutex> slot_lock(slot_mutex_[slot_hint]);
    db_->getCRManager().scheduleJobSync(slot_hint, [&] {
      while (!s->done) {
        jumpmuTry() {
          leanstore::cr::Worker::my().startTX(
              leanstore::TX_MODE::OLTP, leanstore::TX_ISOLATION_LEVEL::SNAPSHOT_ISOLATION, read_only);
          s->value = body(*table_);
          leanstore::cr::Worker::my().commitTX();
          s->done = true;
        }
        jumpmuCatch() {}
      }
    });
    if (!read_only) {
      await_group_durable(slot_hint);
    }
    return std::move(state->value);
  }

  template <typename T, typename Body>
  auto run_on_worker(Body &&body, bool read_only = false) const -> T {
    return run_on_worker<T>(std::forward<Body>(body), read_only, worker_slot());
  }

  // Private default ctor for ensure_master_built()'s scratch instance.
  LeanStoreStore() = default;

  std::unique_ptr<leanstore::LeanStore> db_;
  leanstore::KVInterface *table_ = nullptr;
  mutable std::mutex slot_mutex_[kWorkerThreads];
  mutable std::atomic<uint64_t> op_seq_{0};
  std::string ssd_path_;
  std::string json_path_;
#else
  VMEMKV_RIVAL_DISABLED_STUB(LeanStoreStore, "LeanStore")
#endif
};
