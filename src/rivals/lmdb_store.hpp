// lmdb_store.hpp — Thin LMDB (Lightning Memory-Mapped Database) wrapper exposing byte-span APIs.
//
// Thread safety: unlimited concurrent readers; exactly one writer transaction at a time.

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vmemkv/scan_control.hpp>

#ifdef ENABLE_LMDB
#include <lmdb.h>
#endif

#include "rival_common.hpp"
#include "rival_store_disabled_stub.hpp"

class LMDBStore {
 public:
  static constexpr bool kIsEnabled =
#ifdef ENABLE_LMDB
      true;
#else
      false;
#endif
  static constexpr bool kIsRival = true;
  static void checkpoint() noexcept {}

#ifdef ENABLE_LMDB
  // Opens a fresh environment at a unique subpath (single-file DB via MDB_NOSUBDIR),
  // mirroring RocksDBStore's per-instance uniquing.
  explicit LMDBStore(std::string path) {
    path_ = vmemkv::rivals::make_unique_instance_path(std::move(path), ".lmdb");
    std::filesystem::remove(path_);
    std::filesystem::remove(path_ + "-lock");

    // MDB_WRITEMAP without MDB_MAPASYNC: every commit synchronously msyncs (same durability
    // contract as vmemkv's per-write fsync and RocksDB's sync=true). bulk_load_impl() toggles
    // MDB_MAPASYNC on for its own duration instead, like the other engines' bulk loaders.
    open_env_and_dbi(path_, MDB_NOSUBDIR | MDB_WRITEMAP, "", env_, dbi_);
  }

  // Closes the environment.
  ~LMDBStore() {
    if (env_ != nullptr) {
      mdb_env_close(env_);
    }
    std::error_code ignored;
    std::filesystem::remove(path_, ignored);
    std::filesystem::remove(path_ + "-lock", ignored);
  }

  LMDBStore(const LMDBStore &) = delete;
  auto operator=(const LMDBStore &) -> LMDBStore & = delete;

  // Shared tag selecting the clone-from-master constructor (see rival_common.hpp).
  using CloneFromMasterTag = vmemkv::rivals::CloneFromMasterTag;

  // Builds `master_path` once via bulk_load, then clones it into this instance's own path.
  // A real copy, never a hardlink: LMDB writes B+Tree pages in place, so hardlinked clones
  // would corrupt each other on first write (unlike RocksDB's immutable SSTs).
  // Fixed path, not an ever-incrementing counter -- see rocksdb_store.hpp's identical
  // constructor / bench_kv.cpp's make_vmemkv_clone_from_checkpoint() for why.
  template <typename KeyFn, typename ValueFn>
  LMDBStore(CloneFromMasterTag /*tag*/,
            const std::string &master_path,
            std::size_t key_count,
            KeyFn &&make_key,
            ValueFn &&make_value) {
    ensure_master_built(master_path, key_count, std::forward<KeyFn>(make_key), std::forward<ValueFn>(make_value));
    // Fixed path, not an ever-incrementing counter -- see rocksdb_store.hpp's identical
    // constructor / bench_kv.cpp's make_vmemkv_clone_from_checkpoint() for why.
    path_ = master_path + "_clone.lmdb";
    clone_from(master_path, path_);

    open_env_and_dbi(path_, MDB_NOSUBDIR | MDB_WRITEMAP, " (clone)", env_, dbi_);
  }

  // No-op: LMDB reclaims B+Tree pages via its freelist; no manual reorganize concept.
  void reorganize() {}

  // ─── Low-level byte-span APIs (called by StoreAdapter) ───────────────────────

  template <typename Callback>
  auto get_impl(std::span<const std::byte> key, Callback callback) const -> bool {
    MDB_txn *txn = nullptr;
    if (mdb_txn_begin(env_, nullptr, MDB_RDONLY, &txn) != 0) {
      return false;
    }
    MDB_val mkey = to_val(key);
    MDB_val mval{};
    const bool found = (mdb_get(txn, dbi_, &mkey, &mval) == 0);
    if (found) {
      callback(std::span<const std::byte>(static_cast<const std::byte *>(mval.mv_data), mval.mv_size));
    }
    mdb_txn_abort(txn);
    return found;
  }

  auto insert_impl(std::span<const std::byte> key, std::span<const std::byte> value) -> bool {
    return with_existence_checked_txn(key, /*existence_required=*/false, [&](MDB_txn *txn, MDB_val &mkey) {
      MDB_val mval = to_val(value);
      return mdb_put(txn, dbi_, &mkey, &mval, 0) == 0;
    });
  }

  auto update_impl(std::span<const std::byte> key, std::span<const std::byte> value) -> bool {
    return with_existence_checked_txn(key, /*existence_required=*/true, [&](MDB_txn *txn, MDB_val &mkey) {
      MDB_val mval = to_val(value);
      return mdb_put(txn, dbi_, &mkey, &mval, 0) == 0;
    });
  }

  auto remove_impl(std::span<const std::byte> key) -> bool {
    return with_existence_checked_txn(key, /*existence_required=*/true, [&](MDB_txn *txn, MDB_val &mkey) {
      return mdb_del(txn, dbi_, &mkey, nullptr) == 0;
    });
  }

  template <typename KeyFn, typename ValueFn>
  void bulk_load_impl(std::size_t key_count, KeyFn &&make_key, ValueFn &&make_value) {
    bulk_load_into(env_, dbi_, key_count, std::forward<KeyFn>(make_key), std::forward<ValueFn>(make_value));
  }

  template <typename KeyFn, typename ValueFn>
  static void bulk_load_into(MDB_env *env, MDB_dbi dbi, std::size_t key_count, KeyFn &&make_key, ValueFn &&make_value) {
    if (key_count == 0) {
      return;
    }
    // Non-durable during bulk load (like vmemkv's WAL-skipping bulk_load and RocksDB's
    // disableWAL): MDB_MAPASYNC for the duration, then restore the durable default.
    mdb_env_set_flags(env, MDB_MAPASYNC, 1);
    struct RestoreDurability {
      MDB_env *env;
      ~RestoreDurability() { mdb_env_set_flags(env, MDB_MAPASYNC, 0); }
    } restore_durability{env};

    // Batch many puts per write transaction: per-key commits would pay commit overhead
    // key_count times.
    constexpr std::size_t kBatchKeys = 100'000;

    std::size_t next_index = 0;
    while (next_index < key_count) {
      MDB_txn *txn = nullptr;
      if (mdb_txn_begin(env, nullptr, 0, &txn) != 0) {
        throw std::runtime_error("LMDB txn_begin failed during bulk load");
      }
      const std::size_t chunk_end = std::min(key_count, next_index + kBatchKeys);
      for (std::size_t index = next_index; index < chunk_end; ++index) {
        const std::string key = make_key(index);
        const std::string value = make_value(index);
        MDB_val mkey{key.size(), const_cast<char *>(key.data())};
        MDB_val mval{value.size(), const_cast<char *>(value.data())};
        if (mdb_put(txn, dbi, &mkey, &mval, 0) != 0) {
          mdb_txn_abort(txn);
          throw std::runtime_error("LMDB put failed during bulk load");
        }
      }
      if (mdb_txn_commit(txn) != 0) {
        throw std::runtime_error("LMDB txn_commit failed during bulk load");
      }
      next_index = chunk_end;
    }
  }

  template <typename Cb>
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  [[nodiscard]] auto scan_impl(std::span<const std::byte> lower_bound,
                               std::span<const std::byte> upper_bound,
                               Cb callback) const -> size_t {
    MDB_txn *txn = nullptr;
    if (mdb_txn_begin(env_, nullptr, MDB_RDONLY, &txn) != 0) {
      return 0;
    }
    MDB_cursor *cursor = nullptr;
    if (mdb_cursor_open(txn, dbi_, &cursor) != 0) {
      mdb_txn_abort(txn);
      return 0;
    }

    size_t count = 0;
    MDB_val mkey = to_val(lower_bound);
    MDB_val mval{};
    int rc = mdb_cursor_get(cursor, &mkey, &mval, MDB_SET_RANGE);
    while (rc == 0) {
      if (compare_to_bound(mkey, upper_bound) >= 0) {
        break;
      }
      const vmemkv::ScanControl control = vmemkv::scan_detail::invoke_callback(
          callback,
          std::span<const std::byte>(static_cast<const std::byte *>(mkey.mv_data), mkey.mv_size),
          std::span<const std::byte>(static_cast<const std::byte *>(mval.mv_data), mval.mv_size));
      ++count;
      if (control == vmemkv::ScanControl::Stop) {
        break;
      }
      rc = mdb_cursor_get(cursor, &mkey, &mval, MDB_NEXT);
    }

    mdb_cursor_close(cursor);
    mdb_txn_abort(txn);
    return count;
  }

 private:
  // 1TB virtual address space reservation (sparse; LMDB only maps what's actually
  // used). Matches vmemkv::Config<>::kDefaultCapacityBytes so both engines are
  // given comparable headroom for LTM-scale corpora.
  static constexpr size_t kMapSizeBytes = 1ULL << 40;
  static constexpr unsigned int kMaxReaders = 512;

  // Creates an env with the store-wide size/reader tuning, throwing with `label` on failure.
  // Shared by open_env_and_dbi() and clone_from()'s read-only open below.
  static auto make_sized_env(const char *label) -> MDB_env * {
    MDB_env *env = nullptr;
    if (mdb_env_create(&env) != 0) {
      throw std::runtime_error(std::string("LMDB env_create failed") + label);
    }
    mdb_env_set_mapsize(env, kMapSizeBytes);
    mdb_env_set_maxreaders(env, kMaxReaders);
    return env;
  }

  // Opens an LMDB environment at `path` with `flags` and its single (unnamed) DBI -- the
  // create/set-mapsize/set-maxreaders/open/txn-begin/dbi-open/commit sequence shared by the main
  // constructor, the CloneFromMasterTag constructor, and ensure_master_built(). On any failure,
  // closes whatever partial env it already created (leaving `env_out` null) and throws, with
  // `label_suffix` (e.g. " (clone)", " (master build)", or "" for the main constructor) appended
  // so the error identifies which call site failed.
  static void open_env_and_dbi(
      const std::string &path, unsigned int flags, const char *label_suffix, MDB_env *&env_out, MDB_dbi &dbi_out) {
    MDB_env *env = make_sized_env(label_suffix);
    int rc = mdb_env_open(env, path.c_str(), flags, 0664);
    if (rc != 0) {
      mdb_env_close(env);
      env_out = nullptr;
      throw std::runtime_error(std::string("LMDB env_open failed") + label_suffix + ": " + mdb_strerror(rc));
    }
    MDB_txn *txn = nullptr;
    rc = mdb_txn_begin(env, nullptr, 0, &txn);
    if (rc != 0) {
      mdb_env_close(env);
      env_out = nullptr;
      throw std::runtime_error(std::string("LMDB txn_begin failed") + label_suffix + ": " + mdb_strerror(rc));
    }
    rc = mdb_dbi_open(txn, nullptr, 0, &dbi_out);
    if (rc != 0) {
      mdb_txn_abort(txn);
      mdb_env_close(env);
      env_out = nullptr;
      throw std::runtime_error(std::string("LMDB dbi_open failed") + label_suffix + ": " + mdb_strerror(rc));
    }
    mdb_txn_commit(txn);
    env_out = env;
  }

  // Builds a fresh master at a ".building" sibling, closed before anything copies it (LMDB
  // forbids one process opening the same file twice), renamed into place only once complete --
  // a crash mid-build never leaves a partial master to trust.
  template <typename KeyFn, typename ValueFn>
  static void ensure_master_built(const std::string &master_path,
                                  std::size_t key_count,
                                  KeyFn &&make_key,
                                  ValueFn &&make_value) {
    if (std::filesystem::exists(master_path)) {
      return;
    }
    const std::string building = vmemkv::rivals::building_path(master_path);
    std::error_code ignored;
    std::filesystem::remove(building, ignored);
    std::filesystem::remove(building + "-lock", ignored);

    MDB_env *build_env = nullptr;
    MDB_dbi build_dbi = 0;
    open_env_and_dbi(building, MDB_NOSUBDIR | MDB_WRITEMAP, " (master build)", build_env, build_dbi);

    bulk_load_into(build_env, build_dbi, key_count, std::forward<KeyFn>(make_key), std::forward<ValueFn>(make_value));
    mdb_env_close(build_env);

    std::filesystem::remove(master_path, ignored);
    std::filesystem::remove(master_path + "-lock", ignored);
    vmemkv::rivals::atomic_rename(building, master_path, "LMDB master build rename failed");
  }

  // Copies `source_path` to `dest_path` via mdb_env_copy2(..., MDB_CP_COMPACT), which also
  // drops free/unused pages so the clone is no larger than its live data. See the
  // CloneFromMasterTag constructor above for why this must be a real copy, not a hardlink.
  static void clone_from(const std::string &source_path, const std::string &dest_path) {
    std::error_code ignored;
    std::filesystem::remove(dest_path, ignored);
    std::filesystem::remove(dest_path + "-lock", ignored);

    MDB_env *source_env = make_sized_env(" (clone source)");
    int rc = mdb_env_open(source_env, source_path.c_str(), MDB_NOSUBDIR | MDB_RDONLY, 0664);
    if (rc != 0) {
      mdb_env_close(source_env);
      throw std::runtime_error("LMDB env_open failed (clone source): " + std::string(mdb_strerror(rc)));
    }
    rc = mdb_env_copy2(source_env, dest_path.c_str(), MDB_CP_COMPACT);
    mdb_env_close(source_env);
    if (rc != 0) {
      throw std::runtime_error("LMDB env_copy2 failed: " + std::string(mdb_strerror(rc)));
    }
  }

  static auto to_val(std::span<const std::byte> bytes) noexcept -> MDB_val {
    return MDB_val{bytes.size(), const_cast<std::byte *>(bytes.data())};
  }

  // Shared insert/update/remove body: write txn, existence check against
  // `existence_required`, mutate-or-abort, commit.
  template <typename Mutate>
  auto with_existence_checked_txn(std::span<const std::byte> key, bool existence_required, Mutate &&mutate) -> bool {
    MDB_txn *txn = nullptr;
    if (mdb_txn_begin(env_, nullptr, 0, &txn) != 0) {
      return false;
    }
    MDB_val mkey = to_val(key);
    MDB_val existing{};
    const bool exists = (mdb_get(txn, dbi_, &mkey, &existing) == 0);
    if (exists != existence_required || !mutate(txn, mkey)) {
      mdb_txn_abort(txn);
      return false;
    }
    return mdb_txn_commit(txn) == 0;
  }

  // LMDB's unnamed database orders by plain memcmp; mirror it for the upper-bound check.
  static auto compare_to_bound(const MDB_val &key, std::span<const std::byte> upper_bound) noexcept -> int {
    return vmemkv::bytes_compare_3way(
        std::span<const std::byte>(static_cast<const std::byte *>(key.mv_data), key.mv_size), upper_bound);
  }

  MDB_env *env_ = nullptr;
  MDB_dbi dbi_ = 0;
  std::string path_;
#else
  VMEMKV_RIVAL_DISABLED_STUB(LMDBStore, "LMDB")
#endif
};
