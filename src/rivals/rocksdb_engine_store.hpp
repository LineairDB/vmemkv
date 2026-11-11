// rocksdb_engine_store.hpp — Shared wiring for RocksDBStore/RocksDBBlobDBStore: identical
// construction/cloning/delegation, differing only in engine label and rocksdb::Options
// (see each Policy in rocksdb_store.hpp/rocksdb_blobdb_store.hpp).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef ENABLE_ROCKSDB
#include <rocksdb/db.h>
#include <rocksdb/options.h>
#endif

#include "rival_common.hpp"
#include "rival_store_disabled_stub.hpp"
#include "rocksdb_common.hpp"

namespace vmemkv::rivals {

// `Policy` supplies: kLabel, kCloneLabel (both `const char *`), and `static auto
// make_options() -> rocksdb::Options` (only required/compiled when ENABLE_ROCKSDB is defined).
template <typename Policy>
class RocksDBEngineStore {
 public:
  static constexpr bool kIsEnabled =
#ifdef ENABLE_ROCKSDB
      true;
#else
      false;
#endif
  static constexpr bool kIsRival = true;

  // RivalPolicy surface: checkpoint noop (StoreAdapter unchanged).
  static void checkpoint() noexcept {}

#ifdef ENABLE_ROCKSDB
  // Opens a fresh DB at a unique subpath (avoids ENOLCK lock contention across sweeps).
  explicit RocksDBEngineStore(std::string path) {
    path_ = make_unique_instance_path(std::move(path), "");
    rocksdb::DestroyDB(path_, {});
    db_.reset(rocksdb_common::open_db(Policy::make_options(), path_, Policy::kLabel));
  }

  // Closes and destroys the database.
  ~RocksDBEngineStore() {
    db_.reset();
    rocksdb::DestroyDB(path_, {});
  }

  RocksDBEngineStore(const RocksDBEngineStore &) = delete;
  auto operator=(const RocksDBEngineStore &) -> RocksDBEngineStore & = delete;

  // Shared tag selecting the clone-from-master constructor (see rival_common.hpp).
  using CloneFromMasterTag = ::vmemkv::rivals::CloneFromMasterTag;

  // Builds `master_path` once via bulk_load, then clones via Checkpoint -- see
  // rocksdb_common::clone_from() for why this is cheap and safe.
  template <typename KeyFn, typename ValueFn>
  RocksDBEngineStore(CloneFromMasterTag /*tag*/,
                     const std::string &master_path,
                     std::size_t key_count,
                     KeyFn &&make_key,
                     ValueFn &&make_value) {
    rocksdb_common::ensure_master_built(Policy::make_options(),
                                        master_path,
                                        key_count,
                                        std::forward<KeyFn>(make_key),
                                        std::forward<ValueFn>(make_value),
                                        Policy::kLabel);
    // Fixed path, not an ever-incrementing counter -- see bench_kv.cpp's make_vmemkv_clone_from_
    // checkpoint() for why this is safe (at most one clone of a given master is ever live
    // process-wide) and necessary (an unbounded counter accumulates a fresh multi-GB clone per
    // benchmark case instead of reusing one). clone_from() below already destroys any stale
    // directory at this path before rebuilding it.
    path_ = master_path + "_clone";
    rocksdb_common::clone_from(Policy::make_options(), master_path, path_, Policy::kLabel);
    db_.reset(rocksdb_common::open_db(Policy::make_options(), path_, Policy::kCloneLabel));
  }

  // No-op: self-compacting (blob GC rides compaction for the BlobDB policy).
  void reorganize() {}

  // ─── Low-level byte-span APIs (called by StoreAdapter) ───────────────────────

  template <typename Callback>
  auto get_impl(std::span<const std::byte> key, Callback callback) const -> bool {
    return rocksdb_common::get_from_db(db_.get(), key, callback);
  }

  auto insert_impl(std::span<const std::byte> key, std::span<const std::byte> value) -> bool {
    return rocksdb_common::insert_into_db(db_.get(), key, value);
  }

  auto update_impl(std::span<const std::byte> key, std::span<const std::byte> value) -> bool {
    return rocksdb_common::update_in_db(db_.get(), key, value);
  }

  auto remove_impl(std::span<const std::byte> key) -> bool { return rocksdb_common::remove_from_db(db_.get(), key); }

  template <typename KeyFn, typename ValueFn>
  void bulk_load_impl(std::size_t key_count, KeyFn &&make_key, ValueFn &&make_value) {
    rocksdb_common::bulk_load_into(
        db_.get(), key_count, std::forward<KeyFn>(make_key), std::forward<ValueFn>(make_value), Policy::kLabel);
  }

  template <typename Cb>
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  [[nodiscard]] auto scan_impl(std::span<const std::byte> lower_bound,
                               std::span<const std::byte> upper_bound,
                               Cb callback) const -> size_t {
    return rocksdb_common::scan_db(db_.get(), lower_bound, upper_bound, callback);
  }

 private:
  std::unique_ptr<rocksdb::DB> db_;
  std::string path_;
#else
  // Same stub surface; Policy::kLabel isn't a string literal, but the macro accepts any string
  // expression since it builds the message with std::string().
  VMEMKV_RIVAL_DISABLED_STUB(RocksDBEngineStore, Policy::kLabel)
#endif
};

}  // namespace vmemkv::rivals
