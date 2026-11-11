// rocksdb_common.hpp — Shared low-level helpers for the RocksDB-family rival stores
// (RocksDBStore, RocksDBBlobDBStore): both wrap the same rocksdb::DB API and differ only in
// tuning options (make_base_benchmark_db_options() vs. each store's own extensions of it) and the
// `engine_label` passed here purely to distinguish which one failed in an error message.
#pragma once

#ifdef ENABLE_ROCKSDB
#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <rocksdb/slice.h>
#include <rocksdb/utilities/checkpoint.h>
#include <rocksdb/write_batch.h>
#endif

#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vmemkv/scan_control.hpp>

#include "rival_common.hpp"

namespace vmemkv::rivals::rocksdb_common {

#ifdef ENABLE_ROCKSDB

inline auto to_slice(std::span<const std::byte> bytes) noexcept -> rocksdb::Slice {
  return {reinterpret_cast<const char *>(bytes.data()), bytes.size()};
}

inline auto to_bytes(const rocksdb::Slice &slice) noexcept -> std::span<const std::byte> {
  return {reinterpret_cast<const std::byte *>(slice.data()), slice.size()};
}

inline auto make_bulk_load_write_options() -> rocksdb::WriteOptions {
  rocksdb::WriteOptions opts;
  opts.disableWAL = true;
  return opts;
}

// Insert/Update/Delete fsync the WAL before returning, matching vmemkv's per-write fsync contract.
inline auto make_durable_write_options() -> rocksdb::WriteOptions {
  rocksdb::WriteOptions opts;
  opts.sync = true;
  return opts;
}

// Base tuning shared by every RocksDB-family rival; callers add their own extras (e.g. blob-file
// settings) on top.
inline auto make_base_benchmark_db_options() -> rocksdb::Options {
  rocksdb::Options opts;
  opts.create_if_missing = true;
  opts.compression = rocksdb::kNoCompression;
  opts.write_buffer_size = 256ULL * 1024ULL * 1024ULL;
  opts.max_write_buffer_number = 16;
  opts.min_write_buffer_number_to_merge = 4;
  opts.target_file_size_base = 256ULL * 1024ULL * 1024ULL;
  opts.level_compaction_dynamic_level_bytes = true;
  opts.level0_file_num_compaction_trigger = 16;
  opts.level0_slowdown_writes_trigger = 128;
  opts.level0_stop_writes_trigger = 256;
  opts.max_background_jobs = 8;
  opts.max_subcompactions = 4;
  opts.max_open_files = 256;
  return opts;
}

inline auto open_db(const rocksdb::Options &opts, const std::string &path, const char *engine_label) -> rocksdb::DB * {
  rocksdb::DB *db_handle = nullptr;
  auto status = rocksdb::DB::Open(opts, path, &db_handle);
  if (!status.ok()) {
    throw std::runtime_error(std::string(engine_label) + " Open failed: " + status.ToString());
  }
  return db_handle;
}

template <typename KeyFn, typename ValueFn>
void bulk_load_into(
    rocksdb::DB *db, std::size_t key_count, KeyFn &&make_key, ValueFn &&make_value, const char *engine_label) {
  if (key_count == 0) {
    return;
  }
  // Keep each WriteBatch within a fixed byte budget.
  constexpr std::size_t kBatchBytes = 64ULL * 1024ULL * 1024ULL;
  rocksdb::WriteOptions write_opts = make_bulk_load_write_options();

  std::size_t next_index = 0;
  while (next_index < key_count) {
    rocksdb::WriteBatch batch;
    std::size_t batch_bytes = 0;
    std::size_t index = next_index;
    for (; index < key_count && batch_bytes < kBatchBytes; ++index) {
      const std::string key = make_key(index);
      const std::string value = make_value(index);
      batch_bytes += key.size() + value.size();
      batch.Put(key, value);
    }

    auto status = db->Write(write_opts, &batch);
    if (!status.ok()) {
      throw std::runtime_error(std::string(engine_label) + " WriteBatch failed: " + status.ToString());
    }
    next_index = index;
  }
}

// Builds a fresh master DB at a ".building" sibling path, renamed into place only once fully
// populated and closed, so a crash mid-build never leaves a partial master for a later call to
// trust.
template <typename KeyFn, typename ValueFn>
void ensure_master_built(const rocksdb::Options &opts,
                         const std::string &master_path,
                         std::size_t key_count,
                         KeyFn &&make_key,
                         ValueFn &&make_value,
                         const char *engine_label) {
  if (std::filesystem::exists(master_path)) {
    return;
  }
  const std::string building = building_path(master_path);
  rocksdb::DestroyDB(building, {});
  rocksdb::DB *db_handle = open_db(opts, building, engine_label);
  bulk_load_into(db_handle, key_count, std::forward<KeyFn>(make_key), std::forward<ValueFn>(make_value), engine_label);
  delete db_handle;
  rocksdb::DestroyDB(master_path, {});
  atomic_rename(building, master_path, std::string(engine_label) + " master build rename failed");
}

// Briefly opens `source_path` read-only, checkpoints it to `dest_path` (destroying any stale
// directory already there), then closes the read-only handle. Hardlinks unchanged SST files
// instead of copying bytes, which is safe because SST files are immutable once written:
// compaction produces new files rather than editing a shared hardlink in place, so writes
// against a clone can never corrupt the master or a sibling clone.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
inline void clone_from(const rocksdb::Options &opts,
                       const std::string &source_path,
                       const std::string &dest_path,
                       const char *engine_label) {
  rocksdb::DB *source_db = nullptr;
  auto status = rocksdb::DB::OpenForReadOnly(opts, source_path, &source_db);
  if (!status.ok()) {
    throw std::runtime_error(std::string(engine_label) +
                             " OpenForReadOnly (clone source) failed: " + status.ToString());
  }
  rocksdb::Checkpoint *checkpoint = nullptr;
  status = rocksdb::Checkpoint::Create(source_db, &checkpoint);
  if (!status.ok()) {
    delete source_db;
    throw std::runtime_error(std::string(engine_label) + " Checkpoint::Create failed: " + status.ToString());
  }
  rocksdb::DestroyDB(dest_path, {});
  status = checkpoint->CreateCheckpoint(dest_path);
  delete checkpoint;
  delete source_db;
  if (!status.ok()) {
    throw std::runtime_error(std::string(engine_label) + " CreateCheckpoint failed: " + status.ToString());
  }
}

template <typename Callback>
auto get_from_db(rocksdb::DB *db, std::span<const std::byte> key, Callback callback) -> bool {
  rocksdb::PinnableSlice pinned_value;
  auto status = db->Get({}, db->DefaultColumnFamily(), to_slice(key), &pinned_value);
  if (!status.ok()) {
    return false;
  }
  std::span<const std::byte> val_span(reinterpret_cast<const std::byte *>(pinned_value.data()), pinned_value.size());
  callback(val_span);
  return true;
}

inline auto exists_in_db(rocksdb::DB *db, std::span<const std::byte> key) -> bool {
  rocksdb::PinnableSlice pinned_value;
  return db->Get({}, db->DefaultColumnFamily(), to_slice(key), &pinned_value).ok();
}

inline auto insert_into_db(rocksdb::DB *db, std::span<const std::byte> key, std::span<const std::byte> value) -> bool {
  return require_exists([&] { return exists_in_db(db, key); },
                        false,
                        [&] { return db->Put(make_durable_write_options(), to_slice(key), to_slice(value)).ok(); });
}

inline auto update_in_db(rocksdb::DB *db, std::span<const std::byte> key, std::span<const std::byte> value) -> bool {
  return require_exists([&] { return exists_in_db(db, key); },
                        true,
                        [&] { return db->Put(make_durable_write_options(), to_slice(key), to_slice(value)).ok(); });
}

inline auto remove_from_db(rocksdb::DB *db, std::span<const std::byte> key) -> bool {
  return require_exists([&] { return exists_in_db(db, key); },
                        true,
                        [&] { return db->Delete(make_durable_write_options(), to_slice(key)).ok(); });
}

template <typename Cb>
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
auto scan_db(rocksdb::DB *db,
             std::span<const std::byte> lower_bound,
             std::span<const std::byte> upper_bound,
             Cb callback) -> size_t {
  auto iterator = std::unique_ptr<rocksdb::Iterator>(db->NewIterator({}));
  rocksdb::Slice lower_bound_slice = to_slice(lower_bound);
  rocksdb::Slice upper_bound_slice = to_slice(upper_bound);
  size_t count = 0;
  for (iterator->Seek(lower_bound_slice); iterator->Valid() && iterator->key().compare(upper_bound_slice) < 0;
       iterator->Next()) {
    const ScanControl control =
        scan_detail::invoke_callback(callback, to_bytes(iterator->key()), to_bytes(iterator->value()));
    ++count;
    if (control == ScanControl::Stop) {
      break;
    }
  }
  return count;
}

#endif  // ENABLE_ROCKSDB

}  // namespace vmemkv::rivals::rocksdb_common
