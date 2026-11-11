// rocksdb_blobdb_store.hpp — Thin RocksDB "BlobDB" wrapper exposing byte-span APIs
// for comparison.
//
// Uses RocksDB's integrated blob-file support (enable_blob_files).
// Key-value separation trades
// write amplification for reduced compaction I/O on large values -- the interesting comparison
// point against plain RocksDBStore for the LTM(64KB) scenarios.
//
// Thread safety: RocksDB is internally thread-safe.

#pragma once

#ifdef ENABLE_ROCKSDB
#include <rocksdb/options.h>
#endif

#include "rocksdb_common.hpp"
#include "rocksdb_engine_store.hpp"

namespace vmemkv::rivals {

struct RocksDBBlobDBPolicy {
  static constexpr const char *kLabel = "RocksDB(BlobDB)";
  static constexpr const char *kCloneLabel = "RocksDB(BlobDB) (clone)";
#ifdef ENABLE_ROCKSDB
  static auto make_options() -> rocksdb::Options {
    rocksdb::Options opts = rocksdb_common::make_base_benchmark_db_options();
    // Values below this size stay inline in SST files; larger ones go to blob files.
    opts.enable_blob_files = true;
    opts.min_blob_size = 256;
    opts.blob_file_size = 256ULL * 1024ULL * 1024ULL;
    opts.blob_compression_type = rocksdb::kNoCompression;
    opts.enable_blob_garbage_collection = true;
    return opts;
  }
#endif
};

}  // namespace vmemkv::rivals

// A derived class (not a type alias) so rival identity stays a distinct type
// for the StoreAdapter name hook (see store_names.hpp) -- a `using` alias to
// the template instantiation would collapse the two RocksDB policies to one type.
class RocksDBBlobDBStore : public vmemkv::rivals::RocksDBEngineStore<vmemkv::rivals::RocksDBBlobDBPolicy> {
 public:
  using RocksDBEngineStore::RocksDBEngineStore;
};
