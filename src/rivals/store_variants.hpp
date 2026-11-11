// store_variants.hpp - Store registry for tests and benchmarks: VMemKV read policies
// and rival-backend aliases in one tuple. Internal to tests/benchmark;
// not part of the public API (see include/vmemkv/vmemkv.hpp).
#pragma once

#include <tuple>
#include <vmemkv/variants.hpp>

#include "leanstore_store.hpp"
#include "lmdb_store.hpp"
#include "rocksdb_blobdb_store.hpp"
#include "rocksdb_store.hpp"
#include "store_names.hpp"

namespace vmemkv::variants {

using VMemKV_RocksDB = StoreAdapter<::RocksDBStore>;
using VMemKV_RocksDBBlobDB = StoreAdapter<::RocksDBBlobDBStore>;
using VMemKV_LMDB = StoreAdapter<::LMDBStore>;
using VMemKV_LeanStore = StoreAdapter<::LeanStoreStore>;

// Unified benchmark registration tuple.
using AllPossibleTypes = std::tuple<VMemKV_Default,
                                    VMemKV_ReadRandom,
                                    VMemKV_ReadSeq,
                                    VMemKV_RocksDB,
                                    VMemKV_RocksDBBlobDB,
                                    VMemKV_LMDB,
                                    VMemKV_LeanStore>;

// Read-path policy pins registered on read cells only (Get Hit Zipf, Scan Zipf/Uniform),
// not the full matrix: every other workload is insensitive to the mapping choice.
using ReadPinTypes = std::tuple<VMemKV_ReadNormal, VMemKV_ReadPread>;
}  // namespace vmemkv::variants
