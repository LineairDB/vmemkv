// store_names.hpp — StoreAdapter name-hook specializations for rival backends.
//
// StoreAdapter::name() calls vmemkv::detail::store_name<T>() (see
// src/vmemkv/api/store_adapter.hpp); vmemkv types use the primary template
// (T::name()), rivals specialize here. Included only where rival names are
// needed (tests/benchmark via store_variants.hpp), so the core adapter keeps
// no rival dependency.
#pragma once

#include <string>

#include "leanstore_store.hpp"
#include "lmdb_store.hpp"
#include "rocksdb_blobdb_store.hpp"
#include "rocksdb_store.hpp"

namespace vmemkv::detail {

template <>
inline auto store_name<::RocksDBStore>() -> std::string {
  return "RocksDB";
}

template <>
inline auto store_name<::RocksDBBlobDBStore>() -> std::string {
  return "RocksDB-BlobDB";
}

template <>
inline auto store_name<::LMDBStore>() -> std::string {
  return "LMDB";
}

template <>
inline auto store_name<::LeanStoreStore>() -> std::string {
  return "LeanStore";
}

}  // namespace vmemkv::detail
