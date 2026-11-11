// test_kv_store_fixture.hpp - Shared fixture for test_kv_store.cpp and
// test_kv_store_vmemkv.cpp: store factory, store-type lists, checkpoint config,
// and small insert/check helpers. Both test TUs include this; nothing else does,
// so rival-heavy includes stay out of the other test TUs.
#pragma once

#include <doctest/doctest.h>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <rivals/store_variants.hpp>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <vmemkv/variants.hpp>

#include "test_support.hpp"

namespace test_util {
// Decodes a little-endian u64 from a span (short spans decode their prefix; empty decodes 0).
inline auto decode_u64(std::span<const std::byte> val) -> uint64_t {
  uint64_t res = 0;
  std::memcpy(&res, val.data(), std::min(val.size(), sizeof(uint64_t)));
  return res;
}

template <typename StorePtr, typename Key>
auto get_u64_sync(const StorePtr &store, const Key &key) -> uint64_t {
  uint64_t res = vmemkv::STORE_NOT_FOUND;
  store->get(key, [&](std::span<const std::byte> val) { res = decode_u64(val); });
  return res;
}

}  // namespace test_util

namespace {
constexpr uint32_t kTestByteMask = 0xffU;
constexpr int kCustomKeyId = 42;
constexpr int kCustomKeyCategoryBase = 7;
constexpr uint64_t kRemovedValue = 7;
constexpr uint64_t kSmallValue = 100;
constexpr uint64_t kLargeValue = 200;
constexpr std::size_t kValue64Bytes = 64;
constexpr std::size_t kValue100Bytes = 100;
constexpr std::size_t kValue200Bytes = 200;
constexpr std::size_t kInlineValueBytes = 8;
constexpr std::size_t kShortInlineValueBytes = 5;
constexpr int kBitsPerByte = 8;
constexpr std::array<std::byte, kInlineValueBytes> kCustomKeySerializedTemplate{};
constexpr unsigned int kScanStart = 0U;
constexpr unsigned int kScanTen = 10U;
constexpr unsigned int kScanHundred = 100U;
constexpr unsigned int kScanUpperBound = 1000U;
constexpr std::size_t kLongValueBytes = 12;
}  // namespace

template <typename Store>
struct StoreFactory;

// Helper to reserve temp path for T2 files
static auto reserve_temp_path() -> std::filesystem::path {
  return vmemkv_test::reserve_unique_temp_path("vmemkv_kv", /*also_remove_wal_sibling=*/true);
}

template <typename Impl>
struct VMemKVDeleter {
  std::filesystem::path path;
  void operator()(vmemkv::StoreAdapter<Impl> *store_ptr) const noexcept {
    delete store_ptr;
    vmemkv::remove_store_files(path);
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
};

template <typename Impl>
struct StoreFactory<vmemkv::StoreAdapter<Impl>> {
  static constexpr uint64_t kCapacityBytes = 8U << 20;  // 8 MiB

  static auto make() -> std::unique_ptr<vmemkv::StoreAdapter<Impl>, VMemKVDeleter<Impl>> {
    std::filesystem::path path = reserve_temp_path();
    if constexpr (std::is_same_v<Impl, ::RocksDBStore> || std::is_same_v<Impl, ::RocksDBBlobDBStore> ||
                  std::is_same_v<Impl, ::LMDBStore> || std::is_same_v<Impl, ::LeanStoreStore>) {
      std::error_code ignored;
      std::filesystem::remove(path, ignored);
      auto *store = new vmemkv::StoreAdapter<Impl>(path.string());
      return std::unique_ptr<vmemkv::StoreAdapter<Impl>, VMemKVDeleter<Impl>>(store, VMemKVDeleter<Impl>{path});
    } else {
      auto *store = new vmemkv::StoreAdapter<Impl>(path, kCapacityBytes);
      return std::unique_ptr<vmemkv::StoreAdapter<Impl>, VMemKVDeleter<Impl>>(store, VMemKVDeleter<Impl>{path});
    }
  }
};

// ─── Store type lists ───────────────────────────────────────────────────────

// VMemKV's default and read-policy configurations.
#define VMemKVStores \
  vmemkv::variants::VMemKV_Default, vmemkv::variants::VMemKV_ReadRandom, vmemkv::variants::VMemKV_ReadSeq

// Rival backend variants (RocksDBStore, LMDBStore, etc.)
#ifdef ENABLE_ROCKSDB
#define RocksDBRivalStores , vmemkv::variants::VMemKV_RocksDB, vmemkv::variants::VMemKV_RocksDBBlobDB
#else
#define RocksDBRivalStores
#endif

#ifdef ENABLE_LMDB
#define LMDBRivalStores , vmemkv::variants::VMemKV_LMDB
#else
#define LMDBRivalStores
#endif

#ifdef ENABLE_LEANSTORE
#define LeanStoreRivalStores , vmemkv::variants::VMemKV_LeanStore
#else
#define LeanStoreRivalStores
#endif

#define RivalStores RocksDBRivalStores LMDBRivalStores LeanStoreRivalStores

#define STORE_TYPES VMemKVStores RivalStores

struct FrequentCheckpointConfig : vmemkv::Config<> {
  // Partial override style: inherit all defaults and only tune checkpoint aggressiveness.
  static constexpr size_t kMaxBytesSinceCheckpoint = 64ULL << 10;  // 64 KiB.
};

static_assert(FrequentCheckpointConfig::kAppendCapacityEntries ==
              (size_t{1} << FrequentCheckpointConfig::kAppendCapacityLog2));

using VMemKV_FrequentCheckpoint = vmemkv::StoreAdapter<vmemkv::VMemKVImpl<FrequentCheckpointConfig>>;

template <typename StoreHandle>
static void insert_sequential_u64_values(StoreHandle &store, int key_count) {
  for (int i = 0; i < key_count; ++i) {
    CHECK(store->insert("k" + std::to_string(i), static_cast<uint64_t>(i)));
  }
}

template <typename StoreHandle>
static void check_sequential_u64_values(StoreHandle &store, int key_count) {
  for (int i = 0; i < key_count; ++i) {
    CHECK(test_util::get_u64_sync(store, "k" + std::to_string(i)) == static_cast<uint64_t>(i));
  }
}

template <typename StoreHandle>
static void check_hot_key_upgrade_is_visible(StoreHandle &store) {
  CHECK(store->insert("hot", std::string("a")));
  const std::string large_value(kValue64Bytes, 'x');
  CHECK(store->update("hot", large_value));
  const auto got = vmemkv_test::get_optional_bytes(store, "hot");
  REQUIRE(got.has_value());
  if (!got.has_value()) {
    return;
  }
  CHECK(vmemkv_test::span_to_string(vmemkv_test::as_span(got.value())) == large_value);
}
