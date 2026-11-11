// test_kv_store.cpp — Cross-backend correctness matrix (TEST_CASE_TEMPLATE over STORE_TYPES).
//
// Each scenario is an independent TEST_CASE_TEMPLATE instantiated for every
// store type.  A fresh store is constructed per TEST_CASE (via StoreFactory)
// so tests are fully isolated. vmemkv-specific cases live in
// test_kv_store_vmemkv.cpp; shared fixture in test_kv_store_fixture.hpp.

#include <doctest/doctest.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <rivals/store_variants.hpp>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <unordered_set>
#include <vector>
#include <vmemkv/checkpoint/checkpoint.hpp>
#include <vmemkv/vmemkv.hpp>

#include "test_support.hpp"

// Verify that the default and read-policy stores satisfy the C++20 KVStore concept.
static_assert(vmemkv::KVStore<vmemkv::variants::VMemKV_Default>);
static_assert(vmemkv::KVStore<vmemkv::VMemKVStore>);
static_assert(vmemkv::KVStore<vmemkv::variants::VMemKV_RocksDB>);
static_assert(vmemkv::KVStore<vmemkv::variants::VMemKV_RocksDBBlobDB>);
static_assert(vmemkv::KVStore<vmemkv::variants::VMemKV_LMDB>);
static_assert(vmemkv::KVStore<vmemkv::variants::VMemKV_ReadNormal>);
static_assert(vmemkv::KVStore<vmemkv::variants::VMemKV_ReadPread>);
static_assert(vmemkv::Config<>::UseBloomFilter);
static_assert(vmemkv::Config<>::UseT1InlineValue);

#include "test_kv_store_fixture.hpp"

// ─── Test cases (one per scenario) ───────────────────────────────────────────

TEST_CASE_TEMPLATE("get on empty store returns STORE_NOT_FOUND", Store, STORE_TYPES) {
  auto store = StoreFactory<Store>::make();
  CHECK(test_util::get_u64_sync(store, "x") == vmemkv::STORE_NOT_FOUND);
}

TEST_CASE_TEMPLATE("insert and get", Store, STORE_TYPES) {
  auto store = StoreFactory<Store>::make();
  CHECK(store->insert("a", 10));
  CHECK(test_util::get_u64_sync(store, "a") == 10U);
}

TEST_CASE_TEMPLATE("insert duplicate returns false, value unchanged", Store, STORE_TYPES) {
  auto store = StoreFactory<Store>::make();
  CHECK(store->insert("a", 10));
  CHECK_FALSE(store->insert("a", 99));
  CHECK(test_util::get_u64_sync(store, "a") == 10U);
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE_TEMPLATE("integral keys use the templated convenience API", Store, STORE_TYPES) {
  auto store = StoreFactory<Store>::make();
  constexpr int key = 42;

  CHECK(store->insert(key, 10));
  CHECK(test_util::get_u64_sync(store, key) == 10U);
  CHECK(store->update(key, 11));
  CHECK(test_util::get_u64_sync(store, key) == 11U);
  CHECK(store->remove(key));
  CHECK(test_util::get_u64_sync(store, key) == vmemkv::STORE_NOT_FOUND);
}

namespace test_adl {
struct CustomKey {
  int id;
  int category;
};

inline auto serialize_kvs(const CustomKey &custom_key) noexcept
    -> std::array<std::byte, kCustomKeySerializedTemplate.size()> {
  auto out = kCustomKeySerializedTemplate;
  auto uid = static_cast<uint32_t>(custom_key.id);
  auto ucat = static_cast<uint32_t>(custom_key.category);
  for (int i = 0; i < 4; ++i) {
    out[i] = static_cast<std::byte>((uid >> ((3 - i) * kBitsPerByte)) & kTestByteMask);
    out[i + 4] = static_cast<std::byte>((ucat >> ((3 - i) * kBitsPerByte)) & kTestByteMask);
  }
  return out;
}
}  // namespace test_adl

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE_TEMPLATE("custom serializers (ADL) for user-defined types", Store, STORE_TYPES) {
  auto store = StoreFactory<Store>::make();
  test_adl::CustomKey key_one{kCustomKeyId, kCustomKeyCategoryBase};
  test_adl::CustomKey key_two{kCustomKeyId, kCustomKeyCategoryBase + 1};

  CHECK(store->insert(key_one, kSmallValue));
  CHECK(store->insert(key_two, kLargeValue));

  CHECK(test_util::get_u64_sync(store, key_one) == kSmallValue);
  CHECK(test_util::get_u64_sync(store, key_two) == kLargeValue);

  std::vector<uint64_t> results;
  const test_adl::CustomKey key_end{kCustomKeyId, kCustomKeyCategoryBase + 2};
  const size_t scan_count =
      store->scan(key_one, key_end, [&](std::span<const std::byte>, std::span<const std::byte> value) {
        results.push_back(test_util::decode_u64(value));
      });
  REQUIRE(scan_count == 2U);
  REQUIRE(results.size() == 2);
  CHECK(results[0] == kSmallValue);
  CHECK(results[1] == kLargeValue);
}

// A value that's bit-for-bit identical to T1's STORE_NOT_FOUND sentinel (~0ULL) is fully storable
// and retrievable like any other: VMemKVImpl::try_make_inline_payload() declines to inline this
// exact value, routing it through the ordinary T2-record path instead, whose payload is an
// offset, never the raw value bytes.
TEST_CASE_TEMPLATE("insert accepts a value equal to STORE_NOT_FOUND and it round-trips", Store, STORE_TYPES) {
  auto store = StoreFactory<Store>::make();
  CHECK(store->insert("a", vmemkv::STORE_NOT_FOUND));
  bool found = store->get("a", [](std::span<const std::byte> /*val*/) {});
  CHECK(found);
  CHECK(test_util::get_u64_sync(store, "a") == vmemkv::STORE_NOT_FOUND);
}

TEST_CASE_TEMPLATE("update existing key", Store, STORE_TYPES) {
  auto store = StoreFactory<Store>::make();
  store->insert("a", 1);
  CHECK(store->update("a", 2));
  CHECK(test_util::get_u64_sync(store, "a") == 2U);
}

TEST_CASE_TEMPLATE("update missing key returns false", Store, STORE_TYPES) {
  auto store = StoreFactory<Store>::make();
  CHECK_FALSE(store->update("z", 1));
}

// Same as the insert case above, for update().
TEST_CASE_TEMPLATE("update accepts a value equal to STORE_NOT_FOUND and it round-trips", Store, STORE_TYPES) {
  auto store = StoreFactory<Store>::make();
  store->insert("a", 1);
  store->update("a", vmemkv::STORE_NOT_FOUND);
  CHECK(test_util::get_u64_sync(store, "a") == vmemkv::STORE_NOT_FOUND);
}

// Per-key get() exactness after distinct same-size updates plus interleaved deletes:
// every surviving key must read back exactly its latest value (not a stale version or a
// length-corrupted payload), deleted keys must read absent. Same-size values keep this
// valid for LeanStore (no size-changing update path there); no key is reinserted.
TEST_CASE_TEMPLATE("get returns latest value per key after update and delete churn", Store, STORE_TYPES) {
  auto store = StoreFactory<Store>::make();
  constexpr size_t kKeys = 300;
  for (size_t i = 0; i < kKeys; ++i) {
    store->insert("k" + std::to_string(i), i);
  }
  for (size_t i = 0; i < kKeys; ++i) {
    CHECK(store->update("k" + std::to_string(i), 100000 + i));
  }
  for (size_t i = 0; i < kKeys; i += 3) {
    CHECK(store->remove("k" + std::to_string(i)));
  }
  for (size_t i = 0; i < kKeys; ++i) {
    if (i % 3 == 0) {
      CHECK(test_util::get_u64_sync(store, "k" + std::to_string(i)) == vmemkv::STORE_NOT_FOUND);
    } else {
      CHECK(test_util::get_u64_sync(store, "k" + std::to_string(i)) == 100000 + i);
    }
  }
  CHECK(test_util::get_u64_sync(store, "k_missing") == vmemkv::STORE_NOT_FOUND);
}
// LeanStore-only: clone-from-master construction round-trips data, isolates mutations to
// the clone (the master stays intact for later clones), and a second clone of the same
// master still recovers. Exercises ensure_master_built()/clone_from()/open_recover(),
// including the registry prune across sequential instances.
#ifdef ENABLE_LEANSTORE
TEST_CASE("LeanStore clone from master round-trips and isolates") {
  using Store = vmemkv::StoreAdapter<::LeanStoreStore>;
  const std::string master = reserve_temp_path().string() + "_master";
  auto make_key = [](std::size_t i) { return vmemkv_test::padded_key(i, 5); };
  auto make_value = [](std::size_t i) {
    std::string v = "v" + std::to_string(i);
    v.append(24 - v.size(), '.');
    return v;
  };
  constexpr std::size_t kKeys = 200;
  const auto check_all = [&](Store *store, const std::string &phase) {
    for (size_t i = 0; i < kKeys; ++i) {
      const auto got = vmemkv_test::get_optional_bytes(store, make_key(i));
      if (!got.has_value() || vmemkv_test::span_to_string(vmemkv_test::as_span(*got)) != make_value(i)) {
        MESSAGE("mismatch phase=" << phase << " i=" << i << " got="
                                  << (got.has_value() ? vmemkv_test::span_to_string(vmemkv_test::as_span(*got))
                                                      : "<absent>"));
      }
      REQUIRE(got.has_value());
      CHECK(vmemkv_test::span_to_string(vmemkv_test::as_span(*got)) == make_value(i));
    }
  };
  {
    // Same data through the plain insert path (no master/clone/recover involved).
    auto plain = StoreFactory<Store>::make();
    for (size_t i = 0; i < kKeys; ++i) {
      plain->insert(make_key(i), make_value(i));
    }
    check_all(plain.get(), "plain");
  }
  {
    Store clone(typename ::LeanStoreStore::CloneFromMasterTag{}, master, kKeys, make_key, make_value);
    check_all(&clone, "clone1");
    for (size_t i = 0; i < kKeys; i += 2) {
      CHECK(clone.remove(make_key(i)));
    }
  }
  {
    Store clone2(typename ::LeanStoreStore::CloneFromMasterTag{}, master, kKeys, make_key, make_value);
    check_all(&clone2, "clone2");
  }
  std::error_code ignored;
  std::filesystem::remove(master, ignored);
  std::filesystem::remove(master + ".json", ignored);
  std::filesystem::remove(master + "_clone.leanstore", ignored);
  std::filesystem::remove(master + "_clone.leanstore.json", ignored);
}
#endif

TEST_CASE_TEMPLATE("remove existing key", Store, STORE_TYPES) {
  auto store = StoreFactory<Store>::make();
  store->insert("a", kRemovedValue);
  CHECK(store->remove("a"));
  CHECK(test_util::get_u64_sync(store, "a") == vmemkv::STORE_NOT_FOUND);
}

TEST_CASE_TEMPLATE("remove missing key returns false", Store, STORE_TYPES) {
  auto store = StoreFactory<Store>::make();
  CHECK_FALSE(store->remove("z"));
}

TEST_CASE_TEMPLATE("re-insert after remove", Store, STORE_TYPES) {
  // LeanStore's BTreeVI has no insert-after-remove path (upstream TODO, hits ensure(false)),
  // and no benchmark flow reinserts a removed key.
  if constexpr (std::is_same_v<Store, vmemkv::variants::VMemKV_LeanStore>) {
    MESSAGE("skipped for LeanStore: engine cannot reinsert removed keys");
    return;
  }
  auto store = StoreFactory<Store>::make();
  store->insert("a", 1);
  store->remove("a");
  CHECK(store->insert("a", 2));
  CHECK(test_util::get_u64_sync(store, "a") == 2U);
}

TEST_CASE_TEMPLATE("scan empty range returns 0", Store, STORE_TYPES) {
  auto store = StoreFactory<Store>::make();
  size_t entry_count = 0;
  store->insert("b", 1);
  entry_count = store->scan("a", "a", [](std::span<const std::byte>, std::span<const std::byte>) {});
  CHECK(entry_count == 0U);
}

TEST_CASE_TEMPLATE("scan returns all entries in range", Store, STORE_TYPES) {
  auto store = StoreFactory<Store>::make();
  store->insert("b", 2);
  store->insert("c", 3);
  store->insert("d", 4);
  store->insert("e", 5);
  std::unordered_set<uint64_t> values;
  // Half-open [b, e): the upper bound itself is excluded.
  size_t entry_count = store->scan("b", "e", [&](std::span<const std::byte>, std::span<const std::byte> value) {
    values.insert(test_util::decode_u64(value));
  });
  CHECK(entry_count == 3U);
  CHECK(values.count(2) == 1);
  CHECK(values.count(3) == 1);
  CHECK(values.count(4) == 1);
  CHECK(values.count(5) == 0);
  values.clear();
  entry_count = store->scan("b", "d", [&](std::span<const std::byte>, std::span<const std::byte> value) {
    values.insert(test_util::decode_u64(value));
  });
  CHECK(entry_count == 2U);
  CHECK(values.count(2) == 1);
  CHECK(values.count(3) == 1);
  CHECK(values.count(4) == 0);
}

TEST_CASE_TEMPLATE("scan callback can stop after a live entry", Store, STORE_TYPES) {
  auto store = StoreFactory<Store>::make();
  store->insert("a", 1);
  store->insert("b", std::string("a value stored outside T1"));
  store->insert("c", 3);

  std::vector<std::string> seen;
  size_t entry_count = store->scan("a", "d", [&](std::span<const std::byte> key, std::span<const std::byte>) {
    seen.push_back(vmemkv_test::span_to_string(key));
    return vmemkv::ScanControl::Stop;
  });
  CHECK(entry_count == 1U);
  CHECK(seen == std::vector<std::string>{"a"});

  store->reorganize();
  seen.clear();
  entry_count = store->scan("a", "c", [&](std::span<const std::byte> key, std::span<const std::byte>) {
    const std::string current = vmemkv_test::span_to_string(key);
    seen.push_back(current);
    return current == "b" ? vmemkv::ScanControl::Stop : vmemkv::ScanControl::Continue;
  });
  CHECK(entry_count == 2U);
  CHECK(seen == std::vector<std::string>{"a", "b"});
}

TEST_CASE_TEMPLATE("scan excludes removed entries", Store, STORE_TYPES) {
  auto store = StoreFactory<Store>::make();
  store->insert("a", 1);
  store->insert("b", 2);
  store->remove("a");
  const size_t entry_count = store->scan("a", "c", [](std::span<const std::byte>, std::span<const std::byte>) {});
  CHECK(entry_count == 1U);
}

TEST_CASE_TEMPLATE("scan after churn returns key order with latest values", Store, STORE_TYPES) {
  // 300 keys cross several internal read batches; updating every key with a larger value forces
  // out-of-place appends, decorrelating T2 physical offsets from key order. The scan must still
  // report keys in order, each with its latest value, and count every live entry.
  auto store = StoreFactory<Store>::make();
  constexpr int kKeys = 300;
  for (int i = 0; i < kKeys; ++i) {
    store->insert(vmemkv_test::padded_key(i, 8), static_cast<uint64_t>(i));
  }
  // Descending update order appends tail records in reverse-key sequence, so T2 physical
  // offset order is the opposite of key order: emitting reads in offset order would observably
  // reverse the callback sequence.
  for (int i = kKeys - 1; i >= 0; --i) {
    store->update(vmemkv_test::padded_key(i, 8), static_cast<uint64_t>(kKeys + i));
  }
  std::vector<std::pair<std::string, uint64_t>> seen;
  const size_t entry_count = store->scan(
      "k00000000", "k00000300", [&](std::span<const std::byte> key_bytes, std::span<const std::byte> value) {
        seen.emplace_back(vmemkv_test::span_to_string(key_bytes), test_util::decode_u64(value));
      });
  CHECK(entry_count == static_cast<size_t>(kKeys));
  REQUIRE(seen.size() == static_cast<size_t>(kKeys));
  for (int i = 0; i < kKeys; ++i) {
    CHECK(seen[static_cast<size_t>(i)].first == vmemkv_test::padded_key(i, 8));
    CHECK(seen[static_cast<size_t>(i)].second == static_cast<uint64_t>(kKeys + i));
  }
}

TEST_CASE_TEMPLATE("reorganize: CRUD still works", Store, STORE_TYPES) {
  auto store = StoreFactory<Store>::make();
  store->insert("a", 1);
  store->insert("b", 2);
  store->reorganize();
  CHECK(test_util::get_u64_sync(store, "a") == 1U);
  CHECK(test_util::get_u64_sync(store, "b") == 2U);
  CHECK(store->insert("c", 3));
  const size_t entry_count = store->scan("a", "d", [](std::span<const std::byte>, std::span<const std::byte>) {});
  CHECK(entry_count == 3U);
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE_TEMPLATE("long keys sharing a 16-byte prefix: CRUD", Store, STORE_TYPES) {
  // Same reinsert-after-remove limitation as above (this case reinserts key_one).
  if constexpr (std::is_same_v<Store, vmemkv::variants::VMemKV_LeanStore>) {
    MESSAGE("skipped for LeanStore: engine cannot reinsert removed keys");
    return;
  }
  auto store = StoreFactory<Store>::make();

  const std::string prefix = "0123456789abcdef";
  const std::string key_one = prefix + "-alpha";
  const std::string key_two = prefix + "-bravo";
  const std::string key_three = prefix + "-charlie";

  CHECK(store->insert(key_one, 1));
  CHECK(store->insert(key_two, 2));
  CHECK(store->insert(key_three, 3));

  CHECK(test_util::get_u64_sync(store, key_one) == 1U);
  CHECK(test_util::get_u64_sync(store, key_two) == 2U);
  CHECK(test_util::get_u64_sync(store, key_three) == 3U);

  CHECK(test_util::get_u64_sync(store, prefix) == vmemkv::STORE_NOT_FOUND);

  CHECK_FALSE(store->insert(key_two, 99));
  CHECK(test_util::get_u64_sync(store, key_two) == 2U);

  CHECK(store->update(key_two, 22));
  CHECK(test_util::get_u64_sync(store, key_one) == 1U);
  CHECK(test_util::get_u64_sync(store, key_two) == 22U);
  CHECK(test_util::get_u64_sync(store, key_three) == 3U);

  CHECK(store->remove(key_one));
  CHECK(test_util::get_u64_sync(store, key_one) == vmemkv::STORE_NOT_FOUND);
  CHECK(test_util::get_u64_sync(store, key_two) == 22U);
  CHECK(test_util::get_u64_sync(store, key_three) == 3U);

  CHECK(store->insert(key_one, 111));
  CHECK(test_util::get_u64_sync(store, key_one) == 111U);
}

// Offset64 must disambiguate matching prefixes by hash(full_key).
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("Offset64 + hash-index disambiguates long keys sharing a prefix") {
  using OffsetAppendMapIndex = vmemkv::T1Index<vmemkv::Config<>>;
  // Heap-allocate: Hash index embeds a large bucket array unsuited to the stack.
  auto idx = std::make_unique<OffsetAppendMapIndex>();

  const std::string prefix = "0123456789abcdef";
  const std::string key_one = prefix + "-alpha";
  const std::string key_two = prefix + "-bravo";

  CHECK(idx->put(vmemkv_test::to_span(key_one), 1) == OffsetAppendMapIndex::PutResult::Applied);
  CHECK(idx->put(vmemkv_test::to_span(key_two), 2) == OffsetAppendMapIndex::PutResult::Applied);
  CHECK(idx->get(vmemkv_test::to_span(key_one)) == 1U);
  CHECK(idx->get(vmemkv_test::to_span(key_two)) == 2U);
  CHECK(idx->put(vmemkv_test::to_span(key_one), 9) == OffsetAppendMapIndex::PutResult::Applied);
  CHECK(idx->get(vmemkv_test::to_span(key_one)) == 9U);

  idx->reorganize([](auto) {});
  CHECK(idx->get(vmemkv_test::to_span(key_one)) == 9U);
  CHECK(idx->get(vmemkv_test::to_span(key_two)) == 2U);
}

// Large byte values are only tested for stores backed by Tier 2.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE_TEMPLATE("large value (>= 64B): CRUD still works", Store, STORE_TYPES) {
  // LeanStore's BTreeVI has same-size-only in-place update and no insert-after-remove path,
  // so grow/shrink updates have no correct engine path (benchmark updates are same-size).
  if constexpr (std::is_same_v<Store, vmemkv::variants::VMemKV_LeanStore>) {
    MESSAGE("skipped for LeanStore: engine cannot grow/shrink values in place");
    return;
  }
  auto store = StoreFactory<Store>::make();

  const std::string v64(kValue64Bytes, 'a');
  const std::string v200(kValue200Bytes, 'b');

  CHECK(store->insert("key1", v64));
  auto got = vmemkv_test::get_optional_bytes(store, "key1");
  if (!got.has_value()) {
    FAIL("missing key1 after insert");
  }
  CHECK(vmemkv_test::span_to_string(vmemkv_test::as_span(*got)) == v64);  // NOLINT(bugprone-unchecked-optional-access)

  CHECK_FALSE(store->insert("key1", v200));
  got = vmemkv_test::get_optional_bytes(store, "key1");
  if (!got.has_value()) {
    FAIL("missing key1 after duplicate insert");
  }
  CHECK(vmemkv_test::span_to_string(vmemkv_test::as_span(*got)) == v64);  // NOLINT(bugprone-unchecked-optional-access)

  CHECK(store->update("key1", v200));
  got = vmemkv_test::get_optional_bytes(store, "key1");
  if (!got.has_value()) {
    FAIL("missing key1 after grow update");
  }
  CHECK(vmemkv_test::span_to_string(vmemkv_test::as_span(*got)) == v200);  // NOLINT(bugprone-unchecked-optional-access)

  constexpr std::size_t kShrinkValueBytes = 16;
  const std::string v16(kShrinkValueBytes, 'c');

  CHECK(store->update("key1", v16));
  got = vmemkv_test::get_optional_bytes(store, "key1");
  if (!got.has_value()) {
    FAIL("missing key1 after shrink update");
  }
  CHECK(vmemkv_test::span_to_string(vmemkv_test::as_span(*got)) == v16);  // NOLINT(bugprone-unchecked-optional-access)

  CHECK(store->insert("key2", v64));
  auto got_key2 = vmemkv_test::get_optional_bytes(store, "key2");
  if (!got_key2.has_value()) {
    FAIL("missing key2 after insert");
  }
  CHECK(vmemkv_test::span_to_string(vmemkv_test::as_span(*got_key2)) ==
        v64);  // NOLINT(bugprone-unchecked-optional-access)

  CHECK(store->remove("key2"));
  CHECK_FALSE(vmemkv_test::get_optional_bytes(store, "key2").has_value());
  auto got_key1 = vmemkv_test::get_optional_bytes(store, "key1");
  if (!got_key1.has_value()) {
    FAIL("missing key1 after key2 removal");
  }
  CHECK(vmemkv_test::span_to_string(vmemkv_test::as_span(*got_key1)) ==
        v16);  // NOLINT(bugprone-unchecked-optional-access)
}

TEST_CASE_TEMPLATE("large N: all keys retrievable", Store, STORE_TYPES) {
  auto store = StoreFactory<Store>::make();
  constexpr int key_count = 1000;
  for (int i = 0; i < key_count; ++i) {
    store->insert("k" + std::to_string(i), static_cast<uint64_t>(i));
  }
  for (int i = 0; i < key_count; ++i) {
    CHECK(test_util::get_u64_sync(store, "k" + std::to_string(i)) == static_cast<uint64_t>(i));
  }
}

TEST_CASE_TEMPLATE("[mt] concurrent reads are consistent", Store, STORE_TYPES) {
  auto store = StoreFactory<Store>::make();
  constexpr int key_count = 500;
  constexpr int kReaderThreadCount = 4;
  for (int i = 0; i < key_count; ++i) {
    store->insert("k" + std::to_string(i), static_cast<uint64_t>(i));
  }

  std::atomic<bool> all_ok{true};
  std::vector<std::thread> threads;
  threads.reserve(kReaderThreadCount);
  for (int thread_index = 0; thread_index < kReaderThreadCount; ++thread_index) {
    threads.emplace_back([&] {
      for (int i = 0; i < key_count; ++i) {
        uint64_t value = test_util::get_u64_sync(store, "k" + std::to_string(i));
        if (value != static_cast<uint64_t>(i)) {
          all_ok.store(false);
        }
      }
    });
  }
  for (auto &thread_handle : threads) {
    thread_handle.join();
  }
  CHECK(all_ok.load());
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE_TEMPLATE("scan with integral keys verifies lexicographical ordering", Store, STORE_TYPES) {
  auto store = StoreFactory<Store>::make();
  // Keys would sort wrong as strings (1 < 10 < 100 < 2); verifies numeric order instead.
  store->insert(kScanHundred, kScanHundred);
  store->insert(1U, 1U);
  store->insert(kScanTen, kScanTen);
  store->insert(2U, 2U);

  // No reorganize() here on purpose: scan() merges the sorted and append regions live and sorts
  // the combined result, so ordering holds regardless of which region each key currently sits in.
  std::vector<uint64_t> keys;
  const size_t scan_count =
      store->scan(kScanStart,
                  kScanUpperBound,
                  [&](std::span<const std::byte> key_bytes, [[maybe_unused]] std::span<const std::byte> value) {
                    // Decode big-endian key bytes back to uint32_t.
                    uint64_t key = 0;
                    for (size_t i = 0; i < 4; ++i) {
                      key = (key << kBitsPerByte) | static_cast<uint8_t>(key_bytes[i]);
                    }
                    keys.push_back(key);
                  });
  REQUIRE(scan_count == 4U);

  REQUIRE(keys.size() == 4U);
  CHECK(keys[0] == 1U);
  CHECK(keys[1] == 2U);
  CHECK(keys[2] == kScanTen);
  CHECK(keys[3] == kScanHundred);
}
