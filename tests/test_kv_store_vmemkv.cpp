// test_kv_store_vmemkv.cpp — vmemkv-specific store tests: checkpoint/base-region behavior,
// value inlining, torn-read regressions, and checkpoint/write-barrier races.
//
// The cross-backend matrix (every store type) lives in test_kv_store.cpp; shared fixture
// (StoreFactory, store-type lists, small helpers) in test_kv_store_fixture.hpp.

#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <thread>
#include <tuple>
#include <vector>
#include <vmemkv/variants.hpp>

#include "test_kv_store_fixture.hpp"
#include "test_support.hpp"

TEST_CASE("checkpoint churn during writes is transparent") {
  auto store = StoreFactory<VMemKV_FrequentCheckpoint>::make();

  constexpr int key_count = 25000;
  insert_sequential_u64_values(store, key_count);

  // The append-then-inline-to-non-inline update path should stay transparent even when
  // checkpoints trigger frequently in between.
  check_hot_key_upgrade_is_visible(store);

  check_sequential_u64_values(store, key_count);
}

// StoreAdapter::bulk_load() calls impl_.bulk_load_impl() directly, bypassing insert()/update()'s
// own path -- exercises that a bulk-loaded entry whose 8-byte value equals STORE_NOT_FOUND still
// round-trips via get()/scan() instead of being silently inlined and becoming indistinguishable
// from a tombstone under the default inline-value configuration.
TEST_CASE("bulk_load: an entry whose value equals STORE_NOT_FOUND round-trips via get() and scan()") {
  auto store = StoreFactory<vmemkv::variants::VMemKV_Default>::make();
  const std::string all_ff_value(8, '\xFF');
  store->bulk_load(
      1, [](std::size_t /*index*/) { return std::string("a"); }, [&](std::size_t /*index*/) { return all_ff_value; });

  bool found = store->get("a", [](std::span<const std::byte> /*val*/) {});
  CHECK(found);
  CHECK(test_util::get_u64_sync(store, "a") == vmemkv::STORE_NOT_FOUND);

  size_t scan_hits = 0;
  std::ignore = store->scan("a", "a~", [&](std::span<const std::byte> /*key*/, std::span<const std::byte> value) {
    REQUIRE(value.size() == sizeof(uint64_t));
    uint64_t val_u64 = 0;
    std::memcpy(&val_u64, value.data(), sizeof(uint64_t));
    CHECK(val_u64 == vmemkv::STORE_NOT_FOUND);
    ++scan_hits;
  });
  CHECK(scan_hits == 1U);
}

// Regression test: a same-size update targeting a record in T2's "base" region (already
// checkpointed) must be redirected out-of-place instead of taking the in-place fast path -- see
// update_impl()'s base_boundary check and T2Memory::base_boundary's declaration. Base-region reads
// go through a seqlock-free path (docs/specification/low_level_design.md 7.5) that assumes the bytes never change again
// once a record is base-resident; an in-place write there would violate that assumption. If the redirect were missing
// or wrong, this test would observe Scan (normal_mmap-served) and Get (random_mmap-served) silently disagreeing on
// "key_a"'s value.
TEST_CASE("vmemkv: update after reorganize redirects out-of-place, Scan sees fresh value") {
  auto store = StoreFactory<vmemkv::variants::VMemKV_Default>::make();

  const std::string value_a(kValue200Bytes, 'a');
  const std::string value_b(kValue200Bytes, 'b');
  REQUIRE(store->insert("key_a", value_a));
  REQUIRE(store->insert("key_b", value_b));

  store->checkpoint();  // Advances base_boundary: both records now sit in the base region.

  // Same-size update: would take the in-place fast path if key_a's offset weren't in the base.
  const std::string value_a_updated(kValue200Bytes, 'A');
  REQUIRE(store->update("key_a", value_a_updated));

  std::map<std::string, std::string> seen;
  std::ignore = store->scan("key_a", "key_c", [&](std::span<const std::byte> key, std::span<const std::byte> value) {
    seen.emplace(std::string(reinterpret_cast<const char *>(key.data()), key.size()),
                 std::string(reinterpret_cast<const char *>(value.data()), value.size()));
  });

  REQUIRE(seen.count("key_a") == 1);
  CHECK(seen.at("key_a") == value_a_updated);
  REQUIRE(seen.count("key_b") == 1);
  CHECK(seen.at("key_b") == value_b);

  // Get (random_mmap path) must agree with Scan (normal_mmap path) on the same key.
  const auto get_result = vmemkv_test::get_optional_bytes(store, "key_a");
  REQUIRE(get_result.has_value());
  CHECK(std::string(reinterpret_cast<const char *>(get_result->data()), get_result->size()) == value_a_updated);

  // A second checkpoint should absorb the out-of-place update into a new base region, and a
  // further same-size update should again be redirected rather than corrupting the new base.
  store->checkpoint();
  const std::string value_a_updated2(kValue200Bytes, 'Z');
  REQUIRE(store->update("key_a", value_a_updated2));
  std::map<std::string, std::string> seen2;
  std::ignore = store->scan("key_a", "key_c", [&](std::span<const std::byte> key, std::span<const std::byte> value) {
    seen2.emplace(std::string(reinterpret_cast<const char *>(key.data()), key.size()),
                  std::string(reinterpret_cast<const char *>(value.data()), value.size()));
  });
  REQUIRE(seen2.count("key_a") == 1);
  CHECK(seen2.at("key_a") == value_a_updated2);
}

namespace {

// Builds a store pre-filled with key_count uniform 'a' values for the checkpoint stress tests below.
template <typename TestStore>
auto make_seeded_checkpoint_stress_store(uint64_t capacity_bytes, int key_count, std::size_t value_bytes) {
  auto store = std::make_unique<TestStore>(reserve_temp_path().string(), capacity_bytes);
  for (int i = 0; i < key_count; ++i) {
    REQUIRE(store->insert("key" + std::to_string(i), std::string(value_bytes, 'a')));
  }
  return store;
}

// A value of copies of one repeated character; anything else is a torn read.
inline auto is_uniform(std::span<const std::byte> value) -> bool {
  if (value.empty()) {
    return true;
  }
  const auto expected = value[0];
  for (std::byte b : value) {
    if (b != expected) {
      return false;
    }
  }
  return true;
}

inline void check_uniform_value(std::span<const std::byte> value,
                                std::size_t value_bytes,
                                std::atomic<bool> &corruption_found) {
  if (value.size() != value_bytes || !is_uniform(value)) {
    corruption_found.store(true, std::memory_order_relaxed);
  }
}

// Every pre-seeded key must still be present at its full value size after the stress run.
template <typename StorePtr>
void check_final_value_sizes(StorePtr &store, int key_count, std::size_t value_bytes) {
  for (int i = 0; i < key_count; ++i) {
    const auto final_value = vmemkv_test::get_optional_bytes(store, "key" + std::to_string(i));
    REQUIRE(final_value.has_value());
    CHECK(final_value->size() == value_bytes);
  }
}

}  // namespace

// Regression test for base_boundary coverage across repeated checkpoint() cycles: each cycle
// advances base_boundary to cover everything durabilized so far, not just whatever the first
// forced checkpoint() established. Without this, a long-running insert-only process would have
// Scan's base-region coverage frozen forever at the first checkpoint, even as the live corpus
// kept growing past it.
TEST_CASE("vmemkv: checkpoint() extends base_boundary coverage across repeated cycles") {
  using TestStore = vmemkv::variants::VMemKV_Default;
  constexpr uint64_t kStoreCapacityBytes = 8ULL * 1024 * 1024;
  auto store = std::make_unique<TestStore>(reserve_temp_path().string(), kStoreCapacityBytes);

  const std::string value(kValue200Bytes, 'a');
  for (int i = 0; i < 10; ++i) {
    REQUIRE(store->insert(vmemkv_test::padded_key("key", i, 2), value));
  }
  store->checkpoint();  // First checkpoint: creates the T2 checkpoint file.

  const uint64_t boundary_after_first = store->impl().t2().get_memory()->base_boundary;
  CHECK(boundary_after_first > 0);
  CHECK(store->impl().t2().get_memory()->reads.normal_mmap != nullptr);
  CHECK(store->impl().t2().get_memory()->reads.sequential_mmap != nullptr);
  CHECK(store->impl().t2().get_memory()->reads.pread_fd >= 0);

  // Insert more, then checkpoint again -- base_boundary advances further in place.
  for (int i = 10; i < 20; ++i) {
    REQUIRE(store->insert(vmemkv_test::padded_key("key", i, 2), value));
  }
  store->checkpoint();

  const uint64_t boundary_after_second = store->impl().t2().get_memory()->base_boundary;
  CHECK(boundary_after_second > boundary_after_first);

  // Confirm the newly-promoted range is actually reachable and correct via the normal_mmap
  // path (Scan), not just the always-correct fallback (Get).
  std::map<std::string, std::string> seen;
  std::ignore = store->scan(vmemkv_test::padded_key("key", 0, 2),
                            "key19~",
                            [&](std::span<const std::byte> key, std::span<const std::byte> val) {
                              seen.emplace(std::string(reinterpret_cast<const char *>(key.data()), key.size()),
                                           std::string(reinterpret_cast<const char *>(val.data()), val.size()));
                            });
  for (int i = 0; i < 20; ++i) {
    REQUIRE(seen.count(vmemkv_test::padded_key("key", i, 2)) == 1);
    CHECK(seen.at(vmemkv_test::padded_key("key", i, 2)) == value);
  }
}

// update()'s in-place path racing repeated checkpoint() cycles must never observe a torn value.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE(
    "vmemkv: concurrent update+scan survive repeated cheap checkpoint() "
    "promotions (stress)") {
  using TestStore = vmemkv::variants::VMemKV_Default;
  constexpr uint64_t kStoreCapacityBytes = 8ULL * 1024 * 1024;
  constexpr int kKeyCount = 50;
  constexpr std::size_t kStressValueBytes = 200;
  constexpr int kCheckpointCycles = 60;
  constexpr int kOpsPerCycle = 50;

  auto store = make_seeded_checkpoint_stress_store<TestStore>(kStoreCapacityBytes, kKeyCount, kStressValueBytes);
  store->checkpoint();  // Establish an initial base region for the cheap path to keep promoting.

  std::atomic<bool> corruption_found{false};

  for (int cycle = 0; cycle < kCheckpointCycles; ++cycle) {
    std::thread updater([&, cycle] {
      std::mt19937 rng(static_cast<unsigned>(cycle));
      std::uniform_int_distribution<int> key_dist(0, kKeyCount - 1);
      for (int i = 0; i < kOpsPerCycle; ++i) {
        const std::string key = "key" + std::to_string(key_dist(rng));
        const std::string value(kStressValueBytes, static_cast<char>('a' + (i % 26)));
        store->update(key, value);
      }
    });
    std::thread scanner([&] {
      for (int i = 0; i < kOpsPerCycle; ++i) {
        std::ignore =
            store->scan("key0", "key9", [&](std::span<const std::byte> /*key*/, std::span<const std::byte> value) {
              check_uniform_value(value, kStressValueBytes, corruption_found);
            });
      }
    });

    // A fresh key each cycle so there's always new, not-yet-durable data for the cheap path to
    // actually promote (checkpoint() is a no-op promotion if nothing changed since the last one).
    // Runs on the main thread concurrently with updater/scanner above, joined only after.
    REQUIRE(store->insert("cycle" + std::to_string(cycle), std::string(kStressValueBytes, 'c')));
    store->checkpoint();

    updater.join();
    scanner.join();
  }

  CHECK_FALSE(corruption_found.load());

  check_final_value_sizes(store, kKeyCount, kStressValueBytes);
}

// Edge case: checkpoint() on an empty store still establishes normal_mmap/pread_fd.
TEST_CASE("vmemkv: checkpoint() on an empty store still establishes normal_mmap/pread_fd") {
  using TestStore = vmemkv::variants::VMemKV_Default;
  constexpr uint64_t kStoreCapacityBytes = 8ULL * 1024 * 1024;
  auto store = std::make_unique<TestStore>(reserve_temp_path().string(), kStoreCapacityBytes);

  store->checkpoint();  // No-op promotion: zero live entries, base_boundary stays at 0.
  CHECK(store->impl().t2().get_memory()->reads.normal_mmap != nullptr);
  CHECK(store->impl().t2().get_memory()->reads.sequential_mmap != nullptr);
  CHECK(store->impl().t2().get_memory()->reads.pread_fd >= 0);
  CHECK(store->impl().t2().get_memory()->base_boundary == 0);

  const std::string value(kValue200Bytes, 'a');
  for (int i = 0; i < 10; ++i) {
    REQUIRE(store->insert("key" + std::to_string(i), value));
  }
  store->checkpoint();  // Cheap path now: normal_mmap/pread_fd already existed, just needs promoting.

  CHECK(store->impl().t2().get_memory()->base_boundary > 0);

  std::map<std::string, std::string> seen;
  std::ignore = store->scan("key0", "key9~", [&](std::span<const std::byte> key, std::span<const std::byte> val) {
    seen.emplace(std::string(reinterpret_cast<const char *>(key.data()), key.size()),
                 std::string(reinterpret_cast<const char *>(val.data()), val.size()));
  });
  for (int i = 0; i < 10; ++i) {
    REQUIRE(seen.count("key" + std::to_string(i)) == 1);
    CHECK(seen.at("key" + std::to_string(i)) == value);
  }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("vmemkv: checkpoint preserves correctness across garbage from update/remove") {
  auto store = StoreFactory<vmemkv::variants::VMemKV_Default>::make();

  std::string val1(kValue100Bytes, 'x');
  std::string val2(kValue200Bytes, 'y');

  CHECK(store->insert("k1", val1));
  CHECK(store->insert("k2", val1));

  uint64_t bytes_used_before = store->t2().bytes_used();

  // Grow-update leaves the old value as garbage; remove leaves k1's record as garbage too.
  CHECK(store->update("k2", val2));
  uint64_t bytes_used_after_update = store->t2().bytes_used();
  CHECK(bytes_used_after_update > bytes_used_before);

  CHECK(store->remove("k1"));

  store->checkpoint();

  CHECK_FALSE(vmemkv_test::get_optional_bytes(store, "k1").has_value());
  auto got = vmemkv_test::get_optional_bytes(store, "k2");
  if (!got.has_value()) {
    FAIL("missing k2 after checkpoint");
  }
  CHECK(vmemkv_test::span_to_string(vmemkv_test::as_span(*got)) == val2);  // NOLINT(bugprone-unchecked-optional-access)
}

// try_make_inline_payload() excludes a key whose own last byte is 0x00 from inlining -- T1's
// 16-byte zero-padded prefix would make such a key's true length ambiguous with padding on
// scan_impl()'s inline-value fast path, so it takes the normal T2-record path instead, where the
// full key is stored verbatim. A big-endian-encoded integer key that's a multiple of 256 hits this
// directly (e.g. 256 encodes as {0x00,0x00,0x01,0x00}). This test only makes sense for a store
// with the default inline-value path enabled -- the case is specific to that fast path.
TEST_CASE("Value Inlining: scan() returns the untruncated key for a key ending in a zero byte") {
  auto store = StoreFactory<vmemkv::variants::VMemKV_Default>::make();
  store->insert(256U, 256U);  // big-endian encode(256) = {0x00,0x00,0x01,0x00} -- ends in 0x00.
  store->reorganize();

  std::vector<std::byte> observed_key;
  const size_t scan_count =
      store->scan(0U, 1000U, [&](std::span<const std::byte> key_bytes, std::span<const std::byte> /*value*/) {
        observed_key.assign(key_bytes.begin(), key_bytes.end());
      });
  REQUIRE(scan_count == 1U);
  REQUIRE(observed_key.size() == 4U);
  CHECK(static_cast<uint8_t>(observed_key[0]) == 0x00U);
  CHECK(static_cast<uint8_t>(observed_key[1]) == 0x00U);
  CHECK(static_cast<uint8_t>(observed_key[2]) == 0x01U);
  CHECK(static_cast<uint8_t>(observed_key[3]) == 0x00U);

  // get() must still find it via the exact key the caller already knows (unaffected by this fix
  // either way, since it never needed to recover the key from T1's prefix).
  const auto got = vmemkv_test::get_optional_bytes(store, 256U);
  REQUIRE(got.has_value());
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("Value Inlining: verify that short/8B-aligned values bypass T2 write paths") {
  const std::string path = reserve_temp_path().string();
  constexpr uint64_t kInlineStoreCapacityBytes = 8ULL * 1024ULL * 1024ULL;
  constexpr std::byte kShortFillByte{0xAB};
  constexpr std::byte kLongFillByte{0xCD};
  constexpr uint64_t kOddInlineValue = 0x003456789ABCDEF1ULL;
  constexpr uint64_t kEvenInlineValue = 0x123456789ABCDEF0ULL;

  using InlineStore = vmemkv::variants::VMemKV_Default;
  auto store = std::make_unique<InlineStore>(path, kInlineStoreCapacityBytes);

  uint64_t initial_bytes = store->t2().bytes_used();
  CHECK(initial_bytes == 0);

  // 1-7 bytes: should inline (no T2 usage).
  std::vector<std::byte> val_short(kShortInlineValueBytes, kShortFillByte);
  store->insert("key1", val_short);

  CHECK(store->t2().bytes_used() == 0);

  auto res = vmemkv_test::get_optional_bytes(store, "key1");
  if (!res.has_value()) {
    FAIL("missing key1 inline payload");
  }
  const auto &res_value = *res;  // NOLINT(bugprone-unchecked-optional-access)
  CHECK(res_value.size() == kShortInlineValueBytes);
  CHECK(res_value[0] == kShortFillByte);

  // 8-byte odd integer: should inline.
  uint64_t val_odd = kOddInlineValue;
  std::vector<std::byte> val_odd_bytes(kInlineValueBytes);
  std::memcpy(val_odd_bytes.data(), &val_odd, kInlineValueBytes);

  store->insert("key_odd", val_odd_bytes);
  CHECK(store->t2().bytes_used() == 0);

  auto res_odd = vmemkv_test::get_optional_bytes(store, "key_odd");
  if (!res_odd.has_value()) {
    FAIL("missing key_odd inline payload");
  }
  const auto &res_odd_value = *res_odd;  // NOLINT(bugprone-unchecked-optional-access)
  REQUIRE(res_odd_value.size() == kInlineValueBytes);
  uint64_t read_odd = 0;
  std::memcpy(&read_odd, res_odd_value.data(), kInlineValueBytes);
  CHECK(read_odd == val_odd);

  // 8-byte even integer: should also inline.
  uint64_t val_even = kEvenInlineValue;
  std::vector<std::byte> val_even_bytes(kInlineValueBytes);
  std::memcpy(val_even_bytes.data(), &val_even, kInlineValueBytes);

  store->insert("key_even", val_even_bytes);
  CHECK(store->t2().bytes_used() == 0);

  auto res_even = vmemkv_test::get_optional_bytes(store, "key_even");
  if (!res_even.has_value()) {
    FAIL("missing key_even inline payload");
  }
  const auto &res_even_value = *res_even;  // NOLINT(bugprone-unchecked-optional-access)
  REQUIRE(res_even_value.size() == kInlineValueBytes);
  uint64_t read_even = 0;
  std::memcpy(&read_even, res_even_value.data(), kInlineValueBytes);
  CHECK(read_even == val_even);

  // 9+ bytes: should bypass inlining and go to T2.
  std::vector<std::byte> val_long(kLongValueBytes, kLongFillByte);
  store->insert("key2", val_long);

  CHECK(store->t2().bytes_used() > 0);

  std::filesystem::remove(path);
  vmemkv::remove_wal_segments(vmemkv::derive_wal_path(path));
}

// Read callbacks observe an owned buffer copy validated by the seqlock, never live T2 memory.
namespace {

constexpr std::size_t kTornReadValueBytes = 200;  // Non-inline; same-size updates stay in the in-place path.

template <typename StorePtr, typename ReaderLoop>
void check_no_torn_read(StorePtr &store, ReaderLoop &&reader_loop) {
  REQUIRE(store->insert("hot", std::string(kTornReadValueBytes, 'a')));

  std::atomic<bool> stop{false};
  std::atomic<bool> torn_read_found{false};

  std::thread updater([&] {
    uint64_t i = 0;
    while (!stop.load(std::memory_order_relaxed)) {
      store->update("hot", std::string(kTornReadValueBytes, static_cast<char>('a' + (i++ % 26))));
    }
  });

  std::thread reader([&] {
    while (!stop.load(std::memory_order_relaxed)) {
      reader_loop(store, torn_read_found);
    }
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  stop.store(true, std::memory_order_relaxed);
  updater.join();
  reader.join();

  CHECK_FALSE(torn_read_found.load());
}

}  // namespace

TEST_CASE("vmemkv: scan callback never observes a torn read across a concurrent update (regression)") {
  using TestStore = vmemkv::variants::VMemKV_Default;
  constexpr uint64_t kStoreCapacityBytes = 8ULL * 1024 * 1024;
  auto store = std::make_unique<TestStore>(reserve_temp_path().string(), kStoreCapacityBytes);

  check_no_torn_read(store, [](auto &store, std::atomic<bool> &torn_read_found) {
    std::ignore = store->scan("hot", "hot~", [&](std::span<const std::byte> /*key*/, std::span<const std::byte> value) {
      if (!is_uniform(value)) {
        torn_read_found.store(true, std::memory_order_relaxed);
      }
    });
  });
}

// Same guarantee via get() instead of scan().
TEST_CASE("vmemkv: get callback never observes a torn read across a concurrent update (regression)") {
  using TestStore = vmemkv::variants::VMemKV_Default;
  constexpr uint64_t kStoreCapacityBytes = 8ULL * 1024 * 1024;
  auto store = std::make_unique<TestStore>(reserve_temp_path().string(), kStoreCapacityBytes);

  check_no_torn_read(store, [](auto &store, std::atomic<bool> &torn_read_found) {
    std::ignore = store->get("hot", [&](std::span<const std::byte> value) {
      if (!is_uniform(value)) {
        torn_read_found.store(true, std::memory_order_relaxed);
      }
    });
  });
}

// Generation-verified batched scan walk: concurrent same-size updates churn the
// in-place-write generation, forcing fallback batches, while scans must still
// observe only complete versions and converge exactly once writers join. Uses
// 32-byte values so updates stay out-of-place-free (in-place, tail region) and
// exercise the bracket; keys are reorganized into the sorted region first so scans
// take the batched fast path rather than the append slow path.
// Read-policy variants differ only in base-region read routing; this case never checkpoints,
// so no base region is ever formed and VMemKV_Default alone is representative.
TEST_CASE_TEMPLATE("scan under concurrent same-size updates observes exact versions",
                   Store,
                   vmemkv::variants::VMemKV_Default) {
  auto store = StoreFactory<Store>::make();
  constexpr int kKeys = 32;
  constexpr size_t kValBytes = 32;
  auto make_value = [](uint64_t version) {
    std::string value(kValBytes, '\0');
    for (size_t i = 0; i < kValBytes; ++i) {
      value[i] = static_cast<char>((version >> (8 * (i % 8))) & 0xFF);
    }
    return value;
  };
  auto decode_version = [](std::span<const std::byte> value) {
    uint64_t version = 0;
    for (size_t i = 0; i < sizeof(uint64_t); ++i) {
      version |= static_cast<uint64_t>(value[i]) << (8 * i);
    }
    return version;
  };
  for (int i = 0; i < kKeys; ++i) {
    REQUIRE(store->insert(vmemkv_test::padded_key(i, 8), make_value(0)));
  }
  store->reorganize();
  std::atomic<bool> stop{false};
  std::vector<std::thread> updaters;
  for (int t = 0; t < 2; ++t) {
    updaters.emplace_back([&, t] {
      uint64_t version = 1;
      while (!stop.load(std::memory_order_relaxed)) {
        const int key = (t + static_cast<int>(version)) % kKeys;
        if (store->update(vmemkv_test::padded_key(key, 8), make_value(version))) {
          ++version;
        }
      }
    });
  }
  for (int s = 0; s < 200; ++s) {
    std::ignore = store->scan(
        "k00000000", "k00000099", [&](std::span<const std::byte> key_bytes, std::span<const std::byte> value) {
          const int index = std::stoi(vmemkv_test::span_to_string(key_bytes).substr(1));
          REQUIRE(index >= 0);
          REQUIRE(index < kKeys);
          CHECK(value.size() == kValBytes);
          CHECK(vmemkv_test::span_to_string(value) == make_value(decode_version(value)));
        });
  }
  stop.store(true, std::memory_order_relaxed);
  for (auto &th : updaters) {
    th.join();
  }
  // Converged: every key reads back exactly (versions only ever increased, and the last
  // update per key wins).
  for (int i = 0; i < kKeys; ++i) {
    const auto got = vmemkv_test::get_optional_bytes(store, vmemkv_test::padded_key(i, 8));
    REQUIRE(got.has_value());
    CHECK(got->size() == kValBytes);
  }
}

TEST_CASE(
    "vmemkv: free-spinning update+scan survive with no reorganize/checkpoint "
    "at all (regression)") {
  using TestStore = vmemkv::variants::VMemKV_Default;
  constexpr uint64_t kStoreCapacityBytes = 8ULL * 1024 * 1024;
  auto store = std::make_unique<TestStore>(reserve_temp_path().string(), kStoreCapacityBytes);

  check_no_torn_read(store, [](auto &store, std::atomic<bool> &torn_read_found) {
    std::ignore = store->scan("hot", "hot~", [&](std::span<const std::byte> /*key*/, std::span<const std::byte> value) {
      if (!is_uniform(value)) {
        torn_read_found.store(true, std::memory_order_relaxed);
      }
    });
  });
}

namespace {

// Shared by the two writer-stop-barrier regression tests below: builds a "straggler" T2 record
// through an already-acquired write handle and publishes it into T1, exactly as
// write_entry_lockfree() would for a real append.
template <typename ImplT, typename StorePtr>
void publish_straggler_entry(StorePtr &store,
                             const vmemkv::T2FlatFile::T2MemoryHandle &mem,
                             const std::string &straggler_value) {
  kvs_detail::with_key_serialized(std::string("straggler"), [&](std::span<const std::byte> key_bytes) {
    kvs_detail::with_val_serialized(straggler_value, [&](std::span<const std::byte> val_bytes) {
      const uint64_t offset = vmemkv::T2FlatFile::append_default(mem, key_bytes, val_bytes);
      uint64_t encoded_payload = vmemkv::t2_detail::encode_offset_payload(
          offset, vmemkv::t2_detail::checked_block_count(key_bytes.size(), val_bytes.size()));
      REQUIRE(store->impl().t1().put(key_bytes, encoded_payload, false, 0) == ImplT::T1IndexT::PutResult::Applied);
    });
  });
}

// Shared trailing check: the straggler entry published above must be immediately readable, with
// no second checkpoint/reorganize cycle needed.
template <typename StorePtr>
void verify_straggler_readback(StorePtr &store, const std::string &straggler_value) {
  const auto straggler_readback = vmemkv_test::get_optional_bytes(store, "straggler");
  REQUIRE(straggler_readback.has_value());
  CHECK(straggler_readback->size() == straggler_value.size());
}

}  // namespace

// checkpoint's writer-stop barrier waits for in-flight writers before capturing the frontier.
TEST_CASE(
    "vmemkv: checkpoint's writer-stop barrier waits for an in-flight writer instead of "
    "capturing a frontier underneath it (regression)") {
  using TestStore = vmemkv::variants::VMemKV_Default;
  constexpr uint64_t kStoreCapacityBytes = 8ULL * 1024 * 1024;
  auto store = std::make_unique<TestStore>(reserve_temp_path().string(), kStoreCapacityBytes);

  // 200B: forces a real T2 record even under T1InlineValue (an inline payload never touches T2,
  // so wouldn't reach the write-handle registration this test targets).
  const std::string baseline_value(200, 'v');
  REQUIRE(store->insert("baseline", baseline_value));

  const std::string straggler_value(200, 's');
  std::thread straggler_writer;

  // pre_stop_hook fires strictly before the stop flag goes up, so the spawned thread's
  // acquire_write_handle() call is guaranteed to register with the reference tracker before
  // stop_writers_and_wait() ever scans it.
  using ImplT = std::decay_t<decltype(store->impl())>;
  store->impl().reorganize_internal(ImplT::ReorgMode::Checkpoint,
                                    /*pre_stop_hook=*/
                                    [&] {
                                      straggler_writer = std::thread([&] {
                                        vmemkv::T2FlatFile::T2MemoryHandle mem =
                                            store->impl().t2().acquire_write_handle();
                                        // Simulates being "mid-write": long enough that, absent the stop-and-wait, the
                                        // checkpoint would very likely have already captured its target by the time
                                        // this thread publishes.
                                        std::this_thread::sleep_for(std::chrono::milliseconds(20));
                                        publish_straggler_entry<ImplT>(store, mem, straggler_value);
                                        // mem released here -- only now can stop_writers_and_wait() (blocked on this
                                        // exact handle since before this thread even started sleeping) proceed.
                                      });
                                    });
  straggler_writer.join();

  // This same checkpoint cycle's captured target must already cover "straggler" -- reading it
  // back must work immediately, no second cycle needed.
  verify_straggler_readback(store, straggler_value);
}

// T2FlatFile::acquire_write_handle() registers with the reference tracker *before* checking
// writer_stop_, and retries if that check then finds the flag already true: registering first
// closes a window that ThreadReferenceTracker::wait_until_retired()'s single index-ordered scan
// (it never re-examines a slot once past it) would otherwise open -- a writer whose flag check ran
// before registering could be missed by an already-in-progress scan, letting stop_writers_and_wait()
// complete while that writer goes on to append past the frontier this same cycle just captured.
//
// Reproduces the adversarial timing deterministically via acquire_write_handle()'s hook seam
// (fires once, right after registering and before the writer_stop_ check): the writer thread
// registers, signals pre_stop_hook that it has done so, then sleeps -- guaranteeing
// stop_writers_and_wait() (called on the main thread right after pre_stop_hook returns) sets
// writer_stop_=true and starts scanning while this thread is still paused, already registered but
// not yet having checked the flag. The writer's own check must then see writer_stop_==true and
// back out/retry rather than proceed with a handle that might land past the frontier this same
// cycle already captured.
TEST_CASE(
    "vmemkv: acquire_write_handle()'s register-then-check-retry survives a writer racing the "
    "writer-stop scan (regression)") {
  using TestStore = vmemkv::variants::VMemKV_Default;
  constexpr uint64_t kStoreCapacityBytes = 8ULL * 1024 * 1024;
  auto store = std::make_unique<TestStore>(reserve_temp_path().string(), kStoreCapacityBytes);

  const std::string baseline_value(200, 'v');
  REQUIRE(store->insert("baseline", baseline_value));

  const std::string straggler_value(200, 's');
  std::thread straggler_writer;
  std::atomic<bool> writer_registered{false};

  using ImplT = std::decay_t<decltype(store->impl())>;
  store->impl().reorganize_internal(ImplT::ReorgMode::Checkpoint,
                                    /*pre_stop_hook=*/
                                    [&] {
                                      straggler_writer = std::thread([&] {
                                        // Only the *first* acquire_write_handle() attempt needs to pause here -- that's
                                        // the one stop_writers_and_wait() (below) is guaranteed to observe as "already
                                        // registered, haven't checked the flag yet" (the exact window this test
                                        // targets). A production caller's retries carry no such delay
                                        // (NoOpAcquireWriteHandleHook is instant), so they re-check writer_stop_ within
                                        // nanoseconds of it clearing; re-pausing on every retry here would be purely a
                                        // test-harness artifact -- and a bad one: it can stretch the *net* time this
                                        // slot spends showing a stale value far past what
                                        // ThreadReferenceTracker::wait_until_retired()'s SpinBackoff (yields, then 1ms
                                        // polls) is tuned to catch quickly, since each retry reopens only a
                                        // sub-microsecond release window against a 30ms-wide observation stride.
                                        bool first_attempt = true;
                                        vmemkv::T2FlatFile::T2MemoryHandle mem =
                                            store->impl().t2().acquire_write_handle([&] {
                                              writer_registered.store(true, std::memory_order_release);
                                              if (first_attempt) {
                                                first_attempt = false;
                                                // Long enough that stop_writers_and_wait() below is guaranteed to have
                                                // set writer_stop_=true and started its scan (finding this thread's
                                                // slot registered, hence blocking on it) before this thread wakes up
                                                // and checks the flag itself.
                                                std::this_thread::sleep_for(std::chrono::milliseconds(30));
                                              }
                                            });
                                        publish_straggler_entry<ImplT>(store, mem, straggler_value);
                                      });
                                      // Don't let pre_stop_hook return until the writer has registered -- otherwise
                                      // stop_writers_and_wait() might start scanning before the writer's slot is set at
                                      // all, which wouldn't exercise the "scan already saw/blocked on a registered
                                      // slot, writer only then wakes up and self-checks" path this test targets.
                                      while (!writer_registered.load(std::memory_order_acquire)) {
                                        std::this_thread::yield();
                                      }
                                    });
  straggler_writer.join();

  verify_straggler_readback(store, straggler_value);
}

// An exception before T1 publish during a T2 rebuild leaves the store fully usable.
TEST_CASE("vmemkv: exception before T1 publish during a T2 rebuild leaves the store fully usable") {
  using TestStore = vmemkv::variants::VMemKV_Default;
  constexpr uint64_t kStoreCapacityBytes = 8ULL * 1024 * 1024;
  auto store = std::make_unique<TestStore>(reserve_temp_path().string(), kStoreCapacityBytes);

  const std::string value(200, 'v');
  REQUIRE(store->insert("key", value));

  struct InjectedFault : std::runtime_error {
    InjectedFault() : std::runtime_error("injected fault before T1 publish") {}
  };

  bool threw = false;
  try {
    store->impl().reorganize_internal(std::decay_t<decltype(store->impl())>::ReorgMode::Checkpoint,
                                      /*pre_stop_hook=*/vmemkv::NoOpPreStopHook{},
                                      /*pre_finish_hook=*/[] { throw InjectedFault{}; });
  } catch (const InjectedFault &) {
    threw = true;  // Mirrors reorg_worker_loop()'s catch (...) {} -- swallow and move on.
  }
  REQUIRE(threw);

  // No hang, no rollback needed: T1 was never touched, so the store is immediately usable.
  const auto readback = vmemkv_test::get_optional_bytes(store, "key");
  REQUIRE(readback.has_value());
  CHECK(std::string(reinterpret_cast<const char *>(readback->data()), readback->size()) == value);

  REQUIRE(store->update("key", std::string(200, 'w')));

  // A subsequent, unfaulted checkpoint cycle must still succeed normally.
  store->checkpoint();
  const auto final_readback = vmemkv_test::get_optional_bytes(store, "key");
  REQUIRE(final_readback.has_value());
  CHECK(final_readback->size() == 200);
}

// Read-policy pins: every read path must round-trip, including the base-region
// paths a checkpointed corpus actually exercises. Small values stay inline (policy-independent);
// the 200B values force real T2 base-region reads through the pinned mapping or pread.
#define READ_PIN_STORES vmemkv::variants::VMemKV_ReadNormal, vmemkv::variants::VMemKV_ReadPread

TEST_CASE_TEMPLATE("read-policy pins round-trip through checkpointed base reads", Store, READ_PIN_STORES) {
  auto store = StoreFactory<Store>::make();

  const std::string small_value("abc");
  const std::string large_value(200, 'v');
  REQUIRE(store->insert("key_small", small_value));
  REQUIRE(store->insert("key_large", large_value));
  store->checkpoint();

  const auto small_back = vmemkv_test::get_optional_bytes(store, "key_small");
  REQUIRE(small_back.has_value());
  CHECK(vmemkv_test::span_to_string(vmemkv_test::as_span(*small_back)) == small_value);

  const auto large_back = vmemkv_test::get_optional_bytes(store, "key_large");
  REQUIRE(large_back.has_value());
  CHECK(large_back->size() == large_value.size());

  std::map<std::string, std::string> seen;
  const size_t scan_count =
      store->scan("key_large", "key_small~", [&](std::span<const std::byte> key, std::span<const std::byte> value) {
        seen.emplace(vmemkv_test::span_to_string(key), vmemkv_test::span_to_string(value));
      });
  CHECK(scan_count == 2U);
  CHECK(seen.at("key_small") == small_value);
  CHECK(seen.at("key_large") == large_value);
}
