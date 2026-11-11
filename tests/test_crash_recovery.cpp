// test_crash_recovery.cpp — WAL-backed crash recovery: "restart" means constructing a second
// store over the same path. Torn/corrupt WAL bytes are injected via find_active_wal_segment(),
// bypassing Wal, to simulate a crash mid-append.

#include <doctest/doctest.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <vector>
#include <vmemkv/checkpoint/checkpoint.hpp>
#include <vmemkv/checkpoint/paths.hpp>
#include <vmemkv/impl/hooks.hpp>
#include <vmemkv/impl/t2_ownership.hpp>
#include <vmemkv/variants.hpp>
#include <vmemkv/wal/wal.hpp>

#include "test_support.hpp"

namespace {

constexpr uint64_t kStoreCapacityBytes = 64ULL * 1024 * 1024;

using vmemkv_test::check_range;
using vmemkv_test::insert_range;

// Inserts, injects a torn tail, recovers, and verifies (plus post-recovery usability).
template <typename Store, typename InjectFn>
void check_torn_tail_discarded(const vmemkv_test::ScopedTempPath &path, int key_count, InjectFn &&inject) {
  {
    auto store = std::make_unique<Store>(path, kStoreCapacityBytes);
    insert_range(*store, key_count);
  }
  inject(path);
  {
    auto store = std::make_unique<Store>(path, kStoreCapacityBytes);
    check_range(*store, key_count);
    REQUIRE(store->insert("new_key", vmemkv_test::indexed_value(999)));
    const auto bytes = vmemkv_test::get_optional_bytes(store, "new_key");
    REQUIRE(bytes.has_value());
    CHECK(vmemkv_test::span_to_string(vmemkv_test::as_span(*bytes)) == vmemkv_test::indexed_value(999));
  }
}

}  // namespace

#define CRASH_RECOVERY_STORE_TYPES vmemkv::variants::VMemKV_Default

TEST_CASE_TEMPLATE("crash recovery: clean restart, empty store stays empty", Store, CRASH_RECOVERY_STORE_TYPES) {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_crash", vmemkv::remove_store_files);
  { auto store = std::make_unique<Store>(path, kStoreCapacityBytes); }
  {
    auto store = std::make_unique<Store>(path, kStoreCapacityBytes);
    CHECK_FALSE(vmemkv_test::get_optional_bytes(store, "anything").has_value());
    CHECK(store->insert("k", vmemkv_test::indexed_value(0)));
  }
}

TEST_CASE_TEMPLATE("crash recovery: clean restart recovers 100 inserted keys", Store, CRASH_RECOVERY_STORE_TYPES) {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_crash", vmemkv::remove_store_files);
  constexpr int kKeyCount = 100;
  {
    auto store = std::make_unique<Store>(path, kStoreCapacityBytes);
    insert_range(*store, kKeyCount);
  }
  {
    auto store = std::make_unique<Store>(path, kStoreCapacityBytes);
    check_range(*store, kKeyCount);
  }
}

TEST_CASE_TEMPLATE("crash recovery: update then restart persists only the latest value",
                   Store,
                   CRASH_RECOVERY_STORE_TYPES) {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_crash", vmemkv::remove_store_files);
  {
    auto store = std::make_unique<Store>(path, kStoreCapacityBytes);
    REQUIRE(store->insert("k", vmemkv_test::indexed_value(0)));
    REQUIRE(store->update("k", vmemkv_test::indexed_value(1)));
    REQUIRE(store->update("k", vmemkv_test::indexed_value(2)));
  }
  {
    auto store = std::make_unique<Store>(path, kStoreCapacityBytes);
    const auto val = vmemkv_test::get_optional_bytes(store, "k");
    REQUIRE(val.has_value());
    CHECK(vmemkv_test::span_to_string(vmemkv_test::as_span(*val)) == vmemkv_test::indexed_value(2));
  }
}

TEST_CASE_TEMPLATE("crash recovery: delete then restart tombstone is durable, re-insert works",
                   Store,
                   CRASH_RECOVERY_STORE_TYPES) {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_crash", vmemkv::remove_store_files);
  {
    auto store = std::make_unique<Store>(path, kStoreCapacityBytes);
    REQUIRE(store->insert("k", vmemkv_test::indexed_value(0)));
    REQUIRE(store->remove("k"));
  }
  {
    auto store = std::make_unique<Store>(path, kStoreCapacityBytes);
    CHECK_FALSE(vmemkv_test::get_optional_bytes(store, "k").has_value());
    REQUIRE(store->insert("k", vmemkv_test::indexed_value(1)));
    const auto val = vmemkv_test::get_optional_bytes(store, "k");
    REQUIRE(val.has_value());
    CHECK(vmemkv_test::span_to_string(vmemkv_test::as_span(*val)) == vmemkv_test::indexed_value(1));
  }
}

TEST_CASE_TEMPLATE("crash recovery: insert-delete-reinsert-update replays to final state only",
                   Store,
                   CRASH_RECOVERY_STORE_TYPES) {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_crash", vmemkv::remove_store_files);
  {
    auto store = std::make_unique<Store>(path, kStoreCapacityBytes);
    REQUIRE(store->insert("k", vmemkv_test::indexed_value(0)));
    REQUIRE(store->remove("k"));
    REQUIRE(store->insert("k", vmemkv_test::indexed_value(1)));
    REQUIRE(store->update("k", vmemkv_test::indexed_value(2)));
  }
  {
    auto store = std::make_unique<Store>(path, kStoreCapacityBytes);
    const auto val = vmemkv_test::get_optional_bytes(store, "k");
    REQUIRE(val.has_value());
    CHECK(vmemkv_test::span_to_string(vmemkv_test::as_span(*val)) == vmemkv_test::indexed_value(2));
  }
}

TEST_CASE_TEMPLATE("crash recovery: torn trailing partial header discarded, prior writes recovered, store usable",
                   Store,
                   CRASH_RECOVERY_STORE_TYPES) {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_crash", vmemkv::remove_store_files);
  check_torn_tail_discarded<Store>(path, 20, [](const auto &wal_path) { vmemkv_test::append_torn_header(wal_path); });
}

TEST_CASE_TEMPLATE(
    "crash recovery: torn trailing full-header-truncated-payload discarded, prior writes recovered, store usable",
    Store,
    CRASH_RECOVERY_STORE_TYPES) {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_crash", vmemkv::remove_store_files);
  check_torn_tail_discarded<Store>(path, 20, [](const auto &wal_path) {
    vmemkv::append_truncated_record_for_test(vmemkv_test::resolve_active_wal_segment(wal_path), 10, "xyz");
  });
}

TEST_CASE_TEMPLATE("crash recovery: corrupted checksum on last record discarded, earlier keys intact",
                   Store,
                   CRASH_RECOVERY_STORE_TYPES) {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_crash", vmemkv::remove_store_files);
  constexpr int kKeyCount = 10;
  uint64_t offset_before_last = 0;
  {
    auto store = std::make_unique<Store>(path, kStoreCapacityBytes);
    insert_range(*store, kKeyCount);
    const auto active = vmemkv::find_active_wal_segment(vmemkv::derive_wal_path(path));
    REQUIRE(active.has_value());
    offset_before_last = std::filesystem::file_size(*active);
    REQUIRE(store->insert("doomed", vmemkv_test::indexed_value(999)));
  }
  vmemkv_test::flip_byte_at(vmemkv_test::resolve_active_wal_segment(path),
                            static_cast<std::streamoff>(offset_before_last) +
                                static_cast<std::streamoff>(offsetof(vmemkv::WalRecordHeader, checksum)));
  {
    auto store = std::make_unique<Store>(path, kStoreCapacityBytes);
    check_range(*store, kKeyCount);
    CHECK_FALSE(vmemkv_test::get_optional_bytes(store, "doomed").has_value());
  }
}

TEST_CASE_TEMPLATE("crash recovery: writes after recovery remain durable across a second restart",
                   Store,
                   CRASH_RECOVERY_STORE_TYPES) {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_crash", vmemkv::remove_store_files);
  constexpr int kFirstBatch = 20;
  constexpr int kSecondBatch = 20;
  {
    auto store = std::make_unique<Store>(path, kStoreCapacityBytes);
    for (int i = 0; i < kFirstBatch; ++i) {
      REQUIRE(store->insert("a" + std::to_string(i), vmemkv_test::indexed_value(i)));
    }
  }
  {
    auto store = std::make_unique<Store>(path, kStoreCapacityBytes);
    for (int i = 0; i < kFirstBatch; ++i) {
      REQUIRE(vmemkv_test::get_optional_bytes(store, "a" + std::to_string(i)).has_value());
    }
    for (int i = 0; i < kSecondBatch; ++i) {
      REQUIRE(store->insert("b" + std::to_string(i), vmemkv_test::indexed_value(1000 + i)));
    }
  }
  {
    auto store = std::make_unique<Store>(path, kStoreCapacityBytes);
    for (int i = 0; i < kFirstBatch; ++i) {
      const auto val = vmemkv_test::get_optional_bytes(store, "a" + std::to_string(i));
      REQUIRE(val.has_value());
      CHECK(vmemkv_test::span_to_string(vmemkv_test::as_span(*val)) == vmemkv_test::indexed_value(i));
    }
    for (int i = 0; i < kSecondBatch; ++i) {
      const auto val = vmemkv_test::get_optional_bytes(store, "b" + std::to_string(i));
      REQUIRE(val.has_value());
      CHECK(vmemkv_test::span_to_string(vmemkv_test::as_span(*val)) == vmemkv_test::indexed_value(1000 + i));
    }
  }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE_TEMPLATE("crash recovery: concurrent inserts before crash all recover", Store, CRASH_RECOVERY_STORE_TYPES) {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_crash", vmemkv::remove_store_files);
  constexpr int kThreadCount = 8;
  constexpr int kPerThread = 200;
  std::atomic<bool> all_inserts_ok{true};
  {
    auto store = std::make_unique<Store>(path, kStoreCapacityBytes);
    std::vector<std::thread> threads;
    threads.reserve(kThreadCount);
    for (int thread_index = 0; thread_index < kThreadCount; ++thread_index) {
      threads.emplace_back([&store, &all_inserts_ok, thread_index]() {
        for (int i = 0; i < kPerThread; ++i) {
          const std::string key = "t" + std::to_string(thread_index) + "_" + std::to_string(i);
          if (!store->insert(key, vmemkv_test::indexed_value(thread_index * kPerThread + i))) {
            all_inserts_ok.store(false, std::memory_order_relaxed);
          }
        }
      });
    }
    for (auto &thread : threads) {
      thread.join();
    }
  }
  CHECK(all_inserts_ok.load());

  {
    auto store = std::make_unique<Store>(path, kStoreCapacityBytes);
    for (int thread_index = 0; thread_index < kThreadCount; ++thread_index) {
      for (int i = 0; i < kPerThread; ++i) {
        const std::string key = "t" + std::to_string(thread_index) + "_" + std::to_string(i);
        const auto val = vmemkv_test::get_optional_bytes(store, key);
        REQUIRE(val.has_value());
        CHECK(vmemkv_test::span_to_string(vmemkv_test::as_span(*val)) ==
              vmemkv_test::indexed_value(thread_index * kPerThread + i));
      }
    }
  }
}

TEST_CASE_TEMPLATE("crash recovery: delete-heavy workload restarts to correct split, reorganize still works",
                   Store,
                   CRASH_RECOVERY_STORE_TYPES) {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_crash", vmemkv::remove_store_files);
  constexpr int kInsertCount = 100;
  constexpr int kRemoveCount = 50;
  {
    auto store = std::make_unique<Store>(path, kStoreCapacityBytes);
    for (int i = 0; i < kInsertCount; ++i) {
      REQUIRE(store->insert("k" + std::to_string(i), vmemkv_test::indexed_value(i)));
    }
    for (int i = 0; i < kRemoveCount; ++i) {
      REQUIRE(store->remove("k" + std::to_string(i)));
    }
  }
  {
    auto store = std::make_unique<Store>(path, kStoreCapacityBytes);
    int live_count = 0;
    for (int i = 0; i < kInsertCount; ++i) {
      const auto val = vmemkv_test::get_optional_bytes(store, "k" + std::to_string(i));
      if (i < kRemoveCount) {
        CHECK_FALSE(val.has_value());
      } else {
        REQUIRE(val.has_value());
        CHECK(vmemkv_test::span_to_string(vmemkv_test::as_span(*val)) == vmemkv_test::indexed_value(i));
        ++live_count;
      }
    }
    CHECK(live_count == kInsertCount - kRemoveCount);

    store->reorganize();  // T1-only fast path.

    for (int i = kRemoveCount; i < kInsertCount; ++i) {
      const auto val = vmemkv_test::get_optional_bytes(store, "k" + std::to_string(i));
      REQUIRE(val.has_value());
      CHECK(vmemkv_test::span_to_string(vmemkv_test::as_span(*val)) == vmemkv_test::indexed_value(i));
    }
  }
}

namespace {
// 256-entry append region: shrinks append-region capacity to force frequent reorganizes.
using TinyAppendConfig = vmemkv_test::TinyAppendConfig<8>;

using VMemKV_TinyAppend = vmemkv::StoreAdapter<vmemkv::VMemKVImpl<TinyAppendConfig>>;
}  // namespace

// Livelock guard: during recovery no reorg worker exists yet, so replay into a full append
// region (256 entries here) must not hang waiting for a background drain.
TEST_CASE("crash recovery: recovery under tiny T1 append-region capacity does not hang") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_crash", vmemkv::remove_store_files);
  constexpr int kKeyCount = 5000;
  {
    auto store = std::make_unique<VMemKV_TinyAppend>(path, kStoreCapacityBytes);
    for (int i = 0; i < kKeyCount; ++i) {
      REQUIRE(store->insert("k" + std::to_string(i), vmemkv_test::indexed_value(i)));
    }
  }
  {
    auto store = std::make_unique<VMemKV_TinyAppend>(path, kStoreCapacityBytes);
    for (int i = 0; i < kKeyCount; ++i) {
      const auto val = vmemkv_test::get_optional_bytes(store, "k" + std::to_string(i));
      REQUIRE(val.has_value());
      CHECK(vmemkv_test::span_to_string(vmemkv_test::as_span(*val)) == vmemkv_test::indexed_value(i));
    }
  }
}

namespace {
// >16 bytes unconditionally bypasses T1's inline-value optimization, forcing every write
// through T2 so a tiny T2 capacity is what actually gets exhausted below.
auto capacity_test_key(int index) -> std::string { return "key_over16bytes_" + std::to_string(index); }
}  // namespace

// WAL logging happens only after the T1/T2 mutation applies: logging first would leave a
// phantom record behind a failed mutation (e.g. T2 capacity exceeded), bricking every future
// restart on the same replay throw.
TEST_CASE("crash recovery: insert failing on T2 capacity exceeded leaves no phantom WAL record") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_crash", vmemkv::remove_store_files);
  constexpr uint64_t kTinyT2Capacity = 200;  // Room for 3 of these ~56B records, not 4.

  bool insert_threw = false;
  {
    auto store = std::make_unique<vmemkv::variants::VMemKV_Default>(path, kTinyT2Capacity);
    for (int i = 0; i < 3; ++i) {
      REQUIRE(store->insert(capacity_test_key(i), std::string("01234567")));
    }
    try {
      store->insert(capacity_test_key(99), std::string("01234567"));
    } catch (const std::exception &) {
      insert_threw = true;
    }
  }
  REQUIRE(insert_threw);  // Sanity: T2 capacity is genuinely exhausted by the 4th insert.

  // Restarting with the same capacity must succeed outright (no phantom WAL record to replay
  // for the key whose insert() threw), and that key must be absent from the recovered store.
  auto store = std::make_unique<vmemkv::variants::VMemKV_Default>(path, kTinyT2Capacity);
  for (int i = 0; i < 3; ++i) {
    const auto val = vmemkv_test::get_optional_bytes(store, capacity_test_key(i));
    REQUIRE(val.has_value());
    CHECK(vmemkv_test::span_to_string(vmemkv_test::as_span(*val)) == "01234567");
  }
  CHECK_FALSE(vmemkv_test::get_optional_bytes(store, capacity_test_key(99)).has_value());
}

// Checkpoint durability: appends landing after the checkpoint's target capture
// are excluded from the T1 dump and must come back through WAL replay.
// The pre-finish hook runs after msync and before the T1 dump, so appending
// inside it deterministically places records past the target.
TEST_CASE("checkpoint with concurrent appends recovers all keys") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_crash", vmemkv::remove_store_files);
  constexpr int kBaseKeys = 200;
  constexpr int kLateKeys = 100;
  const std::string big_value(1024, 'v');
  const std::string late_value(1024, 'w');
  using ImplT = vmemkv::VMemKVImpl<vmemkv::Config<>>;
  {
    auto store = std::make_unique<vmemkv::variants::VMemKV_Default>(path, kStoreCapacityBytes);
    for (int i = 0; i < kBaseKeys; ++i) {
      REQUIRE(store->insert("base_" + std::to_string(i), big_value));
    }
    store->checkpoint();
    store->impl().reorganize_internal(ImplT::ReorgMode::Checkpoint, vmemkv::NoOpPreStopHook{}, [&] {
      for (int i = 0; i < kLateKeys; ++i) {
        if (!store->insert("late_" + std::to_string(i), late_value)) {
          throw std::runtime_error("late append failed");
        }
      }
    });
    // The hook appends landed past the capture target: their offsets must lie
    // at or beyond the committed manifest frontier, proving the T1 dump could
    // not have carried them and recovery must replay the WAL.
    const auto manifest = vmemkv::read_manifest(vmemkv::derive_manifest_path(path));
    REQUIRE(manifest.has_value());
    for (int i = 0; i < kLateKeys; i += 10) {
      const auto res = store->impl().t1().get_with_hash(vmemkv_test::to_span("late_" + std::to_string(i)));
      REQUIRE(res.payload_bits != vmemkv::STORE_NOT_FOUND);
      CHECK((res.payload_bits & vmemkv::t2_detail::kPayloadOffsetMask) >= manifest->t2_bytes_used);
    }
  }
  {
    auto store = std::make_unique<vmemkv::variants::VMemKV_Default>(path, kStoreCapacityBytes);
    for (int i = 0; i < kBaseKeys; ++i) {
      const auto val = vmemkv_test::get_optional_bytes(store, "base_" + std::to_string(i));
      REQUIRE(val.has_value());
      CHECK(vmemkv_test::span_to_string(vmemkv_test::as_span(*val)) == big_value);
    }
    for (int i = 0; i < kLateKeys; ++i) {
      const auto val = vmemkv_test::get_optional_bytes(store, "late_" + std::to_string(i));
      REQUIRE(val.has_value());
      CHECK(vmemkv_test::span_to_string(vmemkv_test::as_span(*val)) == late_value);
    }
  }
}

// Same bug, but through update_impl()'s write_entry_lockfree() fallback path (value grew too
// large for the in-place t2_.update_value_at() branch, forcing a fresh T2 append instead).
TEST_CASE("crash recovery: update failing on T2 capacity exceeded leaves no phantom WAL record") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_crash", vmemkv::remove_store_files);
  constexpr uint64_t kTinyT2Capacity = 200;  // 2 inserts (112B) leaves 88B -- not enough for a 96B growth append.

  bool update_threw = false;
  {
    auto store = std::make_unique<vmemkv::variants::VMemKV_Default>(path, kTinyT2Capacity);
    REQUIRE(store->insert(capacity_test_key(0), std::string("01234567")));
    REQUIRE(store->insert(capacity_test_key(1), std::string("01234567")));
    try {
      store->update(capacity_test_key(0), std::string(50, 'x'));
    } catch (const std::exception &) {
      update_threw = true;
    }
  }
  REQUIRE(update_threw);  // Sanity: the growth-triggered re-append genuinely exceeds T2 capacity.

  // Restarting must succeed, and key 0 must still hold its pre-update value (the failed update
  // left no phantom WAL record to replay).
  auto store = std::make_unique<vmemkv::variants::VMemKV_Default>(path, kTinyT2Capacity);
  const auto val0 = vmemkv_test::get_optional_bytes(store, capacity_test_key(0));
  REQUIRE(val0.has_value());
  CHECK(vmemkv_test::span_to_string(vmemkv_test::as_span(*val0)) == "01234567");
  const auto val1 = vmemkv_test::get_optional_bytes(store, capacity_test_key(1));
  REQUIRE(val1.has_value());
  CHECK(vmemkv_test::span_to_string(vmemkv_test::as_span(*val1)) == "01234567");
}
