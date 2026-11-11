// test_wal.cpp — Standalone Wal: append/fsync, LSN monotonicity across reopen, replay order,
// torn/corrupt tail recovery.

#include <doctest/doctest.h>
#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <vmemkv/wal/wal.hpp>

#include "test_support.hpp"

namespace {

// Runs `fn` on a thread; true iff it finishes within `timeout` (then joined). Detaches only
// on timeout (already failed): unconditional detach risks a detached thread's TLS teardown
// aliasing a later TEST_CASE's reused stack.
template <typename Fn>
auto run_with_timeout(Fn &&fn, std::chrono::milliseconds timeout) -> bool {
  auto promise = std::make_shared<std::promise<void>>();
  std::future<void> future = promise->get_future();
  std::thread worker([fn = std::forward<Fn>(fn), promise]() mutable {
    fn();
    promise->set_value();
  });
  const bool completed = future.wait_for(timeout) == std::future_status::ready;
  if (completed) {
    worker.join();
  } else {
    worker.detach();
  }
  return completed;
}

// Joins every thread within `timeout` total; true iff all joined. The worker owns the moved
// threads (via shared_ptr) and keeps joining after detach -- only it ever touches them, so
// the caller must never join stragglers itself (would race the worker's join).
auto join_all_with_timeout(std::vector<std::thread> &threads, std::chrono::milliseconds timeout) -> bool {
  auto owned_threads = std::make_shared<std::vector<std::thread>>(std::move(threads));
  return run_with_timeout(
      [owned_threads]() {
        for (auto &thread : *owned_threads) {
          thread.join();
        }
      },
      timeout);
}

using vmemkv_test::as_span;
using vmemkv_test::bytes_of;
using vmemkv_test::span_to_string;

// Thin wrappers over reserve_*()/await_durable() for tests that only need "durably appended".
auto append_insert(vmemkv::Wal &wal, std::span<const std::byte> key, std::span<const std::byte> value) -> uint64_t {
  return wal.await_durable(wal.reserve_insert(key, value));
}

auto append_update(vmemkv::Wal &wal, std::span<const std::byte> key, std::span<const std::byte> value) -> uint64_t {
  return wal.await_durable(wal.reserve_update(key, value));
}

auto append_delete(vmemkv::Wal &wal, std::span<const std::byte> key) -> uint64_t {
  return wal.await_durable(wal.reserve_delete(key));
}

struct ReplayedRecord {
  vmemkv::WalRecordType type;
  std::string key;
  std::string value;
  uint64_t lsn;
};

// Spawns thread_count joinable threads of per_thread iteration(t, i) calls. The iteration
// callable moves into each thread (a call-site temporary would dangle otherwise).
template <typename Iteration>
auto run_concurrent_appends(int thread_count, int per_thread, Iteration &&iteration) -> std::vector<std::thread> {
  std::vector<std::thread> threads;
  threads.reserve(static_cast<size_t>(thread_count));
  for (int thread_index = 0; thread_index < thread_count; ++thread_index) {
    threads.emplace_back([call = std::forward<Iteration>(iteration), thread_index, per_thread]() mutable {
      for (int i = 0; i < per_thread; ++i) {
        call(thread_index, i);
      }
    });
  }
  return threads;
}

// Replays wal into an LSN-keyed map and checks entry-for-entry equality.
void check_replay_matches(vmemkv::Wal &wal, const std::unordered_map<uint64_t, ReplayedRecord> &recorded_by_lsn) {
  std::unordered_map<uint64_t, ReplayedRecord> replayed_by_lsn;
  wal.replay(
      [&](vmemkv::WalRecordType type, std::span<const std::byte> key, std::span<const std::byte> value, uint64_t lsn) {
        replayed_by_lsn[lsn] = ReplayedRecord{type, span_to_string(key), span_to_string(value), lsn};
      });

  REQUIRE(replayed_by_lsn.size() == recorded_by_lsn.size());
  for (const auto &[lsn, expected] : recorded_by_lsn) {
    const auto it = replayed_by_lsn.find(lsn);
    REQUIRE(it != replayed_by_lsn.end());
    CHECK(it->second.type == expected.type);
    CHECK(it->second.key == expected.key);
    CHECK(it->second.value == expected.value);
  }
}

// Replays wal into an LSN-keyed key map and checks every recorded key round-trips.
void check_replay_matches(vmemkv::Wal &wal, const std::unordered_map<uint64_t, std::string> &recorded_by_lsn) {
  std::unordered_map<uint64_t, std::string> replayed_by_lsn;
  const uint64_t count = wal.replay([&](vmemkv::WalRecordType /*type*/,
                                        std::span<const std::byte> key,
                                        std::span<const std::byte> /*value*/,
                                        uint64_t lsn) { replayed_by_lsn[lsn] = span_to_string(key); });

  CHECK(count == static_cast<uint64_t>(recorded_by_lsn.size()));
  REQUIRE(replayed_by_lsn.size() == recorded_by_lsn.size());
  for (const auto &[lsn, expected_key] : recorded_by_lsn) {
    const auto it = replayed_by_lsn.find(lsn);
    REQUIRE(it != replayed_by_lsn.end());
    CHECK(it->second == expected_key);
  }
}

}  // namespace

TEST_CASE("Wal: fresh file starts at LSN 1 with empty replay") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_wal_test", vmemkv_test::cleanup_wal_segments);
  vmemkv::Wal wal(path);
  CHECK(wal.next_lsn() == 1);

  const uint64_t count = wal.replay([](vmemkv::WalRecordType /*type*/,
                                       std::span<const std::byte> /*key*/,
                                       std::span<const std::byte> /*value*/,
                                       uint64_t /*lsn*/) { FAIL("replay callback invoked on empty WAL"); });
  CHECK(count == 0);
}

TEST_CASE("Wal: insert/update/delete assign strictly increasing LSNs") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_wal_test", vmemkv_test::cleanup_wal_segments);
  vmemkv::Wal wal(path);

  const auto key = bytes_of("k1");
  const auto val = bytes_of("v1");

  // Relative ordering only: strictly increasing per call is the contract, not exact numbers.
  const uint64_t lsn1 = append_insert(wal, as_span(key), as_span(val));
  const uint64_t lsn2 = append_update(wal, as_span(key), as_span(val));
  const uint64_t lsn3 = append_delete(wal, as_span(key));

  CHECK(lsn1 < lsn2);
  CHECK(lsn2 < lsn3);
  CHECK(wal.next_lsn() > lsn3);
}

TEST_CASE("Wal: replay after reopen round-trips type/key/value/order") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_wal_test", vmemkv_test::cleanup_wal_segments);
  {
    vmemkv::Wal wal(path);
    append_insert(wal, as_span(bytes_of("a")), as_span(bytes_of("1")));
    append_update(wal, as_span(bytes_of("b")), as_span(bytes_of("22")));
    append_delete(wal, as_span(bytes_of("c")));
  }

  vmemkv::Wal wal(path);
  std::vector<ReplayedRecord> records;
  const uint64_t count = wal.replay(
      [&](vmemkv::WalRecordType type, std::span<const std::byte> key, std::span<const std::byte> value, uint64_t lsn) {
        records.push_back(ReplayedRecord{type, span_to_string(key), span_to_string(value), lsn});
      });

  CHECK(count == 3);
  REQUIRE(records.size() == 3);

  CHECK(records[0].type == vmemkv::WalRecordType::Insert);
  CHECK(records[0].key == "a");
  CHECK(records[0].value == "1");

  CHECK(records[1].type == vmemkv::WalRecordType::Update);
  CHECK(records[1].key == "b");
  CHECK(records[1].value == "22");

  CHECK(records[2].type == vmemkv::WalRecordType::Delete);
  CHECK(records[2].key == "c");
  CHECK(records[2].value.empty());

  // Relative ordering is the contract; exact numbers are an implementation detail.
  CHECK(records[0].lsn < records[1].lsn);
  CHECK(records[1].lsn < records[2].lsn);
}

TEST_CASE("Wal: delete record replays with empty value span") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_wal_test", vmemkv_test::cleanup_wal_segments);
  vmemkv::Wal wal(path);
  append_delete(wal, as_span(bytes_of("gone")));

  bool saw_record = false;
  wal.replay([&](vmemkv::WalRecordType type,
                 std::span<const std::byte> key,
                 std::span<const std::byte> value,
                 uint64_t /*lsn*/) {
    saw_record = true;
    CHECK(type == vmemkv::WalRecordType::Delete);
    CHECK(span_to_string(key) == "gone");
    CHECK(value.empty());
  });
  CHECK(saw_record);
}

TEST_CASE("Wal: LSN numbering continues across reopen, does not reset") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_wal_test", vmemkv_test::cleanup_wal_segments);
  uint64_t last_lsn_before_reopen = 0;
  {
    vmemkv::Wal wal(path);
    append_insert(wal, as_span(bytes_of("a")), as_span(bytes_of("1")));
    last_lsn_before_reopen = append_insert(wal, as_span(bytes_of("b")), as_span(bytes_of("2")));
  }

  vmemkv::Wal wal(path);
  CHECK(wal.next_lsn() > last_lsn_before_reopen);
  const uint64_t lsn = append_insert(wal, as_span(bytes_of("c")), as_span(bytes_of("3")));
  CHECK(lsn > last_lsn_before_reopen);
}

TEST_CASE("Wal: repeated open/close with zero appends stays empty") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_wal_test", vmemkv_test::cleanup_wal_segments);
  { vmemkv::Wal wal(path); }
  {
    vmemkv::Wal wal(path);
    CHECK(wal.next_lsn() == 1);
  }

  vmemkv::Wal wal(path);
  const uint64_t count = wal.replay([](vmemkv::WalRecordType /*type*/,
                                       std::span<const std::byte> /*key*/,
                                       std::span<const std::byte> /*value*/,
                                       uint64_t /*lsn*/) { FAIL("replay callback invoked on empty WAL"); });
  CHECK(count == 0);
}

TEST_CASE("Wal: torn trailing partial header is discarded and file truncated") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_wal_test", vmemkv_test::cleanup_wal_segments);
  const auto seg1 = vmemkv::derive_wal_segment_path(path, 1);
  uint64_t valid_end = 0;
  uint64_t first_lsn = 0;
  {
    vmemkv::Wal wal(path);
    first_lsn = append_insert(wal, as_span(bytes_of("a")), as_span(bytes_of("1")));
    valid_end = std::filesystem::file_size(seg1);
  }

  // Simulate a crash mid-append: raw sub-header bytes, bypassing Wal.
  vmemkv_test::append_torn_header(path);
  REQUIRE(std::filesystem::file_size(seg1) == valid_end + 10);

  vmemkv::Wal wal(path);
  CHECK(std::filesystem::file_size(seg1) == valid_end);
  CHECK(wal.next_lsn() == first_lsn + 1);

  const uint64_t lsn = append_insert(wal, as_span(bytes_of("b")), as_span(bytes_of("2")));
  CHECK(lsn == first_lsn + 1);

  std::vector<std::string> keys;
  wal.replay([&](vmemkv::WalRecordType /*type*/,
                 std::span<const std::byte> key,
                 std::span<const std::byte> /*value*/,
                 uint64_t /*lsn*/) { keys.push_back(span_to_string(key)); });
  REQUIRE(keys.size() == 2);
  CHECK(keys[0] == "a");
  CHECK(keys[1] == "b");
}

TEST_CASE("Wal: torn trailing record with full header but truncated payload is discarded") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_wal_test", vmemkv_test::cleanup_wal_segments);
  const auto seg1 = vmemkv::derive_wal_segment_path(path, 1);
  uint64_t valid_end = 0;
  uint64_t first_lsn = 0;
  {
    vmemkv::Wal wal(path);
    first_lsn = append_insert(wal, as_span(bytes_of("a")), as_span(bytes_of("1")));
    valid_end = std::filesystem::file_size(seg1);
  }

  // Declares a 10-byte payload with only 3 following -- torn payload.
  vmemkv::append_truncated_record_for_test(seg1, 10, "xyz");

  vmemkv::Wal wal(path);
  CHECK(std::filesystem::file_size(seg1) == valid_end);
  CHECK(wal.next_lsn() == first_lsn + 1);
}

TEST_CASE("Wal: corrupted checksum on trailing record is discarded, earlier records survive") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_wal_test", vmemkv_test::cleanup_wal_segments);
  const auto seg1 = vmemkv::derive_wal_segment_path(path, 1);
  uint64_t after_first = 0;
  uint64_t first_lsn = 0;
  {
    vmemkv::Wal wal(path);
    first_lsn = append_insert(wal, as_span(bytes_of("keep")), as_span(bytes_of("v1")));
    after_first = std::filesystem::file_size(seg1);
    append_insert(wal, as_span(bytes_of("corrupt")), as_span(bytes_of("v2")));
  }

  vmemkv_test::flip_byte_at(seg1,
                            static_cast<std::streamoff>(after_first) +
                                static_cast<std::streamoff>(offsetof(vmemkv::WalRecordHeader, checksum)));

  vmemkv::Wal wal(path);
  CHECK(std::filesystem::file_size(seg1) == after_first);
  CHECK(wal.next_lsn() == first_lsn + 1);

  std::vector<std::string> keys;
  wal.replay([&](vmemkv::WalRecordType /*type*/,
                 std::span<const std::byte> key,
                 std::span<const std::byte> /*value*/,
                 uint64_t /*lsn*/) { keys.push_back(span_to_string(key)); });
  REQUIRE(keys.size() == 1);
  CHECK(keys[0] == "keep");
}

TEST_CASE("Wal: concurrent appends from multiple threads yield unique LSNs that all survive replay") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_wal_test", vmemkv_test::cleanup_wal_segments);
  constexpr int kThreadCount = 8;
  constexpr int kPerThread = 200;
  constexpr int kTotal = kThreadCount * kPerThread;

  // Each thread records its *returned* LSNs, not file layout: the contract is "every
  // acknowledged append gets a distinct LSN and replays", not "gapless 1..N" (a legitimately
  // failed append's reserved LSN would gap it; asserting contiguity would over-test).
  vmemkv::Wal wal(path);
  std::vector<std::vector<uint64_t>> per_thread_lsns(kThreadCount);
  auto threads = run_concurrent_appends(kThreadCount, kPerThread, [&](int thread_index, int i) {
    const auto key = bytes_of("t" + std::to_string(thread_index) + "_" + std::to_string(i));
    const auto val = bytes_of("v");
    per_thread_lsns[static_cast<size_t>(thread_index)].push_back(append_insert(wal, as_span(key), as_span(val)));
  });
  for (auto &thread : threads) {
    thread.join();
  }

  std::vector<uint64_t> acknowledged_lsns;
  acknowledged_lsns.reserve(kTotal);
  for (const auto &lsns : per_thread_lsns) {
    acknowledged_lsns.insert(acknowledged_lsns.end(), lsns.begin(), lsns.end());
  }
  REQUIRE(acknowledged_lsns.size() == static_cast<size_t>(kTotal));
  std::sort(acknowledged_lsns.begin(), acknowledged_lsns.end());
  CHECK(std::adjacent_find(acknowledged_lsns.begin(), acknowledged_lsns.end()) ==
        acknowledged_lsns.end());  // No two appends were handed the same LSN.
  CHECK(wal.next_lsn() > acknowledged_lsns.back());

  std::vector<uint64_t> replayed_lsns;
  const uint64_t count = wal.replay([&](vmemkv::WalRecordType /*type*/,
                                        std::span<const std::byte> /*key*/,
                                        std::span<const std::byte> /*value*/,
                                        uint64_t lsn) { replayed_lsns.push_back(lsn); });
  CHECK(count == static_cast<uint64_t>(kTotal));
  std::sort(replayed_lsns.begin(), replayed_lsns.end());
  CHECK(replayed_lsns == acknowledged_lsns);  // Every acknowledged append survives replay, and no others do.
}

TEST_CASE("Wal: zero-length value on Insert round-trips distinctly from Delete") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_wal_test", vmemkv_test::cleanup_wal_segments);
  {
    vmemkv::Wal wal(path);
    const std::vector<std::byte> empty_value;
    append_insert(wal, as_span(bytes_of("emptyval")), as_span(empty_value));
  }

  vmemkv::Wal wal(path);
  bool saw_record = false;
  wal.replay([&](vmemkv::WalRecordType type,
                 std::span<const std::byte> key,
                 std::span<const std::byte> value,
                 uint64_t /*lsn*/) {
    saw_record = true;
    CHECK(type == vmemkv::WalRecordType::Insert);
    CHECK(span_to_string(key) == "emptyval");
    CHECK(value.empty());
  });
  CHECK(saw_record);
}

TEST_CASE("Wal: large (>64KB) key/value payloads round-trip") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_wal_test", vmemkv_test::cleanup_wal_segments);
  constexpr size_t kLargeSize = 70 * 1024;
  const std::vector<std::byte> big_key(kLargeSize, std::byte{0xAB});
  const std::vector<std::byte> big_value(kLargeSize, std::byte{0xCD});
  {
    vmemkv::Wal wal(path);
    append_insert(wal, as_span(big_key), as_span(big_value));
  }

  vmemkv::Wal wal(path);
  bool saw_record = false;
  wal.replay([&](vmemkv::WalRecordType type,
                 std::span<const std::byte> key,
                 std::span<const std::byte> value,
                 uint64_t /*lsn*/) {
    saw_record = true;
    CHECK(type == vmemkv::WalRecordType::Insert);
    REQUIRE(key.size() == big_key.size());
    REQUIRE(value.size() == big_value.size());
    CHECK(std::equal(key.begin(), key.end(), big_key.begin()));
    CHECK(std::equal(value.begin(), value.end(), big_value.begin()));
  });
  CHECK(saw_record);
}

TEST_CASE(
    "Wal: rotate_segment() rolls onto a new segment, replay() still sees everything (nothing 2 rollovers old yet)") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_wal_test", vmemkv_test::cleanup_wal_segments);
  vmemkv::Wal wal(path);

  append_insert(wal, as_span(bytes_of("a")), as_span(bytes_of("1")));
  append_insert(wal, as_span(bytes_of("b")), as_span(bytes_of("2")));
  append_insert(wal, as_span(bytes_of("c")), as_span(bytes_of("3")));

  wal.rotate_segment();  // 1 -> 2: nothing two rollovers old yet.

  std::vector<ReplayedRecord> records;
  const uint64_t count = wal.replay(
      [&](vmemkv::WalRecordType type, std::span<const std::byte> key, std::span<const std::byte> value, uint64_t lsn) {
        records.push_back(ReplayedRecord{type, span_to_string(key), span_to_string(value), lsn});
      });

  CHECK(count == 3);
  REQUIRE(records.size() == 3);
  CHECK(records[0].key == "a");
  CHECK(records[1].key == "b");
  CHECK(records[2].key == "c");
}

TEST_CASE("Wal: rotate_segment() deletes the generation two rollovers back") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_wal_test", vmemkv_test::cleanup_wal_segments);
  vmemkv::Wal wal(path);

  const uint64_t lsn_a = append_insert(wal, as_span(bytes_of("a")), as_span(bytes_of("1")));
  wal.rotate_segment();  // 1 -> 2: nothing to delete yet.
  CHECK(std::filesystem::exists(vmemkv::derive_wal_segment_path(path, 1)));

  const uint64_t lsn_b = append_insert(wal, as_span(bytes_of("b")), as_span(bytes_of("2")));
  wal.rotate_segment();  // 2 -> 3: generation 1 is two rollovers back -- deleted.
  CHECK_FALSE(std::filesystem::exists(vmemkv::derive_wal_segment_path(path, 1)));
  CHECK(std::filesystem::exists(vmemkv::derive_wal_segment_path(path, 2)));

  std::vector<ReplayedRecord> records;
  wal.replay(
      [&](vmemkv::WalRecordType type, std::span<const std::byte> key, std::span<const std::byte> value, uint64_t lsn) {
        records.push_back(ReplayedRecord{type, span_to_string(key), span_to_string(value), lsn});
      });
  REQUIRE(records.size() == 1);  // "a" (generation 1) is gone; "b" (generation 2) survives.
  CHECK(records[0].key == "b");
  CHECK(records[0].lsn == lsn_b);
  CHECK(lsn_a < lsn_b);  // Sanity on the fixture, not the code under test.
}

TEST_CASE("Wal: rotate_segment() keeps LSN numbering continuous, does not reset") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_wal_test", vmemkv_test::cleanup_wal_segments);
  vmemkv::Wal wal(path);

  append_insert(wal, as_span(bytes_of("a")), as_span(bytes_of("1")));
  const uint64_t lsn_b = append_insert(wal, as_span(bytes_of("b")), as_span(bytes_of("2")));

  wal.rotate_segment();
  CHECK(wal.next_lsn() > lsn_b);

  const uint64_t lsn_after = append_insert(wal, as_span(bytes_of("c")), as_span(bytes_of("3")));
  CHECK(lsn_after > lsn_b);
}

TEST_CASE("Wal: rotate_segment() with nothing written since the last rollover still preserves LSN continuity") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_wal_test", vmemkv_test::cleanup_wal_segments);
  vmemkv::Wal wal(path);

  const uint64_t lsn_a = append_insert(wal, as_span(bytes_of("a")), as_span(bytes_of("1")));
  wal.rotate_segment();  // Gen 2 active, empty.
  wal.rotate_segment();  // Again with nothing on gen 2; gen 1 ("a") deleted.

  CHECK(wal.next_lsn() > lsn_a);

  const uint64_t count =
      wal.replay([](vmemkv::WalRecordType /*type*/,
                    std::span<const std::byte> /*key*/,
                    std::span<const std::byte> /*value*/,
                    uint64_t /*lsn*/) { FAIL("replay callback invoked but no record should survive"); });
  CHECK(count == 0);

  const uint64_t lsn_after = append_insert(wal, as_span(bytes_of("b")), as_span(bytes_of("2")));
  CHECK(lsn_after > lsn_a);
}

TEST_CASE("Wal: rotate_segment() starts the new active segment empty") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_wal_test", vmemkv_test::cleanup_wal_segments);
  uint64_t size_before = 0;
  {
    vmemkv::Wal wal(path);
    append_insert(wal, as_span(bytes_of("a")), as_span(bytes_of("1")));
    for (int i = 0; i < 50; ++i) {
      append_insert(wal, as_span(bytes_of("k" + std::to_string(i))), as_span(bytes_of("v")));
    }
    size_before = std::filesystem::file_size(vmemkv::derive_wal_segment_path(path, 1));
    REQUIRE(size_before > 0);

    wal.rotate_segment();
  }

  const auto seg2 = vmemkv::derive_wal_segment_path(path, 2);
  REQUIRE(std::filesystem::exists(seg2));
  CHECK(std::filesystem::file_size(seg2) == 0);

  // A reopened Wal must see the same state -- the crash-recovery property, not just the live
  // object's view. Gen 1 survives (one rollover only), so all 51 records still replay.
  vmemkv::Wal reopened(path);
  uint64_t replayed_count = reopened.replay([](vmemkv::WalRecordType /*type*/,
                                               std::span<const std::byte> /*key*/,
                                               std::span<const std::byte> /*value*/,
                                               uint64_t lsn) { CHECK(lsn > 0); });
  CHECK(replayed_count == 51);
}

TEST_CASE("Wal: rotate_segment() survives reopen -- every existing segment replays from a new Wal instance") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_wal_test", vmemkv_test::cleanup_wal_segments);
  uint64_t lsn_c = 0;
  {
    vmemkv::Wal wal(path);
    append_insert(wal, as_span(bytes_of("a")), as_span(bytes_of("1")));
    append_insert(wal, as_span(bytes_of("b")), as_span(bytes_of("2")));
    lsn_c = append_insert(wal, as_span(bytes_of("c")), as_span(bytes_of("3")));
    wal.rotate_segment();  // Gen 1 (a, b, c) retained; gen 2 active, empty.
  }

  vmemkv::Wal reopened(path);
  CHECK(reopened.next_lsn() == lsn_c + 1);

  std::vector<std::string> keys;
  reopened.replay([&](vmemkv::WalRecordType /*type*/,
                      std::span<const std::byte> key,
                      std::span<const std::byte> /*value*/,
                      uint64_t /*lsn*/) { keys.push_back(span_to_string(key)); });
  REQUIRE(keys.size() == 3);
  CHECK(keys[0] == "a");
  CHECK(keys[1] == "b");
  CHECK(keys[2] == "c");
}

// ---------------------------------------------------------------------------------------------
// Group-commit (lock-free ring) regression tests.
// ---------------------------------------------------------------------------------------------

TEST_CASE("Wal: append after reopening a non-empty WAL does not hang") {
  // next_to_flush_ must init from next_lsn_ on recovery, not stay 1, or the first leader append
  // spins forever on never-published slots.
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_wal_test", vmemkv_test::cleanup_wal_segments);
  {
    vmemkv::Wal wal(path);
    append_insert(wal, as_span(bytes_of("a")), as_span(bytes_of("1")));
    append_insert(wal, as_span(bytes_of("b")), as_span(bytes_of("2")));
  }

  vmemkv::Wal wal(path);
  const bool completed = run_with_timeout(
      [&wal]() { append_insert(wal, as_span(bytes_of("c")), as_span(bytes_of("3"))); }, std::chrono::seconds(5));
  REQUIRE(completed);

  std::vector<std::string> keys;
  wal.replay([&](vmemkv::WalRecordType /*type*/,
                 std::span<const std::byte> key,
                 std::span<const std::byte> /*value*/,
                 uint64_t /*lsn*/) { keys.push_back(span_to_string(key)); });
  REQUIRE(keys.size() == 3);
  CHECK(keys[2] == "c");
}

TEST_CASE("Wal: concurrent appends replay with exactly the content each thread wrote, keyed by LSN") {
  // Beyond unique LSNs: per-record content catches swapped-bytes batching bugs a pure LSN-set
  // check would miss.
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_wal_test", vmemkv_test::cleanup_wal_segments);
  constexpr int kThreadCount = 32;
  constexpr int kPerThread = 50;

  vmemkv::Wal wal(path);
  std::mutex recorded_mutex;
  std::unordered_map<uint64_t, ReplayedRecord> recorded_by_lsn;

  auto threads = run_concurrent_appends(kThreadCount, kPerThread, [&](int thread_index, int i) {
    const std::string key_str = "t" + std::to_string(thread_index) + "_" + std::to_string(i);
    const std::string value_str = "v" + std::to_string(thread_index) + "_" + std::to_string(i);
    const auto key = bytes_of(key_str);
    const auto val = bytes_of(value_str);
    const uint64_t lsn = append_insert(wal, as_span(key), as_span(val));
    std::lock_guard<std::mutex> lock(recorded_mutex);
    recorded_by_lsn[lsn] = ReplayedRecord{vmemkv::WalRecordType::Insert, key_str, value_str, lsn};
  });
  for (auto &thread : threads) {
    thread.join();
  }

  REQUIRE(recorded_by_lsn.size() == static_cast<size_t>(kThreadCount * kPerThread));

  check_replay_matches(wal, recorded_by_lsn);
}

TEST_CASE("Wal: append storm exceeding ring capacity does not corrupt or duplicate records") {
  // Without the per-pass kWalRingCapacity bound, a burst wrapping the ring pre-flush lets the
  // leader mistake a stale slot for the sought LSN -- wrong bytes under the wrong LSN, stranding
  // the true owner. Threads join via join_all_with_timeout() (never detached: avoids a TSan
  // teardown race).
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_wal_test", vmemkv_test::cleanup_wal_segments);
  vmemkv::Wal wal(path);

  // Must exceed ring capacity (4096) by a comfortable margin.
  constexpr int kAppendCount = 4096 + 512;

  std::mutex recorded_mutex;
  std::unordered_map<uint64_t, std::string> recorded_by_lsn;

  auto threads = run_concurrent_appends(kAppendCount, 1, [&](int thread_index, int /*i*/) {
    const std::string key_str = "k" + std::to_string(thread_index);
    const auto key = bytes_of(key_str);
    const auto val = bytes_of("v");
    const uint64_t lsn = append_insert(wal, as_span(key), as_span(val));
    std::lock_guard<std::mutex> lock(recorded_mutex);
    recorded_by_lsn[lsn] = key_str;
  });

  // Generous timeout: high thread counts are what TSan's per-thread bookkeeping hits hardest
  // (~1s normally, 1min+ under TSan).
  REQUIRE(join_all_with_timeout(threads, std::chrono::minutes(10)));
  REQUIRE(recorded_by_lsn.size() == static_cast<size_t>(kAppendCount));  // every LSN unique

  check_replay_matches(wal, recorded_by_lsn);
}

TEST_CASE("Wal: rotate_segment() under concurrent appends returns promptly and loses nothing") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_wal_test", vmemkv_test::cleanup_wal_segments);
  vmemkv::Wal wal(path);

  std::atomic<bool> stop{false};
  std::atomic<uint64_t> appended_count{0};
  std::mutex recorded_mutex;
  std::vector<uint64_t> all_lsns;

  // Kept joinable (never detached): no race with a later TEST_CASE's stack (see
  // run_with_timeout()).
  std::thread writer([&]() {
    int i = 0;
    while (!stop.load(std::memory_order_relaxed)) {
      const auto key = bytes_of("k" + std::to_string(i));
      const auto val = bytes_of("v");
      const uint64_t lsn = append_insert(wal, as_span(key), as_span(val));
      {
        std::lock_guard<std::mutex> lock(recorded_mutex);
        all_lsns.push_back(lsn);
      }
      appended_count.fetch_add(1, std::memory_order_relaxed);
      ++i;
    }
  });

  {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (appended_count.load(std::memory_order_relaxed) < 50 && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    REQUIRE(appended_count.load(std::memory_order_relaxed) >= 50);
  }

  const bool rotate_completed = run_with_timeout([&]() { wal.rotate_segment(); }, std::chrono::seconds(10));
  REQUIRE(rotate_completed);

  stop.store(true, std::memory_order_relaxed);
  const bool writer_stopped = run_with_timeout([&]() { writer.join(); }, std::chrono::seconds(10));
  REQUIRE(writer_stopped);

  std::vector<uint64_t> replayed_lsns;
  wal.replay([&](vmemkv::WalRecordType /*type*/,
                 std::span<const std::byte> /*key*/,
                 std::span<const std::byte> /*value*/,
                 uint64_t lsn) { replayed_lsns.push_back(lsn); });
  std::sort(replayed_lsns.begin(), replayed_lsns.end());
  CHECK(std::adjacent_find(replayed_lsns.begin(), replayed_lsns.end()) == replayed_lsns.end());  // no duplicates

  std::vector<uint64_t> expected_lsns;
  {
    std::lock_guard<std::mutex> lock(recorded_mutex);
    expected_lsns = all_lsns;
  }
  std::sort(expected_lsns.begin(), expected_lsns.end());
  // One rollover only: nothing two rollovers old yet, so every appended record must still be
  // present regardless of landing segment (see rotate_segment()).
  CHECK(replayed_lsns == expected_lsns);
}

TEST_CASE("Wal: write/fsync failure poisons the Wal and fails outstanding callers without hanging") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_wal_test", vmemkv_test::cleanup_wal_segments);
  vmemkv::Wal wal(path);

  // Establish a small baseline so later writes past this size trigger EFBIG.
  append_insert(wal, as_span(bytes_of("seed")), as_span(bytes_of("1")));
  const auto size_after_seed =
      static_cast<rlim_t>(std::filesystem::file_size(vmemkv::derive_wal_segment_path(path, 1)));

  struct rlimit original_limit {};
  REQUIRE(getrlimit(RLIMIT_FSIZE, &original_limit) == 0);
  struct sigaction original_sigxfsz {};
  REQUIRE(sigaction(SIGXFSZ, nullptr, &original_sigxfsz) == 0);

  struct RestoreGuard {
    struct rlimit limit;
    struct sigaction sigxfsz;
    ~RestoreGuard() {
      setrlimit(RLIMIT_FSIZE, &limit);
      sigaction(SIGXFSZ, &sigxfsz, nullptr);
    }
  } restore_guard{original_limit, original_sigxfsz};

  // Past RLIMIT_FSIZE raises SIGXFSZ (would kill this process) -- ignore it so write()
  // just fails with EFBIG.
  struct sigaction ignore_action {};
  ignore_action.sa_handler = SIG_IGN;
  REQUIRE(sigaction(SIGXFSZ, &ignore_action, nullptr) == 0);

  struct rlimit tight_limit = original_limit;
  tight_limit.rlim_cur = size_after_seed + 4;  // just barely past the seed record
  REQUIRE(setrlimit(RLIMIT_FSIZE, &tight_limit) == 0);

  // Concurrent appends past the limit must all fail (EFBIG), not hang. Joinable +
  // join_all_with_timeout (never detach-and-poll).
  constexpr int kThreadCount = 8;
  std::atomic<int> threw{0};
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);
  for (int i = 0; i < kThreadCount; ++i) {
    threads.emplace_back([&, i]() {
      try {
        append_insert(wal, as_span(bytes_of("x" + std::to_string(i))), as_span(bytes_of("y")));
      } catch (...) {
        threw.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  REQUIRE(join_all_with_timeout(threads, std::chrono::seconds(10)));
  CHECK(threw.load(std::memory_order_relaxed) == kThreadCount);

  // Restore the real limit before further syscalls.
  REQUIRE(setrlimit(RLIMIT_FSIZE, &original_limit) == 0);

  // Permanently poisoned: further appends fail immediately, without another syscall.
  CHECK_THROWS(append_insert(wal, as_span(bytes_of("after")), as_span(bytes_of("z"))));
}
