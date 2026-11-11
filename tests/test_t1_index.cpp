// test_t1_index.cpp — Standalone T1Index: get/put, reorganize merge + chk_writer, checkpoint
// adopt, and epoch-reclamation concurrency stress.

#include <doctest/doctest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <vector>
#include <vmemkv/config.hpp>
#include <vmemkv/t1_index/t1_index.hpp>

#include "test_support.hpp"

namespace {

using vmemkv_test::to_span;

// 256-entry append region: reorganize() triggers without thousands of puts per run.
using TinyAppendConfig = vmemkv_test::TinyAppendConfig<8>;

using TestIndex = vmemkv::T1Index<TinyAppendConfig>;

// Heap-allocate: the append-region hash index embeds a large bucket array unsuited to the stack.
auto make_index() -> std::unique_ptr<TestIndex> { return std::make_unique<TestIndex>(); }

// Builds an EntrySnapshot the way put()/reorganize() would (load_sorted_region_from_checkpoint
// bypasses put()'s own hashing).
auto make_entry(const std::string &key_string, TestIndex::Payload payload) -> TestIndex::EntrySnapshot {
  TestIndex::Key key{};
  std::memcpy(key.data(), key_string.data(), std::min(key_string.size(), key.size()));
  return TestIndex::EntrySnapshot{key, payload, t1_detail::hash_full_key(to_span(key_string))};
}

using vmemkv_test::identity_offset_mapper;
using vmemkv_test::per_entry_offset_mapper;

void fill_sequential(TestIndex &idx, int count) {
  for (int i = 0; i < count; ++i) {
    CHECK(idx.put(to_span("k" + std::to_string(i)), static_cast<uint64_t>(i)) == TestIndex::PutResult::Applied);
  }
}

}  // namespace

TEST_CASE("T1Index: get on empty index returns STORE_NOT_FOUND") {
  auto idx = make_index();
  CHECK(idx->get(to_span("missing")) == vmemkv::STORE_NOT_FOUND);
}

TEST_CASE("T1Index: put then get round-trips the payload") {
  auto idx = make_index();
  CHECK(idx->put(to_span("a"), 42) == TestIndex::PutResult::Applied);
  CHECK(idx->get(to_span("a")) == 42U);
}

TEST_CASE("T1Index: put on an existing key overwrites in place, does not grow the append region") {
  auto idx = make_index();
  CHECK(idx->put(to_span("a"), 1) == TestIndex::PutResult::Applied);
  CHECK(idx->put(to_span("a"), 2) == TestIndex::PutResult::Applied);
  CHECK(idx->get(to_span("a")) == 2U);
  CHECK(idx->append_size() == 1);
}

TEST_CASE("T1Index: put with STORE_NOT_FOUND tombstones the key") {
  auto idx = make_index();
  CHECK(idx->put(to_span("a"), 1) == TestIndex::PutResult::Applied);
  CHECK(idx->put(to_span("a"), vmemkv::STORE_NOT_FOUND) == TestIndex::PutResult::Applied);
  CHECK(idx->get(to_span("a")) == vmemkv::STORE_NOT_FOUND);
}

TEST_CASE("T1Index: append region reports AppendRegionFull once capacity is exhausted") {
  auto idx = make_index();
  const size_t append_cap = idx->append_capacity();
  size_t applied = 0;
  for (size_t i = 0; i < append_cap + 1; ++i) {
    const std::string key = "k" + std::to_string(i);
    if (idx->put(to_span(key), static_cast<uint64_t>(i)) == TestIndex::PutResult::AppendRegionFull) {
      break;
    }
    ++applied;
  }
  CHECK(applied == append_cap);
}

TEST_CASE("T1Index: reorganize merges append region into sorted region, keeps live entries readable") {
  auto idx = make_index();
  constexpr int key_count = 100;
  fill_sequential(*idx, key_count);
  CHECK(idx->append_size() == static_cast<size_t>(key_count));

  idx->reorganize(identity_offset_mapper());

  CHECK(idx->append_size() == 0);
  for (int i = 0; i < key_count; ++i) {
    CHECK(idx->get(to_span("k" + std::to_string(i))) == static_cast<uint64_t>(i));
  }
}

TEST_CASE("T1Index: reorganize drops tombstoned entries") {
  auto idx = make_index();
  CHECK(idx->put(to_span("a"), 1) == TestIndex::PutResult::Applied);
  CHECK(idx->put(to_span("b"), 2) == TestIndex::PutResult::Applied);
  CHECK(idx->put(to_span("a"), vmemkv::STORE_NOT_FOUND) == TestIndex::PutResult::Applied);

  idx->reorganize(identity_offset_mapper());

  CHECK(idx->get(to_span("a")) == vmemkv::STORE_NOT_FOUND);
  CHECK(idx->get(to_span("b")) == 2U);
}

TEST_CASE("T1Index: reorganize's offset_mapper remaps every live payload") {
  auto idx = make_index();
  CHECK(idx->put(to_span("a"), 10) == TestIndex::PutResult::Applied);
  CHECK(idx->put(to_span("b"), 20) == TestIndex::PutResult::Applied);

  idx->reorganize(per_entry_offset_mapper([](uint64_t payload, uint64_t /*hash*/) { return payload + 1000; }));

  CHECK(idx->get(to_span("a")) == 1010U);
  CHECK(idx->get(to_span("b")) == 1020U);
}

TEST_CASE("T1Index: reorganize's chk_writer sees exactly the merged, sorted, offset-mapped live entries") {
  auto idx = make_index();
  CHECK(idx->put(to_span("b"), 2) == TestIndex::PutResult::Applied);
  CHECK(idx->put(to_span("a"), 1) == TestIndex::PutResult::Applied);
  CHECK(idx->put(to_span("c"), 3) == TestIndex::PutResult::Applied);
  // Tombstoned before reorganize: must be excluded from what chk_writer observes.
  CHECK(idx->put(to_span("c"), vmemkv::STORE_NOT_FOUND) == TestIndex::PutResult::Applied);

  std::vector<TestIndex::EntrySnapshot> observed;
  idx->reorganize(
      per_entry_offset_mapper([](uint64_t payload, uint64_t /*hash*/) { return payload + 100; }),
      [&](std::span<const TestIndex::EntrySnapshot> merged) { observed.assign(merged.begin(), merged.end()); });

  REQUIRE(observed.size() == 2);
  // key_prefix ascending: "a" < "b".
  CHECK(observed[0].payload_bits == 101U);
  CHECK(observed[1].payload_bits == 102U);
}

TEST_CASE("T1Index: load_sorted_region_from_checkpoint adopts entries as the live sorted region") {
  auto idx = make_index();
  const std::vector<TestIndex::EntrySnapshot> entries = {make_entry("a", 111), make_entry("b", 222)};

  idx->load_sorted_region_from_checkpoint(entries);

  CHECK(idx->get(to_span("a")) == 111U);
  CHECK(idx->get(to_span("b")) == 222U);
  CHECK(idx->append_size() == 0);
}

TEST_CASE("T1Index: load_sorted_region_from_checkpoint replaces any previously loaded sorted region") {
  auto idx = make_index();
  idx->load_sorted_region_from_checkpoint(std::vector<TestIndex::EntrySnapshot>{make_entry("a", 1)});
  CHECK(idx->get(to_span("a")) == 1U);

  idx->load_sorted_region_from_checkpoint(std::vector<TestIndex::EntrySnapshot>{make_entry("a", 2)});
  CHECK(idx->get(to_span("a")) == 2U);
}

// put/get_with_hash/append_size must all hold the epoch guard for their duration, or
// reorganize() can delete buffers still in use. Hammering reorganize() concurrently with all
// three exercises that directly.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("T1Index: concurrent put/get_with_hash/append_size survive racing reorganize") {
  auto idx = make_index();
  std::atomic<bool> stop{false};
  std::atomic<uint64_t> reorganize_count{0};

  constexpr int writer_thread_count = 4;
  constexpr int reader_thread_count = 4;
  constexpr int iterations_per_writer = 1500;

  std::thread reorganizer([&]() {
    while (!stop.load(std::memory_order_relaxed)) {
      idx->reorganize(identity_offset_mapper());
      reorganize_count.fetch_add(1, std::memory_order_relaxed);
    }
  });

  std::vector<std::thread> readers;
  readers.reserve(reader_thread_count);
  for (int reader_id = 0; reader_id < reader_thread_count; ++reader_id) {
    readers.emplace_back([&]() {
      while (!stop.load(std::memory_order_relaxed)) {
        (void)idx->get_with_hash(to_span(std::string("w0_0")));
        (void)idx->append_size();
      }
    });
  }

  std::vector<std::thread> writers;
  writers.reserve(writer_thread_count);
  for (int writer_id = 0; writer_id < writer_thread_count; ++writer_id) {
    writers.emplace_back([&, writer_id]() {
      for (int i = 0; i < iterations_per_writer; ++i) {
        const std::string key = "w" + std::to_string(writer_id) + "_" + std::to_string(i);
        while (idx->put(to_span(key), static_cast<uint64_t>(i)) == TestIndex::PutResult::AppendRegionFull) {
          std::this_thread::yield();
        }
      }
    });
  }

  for (auto &writer : writers) {
    writer.join();
  }
  stop.store(true, std::memory_order_relaxed);
  reorganizer.join();
  for (auto &reader : readers) {
    reader.join();
  }

  CHECK(reorganize_count.load() > 0);
  for (int writer_id = 0; writer_id < writer_thread_count; ++writer_id) {
    const std::string key = "w" + std::to_string(writer_id) + "_" + std::to_string(iterations_per_writer - 1);
    CHECK(idx->get(to_span(key)) == static_cast<uint64_t>(iterations_per_writer - 1));
  }
}

// Merge dedups sorted-vs-append copies by *clean* hash (inline-metadata bits excluded), so a
// put() racing the frozen-region bypass across an inline transition still resolves to one
// logical key instead of two live entries.
struct TinyInlineConfig : vmemkv::Config<> {
  static constexpr size_t kAppendCapacityLog2 = 8;
  static constexpr size_t kAppendCapacityEntries = size_t{1} << kAppendCapacityLog2;
};
using InlineTestIndex = vmemkv::T1Index<TinyInlineConfig>;

TEST_CASE(
    "T1Index: inline-to-non-inline transition racing an in-flight reorganize does not create a "
    "duplicate entry") {
  auto idx = std::make_unique<InlineTestIndex>();

  // "k" starts inline; its stored hash carries the inline metadata bits.
  REQUIRE(idx->put(to_span("k"), 111, /*is_inline=*/true, /*inline_size=*/8) == InlineTestIndex::PutResult::Applied);

  // reorg #1 freezes "k" into append_immutable_, then parks inside offset_mapper.
  std::atomic<bool> reorg1_entered_offset_mapper{false};
  std::atomic<bool> proceed{false};
  std::thread reorg1([&] {
    idx->reorganize(per_entry_offset_mapper([&](uint64_t payload, uint64_t /*hash*/) {
      reorg1_entered_offset_mapper.store(true, std::memory_order_release);
      reorg1_entered_offset_mapper.notify_all();
      proceed.wait(false, std::memory_order_acquire);
      return payload;
    }));
  });

  reorg1_entered_offset_mapper.wait(false, std::memory_order_acquire);

  // "k" is only live in append_immutable_ now, so resolve()'s bypass must append a *new* entry
  // (with a different, non-inline stored hash) instead of updating the frozen one in place.
  REQUIRE(idx->put(to_span("k"), 222, /*is_inline=*/false) == InlineTestIndex::PutResult::Applied);

  proceed.store(true, std::memory_order_release);
  proceed.notify_all();
  reorg1.join();

  // sorted has (k, inline hash); append_active_ separately holds the bypass (k, non-inline
  // hash), still unmerged.
  CHECK(idx->get(to_span("k")) == 222U);

  // reorg #2 merges both; exactly the latest (222) survives iff the merge keys them together.
  std::vector<InlineTestIndex::EntrySnapshot> merged_out;
  idx->reorganize(identity_offset_mapper(), [&](std::span<const InlineTestIndex::EntrySnapshot> merged) {
    merged_out.assign(merged.begin(), merged.end());
  });

  const auto k_prefix = t1_detail::prefix_from_bytes(to_span("k"));
  const auto count_for_k = std::ranges::count(merged_out, k_prefix, &InlineTestIndex::EntrySnapshot::key);
  CHECK(count_for_k == 1);
  CHECK(idx->get(to_span("k")) == 222U);
}

// store_hash() mutates a live published SortedSlot's hash concurrently with the merge loop
// and readers. Unlike the insert-only stress above, this test updates pre-seeded sorted keys
// continuously under a concurrent reorganize. Best run under ThreadSanitizer.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("T1Index: concurrent updates to already-sorted keys survive racing reorganize") {
  auto idx = make_index();
  constexpr int key_count = 64;
  for (int i = 0; i < key_count; ++i) {
    REQUIRE(idx->put(to_span("k" + std::to_string(i)), 0) == TestIndex::PutResult::Applied);
  }
  // Moves every key into sorted_region, so the in-place update reaches store_hash().
  idx->reorganize(identity_offset_mapper());
  REQUIRE(idx->append_size() == 0);

  std::atomic<bool> stop{false};
  std::atomic<uint64_t> reorganize_count{0};

  std::thread reorganizer([&]() {
    while (!stop.load(std::memory_order_relaxed)) {
      idx->reorganize(identity_offset_mapper());
      reorganize_count.fetch_add(1, std::memory_order_relaxed);
    }
  });

  constexpr int writer_thread_count = 4;
  constexpr int iterations_per_writer = 2000;
  std::vector<std::thread> writers;
  writers.reserve(writer_thread_count);
  for (int writer_id = 0; writer_id < writer_thread_count; ++writer_id) {
    writers.emplace_back([&, writer_id]() {
      for (int i = 0; i < iterations_per_writer; ++i) {
        const std::string key = "k" + std::to_string(i % key_count);
        const auto value = static_cast<uint64_t>((writer_id << 20) | i);
        CHECK(idx->put(to_span(key), value) == TestIndex::PutResult::Applied);
      }
    });
  }

  for (auto &writer : writers) {
    writer.join();
  }
  stop.store(true, std::memory_order_relaxed);
  reorganizer.join();

  CHECK(reorganize_count.load() > 0);
  for (int i = 0; i < key_count; ++i) {
    CHECK(idx->get(to_span("k" + std::to_string(i))) != vmemkv::STORE_NOT_FOUND);
  }
}

// resolve()'s frozen-region bypass ("not found" for frozen slots) vs genuine absence: put()
// must tell them apart, or tombstoning its just-written value on the frozen case briefly
// reports a live key as absent until the bypass insert lands. Only a dedicated reader thread
// polling get() observes the transient window (post-join checks never do).
TEST_CASE("T1Index: get() never observes a live key as absent while a concurrent update races reorganize") {
  auto idx = make_index();
  constexpr int key_count = 64;
  for (int i = 0; i < key_count; ++i) {
    REQUIRE(idx->put(to_span("k" + std::to_string(i)), 0) == TestIndex::PutResult::Applied);
  }

  std::atomic<bool> stop{false};
  std::atomic<bool> saw_missing{false};

  std::thread reorganizer([&]() {
    while (!stop.load(std::memory_order_relaxed)) {
      idx->reorganize(identity_offset_mapper());
    }
  });

  std::thread writer([&]() {
    uint64_t value = 1;
    while (!stop.load(std::memory_order_relaxed)) {
      for (int i = 0; i < key_count; ++i) {
        idx->put(to_span("k" + std::to_string(i)), value);
      }
      ++value;
    }
  });

  std::vector<std::thread> readers;
  for (int r = 0; r < 4; ++r) {
    readers.emplace_back([&]() {
      while (!stop.load(std::memory_order_relaxed)) {
        for (int i = 0; i < key_count; ++i) {
          if (idx->get(to_span("k" + std::to_string(i))) == vmemkv::STORE_NOT_FOUND) {
            saw_missing.store(true, std::memory_order_relaxed);
          }
        }
      }
    });
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  stop.store(true, std::memory_order_relaxed);
  writer.join();
  reorganizer.join();
  for (auto &reader : readers) {
    reader.join();
  }

  CHECK_FALSE(saw_missing.load());
}

// The merge loop snapshots a sorted slot and publishes a brand-new SortedRegion; put()'s
// in-place path must route through sorted_write_frozen_ (see FreezableRegion) so a write
// between snapshot and publish is never lost. One writer per key (well-defined "last write")
// races a hammering reorganizer; the mapper sleep widens the merge-to-publish window toward
// production msync/file-I/O costs.
TEST_CASE("T1Index: dedicated single writer per key survives racing reorganize with exact last value") {
  auto idx = make_index();
  constexpr int key_count = 8;
  constexpr int updates_per_key = 500;
  for (int i = 0; i < key_count; ++i) {
    REQUIRE(idx->put(to_span("k" + std::to_string(i)), 0) == TestIndex::PutResult::Applied);
  }
  idx->reorganize(identity_offset_mapper());

  std::atomic<bool> stop{false};
  std::thread reorganizer([&]() {
    while (!stop.load(std::memory_order_relaxed)) {
      idx->reorganize(per_entry_offset_mapper([](uint64_t payload, uint64_t) {
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        return payload;
      }));
    }
  });

  std::vector<std::thread> writers;
  writers.reserve(key_count);
  for (int i = 0; i < key_count; ++i) {
    writers.emplace_back([&, i]() {
      for (int round = 1; round <= updates_per_key; ++round) {
        const auto value = static_cast<uint64_t>((i << 20) | round);
        REQUIRE(idx->put(to_span("k" + std::to_string(i)), value) == TestIndex::PutResult::Applied);
      }
    });
  }
  for (auto &writer : writers) {
    writer.join();
  }
  stop.store(true, std::memory_order_relaxed);
  reorganizer.join();

  for (int i = 0; i < key_count; ++i) {
    const auto expected = static_cast<uint64_t>((i << 20) | updates_per_key);
    CHECK(idx->get(to_span("k" + std::to_string(i))) == expected);
  }
}

// The immutable-bypass window lets concurrent same-key puts all take "not found, insert new"
// with no mutual re-check. publish_slot() must retire the losers displaced from the hash
// index, or N racers leave N-1 orphaned-but-live duplicates -- invisible to get(), but carried
// by walks into every future checkpoint, forever.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEST_CASE("T1Index: concurrent puts racing the same immutable-bypass window collapse to one entry") {
  // The window is narrower than "reorganize() is paused": once any racer publishes, later
  // racers resolve to it and update in place. A spin barrier plus many racers makes real
  // overlap likely; fresh keys per round compound it.
  constexpr int rounds = 8;
  constexpr int racer_count = 32;

  for (int round = 0; round < rounds; ++round) {
    auto idx = make_index();
    const std::string hot_key = "hot" + std::to_string(round);
    REQUIRE(idx->put(to_span(hot_key), 0) == TestIndex::PutResult::Applied);

    // reorg #1 freezes the key into append_immutable_ and parks inside offset_mapper, giving
    // every racer a window where resolve() is guaranteed not-found.
    std::atomic<bool> reorg1_entered_offset_mapper{false};
    std::atomic<bool> proceed{false};
    std::thread reorg1([&] {
      idx->reorganize(per_entry_offset_mapper([&](uint64_t payload, uint64_t /*hash*/) {
        reorg1_entered_offset_mapper.store(true, std::memory_order_release);
        reorg1_entered_offset_mapper.notify_all();
        proceed.wait(false, std::memory_order_acquire);
        return payload;
      }));
    });
    reorg1_entered_offset_mapper.wait(false, std::memory_order_acquire);

    std::atomic<int> ready_count{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> racers;
    racers.reserve(racer_count);
    for (int i = 0; i < racer_count; ++i) {
      racers.emplace_back([&, i]() {
        ready_count.fetch_add(1, std::memory_order_acq_rel);
        while (!go.load(std::memory_order_acquire)) {
          std::this_thread::yield();
        }
        CHECK(idx->put(to_span(hot_key), static_cast<uint64_t>(i + 1)) == TestIndex::PutResult::Applied);
      });
    }
    while (ready_count.load(std::memory_order_acquire) < racer_count) {
      std::this_thread::yield();
    }
    go.store(true, std::memory_order_release);
    for (auto &racer : racers) {
      racer.join();
    }

    proceed.store(true, std::memory_order_release);
    proceed.notify_all();
    reorg1.join();

    // append_active_ now holds up to racer_count bypass entries for hot_key -- at most one
    // survives live; the rest were retired as orphaned duplicates.
    CHECK(idx->get(to_span(hot_key)) != vmemkv::STORE_NOT_FOUND);

    std::vector<TestIndex::EntrySnapshot> merged_out;
    idx->reorganize(identity_offset_mapper(), [&](std::span<const TestIndex::EntrySnapshot> merged) {
      merged_out.assign(merged.begin(), merged.end());
    });

    const auto hot_prefix = t1_detail::prefix_from_bytes(to_span(hot_key));
    const auto count_for_hot = std::ranges::count(merged_out, hot_prefix, &TestIndex::EntrySnapshot::key);
    CHECK(count_for_hot == 1);
  }
}
