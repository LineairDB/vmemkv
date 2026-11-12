// test_sharded_t1_index.cpp — ShardedT1Index: routing, cross-shard scan, the Closing-split
// protocol, and forwarding-pointer concurrency stress.

#include <doctest/doctest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <set>
#include <span>
#include <string>
#include <thread>
#include <vector>
#include <vmemkv/config.hpp>
#include <vmemkv/t1_index/sharded_t1_index.hpp>

#include "test_support.hpp"

namespace {

using vmemkv_test::identity_offset_mapper;
using vmemkv_test::to_span;

// 1024-entry append region: holds this file's largest key count (800) without overflowing
// before an explicit split -- these tests cover routing/splits, not AppendRegionFull
// backpressure (see test_t1_index.cpp).
using TinyAppendConfig = vmemkv_test::TinyAppendConfig<10>;

using TestIndex = vmemkv::ShardedT1Index<TinyAppendConfig>;

auto make_index() -> std::unique_ptr<TestIndex> { return std::make_unique<TestIndex>(); }

auto make_index_with_target_size(size_t target_shard_size, size_t worker_threads = 2) -> std::unique_ptr<TestIndex> {
  return std::make_unique<TestIndex>(TinyAppendConfig::kAppendCapacityEntries, target_shard_size, worker_threads);
}

// Fixed-width keys ("k" + zero-padded index) so boundary comparisons match numeric order.
auto ikey(int i) -> std::string { return vmemkv_test::padded_key(i, 8); }

void fill_sharded(TestIndex &idx, int count) {
  for (int i = 0; i < count; ++i) {
    CHECK(idx.put(to_span(ikey(i)), static_cast<uint64_t>(i)) == TestIndex::PutResult::Applied);
  }
}

struct ShardSnapshot {
  std::vector<TestIndex::Key> boundaries;
  std::vector<std::vector<TestIndex::EntrySnapshot>> per_shard;
};

auto capture_all(TestIndex &idx) -> ShardSnapshot {
  ShardSnapshot out;
  out.boundaries = idx.checkpoint_all_shards(vmemkv_test::identity_offset_mapper(),
                                             [&](std::span<const TestIndex::EntrySnapshot> merged) {
                                               out.per_shard.emplace_back(merged.begin(), merged.end());
                                             });
  return out;
}

void check_scan_sorted_unique(
    TestIndex &idx, int first, int last, std::size_t expected_count, bool require_sorted = true) {
  std::vector<uint64_t> seen;
  const size_t matched = idx.scan(to_span(ikey(first)),
                                  to_span(ikey(last)),
                                  [&](auto /*key*/, auto payload, auto /*hash*/) { seen.push_back(payload); });
  CHECK(matched == expected_count);
  REQUIRE(seen.size() == expected_count);
  if (require_sorted) {
    CHECK(std::is_sorted(seen.begin(), seen.end()));
  }
  const std::set<uint64_t> unique_seen(seen.begin(), seen.end());
  CHECK(unique_seen.size() == seen.size());
}

}  // namespace

TEST_CASE("ShardedT1Index: get on empty index returns STORE_NOT_FOUND") {
  auto idx = make_index();
  CHECK(idx->get(to_span("missing")) == vmemkv::STORE_NOT_FOUND);
}

TEST_CASE("ShardedT1Index: put then get round-trips, starts at a single shard") {
  auto idx = make_index();
  CHECK(idx->shard_count() == 1);
  CHECK(idx->put(to_span("a"), 42) == TestIndex::PutResult::Applied);
  CHECK(idx->get(to_span("a")) == 42U);
  CHECK(idx->shard_count() == 1);
}

TEST_CASE("ShardedT1Index: put with STORE_NOT_FOUND tombstones the key") {
  auto idx = make_index();
  CHECK(idx->put(to_span("a"), 1) == TestIndex::PutResult::Applied);
  CHECK(idx->put(to_span("a"), vmemkv::STORE_NOT_FOUND) == TestIndex::PutResult::Applied);
  CHECK(idx->get(to_span("a")) == vmemkv::STORE_NOT_FOUND);
}

TEST_CASE("ShardedT1Index: scan on a single shard returns all live entries in order") {
  auto idx = make_index();
  constexpr int kCount = 50;
  fill_sharded(*idx, kCount);
  std::vector<uint64_t> seen;
  const size_t matched = idx->scan(to_span(ikey(0)),
                                   to_span(ikey(kCount)),
                                   [&](auto /*key*/, auto payload, auto /*hash*/) { seen.push_back(payload); });
  CHECK(matched == static_cast<size_t>(kCount));
  REQUIRE(seen.size() == static_cast<size_t>(kCount));
  CHECK(std::is_sorted(seen.begin(), seen.end()));
}

TEST_CASE("ShardedT1Index: split_shard_containing doubles the shard count and preserves all keys") {
  auto idx = make_index();
  constexpr int kCount = 400;
  fill_sharded(*idx, kCount);

  idx->split_shard_containing(to_span(ikey(0)));
  CHECK(idx->shard_count() == 2);

  for (int i = 0; i < kCount; ++i) {
    CHECK(idx->get(to_span(ikey(i))) == static_cast<uint64_t>(i));
  }
}

TEST_CASE("ShardedT1Index: scan across a split boundary is gap-free, duplicate-free, and sorted") {
  auto idx = make_index();
  constexpr int kCount = 400;
  fill_sharded(*idx, kCount);
  idx->split_shard_containing(to_span(ikey(0)));
  REQUIRE(idx->shard_count() == 2);

  check_scan_sorted_unique(*idx, 0, kCount, static_cast<size_t>(kCount));
}

TEST_CASE("ShardedT1Index: scan STOP prevents visiting later shards") {
  auto idx = make_index();
  constexpr int kCount = 50;
  fill_sharded(*idx, kCount);
  idx->split_shard_containing(to_span(ikey(0)));
  REQUIRE(idx->shard_count() == 2U);

  std::vector<uint64_t> seen;
  const size_t matched =
      idx->scan(to_span(ikey(0)), to_span(ikey(kCount)), [&](auto /*key*/, auto payload, auto /*hash*/) {
        seen.push_back(payload);
        return vmemkv::ScanControl::Stop;
      });
  CHECK(matched == 1U);
  CHECK(seen == std::vector<uint64_t>{0U});
}

TEST_CASE("ShardedT1Index: put after split routes to the correct new shard and is visible") {
  auto idx = make_index();
  constexpr int kCount = 400;
  fill_sharded(*idx, kCount);
  idx->split_shard_containing(to_span(ikey(0)));
  REQUIRE(idx->shard_count() == 2);

  // One key on each side of wherever the split landed.
  CHECK(idx->put(to_span(ikey(0)), 9000) == TestIndex::PutResult::Applied);
  CHECK(idx->put(to_span(ikey(kCount - 1)), 9001) == TestIndex::PutResult::Applied);
  CHECK(idx->get(to_span(ikey(0))) == 9000U);
  CHECK(idx->get(to_span(ikey(kCount - 1))) == 9001U);
}

TEST_CASE("ShardedT1Index: splitting an already-splitting shard is a single-flight no-op") {
  auto idx = make_index();
  constexpr int kCount = 400;
  fill_sharded(*idx, kCount);
  idx->split_shard_containing(to_span(ikey(0)));
  REQUIRE(idx->shard_count() == 2);
  // Calling again on a key whose shard was never re-split is a correctness no-op (the target
  // resolved by split_shard_containing() is now one of the fresh post-split shards).
  idx->split_shard_containing(to_span(ikey(0)));
  for (int i = 0; i < kCount; ++i) {
    CHECK(idx->get(to_span(ikey(i))) == static_cast<uint64_t>(i));
  }
}

TEST_CASE("ShardedT1Index: concurrent put/get/scan survive a racing split_shard_containing") {
  auto idx = make_index();
  constexpr int kCount = 800;
  fill_sharded(*idx, kCount);

  std::atomic<bool> stop{false};
  std::atomic<bool> saw_torn_read{false};

  // Writer: overwrites every key with a new generation; readers check values only move forward.
  std::thread writer([&] {
    uint64_t generation = 1;
    while (!stop.load(std::memory_order_relaxed)) {
      for (int i = 0; i < kCount && !stop.load(std::memory_order_relaxed); ++i) {
        idx->put(to_span(ikey(i)), generation * 10000 + static_cast<uint64_t>(i));
      }
      ++generation;
    }
  });

  // Readers: get() every key, scan() the whole range.
  std::vector<std::thread> readers;
  for (int t = 0; t < 4; ++t) {
    readers.emplace_back([&] {
      while (!stop.load(std::memory_order_relaxed)) {
        for (int i = 0; i < kCount; ++i) {
          const uint64_t val = idx->get(to_span(ikey(i)));
          if (val != vmemkv::STORE_NOT_FOUND && val % 10000 != static_cast<uint64_t>(i)) {
            saw_torn_read.store(true, std::memory_order_relaxed);
          }
        }
        size_t total = 0;
        idx->scan(to_span(ikey(0)), to_span(ikey(kCount)), [&](auto /*k*/, auto /*p*/, auto /*h*/) { ++total; });
        if (total > static_cast<size_t>(kCount)) {
          // More matches than keys ever inserted: a duplicate from a botched split.
          saw_torn_read.store(true, std::memory_order_relaxed);
        }
      }
    });
  }

  // Splitter: splits key 0's shard a few times, pausing so the split runs in flight.
  std::thread splitter([&] {
    for (int round = 0; round < 5; ++round) {
      idx->split_shard_containing(to_span(ikey(0)));
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  });

  splitter.join();
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  stop.store(true, std::memory_order_relaxed);
  writer.join();
  for (auto &reader : readers) {
    reader.join();
  }

  CHECK_FALSE(saw_torn_read.load());
  CHECK(idx->shard_count() >= 2);
  for (int i = 0; i < kCount; ++i) {
    CHECK(idx->get(to_span(ikey(i))) != vmemkv::STORE_NOT_FOUND);
  }
}

TEST_CASE("ShardedT1Index: a shard keeps splitting under sustained heavy concurrent writes, never stalling") {
  // Splits attempted constantly under sustained cycling writes; shard_count() must keep climbing.
  constexpr size_t kTargetShardSize = 300;  // Split threshold 200% -> 600 entries.
  constexpr size_t kWorkerThreads = 8;      // High enough for duplicate queue entries to be common.
  constexpr int kWriterThreads = 16;
  constexpr int kKeyRange = 6000;
  constexpr size_t kTargetShardCount = 3;
  auto idx = make_index_with_target_size(kTargetShardSize, kWorkerThreads);

  std::atomic<bool> stop{false};
  std::vector<std::thread> writers;
  writers.reserve(kWriterThreads);
  for (int t = 0; t < kWriterThreads; ++t) {
    writers.emplace_back([&, t] {
      while (!stop.load(std::memory_order_relaxed)) {
        for (int i = t; i < kKeyRange && !stop.load(std::memory_order_relaxed); i += kWriterThreads) {
          idx->put(to_span(ikey(i)), static_cast<uint64_t>(i));
        }
      }
    });
  }

  size_t shard_count = 0;
  for (int round = 0; round < 500 && shard_count < kTargetShardCount; ++round) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    shard_count = idx->shard_count();
  }
  stop.store(true, std::memory_order_relaxed);
  for (auto &writer : writers) {
    writer.join();
  }

  // Cycling writes carry no single correct final value per key, so only split progress is asserted.
  CHECK(shard_count >= kTargetShardCount);
}

TEST_CASE("ShardedT1Index: a split under concurrent writes loses no straggler entry") {
  // A single-pass unique-key insert surfaces the in-flight-write-across-split window.
  constexpr size_t kTargetShardSize = 5000;  // Split threshold 200% -> 10000 entries.
  constexpr int kWriterThreads = 4;
  constexpr int kKeyCount = 30000;  // Comfortably past several splits' worth at this target size.
  auto idx = make_index_with_target_size(kTargetShardSize, /*worker_threads=*/2);

  std::vector<std::thread> writers;
  writers.reserve(kWriterThreads);
  for (int t = 0; t < kWriterThreads; ++t) {
    writers.emplace_back([&, t] {
      for (int i = t; i < kKeyCount; i += kWriterThreads) {
        CHECK(idx->put(to_span(ikey(i)), static_cast<uint64_t>(i)) == TestIndex::PutResult::Applied);
      }
    });
  }
  for (auto &writer : writers) {
    writer.join();
  }

  // Background workers are internal jthreads, never joined directly -- wait for split
  // progress (not stability: on a loaded box the first split can be slow to start, and a
  // stable 1 only means nothing started yet), then a fixed window for any in-flight split.
  // The timeout still fails a genuinely broken splitter below.
  size_t current = idx->shard_count();
  for (int round = 0; round < 300 && current <= 1; ++round) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    current = idx->shard_count();
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  CHECK(current > 1);
  for (int i = 0; i < kKeyCount; ++i) {
    CHECK(idx->get(to_span(ikey(i))) == static_cast<uint64_t>(i));
  }
}

TEST_CASE("ShardedT1Index: maintenance_soft_threshold lowers only under scan at scale") {
  // Production append capacity (2^21): normal threshold is half; an active scan pulls it to the
  // L2 slot budget.
  constexpr size_t kProdCap = size_t{1} << 21;
  CHECK(TestIndex::maintenance_soft_threshold(kProdCap, false) == kProdCap / 2);
  const size_t scan_threshold = TestIndex::maintenance_soft_threshold(kProdCap, true);
  CHECK(scan_threshold ==
        std::min(size_t{1024 * 1024} / vmemkv::T1Index<TinyAppendConfig>::append_slot_bytes(), kProdCap / 2));
  CHECK(scan_threshold < kProdCap / 2);
  // In-tree tiny capacities: the L2 budget never binds, so scan-active changes nothing.
  CHECK(TestIndex::maintenance_soft_threshold(1024, false) == 512);
  CHECK(TestIndex::maintenance_soft_threshold(1024, true) == 512);
}

TEST_CASE("ShardedT1Index: delete pressure triggers maintenance below the occupancy threshold") {
  // 400 live keys stay under the 50%-of-1024 occupancy trigger and the split threshold, so only
  // the tombstone count (400 deletes >= target 400) can request maintenance. Without the
  // delete-pressure trigger the tombstones would sit in append indefinitely; with it, a
  // background merge drains append_size() to 0.
  constexpr size_t kTargetShardSize = 400;
  constexpr int kKeyCount = 400;
  auto idx = make_index_with_target_size(kTargetShardSize, /*worker_threads=*/2);

  for (int i = 0; i < kKeyCount; ++i) {
    CHECK(idx->put(to_span(ikey(i)), static_cast<uint64_t>(i)) == TestIndex::PutResult::Applied);
  }
  CHECK(idx->append_size() > 0);
  for (int i = 0; i < kKeyCount; ++i) {
    CHECK(idx->put(to_span(ikey(i)), vmemkv::STORE_NOT_FOUND) == TestIndex::PutResult::Applied);
  }

  for (int round = 0; round < 300; ++round) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    if (idx->append_size() == 0) {
      break;
    }
  }
  CHECK(idx->append_size() == 0);
  CHECK(idx->shard_count() == 1);
  for (int i = 0; i < kKeyCount; ++i) {
    CHECK(idx->get(to_span(ikey(i))) == vmemkv::STORE_NOT_FOUND);
  }
}

TEST_CASE("ShardedT1Index: checkpoint on clean shards matches a full merge") {
  // Empty append means the merge output is the sorted region as-is. With no writes between two
  // checkpoints, the dump path must produce entry-for-entry identical output to the merge path.
  constexpr size_t kTargetShardSize = 5000;
  constexpr int kKeyCount = 3000;
  auto idx = make_index_with_target_size(kTargetShardSize, /*worker_threads=*/2);

  fill_sharded(*idx, kKeyCount);

  const ShardSnapshot merged = capture_all(*idx);
  size_t merged_total = 0;
  for (const auto &shard : merged.per_shard) {
    merged_total += shard.size();
  }
  CHECK(merged_total == static_cast<size_t>(kKeyCount));

  // No writes since: every shard's append region is empty, so this is the dump path throughout.
  CHECK(idx->append_size() == 0);
  const ShardSnapshot dumped = capture_all(*idx);
  CHECK(dumped.boundaries == merged.boundaries);
  REQUIRE(dumped.per_shard.size() == merged.per_shard.size());
  for (size_t s = 0; s < dumped.per_shard.size(); ++s) {
    REQUIRE(dumped.per_shard[s].size() == merged.per_shard[s].size());
    for (size_t i = 0; i < dumped.per_shard[s].size(); ++i) {
      CHECK(dumped.per_shard[s][i].key == merged.per_shard[s][i].key);
      CHECK(dumped.per_shard[s][i].payload_bits == merged.per_shard[s][i].payload_bits);
      CHECK(dumped.per_shard[s][i].hash == merged.per_shard[s][i].hash);
    }
  }
}

TEST_CASE("ShardedT1Index: cycling updates under tiny shards keep every key at its latest value") {
  // Per-key monotonic rounds: the only correct final value is kRounds - 1 (older is a stale
  // clobber, STORE_NOT_FOUND is a loss). A splitter thread hammers splits across the range so
  // every post-publication window races the writers dozens of times.
  constexpr size_t kTargetShardSize = 100;
  constexpr int kWriterThreads = 8;
  constexpr int kKeyRange = 400;
  constexpr int kRounds = 40;
  auto idx = make_index_with_target_size(kTargetShardSize, /*worker_threads=*/2);
  // Seed the key set before starting the split pressure so every attempted split has live data,
  // even when the splitter is not scheduled until after the update writers finish.
  for (int i = 0; i < kKeyRange; ++i) {
    REQUIRE(idx->put(to_span(ikey(i)), 0) == TestIndex::PutResult::Applied);
  }
  idx->split_shard_containing(to_span(ikey(0)));
  REQUIRE(idx->shard_count() > 1);

  std::atomic<bool> stop{false};
  std::vector<std::thread> writers;
  writers.reserve(kWriterThreads);
  for (int t = 0; t < kWriterThreads; ++t) {
    writers.emplace_back([&, t] {
      for (int round = 0; round < kRounds; ++round) {
        for (int i = t; i < kKeyRange; i += kWriterThreads) {
          CHECK(idx->put(to_span(ikey(i)), static_cast<uint64_t>(round)) == TestIndex::PutResult::Applied);
        }
      }
    });
  }
  // Keep explicit split requests active across the write phase; every target shard is already
  // populated, so scheduling order cannot turn all attempts into empty-shard no-ops.
  std::thread splitter([&] {
    for (int round = 0; !stop.load(std::memory_order_relaxed); ++round) {
      idx->split_shard_containing(to_span(ikey((round * 37) % kKeyRange)));
    }
  });

  for (auto &writer : writers) {
    writer.join();
  }
  stop.store(true, std::memory_order_relaxed);
  splitter.join();

  CHECK(idx->shard_count() > 1);
  for (int i = 0; i < kKeyRange; ++i) {
    CHECK(idx->get(to_span(ikey(i))) == static_cast<uint64_t>(kRounds - 1));
  }
}

TEST_CASE("ShardedT1Index: background worker automatically splits an oversized shard") {
  // target 50 -> split threshold 100, kCount=600 crosses it and the soft threshold, so the
  // worker picks it up with no explicit split call.
  auto idx = make_index_with_target_size(/*target_shard_size=*/50);
  constexpr int kCount = 600;
  fill_sharded(*idx, kCount);

  bool split_happened = false;
  for (int attempt = 0; attempt < 200 && !split_happened; ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    split_happened = idx->shard_count() > 1;
  }
  CHECK(split_happened);

  for (int i = 0; i < kCount; ++i) {
    CHECK(idx->get(to_span(ikey(i))) == static_cast<uint64_t>(i));
  }
}

TEST_CASE("ShardedT1Index: dirty-shard visits re-drive split evaluation") {
  // Regression: for_each_dirty_shard() merges drain appends without splitting, so with
  // interleaved visits the append never crosses the soft threshold and the maintenance
  // queue stays empty. Visits must re-drive split evaluation themselves, or a fast writer
  // keeps the shard big forever. 50 keys per round stay far below the soft threshold.
  auto idx = make_index_with_target_size(/*target_shard_size=*/50);
  constexpr int kRounds = 12;
  constexpr int kPerRound = 50;
  for (int round = 0; round < kRounds; ++round) {
    for (int i = 0; i < kPerRound; ++i) {
      const int key_id = round * kPerRound + i;
      CHECK(idx->put(to_span(ikey(key_id)), static_cast<uint64_t>(key_id)) == TestIndex::PutResult::Applied);
    }
    idx->for_each_dirty_shard([](std::span<const TestIndex::EntrySnapshot>) { return true; });
  }

  // shard_count publishes before continue_split() finishes draining references and increments
  // total_splits(), so wait for both before asserting the completed split.
  bool split_happened = false;
  for (int attempt = 0; attempt < 1000 && !split_happened; ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    split_happened = idx->shard_count() > 1 && idx->total_splits() > 0;
  }
  REQUIRE(split_happened);
  CHECK(idx->total_splits() > 0);
  check_scan_sorted_unique(*idx, 0, kRounds * kPerRound, static_cast<size_t>(kRounds * kPerRound));
}

TEST_CASE("ShardedT1Index: concurrent inserts trigger multiple automatic splits without losing data") {
  auto idx = make_index_with_target_size(/*target_shard_size=*/100);
  constexpr int kThreads = 4;
  constexpr int kPerThread = 500;

  std::vector<std::thread> writers;
  for (int t = 0; t < kThreads; ++t) {
    writers.emplace_back([&, t] {
      for (int i = 0; i < kPerThread; ++i) {
        const int key_id = (t * kPerThread) + i;
        CHECK(idx->put(to_span(ikey(key_id)), static_cast<uint64_t>(key_id)) == TestIndex::PutResult::Applied);
      }
    });
  }
  for (auto &writer : writers) {
    writer.join();
  }

  // Let queued maintenance drain before asserting on contents.
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  CHECK(idx->shard_count() > 1);
  constexpr int kTotalKeys = kThreads * kPerThread;
  for (int i = 0; i < kTotalKeys; ++i) {
    CHECK(idx->get(to_span(ikey(i))) == static_cast<uint64_t>(i));
  }

  check_scan_sorted_unique(*idx, 0, kTotalKeys, static_cast<size_t>(kTotalKeys), /*require_sorted=*/false);
}

TEST_CASE("ShardedT1Index: checkpoint_all_shards captures every live key across all shards") {
  auto idx = make_index();
  constexpr int kCount = 400;
  fill_sharded(*idx, kCount);
  idx->split_shard_containing(to_span(ikey(0)));
  REQUIRE(idx->shard_count() == 2);

  const ShardSnapshot snapshot = capture_all(*idx);

  REQUIRE(snapshot.boundaries.size() == 1);
  REQUIRE(snapshot.per_shard.size() == 2);
  size_t total_entries = 0;
  for (const auto &shard_entries : snapshot.per_shard) {
    total_entries += shard_entries.size();
  }
  CHECK(total_entries == static_cast<size_t>(kCount));
}

TEST_CASE("ShardedT1Index: recovers via load_from_checkpoint into a fresh instance") {
  auto idx = make_index();
  constexpr int kCount = 400;
  fill_sharded(*idx, kCount);
  idx->split_shard_containing(to_span(ikey(0)));
  REQUIRE(idx->shard_count() == 2);

  const ShardSnapshot snapshot = capture_all(*idx);

  auto recovered = make_index();
  recovered->load_from_checkpoint(snapshot.boundaries, snapshot.per_shard);

  CHECK(recovered->shard_count() == 2);
  for (int i = 0; i < kCount; ++i) {
    CHECK(recovered->get(to_span(ikey(i))) == static_cast<uint64_t>(i));
  }

  // The recovered instance is fully live: further put/get/split work normally.
  CHECK(recovered->put(to_span(ikey(kCount)), 999) == TestIndex::PutResult::Applied);
  CHECK(recovered->get(to_span(ikey(kCount))) == 999U);
}

TEST_CASE("ShardedT1Index: checkpoint concurrent with put/get/scan/split loses no live key") {
  auto idx = make_index();
  constexpr int kCount = 800;
  fill_sharded(*idx, kCount);

  std::atomic<bool> stop{false};
  std::thread writer([&] {
    uint64_t generation = 1;
    while (!stop.load(std::memory_order_relaxed)) {
      for (int i = 0; i < kCount && !stop.load(std::memory_order_relaxed); ++i) {
        idx->put(to_span(ikey(i)), (generation * 10000) + static_cast<uint64_t>(i));
      }
      ++generation;
    }
  });
  std::thread splitter([&] {
    for (int round = 0; round < 5 && !stop.load(std::memory_order_relaxed); ++round) {
      idx->split_shard_containing(to_span(ikey(0)));
      std::this_thread::sleep_for(std::chrono::milliseconds(15));
    }
  });

  std::vector<std::vector<TestIndex::EntrySnapshot>> per_shard;
  std::vector<TestIndex::Key> boundaries;
  for (int round = 0; round < 5; ++round) {
    per_shard.clear();
    boundaries = idx->checkpoint_all_shards(vmemkv_test::identity_offset_mapper(),
                                            [&](std::span<const TestIndex::EntrySnapshot> merged) {
                                              per_shard.emplace_back(merged.begin(), merged.end());
                                            });
    CHECK(boundaries.size() + 1 == per_shard.size());
    size_t total_entries = 0;
    for (const auto &shard_entries : per_shard) {
      total_entries += shard_entries.size();
    }
    CHECK(total_entries == static_cast<size_t>(kCount));
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  stop.store(true, std::memory_order_relaxed);
  splitter.join();
  writer.join();
}

TEST_CASE("ShardedT1Index: worker_threads=0 defers background maintenance until start_workers()") {
  // target 50 with 0 workers: puts land and cross thresholds, but nothing splits -- the
  // deferred-start contract for single-threaded setup (e.g. checkpoint load) before any
  // background thread may touch the instance.
  auto idx = std::make_unique<TestIndex>(TinyAppendConfig::kAppendCapacityEntries,
                                         /*target_shard_size=*/50,
                                         /*worker_threads=*/0);
  constexpr int kCount = 600;
  fill_sharded(*idx, kCount);

  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  CHECK(idx->shard_count() == 1);  // No worker yet -- no automatic split could have happened.

  idx->start_workers(2);

  bool split_happened = false;
  for (int attempt = 0; attempt < 200 && !split_happened; ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    split_happened = idx->shard_count() > 1;
  }
  CHECK(split_happened);
  for (int i = 0; i < kCount; ++i) {
    CHECK(idx->get(to_span(ikey(i))) == static_cast<uint64_t>(i));
  }
}

TEST_CASE(
    "ShardedT1Index: background-worker-driven splits survive a concurrent explicit "
    "split_shard_containing on the same shards") {
  // run_maintenance() must dequeue inside with_routing_guard(): an explicit split resolving
  // and CASing independently could otherwise win, complete, and delete the shard while a
  // worker still holds it unguarded. Small target size puts both paths in direct contention.
  auto idx = make_index_with_target_size(/*target_shard_size=*/20, /*worker_threads=*/4);
  constexpr int kCount = 2000;

  std::atomic<bool> stop{false};

  std::vector<std::thread> writers;
  for (int t = 0; t < 4; ++t) {
    writers.emplace_back([&, t] {
      for (int i = t; i < kCount; i += 4) {
        idx->put(to_span(ikey(i)), static_cast<uint64_t>(i));
      }
    });
  }

  // Splitter: spreads across the whole range, contending with organic worker splits.
  std::thread splitter([&] {
    int round = 0;
    while (!stop.load(std::memory_order_relaxed)) {
      idx->split_shard_containing(to_span(ikey((round * 37) % kCount)));
      ++round;
    }
  });

  // Reader: get()/scan() throughout, as in the sibling stress tests.
  std::thread reader([&] {
    while (!stop.load(std::memory_order_relaxed)) {
      for (int i = 0; i < kCount; i += 17) {
        idx->get(to_span(ikey(i)));
      }
      idx->scan(to_span(ikey(0)), to_span(ikey(kCount)), [](auto /*k*/, auto /*p*/, auto /*h*/) {});
    }
  });

  for (auto &writer : writers) {
    writer.join();
  }
  // Let workers/splitter keep contending on the populated index -- organic and explicit
  // splits race most likely here.
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  stop.store(true, std::memory_order_relaxed);
  splitter.join();
  reader.join();

  CHECK(idx->shard_count() > 1);
  check_scan_sorted_unique(*idx, 0, kCount, static_cast<size_t>(kCount), /*require_sorted=*/false);
  for (int i = 0; i < kCount; ++i) {
    CHECK(idx->get(to_span(ikey(i))) == static_cast<uint64_t>(i));
  }
}

TEST_CASE("ShardedT1Index: for_each_dirty_shard early stop releases unvisited pins") {
  // Every shard pin acquired for the visit is released when the visitor stops early.
  constexpr size_t kTargetShardSize = 300;  // Split threshold 200% -> 600 entries.
  constexpr int kKeyCount = 1500;
  auto idx = make_index_with_target_size(kTargetShardSize, /*worker_threads=*/0);
  // Construct three shards and fill their append regions below capacity.
  fill_sharded(*idx, 800);
  idx->split_shard_containing(to_span(ikey(400)));
  for (int key = 800; key < kKeyCount; ++key) {
    CHECK(idx->put(to_span(ikey(key)), static_cast<uint64_t>(key)) == TestIndex::PutResult::Applied);
  }
  idx->split_shard_containing(to_span(ikey(1000)));
  REQUIRE(idx->shard_count() == 3);

  // Mark every shard dirty for the snapshot below.
  for (int key = 0; key < kKeyCount; ++key) {
    CHECK(idx->put(to_span(ikey(key)), static_cast<uint64_t>(key)) == TestIndex::PutResult::Applied);
  }

  // Stop at the first (lowest-range) shard; the remaining two must still be released.
  idx->for_each_dirty_shard([](std::span<const TestIndex::EntrySnapshot> /*entries*/) { return false; });

  // A split on a shard outside the early-stopped visit must complete.
  const uint64_t splits_before = idx->total_splits();
  std::atomic<bool> split_completed{false};
  std::thread splitter([&] {
    idx->split_shard_containing(to_span(ikey(kKeyCount - 1)));
    split_completed.store(true, std::memory_order_release);
  });
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
  while (!split_completed.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  if (!split_completed.load(std::memory_order_acquire)) {
    std::fprintf(stderr, "split stalled after for_each_dirty_shard early stop\n");
    std::fflush(stderr);
    std::_Exit(1);
  }
  splitter.join();
  CHECK(idx->total_splits() == splits_before + 1);
}
