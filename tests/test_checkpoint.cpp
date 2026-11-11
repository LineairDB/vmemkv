// test_checkpoint.cpp — Sharded T1 checkpoint file + manifest formats, in isolation
// (no VMemKVImpl).

#include <doctest/doctest.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <vmemkv/checkpoint/checkpoint.hpp>
#include <vmemkv/t1_index/t1_index.hpp>

#include "test_support.hpp"

namespace {

using CheckpointEntrySnapshot = vmemkv::T1Index<vmemkv::Config<>>::EntrySnapshot;

auto make_chk_prefix(std::string_view text) -> vmemkv::T1ChkKeyPrefix {
  vmemkv::T1ChkKeyPrefix key{};
  std::memcpy(key.data(), text.data(), std::min(text.size(), key.size()));
  return key;
}

}  // namespace

TEST_CASE("Manifest: write then read round-trips generation/t2_bytes_used") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_chk_test", vmemkv::remove_quiet);
  vmemkv::write_manifest(path, 42, 12345);

  const auto manifest = vmemkv::read_manifest(path);
  REQUIRE(manifest.has_value());
  CHECK(manifest->generation == 42);
  CHECK(manifest->t2_bytes_used == 12345);
}

TEST_CASE("Manifest: re-writing replaces the generation atomically") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_chk_test", vmemkv::remove_quiet);
  vmemkv::write_manifest(path, 1, 100);
  vmemkv::write_manifest(path, 2, 200);

  const auto manifest = vmemkv::read_manifest(path);
  REQUIRE(manifest.has_value());
  CHECK(manifest->generation == 2);
  CHECK(manifest->t2_bytes_used == 200);
}

TEST_CASE("Manifest: missing file reads as nullopt, not an error") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_chk_test", vmemkv::remove_quiet);
  const auto manifest = vmemkv::read_manifest(path);
  CHECK_FALSE(manifest.has_value());
}

TEST_CASE("Manifest: corrupted checksum reads as nullopt") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_chk_test", vmemkv::remove_quiet);
  vmemkv::write_manifest(path, 7, 70);

  vmemkv_test::flip_byte_at(path.get(), static_cast<std::streamoff>(offsetof(vmemkv::ManifestHeader, checksum)));

  CHECK_FALSE(vmemkv::read_manifest(path).has_value());
}

TEST_CASE("Manifest: truncated file reads as nullopt") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_chk_test", vmemkv::remove_quiet);
  {
    std::ofstream out(path.get(), std::ios::binary);
    constexpr std::array<char, 4> garbage{};
    out.write(garbage.data(), garbage.size());
  }
  CHECK_FALSE(vmemkv::read_manifest(path).has_value());
}

TEST_CASE("Sharded T1 checkpoint: write then read round-trips multiple shards and boundaries") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_chk_test", vmemkv::remove_quiet);
  const std::vector<CheckpointEntrySnapshot> shard0 = {{make_chk_prefix("aaa"), 1, 0x11},
                                                       {make_chk_prefix("bbb"), 2, 0x22}};
  const std::vector<CheckpointEntrySnapshot> shard1 = {{make_chk_prefix("ccc"), 3, 0x33}};
  const std::vector<vmemkv::T1ChkKeyPrefix> boundaries = {make_chk_prefix("ccc")};

  {
    vmemkv::ShardedT1CheckpointWriter writer(path);
    writer.add_shard(std::span<const CheckpointEntrySnapshot>(shard0));
    writer.add_shard(std::span<const CheckpointEntrySnapshot>(shard1));
    writer.finish(boundaries);
  }

  vmemkv::ShardedT1CheckpointFile loaded(path);
  REQUIRE(loaded.shard_count() == 2);
  REQUIRE(loaded.boundaries().size() == 1);
  CHECK(loaded.boundaries()[0] == make_chk_prefix("ccc"));

  const auto loaded_shard0 = loaded.shard_entries(0);
  REQUIRE(loaded_shard0.size() == 2);
  CHECK(loaded_shard0[0].key_prefix == make_chk_prefix("aaa"));
  CHECK(loaded_shard0[1].key_prefix == make_chk_prefix("bbb"));

  const auto loaded_shard1 = loaded.shard_entries(1);
  REQUIRE(loaded_shard1.size() == 1);
  CHECK(loaded_shard1[0].key_prefix == make_chk_prefix("ccc"));
  CHECK(loaded_shard1[0].payload_bits == 3);
}

TEST_CASE("Sharded T1 checkpoint: single shard, zero boundaries round-trips (K=1 case)") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_chk_test", vmemkv::remove_quiet);
  const std::vector<CheckpointEntrySnapshot> shard0 = {{make_chk_prefix("a"), 1, 1}};

  {
    vmemkv::ShardedT1CheckpointWriter writer(path);
    writer.add_shard(std::span<const CheckpointEntrySnapshot>(shard0));
    writer.finish({});
  }

  vmemkv::ShardedT1CheckpointFile loaded(path);
  CHECK(loaded.shard_count() == 1);
  CHECK(loaded.boundaries().empty());
  REQUIRE(loaded.shard_entries(0).size() == 1);
  CHECK(loaded.shard_entries(0)[0].key_prefix == make_chk_prefix("a"));
}

TEST_CASE("Sharded T1 checkpoint: a shard with zero live entries round-trips") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_chk_test", vmemkv::remove_quiet);
  const std::vector<CheckpointEntrySnapshot> empty_shard;
  const std::vector<CheckpointEntrySnapshot> shard1 = {{make_chk_prefix("z"), 9, 9}};

  {
    vmemkv::ShardedT1CheckpointWriter writer(path);
    writer.add_shard(std::span<const CheckpointEntrySnapshot>(empty_shard));
    writer.add_shard(std::span<const CheckpointEntrySnapshot>(shard1));
    const std::vector<vmemkv::T1ChkKeyPrefix> boundaries = {make_chk_prefix("z")};
    writer.finish(boundaries);
  }

  vmemkv::ShardedT1CheckpointFile loaded(path);
  REQUIRE(loaded.shard_count() == 2);
  CHECK(loaded.shard_entries(0).empty());
  REQUIRE(loaded.shard_entries(1).size() == 1);
}

TEST_CASE("Sharded T1 checkpoint: missing file throws") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_chk_test", vmemkv::remove_quiet);
  CHECK_THROWS(vmemkv::ShardedT1CheckpointFile(path));
}

TEST_CASE("Sharded T1 checkpoint: destroying the writer without calling finish() leaves no file") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_chk_test", vmemkv::remove_quiet);
  {
    vmemkv::ShardedT1CheckpointWriter writer(path);
    writer.add_shard(
        std::span<const CheckpointEntrySnapshot>(std::vector<CheckpointEntrySnapshot>{{make_chk_prefix("a"), 1, 1}}));
    // No finish(): the destructor discards the temp file, never touching `path`.
  }
  CHECK_FALSE(std::filesystem::exists(path));
}

TEST_CASE("Sharded T1 checkpoint: corrupted checksum is detected") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_chk_test", vmemkv::remove_quiet);
  {
    vmemkv::ShardedT1CheckpointWriter writer(path);
    writer.add_shard(
        std::span<const CheckpointEntrySnapshot>(std::vector<CheckpointEntrySnapshot>{{make_chk_prefix("k"), 1, 1}}));
    writer.finish({});
  }

  // Flip a payload byte, pre-trailer: magic/version/layout checks can't catch this, only the
  // checksum can.
  constexpr std::streamoff kEntryCountBytes = 8;
  vmemkv_test::flip_byte_at(path.get(), kEntryCountBytes);

  CHECK_THROWS(vmemkv::ShardedT1CheckpointFile(path));
}

TEST_CASE("Sharded T1 checkpoint: truncated file is detected") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_chk_test", vmemkv::remove_quiet);
  {
    vmemkv::ShardedT1CheckpointWriter writer(path);
    writer.add_shard(
        std::span<const CheckpointEntrySnapshot>(std::vector<CheckpointEntrySnapshot>{{make_chk_prefix("k"), 1, 1}}));
    writer.finish({});
  }

  const auto full_size = std::filesystem::file_size(path);
  std::filesystem::resize_file(path, full_size - 4);

  CHECK_THROWS(vmemkv::ShardedT1CheckpointFile(path));
}

TEST_CASE("Path derivation: manifest/t1chk/t2chk paths are distinct and stable") {
  const std::filesystem::path t2_path = "/tmp/some_store";

  const auto manifest_path = vmemkv::derive_manifest_path(t2_path);
  const auto t1chk_path = vmemkv::derive_t1_chk_path(t2_path);
  const auto t1chk_path_again = vmemkv::derive_t1_chk_path(t2_path);
  const auto t2chk_path = vmemkv::derive_t2_chk_path(t2_path);

  CHECK(manifest_path != t1chk_path);
  CHECK(t1chk_path == t1chk_path_again);  // deterministic for the same t2_path
  CHECK(t1chk_path != t2chk_path);        // T1 chk and T2 data files are separate paths
}

#include <chrono>
#include <memory>
#include <thread>
#include <vmemkv/variants.hpp>

namespace {

using vmemkv_test::check_range;
using vmemkv_test::insert_range;

constexpr uint64_t kStoreCapacityBytes = 64ULL * 1024 * 1024;

#define CRASH_CHECKPOINT_STORE_TYPES vmemkv::variants::VMemKV_Default

}  // namespace

namespace {
// 1024-entry append region.
struct TinyShardConfig : vmemkv_test::TinyAppendConfig<10> {
  static constexpr size_t kShardTargetSizeEntries = 50;  // Forces splits well before kKeyCount below.
};
using VMemKV_TinyShard = vmemkv::StoreAdapter<vmemkv::VMemKVImpl<TinyShardConfig>>;
}  // namespace

// ─── Checkpoint / Reload integration tests ──────────────────────────────────────────────────
// See docs/specification/low_level_design.md 5.2-5.5. Through the public Store interface plus manifest/file
// inspection only -- never VMemKVImpl internals directly.

TEST_CASE_TEMPLATE("checkpoint: explicit checkpoint() persists a manifest and survives restart",
                   Store,
                   CRASH_CHECKPOINT_STORE_TYPES) {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_crash", vmemkv::remove_store_files);
  constexpr int kKeyCount = 50;
  {
    auto store = std::make_unique<Store>(path, kStoreCapacityBytes);
    insert_range(*store, kKeyCount);
    store->impl().checkpoint();
    CHECK(std::filesystem::exists(vmemkv::derive_manifest_path(path)));
  }
  {
    auto store = std::make_unique<Store>(path, kStoreCapacityBytes);
    check_range(*store, kKeyCount);
  }
}

TEST_CASE("checkpoint: rolls the WAL onto a fresh, empty active segment") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_crash", vmemkv::remove_store_files);
  constexpr int kKeyCount = 200;
  auto store = std::make_unique<vmemkv::variants::VMemKV_Default>(path, kStoreCapacityBytes);
  for (int i = 0; i < kKeyCount; ++i) {
    REQUIRE(store->insert("k" + std::to_string(i), vmemkv_test::indexed_value(i)));
  }
  const auto wal_path = vmemkv::derive_wal_path(path);
  const auto active_before = vmemkv::find_active_wal_segment(wal_path);
  REQUIRE(active_before.has_value());
  const uint64_t wal_size_before = std::filesystem::file_size(*active_before);

  store->impl().checkpoint();

  const auto active_after = vmemkv::find_active_wal_segment(wal_path);
  REQUIRE(active_after.has_value());
  CHECK(*active_after != *active_before);  // rotate_segment() rolled onto a new generation.
  const uint64_t wal_size_after = std::filesystem::file_size(*active_after);
  CHECK(wal_size_after < wal_size_before);  // The new active segment starts empty.
}

TEST_CASE("checkpoint: deletes and updates after a checkpoint are correctly reflected after restart") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_crash", vmemkv::remove_store_files);
  constexpr int kKeyCount = 30;
  {
    auto store = std::make_unique<vmemkv::variants::VMemKV_Default>(path, kStoreCapacityBytes);
    for (int i = 0; i < kKeyCount; ++i) {
      REQUIRE(store->insert("k" + std::to_string(i), vmemkv_test::indexed_value(i)));
    }
    store->impl().checkpoint();  // Checkpoint captures all 30 keys.

    // Tail activity after the checkpoint: delete half, update the rest.
    for (int i = 0; i < kKeyCount / 2; ++i) {
      REQUIRE(store->remove("k" + std::to_string(i)));
    }
    for (int i = kKeyCount / 2; i < kKeyCount; ++i) {
      REQUIRE(store->update("k" + std::to_string(i), vmemkv_test::indexed_value(1000 + i)));
    }
  }
  {
    auto store = std::make_unique<vmemkv::variants::VMemKV_Default>(path, kStoreCapacityBytes);
    for (int i = 0; i < kKeyCount / 2; ++i) {
      CHECK_FALSE(vmemkv_test::get_optional_bytes(store, "k" + std::to_string(i)).has_value());
    }
    for (int i = kKeyCount / 2; i < kKeyCount; ++i) {
      const auto val = vmemkv_test::get_optional_bytes(store, "k" + std::to_string(i));
      REQUIRE(val.has_value());
      CHECK(vmemkv_test::span_to_string(vmemkv_test::as_span(*val)) == vmemkv_test::indexed_value(1000 + i));
    }
  }
}

TEST_CASE("checkpoint: insert/checkpoint/insert-more/restart preserves both pre- and post-checkpoint keys") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_crash", vmemkv::remove_store_files);
  constexpr int kFirstBatch = 20;
  constexpr int kSecondBatch = 20;
  {
    auto store = std::make_unique<vmemkv::variants::VMemKV_Default>(path, kStoreCapacityBytes);
    for (int i = 0; i < kFirstBatch; ++i) {
      REQUIRE(store->insert("a" + std::to_string(i), vmemkv_test::indexed_value(i)));
    }
    store->impl().checkpoint();
    for (int i = 0; i < kSecondBatch; ++i) {
      REQUIRE(store->insert("b" + std::to_string(i), vmemkv_test::indexed_value(1000 + i)));
    }
  }
  {
    auto store = std::make_unique<vmemkv::variants::VMemKV_Default>(path, kStoreCapacityBytes);
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

TEST_CASE("checkpoint: a valid manifest pointing at a missing T1 checkpoint file fails construction loudly") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_crash", vmemkv::remove_store_files);
  {
    auto store = std::make_unique<vmemkv::variants::VMemKV_Default>(path, kStoreCapacityBytes);
    REQUIRE(store->insert("k", vmemkv_test::indexed_value(0)));
    store->impl().checkpoint();
  }
  const auto manifest = vmemkv::read_manifest(vmemkv::derive_manifest_path(path));
  REQUIRE(manifest.has_value());
  // T1 file lost with the WAL already rotated to the tail: full replay would silently lose the
  // pre-checkpoint key, so construction must fail loudly instead (see load_checkpoint_if_present).
  std::filesystem::remove(vmemkv::derive_t1_chk_path(path));

  CHECK_THROWS(std::make_unique<vmemkv::variants::VMemKV_Default>(path, kStoreCapacityBytes));
}

// checkpoint()'s first-ever call must create the T2 checkpoint file (O_CREAT), not assume one
// exists. Default (large) kMaxBytesSinceCheckpoint: a tiny override would let the worker's
// auto-trigger race the explicit call below, flaking the exact-count assertion.
TEST_CASE("checkpoint(): first call on a fresh store creates the T2 checkpoint file") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_crash", vmemkv::remove_store_files);
  auto store = std::make_unique<vmemkv::variants::VMemKV_Default>(path, kStoreCapacityBytes);
  for (int i = 0; i < 200; ++i) {
    REQUIRE(store->insert("k" + std::to_string(i), vmemkv_test::indexed_value(i)));
  }
  const auto stats_before = store->impl().get_statistics();
  store->impl().checkpoint();
  const auto stats_after = store->impl().get_statistics();

  CHECK(std::filesystem::exists(vmemkv::derive_manifest_path(path)));
  CHECK(stats_after.checkpoint_count == stats_before.checkpoint_count + 1);
  CHECK(std::filesystem::exists(vmemkv::derive_t2_chk_path(path)));

  for (int i = 0; i < 200; ++i) {
    const auto val = vmemkv_test::get_optional_bytes(store, "k" + std::to_string(i));
    REQUIRE(val.has_value());
    CHECK(vmemkv_test::span_to_string(vmemkv_test::as_span(*val)) == vmemkv_test::indexed_value(i));
  }
}

// Repeated checkpoint()s durabilize each cycle's tail in place (same paths, advancing LSN)
// and restart adopts the final state. Default kMaxBytesSinceCheckpoint (see above: a tiny
// override would race auto-triggered cycles).
TEST_CASE("checkpoint: repeated checkpoint() calls durabilize in place and survive restart") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_crash", vmemkv::remove_store_files);
  auto store = std::make_unique<vmemkv::variants::VMemKV_Default>(path, kStoreCapacityBytes);

  REQUIRE(store->insert("seed", vmemkv_test::indexed_value(-1)));
  store->impl().checkpoint();
  const auto manifest1 = vmemkv::read_manifest(vmemkv::derive_manifest_path(path));
  REQUIRE(manifest1.has_value());
  const auto t2_chk_path = vmemkv::derive_t2_chk_path(path);
  REQUIRE(std::filesystem::exists(t2_chk_path));

  const auto stats_after_seed = store->impl().get_statistics();

  // Cycles 2-4: insert-only, then an explicit checkpoint() call each round.
  constexpr int kCycles = 3;
  constexpr int kKeysPerCycle = 200;
  uint64_t last_generation = manifest1->generation;
  for (int cycle = 0; cycle < kCycles; ++cycle) {
    for (int i = 0; i < kKeysPerCycle; ++i) {
      REQUIRE(store->insert("c" + std::to_string(cycle) + "_" + std::to_string(i), vmemkv_test::indexed_value(i)));
    }
    store->impl().checkpoint();

    const auto manifest_now = vmemkv::read_manifest(vmemkv::derive_manifest_path(path));
    REQUIRE(manifest_now.has_value());
    CHECK(manifest_now->generation != last_generation);  // checkpoint_lsn advances every cycle.
    last_generation = manifest_now->generation;
    REQUIRE(std::filesystem::exists(t2_chk_path));  // Same file, every cycle.
  }

  const auto stats_after_cycles = store->impl().get_statistics();
  CHECK(stats_after_cycles.checkpoint_count ==
        stats_after_seed.checkpoint_count + kCycles);  // Every checkpoint() ran a cycle.

  // A genuine restart adopts the final checkpoint correctly.
  store.reset();
  auto restarted = std::make_unique<vmemkv::variants::VMemKV_Default>(path, kStoreCapacityBytes);
  CHECK(vmemkv_test::get_optional_bytes(restarted, "seed").has_value());
  for (int cycle = 0; cycle < kCycles; ++cycle) {
    for (int i = 0; i < kKeysPerCycle; ++i) {
      const auto val =
          vmemkv_test::get_optional_bytes(restarted, "c" + std::to_string(cycle) + "_" + std::to_string(i));
      REQUIRE(val.has_value());
      CHECK(vmemkv_test::span_to_string(vmemkv_test::as_span(*val)) == vmemkv_test::indexed_value(i));
    }
  }
}

// Stress: in-place updates racing concurrent checkpoint()s stay correct live and across
// restart. Black-box, no hooks needed: durabilization is a byte-range msync() over the live
// mapping itself, so no captured snapshot can go stale mid-update. Per-key dedicated updaters
// pin down each key's last value (update() returns past its WAL fsync); a thread hammers
// checkpoint() throughout.
TEST_CASE("checkpoint: in-place updates racing concurrent checkpoint() stay correct live and survive a restart") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_crash", vmemkv::remove_store_files);
  constexpr int kKeyCount = 8;
  constexpr int kUpdatesPerKey = 500;
  constexpr std::size_t kValueBytes = 200;

  auto key_for = [](int i) { return "racer" + std::to_string(i); };
  auto value_for = [](int i, int round) {
    return std::string(kValueBytes, static_cast<char>('a' + ((i + round) % 26)));
  };

  std::array<std::string, kKeyCount> last_written;
  {
    auto store = std::make_unique<vmemkv::variants::VMemKV_Default>(path, kStoreCapacityBytes);
    for (int i = 0; i < kKeyCount; ++i) {
      REQUIRE(store->insert(key_for(i), value_for(i, 0)));
    }

    std::atomic<bool> stop{false};
    std::thread checkpointer([&] {
      while (!stop.load(std::memory_order_relaxed)) {
        store->checkpoint();
      }
    });

    std::vector<std::thread> updaters;
    updaters.reserve(kKeyCount);
    for (int i = 0; i < kKeyCount; ++i) {
      updaters.emplace_back([&, i] {
        for (int round = 1; round <= kUpdatesPerKey; ++round) {
          REQUIRE(store->update(key_for(i), value_for(i, round)));
        }
      });
    }
    for (auto &updater : updaters) {
      updater.join();
    }
    for (int i = 0; i < kKeyCount; ++i) {
      last_written[static_cast<std::size_t>(i)] = value_for(i, kUpdatesPerKey);
    }

    stop.store(true, std::memory_order_relaxed);
    checkpointer.join();

    // Live reads must reflect each key's actual last-written value, never a stale snapshot from
    // a checkpoint cycle that raced it.
    for (int i = 0; i < kKeyCount; ++i) {
      const auto live = vmemkv_test::get_optional_bytes(store, key_for(i));
      REQUIRE(live.has_value());
      CHECK(vmemkv_test::span_to_string(vmemkv_test::as_span(*live)) == last_written[static_cast<std::size_t>(i)]);
    }
  }
  // Restart without a final checkpoint: recovery must reach the same state via WAL replay,
  // regardless of what any racing checkpoint cycle durabilized.
  {
    auto store = std::make_unique<vmemkv::variants::VMemKV_Default>(path, kStoreCapacityBytes);
    for (int i = 0; i < kKeyCount; ++i) {
      const auto val = vmemkv_test::get_optional_bytes(store, key_for(i));
      REQUIRE(val.has_value());
      CHECK(vmemkv_test::span_to_string(vmemkv_test::as_span(*val)) == last_written[static_cast<std::size_t>(i)]);
    }
  }
}

// Sharding round trip: force real splits (small target size) before checkpointing, so the
// sharded writer/reader pair is exercised past the K=1 case every other test stays within.
TEST_CASE("checkpoint: sharded T1 survives checkpoint and restart with multiple shards") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_crash", vmemkv::remove_store_files);
  constexpr int kKeyCount = 2000;
  size_t shard_count_before_restart = 0;
  {
    auto store = std::make_unique<VMemKV_TinyShard>(path, kStoreCapacityBytes);
    for (int i = 0; i < kKeyCount; ++i) {
      REQUIRE(store->insert("k" + std::to_string(i), vmemkv_test::indexed_value(i)));
    }
    // Give the background worker pool time to notice and split.
    for (int attempt = 0; attempt < 200 && store->impl().t1().shard_count() == 1; ++attempt) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    shard_count_before_restart = store->impl().t1().shard_count();
    REQUIRE(shard_count_before_restart > 1);
    store->checkpoint();
  }
  {
    auto store = std::make_unique<VMemKV_TinyShard>(path, kStoreCapacityBytes);
    CHECK(store->impl().t1().shard_count() == shard_count_before_restart);
    for (int i = 0; i < kKeyCount; ++i) {
      const auto val = vmemkv_test::get_optional_bytes(store, "k" + std::to_string(i));
      REQUIRE(val.has_value());
      CHECK(vmemkv_test::span_to_string(vmemkv_test::as_span(*val)) == vmemkv_test::indexed_value(i));
    }
  }
}
