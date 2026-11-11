// test_compaction.cpp -- T2 segment evacuation, dirty-shard order repair, and punch tests.

#include <doctest/doctest.h>
#include <sys/stat.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <thread>
#include <vector>
#include <vmemkv/checkpoint/paths.hpp>
#include <vmemkv/impl/t2_ownership.hpp>
#include <vmemkv/t1_index/t1_codec.hpp>
#include <vmemkv/variants.hpp>

#include "test_support.hpp"

namespace {

using Store = vmemkv::StoreAdapter<vmemkv::VMemKVImpl<vmemkv::Config<>>>;
constexpr size_t kTestCapacityBytes = 256ULL * 1024 * 1024;

struct TinyShardConfig : vmemkv::Config<> {
  static constexpr size_t kAppendCapacityEntries = 128;
  static constexpr size_t kShardTargetSizeEntries = 64;
};
using TinyShardStore = vmemkv::StoreAdapter<vmemkv::VMemKVImpl<TinyShardConfig>>;

struct TinySegmentConfig : vmemkv::Config<> {
  static constexpr uint64_t kSegmentBytes = 64ULL << 10;
};
using TinySegmentStore = vmemkv::StoreAdapter<vmemkv::VMemKVImpl<TinySegmentConfig>>;

struct TinyConvergenceConfig : vmemkv::Config<> {
  static constexpr uint64_t kSegmentBytes = 64ULL << 10;
  static constexpr uint64_t kPunchBacklogCheckpointCapSegments = 4;
  static constexpr size_t kMaxBytesSinceCheckpoint = 128ULL << 10;
};
using TinyConvergenceStore = vmemkv::StoreAdapter<vmemkv::VMemKVImpl<TinyConvergenceConfig>>;

struct AutoCompactFixture {
  AutoCompactFixture() {
    if (const char *value = std::getenv("VMEMKV_SUPPRESS_AUTO_COMPACT"); value != nullptr) {
      had_previous_value = true;
      previous_value = value;
    }
    ::setenv("VMEMKV_SUPPRESS_AUTO_COMPACT", "1", 1);
  }

  ~AutoCompactFixture() {
    if (had_previous_value) {
      ::setenv("VMEMKV_SUPPRESS_AUTO_COMPACT", previous_value.c_str(), 1);
    } else {
      ::unsetenv("VMEMKV_SUPPRESS_AUTO_COMPACT");
    }
  }

  void enable() const { ::unsetenv("VMEMKV_SUPPRESS_AUTO_COMPACT"); }

 private:
  bool had_previous_value = false;
  std::string previous_value;
};

auto make_store(const std::filesystem::path &path, size_t capacity = kTestCapacityBytes) -> std::unique_ptr<Store> {
  return std::make_unique<Store>(path, capacity);
}

template <typename StoreT>
void wait_for_compaction(StoreT &store) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
  constexpr auto kIdleInterval = std::chrono::milliseconds(100);
  auto idle_since = std::chrono::steady_clock::time_point{};
  bool drained = false;
  const auto pending = [&] {
    return store.get_statistics().t2_dirty_shard_count != 0 || store.impl().pending_segment_count() != 0 ||
           store.impl().compaction_running();
  };
  while (pending() && std::chrono::steady_clock::now() < deadline) {
    idle_since = {};
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  while (std::chrono::steady_clock::now() < deadline) {
    if (pending()) {
      idle_since = {};
    } else if (idle_since == std::chrono::steady_clock::time_point{}) {
      idle_since = std::chrono::steady_clock::now();
    } else if (std::chrono::steady_clock::now() - idle_since >= kIdleInterval) {
      drained = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  REQUIRE(drained);
}

template <typename StoreT>
void drain_compaction(StoreT &store, const AutoCompactFixture &compaction) {
  compaction.enable();
  wait_for_compaction(*store);
}

template <typename Predicate>
void wait_for(Predicate &&predicate) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
  while (!predicate() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  REQUIRE(predicate());
}

auto make_key(size_t index) -> std::string { return "k" + std::to_string(100000 + index); }

auto pattern_value(size_t value_bytes, size_t index) -> std::string {
  std::string value(value_bytes, '\0');
  uint64_t state = static_cast<uint64_t>(index) * 0x9E3779B97F4A7C15ULL + 1;
  for (size_t off = 0; off < value_bytes; off += sizeof(uint64_t)) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    std::memcpy(value.data() + off, &state, std::min(sizeof(uint64_t), value_bytes - off));
  }
  return value;
}

auto shuffled_order(size_t n) -> std::vector<size_t> {
  std::vector<size_t> order(n);
  std::iota(order.begin(), order.end(), size_t{0});
  std::mt19937_64 rng(0x9E3779B97F4A7C15ULL);
  std::shuffle(order.begin(), order.end(), rng);
  return order;
}

inline const std::array<std::byte, 16> kScanLo{};
inline const std::array<std::byte, 16> kScanHi = [] {
  std::array<std::byte, 16> hi{};
  hi.fill(std::byte{0xFF});
  return hi;
}();

template <typename S>
auto inversion_rate(S &store) -> double {
  uint64_t pairs = 0;
  uint64_t inversions = 0;
  uint64_t prev_off = 0;
  bool have_prev = false;
  store->impl().t1().scan(std::span<const std::byte>(kScanLo.data(), kScanLo.size()),
                          std::span<const std::byte>(kScanHi.data(), kScanHi.size()),
                          [&](std::span<const std::byte>, uint64_t payload, uint64_t hash) {
                            if (payload == vmemkv::STORE_NOT_FOUND || t1_detail::is_inline(hash)) {
                              return;
                            }
                            const uint64_t off = payload & vmemkv::t2_detail::kPayloadOffsetMask;
                            if (have_prev) {
                              ++pairs;
                              inversions += static_cast<uint64_t>(off < prev_off);
                            }
                            prev_off = off;
                            have_prev = true;
                          });
  return pairs == 0 ? 0.0 : static_cast<double>(inversions) / static_cast<double>(pairs);
}

auto file_blocks(const std::filesystem::path &path) -> uint64_t {
  struct stat st {};
  if (::stat(path.c_str(), &st) != 0) {
    return 0;
  }
  return static_cast<uint64_t>(st.st_blocks);
}

auto t2_data_path(const std::filesystem::path &store_path) -> std::filesystem::path {
  return vmemkv::derive_t2_chk_path(store_path);
}

}  // namespace

TEST_CASE_FIXTURE(AutoCompactFixture, "ordered dirty shard needs no T2 relocation") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_compact", vmemkv::remove_store_files);
  constexpr size_t kKeys = 3000;
  constexpr size_t kValBytes = 128;
  auto store = make_store(path);
  store->bulk_load(kKeys, [&](size_t i) { return make_key(i); }, [&](size_t i) { return pattern_value(kValBytes, i); });
  store->checkpoint();
  REQUIRE(store->get_statistics().t2_dirty_shard_count > 0);
  const uint64_t used_before = store->impl().t2().bytes_used();
  const uint64_t cycles_before = store->get_statistics().compaction_cycle_count;
  enable();
  wait_for_compaction(*store);
  CHECK(store->impl().t2().bytes_used() == used_before);
  CHECK(store->get_statistics().compaction_cycle_count == cycles_before);
  CHECK(store->get_statistics().t2_dirty_shard_count == 0);
  CHECK(inversion_rate(store) == 0.0);
}

TEST_CASE_FIXTURE(AutoCompactFixture, "bulk load runs background compaction while ingesting") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_compact", vmemkv::remove_store_files);
  constexpr size_t kKeys = 10000;
  constexpr size_t kValBytes = 128;
  constexpr size_t kPauseAt = 128;
  auto store = make_store(path);
  const auto order = shuffled_order(kKeys);
  uint64_t cycles_during_bulk_load = 0;

  enable();
  store->bulk_load(
      kKeys,
      [&](size_t i) {
        if (i == kPauseAt) {
          CHECK(store->impl().pending_segment_count() == 0);
          if (store->get_statistics().compaction_cycle_count == 0) {
            CHECK(store->get_statistics().t2_dirty_shard_count > 0);
          }
          const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
          while (store->get_statistics().compaction_cycle_count == 0 && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
          }
          cycles_during_bulk_load = store->get_statistics().compaction_cycle_count;
        }
        return make_key(order[i]);
      },
      [&](size_t i) { return pattern_value(kValBytes, order[i]); });

  CHECK(cycles_during_bulk_load > 0);
  wait_for_compaction(*store);
  CHECK(store->get_statistics().t2_dirty_shard_count == 0);
  CHECK(inversion_rate(store) == 0.0);
}

TEST_CASE_FIXTURE(AutoCompactFixture, "dirty shard repack preserves live values") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_compact", vmemkv::remove_store_files);
  constexpr size_t kKeys = 10000;
  constexpr size_t kValBytes = 256;
  auto store = make_store(path);
  const auto order = shuffled_order(kKeys);
  store->bulk_load(
      kKeys,
      [&](size_t i) { return make_key(order[i]); },
      [&](size_t i) { return pattern_value(kValBytes, order[i]); });
  REQUIRE(inversion_rate(store) > 0.3);
  store->checkpoint();
  const uint64_t cycles_before = store->get_statistics().compaction_cycle_count;
  enable();
  wait_for_compaction(*store);
  CHECK(store->get_statistics().compaction_cycle_count > cycles_before);
  CHECK(store->get_statistics().t2_dirty_shard_count == 0);
  for (size_t i = 0; i < kKeys; ++i) {
    const auto got = vmemkv_test::get_optional_bytes(store, make_key(i));
    REQUIRE(got.has_value());
    CHECK(vmemkv_test::span_to_string(vmemkv_test::as_span(*got)) == pattern_value(kValBytes, i));
  }
}

TEST_CASE_FIXTURE(AutoCompactFixture, "dirty-segment evacuation preserves concurrent updates") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_compact", vmemkv::remove_store_files);
  constexpr size_t kKeys = 3000;
  constexpr size_t kValBytes = 128;
  auto store = make_store(path);
  const auto order = shuffled_order(kKeys);
  store->bulk_load(
      kKeys,
      [&](size_t i) { return make_key(order[i]); },
      [&](size_t i) { return pattern_value(kValBytes, order[i]); });
  store->checkpoint();
  std::atomic<bool> writer_paused{false};
  std::thread writer([&] {
    for (size_t i = 0; i < kKeys; ++i) {
      std::ignore = store->update(make_key(i), pattern_value(kValBytes, kKeys + i));
      if (i == 64) {
        writer_paused.store(true, std::memory_order_release);
        std::this_thread::sleep_for(std::chrono::milliseconds(1200));
      }
    }
  });
  while (!writer_paused.load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  enable();
  writer.join();
  drain_compaction(store, *this);
  CHECK(store->get_statistics().t2_dirty_shard_count == 0);
  for (size_t i = 0; i < kKeys; ++i) {
    const auto got = vmemkv_test::get_optional_bytes(store, make_key(i));
    REQUIRE(got.has_value());
    CHECK(vmemkv_test::span_to_string(vmemkv_test::as_span(*got)) == pattern_value(kValBytes, kKeys + i));
  }
}

TEST_CASE_FIXTURE(AutoCompactFixture, "T2 segment evacuation includes current versions of replaced keys") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_compact_segment", vmemkv::remove_store_files);
  constexpr size_t kKeys = 2400;
  constexpr size_t kValBytes = 128;
  auto store = std::make_unique<TinySegmentStore>(path, kTestCapacityBytes);
  store->bulk_load(kKeys, [&](size_t i) { return make_key(i); }, [&](size_t i) { return pattern_value(kValBytes, i); });
  store->checkpoint();

  std::vector<uint64_t> old_offsets(kKeys);
  for (size_t i = 0; i < kKeys; ++i) {
    const auto lookup = store->impl().t1().get_with_hash(vmemkv_test::to_span(make_key(i)));
    REQUIRE(lookup.payload_bits != vmemkv::STORE_NOT_FOUND);
    old_offsets[i] = lookup.payload_bits & vmemkv::t2_detail::kPayloadOffsetMask;
  }
  constexpr size_t kUpdatedKey = 100;
  constexpr size_t kDeletedKey = 101;
  constexpr size_t kInlineKey = 102;
  const uint64_t source_segment = vmemkv::T2Ownership<TinySegmentConfig>::seg_index(old_offsets[kUpdatedKey]);
  REQUIRE(store->update(make_key(kUpdatedKey), pattern_value(kValBytes, kKeys + kUpdatedKey)));
  REQUIRE(store->update(make_key(kUpdatedKey), pattern_value(2 * kValBytes, 2 * kKeys + kUpdatedKey)));
  REQUIRE(store->remove(make_key(kDeletedKey)));
  REQUIRE(store->update(make_key(kInlineKey), "inline"));
  const uint64_t updated_offset =
      store->impl().t1().get_with_hash(vmemkv_test::to_span(make_key(kUpdatedKey))).payload_bits &
      vmemkv::t2_detail::kPayloadOffsetMask;
  const uint64_t used_before_compact = store->impl().t2().bytes_used();

  enable();
  wait_for_compaction(*store);

  const uint64_t moved_bytes = store->impl().t2().bytes_used() - used_before_compact;
  CHECK(moved_bytes > 0);
  CHECK(moved_bytes < used_before_compact / 2);

  for (size_t i = 0; i < kKeys; ++i) {
    if (i == kDeletedKey || i == kInlineKey ||
        vmemkv::T2Ownership<TinySegmentConfig>::seg_index(old_offsets[i]) != source_segment) {
      continue;
    }
    const auto lookup = store->impl().t1().get_with_hash(vmemkv_test::to_span(make_key(i)));
    REQUIRE(lookup.payload_bits != vmemkv::STORE_NOT_FOUND);
    const uint64_t new_offset = lookup.payload_bits & vmemkv::t2_detail::kPayloadOffsetMask;
    CHECK(new_offset > old_offsets[i]);
    if (i == kUpdatedKey) {
      CHECK(new_offset >= updated_offset);
    }
  }
  drain_compaction(store, *this);
  for (size_t i = 0; i < kKeys; ++i) {
    const auto got = vmemkv_test::get_optional_bytes(store, make_key(i));
    if (i == kDeletedKey) {
      CHECK_FALSE(got.has_value());
      continue;
    }
    REQUIRE(got.has_value());
    const std::string expected = i == kInlineKey    ? "inline"
                                 : i == kUpdatedKey ? pattern_value(2 * kValBytes, 2 * kKeys + i)
                                                    : pattern_value(kValBytes, i);
    CHECK(vmemkv_test::span_to_string(vmemkv_test::as_span(*got)) == expected);
  }
}

// Punch needs FALLOC_FL_PUNCH_HOLE (tmpfs cannot punch; the store's process-level probe,
// src/vmemkv/impl/compaction.hpp punch_supported(), disables punching there), so the store
// path sits under punch_capable_root(). Corpus sizing: the bulk-loaded records span at least
// two complete 8MiB segments and the update tail appends only beyond them, so the leading
// segments fully evacuate once their records are updated away or removed, and the punches
// become observable in st_blocks.
TEST_CASE_FIXTURE(AutoCompactFixture, "overwrite and delete segment evacuation reclaims frozen segments") {
  if (vmemkv_test::punch_capable_root().empty()) {
    MESSAGE("skipped: no non-tmpfs temp root; hole punch unsupported");
    return;
  }
  const auto path =
      vmemkv_test::ScopedTempPath("vmemkv_compact", vmemkv::remove_store_files, vmemkv_test::punch_capable_root());
  constexpr size_t kKeys = 2048;
  constexpr size_t kValBytes = 8192;
  auto store = make_store(path);
  store->bulk_load(kKeys, [&](size_t i) { return make_key(i); }, [&](size_t i) { return pattern_value(kValBytes, i); });
  store->checkpoint();
  for (size_t i = 0; i < kKeys; ++i) {
    REQUIRE(store->update(make_key(i), pattern_value(2 * kValBytes, kKeys + i)));
  }
  for (size_t i = 0; i < kKeys; i += 2) {
    REQUIRE(store->remove(make_key(i)));
  }
  store->checkpoint();
  drain_compaction(store, *this);
  const uint64_t blocks_after_drain = file_blocks(t2_data_path(path));
  store->checkpoint();
  wait_for_compaction(*store);
  store->checkpoint();
  wait_for([&] { return file_blocks(t2_data_path(path)) < blocks_after_drain; });
  CHECK(file_blocks(t2_data_path(path)) < blocks_after_drain);
  for (size_t i = 0; i < kKeys; ++i) {
    const auto got = vmemkv_test::get_optional_bytes(store, make_key(i));
    if (i % 2 == 0) {
      CHECK_FALSE(got.has_value());
    } else {
      REQUIRE(got.has_value());
      CHECK(vmemkv_test::span_to_string(vmemkv_test::as_span(*got)) == pattern_value(2 * kValBytes, kKeys + i));
    }
  }
}

TEST_CASE("punch backlog cap triggers checkpoint coverage") {
  vmemkv::T2Ownership<TinyConvergenceConfig> own(16);
  constexpr uint64_t kSegBytes = TinyConvergenceConfig::kSegmentBytes;
  const auto payload_at = [](uint64_t seg) {
    return vmemkv::t2_detail::encode_offset_payload(seg * kSegBytes + 4096, /*block_count=*/1);
  };
  constexpr uint64_t kPlainHash = 0;
  for (uint64_t seg = 0; seg < TinyConvergenceConfig::kPunchBacklogCheckpointCapSegments; ++seg) {
    own.note_replace(vmemkv::STORE_NOT_FOUND, kPlainHash, true, payload_at(seg));
    own.note_replace(payload_at(seg), kPlainHash, false, 0);
  }
  CHECK(own.punch_backlog_segments() == TinyConvergenceConfig::kPunchBacklogCheckpointCapSegments);
  CHECK(vmemkv::checkpoint_detail::punch_backlog_over_cap<TinyConvergenceConfig>(own));
  // Below cap: fewer segments do not trigger the cap.
  vmemkv::T2Ownership<TinyConvergenceConfig> own_small(16);
  own_small.note_replace(vmemkv::STORE_NOT_FOUND, kPlainHash, true, payload_at(0));
  own_small.note_replace(payload_at(0), kPlainHash, false, 0);
  CHECK(own_small.punch_backlog_segments() == 1);
  CHECK_FALSE(vmemkv::checkpoint_detail::punch_backlog_over_cap<TinyConvergenceConfig>(own_small));
}

TEST_CASE("T2 garbage-segment selection ignores an unfrozen tail") {
  vmemkv::T2Ownership<TinySegmentConfig> own(2);
  const uint64_t payload = vmemkv::t2_detail::encode_offset_payload(/*offset=*/4096, /*block_count=*/1);
  own.note_replace(vmemkv::STORE_NOT_FOUND, /*old_hash=*/0, /*new_is_offset=*/true, payload);
  own.mark_segment_dirty(0);

  CHECK_FALSE(own.has_garbage_segment(0));
  CHECK(own.find_garbage_segment(0) == vmemkv::T2Ownership<TinySegmentConfig>::kInvalidSegment);
  CHECK(own.has_garbage_segment(TinySegmentConfig::kSegmentBytes));
  CHECK(own.find_garbage_segment(TinySegmentConfig::kSegmentBytes) == 0);
}

TEST_CASE_FIXTURE(AutoCompactFixture, "epoch gate defers punch by one checkpoint") {
  if (vmemkv_test::punch_capable_root().empty()) {
    MESSAGE("skipped: no non-tmpfs temp root; hole punch unsupported");
    return;
  }
  const auto path =
      vmemkv_test::ScopedTempPath("vmemkv_compact", vmemkv::remove_store_files, vmemkv_test::punch_capable_root());
  constexpr size_t kKeys = 2048;
  constexpr size_t kValBytes = 8192;
  auto store = make_store(path);
  store->bulk_load(kKeys, [&](size_t i) { return make_key(i); }, [&](size_t i) { return pattern_value(kValBytes, i); });
  store->checkpoint();
  // Updates only (no removes): every emptied segment is tagged with the same
  // epoch, so the gate's one-checkpoint deferral is observable without
  // mixed-epoch punch-through.
  for (size_t i = 0; i < kKeys; ++i) {
    REQUIRE(store->update(make_key(i), pattern_value(2 * kValBytes, kKeys + i)));
  }
  drain_compaction(store, *this);
  const uint64_t blocks_after_drain = file_blocks(t2_data_path(path));
  // First post-evacuation checkpoint: eligible only after one MORE checkpoint,
  // because an evacuation finishing after a checkpoint's writer-stop resumes can
  // miss that checkpoint's T1 dump.
  store->checkpoint();
  wait_for_compaction(*store);
  CHECK(file_blocks(t2_data_path(path)) == blocks_after_drain);
  store->checkpoint();
  wait_for([&] { return file_blocks(t2_data_path(path)) < blocks_after_drain; });
  for (size_t i = 0; i < kKeys; i += 257) {
    const auto got = vmemkv_test::get_optional_bytes(store, make_key(i));
    REQUIRE(got.has_value());
    CHECK(vmemkv_test::span_to_string(vmemkv_test::as_span(*got)) == pattern_value(2 * kValBytes, kKeys + i));
  }
}

TEST_CASE_FIXTURE(AutoCompactFixture, "punched evacuation survives restart") {
  if (vmemkv_test::punch_capable_root().empty()) {
    MESSAGE("skipped: no non-tmpfs temp root; hole punch unsupported");
    return;
  }
  const auto path =
      vmemkv_test::ScopedTempPath("vmemkv_compact", vmemkv::remove_store_files, vmemkv_test::punch_capable_root());
  constexpr size_t kKeys = 2048;
  constexpr size_t kValBytes = 8192;
  {
    auto store = make_store(path);
    store->bulk_load(
        kKeys, [&](size_t i) { return make_key(i); }, [&](size_t i) { return pattern_value(kValBytes, i); });
    store->checkpoint();
    for (size_t i = 0; i < kKeys; ++i) {
      REQUIRE(store->update(make_key(i), pattern_value(2 * kValBytes, kKeys + i)));
    }
    for (size_t i = 0; i < kKeys; i += 2) {
      REQUIRE(store->remove(make_key(i)));
    }
    store->checkpoint();
    const uint64_t blocks_before = file_blocks(t2_data_path(path));
    drain_compaction(store, *this);
    store->checkpoint();
    store->checkpoint();
    wait_for([&] { return file_blocks(t2_data_path(path)) < blocks_before; });
    // All evacuations are punchable after two post-evacuation checkpoints;
    // relocation appended ~1.1 GB while punching reclaims ~1.4 GB.
    REQUIRE(file_blocks(t2_data_path(path)) < blocks_before);
    // Post-punch foreground updates, closed without a checkpoint: recovery must
    // replay them over the punched segments.
    for (size_t i = 1; i < kKeys; i += 2) {
      REQUIRE(store->update(make_key(i), pattern_value(2 * kValBytes, 3 * kKeys + i)));
    }
  }
  {
    auto store = make_store(path);
    for (size_t i = 0; i < kKeys; ++i) {
      const auto got = vmemkv_test::get_optional_bytes(store, make_key(i));
      if (i % 2 == 0) {
        CHECK_FALSE(got.has_value());
      } else {
        REQUIRE(got.has_value());
        CHECK(vmemkv_test::span_to_string(vmemkv_test::as_span(*got)) == pattern_value(2 * kValBytes, 3 * kKeys + i));
      }
    }
  }
}

TEST_CASE_FIXTURE(AutoCompactFixture, "same-size in-place updates survive checkpoint and segment evacuation") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_compact", vmemkv::remove_store_files);
  constexpr size_t kKeys = 5000;
  constexpr size_t kValBytes = 1024;
  auto store = make_store(path);
  for (size_t i = 0; i < kKeys; ++i) {
    REQUIRE(store->insert(make_key(i), pattern_value(kValBytes, i)));
  }
  store->checkpoint();
  for (size_t i = 0; i < kKeys; ++i) {
    REQUIRE(store->update(make_key(i), pattern_value(kValBytes, kKeys + i)));
  }
  store->checkpoint();
  drain_compaction(store, *this);
  for (size_t i = 0; i < kKeys; ++i) {
    const auto got = vmemkv_test::get_optional_bytes(store, make_key(i));
    REQUIRE(got.has_value());
    CHECK(vmemkv_test::span_to_string(vmemkv_test::as_span(*got)) == pattern_value(kValBytes, kKeys + i));
  }
}

TEST_CASE_FIXTURE(AutoCompactFixture, "restart rebuilds shard order and segment accounting") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_compact", vmemkv::remove_store_files);
  constexpr size_t kKeys = 5000;
  constexpr size_t kValBytes = 256;
  const auto order = shuffled_order(kKeys);
  {
    auto store = make_store(path);
    store->bulk_load(
        kKeys,
        [&](size_t i) { return make_key(order[i]); },
        [&](size_t i) { return pattern_value(kValBytes, order[i]); });
    store->checkpoint();
  }
  {
    auto store = make_store(path);
    REQUIRE(store->get_statistics().t2_dirty_shard_count > 0);
    enable();
    wait_for_compaction(*store);
    CHECK(store->get_statistics().t2_dirty_shard_count == 0);
    for (size_t i = 0; i < kKeys; ++i) {
      const auto got = vmemkv_test::get_optional_bytes(store, make_key(i));
      REQUIRE(got.has_value());
      CHECK(vmemkv_test::span_to_string(vmemkv_test::as_span(*got)) == pattern_value(kValBytes, i));
    }
  }
}

TEST_CASE_FIXTURE(AutoCompactFixture, "T1 shard splits preserve T2 dirty state") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_compact", vmemkv::remove_store_files);
  constexpr size_t kKeys = 2000;
  constexpr size_t kValBytes = 64;
  const auto order = shuffled_order(kKeys);
  auto store = std::make_unique<TinyShardStore>(path, kTestCapacityBytes);
  for (size_t i = 0; i < kKeys; ++i) {
    REQUIRE(store->insert(make_key(order[i]), pattern_value(kValBytes, order[i])));
  }
  for (size_t retry = 0; retry < 500 && store->impl().t1().shard_count() == 1; ++retry) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  REQUIRE(store->impl().t1().shard_count() > 1);
  store->checkpoint();
  CHECK(store->get_statistics().t2_dirty_shard_count > 1);
  enable();
  wait_for_compaction(*store);
  CHECK(store->get_statistics().t2_dirty_shard_count == 0);
  for (size_t i = 0; i < kKeys; ++i) {
    const auto got = vmemkv_test::get_optional_bytes(store, make_key(i));
    REQUIRE(got.has_value());
    CHECK(vmemkv_test::span_to_string(vmemkv_test::as_span(*got)) == pattern_value(kValBytes, i));
  }
}

TEST_CASE_FIXTURE(AutoCompactFixture, "compaction convergence and punch backlog bounding under continuous churn") {
  if (vmemkv_test::punch_capable_root().empty()) {
    MESSAGE("skipped: no non-tmpfs temp root; hole punch unsupported");
    return;
  }
  const auto path =
      vmemkv_test::ScopedTempPath("vmemkv_churn", vmemkv::remove_store_files, vmemkv_test::punch_capable_root());
  constexpr size_t kKeys = 600;
  constexpr size_t kValBytes = 1024;
  auto store = std::make_unique<TinyConvergenceStore>(path, kTestCapacityBytes);

  // 1. Initial population: 600 keys with 1KB values (~600KB live data)
  for (size_t i = 0; i < kKeys; ++i) {
    REQUIRE(store->insert(make_key(i), pattern_value(kValBytes, i)));
  }
  store->checkpoint();

  // Enable background compaction during continuous churn
  enable();

  // 2. Continuous churn: multiple generations of updates creating dead segments
  constexpr size_t kGenerations = 6;
  for (size_t gen = 0; gen < kGenerations; ++gen) {
    for (size_t i = 0; i < kKeys; ++i) {
      REQUIRE(store->update(make_key(i), pattern_value(kValBytes, (gen + 1) * kKeys + i)));
    }
    // Punch backlog must stay bounded by the cap (4) plus processing latency margin
    CHECK(store->impl().punch_backlog_segments() <= 16);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }

  // 3. Quiesce and drain compaction through 2-checkpoint epoch gate
  wait_for_compaction(*store);
  store->checkpoint();
  wait_for_compaction(*store);
  store->checkpoint();

  // 4. Verify convergence: all punch candidates drained
  wait_for([&] { return store->impl().punch_backlog_segments() == 0; });
  CHECK(store->impl().punch_backlog_segments() == 0);

  // 5. Verify data integrity: all keys read back correctly with latest values
  for (size_t i = 0; i < kKeys; ++i) {
    const auto got = vmemkv_test::get_optional_bytes(store, make_key(i));
    REQUIRE(got.has_value());
    CHECK(vmemkv_test::span_to_string(vmemkv_test::as_span(*got)) ==
          pattern_value(kValBytes, kGenerations * kKeys + i));
  }
}

TEST_CASE_FIXTURE(AutoCompactFixture, "16-byte keys with shared upper 8 bytes detect inversions and repack") {
  const auto path = vmemkv_test::ScopedTempPath("vmemkv_compact_16b", vmemkv::remove_store_files);
  constexpr size_t kKeys = 10000;
  constexpr size_t kValBytes = 128;
  auto make_key_16b = [](size_t index) -> std::string {
    std::array<char, 32> buf{};
    std::snprintf(buf.data(), buf.size(), "k%015zx", index);
    return buf.data();
  };
  auto store = make_store(path);
  const auto order = shuffled_order(kKeys);
  store->bulk_load(
      kKeys,
      [&](size_t i) { return make_key_16b(order[i]); },
      [&](size_t i) { return pattern_value(kValBytes, order[i]); });
  REQUIRE(inversion_rate(store) > 0.3);
  enable();
  store->drain_compaction();
  wait_for_compaction(*store);
  CHECK(store->get_statistics().t2_dirty_shard_count == 0);
  CHECK(inversion_rate(store) == 0.0);
}
