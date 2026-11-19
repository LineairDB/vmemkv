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
#include <vmemkv/core/paths.hpp>
#include <vmemkv/impl/compaction.hpp>
#include <vmemkv/impl/t2_ownership.hpp>
#include <vmemkv/t1_index/t1_codec.hpp>
#include <vmemkv/variants.hpp>

#include "test_support.hpp"

namespace {

using Store = vmemkv::StoreAdapter<vmemkv::VMemKVImpl>;
constexpr size_t kTestCapacityBytes = 256ULL * 1024 * 1024;

constexpr size_t kTinyShardMemTableEntries = 128;
constexpr size_t kTinyShardTargetEntries = 64;
struct TinyShardImpl : vmemkv::VMemKVImpl {
  TinyShardImpl(const std::filesystem::path &path, uint64_t cap)
      : VMemKVImpl(path, cap, [] {
          vmemkv::StoreOptions opts;
          opts.memtable_capacity_entries = kTinyShardMemTableEntries;
          opts.shard_target_size_entries = kTinyShardTargetEntries;
          return opts;
        }()) {}
};
using TinyShardStore = vmemkv::StoreAdapter<TinyShardImpl>;

constexpr uint64_t kTinySegmentBytes = 64ULL << 10;
struct TinySegmentImpl : vmemkv::VMemKVImpl {
  TinySegmentImpl(const std::filesystem::path &path, uint64_t cap)
      : VMemKVImpl(path, cap, [] {
          vmemkv::StoreOptions opts;
          opts.segment_bytes = kTinySegmentBytes;
          opts.shard_target_size_entries = 256;
          return opts;
        }()) {}
};
using TinySegmentStore = vmemkv::StoreAdapter<TinySegmentImpl>;

constexpr uint64_t kTinyConvergenceCapSegments = 4;
struct TinyConvergenceImpl : vmemkv::VMemKVImpl {
  TinyConvergenceImpl(const std::filesystem::path &path, uint64_t cap)
      : VMemKVImpl(path, cap, [] {
          vmemkv::StoreOptions opts;
          opts.segment_bytes = kTinySegmentBytes;
          opts.punch_backlog_checkpoint_cap_segments = kTinyConvergenceCapSegments;
          opts.wal_max_bytes_since_checkpoint = 128ULL << 10;
          return opts;
        }()) {}
};
using TinyConvergenceStore = vmemkv::StoreAdapter<TinyConvergenceImpl>;

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
auto count_inversions(S &store, uint64_t *out_pairs = nullptr) -> uint64_t {
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
  if (out_pairs != nullptr) {
    *out_pairs = pairs;
  }
  return inversions;
}

template <typename S>
auto inversion_rate(S &store) -> double {
  uint64_t pairs = 0;
  const uint64_t inversions = count_inversions(store, &pairs);
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
  store->impl().t1().drain_maintenance();
  store->checkpoint();
  REQUIRE(store->get_statistics().t2_dirty_shard_count > 0);
  const uint64_t used_before = store->impl().t2().tail_offset();
  const uint64_t cycles_before = store->get_statistics().compaction_cycle_count;
  enable();
  wait_for_compaction(*store);
  CHECK(store->impl().t2().tail_offset() == used_before);
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
  store->impl().t1().drain_maintenance();
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
  const uint64_t source_segment = vmemkv::T2Ownership::seg_index(old_offsets[kUpdatedKey], kTinySegmentBytes);
  auto is_deleted_key = [&](size_t i) -> bool {
    if (i == kDeletedKey) {
      return true;
    }
    if (i == kUpdatedKey || i == kInlineKey) {
      return false;
    }
    return i >= 150 && vmemkv::T2Ownership::seg_index(old_offsets[i], kTinySegmentBytes) == source_segment;
  };
  REQUIRE(store->update(make_key(kUpdatedKey), pattern_value(kValBytes, kKeys + kUpdatedKey)));
  REQUIRE(store->update(make_key(kUpdatedKey), pattern_value(2 * kValBytes, 2 * kKeys + kUpdatedKey)));
  for (size_t i = 0; i < kKeys; ++i) {
    if (is_deleted_key(i)) {
      REQUIRE(store->remove(make_key(i)));
    }
  }
  REQUIRE(store->update(make_key(kInlineKey), "inline"));
  const uint64_t updated_offset =
      store->impl().t1().get_with_hash(vmemkv_test::to_span(make_key(kUpdatedKey))).payload_bits &
      vmemkv::t2_detail::kPayloadOffsetMask;
  const uint64_t used_before_compact = store->impl().t2().tail_offset();

  enable();
  wait_for_compaction(*store);

  const uint64_t moved_bytes = store->impl().t2().tail_offset() - used_before_compact;
  CHECK(moved_bytes > 0);
  CHECK(moved_bytes < used_before_compact / 2);

  for (size_t i = 0; i < kKeys; ++i) {
    if (is_deleted_key(i) || i == kInlineKey ||
        vmemkv::T2Ownership::seg_index(old_offsets[i], kTinySegmentBytes) != source_segment) {
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
    if (is_deleted_key(i)) {
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
  constexpr size_t kKeys = 192;
  constexpr size_t kValBytes = 2048;
  auto store = std::make_unique<TinySegmentStore>(path, kTestCapacityBytes);
  store->bulk_load(kKeys, [&](size_t i) { return make_key(i); }, [&](size_t i) { return pattern_value(kValBytes, i); });
  store->impl().t1().drain_maintenance();
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
  constexpr uint64_t kSegBytes = kTinySegmentBytes;
  vmemkv::T2Ownership own(16, kSegBytes);
  const auto payload_at = [](uint64_t seg) {
    return vmemkv::t2_detail::encode_offset_payload(seg * kSegBytes + 4096, /*block_count=*/1);
  };
  constexpr uint64_t kPlainHash = 0;
  for (uint64_t seg = 0; seg < kTinyConvergenceCapSegments; ++seg) {
    own.note_replace(vmemkv::STORE_NOT_FOUND, kPlainHash, true, payload_at(seg));
    own.note_replace(payload_at(seg), kPlainHash, false, 0);
  }
  CHECK(own.punch_backlog_segments() == kTinyConvergenceCapSegments);
  CHECK(vmemkv::checkpoint_detail::punch_backlog_over_cap(own, kTinyConvergenceCapSegments));
  // Below cap: fewer segments do not trigger the cap.
  vmemkv::T2Ownership own_small(16, kSegBytes);
  own_small.note_replace(vmemkv::STORE_NOT_FOUND, kPlainHash, true, payload_at(0));
  own_small.note_replace(payload_at(0), kPlainHash, false, 0);
  CHECK(own_small.punch_backlog_segments() == 1);
  CHECK_FALSE(vmemkv::checkpoint_detail::punch_backlog_over_cap(own_small, kTinyConvergenceCapSegments));
}

TEST_CASE("T2 garbage-segment selection ignores an unfrozen tail") {
  vmemkv::T2Ownership own(2, kTinySegmentBytes);
  const uint64_t payload = vmemkv::t2_detail::encode_offset_payload(/*offset=*/4096, /*block_count=*/1);
  own.note_replace(vmemkv::STORE_NOT_FOUND, /*old_hash=*/0, /*new_is_offset=*/true, payload);
  own.mark_segment_dirty(0);

  CHECK_FALSE(own.has_garbage_segment(0));
  CHECK_FALSE(own.is_garbage_segment(0, 0));
  CHECK(own.garbage_segment_count(0) == 0);
  CHECK(own.has_garbage_segment(kTinySegmentBytes));
  CHECK(own.is_garbage_segment(0, kTinySegmentBytes));
  CHECK(own.garbage_segment_count(kTinySegmentBytes) == 1);
}

TEST_CASE_FIXTURE(AutoCompactFixture, "epoch gate defers punch by one checkpoint") {
  if (vmemkv_test::punch_capable_root().empty()) {
    MESSAGE("skipped: no non-tmpfs temp root; hole punch unsupported");
    return;
  }
  const auto path =
      vmemkv_test::ScopedTempPath("vmemkv_compact", vmemkv::remove_store_files, vmemkv_test::punch_capable_root());
  constexpr size_t kKeys = 192;
  constexpr size_t kValBytes = 2048;
  auto store = std::make_unique<TinySegmentStore>(path, kTestCapacityBytes);
  store->bulk_load(kKeys, [&](size_t i) { return make_key(i); }, [&](size_t i) { return pattern_value(kValBytes, i); });
  store->impl().t1().drain_maintenance();
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
  // First post-evacuation checkpoint may flush delayed-allocated ext4 pages (increasing
  // st_blocks slightly), but must not punch yet.
  CHECK(store->impl().punch_backlog_segments() > 0);
  CHECK(file_blocks(t2_data_path(path)) >= blocks_after_drain);
  store->checkpoint();
  wait_for([&] { return file_blocks(t2_data_path(path)) < blocks_after_drain; });
  for (size_t i = 0; i < kKeys; i += 19) {
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
  constexpr size_t kKeys = 192;
  constexpr size_t kValBytes = 2048;
  {
    auto store = std::make_unique<TinySegmentStore>(path, kTestCapacityBytes);
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
    REQUIRE(file_blocks(t2_data_path(path)) < blocks_before);
    // Post-punch foreground updates, closed without a checkpoint: recovery must
    // replay them over the punched segments.
    for (size_t i = 1; i < kKeys; i += 2) {
      REQUIRE(store->update(make_key(i), pattern_value(2 * kValBytes, 3 * kKeys + i)));
    }
  }
  {
    auto store = std::make_unique<TinySegmentStore>(path, kTestCapacityBytes);
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
  store->impl().t1().drain_maintenance();
  const size_t shards_before_checkpoint = store->impl().t1().shard_count();
  REQUIRE(shards_before_checkpoint > 1);
  store->checkpoint();
  store->impl().t1().drain_maintenance();
  CHECK(store->get_statistics().t2_dirty_shard_count == store->impl().t1().shard_count());
  enable();
  wait_for_compaction(*store);
  store->impl().t1().drain_maintenance();
  CHECK(store->get_statistics().t2_dirty_shard_count == 0);

  // Every shard must independently reach intra-shard page locality after compaction,
  // regardless of where asynchronous split boundaries landed.
  const vmemkv::T2Memory *mem = store->impl().t2().get_memory();
  REQUIRE(mem != nullptr);
  size_t visited_shards = 0;
  store->impl().t1().for_each_shard_for_compaction(
      [&](std::span<const TinyShardStore::Impl::T1IndexT::EntrySnapshot> entries) {
        ++visited_shards;
        CHECK(vmemkv::compaction_detail::shard_pages_local(entries, mem));
        return true;
      });
  CHECK(visited_shards == store->impl().t1().shard_count());
  CHECK(count_inversions(store) < store->impl().t1().shard_count());

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
  constexpr size_t kKeys = 128;
  constexpr size_t kValBytes = 4096;
  auto store = std::make_unique<TinyConvergenceStore>(path, kTestCapacityBytes);

  // 1. Initial population: 128 keys with 4KB values (~512KB live data)
  for (size_t i = 0; i < kKeys; ++i) {
    REQUIRE(store->insert(make_key(i), pattern_value(kValBytes, i)));
  }
  store->checkpoint();

  // Enable background compaction during continuous churn
  enable();

  // 2. Continuous churn: multiple generations of updates creating dead segments
  constexpr size_t kGenerations = 4;
  for (size_t gen = 0; gen < kGenerations; ++gen) {
    for (size_t i = 0; i < kKeys; ++i) {
      REQUIRE(store->update(make_key(i), pattern_value(kValBytes, (gen + 1) * kKeys + i)));
    }
    // Punch backlog must stay bounded by the cap (4) plus processing latency margin
    CHECK(store->impl().punch_backlog_segments() <= 32);
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

TEST_CASE_FIXTURE(AutoCompactFixture, "64KB shuffled bulk load bounds T2 write amplification and reclaims segments") {
  if (vmemkv_test::punch_capable_root().empty()) {
    MESSAGE("skipped: no non-tmpfs temp root; hole punch unsupported");
    return;
  }
  const auto path =
      vmemkv_test::ScopedTempPath("vmemkv_compact_64k", vmemkv::remove_store_files, vmemkv_test::punch_capable_root());
  constexpr size_t kKeys = 280;
  constexpr size_t kValBytes = 64ULL * 1024ULL;
  auto store = make_store(path);
  const auto order = shuffled_order(kKeys);
  store->bulk_load(
      kKeys,
      [&](size_t i) { return make_key(order[i]); },
      [&](size_t i) { return pattern_value(kValBytes, order[i]); });
  store->impl().t1().drain_maintenance();
  REQUIRE(store->impl().t1().shard_count() > 1);
  REQUIRE(inversion_rate(store) > 0.3);

  const uint64_t used_after_load = store->impl().t2().tail_offset();
  REQUIRE(used_after_load > 2 * vmemkv::T2Config::kSegmentBytes);
  store->checkpoint();

  drain_compaction(store, *this);
  CHECK(store->get_statistics().t2_dirty_shard_count == 0);
  // Dirty shards compact independently in FIFO order, so only shard boundaries can invert.
  CHECK(count_inversions(store) < store->impl().t1().shard_count());

  const uint64_t used_after_compact = store->impl().t2().tail_offset();
  const uint64_t relocated_bytes = used_after_compact - used_after_load;
  CHECK(relocated_bytes <= (used_after_load * 11) / 10);

  const uint64_t blocks_before_punch = file_blocks(t2_data_path(path));
  store->checkpoint();
  store->checkpoint();
  wait_for([&] { return store->impl().punch_backlog_segments() == 0; });

  const uint64_t blocks_after_punch = file_blocks(t2_data_path(path));
  CHECK(blocks_after_punch < blocks_before_punch);
  CHECK(blocks_after_punch * 512 <= (used_after_load * 3) / 2);

  // Growing-value updates (out-of-place appends) across 75% of keys must trigger
  // garbage-segment evacuation and hole punching without unbounded T2 growth.
  constexpr size_t kGrownValBytes = kValBytes + 512;
  const uint64_t used_before_updates = store->impl().t2().tail_offset();
  for (size_t i = 0; i < kKeys; ++i) {
    if (i % 4 != 0) {
      REQUIRE(store->update(make_key(i), pattern_value(kGrownValBytes, kKeys + i)));
    }
  }
  drain_compaction(store, *this);
  const uint64_t used_after_updates = store->impl().t2().tail_offset();
  CHECK((used_after_updates - used_before_updates) <= (used_after_load * 3) / 2);

  // This checkpoint can advance read_only_offset across dirty segments updated in Phase 2,
  // exposing one final frozen garbage evacuation before the two-checkpoint epoch gate runs.
  store->checkpoint();
  wait_for_compaction(*store);
  store->checkpoint();
  wait_for_compaction(*store);
  store->checkpoint();
  wait_for([&] { return store->impl().punch_backlog_segments() == 0; });
  CHECK(file_blocks(t2_data_path(path)) * 512 <= used_after_load * 2);

  for (size_t i = 0; i < kKeys; i += 31) {
    const auto got = vmemkv_test::get_optional_bytes(store, make_key(i));
    REQUIRE(got.has_value());
    const size_t expected_len = (i % 4 != 0) ? kGrownValBytes : kValBytes;
    const size_t expected_idx = (i % 4 != 0) ? (kKeys + i) : i;
    CHECK(vmemkv_test::span_to_string(vmemkv_test::as_span(*got)) == pattern_value(expected_len, expected_idx));
  }
}
