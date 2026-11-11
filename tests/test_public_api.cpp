// test_public_api.cpp — the installed public surface and nothing else.
//
// This TU intentionally stays self-contained (standard library + installed
// headers only, no tests/test_temp.hpp): it proves a downstream user gets the
// full API from the public headers, exercising VMemKVStore directly instead of
// the internal store configurations the other TUs cover.
#include <doctest/doctest.h>
#include <unistd.h>

#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <vmemkv/vmemkv.hpp>

namespace {

template <typename Store>
concept HasCompact = requires(Store &store) { store.compact(); };

static_assert(!HasCompact<vmemkv::VMemKVStore>);

// Store dirs resolve like tests/test_temp.hpp test_temp_root() (VMEMKV_TEST_TMPDIR override,
// then the memory-backed /dev/shm, then the platform temp dir) while remaining standard-library
// only: this TU must not include the internal test helpers.
auto test_temp_root() -> std::filesystem::path {
  if (const char *env = std::getenv("VMEMKV_TEST_TMPDIR"); env != nullptr && *env != '\0') {
    return {env};
  }
  std::error_code ec;
  if (std::filesystem::is_directory("/dev/shm", ec)) {
    return {"/dev/shm"};
  }
  return std::filesystem::temp_directory_path();
}

auto reserve_store_dir() -> std::filesystem::path {
  static std::atomic<uint64_t> counter{0};
  const auto dir = test_temp_root() / ("vmemkv_public_" + std::to_string(static_cast<long>(::getpid())) + "_" +
                                       std::to_string(counter.fetch_add(1)));
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  return dir;
}

struct ScopedStoreDir {
  std::filesystem::path dir = reserve_store_dir();
  ~ScopedStoreDir() {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
  }
};

auto make_store(const std::filesystem::path &dir) -> vmemkv::VMemKVStore {
  return vmemkv::VMemKVStore(dir / "test.bin", 8ULL << 20);
}

auto span_to_string(std::span<const std::byte> value) -> std::string {
  return {reinterpret_cast<const char *>(value.data()), value.size()};
}

}  // namespace

TEST_CASE("public API: insert/get/update/remove round-trip") {
  ScopedStoreDir guard;
  auto store = make_store(guard.dir);

  CHECK(store.insert("hello", "world"));
  CHECK_FALSE(store.insert("hello", "again"));

  std::string seen;
  CHECK(store.get("hello", [&](std::span<const std::byte> value) { seen = span_to_string(value); }));
  CHECK(seen == "world");

  CHECK(store.update("hello", "new"));
  CHECK(store.get("hello", [&](std::span<const std::byte> value) { seen = span_to_string(value); }));
  CHECK(seen == "new");
  CHECK_FALSE(store.update("absent", "x"));

  CHECK(store.remove("hello"));
  CHECK_FALSE(store.remove("hello"));
  CHECK_FALSE(store.get("hello", [](std::span<const std::byte>) {}));
}

TEST_CASE("public API: integral keys order lexicographically, scan is half-open and ordered") {
  ScopedStoreDir guard;
  auto store = make_store(guard.dir);
  for (uint64_t key : {3ULL, 1ULL, 2ULL}) {
    REQUIRE(store.insert(key, key * 10));
  }

  const auto decode = [](std::span<const std::byte> key, std::span<const std::byte> value) {
    // Keys travel big-endian; swap back to compare numerically.
    uint64_t wire_order = 0;
    std::memcpy(&wire_order, key.data(), sizeof(wire_order));
    const uint64_t decoded_key = std::byteswap(wire_order);
    uint64_t decoded_value = 0;
    std::memcpy(&decoded_value, value.data(), sizeof(decoded_value));
    CHECK(decoded_value == decoded_key * 10);
    return decoded_key;
  };
  // The upper bound itself is excluded: [1, 3) holds {1, 2}.
  std::vector<uint64_t> scanned;
  const size_t count = store.scan(1ULL, 3ULL, [&](std::span<const std::byte> key, std::span<const std::byte> value) {
    scanned.push_back(decode(key, value));
  });
  CHECK(count == 2U);
  CHECK(scanned == std::vector<uint64_t>{1ULL, 2ULL});

  // Extending the upper bound by one recovers the tail key.
  std::vector<uint64_t> rescanned;
  const size_t recount = store.scan(1ULL, 4ULL, [&](std::span<const std::byte> key, std::span<const std::byte> value) {
    rescanned.push_back(decode(key, value));
  });
  CHECK(recount == 3U);
  CHECK(rescanned == std::vector<uint64_t>{1ULL, 2ULL, 3ULL});

  std::vector<uint64_t> stopped;
  const size_t stop_count =
      store.scan(1ULL, 4ULL, [&](std::span<const std::byte> key, std::span<const std::byte> value) {
        stopped.push_back(decode(key, value));
        return vmemkv::ScanControl::Stop;
      });
  CHECK(stop_count == 1U);
  CHECK(stopped == std::vector<uint64_t>{1ULL});
}

namespace public_api_adl {

struct Point {
  int x;
  int y;
};

inline auto serialize_kvs(const Point &point) noexcept -> std::array<std::byte, 8> {
  std::array<std::byte, 8> out{};
  std::memcpy(out.data(), &point.x, sizeof(point.x));
  std::memcpy(out.data() + sizeof(point.x), &point.y, sizeof(point.y));
  return out;
}

}  // namespace public_api_adl

TEST_CASE("public API: value spellings round-trip (integral, bytes, ADL type)") {
  ScopedStoreDir guard;
  auto store = make_store(guard.dir);

  REQUIRE(store.insert("u64", 0x0102030405060708ULL));
  uint64_t decoded = 0;
  REQUIRE(store.get("u64",
                    [&](std::span<const std::byte> value) { std::memcpy(&decoded, value.data(), sizeof(decoded)); }));
  CHECK(decoded == 0x0102030405060708ULL);

  const std::vector<std::byte> raw{std::byte{1}, std::byte{2}, std::byte{3}};
  REQUIRE(store.insert("raw", raw));
  std::vector<std::byte> raw_back;
  REQUIRE(store.get("raw", [&](std::span<const std::byte> value) { raw_back.assign(value.begin(), value.end()); }));
  CHECK(raw_back == raw);

  const public_api_adl::Point point{7, 9};
  REQUIRE(store.insert("point", point));
  public_api_adl::Point point_back{0, 0};
  REQUIRE(store.get("point",
                    [&](std::span<const std::byte> value) { std::memcpy(&point_back, value.data(), value.size()); }));
  CHECK(point_back.x == 7);
  CHECK(point_back.y == 9);
}

TEST_CASE("public API: bulk_load round-trips through get and scan") {
  ScopedStoreDir guard;
  auto store = make_store(guard.dir);
  store.bulk_load(
      4,
      [](std::size_t i) { return "bulk" + std::to_string(i); },
      [](std::size_t i) { return "v" + std::to_string(i); });
  for (std::size_t i = 0; i < 4; ++i) {
    const auto key = "bulk" + std::to_string(i);
    const auto expected = "v" + std::to_string(i);
    std::string seen;
    REQUIRE(store.get(key, [&](std::span<const std::byte> value) { seen = span_to_string(value); }));
    CHECK(seen == expected);
  }
  size_t count = 0;
  std::ignore = store.scan("bulk0", "bulk9", [&](std::span<const std::byte>, std::span<const std::byte>) { ++count; });
  CHECK(count == 4U);
}

TEST_CASE("public API: maintenance and statistics") {
  ScopedStoreDir guard;
  auto store = make_store(guard.dir);
  REQUIRE(store.insert("a", "1"));

  const auto before = store.get_statistics();
  store.reorganize();
  store.checkpoint();
  const auto after = store.get_statistics();
  CHECK(after.checkpoint_count == before.checkpoint_count + 1);

  std::string seen;
  CHECK(store.get("a", [&](std::span<const std::byte> value) { seen = span_to_string(value); }));
  CHECK(seen == "1");
}

TEST_CASE("public API: store handle is move-only") {
  ScopedStoreDir guard;
  auto store = make_store(guard.dir);
  REQUIRE(store.insert("k", "v"));

  auto moved = std::move(store);
  std::string seen;
  CHECK(moved.get("k", [&](std::span<const std::byte> value) { seen = span_to_string(value); }));
  CHECK(seen == "v");
}

TEST_CASE("public API: update verifies full key under prefix collision in base and tail") {
  ScopedStoreDir guard;
  auto store = make_store(guard.dir);

  // Two keys sharing the first 16 bytes (> kPrefixBytes)
  const std::string key_prefix_alice = "prefix_matching_16b_alice_001";
  const std::string key_prefix_bob = "prefix_matching_16b_bob____002";
  const std::string val_initial = "initial_value_payload_100";
  const std::string val_updated = "updated_value_payload_200";

  // 1. Insert alice only into mutable tail
  REQUIRE(store.insert(key_prefix_alice, val_initial));

  // Mutable tail: updating nonexistent bob must fail (not fall through to insert)
  CHECK_FALSE(store.update(key_prefix_bob, val_updated));
  CHECK_FALSE(store.get(key_prefix_bob, [](std::span<const std::byte>) {}));

  // Updating existing alice must succeed
  CHECK(store.update(key_prefix_alice, val_updated));
  std::string seen;
  CHECK(store.get(key_prefix_alice, [&](std::span<const std::byte> val) { seen = span_to_string(val); }));
  CHECK(seen == val_updated);

  // 2. Checkpoint moves records into immutable base region (offset < base_boundary)
  store.checkpoint();

  // Base region: updating nonexistent bob must still fail (not fall through to insert)
  CHECK_FALSE(store.update(key_prefix_bob, "should_not_exist"));
  CHECK_FALSE(store.get(key_prefix_bob, [](std::span<const std::byte>) {}));

  // Updating existing alice in base region redirects out-of-place and succeeds
  const std::string val_base_updated = "base_region_updated_value";
  CHECK(store.update(key_prefix_alice, val_base_updated));
  CHECK(store.get(key_prefix_alice, [&](std::span<const std::byte> val) { seen = span_to_string(val); }));
  CHECK(seen == val_base_updated);
}
