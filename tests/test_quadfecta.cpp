// test_quadfecta.cpp -- Standalone unit tests for quadfecta::MappedReader (VMemKV-independent).

#include <doctest/doctest.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
#include <system_error>
#include <utility>
#include <vector>
#include <vmemkv/core/quadfecta.hpp>
#include <vmemkv/core/utils.hpp>

#include "test_support.hpp"

namespace {

auto make_pattern_bytes(size_t size) -> std::vector<std::byte> {
  std::vector<std::byte> payload(size);
  for (size_t i = 0; i < size; ++i) {
    payload[i] = static_cast<std::byte>((i * 31 + 7) & 0xffU);
  }
  return payload;
}

void write_binary_file(const std::filesystem::path &path, const std::vector<std::byte> &payload) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  REQUIRE(out.good());
  if (!payload.empty()) {
    out.write(reinterpret_cast<const char *>(payload.data()), static_cast<std::streamsize>(payload.size()));
    REQUIRE(out.good());
  }
}

}  // namespace

TEST_CASE("Quadfecta: open_readonly serves small and large reads across all policies") {
  const auto path = vmemkv_test::ScopedTempPath("quadfecta_all_policies", vmemkv::remove_quiet);
  const uint64_t page_size = quadfecta::MappedReader::page_size();
  const size_t total_size = static_cast<size_t>(page_size * 4);
  const auto payload = make_pattern_bytes(total_size);
  write_binary_file(path.get(), payload);

  auto reader = quadfecta::MappedReader::open_readonly(path.get());
  CHECK(reader.capacity() == total_size);
  CHECK(reader.random_mmap() != nullptr);
  CHECK(reader.sequential_mmap() != nullptr);
  CHECK(reader.normal_mmap() != nullptr);
  CHECK(reader.pread_fd() >= 0);

  std::vector<std::byte> scratch;
  const uint64_t large_len = page_size + 256;
  for (quadfecta::Policy policy : {quadfecta::Policy::Adaptive,
                                   quadfecta::Policy::RandomOnly,
                                   quadfecta::Policy::SeqOnly,
                                   quadfecta::Policy::NormalOnly,
                                   quadfecta::Policy::PreadOnly}) {
    for (quadfecta::Access access : {quadfecta::Access::Random, quadfecta::Access::Sequential}) {
      const auto small_view = reader.read(128, 64, access, &scratch, policy);
      REQUIRE(small_view.has_value());
      CHECK(std::memcmp(small_view->data(), payload.data() + 128, 64) == 0);

      const auto large_view = reader.read(page_size, large_len, access, &scratch, policy);
      REQUIRE(large_view.has_value());
      CHECK(std::memcmp(large_view->data(), payload.data() + page_size, large_len) == 0);
    }
  }

  // Exact-page boundary stays on mmap; >page_size routes large random to pread and requires scratch.
  CHECK(reader.read(0, page_size, quadfecta::Access::Random, nullptr, quadfecta::Policy::Adaptive).has_value());
  CHECK_FALSE(
      reader.read(page_size, large_len, quadfecta::Access::Random, nullptr, quadfecta::Policy::Adaptive).has_value());
  // Policy::PreadOnly uses an internal thread_local scratch buffer when caller supplies nullptr.
  const auto tl_pread = reader.read(64, 32, quadfecta::Access::Random, nullptr, quadfecta::Policy::PreadOnly);
  REQUIRE(tl_pread.has_value());
  CHECK(std::memcmp(tl_pread->data(), payload.data() + 64, 32) == 0);

  reader.prefetch(0, page_size);
  reader.prefetch(128, 64);  // unaligned offset rounds down to page boundary without error
  reader.prefetch(total_size, 64);
  reader.prefetch(0, 0);

  // Zero-byte fixed reads are consistently empty spans across policies within [0, capacity()].
  for (quadfecta::Policy policy : {quadfecta::Policy::Adaptive, quadfecta::Policy::PreadOnly}) {
    const auto zero_read = reader.read(0, 0, quadfecta::Access::Random, &scratch, policy);
    REQUIRE(zero_read.has_value());
    CHECK(zero_read->empty());
  }
  const auto zero_tail = reader.read(total_size, 0, quadfecta::Access::Random, &scratch);
  REQUIRE(zero_tail.has_value());
  CHECK(zero_tail->empty());

  // Repeated open_secondary_handles() is idempotent and preserves existing handles without leaking.
  const std::byte *orig_seq = reader.sequential_mmap();
  const std::byte *orig_norm = reader.normal_mmap();
  const int orig_fd = reader.pread_fd();
  const int dup_fd = ::open(path.get().c_str(), O_RDONLY | O_CLOEXEC);
  REQUIRE(dup_fd >= 0);
  reader.open_secondary_handles(dup_fd);
  ::close(dup_fd);
  CHECK(reader.sequential_mmap() == orig_seq);
  CHECK(reader.normal_mmap() == orig_norm);
  CHECK(reader.pread_fd() == orig_fd);
}

TEST_CASE("Quadfecta: view_at_most clamps by valid_end and capacity") {
  const auto path = vmemkv_test::ScopedTempPath("quadfecta_valid_end", vmemkv::remove_quiet);
  const uint64_t page_size = quadfecta::MappedReader::page_size();
  const auto payload = make_pattern_bytes(static_cast<size_t>(page_size * 2));
  write_binary_file(path.get(), payload);

  auto reader = quadfecta::MappedReader::open_readonly(path.get());
  std::vector<std::byte> scratch;

  const uint64_t valid_end = 512;
  CHECK_FALSE(reader.view_at_most(valid_end, 64, valid_end, quadfecta::Access::Random, &scratch).has_value());
  const auto mmap_tail = reader.view_at_most(400, 256, valid_end, quadfecta::Access::Random, &scratch);
  REQUIRE(mmap_tail.has_value());
  CHECK(mmap_tail->size() == valid_end - 400);
  CHECK(std::memcmp(mmap_tail->data(), payload.data() + 400, mmap_tail->size()) == 0);

  const uint64_t overshoot_end = payload.size() + 1024;
  const auto clamped_pread =
      reader.view_at_most(payload.size() - 100, page_size + 512, overshoot_end, quadfecta::Access::Random, &scratch);
  REQUIRE(clamped_pread.has_value());
  CHECK(clamped_pread->size() == 100);
  CHECK(std::memcmp(clamped_pread->data(), payload.data() + payload.size() - 100, 100) == 0);

  CHECK_FALSE(reader.read(payload.size(), 1, quadfecta::Access::Random, &scratch).has_value());
  CHECK_FALSE(reader.read(16, payload.size(), quadfecta::Access::Random, &scratch).has_value());
}

TEST_CASE("Quadfecta: open_readonly handles missing and empty files and primary-only fallback") {
  // Non-regular paths (directories) are rejected by open_readonly().
  const auto dir = vmemkv_test::reserve_unique_temp_dir("quadfecta_dir");
  CHECK_THROWS_AS((void)quadfecta::MappedReader::open_readonly(dir), std::system_error);

  const auto missing = vmemkv_test::reserve_unique_temp_path("quadfecta_missing");
  CHECK_THROWS_AS((void)quadfecta::MappedReader::open_readonly(missing), std::system_error);

  const auto empty_path = vmemkv_test::ScopedTempPath("quadfecta_empty", vmemkv::remove_quiet);
  write_binary_file(empty_path.get(), {});
  auto empty_reader = quadfecta::MappedReader::open_readonly(empty_path.get());
  CHECK(empty_reader.capacity() == 0);
  CHECK(empty_reader.random_mmap() == nullptr);
  CHECK(empty_reader.sequential_mmap() == nullptr);
  CHECK(empty_reader.normal_mmap() == nullptr);
  std::vector<std::byte> scratch;
  CHECK_FALSE(empty_reader.read(0, 1, quadfecta::Access::Random, &scratch).has_value());
  empty_reader.prefetch(0, 64);

  // Primary-only MappedReader (no secondary handles): sequential/normal fall back to random_mmap; pread returns
  // nullopt.
  const uint64_t cap = quadfecta::MappedReader::page_size() * 2;
  void *raw = ::mmap(nullptr, static_cast<size_t>(cap), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  REQUIRE(raw != MAP_FAILED);
  auto *bytes = static_cast<std::byte *>(raw);
  for (uint64_t i = 0; i < cap; ++i) {
    bytes[i] = static_cast<std::byte>(i & 0xffU);
  }

  quadfecta::MappedReader primary_only(bytes, cap);
  CHECK(primary_only.mapping_for(quadfecta::Access::Sequential, true, quadfecta::Policy::Adaptive) == bytes);
  CHECK(primary_only.mapping_for(quadfecta::Access::Sequential, false, quadfecta::Policy::Adaptive) == bytes);
  CHECK(primary_only.mapping_for(quadfecta::Access::Random, true, quadfecta::Policy::SeqOnly) == bytes);
  CHECK(primary_only.mapping_for(quadfecta::Access::Random, true, quadfecta::Policy::NormalOnly) == bytes);
  CHECK_FALSE(primary_only.read(0, 32, quadfecta::Access::Random, &scratch, quadfecta::Policy::PreadOnly).has_value());

  // Move semantics transfer ownership cleanly and reset source handles.
  quadfecta::MappedReader moved(std::move(primary_only));
  CHECK(primary_only.capacity() == 0);
  CHECK(primary_only.random_mmap() == nullptr);
  CHECK(moved.capacity() == cap);
  CHECK(moved.random_mmap() == bytes);

  quadfecta::MappedReader reassigned;
  reassigned = std::move(moved);
  CHECK(moved.capacity() == 0);
  CHECK(reassigned.capacity() == cap);
  const auto span = reassigned.read(17, 16, quadfecta::Access::Sequential, &scratch);
  REQUIRE(span.has_value());
  CHECK(std::memcmp(span->data(), bytes + 17, 16) == 0);
}
