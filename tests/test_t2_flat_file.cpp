// test_t2_flat_file.cpp — T2FlatFile in isolation (no VMemKVImpl/T1Index/WAL):
// append round-tripping and in-place update_value_at.

#include <doctest/doctest.h>

#include <filesystem>
#include <span>
#include <string>
#include <vector>
#include <vmemkv/api/serializer.hpp>     // codec parity check below
#include <vmemkv/api/store_adapter.hpp>  // needed by vmemkv_impl.hpp below (get_impl()'s inline-value path)
#include <vmemkv/t2_flat_file/t2_flat_file.hpp>
#include <vmemkv/vmemkv_impl.hpp>  // for the free function vmemkv::read_t2_record_seqlock() only

#include "test_support.hpp"

namespace {

using vmemkv_test::as_span;
using vmemkv_test::bytes_of;
using vmemkv_test::span_to_string;

constexpr uint64_t kTestCapacityBytes = 4ULL * 1024 * 1024;

struct TempT2File {
  vmemkv_test::ScopedTempPath path_holder{"vmemkv_t2_test", vmemkv::remove_quiet};
  vmemkv::T2FlatFile file{path_holder.get(), kTestCapacityBytes};
};

}  // namespace

TEST_CASE("T2FlatFile: append_default then at() round-trips key/value") {
  TempT2File t2;
  auto mem = t2.file.get_memory();
  const uint64_t offset = t2.file.append_default(mem, as_span(bytes_of("hello")), as_span(bytes_of("world")));

  const auto record = t2.file.at(offset, mem);
  CHECK(span_to_string(record.key) == "hello");
  CHECK(span_to_string(record.value) == "world");
}

TEST_CASE("T2FlatFile: append_default keeps one-page records within a single page") {
  TempT2File t2;
  auto mem = t2.file.get_memory();
  constexpr size_t kRecords = 64;
  constexpr size_t kValueBytes = 1024;
  std::vector<uint64_t> offsets;
  offsets.reserve(kRecords);
  for (size_t i = 0; i < kRecords; ++i) {
    const std::string key = "k" + std::to_string(i);
    const std::string value(kValueBytes, static_cast<char>('a' + (i % 26)));
    offsets.push_back(t2.file.append_default(mem, as_span(bytes_of(key)), as_span(bytes_of(value))));
  }
  size_t page_aligned = 0;
  for (size_t i = 0; i < kRecords; ++i) {
    const std::string key = "k" + std::to_string(i);
    const uint64_t aligned_len = vmemkv::record_aligned_len(sizeof(ValueRecordHeader), key.size(), kValueBytes);
    REQUIRE(aligned_len <= 4096);
    CHECK((offsets[i] % 4096) + aligned_len <= 4096);
    if (offsets[i] % 4096 == 0 && i > 0) {
      ++page_aligned;
    }
    const std::string expected(kValueBytes, static_cast<char>('a' + (i % 26)));
    CHECK(span_to_string(t2.file.at(offsets[i], mem).value) == expected);
  }
  // Crossing page boundaries must actually exercise the skip path.
  CHECK(page_aligned > 0);
}

TEST_CASE("T2FlatFile: append_default packs multi-page records contiguously") {
  TempT2File t2;
  auto mem = t2.file.get_memory();
  // A misaligned frontier is where a containment gap would be inserted; multi-page records
  // pack at the exact frontier instead.
  const uint64_t pad_offset = t2.file.append_default(mem, as_span(bytes_of("pad")), as_span(bytes_of("x")));
  const uint64_t pad_len =
      vmemkv::record_aligned_len(sizeof(ValueRecordHeader), std::string("pad").size(), std::string("x").size());
  const std::string big_value(65536, 'z');
  const uint64_t offset = t2.file.append_default(mem, as_span(bytes_of("big")), as_span(bytes_of(big_value)));
  CHECK(pad_offset == 0);
  CHECK(offset == pad_len);
  CHECK(span_to_string(t2.file.at(offset, mem).value) == big_value);
}

TEST_CASE("T2FlatFile: append_default over capacity throws without consuming bytes") {
  TempT2File t2;
  auto mem = t2.file.get_memory();
  const std::string huge_value(5 * 1024 * 1024, 'q');
  CHECK_THROWS(t2.file.append_default(mem, as_span(bytes_of("huge")), as_span(bytes_of(huge_value))));
  CHECK(t2.file.bytes_used() == 0);
}

// scan_bounds_clean(): clean bounds (no 0x00 in the first 16 bytes, length at most 16) trust
// the sorted walk's prefix containment; dirty bounds keep the per-entry recheck.
TEST_CASE("scan_bounds_clean trusts clean bounds, keeps dirty ones exact") {
  CHECK(vmemkv::scan_bounds_clean(as_span(bytes_of("k00000000")), as_span(bytes_of("k00000099"))));
  // NOTE: std::string("ab\0") would terminate at the null; spell the length out.
  CHECK_FALSE(vmemkv::scan_bounds_clean(as_span(bytes_of(std::string("ab\0", 3))), as_span(bytes_of("ac"))));
  CHECK_FALSE(vmemkv::scan_bounds_clean(as_span(bytes_of("k00000000")), as_span(bytes_of("k000009990123456789"))));
}

TEST_CASE("T2FlatFile: update_value_at overwrites in place when the new value fits alloc_len") {
  TempT2File t2;
  auto mem = t2.file.get_memory();
  const uint64_t offset = t2.file.append_default(mem, as_span(bytes_of("k")), as_span(bytes_of("0123456789")));

  CHECK(vmemkv::T2FlatFile::update_value_at(offset, as_span(bytes_of("abc")), mem));

  CHECK(span_to_string(t2.file.at(offset, mem).value) == "abc");
}

TEST_CASE("T2FlatFile: update_value_at fails when the new value exceeds alloc_len") {
  TempT2File t2;
  auto mem = t2.file.get_memory();
  const uint64_t offset = t2.file.append_default(mem, as_span(bytes_of("k")), as_span(bytes_of("abc")));

  CHECK_FALSE(vmemkv::T2FlatFile::update_value_at(offset, as_span(bytes_of("0123456789")), mem));
}

// read_t2_record_seqlock() re-invokes AtFunc every retry, so it always sizes spans from the
// current record -- even right after a shrinking update_value_at().
TEST_CASE(
    "T2FlatFile: read_t2_record_seqlock always observes the current value, even immediately "
    "after a shrinking update_value_at() (regression)") {
  TempT2File t2;
  auto mem = t2.file.get_memory();
  const std::string original_value(64, 'A');  // alloc_len becomes 64.
  const uint64_t offset = t2.file.append_default(mem, as_span(bytes_of("k")), as_span(bytes_of(original_value)));

  const std::string shrunk_value = "new";  // Fits within alloc_len=64; update_value_at() succeeds.
  REQUIRE(vmemkv::T2FlatFile::update_value_at(offset, as_span(bytes_of(shrunk_value)), mem));

  // AtFunc reads the record fresh, as of *right now*.
  std::vector<std::byte> copied;
  vmemkv::read_t2_record_seqlock([&]() -> T2RecordView { return t2.file.at(offset, mem); },
                                 [&](const T2RecordView &record) -> bool {
                                   copied.assign(record.value.begin(), record.value.end());
                                   return true;
                                 });
  CHECK(copied.size() == shrunk_value.size());
  CHECK(span_to_string(std::span<const std::byte>(copied.data(), copied.size())) == shrunk_value);
}

TEST_CASE("codec: public vmemkv.hpp and internal serializer agree byte-for-byte") {
  CHECK_NOTHROW(kvs_detail::check_public_internal_codec_parity());
}
