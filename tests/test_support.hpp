// test_support.hpp — Umbrella for the split test helpers (temp/bytes/WAL-corrupt).
//
// New code may include the narrower header it needs directly (test_temp.hpp,
// test_bytes.hpp, test_wal_corrupt.hpp); existing TUs keep including this file.
#pragma once

#include <cstdint>
#include <utility>
#include <vmemkv/checkpoint/checkpoint.hpp>
#include <vmemkv/config.hpp>

#include "test_bytes.hpp"
#include "test_temp.hpp"
#include "test_wal_corrupt.hpp"

namespace vmemkv_test {

// Config override providing a small T1 append-region capacity (2^Log2 entries).
template <size_t Log2>
struct TinyAppendConfig : vmemkv::Config<> {
  static constexpr size_t kAppendCapacityLog2 = Log2;
  static constexpr size_t kAppendCapacityEntries = size_t{1} << Log2;
};

// Adapts a per-entry function to T1Index::reorganize()'s OffsetMapper contract.
template <typename PerEntryFn>
auto per_entry_offset_mapper(PerEntryFn fn) {
  return [fn = std::move(fn)](auto merged) {
    for (auto &entry : merged) {
      entry.payload_bits = fn(entry.payload_bits, entry.hash);
    }
  };
}

// Identity offset mapper for reorganize() calls that relocate nothing.
inline auto identity_offset_mapper() {
  return per_entry_offset_mapper([](uint64_t payload, uint64_t /*hash*/) { return payload; });
}

}  // namespace vmemkv_test
