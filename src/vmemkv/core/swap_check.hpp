// swap_check.hpp -- Startup swap validation for larger-than-memory operation.
//
// Spill goes through OS virtual memory; without swap, an over-RAM workload meets the OOM
// killer instead of paging. Warns once on zero swap, throws on a VMEMKV_REQUIRE_SWAP_BYTES
// floor miss. Silent without /proc/meminfo (non-Linux).
#pragma once

#include <atomic>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vmemkv/core/env.hpp>

namespace vmemkv {

inline auto system_swap_total_bytes(const std::string &meminfo_path = "/proc/meminfo") -> std::optional<uint64_t> {
  return read_meminfo_bytes(meminfo_path, "SwapTotal:");
}

inline void validate_swap_for_ltm(const std::string &meminfo_path = "/proc/meminfo") {
  const std::optional<uint64_t> swap_total = system_swap_total_bytes(meminfo_path);
  if (!swap_total.has_value()) {
    return;
  }
  if (const std::string_view env = getenv_view("VMEMKV_REQUIRE_SWAP_BYTES"); !env.empty()) {
    const uint64_t required = getenv_uint64("VMEMKV_REQUIRE_SWAP_BYTES", 0);
    if (*swap_total < required) {
      throw std::runtime_error("vmemkv requires " + std::to_string(required) +
                               " swap bytes (VMEMKV_REQUIRE_SWAP_BYTES) but the host provides only " +
                               std::to_string(*swap_total));
    }
    return;
  }
  if (*swap_total == 0) {
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true, std::memory_order_relaxed)) {
      std::cerr << "[vmemkv] warning: host has no swap; larger-than-memory workloads risk the OOM "
                   "killer instead of paging. Set VMEMKV_REQUIRE_SWAP_BYTES to enforce a minimum."
                << '\n';
    }
  }
}

}  // namespace vmemkv
