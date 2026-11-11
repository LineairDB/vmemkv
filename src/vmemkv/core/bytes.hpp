// bytes.hpp - Shared byte-span ordering and comparison helpers.
#pragma once

#include <algorithm>
#include <compare>
#include <cstddef>
#include <cstring>
#include <span>

namespace vmemkv {

// Byte-wise key ordering: memcmp over the shared prefix, shorter key first on ties.
inline auto bytes_compare_3way(std::span<const std::byte> first, std::span<const std::byte> second) noexcept -> int {
  const std::size_t min_len = std::min(first.size(), second.size());
  const int cmp = min_len == 0 ? 0 : std::memcmp(first.data(), second.data(), min_len);
  if (cmp != 0) {
    return cmp;
  }
  if (first.size() == second.size()) {
    return 0;
  }
  return first.size() < second.size() ? -1 : 1;
}

inline auto bytes_equal(std::span<const std::byte> first, std::span<const std::byte> second) noexcept -> bool {
  return first.size() == second.size() && bytes_compare_3way(first, second) == 0;
}

inline auto bytes_in_range(std::span<const std::byte> key,
                           std::span<const std::byte> lower,
                           std::span<const std::byte> upper) noexcept -> bool {
  return bytes_compare_3way(key, lower) >= 0 && bytes_compare_3way(key, upper) < 0;
}

}  // namespace vmemkv
