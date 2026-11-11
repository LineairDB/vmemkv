// env.hpp - Environment-variable and small-file numeric helpers.
#pragma once

#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>

namespace vmemkv {

// Value of `name`, or empty when unset or empty.
inline auto getenv_view(const char *name) noexcept -> std::string_view {
  if (const char *env = std::getenv(name); env != nullptr && *env != '\0') {
    return {env};
  }
  return {};
}

// Decimal value of `name`, or `dflt` when unset, empty, or unparsable.
inline auto getenv_uint64(const char *name, uint64_t dflt) noexcept -> uint64_t {
  const std::string_view text = getenv_view(name);
  if (text.empty()) {
    return dflt;
  }
  uint64_t value = 0;
  const auto *first = text.data();
  const auto *last = text.data() + text.size();
  if (std::from_chars(first, last, value).ec != std::errc{}) {
    return dflt;
  }
  return value;
}

// Reads one decimal integer from a sysfs/cgroup-style file. False when the file
// is missing or holds non-numeric content such as memory.high's "max".
inline auto read_uint_file(const std::string &path, uint64_t &out) -> bool {
  std::ifstream file(path);
  if (!file.is_open()) {
    return false;
  }
  uint64_t value = 0;
  file >> value;
  if (file.fail()) {
    return false;
  }
  out = value;
  return true;
}

// Value (in bytes) of `key` in a meminfo-style file ("Key: <n> kB"), or nullopt when
// missing/unparsable. Shared by swap_check.hpp (/proc/meminfo SwapTotal:) and the benchmark
// harness (/proc/meminfo MemTotal:).
inline auto read_meminfo_bytes(const std::string &meminfo_path, std::string_view key) -> std::optional<uint64_t> {
  std::ifstream meminfo(meminfo_path);
  if (!meminfo.is_open()) {
    return std::nullopt;
  }
  std::string word;
  std::size_t value_kb = 0;
  std::string unit;
  while (meminfo >> word >> value_kb >> unit) {
    if (word == key) {
      constexpr uint64_t kBytesPerKibibyte = 1024ULL;  // meminfo reports kB.
      return value_kb * kBytesPerKibibyte;
    }
    meminfo.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
  return std::nullopt;
}

}  // namespace vmemkv
