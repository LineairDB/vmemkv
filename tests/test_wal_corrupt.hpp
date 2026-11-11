// test_wal_corrupt.hpp — WAL corruption helpers for tests (byte flips, torn headers).
#pragma once

#include <doctest/doctest.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <vmemkv/wal/wal.hpp>

namespace vmemkv_test {

// Flips a single byte at `offset` in the file at `path`.
inline void flip_byte_at(const std::filesystem::path &path, std::streamoff offset) {
  std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
  file.seekg(offset);
  char original = 0;
  file.read(&original, 1);
  const char flipped = static_cast<char>(~original);
  file.seekp(offset);
  file.write(&flipped, 1);
}

// Resolves the WAL's active segment for either a WAL identity path or a T2 store path.
inline auto resolve_active_wal_segment(const std::filesystem::path &path) -> std::filesystem::path {
  if (const auto direct = vmemkv::find_active_wal_segment(path); direct.has_value()) {
    return *direct;
  }
  const auto derived = vmemkv::find_active_wal_segment(vmemkv::derive_wal_path(path));
  REQUIRE(derived.has_value());
  return *derived;
}

// Appends raw bytes shorter than a full WAL record header to the active segment.
inline void append_torn_header(const std::filesystem::path &path) {
  const auto active = resolve_active_wal_segment(path);
  std::ofstream out(active, std::ios::binary | std::ios::app);
  constexpr std::array<char, 10> garbage{};
  out.write(garbage.data(), garbage.size());
}

}  // namespace vmemkv_test
