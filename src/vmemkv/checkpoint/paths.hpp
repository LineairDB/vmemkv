// paths.hpp - Sibling-path derivation for one store's files.
#pragma once

#include <filesystem>
#include <string_view>
#include <vmemkv/core/env.hpp>

namespace vmemkv {

// Sibling manifest path for a given T2 flat-file path.
inline auto derive_manifest_path(const std::filesystem::path &t2_path) -> std::filesystem::path {
  return {t2_path.string() + ".manifest"};
}

// T1 checkpoint / T2 checkpoint file paths. Each is a single path reused across
// every checkpoint this store ever commits: the T1 file is fully rewritten each
// cycle via temp+rename, and the T2 file is the store's one persistent data file,
// appended to in place. Neither is trusted by a reader until the manifest names the
// generation that last wrote them.
inline auto derive_t1_chk_path(const std::filesystem::path &t2_path) -> std::filesystem::path {
  return {t2_path.string() + ".t1chk"};
}
inline auto derive_t2_chk_path(const std::filesystem::path &t2_path) -> std::filesystem::path {
  return {t2_path.string() + ".t2chk"};
}

// Sibling WAL identity path (path + ".wal"); the on-disk segments are derive_wal_segment_path()
// below. `wal_dir` redirects segments to a second device (T2 bulk writeback and WAL fsyncs on
// one device contend under memory pressure); defaults to VMEMKV_WAL_DIR, unset for
// single-device layout. Injectable for tests.
inline auto derive_wal_path(const std::filesystem::path &t2_path,
                            std::string_view wal_dir = {}) -> std::filesystem::path {
  const std::string_view dir = wal_dir.empty() ? getenv_view("VMEMKV_WAL_DIR") : wal_dir;
  if (!dir.empty()) {
    return {std::filesystem::path(dir) / (t2_path.filename().string() + ".wal")};
  }
  return {t2_path.string() + ".wal"};
}

// One segment's path: <wal_path>.<generation>, oldest first.
inline auto derive_wal_segment_path(const std::filesystem::path &wal_path,
                                    uint64_t generation) -> std::filesystem::path {
  return {wal_path.string() + "." + std::to_string(generation)};
}

}  // namespace vmemkv
