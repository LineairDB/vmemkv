// test_temp.hpp — Temp-file/dir helpers for tests (unique paths, RAII guard).
#pragma once

#include <linux/magic.h>
#include <sys/vfs.h>
#include <unistd.h>

#include <atomic>
#include <filesystem>
#include <functional>
#include <string>
#include <string_view>
#include <vmemkv/core/env.hpp>
#include <vmemkv/wal/wal.hpp>

namespace vmemkv_test {

// ScopedTempPath cleanup for WAL-only test dirs (replaces a lambda pasted per call site).
inline void cleanup_wal_segments(const std::filesystem::path &path) { vmemkv::remove_wal_segments(path); }

// Base directory for test-created T2/WAL/checkpoint files. VMEMKV_TEST_TMPDIR overrides;
// otherwise the default is the memory-backed /dev/shm when present, so store-heavy tests are
// not wall-clock-bound by disk fdatasync latency, and the platform temp dir otherwise.
inline auto test_temp_root() -> const std::filesystem::path & {
  static const std::filesystem::path root = [] {
    if (const std::string_view env = vmemkv::getenv_view("VMEMKV_TEST_TMPDIR"); !env.empty()) {
      return std::filesystem::path(std::string(env));
    }
    std::error_code ignored;
    if (std::filesystem::is_directory("/dev/shm", ignored)) {
      return std::filesystem::path("/dev/shm");
    }
    return std::filesystem::temp_directory_path();
  }();
  return root;
}

// Root for tests whose assertions require hole-punch (tmpfs cannot punch; see
// src/vmemkv/impl/compaction.hpp punch_supported()). VMEMKV_TEST_TMPDIR is honored when it
// names a non-tmpfs filesystem. Empty when only tmpfs roots exist: such tests must skip.
inline auto punch_capable_root() -> const std::filesystem::path & {
  static const std::filesystem::path root = [] {
    const auto is_tmpfs = [](const std::filesystem::path &candidate) {
      struct statfs sfs {};
      return ::statfs(candidate.c_str(), &sfs) == 0 && sfs.f_type == TMPFS_MAGIC;
    };
    if (const std::string_view env = vmemkv::getenv_view("VMEMKV_TEST_TMPDIR"); !env.empty()) {
      const std::filesystem::path env_root{std::string(env)};
      if (!is_tmpfs(env_root)) {
        return env_root;
      }
    }
    for (const char *fallback : {"/tmp", "/var/tmp"}) {
      const std::filesystem::path candidate(fallback);
      if (!is_tmpfs(candidate)) {
        return candidate;
      }
    }
    return std::filesystem::path();
  }();
  return root;
}

// Builds a unique-per-process-per-call temp file path under `prefix`.
inline auto reserve_unique_temp_path(std::string_view prefix,
                                     bool also_remove_wal_sibling = false,
                                     const std::filesystem::path &root = test_temp_root()) -> std::filesystem::path {
  static std::atomic<uint64_t> counter{0};
  const uint64_t sequence_number = counter.fetch_add(1, std::memory_order_relaxed);
  std::filesystem::path temp_path = root / (std::string(prefix) + "_" + std::to_string(static_cast<long>(::getpid())) +
                                            "_" + std::to_string(sequence_number));
  std::error_code ignored;
  std::filesystem::remove(temp_path, ignored);
  if (also_remove_wal_sibling) {
    vmemkv::remove_wal_segments(vmemkv::derive_wal_path(temp_path));
  }
  return temp_path;
}

// Unique temp directory (created), for fake sysfs/cgroup trees.
inline auto reserve_unique_temp_dir(std::string_view prefix) -> std::filesystem::path {
  static std::atomic<uint64_t> counter{0};
  const std::filesystem::path dir =
      test_temp_root() / (std::string(prefix) + "_" + std::to_string(static_cast<long>(::getpid())) + "_" +
                          std::to_string(counter.fetch_add(1, std::memory_order_relaxed)));
  std::error_code ignored;
  std::filesystem::create_directories(dir, ignored);
  return dir;
}

// RAII guard around a path from reserve_unique_temp_path().
class ScopedTempPath {
 public:
  explicit ScopedTempPath(std::string_view prefix,
                          std::function<void(const std::filesystem::path &)> cleanup,
                          const std::filesystem::path &root = test_temp_root())
      : path_(reserve_unique_temp_path(prefix, /*also_remove_wal_sibling=*/false, root)), cleanup_(std::move(cleanup)) {
    cleanup_(path_);
  }
  ~ScopedTempPath() { cleanup_(path_); }

  ScopedTempPath(const ScopedTempPath &) = delete;
  auto operator=(const ScopedTempPath &) -> ScopedTempPath & = delete;

  operator const std::filesystem::path &() const noexcept { return path_; }
  [[nodiscard]] auto get() const noexcept -> const std::filesystem::path & { return path_; }

 private:
  std::filesystem::path path_;
  std::function<void(const std::filesystem::path &)> cleanup_;
};

}  // namespace vmemkv_test
