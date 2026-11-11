// cgroup_memory_throttle.hpp -- Voluntary pacing for bulk ingest under cgroup v2 limits.
//
// Under a memory.high limit there is no asynchronous reclaim: overage is served by the
// faulting thread's own synchronous direct reclaim, so a sustained dirtying burst stalls
// instead of converging. This throttle samples memory.current against memory.high and
// sleeps as usage nears the limit. Without a limit it is permanently inactive (one
// counter check per call).
#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <thread>
#include <vmemkv/core/env.hpp>

namespace vmemkv {

inline constexpr std::size_t kCgroupCheckIntervalKeys = 1024;
inline constexpr uint64_t kCgroupPercentScale = 100;
inline constexpr uint64_t kCgroupSoftLimitPercent = 95;
inline constexpr uint64_t kCgroupSoftBackoffUsec = 200;

class CgroupMemoryThrottle {
 public:
  // check_interval: sample pressure once per this many keys.
  // cgroup_dir_override (or VMEMKV_CGROUP_ROOT): use this directory instead of
  // auto-detecting from /proc/self/cgroup.
  explicit CgroupMemoryThrottle(size_t check_interval = kCgroupCheckIntervalKeys,
                                const char *cgroup_dir_override = nullptr)
      : check_interval_(check_interval == 0 ? 1 : check_interval) {
    const std::string_view dir =
        (cgroup_dir_override != nullptr) ? std::string_view(cgroup_dir_override) : getenv_view("VMEMKV_CGROUP_ROOT");
    if (!dir.empty()) {
      init_from_dir(std::string(dir));
    } else {
      init_from_self_cgroup();
    }
  }

  // Samples pressure (at most once per check_interval keys); sleeps near the limit.
  void maybe_throttle(size_t index) {
    if (!active_ || (index % check_interval_) != 0) {
      return;
    }
    uint64_t current = 0;
    if (!read_uint_file(dir_ + "/memory.current", current)) {
      return;
    }
    // Over budget: back off hard. At 95%: ease off before the race starts.
    if (current >= high_bytes_) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      throttle_events_.fetch_add(1, std::memory_order_relaxed);
    } else if (current * kCgroupPercentScale >= high_bytes_ * kCgroupSoftLimitPercent) {
      std::this_thread::sleep_for(std::chrono::microseconds(kCgroupSoftBackoffUsec));
      throttle_events_.fetch_add(1, std::memory_order_relaxed);
    }
  }

  [[nodiscard]] auto active() const noexcept -> bool { return active_; }

  // Lifetime count of backpressure sleeps taken (diagnostic).
  [[nodiscard]] auto throttle_events() const noexcept -> uint64_t {
    return throttle_events_.load(std::memory_order_relaxed);
  }

 private:
  void init_from_dir(const std::string &dir) {
    uint64_t high = 0;
    if (!read_uint_file(dir + "/memory.high", high) || high == 0) {
      return;
    }
    dir_ = dir;
    high_bytes_ = high;
    active_ = true;
  }

  // Walks up from this process's cgroup to the hierarchy root for a numeric memory.high.
  void init_from_self_cgroup() {
    std::ifstream self_cgroup("/proc/self/cgroup");
    if (!self_cgroup.is_open()) {
      return;
    }
    std::string subpath;
    for (std::string line; std::getline(self_cgroup, line);) {
      // cgroup v2 entry: "0::<subpath>".
      if (line.starts_with("0::")) {
        subpath = line.substr(3);
        break;
      }
    }
    if (subpath.empty()) {
      return;
    }
    std::filesystem::path dir = std::filesystem::path("/sys/fs/cgroup") / subpath.substr(1);
    const std::filesystem::path root("/sys/fs/cgroup");
    for (; dir.string().size() >= root.string().size(); dir = dir.parent_path()) {
      uint64_t high = 0;
      if (read_uint_file((dir / "memory.high").string(), high) && high != 0) {
        dir_ = dir.string();
        high_bytes_ = high;
        active_ = true;
        return;
      }
      if (dir == root) {
        break;
      }
    }
  }

  size_t check_interval_;
  bool active_ = false;
  uint64_t high_bytes_ = 0;
  std::string dir_;
  std::atomic<uint64_t> throttle_events_{0};
};

}  // namespace vmemkv
