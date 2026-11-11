// test_cgroup_throttle.cpp -- Unit tests for CgroupMemoryThrottle against fake cgroup
// directories (no real cgroup needed; the constructor's directory override is always used so
// the host environment, including VMEMKV_CGROUP_ROOT, cannot affect the results).

#include <doctest/doctest.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <concepts>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vmemkv/core/cgroup_memory_throttle.hpp>

#include "test_support.hpp"

namespace {

void write_cgroup_file(const std::filesystem::path &dir, const char *name, const std::string &content) {
  std::ofstream file(dir / name, std::ios::trunc);
  file << content;
}

struct FakeCgroup {
  std::filesystem::path dir = vmemkv_test::reserve_unique_temp_dir("cgthrottle");
  ~FakeCgroup() {
    std::error_code ignored;
    std::filesystem::remove_all(dir, ignored);
  }
};

auto elapsed_for(std::invocable auto &&fn) -> std::chrono::microseconds {
  const auto start = std::chrono::steady_clock::now();
  fn();
  return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start);
}

}  // namespace

TEST_CASE("CgroupMemoryThrottle: missing directory stays inactive") {
  vmemkv::CgroupMemoryThrottle throttle(1, "/nonexistent-cgroup-dir-xyz");
  CHECK(!throttle.active());
  CHECK(elapsed_for([&] { throttle.maybe_throttle(0); }) < std::chrono::milliseconds(50));
}

TEST_CASE("CgroupMemoryThrottle: memory.high=max stays inactive") {
  FakeCgroup cg;
  write_cgroup_file(cg.dir, "memory.high", "max\n");
  write_cgroup_file(cg.dir, "memory.current", "12345\n");
  vmemkv::CgroupMemoryThrottle throttle(1, cg.dir.c_str());
  CHECK(!throttle.active());
}

TEST_CASE("CgroupMemoryThrottle: low usage never sleeps") {
  FakeCgroup cg;
  write_cgroup_file(cg.dir, "memory.high", "1000000\n");
  write_cgroup_file(cg.dir, "memory.current", "500000\n");
  vmemkv::CgroupMemoryThrottle throttle(1, cg.dir.c_str());
  CHECK(throttle.active());
  // 200 samples with no sleep cost file reads only (~µs each); a single 2ms sleep tier firing
  // by mistake would already blow this bound by 400ms.
  CHECK(elapsed_for([&] {
          for (size_t i = 0; i < 200; ++i) {
            throttle.maybe_throttle(i);
          }
        }) < std::chrono::milliseconds(100));
  CHECK(throttle.throttle_events() == 0);
}

TEST_CASE("CgroupMemoryThrottle: usage at the limit backs off hard") {
  FakeCgroup cg;
  write_cgroup_file(cg.dir, "memory.high", "1000000\n");
  write_cgroup_file(cg.dir, "memory.current", "1000000\n");
  vmemkv::CgroupMemoryThrottle throttle(1, cg.dir.c_str());
  CHECK(throttle.active());
  // 3 samples x 2ms sleep each; relative sleeps never return early, only late.
  CHECK(elapsed_for([&] {
          for (size_t i = 0; i < 3; ++i) {
            throttle.maybe_throttle(i);
          }
        }) >= std::chrono::milliseconds(5));
  CHECK(throttle.throttle_events() == 3);
}

TEST_CASE("CgroupMemoryThrottle: usage at 96% eases off early") {
  FakeCgroup cg;
  write_cgroup_file(cg.dir, "memory.high", "1000000\n");
  write_cgroup_file(cg.dir, "memory.current", "960000\n");
  vmemkv::CgroupMemoryThrottle throttle(1, cg.dir.c_str());
  CHECK(throttle.active());
  CHECK(elapsed_for([&] { throttle.maybe_throttle(0); }) >= std::chrono::microseconds(50));
}

TEST_CASE("CgroupMemoryThrottle: sampling honors the check interval") {
  FakeCgroup cg;
  write_cgroup_file(cg.dir, "memory.high", "1000000\n");
  write_cgroup_file(cg.dir, "memory.current", "1000000\n");
  vmemkv::CgroupMemoryThrottle throttle(100, cg.dir.c_str());
  CHECK(throttle.active());
  // Index 1 is not a sampling point at interval 100: no file read, no sleep.
  CHECK(elapsed_for([&] { throttle.maybe_throttle(1); }) < std::chrono::milliseconds(50));
}
