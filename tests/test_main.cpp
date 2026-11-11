// Defines the entry point for doctest. Separated into a separate file to speed up incremental builds.
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

// The compaction worker stays suppressed by default; compaction tests enable it explicitly.
namespace {
struct AutoCompactSuppressor {
  AutoCompactSuppressor() { ::setenv("VMEMKV_SUPPRESS_AUTO_COMPACT", "1", 1); }
} auto_compact_suppressor;

// Sweeps every test temp file/directory this process created under the test root at process
// exit: paths that outlive their cases (crash simulations destroy stores mid-test) would
// otherwise accumulate in the shared tmpfs across runs. Root resolution duplicates
// test_temp.hpp test_temp_root() in a standard-library-only form: this entry point also builds
// the public-API test, which must not include the internal test helpers.
struct TempFileSweeper {
  TempFileSweeper() = default;
  TempFileSweeper(const TempFileSweeper &) = delete;
  auto operator=(const TempFileSweeper &) -> TempFileSweeper & = delete;
  ~TempFileSweeper() {
    const std::string pid_token = "_" + std::to_string(static_cast<long>(::getpid())) + "_";
    std::filesystem::path root;
    if (const char *env = std::getenv("VMEMKV_TEST_TMPDIR"); env != nullptr && *env != '\0') {
      root = env;
    } else {
      std::error_code ignored;
      root = std::filesystem::is_directory("/dev/shm", ignored) ? std::filesystem::path("/dev/shm")
                                                                : std::filesystem::temp_directory_path();
    }
    std::error_code ignored;
    for (const auto &entry : std::filesystem::directory_iterator(root, ignored)) {
      if (entry.path().filename().string().find(pid_token) == std::string::npos) {
        continue;
      }
      std::filesystem::remove_all(entry.path(), ignored);
    }
  }
} temp_file_sweeper;
}  // namespace

// CTest (and CI generally) runs this binary with stdout piped, not attached to a TTY, so libc
// switches from line-buffered to fully-buffered stdio -- doctest's per-TEST_CASE progress lines
// then sit in an internal buffer instead of reaching the CI log until the buffer fills or the
// process exits normally. On a timeout kill (e.g. ctest's TIMEOUT), that buffered-but-unflushed
// output is lost, so a hang shows up as a blank log with no indication of which test was running.
// Forcing line buffering here means every TEST_CASE header reaches the log in real time.
int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, BUFSIZ);
  doctest::Context context;
  context.applyCommandLine(argc, argv);
  const int result = context.run();
  if (context.shouldExit()) {
    return result;
  }
  return result;
}
