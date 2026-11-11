// NOLINTBEGIN
// bench_kv.cpp -- vmemkv microbenchmarks using Google Benchmark
//
// All benchmark functions are dynamic registrations over store variants.
// Runs are executed across a dynamic range of threads to measure scalability.
// Real-time (wall-clock) measurement is used for accurate throughput.

#include <benchmark/benchmark.h>
#include <fcntl.h>
#include <linux/fs.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <rivals/store_variants.hpp>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <vector>
#include <vmemkv/core/env.hpp>
#include <vmemkv/core/io.hpp>
#include <vmemkv/variants.hpp>

#include "bench_helpers.hpp"

namespace {

struct ProcIO {
  uint64_t read_bytes{0};
  uint64_t write_bytes{0};
};

static inline ProcIO read_proc_self_io() {
  ProcIO io{};
  std::ifstream f("/proc/self/io");
  if (!f.is_open()) return io;
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("read_bytes:", 0) == 0) {
      io.read_bytes = std::strtoull(line.c_str() + 11, nullptr, 10);
    } else if (line.rfind("write_bytes:", 0) == 0) {
      io.write_bytes = std::strtoull(line.c_str() + 12, nullptr, 10);
    }
  }
  return io;
}

static inline uint64_t measure_physical_disk_bytes(const std::string &db_dir) {
  uint64_t total_bytes = 0;
  std::error_code ec;
  std::set<std::pair<uint64_t, uint64_t>> seen_inodes;
  for (const auto &entry : std::filesystem::recursive_directory_iterator(db_dir, ec)) {
    if (ec) break;
    if (entry.is_regular_file(ec)) {
      const std::string path_str = entry.path().string();
      const std::string filename = entry.path().filename().string();
      const bool is_master =
          (path_str.find("_master") != std::string::npos && path_str.find("_clone") == std::string::npos);
      if (filename == "swapfile" || is_master) {
        continue;
      }
      struct stat st {};
      if (::stat(entry.path().c_str(), &st) == 0) {
        if (seen_inodes.insert({static_cast<uint64_t>(st.st_dev), static_cast<uint64_t>(st.st_ino)}).second) {
          total_bytes += static_cast<uint64_t>(st.st_blocks) * 512;
        }
      }
    }
  }
  return total_bytes;
}

static inline int get_ycsb_duration_seconds() {
  const char *env = std::getenv("VMEMKV_YCSB_DURATION_SEC");
  if (!env) env = std::getenv("YCSB_DURATION_SEC");
  if (env && env[0] != '\0') {
    try {
      int d = std::stoi(env);
      if (d > 0) return d;
    } catch (...) {
    }
  }
  return 30;
}

static inline int get_ycsb_warmup_seconds() {
  const char *env = std::getenv("VMEMKV_YCSB_WARMUP_SEC");
  if (!env) env = std::getenv("YCSB_WARMUP_SEC");
  if (env && env[0] != '\0') {
    try {
      int w = std::stoi(env);
      if (w >= 0) return w;
    } catch (...) {
    }
  }
  return 0;
}

static inline bool is_forced_triggers_disabled() {
  const char *env = std::getenv("VMEMKV_YCSB_DISABLE_FORCED_TRIGGERS");
  if (!env || env[0] == '\0') {
    return get_ycsb_duration_seconds() > 30;
  }
  return (std::string_view(env) == "1" || std::string_view(env) == "true");
}

static inline uint64_t compute_logical_bytes(uint64_t keys, size_t val_size) {
  constexpr size_t key_len = bench_help::kIndexKeyBytes;
  if (val_size == bench_help::kInlineValueBytes) {
    return keys * (key_len + val_size);
  }
  uint64_t count_8b = (keys + bench_help::kMixEveryNth - 1) / bench_help::kMixEveryNth;
  uint64_t count_large = keys - count_8b;
  return keys * key_len + count_8b * bench_help::kInlineValueBytes + count_large * val_size;
}

class YCSBTimelineCollector {
 public:
  const int duration_seconds;
  uint64_t initial_logical_bytes;

  void reset_after_warmup(size_t val_size) {
    uint64_t current_keys = next_key_index.load(std::memory_order_relaxed);
    initial_logical_bytes = compute_logical_bytes(current_keys, val_size);
    for (auto &tc : counters) {
      for (auto &sec : tc->per_sec) {
        sec.scan_count.store(0, std::memory_order_relaxed);
        sec.insert_count.store(0, std::memory_order_relaxed);
        sec.scan_req_bytes.store(0, std::memory_order_relaxed);
        sec.insert_req_bytes.store(0, std::memory_order_relaxed);
      }
    }
  }

  struct PerSecCounts {
    alignas(64) std::atomic<uint64_t> scan_count{0};
    alignas(64) std::atomic<uint64_t> insert_count{0};
    alignas(64) std::atomic<uint64_t> scan_req_bytes{0};
    alignas(64) std::atomic<uint64_t> insert_req_bytes{0};
  };

  struct ThreadCounter {
    std::vector<PerSecCounts> per_sec;
    explicit ThreadCounter(int duration_sec) : per_sec(duration_sec) {}
  };

  struct ForcedEvent {
    int scheduled_sec;
    int fired_sec;
    std::string kind;
    double elapsed_sec;
  };

  struct BackgroundJobEvent {
    std::string kind;
    double start_sec{0.0};
    double end_sec{0.0};
    double duration_sec{0.0};
    uint64_t bytes{0};
  };

  struct SecondSample {
    uint64_t disk_read_bytes{0};
    uint64_t disk_write_bytes{0};
    uint64_t physical_disk_bytes{0};
    uint64_t logical_live_bytes{0};
    uint64_t t2_checkpoint_ops{0};
    uint64_t t2_forced_checkpoint_ops{0};
    uint64_t defrag_cycle_ops{0};
  };

  std::vector<std::unique_ptr<ThreadCounter>> counters;
  std::vector<SecondSample> samples;
  std::vector<ForcedEvent> forced_events;
  std::vector<BackgroundJobEvent> background_jobs;

  std::atomic<uint64_t> last_recorded_checkpoint_t2{0};
  std::atomic<uint64_t> last_recorded_compaction_cycles{0};
  ProcIO prev_proc_io{};
  std::atomic<uint64_t> next_key_index;
  std::chrono::steady_clock::time_point start_time;

  YCSBTimelineCollector(size_t num_threads, uint64_t initial_keys, size_t val_size)
      : duration_seconds(get_ycsb_duration_seconds()),
        initial_logical_bytes(compute_logical_bytes(initial_keys, val_size)),
        samples(duration_seconds),
        next_key_index(initial_keys) {
    counters.reserve(num_threads);
    for (size_t t = 0; t < num_threads; ++t) {
      counters.push_back(std::make_unique<ThreadCounter>(duration_seconds));
    }
  }

  void start() {
    start_time = std::chrono::steady_clock::now();
    prev_proc_io = read_proc_self_io();
  }

  void record_op(size_t thread_idx, bool is_scan, uint64_t bytes) {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
    if (elapsed >= 0 && elapsed < duration_seconds && thread_idx < counters.size()) {
      auto &sec = counters[thread_idx]->per_sec[elapsed];
      if (is_scan) {
        sec.scan_count.fetch_add(1, std::memory_order_relaxed);
        sec.scan_req_bytes.fetch_add(bytes, std::memory_order_relaxed);
      } else {
        sec.insert_count.fetch_add(1, std::memory_order_relaxed);
        sec.insert_req_bytes.fetch_add(bytes, std::memory_order_relaxed);
      }
    }
  }

  void poll_second(int elapsed_sec, const std::string &db_dir, const vmemkv::VMemKVStatistics &stats) {
    if (elapsed_sec < 0 || elapsed_sec >= duration_seconds) return;

    auto cur_io = read_proc_self_io();
    uint64_t dr = (cur_io.read_bytes >= prev_proc_io.read_bytes) ? (cur_io.read_bytes - prev_proc_io.read_bytes) : 0;
    uint64_t dw =
        (cur_io.write_bytes >= prev_proc_io.write_bytes) ? (cur_io.write_bytes - prev_proc_io.write_bytes) : 0;
    prev_proc_io = cur_io;

    auto &s = samples[elapsed_sec];
    s.disk_read_bytes = dr;
    s.disk_write_bytes = dw;
    s.physical_disk_bytes = measure_physical_disk_bytes(db_dir);

    uint64_t cum_insert_bytes = 0;
    for (const auto &tc : counters) {
      for (int i = 0; i <= elapsed_sec; ++i) {
        cum_insert_bytes += tc->per_sec[i].insert_req_bytes.load(std::memory_order_relaxed);
      }
    }
    s.logical_live_bytes = initial_logical_bytes + cum_insert_bytes;

    uint64_t cur_ckpt = stats.checkpoint_count;
    uint64_t prev_ckpt = last_recorded_checkpoint_t2.load(std::memory_order_relaxed);
    if (cur_ckpt > prev_ckpt) {
      uint64_t diff = cur_ckpt - prev_ckpt;
      s.t2_checkpoint_ops += diff;
      last_recorded_checkpoint_t2.store(cur_ckpt, std::memory_order_relaxed);

      double dur_sec = static_cast<double>(stats.last_checkpoint_duration_us) / 1e6;
      double end_t = static_cast<double>(elapsed_sec + 1);
      double start_t = std::max(0.0, end_t - dur_sec);
      background_jobs.push_back({"checkpoint", start_t, end_t, dur_sec, stats.last_checkpoint_bytes_synced});
    }

    uint64_t cur_compaction = stats.compaction_cycle_count;
    uint64_t prev_compaction = last_recorded_compaction_cycles.load(std::memory_order_relaxed);
    if (cur_compaction > prev_compaction) {
      uint64_t diff = cur_compaction - prev_compaction;
      s.defrag_cycle_ops += diff;
      last_recorded_compaction_cycles.store(cur_compaction, std::memory_order_relaxed);

      double dur_sec = 0.0;
      double end_t = static_cast<double>(elapsed_sec + 1);
      double start_t = end_t;
      background_jobs.push_back({"compaction", start_t, end_t, dur_sec, 0});
    }
  }

  void record_forced_checkpoint(int scheduled_sec, int bucket_sec, double call_elapsed_sec, uint64_t bytes_synced) {
    if (bucket_sec >= 0 && bucket_sec < duration_seconds) {
      samples[bucket_sec].t2_forced_checkpoint_ops += 1;
    }
    forced_events.push_back({scheduled_sec, bucket_sec, "checkpoint", call_elapsed_sec});
    double end_t = static_cast<double>(bucket_sec + 1);
    double start_t = std::max(0.0, end_t - call_elapsed_sec);
    background_jobs.push_back({"forced_checkpoint", start_t, end_t, call_elapsed_sec, bytes_synced});
  }

  void dump_json(std::string store_name, std::string variant_name, std::string val_name, std::string scenario_tag) {
    std::vector<uint64_t> total_scan(duration_seconds, 0);
    std::vector<uint64_t> total_insert(duration_seconds, 0);
    std::vector<uint64_t> total_scan_bytes(duration_seconds, 0);
    std::vector<uint64_t> total_insert_bytes(duration_seconds, 0);

    for (const auto &tc : counters) {
      for (int i = 0; i < duration_seconds; ++i) {
        total_scan[i] += tc->per_sec[i].scan_count.load(std::memory_order_relaxed);
        total_insert[i] += tc->per_sec[i].insert_count.load(std::memory_order_relaxed);
        total_scan_bytes[i] += tc->per_sec[i].scan_req_bytes.load(std::memory_order_relaxed);
        total_insert_bytes[i] += tc->per_sec[i].insert_req_bytes.load(std::memory_order_relaxed);
      }
    }

    // Sanitize parameters to prevent slash / from breaking folder path
    for (auto *s : {&store_name, &variant_name, &val_name, &scenario_tag}) {
      std::replace(s->begin(), s->end(), '/', '-');
      std::replace(s->begin(), s->end(), ' ', '_');
      std::replace(s->begin(), s->end(), '(', '_');
      std::replace(s->begin(), s->end(), ')', '_');
      std::replace(s->begin(), s->end(), '%', '_');
    }

    std::string filename =
        "/tmp/ycsb_e_timeline_" + scenario_tag + "_" + store_name + "_" + variant_name + "_" + val_name + ".json";
    std::ofstream out(filename);
    if (out.is_open()) {
      out << "{\n";
      out << "  \"store\": \"" << store_name << "\",\n";
      out << "  \"variant\": \"" << variant_name << "\",\n";
      out << "  \"value_size\": \"" << val_name << "\",\n";
      out << "  \"duration_seconds\": " << duration_seconds << ",\n";
      out << "  \"initial_logical_bytes\": " << initial_logical_bytes << ",\n";
      out << "  \"timeline\": [\n";
      for (int i = 0; i < duration_seconds; ++i) {
        const auto &s = samples[i];
        out << "    {\"sec\": " << (i + 1) << ", \"scan_ops\": " << total_scan[i]
            << ", \"insert_ops\": " << total_insert[i] << ", \"scan_req_bytes\": " << total_scan_bytes[i]
            << ", \"insert_req_bytes\": " << total_insert_bytes[i] << ", \"disk_read_bytes\": " << s.disk_read_bytes
            << ", \"disk_write_bytes\": " << s.disk_write_bytes
            << ", \"physical_disk_bytes\": " << s.physical_disk_bytes
            << ", \"logical_live_bytes\": " << s.logical_live_bytes
            << ", \"t2_checkpoint_ops\": " << s.t2_checkpoint_ops
            << ", \"t2_forced_checkpoint_ops\": " << s.t2_forced_checkpoint_ops
            << ", \"defrag_cycle_ops\": " << s.defrag_cycle_ops << "}";
        if (i < duration_seconds - 1) out << ",";
        out << "\n";
      }
      out << "  ],\n";
      out << "  \"forced_events\": [\n";
      for (size_t i = 0; i < forced_events.size(); ++i) {
        const auto &ev = forced_events[i];
        out << "    {\"scheduled_sec\": " << ev.scheduled_sec << ", \"fired_sec\": " << ev.fired_sec << ", \"kind\": \""
            << ev.kind << "\", \"elapsed_sec\": " << ev.elapsed_sec << "}";
        if (i + 1 < forced_events.size()) out << ",";
        out << "\n";
      }
      out << "  ],\n";
      out << "  \"background_jobs\": [\n";
      for (size_t i = 0; i < background_jobs.size(); ++i) {
        const auto &bj = background_jobs[i];
        out << "    {\"kind\": \"" << bj.kind << "\", \"start_sec\": " << bj.start_sec
            << ", \"end_sec\": " << bj.end_sec << ", \"duration_sec\": " << bj.duration_sec
            << ", \"bytes\": " << bj.bytes << "}";
        if (i + 1 < background_jobs.size()) out << ",";
        out << "\n";
      }
      out << "  ]\n";
      out << "}\n";
    }
  }
};

// Fixed schedule for YCSB-E's forced checkpoint() calls (see the trigger site in
// register_ycsb_e_benchmark() below): natural triggering is unreliable within the 30s window
// (especially under LTM's lower throughput), so YCSB-E's timeline would otherwise show little to
// no checkpoint activity for some scenarios. Firing these deterministically guarantees comparable
// data points every run instead of leaving it to chance.
//
// t=10s and t=25s: two checkpoint() calls, spaced 15s apart -- enough room for each to complete
// and for surrounding throughput to resettle before/after. Two calls is enough to see whether a
// second cycle looks like the first.
//
// No forced reorganize() here: post-sharding, store->reorganize() forces every T1 shard through a
// full synchronous merge (see store_adapter.hpp's doc comment), so its cost scales with total
// corpus size like checkpoint()'s does -- it is no longer a cheap, distinct reference point worth
// contrasting against checkpoint() in this chart. It would also duplicate
// probe bg-jobs's dedicated, cleaner measurement of that same forced-job cost.
//
// Deliberately no guard against a late-running call pushing a later trigger's fire time past its
// own schedule mark, or even past kDurationSeconds entirely: if checkpoint() at t=10s runs long
// enough to blow through the t=25s mark, the next check (on this thread's very next loop
// iteration, see the trigger site) fires it immediately back-to-back -- itself a legitimate,
// informative result, not a bug to engineer around.
struct ForcedTrigger {
  int second_mark;
};
constexpr std::array<ForcedTrigger, 2> kForcedTriggers{{
    {10},
    {25},
}};

constexpr std::size_t kInMemoryInlineCorpusEntries = 20'000'000;
// 1KB in-memory is fixed at roughly the same scale as the LTM/1KB scenario's own corpus (~8.26M
// keys/8.6GB, itself sized from a small fixed budget x ratio, not host RAM) rather than scaling
// with the host's real RAM (target_ratio=0.5) like every other non-8B case -- this keeps
// in_memory/1KB directly comparable to ltm/1KB (same data, different memory pressure) and fast to
// populate on real NVMe. 8B keeps its own separate, smaller fixed cap (kInMemoryInlineCorpusEntries)
// since even this same entry count at 8B's much smaller footprint would be needlessly small.
constexpr std::size_t kInMemory1KBCorpusEntries = 8'000'000;
static inline bool get_env_flag01(const char *name, bool def);
static inline uint64_t get_env_uint64(const char *name, uint64_t def);
static inline bool is_ltm_mode() { return get_env_flag01("VMEMKV_BENCH_LTM", false); }

static inline bool force_host_memory() { return get_env_flag01("VMEMKV_BENCH_FORCE_HOST_MEMORY", false); }

static inline std::optional<std::size_t> read_size_from_file(const char *path) {
  std::ifstream input(path);
  if (!input.is_open()) {
    return std::nullopt;
  }

  std::string token;
  input >> token;
  if (!input) {
    return std::nullopt;
  }

  if (token == "max") {
    return std::nullopt;
  }

  try {
    return static_cast<std::size_t>(std::stoull(token));
  } catch (...) {
    return std::nullopt;
  }
}

static inline std::optional<std::size_t> read_size_from_env(const char *name) {
  const char *value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return std::nullopt;
  }

  try {
    return static_cast<std::size_t>(std::stoull(value));
  } catch (...) {
    return std::nullopt;
  }
}

static inline std::size_t detect_machine_memory_bytes() {
  constexpr std::size_t kHugeLimitThreshold = 1ULL << 60;

  if (const auto context_budget = read_size_from_env("VMEMKV_CONTEXT_memory_budget_bytes");
      context_budget.has_value() && *context_budget > 0 && *context_budget < kHugeLimitThreshold) {
    return *context_budget;
  }

  if (force_host_memory()) {
    if (const auto mem_total = vmemkv::read_meminfo_bytes("/proc/meminfo", "MemTotal:")) {
      return *mem_total;
    }
    return 256ULL * 1024ULL * 1024ULL;
  }

  // Prefer cgroup limits when present so containerized runs use their effective
  // memory ceiling instead of the host's MemTotal.
  if (const auto cgroup_v2 = read_size_from_file("/sys/fs/cgroup/memory.max");
      cgroup_v2.has_value() && *cgroup_v2 < kHugeLimitThreshold) {
    return *cgroup_v2;
  }
  if (const auto cgroup_v1 = read_size_from_file("/sys/fs/cgroup/memory/memory.limit_in_bytes");
      cgroup_v1.has_value() && *cgroup_v1 < kHugeLimitThreshold) {
    return *cgroup_v1;
  }

  if (const auto mem_total = vmemkv::read_meminfo_bytes("/proc/meminfo", "MemTotal:")) {
    return *mem_total;
  }

  // Conservative fallback if /proc/meminfo is unavailable.
  return 256ULL * 1024ULL * 1024ULL;
}

static inline std::string get_db_dir() {
  const char *env = std::getenv("VMEMKV_DB_DIR");
  return env != nullptr ? std::string(env) : ".";
}

// Sweeps get_db_dir() for every file/directory this harness could have left behind (T2 files,
// WAL/manifest/.chkN.* checkpoints, rival on-disk directories, and the
// "*_master"/"*_master.building"/"*_clone_N" families in for_each_store_variant()), based purely
// on the "bench_" filename prefix every one of these paths shares (see
// for_each_store_variant()'s `filename` construction). `except_prefixes` spares any entry whose
// name starts with one of them -- the caller's own path, either about to be freshly rebuilt (safe
// to keep whatever partial state is there; the fresh construction wipes it anyway) or an
// already-built master a sibling holder (Get/Update/Delete/Scan/YCSB-E can all share one master
// -- see master_corpus_path() in for_each_store_variant()) may still need.
//
// Called from two places, deliberately not more: once unconditionally at process startup (see
// main()), and once per *fresh* (never a reuse/clone-of-existing-master) construction --
// make_vmemkv_fresh() below, and cleanup_before_master_build_if_needed() the instant before a
// rival or vmemkv master build that doesn't exist on disk yet. Never called on a path that's
// reusing an already-built master or checkpoint: doing so there would delete a sibling holder's
// still-needed data out from under it, not just this holder's own. This is what keeps at most one
// store-variant/value-size "generation" of files on disk at a time instead of letting every
// variant/value-size this process ever touches pile up simultaneously -- which is what was
// actually filling the disk (a single run's own *peak* usage, one generation at a time, is much
// smaller than its *cumulative* usage across every variant it touches without ever clearing the
// previous one).
static void cleanup_stale_benchmark_files(const std::set<std::string> &except_prefixes = {}) {
  const std::filesystem::path dir(get_db_dir());
  std::error_code list_error;
  std::size_t removed_count = 0;
  for (const auto &entry : std::filesystem::directory_iterator(dir, list_error)) {
    const std::string entry_name = entry.path().filename().string();
    if (!entry_name.starts_with("bench_")) {
      continue;
    }
    const bool spared = std::any_of(except_prefixes.begin(), except_prefixes.end(), [&](const std::string &prefix) {
      return entry_name.starts_with(prefix);
    });
    if (spared) {
      continue;
    }
    std::error_code remove_error;
    const auto removed = std::filesystem::remove_all(entry.path(), remove_error);
    if (!remove_error) {
      removed_count += static_cast<std::size_t>(removed > 0);
    }
  }
  if (removed_count > 0) {
    std::cout << "Cleaned up " << removed_count << " stale bench_* path(s) in " << dir.string() << std::endl;
  }
}

// Opt-out for main()'s startup cleanup_stale_benchmark_files() sweep and for the "sweep other
// stale corpora" step of every master build (cleanup_before_master_build_if_needed() for rivals,
// make_vmemkv_clone_from_checkpoint()'s own call into make_vmemkv_fresh() for vmemkv) -- *not*
// for make_vmemkv_fresh()'s ordinary (non-master) caller, Insert's non-LTM path, which stays
// unconditional (see its own comment).
//
// Needed by a two-pass "prime masters unconstrained, then measure under a memory-constrained
// cgroup" runner strategy (see run_bench.sh local --cgroup's LTM path). The priming pass builds every shared
// master corpus a run will need, at full (uncapped) disk speed, *before* the actual measurement
// pass enters the cgroup -- a real per-master populate measured there was 200-1200s (vs. ~20-30s
// unconstrained). Both passes need this flag:
// - The measurement pass's own *startup* cleanup must not wipe every master priming just spent
//   real time building, the instant it starts -- that master was built by a separate, now-exited
//   process, so nothing in this process's own state can tell "just built, keep it" apart from
//   "genuinely stale" without this flag.
// - The priming pass itself needs it for the same reason across *retries*: primed_master_
//   prefixes() (see register_primed_master_prefix()) already keeps one store's build from
//   sweeping another's within a single successful run, but that tracking is empty again on a
//   fresh process, so a retried priming invocation still needs this flag to avoid re-sweeping
//   masters a previous, separate priming attempt already finished.
//
// Callers using this flag are expected to have already done their own "start from a clean
// slate" sweep once, up front, before priming begins (see run_bench.sh local) -- this flag does not
// replace that, it only suppresses the *automatic*, per-construction/per-startup sweeps that
// would otherwise fight a deliberate multi-process, multi-master-generation priming pass.
static inline bool should_skip_cleanup() { return get_env_flag01("VMEMKV_BENCH_SKIP_CLEANUP", false); }

// Every master path this process has registered as one to never sweep away as "stale" for the
// rest of this run, whether because it already existed or because this process is about to build
// it -- see register_primed_master_prefix() below.
static auto primed_master_prefixes() -> std::set<std::string> & {
  static std::set<std::string> prefixes;
  return prefixes;
}
static std::mutex primed_master_prefixes_mutex;

// Registers `master_path`'s filename into primed_master_prefixes() and returns a snapshot of the
// full, up-to-date set (self included) for immediate use as a cleanup_stale_benchmark_files()
// spare-list -- copied out under the lock, safe to read lock-free afterward. A run that builds
// several stores' masters in one process (e.g. an LTM priming pass covering vmemkv, RocksDB,
// RocksDB-BlobDB, LMDB, ...) needs every prior store's master spared here, not just the one
// currently being built, or each store's build would sweep the previous stores' just-built
// masters right back out.
static auto register_primed_master_prefix(const std::string &master_path) -> std::set<std::string> {
  std::lock_guard<std::mutex> lock(primed_master_prefixes_mutex);
  primed_master_prefixes().insert(std::filesystem::path(master_path).filename().string());
  return primed_master_prefixes();
}

// Called immediately before a rival's CloneFromMasterTag constructor hands off to the actual
// build. If that master doesn't exist on disk yet, this is about to be a real build -- sweep away
// every *other* stale corpus first (sparing every master primed so far this process, per
// register_primed_master_prefix()), so switching which master gets built next never lets more
// than one generation of files pile up. A no-op once the master already exists (the ordinary
// repeat-reuse/clone case) -- see cleanup_stale_benchmark_files()'s comment for why this must
// stay skipped there.
//
// VMEMKV_BENCH_SKIP_CLEANUP (should_skip_cleanup()) still gates this sweep on top of
// register_primed_master_prefix()'s in-process tracking, for the one thing an in-process set can
// never know: whether an on-disk master this same process hasn't touched yet was actually built
// by an *earlier, separate* process invocation (e.g. a retried priming pass) rather than being
// genuinely stale.
static void cleanup_before_master_build_if_needed(const std::string &master_path) {
  const std::set<std::string> primed = register_primed_master_prefix(master_path);
  if (!should_skip_cleanup() && !std::filesystem::exists(master_path)) {
    cleanup_stale_benchmark_files(primed);
  }
}

// Positive-double env override, or nullopt when unset/empty/unparsable/non-positive.
static inline std::optional<double> env_positive_double(const char *name) {
  const char *env = std::getenv(name);
  if (env == nullptr || *env == '\0') {
    return std::nullopt;
  }
  char *end = nullptr;
  const double parsed = std::strtod(env, &end);
  if (end == env || *end != '\0' || parsed <= 0.0) {
    return std::nullopt;
  }
  return parsed;
}

// Unsigned-integer env override, or def when unset/empty/unparsable.
static inline uint64_t get_env_uint64(const char *name, uint64_t def) {
  const char *env = std::getenv(name);
  if (env == nullptr || *env == '\0' || *env == '-') {
    return def;
  }
  try {
    return static_cast<uint64_t>(std::stoull(env));
  } catch (...) {
    return def;
  }
}

// "1"-valued env flag, or def when unset.
static inline bool get_env_flag01(const char *name, bool def) {
  const char *env = std::getenv(name);
  if (env == nullptr) {
    return def;
  }
  return std::string_view(env) == "1";
}

static inline double get_target_ratio() {
  if (const auto override = env_positive_double("VMEMKV_BENCH_TARGET_RATIO")) {
    return *override;
  }
  return is_ltm_mode() ? 2.0 : 0.5;
}

static inline std::size_t get_target_base_bytes() { return detect_machine_memory_bytes(); }

static inline std::size_t get_target_corpus_bytes() {
  static const std::size_t cached = []() {
    const double target = static_cast<double>(get_target_base_bytes()) * get_target_ratio();
    return static_cast<std::size_t>(std::llround(target));
  }();
  return cached;
}

// Shared by for_each_store_variant()'s registration loop (which needs corpus_size to size
// Insert/YCSB-E's own workload) and make_fresh_corpus()/make_fresh_corpus_checkpoint()'s
// rival-clone paths (which need it to know how many keys to bulk_load into a master corpus
// they haven't seen the caller compute yet).
static inline std::size_t corpus_size_for_value(std::size_t val_size) {
  if (!is_ltm_mode() && val_size == bench_help::kInlineValueBytes) {
    return kInMemoryInlineCorpusEntries;
  }
  if (!is_ltm_mode() && val_size == 1024ULL) {
    return kInMemory1KBCorpusEntries;
  }
  const size_t target_corpus_bytes = get_target_corpus_bytes();
  return std::max<std::size_t>(1, target_corpus_bytes / (bench_help::kIndexKeyBytes + val_size));
}

// Benchmark value-size matrix for the current scenario. LeanStore's BTreeVI lengths are u16
// (see LeanStoreStore::kMaxValueBytes), so the 64KB corpus is unstorable there: excluding it
// at registration means no benchmark filter can ever match an un-runnable cell (previously
// the exclusion lived only in run_bench.sh's matrix filters, and any broader custom filter
// would match the registered-but-aborting cells and kill the run mid-matrix).
static inline std::vector<std::size_t> value_sizes_for_mode(const std::string &store_name) {
  if (is_ltm_mode()) {
    if (store_name == "LeanStore") {
      return {1024ULL};
    }
    return {1024ULL, 64ULL * 1024ULL};
  }
  return {bench_help::kInlineValueBytes, 1024ULL};
}

// Non-8B corpora mix one inline-eligible 8B value per kMixEveryNth keys.
static inline bool use_mixed_8b_value(std::size_t val_size, std::size_t key_index) {
  return val_size != bench_help::kInlineValueBytes && key_index % bench_help::kMixEveryNth == 0;
}

// Scan range length for YCSB-E and the background-job probe.
constexpr std::size_t kScanLength = 100;

static void add_custom_context_from_env(const char *context_key, const char *env_name) {
  const std::string_view value = vmemkv::getenv_view(env_name);
  if (!value.empty()) {
    benchmark::AddCustomContext(context_key, std::string(value));
  }
}

static inline double insert_time_budget_seconds();
static inline double delete_time_budget_seconds();

static void register_benchmark_context() {
  add_custom_context_from_env("git_revision", "VMEMKV_CONTEXT_git_revision");
  add_custom_context_from_env("git_dirty", "VMEMKV_CONTEXT_git_dirty");
  add_custom_context_from_env("run_kind", "VMEMKV_CONTEXT_run_kind");
  add_custom_context_from_env("build_type", "VMEMKV_CONTEXT_build_type");
  add_custom_context_from_env("build_dir", "VMEMKV_CONTEXT_build_dir");
  add_custom_context_from_env("enable_rocksdb", "VMEMKV_CONTEXT_enable_rocksdb");
  benchmark::AddCustomContext("populate_bytes_scale_by_memory", std::to_string(get_target_ratio()));
  add_custom_context_from_env("scenario_order", "VMEMKV_CONTEXT_scenario_order");
  add_custom_context_from_env("flags", "VMEMKV_CONTEXT_flags");
  add_custom_context_from_env("memory_budget_source", "VMEMKV_CONTEXT_memory_budget_source");
  add_custom_context_from_env("memory_budget_bytes", "VMEMKV_CONTEXT_memory_budget_bytes");
  add_custom_context_from_env("memory_swap_max_bytes", "VMEMKV_CONTEXT_memory_swap_max_bytes");
  add_custom_context_from_env("swap_storage_media", "VMEMKV_CONTEXT_swap_storage_media");
  add_custom_context_from_env("kernel_release", "VMEMKV_CONTEXT_kernel_release");
  add_custom_context_from_env("cpu_model", "VMEMKV_CONTEXT_cpu_model");
  add_custom_context_from_env("cpu_count", "VMEMKV_CONTEXT_cpu_count");
  add_custom_context_from_env("mem_total_bytes", "VMEMKV_CONTEXT_mem_total_bytes");
  add_custom_context_from_env("cgroup_memory_limit_bytes", "VMEMKV_CONTEXT_cgroup_memory_limit_bytes");
  add_custom_context_from_env("swap_total_bytes", "VMEMKV_CONTEXT_swap_total_bytes");
  add_custom_context_from_env("lru_gen_enabled", "VMEMKV_CONTEXT_lru_gen_enabled");
  add_custom_context_from_env("instance_type", "VMEMKV_CONTEXT_instance_type");
  add_custom_context_from_env("aws_region", "VMEMKV_CONTEXT_aws_region");
  add_custom_context_from_env("ami_id", "VMEMKV_CONTEXT_ami_id");
  add_custom_context_from_env("memo", "VMEMKV_CONTEXT_memo");
  add_custom_context_from_env("min_time_seconds", "VMEMKV_CONTEXT_min_time_seconds");
  add_custom_context_from_env("repetitions", "VMEMKV_CONTEXT_repetitions");
  add_custom_context_from_env("rocksdb_version", "VMEMKV_CONTEXT_rocksdb_version");
  add_custom_context_from_env("lmdb_version", "VMEMKV_CONTEXT_lmdb_version");
  add_custom_context_from_env("leanstore_pinned_commit", "VMEMKV_CONTEXT_leanstore_pinned_commit");
  add_custom_context_from_env("compiler_version", "VMEMKV_CONTEXT_compiler_version");
  benchmark::AddCustomContext("insert_time_budget_seconds", std::to_string(insert_time_budget_seconds()));
  benchmark::AddCustomContext("delete_time_budget_seconds", std::to_string(delete_time_budget_seconds()));
  benchmark::AddCustomContext("ycsb_window_seconds", std::to_string(get_ycsb_duration_seconds()));
  benchmark::AddCustomContext("warmup_policy",
                              "no warmup discard; corpus pre-populated on the same host, LTM cells cgroup-isolated");
  benchmark::AddCustomContext("t1_append_capacity_log2", std::to_string(vmemkv::Config<>::kAppendCapacityLog2));
  benchmark::AddCustomContext("t1_append_capacity_entries", std::to_string(vmemkv::Config<>::kAppendCapacityEntries));
  benchmark::AddCustomContext("t1_shard_target_size_entries",
                              std::to_string(vmemkv::Config<>::kShardTargetSizeEntries));
  benchmark::AddCustomContext("t1_shard_split_threshold_percent",
                              std::to_string(vmemkv::Config<>::kShardSplitThresholdPercent));
}

static inline bool prefer_large_value_first() { return get_env_flag01("VMEMKV_BENCH_LARGE_VALUE_FIRST", false); }
// Bounded wall-clock loop budget shared by Insert/Delete/YCSB-E.
static inline double time_budget_seconds(const char *env_name, double default_seconds) {
  if (const auto override_seconds = env_positive_double(env_name)) {
    return *override_seconds;
  }
  return default_seconds;
}
// Insert/Delete/YCSB-E have no min-time-adaptive path via iteration counting (a fixed count
// baked in for one engine's latency badly misses elsewhere -- e.g. LMDB Insert cost grows with
// tree depth and would run 26-42s against a 5s target). They run bounded wall-clock loops
// instead, so every engine gets the same wall-clock cap regardless of per-op cost.
static inline double insert_time_budget_seconds() {
  return time_budget_seconds("VMEMKV_BENCH_INSERT_TIME_BUDGET_SECONDS", 5.0);
}

// Delete runs a bounded wall-clock loop (same technique as YCSB-E): deleting keys until the
// time budget or the corpus exhausts. Caps worst-case cell time regardless of degradation
// (e.g. LMDB/1KB falls 212K to 5K deletes/sec across corpus sizes), while still populating the
// full corpus so LTM keeps genuine memory pressure (a small delete-only corpus would fit in
// budget and stop being "larger than memory").
static inline double delete_time_budget_seconds() {
  return time_budget_seconds("VMEMKV_BENCH_DELETE_TIME_BUDGET_SECONDS", 5.0);
}

static auto set_benchmark_metadata(benchmark::State &state, const bench_help::BenchmarkMetadata &meta) -> void {
  state.counters["CorpusKeys"] =
      benchmark::Counter(static_cast<double>(meta.corpus_keys), benchmark::Counter::kAvgThreads);
  state.counters["CorpusBytes"] =
      benchmark::Counter(static_cast<double>(meta.corpus_bytes), benchmark::Counter::kAvgThreads);
}
}  // namespace

// Always starts from a genuinely wiped, empty store: the T2 file, WAL, manifest, and the T1/T2
// checkpoint files (path + ".t1chk" / ".t2chk"). Used for rival backends (their own on-disk
// format has no relationship to vmemkv's checkpoint reuse below) and for scenarios that genuinely
// need a fresh/empty corpus every time -- Insert (measures inserting into an empty store).
// Leaving the manifest/checkpoint files behind would let a "fresh" store silently fast-boot from
// an unrelated scenario's leftover checkpoint whenever it happened to share this same unscoped
// path and had called checkpoint() -- corrupting later scenarios sharing that path (a
// fresh-looking Insert silently rejecting every key as already-present, since it isn't actually
// empty).
//
// `extra_spared_prefixes`/`sweep_other_stale_paths` default to "spare nothing extra, always
// sweep" for the ordinary (Insert non-LTM) caller, whose own files are genuinely
// one-generation-at-a-time even across a whole measurement pass and so deliberately ignore
// VMEMKV_BENCH_SKIP_CLEANUP (see should_skip_cleanup()'s comment). make_vmemkv_clone_from_
// checkpoint() overrides both instead: its own in-process master registry as
// `extra_spared_prefixes`, so building this path's master doesn't sweep away a sibling store
// variant's master this same process already primed, and `!should_skip_cleanup()` as
// `sweep_other_stale_paths`, for parity with rivals' cleanup_before_master_build_if_needed() --
// this path's own wipe/rebuild of `path` itself (below) always happens regardless.
template <typename Constructor>
static auto make_vmemkv_fresh(const std::string &path,
                              Constructor &&constructor,
                              std::set<std::string> extra_spared_prefixes = {},
                              bool sweep_other_stale_paths = true) {
  // See cleanup_stale_benchmark_files()'s comment: every fresh (non-reuse) construction also
  // sweeps away whatever other variant/value-size's files are still sitting on disk, so at most
  // one generation ever accumulates at a time.
  if (sweep_other_stale_paths) {
    extra_spared_prefixes.insert(std::filesystem::path(path).filename().string());
    cleanup_stale_benchmark_files(extra_spared_prefixes);
  }

  vmemkv::remove_store_files(path);

  return constructor();
}

// Copies only `[0, live_bytes)` of `source` into `dest` (creating/truncating `dest` first),
// preserving `source`'s full logical size as a sparse hole beyond that point instead of
// materializing it. Needed because a T2 checkpoint file's logical size is its *capacity*
// (kDefaultCapacityBytes, currently 1 TiB -- see load_checkpoint_if_present()'s fstat()-based
// capacity detection, which relies on the file's apparent size matching what the store was
// originally constructed with), almost all of which is an untouched sparse hole beyond
// `live_bytes` for any real corpus. std::filesystem::copy_file() has no notion of holes and
// copies the full logical extent byte-for-byte, turning a multi-GB-real/1-TiB-logical sparse
// master into a 1-TiB-real, fully-allocated clone -- easily exhausting disk space on a real host.
static void copy_t2_checkpoint_sparse(const std::filesystem::path &source,
                                      const std::filesystem::path &dest,
                                      uint64_t live_bytes) {
  const vmemkv::FdGuard src_guard{vmemkv::open_file(source.c_str(), O_RDONLY, "Failed to open T2 checkpoint source")};
  const int src_fd = src_guard.fd;
  const auto logical_size = vmemkv::fstat_size(src_fd, "Failed to stat T2 checkpoint source for clone");

  const vmemkv::FdGuard dst_guard{
      vmemkv::open_create_0600(dest.c_str(), O_RDWR | O_CREAT | O_TRUNC, "Failed to create T2 checkpoint clone")};
  const int dst_fd = dst_guard.fd;

#if defined(FICLONE)
  // Try zero-cost reflink CoW clone first. If the underlying filesystem supports CoW (e.g. XFS/Btrfs),
  // this shares the entire 1TiB sparse extent in sub-millisecond time with zero page-cache pollution.
  if (::ioctl(dst_fd, FICLONE, src_fd) == 0) {
    return;
  }
#endif

  if (::ftruncate(dst_fd, static_cast<off_t>(logical_size)) != 0) {
    throw std::runtime_error("Failed to size T2 checkpoint clone: " + dest.string());
  }

  // Fast in-kernel copy via copy_file_range()
  loff_t off_in = 0;
  loff_t off_out = 0;
  size_t remaining = live_bytes;
  bool copy_file_range_failed = false;
  while (remaining > 0) {
    constexpr size_t kChunkBytes = 1ULL * 1024 * 1024 * 1024;  // 1 GiB chunks
    const size_t chunk = std::min<size_t>(remaining, kChunkBytes);
    const ssize_t copied = ::copy_file_range(src_fd, &off_in, dst_fd, &off_out, chunk, 0);
    if (copied > 0) {
      remaining -= static_cast<size_t>(copied);
      continue;
    }
    if (copied < 0 && (errno == EXDEV || errno == ENOSYS || errno == EINVAL || errno == EOPNOTSUPP)) {
      copy_file_range_failed = true;
      break;
    }
    if (copied == 0) {
      break;  // Source live region exhausted
    }
    throw std::runtime_error("copy_file_range failed for T2 clone: " + dest.string());
  }

  if (!copy_file_range_failed) {
    return;
  }

  // Fallback to user-space pread/pwrite
  std::vector<std::byte> buffer(4ULL * 1024 * 1024);
  uint64_t offset = static_cast<uint64_t>(off_in);
  while (offset < live_bytes) {
    const size_t chunk = static_cast<size_t>(std::min<uint64_t>(buffer.size(), live_bytes - offset));
    const ssize_t bytes_read = ::pread(src_fd, buffer.data(), chunk, static_cast<off_t>(offset));
    if (bytes_read < 0) {
      throw std::runtime_error("Failed to read T2 checkpoint for clone: " + source.string());
    }
    if (bytes_read == 0) {
      break;  // Live region ends before live_bytes (e.g. a still-sparse tail) -- nothing more to copy.
    }
    if (::pwrite(dst_fd, buffer.data(), static_cast<size_t>(bytes_read), static_cast<off_t>(offset)) != bytes_read) {
      throw std::runtime_error("Failed to write T2 checkpoint clone: " + dest.string());
    }
    offset += static_cast<uint64_t>(bytes_read);
  }
}

// Builds (once) a vmemkv checkpoint at `master_path` -- a real bench_help::populate() + checkpoint() -- if
// one isn't already there, then clones it into a fresh instance path and writing a matching
// manifest, with *no* WAL at the new path -- so constructing a VMemKVImpl there fast-boots
// straight from the checkpoint with nothing to replay.
//
// The T1 checkpoint file is hardlinked (checkpoint.hpp's ShardedT1CheckpointWriter: always written
// to a temp path and rename()'d onto the final one, so a later cycle on the clone replaces the
// clone's directory entry with a fresh inode rather than mutating the one still shared with the
// master -- exactly like RocksDB's SST files, see RocksDBStore::clone_from()'s comment for the
// same reasoning). The T2 checkpoint file cannot use the same trick: checkpoint_internal()
// durabilizes T2's tail via pwrite() directly into the persistent file, in place, so a hardlinked
// T2 file would let the clone's own later checkpoint cycles corrupt the master (and any other
// clone sharing that inode) -- it's copied instead (see copy_t2_checkpoint_sparse() above).
//
// Post-clone quiesce, standard on LTM passes (VMEMKV_BENCH_QUIESCE_AFTER_CLONE=1,
// opt out with =0): the T2 sparse copy above streams gigabytes through the page cache
// and leaves gigabytes dirty, so a benchmark starting immediately after measures an
// indeterminate mix of draining writeback and timing-dependent cache retention -- the
// confirmed cause of run-to-run write variance under cgroup pressure. sync() drains
// the copy's writeback, drop_caches() restores a deterministic cold cache, and a short
// settle lets kworker/kswapd go idle. Best-effort (drop_caches needs privilege; local
// runs without it simply skip that step) and off by default.
static void quiesce_after_clone() {
  if (!get_env_flag01("VMEMKV_BENCH_QUIESCE_AFTER_CLONE", false)) {
    return;
  }
  ::sync();
  if (std::ofstream drop_caches("/proc/sys/vm/drop_caches"); drop_caches.good()) {
    drop_caches << 3 << std::flush;
  }
  const auto settle = std::chrono::seconds(get_env_uint64("VMEMKV_BENCH_QUIESCE_SECS", 3));
  std::this_thread::sleep_for(settle);
}

template <typename StorePtr>
static void wait_for_background_compaction(StorePtr &store) {
  store->drain_compaction();
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(10);
  const auto pending = [&] {
    return store->get_statistics().t2_dirty_shard_count != 0 || store->impl().pending_segment_count() != 0 ||
           store->impl().compaction_running();
  };
  while (pending() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  if (pending()) {
    throw std::runtime_error("background compaction did not drain before timeout");
  }
}

// This is what lets Get/Update/Delete/YCSB-E/Scan (see their registrations below) all get a
// fresh, fully-populated, fully-reorganized instance on every construction without paying
// bulk_load's cost more than once per (variant, val_size) -- and, since they all pass the same
// (val_size, key_count), they transparently share that one on-disk master rather than each
// building their own. Every clone here is independent and starts with no WAL of its own, so a
// later mutation (e.g. Delete removing keys, or YCSB-E's insert mix growing the corpus) on one
// clone never touches the master or any sibling clone taken from it.
template <typename Store>
static auto make_vmemkv_clone_from_checkpoint(const std::string &master_path,
                                              std::size_t val_size,
                                              std::size_t key_count,
                                              bool shuffled = false) -> std::unique_ptr<Store> {
  const std::filesystem::path master_manifest_path = vmemkv::derive_manifest_path(master_path);
  auto manifest = vmemkv::read_manifest(master_manifest_path);
  if (!manifest.has_value()) {
    // First time for this (variant, val_size): build the master for real, then discard the
    // in-memory instance -- only the checkpoint files it leaves behind on disk matter from here
    // on. register_primed_master_prefix() spares every master this process has already primed
    // (spanning other store variants sharing this same run -- see its comment) from
    // make_vmemkv_fresh()'s sweep, so building this one doesn't wipe them back out; should_skip_
    // cleanup() additionally suppresses that sweep entirely, for parity with rivals' cleanup_
    // before_master_build_if_needed() -- see should_skip_cleanup()'s comment.
    auto master_store = make_vmemkv_fresh(
        master_path,
        [master_path]() { return std::make_unique<Store>(master_path, Store::ConfigType::kDefaultCapacityBytes); },
        register_primed_master_prefix(master_path),
        !should_skip_cleanup());
    bench_help::populate(*master_store, {key_count, val_size}, shuffled);
    master_store->impl().t1().drain_maintenance();
    if (shuffled) {
      master_store->impl().checkpoint();
      wait_for_background_compaction(master_store);
      master_store->impl().checkpoint();
      master_store->impl().checkpoint();
      wait_for_background_compaction(master_store);
    } else {
      master_store->impl().checkpoint();
    }
    master_store.reset();
    manifest = vmemkv::read_manifest(master_manifest_path);
    if (!manifest.has_value()) {
      throw std::runtime_error("vmemkv master checkpoint build did not produce a valid manifest: " + master_path);
    }
  }

  // Fixed path, not an ever-incrementing counter: g_active_store_holder (see StoreHolder below)
  // is a single global slot, so at most one clone of any given master is ever live process-wide
  // -- switching holders always resets the previous one before this runs again. Reusing one path
  // means every subsequent clone's own remove-then-rebuild below (already required, to discard
  // whatever the *previous* master-build or clone left there) also bounds this master's cumulative
  // on-disk clone footprint to one clone's worth instead of accumulating a fresh multi-GB copy per
  // benchmark case for the rest of the process's run.
  const std::string instance_path = master_path + "_clone";
  std::error_code ignored;
  std::filesystem::remove(instance_path, ignored);
  vmemkv::remove_wal_segments(vmemkv::derive_wal_path(instance_path));
  std::filesystem::remove(vmemkv::derive_manifest_path(instance_path), ignored);

  const auto master_t1 = vmemkv::derive_t1_chk_path(master_path);
  const auto master_t2 = vmemkv::derive_t2_chk_path(master_path);
  const auto instance_t1 = vmemkv::derive_t1_chk_path(instance_path);
  const auto instance_t2 = vmemkv::derive_t2_chk_path(instance_path);
  std::filesystem::remove(instance_t1, ignored);
  std::filesystem::remove(instance_t2, ignored);
  std::error_code link_error;
  std::filesystem::create_hard_link(master_t1, instance_t1, link_error);
  if (link_error) {
    throw std::runtime_error("Failed to hardlink T1 checkpoint for clone: " + link_error.message());
  }
  copy_t2_checkpoint_sparse(master_t2, instance_t2, manifest->t2_bytes_used);
  vmemkv::write_manifest(vmemkv::derive_manifest_path(instance_path), manifest->generation, manifest->t2_bytes_used);

  quiesce_after_clone();

  return std::make_unique<Store>(instance_path, Store::ConfigType::kDefaultCapacityBytes);
}

template <typename Tuple, typename Visitor, std::size_t... Indices>
void for_each_in_tuple_impl(Visitor &&visitor, std::index_sequence<Indices...> index_sequence_unused) {
  (void)index_sequence_unused;
  (visitor(static_cast<std::tuple_element_t<Indices, Tuple> *>(nullptr)), ...);
}

template <typename Tuple, typename Visitor>
void for_each_in_tuple(Visitor &&visitor) {
  for_each_in_tuple_impl<Tuple>(std::forward<Visitor>(visitor), std::make_index_sequence<std::tuple_size_v<Tuple>>{});
}

using BenchmarkTypes = vmemkv::variants::AllPossibleTypes;

// Rival backends take a bare path (they manage their own on-disk capacity/layout);
// only vmemkv configurations take the extra T2 capacity argument.
template <typename Store>
inline constexpr bool kUsesSingleArgConstructor = vmemkv::detail::is_rival_store_v<typename Store::Impl>;

// A per-(val_size, key_count) master-corpus path shared by every scenario in visit_one() below
// that wants a fully-populated, fully-reorganized/committed corpus (Get/Update/Delete/YCSB-E/
// Scan/Insert's LTM pre-populate): as long as they pass the same (val_size, key_count), they all
// clone from -- and, the first time, jointly pay to build -- exactly one on-disk master, instead
// of each independently paying its own real-populate cost. Not used by Insert's non-LTM path
// (measures inserting into a genuinely empty store, so there is nothing to share) -- that one
// keeps using make_bench_store_fresh() unchanged.
static auto bench_master_corpus_path(const std::string &filename,
                                     std::size_t val_size,
                                     std::size_t key_count,
                                     bool shuffled = false) -> std::string {
  return filename + "_v" + std::to_string(val_size) + "_n" + std::to_string(key_count) + (shuffled ? "_shuffled" : "") +
         "_master";
}

template <typename Store>
static auto make_bench_store_fresh(const std::string &filename) {
  return make_vmemkv_fresh(filename, [filename]() {
    if constexpr (kUsesSingleArgConstructor<Store>) {
      return std::make_unique<Store>(filename);
    } else {
      return std::make_unique<Store>(filename, Store::ConfigType::kDefaultCapacityBytes);
    }
  });
}

// The rival-backend half of make_bench_fresh_corpus_checkpoint() below: rivals have no
// vmemkv-specific checkpoint/manifest concept, so their "clone from a shared master" mechanism is
// instead CloneFromMasterTag, driven off bench_master_corpus_path() directly. vmemkv itself never
// takes this branch (see make_bench_fresh_corpus_checkpoint()'s own branch). Takes key_count
// explicitly (rather than deriving it from val_size) so callers whose corpus size doesn't match
// corpus_size_for_value() still get a correctly-sized master instead of silently
// reusing/misreporting a mismatched one.
template <typename Store>
static auto make_bench_fresh_corpus(const std::string &filename,
                                    std::size_t val_size,
                                    std::size_t key_count,
                                    bool shuffled = false) {
  if constexpr (kUsesSingleArgConstructor<Store>) {
    const std::string master_path = bench_master_corpus_path(filename, val_size, key_count, shuffled);
    cleanup_before_master_build_if_needed(master_path);
    if (!shuffled) {
      return std::make_unique<Store>(
          typename Store::Impl::CloneFromMasterTag{},
          master_path,
          key_count,
          [](std::size_t index) { return bench_help::make_key(index); },
          [val_size](std::size_t index) { return bench_help::make_value_for_key(index, val_size); });
    }
    // Same logical key set in fixed-seed-shuffled insert order, so the on-disk layout ends up
    // scattered relative to key order. Deterministic, so rebuilds coincide.
    const auto order = std::make_shared<std::vector<std::size_t>>(bench_help::shuffled_order(key_count));
    return std::make_unique<Store>(
        typename Store::Impl::CloneFromMasterTag{},
        master_path,
        key_count,
        [order](std::size_t index) { return bench_help::make_key((*order)[index]); },
        [order, val_size](std::size_t index) { return bench_help::make_value_for_key((*order)[index], val_size); });
  } else {
    return make_vmemkv_fresh(
        filename, [filename]() { return std::make_unique<Store>(filename, Store::ConfigType::kDefaultCapacityBytes); });
  }
}

// Get/Update/Delete/YCSB-E/Scan all want the same thing -- a fresh, fully-populated,
// committed instance, cheaply, on every construction -- so vmemkv clones the shared checkpoint
// (rivals: CloneFromMasterTag off bench_master_corpus_path()). Same (val_size, key_count) shares
// one on-disk master, built at most once per (variant, val_size). The base-region read handles
// ride along on every fresh construction, including checkpoint adopts
// (see load_checkpoint_if_present()).
template <typename Store>
static auto make_bench_fresh_corpus_checkpoint(const std::string &filename,
                                               std::size_t val_size,
                                               std::size_t key_count,
                                               bool shuffled = false) {
  if constexpr (kUsesSingleArgConstructor<Store>) {
    return make_bench_fresh_corpus<Store>(filename, val_size, key_count, shuffled);
  } else {
    // Scoped by (val_size, key_count), same as the rival masters above (see
    // bench_master_corpus_path()'s comment) -- different value sizes (or a YCSB_E_POPULATE
    // override) must never collide on the same master checkpoint.
    return make_vmemkv_clone_from_checkpoint<Store>(
        bench_master_corpus_path(filename, val_size, key_count, shuffled), val_size, key_count, shuffled);
  }
}

template <typename Store, typename Visitor>
static void visit_one_store(Visitor &&visitor) {
  if constexpr (Store::kIsEnabled) {
    std::string filename = get_db_dir() + "/bench_" + Store::name() + ".bin";
    std::replace(filename.begin() + get_db_dir().size() + 1, filename.end(), '/', '_');

    auto make = [filename]() { return make_bench_store_fresh<Store>(filename); };
    auto make_fresh_corpus_checkpoint = [filename](std::size_t val_size, std::size_t key_count, bool shuffled = false) {
      return make_bench_fresh_corpus_checkpoint<Store>(filename, val_size, key_count, shuffled);
    };

    visitor(Store::name().c_str(), make, make_fresh_corpus_checkpoint);
  }
}

template <typename Visitor>
static void for_each_store_variant(Visitor &&visitor) {
  for_each_in_tuple<BenchmarkTypes>([&](auto *dummy) {
    using Store = std::remove_pointer_t<decltype(dummy)>;
    if constexpr (Store::kIsEnabled && std::is_same_v<Store, vmemkv::variants::VMemKV_RocksDB>) {
      visit_one_store<Store>(visitor);
    }
  });

  for_each_in_tuple<BenchmarkTypes>([&](auto *dummy) {
    using Store = std::remove_pointer_t<decltype(dummy)>;
    if constexpr (Store::kIsEnabled && !std::is_same_v<Store, vmemkv::variants::VMemKV_RocksDB>) {
      visit_one_store<Store>(visitor);
    }
  });
}

// Read-policy pins: Get Hit Zipf + Scan (both dists) only, unlike the full-matrix variants above.
// Benchmark names, masters, and filters flow through the same helpers.
template <typename Visitor>
static void for_each_read_pin_variant(Visitor &&visitor) {
  for_each_in_tuple<vmemkv::variants::ReadPinTypes>([&](auto *dummy) {
    using Store = std::remove_pointer_t<decltype(dummy)>;
    visit_one_store<Store>(visitor);
  });
}

// =============================================================================
// Helper for managing store lifecycle across threads in Google Benchmark
// =============================================================================
struct StoreHolderBase {
  virtual ~StoreHolderBase() = default;
  virtual void reset_store() = 0;
};

template <typename StorePtr>
struct StoreHolder : public StoreHolderBase {
  std::mutex mutex;
  StorePtr store;
  int active_threads = 0;

  void reset_store() override {
    std::lock_guard<std::mutex> lock(mutex);
    store.reset();
  }
};

static std::mutex g_active_store_mutex;
static std::shared_ptr<StoreHolderBase> g_active_store_holder = nullptr;

static inline bool should_log_phase_steps(const std::string &name) {
  return name.find("/Op=Insert/") != std::string::npos || name.find("/Op=Delete/") != std::string::npos;
}

template <typename StorePtr, typename MakeStore, typename InitFn>
static void ensure_store_ready(std::shared_ptr<StoreHolder<StorePtr>> holder_ptr,
                               MakeStore &make,
                               InitFn &init_fn,
                               const std::string &name,
                               benchmark::State &state,
                               std::optional<double> &populate_seconds) {
  {
    std::lock_guard<std::mutex> lock(g_active_store_mutex);
    if (g_active_store_holder && g_active_store_holder != holder_ptr) {
      g_active_store_holder->reset_store();
    }
    g_active_store_holder = holder_ptr;
  }

  StoreHolder<StorePtr> &holder = *holder_ptr;
  std::lock_guard<std::mutex> lock(holder.mutex);
  if (!holder.store) {
    if (should_log_phase_steps(name)) {
      std::cout << "[phase] start init benchmark=" << name << " threads=" << state.threads() << std::endl;
    }
    // Started before make() (not just init_fn) so Populate_Sec reflects the full cost of
    // getting a ready-to-measure store, including construction itself -- the make() passed in
    // here (see for_each_store_variant() in this file) can do real work there (a one-time
    // master bulk_load, or a clone from it; see rivals/*.hpp's CloneFromMasterTag constructors);
    // starting the timer only after make() returns would leave that work completely unmeasured.
    auto init_start = std::chrono::steady_clock::now();
    holder.store = make();
    init_fn(*(holder.store));
    auto init_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> init_elapsed = init_end - init_start;
    populate_seconds = init_elapsed.count();
    if (should_log_phase_steps(name)) {
      std::cout << "[phase] end init benchmark=" << name << " populate_sec=" << init_elapsed.count() << std::endl;
    }
  }
  holder.active_threads++;
}

template <typename StorePtr>
static void record_store_statistics(benchmark::State &state, StoreHolder<StorePtr> &holder) {
  if (state.thread_index() != 0) {
    return;
  }
  auto stats = holder.store->get_statistics();
  state.counters["T1_Splits"] =
      benchmark::Counter(static_cast<double>(stats.t1_split_count), benchmark::Counter::kDefaults);
  state.counters["Checkpoints"] =
      benchmark::Counter(static_cast<double>(stats.checkpoint_count), benchmark::Counter::kDefaults);
  state.counters["Reorganize_Wait_Duration_us"] =
      benchmark::Counter(static_cast<double>(stats.total_reorganize_wait_duration_us), benchmark::Counter::kDefaults);
  state.counters["Append_Region_Live"] =
      benchmark::Counter(static_cast<double>(stats.append_region_live_count), benchmark::Counter::kDefaults);
  state.counters["Append_Region_Peak"] =
      benchmark::Counter(static_cast<double>(stats.append_region_peak_count), benchmark::Counter::kDefaults);
  state.counters["Compaction_Cycles"] =
      benchmark::Counter(static_cast<double>(stats.compaction_cycle_count), benchmark::Counter::kDefaults);
  state.counters["T2_Dirty_Shards"] =
      benchmark::Counter(static_cast<double>(stats.t2_dirty_shard_count), benchmark::Counter::kDefaults);
  state.counters["Throttle_Events"] =
      benchmark::Counter(static_cast<double>(stats.bulk_load_throttle_events), benchmark::Counter::kDefaults);
  const uint64_t samples = vmemkv::write_path_debug::g_sample_count.load(std::memory_order_relaxed);
  if (samples > 0) {
    state.counters["Write_Lock_Avg_ns"] = benchmark::Counter(
        static_cast<double>(vmemkv::write_path_debug::g_stripe_lock_ns.load(std::memory_order_relaxed)) /
            static_cast<double>(samples),
        benchmark::Counter::kDefaults);
    state.counters["Write_Mutation_Avg_ns"] = benchmark::Counter(
        static_cast<double>(vmemkv::write_path_debug::g_mutation_ns.load(std::memory_order_relaxed)) /
            static_cast<double>(samples),
        benchmark::Counter::kDefaults);
    state.counters["Write_Await_Avg_ns"] = benchmark::Counter(
        static_cast<double>(vmemkv::write_path_debug::g_await_durable_ns.load(std::memory_order_relaxed)) /
            static_cast<double>(samples),
        benchmark::Counter::kDefaults);
  }
  const uint64_t fsync_count = vmemkv::wal_debug::g_fsync_count.load(std::memory_order_relaxed);
  if (fsync_count > 0) {
    state.counters["Wal_Fsync_Avg_ns"] =
        benchmark::Counter(static_cast<double>(vmemkv::wal_debug::g_fsync_ns.load(std::memory_order_relaxed)) /
                               static_cast<double>(fsync_count),
                           benchmark::Counter::kDefaults);
    state.counters["Wal_Batch_Avg_Size"] =
        benchmark::Counter(static_cast<double>(vmemkv::wal_debug::g_batch_records.load(std::memory_order_relaxed)) /
                               static_cast<double>(fsync_count),
                           benchmark::Counter::kDefaults);
  }
}

template <typename StorePtr>
static void release_store(StoreHolder<StorePtr> &holder,
                          bool destroy_store_after_benchmark,
                          std::optional<double> &reset_seconds) {
  std::lock_guard<std::mutex> lock(holder.mutex);
  holder.active_threads--;
  if (holder.active_threads != 0 || !destroy_store_after_benchmark) {
    return;
  }

  auto reset_start = std::chrono::steady_clock::now();
  holder.store.reset();
  auto reset_end = std::chrono::steady_clock::now();
  std::chrono::duration<double> reset_elapsed = reset_end - reset_start;
  reset_seconds = reset_elapsed.count();
}

template <typename StorePtr, typename RunFn>
static void execute_benchmark_run(StoreHolder<StorePtr> &holder,
                                  RunFn &run_fn,
                                  const std::string &name,
                                  benchmark::State &state) {
  if (state.thread_index() == 0) {
    vmemkv::write_path_debug::reset();
    vmemkv::wal_debug::reset();
    if (should_log_phase_steps(name)) {
      std::cout << "[phase] start run benchmark=" << name << " threads=" << state.threads() << std::endl;
    }
  }

  auto start = std::chrono::steady_clock::now();
  run_fn(state, *(holder.store));
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  state.counters["Elapsed_Sec"] = benchmark::Counter(elapsed.count(), benchmark::Counter::kAvgThreads);

  if (should_log_phase_steps(name) && state.thread_index() == 0) {
    std::cout << "[phase] end run benchmark=" << name << " elapsed_sec=" << elapsed.count() << std::endl;
  }
}

template <typename StorePtr, typename MakeStore, typename InitFn, typename RunFn>
static void register_bench(std::shared_ptr<StoreHolder<StorePtr>> holder,
                           const std::string &name,
                           const bench_help::BenchmarkMetadata &metadata,
                           MakeStore make,
                           InitFn &&init_fn,
                           RunFn &&run_fn,
                           const std::vector<int> &thread_counts,
                           bool destroy_store_after_benchmark = true) {
  auto init_fn_copy = std::forward<InitFn>(init_fn);
  auto run_fn_copy = std::forward<RunFn>(run_fn);

  for (int threads : thread_counts) {
    auto populate_seconds = std::make_shared<std::optional<double>>();
    auto reset_seconds = std::make_shared<std::optional<double>>();

    auto bench_func = [holder,
                       name,
                       make,
                       metadata,
                       init_fn_copy,
                       run_fn_copy,
                       destroy_store_after_benchmark,
                       populate_seconds,
                       reset_seconds](benchmark::State &state) {
      ensure_store_ready(holder, make, init_fn_copy, name, state, *populate_seconds);
      set_benchmark_metadata(state, metadata);
      execute_benchmark_run(*holder, run_fn_copy, name, state);
      record_store_statistics(state, *holder);
      release_store(*holder, destroy_store_after_benchmark, *reset_seconds);

      if (state.thread_index() == 0) {
        state.counters["Populate_Sec"] =
            benchmark::Counter(populate_seconds->value_or(0.0), benchmark::Counter::kDefaults);
        state.counters["Reset_Sec"] = benchmark::Counter(reset_seconds->value_or(0.0), benchmark::Counter::kDefaults);
      }
    };

    auto *reg = benchmark::RegisterBenchmark(name.c_str(), bench_func)->Threads(threads)->UseRealTime();
    if (name.find("/Op=YCSB-E/") != std::string::npos || name.find("/Op=Delete/") != std::string::npos ||
        name.find("/Op=Insert/") != std::string::npos) {
      // YCSB-E, Delete, and Insert all enforce their own wall-clock window internally.
      // Keep Google Benchmark to a single execution so it does not add another
      // adaptive MinTime pass on top of the benchmark's own timing.
      reg->Iterations(1);
    }
  }
}

template <typename MakeStore, typename InitFn, typename RunFn>
static void register_bench(const std::string &name,
                           const bench_help::BenchmarkMetadata &metadata,
                           MakeStore make,
                           InitFn &&init_fn,
                           RunFn &&run_fn,
                           const std::vector<int> &thread_counts) {
  auto holder = std::make_shared<StoreHolder<decltype(make())>>();
  register_bench(
      holder, name, metadata, make, std::forward<InitFn>(init_fn), std::forward<RunFn>(run_fn), thread_counts, true);
}

// YCSB-E epoch join: thread 0 opens a new epoch, other threads wait for it.
template <typename YCSBStateT>
static uint64_t ycsb_join_epoch(YCSBStateT &st,
                                std::size_t thread_idx,
                                uint64_t my_last_epoch,
                                int num_threads,
                                uint64_t populate_size,
                                std::size_t val_size) {
  uint64_t local_epoch = 0;
  if (thread_idx == 0) {
    local_epoch = st.current_epoch.load(std::memory_order_relaxed) + 1;
    st.collector = std::make_unique<YCSBTimelineCollector>(num_threads, populate_size, val_size);
    st.ready_threads.store(0, std::memory_order_release);
    st.warmup_finished_threads.store(0, std::memory_order_release);
    st.current_epoch.store(local_epoch, std::memory_order_release);
    st.init_done_epoch.store(local_epoch, std::memory_order_release);
  } else {
    while (st.init_done_epoch.load(std::memory_order_acquire) <= my_last_epoch) {
      std::this_thread::yield();
    }
    local_epoch = st.init_done_epoch.load(std::memory_order_relaxed);
  }
  if (thread_idx < 128) {
    st.thread_last_epochs[thread_idx].store(local_epoch, std::memory_order_relaxed);
  }
  return local_epoch;
}

// YCSB-E start barrier: all threads rendezvous, thread 0 starts the timeline.
template <typename YCSBStateT, typename StoreT>
static void ycsb_barrier_start(YCSBStateT &st,
                               StoreT &store,
                               YCSBTimelineCollector *col,
                               std::size_t thread_idx,
                               uint64_t local_epoch,
                               int num_threads) {
  st.ready_threads.fetch_add(1, std::memory_order_acq_rel);
  if (thread_idx == 0) {
    if (st.start_done_epoch.load(std::memory_order_relaxed) < local_epoch) {
      while (st.ready_threads.load(std::memory_order_acquire) < static_cast<std::size_t>(num_threads)) {
        std::this_thread::yield();
      }
      col->start();
      auto stats = store.get_statistics();
      col->last_recorded_checkpoint_t2.store(stats.checkpoint_count, std::memory_order_relaxed);
      col->last_recorded_compaction_cycles.store(stats.compaction_cycle_count, std::memory_order_relaxed);
      st.start_done_epoch.store(local_epoch, std::memory_order_release);
    }
  } else {
    while (st.start_done_epoch.load(std::memory_order_acquire) < local_epoch) {
      std::this_thread::yield();
    }
  }
}

// YCSB-E 95/5 scan/insert step.
template <typename StoreT>
static void ycsb_do_mixed_op(StoreT &store,
                             YCSBTimelineCollector *col,
                             std::mt19937_64 &rng,
                             std::uniform_int_distribution<int> &op_dist,
                             std::size_t thread_idx,
                             std::size_t val_size,
                             const std::string &dummy_large,
                             const std::string &dummy_8b) {
  int op_choice = op_dist(rng);
  if (op_choice < 95) {
    uint64_t max_keys = col->next_key_index.load(std::memory_order_relaxed);
    bench_help::ZipfDistribution dynamic_zipf({max_keys > kScanLength ? max_keys - kScanLength : 1, 1.0});
    uint64_t scan_start = dynamic_zipf(rng);
    size_t scan_bytes = 0;
    size_t result_count = store.scan(bench_help::make_key(scan_start),
                                     bench_help::make_key(scan_start + kScanLength - 1),
                                     [&](std::span<const std::byte> k, std::span<const std::byte> value) {
                                       scan_bytes += k.size() + value.size();
                                       benchmark::DoNotOptimize(bench_help::touch_bytes(value));
                                     });
    benchmark::DoNotOptimize(result_count);
    col->record_op(thread_idx, true, scan_bytes);
  } else {
    uint64_t next_idx = col->next_key_index.fetch_add(1, std::memory_order_relaxed);
    bool inserted;
    size_t insert_bytes = 0;
    if (use_mixed_8b_value(val_size, next_idx)) {
      inserted = store.insert(bench_help::make_key(next_idx), dummy_8b);
      insert_bytes = bench_help::kIndexKeyBytes + dummy_8b.size();
    } else {
      inserted = store.insert(bench_help::make_key(next_idx), dummy_large);
      insert_bytes = bench_help::kIndexKeyBytes + dummy_large.size();
    }
    benchmark::DoNotOptimize(inserted);
    col->record_op(thread_idx, false, insert_bytes);
  }
}

// YCSB-E Benchmark (Short Range Scans, 30s mixed workload, hw_threads threads max), registered
// once per (store variant, value size) by register_all_benchmarks() below.
template <typename Holder, typename MakeFreshCorpusCheckpoint>
static void register_ycsb_e_benchmark(Holder ycsb_holder,
                                      int hw_threads,
                                      const std::string &sname,
                                      const std::string &value_name,
                                      MakeFreshCorpusCheckpoint make_fresh_corpus_checkpoint,
                                      std::size_t val_size,
                                      std::size_t corpus_size) {
  auto ycsb_meta = bench_help::make_metadata(corpus_size, val_size);
  std::vector<int> ycsb_threads = {hw_threads};

  struct YCSBState {
    std::unique_ptr<YCSBTimelineCollector> collector;
    std::mutex init_mutex;
    std::atomic<uint64_t> current_epoch{0};
    std::atomic<uint64_t> init_done_epoch{0};
    std::atomic<uint64_t> start_done_epoch{0};
    std::atomic<uint64_t> warmup_done_epoch{0};
    std::array<std::atomic<uint64_t>, 128> thread_last_epochs{};
    std::atomic<size_t> ready_threads{0};
    std::atomic<size_t> warmup_finished_threads{0};
  };
  auto ycsb_state = std::make_shared<YCSBState>();

  // YCSB-E populate size: respect YCSB_E_POPULATE env var, else use corpus_size (same scale as other benchmarks)
  const size_t ycsb_populate_size = get_env_uint64("YCSB_E_POPULATE", corpus_size);

  register_bench(
      ycsb_holder,
      bench_help::benchmark_name(sname, "YCSB-E", std::nullopt, "Zipf", value_name),
      ycsb_meta,
      [make_fresh_corpus_checkpoint, val_size, ycsb_populate_size]() {
        return make_fresh_corpus_checkpoint(val_size, ycsb_populate_size);
      },
      [ycsb_state](auto & /*store*/) {
        // Both vmemkv (checkpoint clone) and rivals (native snapshot clone) already hand back a
        // fresh, fully-populated instance above -- nothing left to initialize. Deliberately *not*
        // calling store.checkpoint() defensively: it always forces a real cycle for zero benefit
        // here. The run phase may grow the corpus past ycsb_populate_size afterward, but that
        // mutates only this clone, never the master, so later clones are unaffected.
        //
        // NOTE: collector setup and background reorg thread are launched per-run inside the
        // benchmark body (via epoch synchronization) to handle multiple trial/warmup runs correctly.
      },
      [ycsb_state, ycsb_populate_size, sname, variant_label = bench_help::variant_label(sname), value_name, val_size](
          benchmark::State &state, auto &store) {
        // NOTE: drives its own wall-clock window instead of gbench iteration counting. Still
        // wrapped in `for (auto _ : state)` (exactly once, via Iterations(1)) purely so
        // StartKeepRunning/FinishKeepRunning bracket the run -- otherwise `real_time` stays 0
        // (per-second data comes from the YCSBTimelineCollector JSON dump either way).
        for (auto _ : state) {
          std::mt19937_64 rng(bench_help::kBenchmarkSeed + state.thread_index());
          std::uniform_int_distribution<int> op_dist(0, 99);

          const size_t thread_idx = state.thread_index();
          uint64_t my_last_epoch =
              (thread_idx < 128) ? ycsb_state->thread_last_epochs[thread_idx].load(std::memory_order_relaxed) : 0;
          uint64_t local_epoch =
              ycsb_join_epoch(*ycsb_state, thread_idx, my_last_epoch, state.threads(), ycsb_populate_size, val_size);

          auto *col = ycsb_state->collector.get();
          std::string dummy_large(val_size, 'a');
          std::string dummy_8b(8, 'a');
          uint64_t local_ops = 0;
          std::size_t next_forced_trigger_idx = 0;
          const bool disable_forced = is_forced_triggers_disabled();
          const int warmup_sec = get_ycsb_warmup_seconds();

          ycsb_barrier_start(*ycsb_state, store, col, thread_idx, local_epoch, state.threads());

          if (warmup_sec > 0) {
            auto warmup_t0 = std::chrono::steady_clock::now();
            while (true) {
              auto now = std::chrono::steady_clock::now();
              auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - warmup_t0).count();
              if (elapsed >= warmup_sec) {
                break;
              }
              ycsb_do_mixed_op(store, col, rng, op_dist, thread_idx, val_size, dummy_large, dummy_8b);
            }
            ycsb_state->warmup_finished_threads.fetch_add(1, std::memory_order_acq_rel);
            if (thread_idx == 0) {
              while (ycsb_state->warmup_finished_threads.load(std::memory_order_acquire) <
                     static_cast<size_t>(state.threads())) {
                std::this_thread::yield();
              }
              col->reset_after_warmup(val_size);
              col->start();
              auto stats = store.get_statistics();
              col->last_recorded_checkpoint_t2.store(stats.checkpoint_count, std::memory_order_relaxed);
              col->last_recorded_compaction_cycles.store(stats.compaction_cycle_count, std::memory_order_relaxed);
              ycsb_state->warmup_done_epoch.store(local_epoch, std::memory_order_release);
            } else {
              while (ycsb_state->warmup_done_epoch.load(std::memory_order_acquire) < local_epoch) {
                std::this_thread::yield();
              }
            }
          }

          int last_polled_sec = -1;
          while (true) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - col->start_time).count();
            if (elapsed >= col->duration_seconds) {
              break;
            }

            // Thread 0: poll and record stats per second
            if (thread_idx == 0 && elapsed > last_polled_sec && elapsed < col->duration_seconds) {
              last_polled_sec = static_cast<int>(elapsed);
              auto stats = store.get_statistics();
              col->poll_second(last_polled_sec, get_db_dir(), stats);

              if (!disable_forced && next_forced_trigger_idx < kForcedTriggers.size() &&
                  elapsed >= kForcedTriggers[next_forced_trigger_idx].second_mark) {
                const ForcedTrigger &trigger = kForcedTriggers[next_forced_trigger_idx];
                const auto call_t0 = std::chrono::steady_clock::now();
                store.checkpoint();
                const double call_elapsed_sec =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - call_t0).count();

                const auto fired_elapsed =
                    std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - col->start_time)
                        .count();
                const int bucket = static_cast<int>(std::clamp<int64_t>(fired_elapsed, 0, col->duration_seconds - 1));

                auto post_stats = store.get_statistics();
                col->last_recorded_checkpoint_t2.store(post_stats.checkpoint_count, std::memory_order_relaxed);
                col->record_forced_checkpoint(
                    trigger.second_mark, bucket, call_elapsed_sec, post_stats.last_checkpoint_bytes_synced);
                ++next_forced_trigger_idx;
              }
            }

            ycsb_do_mixed_op(store, col, rng, op_dist, thread_idx, val_size, dummy_large, dummy_8b);
            ++local_ops;
          }

          if (thread_idx == 0) {
            auto final_stats = store.get_statistics();
            for (int s = last_polled_sec + 1; s < col->duration_seconds; ++s) {
              col->poll_second(s, get_db_dir(), final_stats);
            }
          }

          if (state.thread_index() == 0) {
            col->dump_json(sname, variant_label, value_name, is_ltm_mode() ? "ltm" : "in_memory");
          }
          state.SetItemsProcessed(local_ops);
        }  // for (auto _ : state)
      },
      ycsb_threads,
      false);
}

// Shared no-op init_fn for scenarios whose make_fn (make_corpus / make_fresh_corpus_checkpoint)
// already hands back a fresh, fully-populated, durably-committed instance -- there is nothing
// left to initialize.
inline constexpr auto noop_already_populated_init = [](auto & /*store*/) {};

// Runs body(i) until the time budget elapses; body returns false to stop early (e.g. the
// corpus slice is exhausted). Returns iterations completed; reports via SetItemsProcessed.
// CorpusExhausted is 1 when any repetition ended early (Delete draining its corpus), else 0.
template <typename Body>
static void run_wall_clock(benchmark::State &state, std::chrono::duration<double> budget, Body &&body) {
  for (auto _ : state) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::duration_cast<std::chrono::steady_clock::duration>(budget);
    std::size_t i = 0;
    bool exhausted = false;
    while (std::chrono::steady_clock::now() < deadline) {
      if (!body(i)) {
        exhausted = true;
        break;
      }
      ++i;
    }
    state.counters["CorpusExhausted"] = exhausted ? 1.0 : 0.0;
    state.SetItemsProcessed(i);
  }
}

// Insert, registered once per (store variant, value size) by register_all_benchmarks() below. In
// LTM mode this pre-populates the scenario-sized corpus first, then measures incremental inserts
// on top of it; in in-memory mode it starts from an empty store.
template <typename Holder, typename Make, typename MakeFreshCorpusCheckpoint>
static void register_insert_benchmark(Holder insert_holder,
                                      const std::string &sname,
                                      const std::string &value_name,
                                      Make make,
                                      MakeFreshCorpusCheckpoint make_fresh_corpus_checkpoint,
                                      std::size_t val_size,
                                      std::size_t corpus_size,
                                      const std::vector<int> &thread_counts) {
  const bool ltm_insert_mode = is_ltm_mode();
  auto insert_meta = bench_help::make_metadata(corpus_size, val_size);
  register_bench(
      insert_holder,
      bench_help::benchmark_name(sname, "Insert", std::nullopt, std::nullopt, value_name),
      insert_meta,
      // LTM pre-population shares every other scenario's master (see
      // make_fresh_corpus_checkpoint()): clone from it instead of bulk_load()+checkpoint()ing once
      // per thread-count entry, so that cost is paid once (outside the cgroup, via the priming
      // pass). Non-LTM keeps `make` (genuinely empty store).
      [make, make_fresh_corpus_checkpoint, val_size, corpus_size, ltm_insert_mode]() {
        if (ltm_insert_mode) {
          return make_fresh_corpus_checkpoint(val_size, corpus_size);
        }
        return make();
      },
      noop_already_populated_init,
      [corpus_size, val_size](benchmark::State &state, auto &store) {
        std::string dummy_large(val_size, 'a');
        std::string dummy_8b(8, 'a');
        const std::size_t insert_start = is_ltm_mode() ? corpus_size : 0;
        std::size_t threads = static_cast<std::size_t>(state.threads());
        std::size_t thread_idx = static_cast<std::size_t>(state.thread_index());
        const auto time_budget = std::chrono::duration<double>(insert_time_budget_seconds());
        run_wall_clock(state, time_budget, [&](std::size_t i) {
          std::size_t key_index = insert_start + thread_idx + i * threads;
          bool inserted;
          if (use_mixed_8b_value(val_size, key_index)) {
            inserted = store.insert(bench_help::make_key(key_index), dummy_8b);
          } else {
            inserted = store.insert(bench_help::make_key(key_index), dummy_large);
          }
          benchmark::DoNotOptimize(inserted);
          return true;
        });
      },
      thread_counts);
}

// Get/Hit (all Dist values) and Get/Miss, registered once per (store variant, value size) by
// register_all_benchmarks() below. Both are read-only against the shared corpus `make_corpus`
// clones. dists/include_miss narrow the registration for read-policy pins, which compare
// Get/Hit/Zipf only; the default registers everything.
template <typename Holder, typename MakeCorpus>
static void register_get_benchmarks(Holder crud_holder,
                                    const std::string &sname,
                                    const std::string &value_name,
                                    MakeCorpus make_corpus,
                                    std::size_t val_size,
                                    std::size_t corpus_size,
                                    const std::vector<int> &thread_counts,
                                    std::initializer_list<const char *> dists = {"Zipf", "Uniform"},
                                    bool include_miss = true) {
  for (const char *dist : dists) {
    auto get_hit_meta = bench_help::make_metadata(corpus_size, val_size);
    register_bench(
        crud_holder,
        bench_help::benchmark_name(sname, "Get", "Hit", dist, value_name),
        get_hit_meta,
        make_corpus,
        noop_already_populated_init,
        [corpus_size, dist](benchmark::State &state, auto &store) {
          std::mt19937_64 rng(bench_help::kBenchmarkSeed + state.thread_index());
          bench_help::ZipfDistribution zipf({corpus_size, 1.0});
          std::uniform_int_distribution<std::size_t> uniform_index(0, corpus_size - 1);
          const bool zipf_dist = std::string(dist) == "Zipf";
          for (auto _ : state) {
            std::size_t key_index = zipf_dist ? zipf(rng) : uniform_index(rng);
            store.get(bench_help::make_key(key_index), [](std::span<const std::byte> value) {
              benchmark::DoNotOptimize(bench_help::touch_bytes(value));
            });
          }
          state.SetItemsProcessed(state.iterations());
        },
        thread_counts,
        false);
  }

  auto get_miss_meta = bench_help::make_metadata(corpus_size, val_size);
  if (include_miss) {
    register_bench(
        crud_holder,
        bench_help::benchmark_name(sname, "Get", "Miss", "Zipf", value_name),
        get_miss_meta,
        make_corpus,
        noop_already_populated_init,
        [corpus_size](benchmark::State &state, auto &store) {
          std::mt19937_64 rng(bench_help::kBenchmarkSeed + state.thread_index());
          bench_help::ZipfDistribution zipf({corpus_size, 1.0});
          for (auto _ : state) {
            std::size_t key_index = corpus_size + zipf(rng);
            store.get(bench_help::make_key(key_index),
                      [](std::span<const std::byte> value) { benchmark::DoNotOptimize(value); });
          }
          state.SetItemsProcessed(state.iterations());
        },
        thread_counts,
        false);
  }
}

// Update (stateful, against the shared corpus), registered once per (store variant, value size)
// by register_all_benchmarks() below.
template <typename Holder, typename MakeCorpus>
static void register_update_benchmark(Holder crud_holder,
                                      const std::string &sname,
                                      const std::string &value_name,
                                      MakeCorpus make_corpus,
                                      std::size_t val_size,
                                      std::size_t corpus_size,
                                      const std::vector<int> &thread_counts) {
  auto update_meta = bench_help::make_metadata(corpus_size, val_size);
  register_bench(
      crud_holder,
      bench_help::benchmark_name(sname, "Update", std::nullopt, "Zipf", value_name),
      update_meta,
      make_corpus,
      noop_already_populated_init,
      [corpus_size, val_size](benchmark::State &state, auto &store) {
        std::string dummy(val_size, 'a');
        std::string dummy_8b(8, 'a');
        std::mt19937_64 rng(bench_help::kBenchmarkSeed + state.thread_index());
        bench_help::ZipfDistribution zipf({corpus_size, 1.0});
        for (auto _ : state) {
          std::size_t key_index = zipf(rng);
          // Per-key sizes matching the corpus distribution (see bench_help::get_value_size_for_key() and
          // the Insert loops): 8B-valued keys get 8B updates (exercising the inline path),
          // so updates never change a record's size class.
          bool updated;
          if (use_mixed_8b_value(val_size, key_index)) {
            updated = store.update(bench_help::make_key(key_index), dummy_8b);
          } else {
            updated = store.update(bench_help::make_key(key_index), dummy);
          }
          benchmark::DoNotOptimize(updated);
        }
        state.SetItemsProcessed(state.iterations());
      },
      thread_counts,
      false);
}

// Delete, registered once per (store variant, value size) by register_all_benchmarks() below.
// Mutates its corpus, so (unlike Get/Update/Scan) it starts from its own fresh clone for every
// thread-count instance rather than sharing crud_holder's.
template <typename MakeFreshCorpusCheckpoint>
static void register_delete_benchmark(const std::string &sname,
                                      const std::string &value_name,
                                      MakeFreshCorpusCheckpoint make_fresh_corpus_checkpoint,
                                      std::size_t val_size,
                                      std::size_t corpus_size,
                                      const std::vector<int> &thread_counts) {
  auto delete_meta = bench_help::make_metadata(corpus_size, val_size);
  register_bench(
      bench_help::benchmark_name(sname, "Delete", std::nullopt, std::nullopt, value_name),
      delete_meta,
      [make_fresh_corpus_checkpoint, val_size, corpus_size]() {
        return make_fresh_corpus_checkpoint(val_size, corpus_size);
      },
      noop_already_populated_init,
      [corpus_size](benchmark::State &state, auto &store) {
        std::size_t threads = static_cast<std::size_t>(state.threads());
        std::size_t thread_idx = static_cast<std::size_t>(state.thread_index());
        const auto time_budget = std::chrono::duration<double>(delete_time_budget_seconds());
        run_wall_clock(state, time_budget, [&](std::size_t i) {
          std::size_t key_index = thread_idx + i * threads;
          if (key_index >= corpus_size) {
            return false;
          }
          bool removed = store.remove(bench_help::make_key(key_index));
          benchmark::DoNotOptimize(removed);
          return true;
        });
      },
      thread_counts);
}

// Scan (both Dist values), registered once per (store variant, value size) by
// register_all_benchmarks() below.
template <typename Make, typename MakeFreshCorpusCheckpoint>
static void register_scan_benchmarks(const std::string &sname,
                                     const std::string &value_name,
                                     Make make,
                                     MakeFreshCorpusCheckpoint make_fresh_corpus_checkpoint,
                                     std::size_t val_size,
                                     std::size_t corpus_size,
                                     const std::vector<int> &thread_counts,
                                     bool include_shuffled) {
  constexpr int scan_count_reorg = 100;
  const size_t reorg_dataset_size = corpus_size;
  // Cyclic steady-state twin (env-gated): G interleaved append-only generations
  // instead of one bulk load, so each T1 range contains offsets from multiple
  // generations. Build in place so checkpoint adoption does not change layout.
  // LTM 1KB only, VMemKV only.
  const std::size_t cyclic_gens = get_env_uint64("VMEMKV_BENCH_CYCLIC_GENS", 0);
  using MadeStore = typename decltype(std::declval<Make>()())::element_type;
  constexpr bool kCyclicEligible = !vmemkv::detail::is_rival_store_v<typename MadeStore::Impl>;
  const bool include_cyclic = cyclic_gens > 1 && kCyclicEligible && is_ltm_mode() && val_size == 1024;
  for (int pass = 0; pass < (include_shuffled ? 2 : 1) + (include_cyclic ? 1 : 0); ++pass) {
    // Shuffled-load twin: same cells over a master populated in fixed-seed-shuffled key order,
    // so key order no longer matches layout order (see make_bench_fresh_corpus()).
    const bool cyclic = include_cyclic && (pass == (include_shuffled ? 2 : 1));
    const bool shuffled = !cyclic && (pass == 1);
    auto scan_holder = std::make_shared<StoreHolder<decltype(make())>>();
    auto make_ordered = [make_fresh_corpus_checkpoint, val_size, corpus_size, shuffled]() {
      auto store = make_fresh_corpus_checkpoint(val_size, corpus_size, shuffled);
      if (shuffled) {
        if constexpr (kCyclicEligible) {
          wait_for_background_compaction(store);
          store->checkpoint();
        }
      }
      return store;
    };
    auto make_cyclic = [make, val_size, corpus_size, cyclic_gens]() {
      // Build all generations before allowing the background compaction worker to drain them.
      // Save/restore keeps measurement-time behavior intact.
      const char *prev_suppress =
          ::getenv("VMEMKV_SUPPRESS_AUTO_COMPACT") == nullptr ? nullptr : ::getenv("VMEMKV_SUPPRESS_AUTO_COMPACT");
      const std::string saved = prev_suppress == nullptr ? std::string() : std::string(prev_suppress);
      ::setenv("VMEMKV_SUPPRESS_AUTO_COMPACT", "1", 1);
      auto store = make();
      const std::size_t per_gen = corpus_size / cyclic_gens;
      for (std::size_t gen = 0; gen < cyclic_gens; ++gen) {
        const std::size_t count = (gen + 1 == cyclic_gens) ? corpus_size - gen * per_gen : per_gen;
        store->bulk_load(
            count,
            [gen, cyclic_gens](std::size_t i) { return bench_help::make_key(gen + i * cyclic_gens); },
            [gen, cyclic_gens, val_size](std::size_t i) {
              return bench_help::make_value_for_key(gen + i * cyclic_gens, val_size);
            });
        store->checkpoint();
      }
      if constexpr (kCyclicEligible) {
        ::unsetenv("VMEMKV_SUPPRESS_AUTO_COMPACT");
        wait_for_background_compaction(store);
      }
      // Freeze the repacked frontier so the measured scan uses the base-region
      // fast path, matching production steady state (periodic checkpoints).
      store->checkpoint();
      if (saved.empty()) {
        ::unsetenv("VMEMKV_SUPPRESS_AUTO_COMPACT");
      } else {
        ::setenv("VMEMKV_SUPPRESS_AUTO_COMPACT", saved.c_str(), 1);
      }
      return store;
    };
    std::function<decltype(make())()> make_corpus =
        cyclic ? make_cyclic : static_cast<std::function<decltype(make())()>>(make_ordered);
    const std::string load_suffix = shuffled ? "/Load=Shuffled" : (cyclic ? "/Load=Cyclic" : "");
    for (int dist_idx = 0; dist_idx < 2; ++dist_idx) {
      const char *dist = (dist_idx == 0) ? "Zipf" : "Uniform";
      auto scan_meta = bench_help::make_metadata(reorg_dataset_size, val_size);
      register_bench(
          scan_holder,
          bench_help::benchmark_name(sname, "Scan", std::nullopt, dist, value_name) + load_suffix,
          scan_meta,
          make_corpus,
          noop_already_populated_init,
          [reorg_dataset_size, dist](benchmark::State &state, auto &store) {
            std::mt19937_64 rng(bench_help::kBenchmarkSeed + state.thread_index());
            const std::size_t scan_span = reorg_dataset_size > static_cast<std::size_t>(scan_count_reorg)
                                              ? reorg_dataset_size - static_cast<std::size_t>(scan_count_reorg)
                                              : 1;
            bench_help::ZipfDistribution zipf({scan_span, 1.0});
            std::uniform_int_distribution<std::size_t> uniform_index(0, scan_span - 1);
            const bool zipf_dist = std::string(dist) == "Zipf";
            for (auto _ : state) {
              std::size_t scan_start = zipf_dist ? zipf(rng) : uniform_index(rng);
              size_t result_count = store.scan(bench_help::make_key(scan_start),
                                               bench_help::make_key(scan_start + scan_count_reorg - 1),
                                               [](std::span<const std::byte>, std::span<const std::byte> value) {
                                                 benchmark::DoNotOptimize(bench_help::touch_bytes(value));
                                               });
              benchmark::DoNotOptimize(result_count);
            }
            state.SetItemsProcessed(state.iterations());
          },
          thread_counts,
          false);
    }
  }
}

// =============================================================================
// Benchmark definitions
// =============================================================================
// Read-policy pins: Get Hit Zipf/Uniform + Scan (both dists) only. Shares the per-value
// corpus scheme above; Insert/Update/Delete/YCSB-E are insensitive to the read mapping.
template <typename Make, typename MakeFreshCorpusCheckpoint>
static void register_read_pin_cells(const char *name,
                                    Make make,
                                    MakeFreshCorpusCheckpoint make_fresh_corpus_checkpoint,
                                    const std::vector<int> &thread_counts) {
  std::string sname(name);
  std::vector<size_t> value_sizes = value_sizes_for_mode(sname);
  if (prefer_large_value_first()) {
    std::reverse(value_sizes.begin(), value_sizes.end());
  }
  for (size_t val_size : value_sizes) {
    const std::string value_name = bench_help::value_label(val_size);
    const size_t corpus_size = corpus_size_for_value(val_size);
    auto make_corpus = [make_fresh_corpus_checkpoint, val_size, corpus_size]() {
      return make_fresh_corpus_checkpoint(val_size, corpus_size);
    };
    auto crud_holder = std::make_shared<StoreHolder<decltype(make())>>();
    register_get_benchmarks(
        crud_holder, sname, value_name, make_corpus, val_size, corpus_size, thread_counts, {"Zipf", "Uniform"}, false);
    register_scan_benchmarks(
        sname, value_name, make, make_fresh_corpus_checkpoint, val_size, corpus_size, thread_counts, true);
  }
}

void register_all_benchmarks() {
  int hw_threads = static_cast<int>(std::thread::hardware_concurrency());

  // Full 4-point thread counts for Insert/Update scalability: {1, 4, 16, hw_threads}
  std::vector<int> full_thread_counts = {1, 4};
  if (16 < hw_threads) {
    full_thread_counts.push_back(16);
  }
  if (hw_threads > full_thread_counts.back()) {
    full_thread_counts.push_back(hw_threads);
  }
  std::sort(full_thread_counts.begin(), full_thread_counts.end());
  full_thread_counts.erase(std::unique(full_thread_counts.begin(), full_thread_counts.end()), full_thread_counts.end());

  // 2-point thread counts for other workloads: {1, hw_threads}
  std::vector<int> reduced_thread_counts = {1};
  if (hw_threads > 1) {
    reduced_thread_counts.push_back(hw_threads);
  }

  auto is_vmemkv_read_policy_ablation = [](const std::string &sname) {
    return sname.find("ReadRandom") != std::string::npos || sname.find("ReadSeq") != std::string::npos;
  };

  for_each_store_variant([&](const char *name, auto make, auto make_fresh_corpus_checkpoint) {
    std::string sname(name);

    // Keep the benchmark matrix aligned with the scenario model:
    // 8B for in-memory, 1KB/64KB for LTM.
    std::vector<size_t> value_sizes = value_sizes_for_mode(sname);
    if (prefer_large_value_first()) {
      std::reverse(value_sizes.begin(), value_sizes.end());
    }
    for (size_t val_size : value_sizes) {
      const std::string value_name = bench_help::value_label(val_size);
      const size_t corpus_size = corpus_size_for_value(val_size);
      // Bound to this val_size: constructs at the shared, checkpoint-reusable corpus path
      // (see for_each_store_variant()). Used by every scenario below that shares this
      // read-mostly corpus (Get/Update/YCSB-E/Scan) instead of `make`.
      auto make_corpus = [make_fresh_corpus_checkpoint, val_size, corpus_size]() {
        return make_fresh_corpus_checkpoint(val_size, corpus_size);
      };
      auto crud_holder = std::make_shared<StoreHolder<decltype(make())>>();
      auto insert_holder = std::make_shared<StoreHolder<decltype(make())>>();
      // YCSB-E gets its own holder/path rather than sharing crud_holder's: its own run phase
      // legitimately inserts new keys past corpus_size (a 5% insert mix), which would leave a
      // *live, in-process* corpus with extra, unexpected keys if crud_holder's instance were
      // reused afterward -- and could make Get/Miss's "these indices don't exist" assumption
      // false. A separate holder means YCSB-E always clones its own fresh instance from the
      // shared master (see make_fresh_corpus_checkpoint()'s comment) instead of ever risking a
      // reuse of an instance it (or something else) already grew.
      auto ycsb_holder = std::make_shared<StoreHolder<decltype(make())>>();

      if (is_vmemkv_read_policy_ablation(sname)) {
        // Read-policy ablation variants (+PinRandom, +PinSeq): measure Get-Hit and Scan only.
        register_get_benchmarks(crud_holder,
                                sname,
                                value_name,
                                make_corpus,
                                val_size,
                                corpus_size,
                                reduced_thread_counts,
                                {"Zipf", "Uniform"},
                                false);
        register_scan_benchmarks(
            sname, value_name, make, make_fresh_corpus_checkpoint, val_size, corpus_size, reduced_thread_counts, true);
      } else {
        // 1. Insert (1, 4, 16, 32 threads)
        register_insert_benchmark(insert_holder,
                                  sname,
                                  value_name,
                                  make,
                                  make_fresh_corpus_checkpoint,
                                  val_size,
                                  corpus_size,
                                  full_thread_counts);

        // 2. Get/Hit (both dists) and 3. Get/Miss (1, 32 threads)
        register_get_benchmarks(
            crud_holder, sname, value_name, make_corpus, val_size, corpus_size, reduced_thread_counts);

        // 4. Update (1, 4, 16, 32 threads)
        register_update_benchmark(
            crud_holder, sname, value_name, make_corpus, val_size, corpus_size, full_thread_counts);

        // 5. YCSB-E Benchmark (Short Range Scans, 30s mixed workload, hw_threads threads max)
        register_ycsb_e_benchmark(
            ycsb_holder, hw_threads, sname, value_name, make_fresh_corpus_checkpoint, val_size, corpus_size);

        // 6. Delete (1, 32 threads)
        register_delete_benchmark(
            sname, value_name, make_fresh_corpus_checkpoint, val_size, corpus_size, reduced_thread_counts);

        // 7. Scan (both Dist values, 1, 32 threads)
        register_scan_benchmarks(
            sname, value_name, make, make_fresh_corpus_checkpoint, val_size, corpus_size, reduced_thread_counts, true);
      }
    }
  });

  // Read-policy ablation pins (+PinNormal, +PinPread): read cells only (Get-Hit + Scan, 1, 32 threads).
  for_each_read_pin_variant([&](const char *pin_name, auto pin_make, auto pin_make_fresh_corpus_checkpoint) {
    register_read_pin_cells(pin_name, pin_make, pin_make_fresh_corpus_checkpoint, reduced_thread_counts);
  });
}

// ─── Reorg-scaling probe (standalone CLI mode, bypasses Google Benchmark) ───────────────────
//
// Measures how a reorganize()/checkpoint() call's cost and effect on concurrent traffic scale
// with corpus size. Google Benchmark's own registration model assumes the same operation repeats
// many times to build a statistic; here we want to time exactly one, possibly very slow, blocking
// call and be able to tell a driver script "this is taking too long" without waiting indefinitely
// -- hence a standalone CLI mode instead of a registered benchmark case.
//
// Population/setup is deliberately NOT time-limited here -- only the outer shell driver's own
// generous backstop timeout (wrapping this whole process) covers it. Only the
// reorganize()/checkpoint() call itself is capped (timed_run(), below), via a background thread
// plus future::wait_for(), so a runaway call is reported on its own without also charging setup
// time against the same budget (conflating the two would make a timeout ambiguous: slow setup, or
// slow reorganize/checkpoint?).
//
// Two modes, selected via --mode:
//   background_job_probe (run_background_job_probe(), below): fixed 10,000,000-record corpus;
//     measures one reorganize()/checkpoint() call's own duration plus the QPS degradation it
//     causes to concurrent Insert/Update/Scan workloads -- see benchmark/run_bench.sh probe bg-jobs. This
//     is a forced, whole-store operation (every shard synchronously merged in one call) -- not
//     what happens during ordinary operation, where ShardedT1Index's own background workers
//     maintain each shard independently and incrementally. See organic_split_probe below for that.
//   organic_split_probe (run_organic_split_probe(), below): starts from an empty store with real
//     background workers running (not the forced calls above) and inserts continuously, polling
//     for each organic per-shard split as it completes and comparing Insert QPS just before it to
//     Insert QPS during it -- see benchmark/run_bench.sh probe organic-split.
namespace reorg_probe {

constexpr int kReorgTimeoutSecondsDefault = 60;

// Overridable so a single-call reorganize()/checkpoint() timeout can be tightened for
// a specific experiment without changing the default used everywhere else this timeout applies.
static inline int reorg_timeout_seconds() {
  const uint64_t parsed = get_env_uint64("VMEMKV_BENCH_REORG_TIMEOUT_SECONDS", kReorgTimeoutSecondsDefault);
  if (parsed == 0) {
    return kReorgTimeoutSecondsDefault;
  }
  return static_cast<int>(parsed);
}

enum class ProbeMode {
  kBackgroundJobProbe,
  kOrganicSplitProbe,
  kCheckpointScaleProbe,
  kScanUpdateScanProbe,
  kCompactionChurnProbe,
};

struct ProbeArgs {
  bool is_ltm = false;
  std::size_t val_size = 0;
  ProbeMode mode = ProbeMode::kBackgroundJobProbe;
  double ratio = 1.0;
  std::string job;  // "reorganize" | "checkpoint" -- only read by kBackgroundJobProbe.
  // "monotonic" (every insert a fresh increasing key, a single hot shard) or "random" (fresh
  // keys uniform over the key space, writes spread over all shards) -- only read by
  // kOrganicSplitProbe. Random is what shows whether one split's throughput impact shrinks as
  // the shard count grows; monotonic structurally cannot (see run_organic_split_probe()).
  std::string key_pattern = "monotonic";
  // Only read by kCheckpointScaleProbe: base corpus keys, interleaved prime generations,
  // measured insert+checkpoint+compact cycles, and fresh keys per measured cycle.
  std::size_t key_count = 0;
  std::size_t gens = 10;
  std::size_t cycles = 5;
  std::size_t chunk_keys = 0;
  double duration_sec = 60.0;  // Only read by kCompactionChurnProbe.
};

[[noreturn]] void fail(const std::string &msg) {
  std::cerr << "[reorg-probe] " << msg << std::endl;
  std::exit(1);
}

auto parse_args(int argc, char **argv) -> ProbeArgs {
  ProbeArgs args;
  bool has_scenario = false;
  bool has_value_size = false;
  bool has_mode = false;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (arg == "--reorg-probe") {
      continue;
    }
    const auto eq = arg.find('=');
    if (eq == std::string_view::npos) {
      continue;
    }
    const std::string_view key = arg.substr(0, eq);
    const std::string_view value = arg.substr(eq + 1);
    if (key == "--scenario") {
      if (value == "ltm") {
        args.is_ltm = true;
      } else if (value == "in_memory") {
        args.is_ltm = false;
      } else {
        fail("unknown --scenario: " + std::string(value));
      }
      has_scenario = true;
    } else if (key == "--value-size") {
      if (value == "8B") {
        args.val_size = 8;
      } else if (value == "1KB") {
        args.val_size = 1024;
      } else if (value == "64KB") {
        args.val_size = 65536;
      } else {
        fail("unknown --value-size: " + std::string(value));
      }
      has_value_size = true;
    } else if (key == "--mode") {
      if (value == "background_job_probe") {
        args.mode = ProbeMode::kBackgroundJobProbe;
      } else if (value == "organic_split_probe") {
        args.mode = ProbeMode::kOrganicSplitProbe;
      } else if (value == "checkpoint_scale_probe") {
        args.mode = ProbeMode::kCheckpointScaleProbe;
      } else if (value == "scan_update_scan_probe") {
        args.mode = ProbeMode::kScanUpdateScanProbe;
      } else if (value == "compaction_churn_probe") {
        args.mode = ProbeMode::kCompactionChurnProbe;
      } else {
        fail("unknown --mode: " + std::string(value));
      }
      has_mode = true;
    } else if (key == "--ratio") {
      args.ratio = std::stod(std::string(value));
    } else if (key == "--job") {
      // Only read by kBackgroundJobProbe.
      args.job = std::string(value);
    } else if (key == "--key-pattern") {
      // Only read by kOrganicSplitProbe.
      if (value != "monotonic" && value != "random") {
        fail("unknown --key-pattern: " + std::string(value));
      }
      args.key_pattern = std::string(value);
    } else if (key == "--key-count") {
      // Only read by kCheckpointScaleProbe and kCompactionChurnProbe.
      args.key_count = static_cast<std::size_t>(std::stoull(std::string(value)));
    } else if (key == "--gens") {
      // Only read by kCheckpointScaleProbe.
      args.gens = static_cast<std::size_t>(std::stoull(std::string(value)));
    } else if (key == "--cycles") {
      // Only read by kCheckpointScaleProbe.
      args.cycles = static_cast<std::size_t>(std::stoull(std::string(value)));
    } else if (key == "--chunk-keys") {
      // Only read by kCheckpointScaleProbe.
      args.chunk_keys = static_cast<std::size_t>(std::stoull(std::string(value)));
    } else if (key == "--duration-sec") {
      // Only read by kCompactionChurnProbe.
      args.duration_sec = std::stod(std::string(value));
    }
  }
  if (!has_scenario || !has_value_size || !has_mode) {
    fail(
        "usage: --reorg-probe --scenario=<in_memory|ltm> --value-size=<8B|1KB|64KB> "
        "--mode=<background_job_probe|organic_split_probe|checkpoint_scale_probe|scan_update_scan_probe|compaction_"
        "churn_probe> "
        "--ratio=<0.0-1.0> [--job=<reorganize|checkpoint>] [--key-pattern=<monotonic|random>] [--duration-sec=<sec>]");
  }
  if (args.mode == ProbeMode::kBackgroundJobProbe && args.job != "reorganize" && args.job != "checkpoint") {
    fail("--mode=background_job_probe requires --job=<reorganize|checkpoint>");
  }
  if (args.ratio <= 0.0 || args.ratio > 1.0) {
    fail("--ratio must be in (0.0, 1.0]");
  }
  return args;
}

// Times a single call to `fn` (reorganize()/checkpoint()) in a detached background thread capped
// at reorg_timeout_seconds() -- see the file-level comment above this namespace for why
// google-benchmark's iteration model doesn't fit timing exactly one, possibly very slow, blocking
// call.
//
// Every caller must terminate via std::_Exit(), not a normal return, once this returns: on
// timeout, the worker thread above is still inside reorganize()/checkpoint() touching the store,
// so returning normally and running destructors (in particular the store's, which would join
// reorg_worker_) could itself block forever. _Exit() is also the simplest way to avoid that same
// destructor path racing the now-finished-but-still-detached worker thread on a non-timeout exit.
template <typename Fn>
auto timed_run(Fn &&fn) -> std::pair<double, bool> {
  const int timeout_seconds = reorg_timeout_seconds();
  std::promise<void> done_promise;
  auto done_future = done_promise.get_future();
  const auto t0 = std::chrono::steady_clock::now();
  std::thread worker([fn = std::forward<Fn>(fn), promise = std::move(done_promise)]() mutable {
    fn();
    promise.set_value();
  });
  worker.detach();

  const auto status = done_future.wait_for(std::chrono::seconds(timeout_seconds));
  const bool timed_out = status != std::future_status::ready;
  if (timed_out) {
    std::cerr << "[timed_run] TIMED OUT after " << timeout_seconds << "s! Last checkpoint phase was: "
              << vmemkv::checkpoint_detail::g_checkpoint_debug_phase.load(std::memory_order_acquire)
              << " (shard: " << vmemkv::t1_debug::g_checkpoint_current_shard.load(std::memory_order_acquire) << "/"
              << vmemkv::t1_debug::g_checkpoint_total_shards.load(std::memory_order_acquire) << ")" << std::endl;
  }
  const double elapsed_sec = timed_out ? static_cast<double>(timeout_seconds)
                                       : std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return {elapsed_sec, timed_out};
}

static std::string reorg_probe_path(const ProbeArgs &args, const std::string &suffix) {
  return get_db_dir() + "/reorg_probe_" + (args.is_ltm ? "ltm" : "inmem") + "_" + std::to_string(args.val_size) + "_" +
         suffix;
}

static std::size_t resolve_writer_threads() {
  return std::min<std::size_t>({std::max(1u, std::thread::hardware_concurrency()), 32});
}

// Background-job probe: for a fixed corpus (args.val_size, kBackgroundJobProbeKeyCount records --
// in-memory unconstrained, or LTM cgroup-constrained by the *caller* script, this binary itself
// applies no memory limit), measures one call to reorganize()/checkpoint() (args.job) and, for
// each of three concurrent workloads (Insert, Update, Scan), the percentage QPS degradation that
// workload suffers while the job call is in flight. See benchmark/run_bench.sh probe bg-jobs.
//
// Each workload gets its own paired isolated/concurrent measurement, both windows sized to the
// SAME wall-clock duration (the job call's own measured elapsed time) -- comparing TPS across two
// differently-sized windows would not be a fair measurement. Phase B (concurrent) runs the job
// once while workload threads hammer continuously (stop-flag controlled) and measures its own
// elapsed time; phase A (isolated) then runs the identical thread/workload setup for exactly that
// same elapsed time (timer-controlled, no concurrent job) as the baseline.
enum class BackgroundJobWorkload { kInsert, kUpdate, kScan };

// kInsert draws from `next_fresh_key` (shared across a phase's threads, never reused within one
// measure_workload_degradation() call); kUpdate/kScan pick a uniformly random existing key.
// kScan issues a small (100-key) ranged scan, mirroring the main Google Benchmark-registered
// Op=Scan implementation above.
template <typename Store>
static void background_job_do_one_op(Store &store,
                                     uint32_t val_size,
                                     BackgroundJobWorkload workload,
                                     std::mt19937_64 &rng,
                                     std::uniform_int_distribution<std::size_t> &key_dist,
                                     std::atomic<std::size_t> &next_fresh_key) {
  switch (workload) {
    case BackgroundJobWorkload::kInsert: {
      const std::size_t idx = next_fresh_key.fetch_add(1, std::memory_order_relaxed);
      store->insert(bench_help::make_key(idx), bench_help::make_value_for_key(idx, val_size));
      return;
    }
    case BackgroundJobWorkload::kUpdate: {
      const std::size_t idx = key_dist(rng);
      store->update(bench_help::make_key(idx), bench_help::make_value_for_key(idx, val_size));
      return;
    }
    case BackgroundJobWorkload::kScan: {
      const std::size_t start = key_dist(rng);
      std::size_t result_count = store->scan(bench_help::make_key(start),
                                             bench_help::make_key(start + kScanLength - 1),
                                             [](std::span<const std::byte>, std::span<const std::byte> value) {
                                               benchmark::DoNotOptimize(bench_help::touch_bytes(value));
                                             });
      benchmark::DoNotOptimize(result_count);
      return;
    }
  }
}

// Worker threads for one measure_workload_degradation() phase.
template <typename Store>
static void spawn_workload_workers(Store &store,
                                   uint32_t val_size,
                                   BackgroundJobWorkload workload,
                                   std::size_t key_count,
                                   std::size_t thread_count,
                                   uint64_t seed_base,
                                   std::atomic<std::size_t> &next_fresh_key,
                                   std::atomic<bool> &stop,
                                   std::vector<std::thread> &workers,
                                   std::vector<std::size_t> &counts) {
  workers.reserve(thread_count);
  for (std::size_t t = 0; t < thread_count; ++t) {
    workers.emplace_back([&, t]() {
      std::mt19937_64 rng(bench_help::kBenchmarkSeed + seed_base + t);
      std::uniform_int_distribution<std::size_t> key_dist(0, key_count > kScanLength ? key_count - kScanLength : 1);
      std::size_t done = 0;
      while (!stop.load(std::memory_order_relaxed)) {
        background_job_do_one_op(store, val_size, workload, rng, key_dist, next_fresh_key);
        ++done;
      }
      counts[t] = done;
    });
  }
}

struct WorkloadDegradation {
  double job_elapsed_sec = 0.0;
  double degradation_pct = 0.0;
  bool timed_out = false;
};

template <typename Store, typename JobFn>
static auto measure_workload_degradation(Store &store,
                                         std::size_t key_count,
                                         uint32_t val_size,
                                         BackgroundJobWorkload workload,
                                         std::size_t thread_count,
                                         JobFn &&run_job) -> WorkloadDegradation {
  // Phase B: concurrent -- workload threads run continuously while the job executes once.
  std::atomic<std::size_t> next_fresh_key_b{key_count};
  std::atomic<bool> stop_b{false};
  std::vector<std::thread> workers_b;
  std::vector<std::size_t> counts_b(thread_count, 0);
  spawn_workload_workers(
      store, val_size, workload, key_count, thread_count, 9000, next_fresh_key_b, stop_b, workers_b, counts_b);
  auto [job_elapsed_sec, timed_out] = timed_run(std::forward<JobFn>(run_job));
  stop_b.store(true, std::memory_order_relaxed);
  for (auto &w : workers_b) {
    w.join();
  }
  if (timed_out || job_elapsed_sec <= 0.0) {
    return {job_elapsed_sec, 0.0, timed_out};
  }
  std::size_t concurrent_ops = 0;
  for (auto c : counts_b) {
    concurrent_ops += c;
  }
  const double concurrent_tps = static_cast<double>(concurrent_ops) / job_elapsed_sec;

  // Phase A: isolated -- same setup, run for exactly job_elapsed_sec (timer-controlled), not a
  // fixed op count -- matches Phase B's own window so the two TPS figures are comparable.
  std::atomic<std::size_t> next_fresh_key_a{key_count};
  std::atomic<bool> stop_a{false};
  std::vector<std::thread> workers_a;
  std::vector<std::size_t> counts_a(thread_count, 0);
  spawn_workload_workers(
      store, val_size, workload, key_count, thread_count, 5000, next_fresh_key_a, stop_a, workers_a, counts_a);
  std::this_thread::sleep_for(std::chrono::duration<double>(job_elapsed_sec));
  stop_a.store(true, std::memory_order_relaxed);
  for (auto &w : workers_a) {
    w.join();
  }
  std::size_t isolated_ops = 0;
  for (auto c : counts_a) {
    isolated_ops += c;
  }
  const double isolated_tps = static_cast<double>(isolated_ops) / job_elapsed_sec;
  const double degradation_pct = isolated_tps > 0.0 ? (1.0 - concurrent_tps / isolated_tps) * 100.0 : 0.0;
  return {job_elapsed_sec, degradation_pct, false};
}

// Fixed corpus size -- a single reproducible reference point (see this file's own doc comment
// above) rather than a sweep across the matrix's 4 scenario/value-size combos.
constexpr std::size_t kBackgroundJobProbeKeyCount = 10'000'000;

[[noreturn]] void run_background_job_probe(const ProbeArgs &args) {
  using Store = vmemkv::variants::VMemKV_Default;

  const std::size_t key_count = kBackgroundJobProbeKeyCount;
  const std::string path = reorg_probe_path(args, "backgroundjob_" + args.job);
  const std::size_t resolved_writer_threads = resolve_writer_threads();

  setenv("VMEMKV_SUPPRESS_AUTO_COMPACT", "1", 1);
  auto store = make_vmemkv_fresh(
      path, [&path]() { return std::make_unique<Store>(path, Store::ConfigType::kDefaultCapacityBytes); });
  bench_help::populate(*store, {key_count, args.val_size}, /*shuffled=*/true);
  // Ensure the 10M-key corpus is properly sharded (target ~8-10 shards)
  // before measuring whole-store maintenance under concurrent traffic.
  std::cerr << "[run_background_job_probe] starting manual sharding..." << std::endl;
  for (std::size_t split_step = 1; split_step <= 3; ++split_step) {
    const std::size_t num_splits = std::size_t{1} << (split_step - 1);
    for (std::size_t i = 0; i < num_splits; ++i) {
      const std::size_t key_idx = (2 * i + 1) * (key_count / (num_splits * 2));
      const std::string k = bench_help::make_key(key_idx);
      store->impl().t1().split_shard_containing(
          std::span<const std::byte>(reinterpret_cast<const std::byte *>(k.data()), k.size()));
    }
    std::cerr << "[run_background_job_probe] split step " << split_step
              << " finished, shard_count=" << store->impl().t1().shard_count()
              << " total_splits=" << store->impl().t1().total_splits() << std::endl;
  }
  store->checkpoint();
  // Suppress background auto-reorganization during measurement so the foreground
  // reorganize/checkpoint invocation being measured does not contend for per-shard reorg locks.
  setenv("VMEMKV_SUPPRESS_AUTO_REORG", "1", 1);
  std::cerr << "[run_background_job_probe] initial shard_count=" << store->impl().t1().shard_count()
            << " total_splits=" << store->impl().t1().total_splits() << std::endl;

  std::function<void()> run_job;
  if (args.job == "reorganize") {
    run_job = [&store]() { store->reorganize(); };
  } else {
    run_job = [&store]() { store->checkpoint(); };
  }

  bool any_timed_out = false;
  double reported_elapsed_sec = 0.0;
  std::optional<double> insert_pct;
  std::optional<double> update_pct;
  std::optional<double> scan_pct;
  const std::array<std::pair<BackgroundJobWorkload, std::optional<double> *>, 3> workloads{{
      {BackgroundJobWorkload::kInsert, &insert_pct},
      {BackgroundJobWorkload::kUpdate, &update_pct},
      {BackgroundJobWorkload::kScan, &scan_pct},
  }};
  for (const auto &[workload, out_pct] : workloads) {
    auto result =
        measure_workload_degradation(store, key_count, args.val_size, workload, resolved_writer_threads, run_job);
    reported_elapsed_sec = result.job_elapsed_sec;
    if (result.timed_out) {
      any_timed_out = true;
      break;
    }
    *out_pct = result.degradation_pct;
  }

  auto pct_json = [](std::optional<double> v) -> std::string { return v.has_value() ? std::to_string(*v) : "null"; };
  // Phase breakdown: only meaningful for job=="checkpoint" (reorganize()'s T1Only path never runs
  // checkpoint_internal(), so these fields would just be stale/zero) -- last_checkpoint_* reflects
  // whichever of the 3 measure_workload_degradation() calls above most recently ran checkpoint(),
  // representative since corpus size barely changes call to call.
  std::optional<double> t1_reorganize_ms;
  std::optional<double> t1_merge_ms;
  std::optional<double> t1_write_ms;
  std::optional<double> wal_rotate_ms;
  std::optional<double> msync_ms;
  if (!any_timed_out && args.job == "checkpoint") {
    const auto stats = store->get_statistics();
    t1_reorganize_ms = stats.last_checkpoint_t1_reorganize_duration_us / 1000.0;
    t1_merge_ms = stats.last_checkpoint_t1_merge_duration_us / 1000.0;
    t1_write_ms = stats.last_checkpoint_t1_write_duration_us / 1000.0;
    wal_rotate_ms = stats.last_checkpoint_wal_rotate_duration_us / 1000.0;
    msync_ms = stats.last_checkpoint_msync_duration_us / 1000.0;
  }
  std::cout << "{\"job\":\"" << args.job << "\"," << "\"scenario\":\"" << (args.is_ltm ? "ltm" : "in_memory") << "\","
            << "\"value_size\":" << args.val_size << "," << "\"key_count\":" << key_count << ","
            << "\"writer_threads\":" << resolved_writer_threads << "," << "\"job_elapsed_sec\":" << reported_elapsed_sec
            << "," << "\"insert_degradation_pct\":" << pct_json(insert_pct) << ","
            << "\"update_degradation_pct\":" << pct_json(update_pct) << ","
            << "\"scan_degradation_pct\":" << pct_json(scan_pct) << ","
            << "\"t1_reorganize_phase_ms\":" << pct_json(t1_reorganize_ms) << ","
            << "\"t1_merge_phase_ms\":" << pct_json(t1_merge_ms) << ","
            << "\"t1_write_phase_ms\":" << pct_json(t1_write_ms) << ","
            << "\"wal_rotate_phase_ms\":" << pct_json(wal_rotate_ms) << ","
            << "\"msync_phase_ms\":" << pct_json(msync_ms) << ","
            << "\"timed_out\":" << (any_timed_out ? "true" : "false") << "}" << std::endl;
  std::_Exit(any_timed_out ? 124 : 0);
}

// Organic per-shard split probe: unlike run_background_job_probe() above (a forced, whole-store
// reorganize()/checkpoint() call), this measures the Insert-QPS impact of ShardedT1Index's own
// automatic background splitting under sustained write load -- the maintenance path that actually
// The probe starts with an empty store and active background workers. Insert workers use
// monotonically increasing or uniformly random keys. Statistics include completed split count,
// writer-pause duration, and pause-end timestamp. QPS uses a local pre-pause window and the
// measured pause interval. See docs/specification/t1_sharding_design.md.
constexpr int kOrganicSplitProbeDurationSec = 90;
constexpr auto kOrganicSplitPollInterval = std::chrono::milliseconds(100);
// Duration of the pre-pause QPS baseline window.
constexpr double kOrganicSplitWindowSec = 1.2;

[[noreturn]] void run_organic_split_probe(const ProbeArgs &args) {
  using Store = vmemkv::variants::VMemKV_Default;

  const std::string path = reorg_probe_path(args, "organic_split");
  const std::size_t writer_threads = resolve_writer_threads();

  setenv("VMEMKV_SUPPRESS_AUTO_REORG", "1", 1);
  setenv("VMEMKV_SUPPRESS_AUTO_COMPACT", "1", 1);
  auto store = make_vmemkv_fresh(
      path, [&path]() { return std::make_unique<Store>(path, Store::ConfigType::kDefaultCapacityBytes); });

  // Automatic maintenance is suppressed during the probe.

  // Monotonic keys increase from zero. Random keys are uniformly selected from the 2^48 key space.
  const bool random_keys = args.key_pattern == "random";
  std::atomic<std::size_t> next_key{0};
  // Completed inserts used by QPS sampling and the final total.
  std::atomic<std::size_t> inserted{0};
  std::atomic<bool> stop{false};
  std::vector<std::thread> workers;
  workers.reserve(writer_threads);
  for (std::size_t t = 0; t < writer_threads; ++t) {
    workers.emplace_back([&, t]() {
      std::mt19937_64 rng(bench_help::kBenchmarkSeed + t);
      std::uniform_int_distribution<std::size_t> dist(0, (std::size_t{1} << 48) - 1);
      while (!stop.load(std::memory_order_relaxed)) {
        const std::size_t idx = random_keys ? dist(rng) : next_key.fetch_add(1, std::memory_order_relaxed);
        store->insert(bench_help::make_key(idx), bench_help::make_value_for_key(idx, args.val_size));
        inserted.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  struct Sample {
    double t_sec;
    std::size_t inserted;
    uint64_t split_count;
    uint64_t last_pause_us;      // stats.t1_last_split_pause_us as of this poll
    uint64_t last_pause_end_ns;  // stats.t1_last_split_pause_end_ns as of this poll
  };
  std::vector<Sample> samples;
  const auto t0 = std::chrono::steady_clock::now();
  const auto t0_epoch_ns = static_cast<uint64_t>(t0.time_since_epoch().count());
  auto elapsed_sec = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
  while (elapsed_sec() < kOrganicSplitProbeDurationSec) {
    const auto stats = store->get_statistics();
    samples.push_back({elapsed_sec(),
                       inserted.load(std::memory_order_relaxed),
                       stats.t1_split_count,
                       stats.t1_last_split_pause_us,
                       stats.t1_last_split_pause_end_ns});
    std::this_thread::sleep_for(kOrganicSplitPollInterval);
  }
  stop.store(true, std::memory_order_relaxed);
  for (auto &w : workers) {
    w.join();
  }
  unsetenv("VMEMKV_SUPPRESS_AUTO_REORG");
  const uint64_t final_split_count = store->get_statistics().t1_split_count;

  // Nearest sample at or before `t_sec` (clamped to the first sample if `t_sec` predates the run).
  auto sample_at_or_before = [&](double t_sec) -> const Sample & {
    auto it =
        std::upper_bound(samples.begin(), samples.end(), t_sec, [](double t, const Sample &s) { return t < s.t_sec; });
    if (it == samples.begin()) {
      return samples.front();
    }
    return *std::prev(it);
  };
  auto qps_between = [&](double from_sec, double to_sec) -> std::optional<double> {
    if (from_sec < samples.front().t_sec) {
      return std::nullopt;
    }
    const Sample &a = sample_at_or_before(from_sec);
    const Sample &b = sample_at_or_before(to_sec);
    const double dt = b.t_sec - a.t_sec;
    if (dt <= 0.0 || b.inserted <= a.inserted) {
      return std::nullopt;
    }
    return static_cast<double>(b.inserted - a.inserted) / dt;
  };

  struct SplitEvent {
    double pause_start_sec;
    double pause_us;
    uint64_t shard_count_after;
    double baseline_qps;
    double during_qps;
  };
  std::vector<SplitEvent> events;
  uint64_t last_recorded_pause_end_ns = 0;
  for (std::size_t i = 1; i < samples.size(); ++i) {
    if (samples[i].split_count <= samples[i - 1].split_count) {
      continue;
    }
    // The probe and split timestamps use the same process-local steady_clock epoch.
    const uint64_t pause_end_ns = samples[i].last_pause_end_ns;
    if (pause_end_ns <= last_recorded_pause_end_ns) {
      continue;
    }
    last_recorded_pause_end_ns = pause_end_ns;
    const double pause_end_sec = static_cast<double>(pause_end_ns - t0_epoch_ns) / 1e9;
    const double pause_us = static_cast<double>(samples[i].last_pause_us);
    const double pause_start_sec = pause_end_sec - pause_us / 1e6;

    const auto baseline = qps_between(pause_start_sec - kOrganicSplitWindowSec, pause_start_sec);
    const auto during = qps_between(pause_start_sec, pause_end_sec);
    if (!baseline.has_value() || !during.has_value() || *baseline <= 0.0) {
      continue;
    }
    events.push_back({pause_start_sec, pause_us, samples[i].split_count + 1, *baseline, *during});
  }

  std::cout << "{\"job\":\"organic_split\",\"scenario\":\"" << (args.is_ltm ? "ltm" : "in_memory") << "\","
            << "\"value_size\":" << args.val_size << ",\"writer_threads\":" << writer_threads << ",\"key_pattern\":\""
            << args.key_pattern << "\"," << "\"duration_sec\":" << kOrganicSplitProbeDurationSec << ","
            << "\"total_inserted\":" << inserted.load(std::memory_order_relaxed) << ","
            << "\"completed_splits\":" << final_split_count << ","
            << "\"final_shard_count\":" << store->impl().t1().shard_count() << ",\"splits\":[";
  for (std::size_t i = 0; i < events.size(); ++i) {
    const auto &e = events[i];
    const double degradation_pct = (1.0 - e.during_qps / e.baseline_qps) * 100.0;
    std::cout << (i > 0 ? "," : "") << "{\"at_sec\":" << e.pause_start_sec << ",\"pause_us\":" << e.pause_us
              << ",\"shard_count_after\":" << e.shard_count_after << ",\"baseline_qps\":" << e.baseline_qps
              << ",\"during_qps\":" << e.during_qps << ",\"degradation_pct\":" << degradation_pct << "}";
  }
  std::cout << "],\"timed_out\":false}" << std::endl;
  std::_Exit(0);
}

[[noreturn]] void run_checkpoint_scale_probe(const ProbeArgs &args) {
  using Store = vmemkv::variants::VMemKV_Default;

  const std::size_t key_count = args.key_count;
  const std::size_t gens = args.gens == 0 ? 1 : args.gens;
  const std::size_t cycles = args.cycles;
  const std::size_t chunk = args.chunk_keys == 0 ? std::max<std::size_t>(key_count / 10, 1) : args.chunk_keys;
  const std::size_t val_size = args.val_size;
  if (key_count == 0 || cycles == 0) {
    fail("--key-count and --cycles must be nonzero for checkpoint_scale_probe");
  }
  const std::string path = reorg_probe_path(args, "checkpointscale");

  // Suppressed throughout (prime and measured cycles): automatic reorganize/checkpoint
  // would interleave with the per-cycle checkpoint and pollute the accounting.
  setenv("VMEMKV_SUPPRESS_AUTO_REORG", "1", 1);
  setenv("VMEMKV_SUPPRESS_AUTO_COMPACT", "1", 1);
  auto store = make_vmemkv_fresh(
      path, [&path]() { return std::make_unique<Store>(path, Store::ConfigType::kDefaultCapacityBytes); });

  // Interleaved append-only prime: gens overlapping runs over the same key range,
  // then settle to at most K runs (production steady state).
  const std::size_t per_gen = key_count / gens;
  for (std::size_t gen = 0; gen < gens; ++gen) {
    const std::size_t count = (gen + 1 == gens) ? key_count - gen * per_gen : per_gen;
    store->bulk_load(
        count,
        [gen, gens](std::size_t i) { return bench_help::make_key(gen + i * gens); },
        [gen, gens, val_size](std::size_t i) { return bench_help::make_value_for_key(gen + i * gens, val_size); });
    store->checkpoint();
  }
  ::unsetenv("VMEMKV_SUPPRESS_AUTO_COMPACT");
  wait_for_background_compaction(store);
  ::setenv("VMEMKV_SUPPRESS_AUTO_COMPACT", "1", 1);

  for (std::size_t cycle = 0; cycle < cycles; ++cycle) {
    const std::size_t base = key_count + cycle * chunk;
    for (std::size_t i = 0; i < chunk; ++i) {
      std::ignore = store->insert(bench_help::make_key(base + i), bench_help::make_value_for_key(base + i, val_size));
    }
    const auto before = store->get_statistics();
    store->checkpoint();
    const auto after_checkpoint = store->get_statistics();
    const auto compact_start = std::chrono::steady_clock::now();
    ::unsetenv("VMEMKV_SUPPRESS_AUTO_COMPACT");
    wait_for_background_compaction(store);
    ::setenv("VMEMKV_SUPPRESS_AUTO_COMPACT", "1", 1);
    const auto compact_us =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - compact_start).count();
    const auto after = store->get_statistics();
    std::cout << "{\"probe\":\"checkpoint_scale\",\"scenario\":\"" << (args.is_ltm ? "ltm" : "in_memory") << "\","
              << "\"value_size\":" << val_size << "," << "\"base_keys\":" << key_count << "," << "\"gens\":" << gens
              << "," << "\"cycle\":" << cycle << "," << "\"corpus_keys\":" << (base + chunk) << ","
              << "\"corpus_bytes\":" << after_checkpoint.last_checkpoint_corpus_bytes << ","
              << "\"checkpoint_total_ms\":" << after_checkpoint.last_checkpoint_duration_us / 1000.0 << ","
              << "\"msync_ms\":" << after_checkpoint.last_checkpoint_msync_duration_us / 1000.0 << ","
              << "\"t1_reorganize_ms\":" << after_checkpoint.last_checkpoint_t1_reorganize_duration_us / 1000.0 << ","
              << "\"t1_merge_ms\":" << after_checkpoint.last_checkpoint_t1_merge_duration_us / 1000.0 << ","
              << "\"t1_write_ms\":" << after_checkpoint.last_checkpoint_t1_write_duration_us / 1000.0 << ","
              << "\"stop_ms\":" << after_checkpoint.last_checkpoint_stop_writers_duration_us / 1000.0 << ","
              << "\"barrier_ms\":" << after_checkpoint.last_checkpoint_barrier_drain_duration_us / 1000.0 << ","
              << "\"wal_rotate_ms\":" << after_checkpoint.last_checkpoint_wal_rotate_duration_us / 1000.0 << ","
              << "\"leader_wait_ms\":" << after_checkpoint.last_checkpoint_wal_rotate_leader_wait_us / 1000.0 << ","
              << "\"bytes_synced\":" << after_checkpoint.last_checkpoint_bytes_synced << ","
              << "\"compact_ms\":" << compact_us / 1000.0 << ","
              << "\"merge_cycles_added\":" << (after.compaction_cycle_count - before.compaction_cycle_count) << ","
              << "\"dirty_shards\":" << after.t2_dirty_shard_count << "}" << std::endl;
  }
  std::_Exit(0);
}

struct ScanPhaseResult {
  double milliseconds_per_window = 0.0;
  double t1_only_milliseconds_per_window = 0.0;
  long major_faults = 0;
  long minor_faults = 0;
  long input_blocks = 0;
  long max_rss_kb = 0;
  uint64_t records = 0;
  uint64_t value_bytes = 0;
  uint64_t base_records = 0;
  uint64_t tail_records = 0;
  std::vector<double> window_milliseconds;
  std::vector<long> window_major_faults;
  std::vector<long> window_minor_faults;
  std::vector<long> window_input_blocks;
};

auto drop_page_cache() -> bool {
  ::sync();
  const int fd = ::open("/proc/sys/vm/drop_caches", O_WRONLY | O_CLOEXEC);
  if (fd < 0) {
    return false;
  }
  constexpr char kDropCaches[] = "3\n";
  const ssize_t written = ::write(fd, kDropCaches, sizeof(kDropCaches) - 1);
  ::close(fd);
  if (written != static_cast<ssize_t>(sizeof(kDropCaches) - 1)) {
    return false;
  }
  return true;
}

struct MemPressureSnapshot {
  bool ok = false;
  uint64_t current = 0;
  uint64_t peak = 0;
  uint64_t swap_current = 0;
  uint64_t pgfault = 0;
  uint64_t pgmajfault = 0;
};

inline auto read_cgroup_uint(const std::string &path, uint64_t &out) -> bool {
  std::ifstream file(path);
  if (!file) {
    return false;
  }
  std::string line;
  if (!std::getline(file, line)) {
    return false;
  }
  try {
    out = std::stoull(line);
  } catch (...) {
    return false;
  }
  return true;
}

// Best-effort snapshot of the calling process's cgroup memory pressure.
// Fields stay zero with ok=false when cgroup accounting is unavailable.
inline auto take_mem_snapshot() -> MemPressureSnapshot {
  MemPressureSnapshot snapshot;
  std::ifstream group_file("/proc/self/cgroup");
  if (!group_file) {
    return snapshot;
  }
  std::string line;
  std::string cgroup_rel;
  while (std::getline(group_file, line)) {
    const std::string::size_type colon = line.rfind(':');
    if (colon != std::string::npos) {
      cgroup_rel = line.substr(colon + 1);
    }
  }
  if (cgroup_rel.empty()) {
    return snapshot;
  }
  const std::string dir = "/sys/fs/cgroup" + cgroup_rel;
  uint64_t current = 0;
  uint64_t peak = 0;
  uint64_t swap_current = 0;
  if (!read_cgroup_uint(dir + "/memory.current", current)) {
    return snapshot;
  }
  read_cgroup_uint(dir + "/memory.peak", peak);
  read_cgroup_uint(dir + "/memory.swap.current", swap_current);
  std::ifstream events(dir + "/memory.events");
  uint64_t pgfault = 0;
  uint64_t pgmajfault = 0;
  if (events) {
    std::string event_line;
    while (std::getline(events, event_line)) {
      std::istringstream fields(event_line);
      std::string name;
      uint64_t value = 0;
      if (fields >> name >> value) {
        if (name == "pgfault") {
          pgfault = value;
        } else if (name == "pgmajfault") {
          pgmajfault = value;
        }
      }
    }
  }
  snapshot.ok = true;
  snapshot.current = current;
  snapshot.peak = peak;
  snapshot.swap_current = swap_current;
  snapshot.pgfault = pgfault;
  snapshot.pgmajfault = pgmajfault;
  return snapshot;
}

template <typename Store>
auto scan_windows(Store &store, const std::vector<std::size_t> &starts) -> ScanPhaseResult {
  uint64_t records = 0;
  uint64_t value_bytes = 0;
  uint64_t base_records = 0;
  uint64_t tail_records = 0;
  const uint64_t base_boundary = store.impl().t2().get_memory()->base_boundary.load(std::memory_order_acquire);
  ScanPhaseResult result;
  result.window_milliseconds.reserve(starts.size());
  result.window_major_faults.reserve(starts.size());
  result.window_minor_faults.reserve(starts.size());
  result.window_input_blocks.reserve(starts.size());
  // T1-only pass first: measures cold-T1 range-scan cost, and its result tells
  // whether the store.scan that follows is paying T1 faults or T2 reads.
  const auto t1_start = std::chrono::steady_clock::now();
  for (const std::size_t scan_start : starts) {
    const std::string lower = bench_help::make_key(scan_start);
    const std::string upper = bench_help::make_key(scan_start + 99);
    const auto lo = std::as_bytes(std::span(lower));
    const auto hi = std::as_bytes(std::span(upper));
    store.impl().t1().scan(lo, hi, [](std::span<const std::byte>, uint64_t, uint64_t) {});
  }
  const auto t1_end = std::chrono::steady_clock::now();
  result.t1_only_milliseconds_per_window =
      std::chrono::duration<double, std::milli>(t1_end - t1_start).count() / static_cast<double>(starts.size());
  struct rusage previous_ru {};
  if (::getrusage(RUSAGE_SELF, &previous_ru) != 0) {
    fail("getrusage before scan failed");
  }
  const struct rusage scan_base = previous_ru;
  const auto start_time = std::chrono::steady_clock::now();
  auto previous_time = start_time;
  for (const std::size_t scan_start : starts) {
    const std::string lower = bench_help::make_key(scan_start);
    const std::string upper = bench_help::make_key(scan_start + 99);
    const auto lo = std::as_bytes(std::span(lower));
    const auto hi = std::as_bytes(std::span(upper));
    store.impl().t1().scan(lo, hi, [&](std::span<const std::byte>, uint64_t payload, uint64_t hash) {
      if (payload == vmemkv::STORE_NOT_FOUND || t1_detail::is_inline(hash)) {
        return;
      }
      const uint64_t off = payload & vmemkv::t2_detail::kPayloadOffsetMask;
      if (off < base_boundary) {
        ++base_records;
      } else {
        ++tail_records;
      }
    });
    records += store.scan(lower, upper, [&](std::span<const std::byte>, std::span<const std::byte> value) {
      value_bytes += value.size();
      benchmark::DoNotOptimize(bench_help::touch_bytes(value));
    });
    const auto window_end = std::chrono::steady_clock::now();
    struct rusage window_ru {};
    if (::getrusage(RUSAGE_SELF, &window_ru) != 0) {
      fail("getrusage during scan failed");
    }
    result.window_milliseconds.push_back(std::chrono::duration<double, std::milli>(window_end - previous_time).count());
    result.window_major_faults.push_back(window_ru.ru_majflt - previous_ru.ru_majflt);
    result.window_minor_faults.push_back(window_ru.ru_minflt - previous_ru.ru_minflt);
    result.window_input_blocks.push_back(window_ru.ru_inblock - previous_ru.ru_inblock);
    previous_time = window_end;
    previous_ru = window_ru;
  }
  const auto end_time = std::chrono::steady_clock::now();
  const double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
  result.milliseconds_per_window = elapsed_ms / static_cast<double>(starts.size());
  result.major_faults = previous_ru.ru_majflt - scan_base.ru_majflt;
  result.minor_faults = previous_ru.ru_minflt - scan_base.ru_minflt;
  result.input_blocks = previous_ru.ru_inblock - scan_base.ru_inblock;
  result.max_rss_kb = previous_ru.ru_maxrss;
  result.records = records;
  result.value_bytes = value_bytes;
  result.base_records = base_records;
  result.tail_records = tail_records;
  return result;
}

template <typename Store>
auto offset_inversions(Store &store) -> uint64_t {
  std::array<std::byte, t1_detail::kPrefixBytes> lower{};
  std::array<std::byte, t1_detail::kPrefixBytes> upper{};
  upper.fill(std::byte{0xff});
  uint64_t inversions = 0;
  uint64_t previous = 0;
  bool have_previous = false;
  store.impl().t1().scan(std::span<const std::byte>(lower),
                         std::span<const std::byte>(upper),
                         [&](std::span<const std::byte>, uint64_t payload, uint64_t hash) {
                           if (payload == vmemkv::STORE_NOT_FOUND || t1_detail::is_inline(hash)) {
                             return;
                           }
                           const uint64_t offset = payload & vmemkv::t2_detail::kPayloadOffsetMask;
                           if (have_previous && offset < previous) {
                             ++inversions;
                           }
                           previous = offset;
                           have_previous = true;
                         });
  return inversions;
}

struct ScanLayout {
  uint64_t t2_records = 0;
  uint64_t backward = 0;
  uint64_t forward_page_gap = 0;
  uint64_t base_records = 0;
  uint64_t tail_records = 0;
  // Actual T2 offsets per scan window in key order. Saved so the offset-based
  // fragmentation metric can be compared window-by-window with measured Scan time.
  std::vector<std::vector<uint64_t>> window_offsets;
  std::vector<uint64_t> window_backward;
};

template <typename Store>
auto scan_layout(Store &store, const std::vector<std::size_t> &starts) -> ScanLayout {
  ScanLayout layout;
  layout.window_offsets.reserve(starts.size());
  layout.window_backward.reserve(starts.size());
  const uint64_t page_size = static_cast<uint64_t>(::sysconf(_SC_PAGESIZE));
  const uint64_t base_boundary = store.impl().t2().get_memory()->base_boundary.load(std::memory_order_acquire);
  for (const std::size_t start : starts) {
    uint64_t previous = 0;
    bool have_previous = false;
    uint64_t window_backward = 0;
    std::vector<uint64_t> offsets;
    offsets.reserve(100);
    const std::string lower = bench_help::make_key(start);
    const std::string upper = bench_help::make_key(start + 99);
    const auto lo = std::as_bytes(std::span(lower));
    const auto hi = std::as_bytes(std::span(upper));
    store.impl().t1().scan(lo, hi, [&](std::span<const std::byte>, uint64_t payload, uint64_t hash) {
      if (payload == vmemkv::STORE_NOT_FOUND || t1_detail::is_inline(hash)) {
        return;
      }
      const uint64_t offset = payload & vmemkv::t2_detail::kPayloadOffsetMask;
      ++layout.t2_records;
      if (offset < base_boundary) {
        ++layout.base_records;
      } else {
        ++layout.tail_records;
      }
      offsets.push_back(offset);
      if (have_previous) {
        if (offset < previous) {
          ++layout.backward;
          ++window_backward;
        } else if (offset / page_size > previous / page_size + 1) {
          ++layout.forward_page_gap;
        }
      }
      previous = offset;
      have_previous = true;
    });
    layout.window_offsets.push_back(std::move(offsets));
    layout.window_backward.push_back(window_backward);
  }
  return layout;
}

[[noreturn]] void run_scan_update_scan_probe(const ProbeArgs &args) {
  using Store = vmemkv::variants::VMemKV_Default;
  constexpr std::size_t kScanWindows = 1000;
  constexpr std::size_t kUpdates = 1000;
  constexpr std::size_t kScanWidth = 100;

  if (!args.is_ltm || args.val_size != 1024) {
    fail("scan_update_scan_probe requires --scenario=ltm --value-size=1KB");
  }
  ::setenv("VMEMKV_SUPPRESS_AUTO_COMPACT", "1", 1);
  ::setenv("VMEMKV_SEGMENT_PROBE", "1", 1);

  const std::size_t key_count = args.key_count == 0 ? corpus_size_for_value(args.val_size) : args.key_count;
  if (key_count <= kScanWidth) {
    fail("scan_update_scan_probe corpus is too small");
  }
  const std::string filename = [&] {
    std::string out = get_db_dir() + "/bench_" + Store::name() + ".bin";
    std::replace(out.begin() + static_cast<std::ptrdiff_t>(get_db_dir().size() + 1), out.end(), '/', '_');
    return out;
  }();
  const std::string master_path = bench_master_corpus_path(filename, args.val_size, key_count, false);
  const std::string clone_path = master_path + "_clone";
  auto store = make_bench_fresh_corpus_checkpoint<Store>(filename, args.val_size, key_count, false);

  std::mt19937_64 scan_rng(bench_help::kBenchmarkSeed);
  bench_help::ZipfDistribution scan_zipf({key_count - kScanWidth, 1.0});
  std::vector<std::size_t> scan_starts;
  scan_starts.reserve(kScanWindows);
  for (std::size_t i = 0; i < kScanWindows; ++i) {
    scan_starts.push_back(scan_zipf(scan_rng));
  }

  const uint64_t before_offset_inversions = offset_inversions(*store);
  const MemPressureSnapshot mem_init = take_mem_snapshot();
  const std::size_t append_entries_before_first_scan = store->impl().t1().append_size();
  const uint64_t dirty_shards_before_first_scan = store->get_statistics().t2_dirty_shard_count;
  const size_t pending_segments_before_first_scan = store->impl().pending_segment_count();
  const bool compaction_running_before_first_scan = store->impl().compaction_running();
  const ScanLayout layout_before = scan_layout(*store, scan_starts);
  const bool cache_dropped_before = drop_page_cache();
  const ScanPhaseResult scan_before = scan_windows(*store, scan_starts);
  const MemPressureSnapshot mem_post_scan1 = take_mem_snapshot();

  const uint64_t t2_bytes_before = store->impl().t2().bytes_used();
  const std::string large_value(args.val_size, 'a');
  const std::string inline_value(8, 'a');
  std::mt19937_64 update_rng(bench_help::kBenchmarkSeed + 1);
  bench_help::ZipfDistribution update_zipf({key_count, 1.0});
  const auto update_start = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < kUpdates; ++i) {
    const std::size_t key_index = update_zipf(update_rng);
    const bool updated = use_mixed_8b_value(args.val_size, key_index)
                             ? store->update(bench_help::make_key(key_index), inline_value)
                             : store->update(bench_help::make_key(key_index), large_value);
    if (!updated) {
      fail("scan_update_scan_probe update missed a corpus key");
    }
  }
  const auto update_end = std::chrono::steady_clock::now();
  const std::size_t append_entries_after_updates = store->impl().t1().append_size();
  const uint64_t dirty_shards_after_updates = store->get_statistics().t2_dirty_shard_count;
  const size_t pending_segments_after_updates = store->impl().pending_segment_count();
  const bool compaction_running_after_updates = store->impl().compaction_running();
  const uint64_t t2_bytes_after_updates = store->impl().t2().bytes_used();
  const MemPressureSnapshot mem_post_update = take_mem_snapshot();

  const auto compaction_start = std::chrono::steady_clock::now();
  ::unsetenv("VMEMKV_SUPPRESS_AUTO_COMPACT");
  try {
    wait_for_background_compaction(store);
  } catch (const std::exception &error) {
    fail(error.what());
  }
  const auto compaction_end = std::chrono::steady_clock::now();
  const auto after_compaction = store->get_statistics();
  const auto pending_segments_after_compaction = store->impl().pending_segment_count();
  const bool compaction_running_after_compaction = store->impl().compaction_running();
  const std::size_t append_entries_after_compaction = store->impl().t1().append_size();
  const MemPressureSnapshot mem_post_drain = take_mem_snapshot();

  const auto checkpoint_start = std::chrono::steady_clock::now();
  store->checkpoint();
  const auto checkpoint_end = std::chrono::steady_clock::now();
  const std::size_t append_entries_after_checkpoint = store->impl().t1().append_size();
  const auto after_checkpoint = store->get_statistics();
  const MemPressureSnapshot mem_post_ckpt = take_mem_snapshot();
  const uint64_t after_offset_inversions = offset_inversions(*store);
  const ScanLayout layout_after = scan_layout(*store, scan_starts);
  const auto &compaction_metrics = store->impl().compaction_probe_state();

  const bool cache_dropped_after = drop_page_cache();
  const ScanPhaseResult scan_after = scan_windows(*store, scan_starts);
  const MemPressureSnapshot mem_post_scan2 = take_mem_snapshot();
  const double update_ms = std::chrono::duration<double, std::milli>(update_end - update_start).count();
  const double compaction_ms = std::chrono::duration<double, std::milli>(compaction_end - compaction_start).count();
  const double checkpoint_ms = std::chrono::duration<double, std::milli>(checkpoint_end - checkpoint_start).count();
  const double scan_ratio = scan_before.milliseconds_per_window == 0.0
                                ? 0.0
                                : scan_after.milliseconds_per_window / scan_before.milliseconds_per_window;

  std::cout
      << "{\"probe\":\"scan_update_scan\",\"scenario\":\"ltm\",\"value_size\":" << args.val_size << ",\"mode\":\""
      << (std::getenv("VMEMKV_SEGMENT_PROBE_BASELINE") == nullptr ? "with_replaced_keys" : "baseline") << "\""
      << ",\"key_count\":" << key_count << ",\"scan_windows\":" << kScanWindows << ",\"scan_width\":" << kScanWidth
      << ",\"updates\":" << kUpdates << ",\"cache_dropped_before\":" << (cache_dropped_before ? "true" : "false")
      << ",\"cache_dropped_after\":" << (cache_dropped_after ? "true" : "false")
      << ",\"append_entries_before_first_scan\":" << append_entries_before_first_scan
      << ",\"dirty_shards_before_first_scan\":" << dirty_shards_before_first_scan
      << ",\"pending_segments_before_first_scan\":" << pending_segments_before_first_scan
      << ",\"compaction_running_before_first_scan\":" << (compaction_running_before_first_scan ? "true" : "false")
      << ",\"append_entries_after_updates\":" << append_entries_after_updates
      << ",\"dirty_shards_after_updates\":" << dirty_shards_after_updates
      << ",\"pending_segments_after_updates\":" << pending_segments_after_updates
      << ",\"compaction_running_after_updates\":" << (compaction_running_after_updates ? "true" : "false")
      << ",\"scan_before_ms_per_window\":" << scan_before.milliseconds_per_window
      << ",\"scan_before_t1_only_ms_per_window\":" << scan_before.t1_only_milliseconds_per_window
      << ",\"scan_after_ms_per_window\":" << scan_after.milliseconds_per_window
      << ",\"scan_after_t1_only_ms_per_window\":" << scan_after.t1_only_milliseconds_per_window
      << ",\"scan_after_over_before\":" << scan_ratio << ",\"scan_before_major_faults\":" << scan_before.major_faults
      << ",\"scan_after_major_faults\":" << scan_after.major_faults
      << ",\"scan_before_minor_faults\":" << scan_before.minor_faults
      << ",\"scan_after_minor_faults\":" << scan_after.minor_faults
      << ",\"scan_before_input_blocks\":" << scan_before.input_blocks
      << ",\"scan_after_input_blocks\":" << scan_after.input_blocks
      << ",\"scan_before_maxrss_kb\":" << scan_before.max_rss_kb
      << ",\"scan_after_maxrss_kb\":" << scan_after.max_rss_kb
      << ",\"mem_init_ok\":" << (mem_init.ok ? "true" : "false") << ",\"mem_init_current\":" << mem_init.current
      << ",\"mem_init_peak\":" << mem_init.peak << ",\"mem_init_swap\":" << mem_init.swap_current
      << ",\"mem_init_pgfault\":" << mem_init.pgfault << ",\"mem_init_pgmajfault\":" << mem_init.pgmajfault
      << ",\"mem_scan1_ok\":" << (mem_post_scan1.ok ? "true" : "false")
      << ",\"mem_scan1_current\":" << mem_post_scan1.current << ",\"mem_scan1_peak\":" << mem_post_scan1.peak
      << ",\"mem_scan1_swap\":" << mem_post_scan1.swap_current << ",\"mem_scan1_pgfault\":" << mem_post_scan1.pgfault
      << ",\"mem_scan1_pgmajfault\":" << mem_post_scan1.pgmajfault
      << ",\"mem_update_ok\":" << (mem_post_update.ok ? "true" : "false")
      << ",\"mem_update_current\":" << mem_post_update.current << ",\"mem_update_peak\":" << mem_post_update.peak
      << ",\"mem_update_swap\":" << mem_post_update.swap_current
      << ",\"mem_update_pgfault\":" << mem_post_update.pgfault
      << ",\"mem_update_pgmajfault\":" << mem_post_update.pgmajfault
      << ",\"mem_drain_ok\":" << (mem_post_drain.ok ? "true" : "false")
      << ",\"mem_drain_current\":" << mem_post_drain.current << ",\"mem_drain_peak\":" << mem_post_drain.peak
      << ",\"mem_drain_swap\":" << mem_post_drain.swap_current << ",\"mem_drain_pgfault\":" << mem_post_drain.pgfault
      << ",\"mem_drain_pgmajfault\":" << mem_post_drain.pgmajfault
      << ",\"mem_ckpt_ok\":" << (mem_post_ckpt.ok ? "true" : "false")
      << ",\"mem_ckpt_current\":" << mem_post_ckpt.current << ",\"mem_ckpt_peak\":" << mem_post_ckpt.peak
      << ",\"mem_ckpt_swap\":" << mem_post_ckpt.swap_current << ",\"mem_ckpt_pgfault\":" << mem_post_ckpt.pgfault
      << ",\"mem_ckpt_pgmajfault\":" << mem_post_ckpt.pgmajfault
      << ",\"mem_scan2_ok\":" << (mem_post_scan2.ok ? "true" : "false")
      << ",\"mem_scan2_current\":" << mem_post_scan2.current << ",\"mem_scan2_peak\":" << mem_post_scan2.peak
      << ",\"mem_scan2_swap\":" << mem_post_scan2.swap_current << ",\"mem_scan2_pgfault\":" << mem_post_scan2.pgfault
      << ",\"mem_scan2_pgmajfault\":" << mem_post_scan2.pgmajfault << ",\"scan_records_before\":" << scan_before.records
      << ",\"scan_records_after\":" << scan_after.records << ",\"scan_value_bytes_before\":" << scan_before.value_bytes
      << ",\"scan_value_bytes_after\":" << scan_after.value_bytes
      << ",\"scan_base_records_before\":" << scan_before.base_records
      << ",\"scan_base_records_after\":" << scan_after.base_records
      << ",\"scan_tail_records_before\":" << scan_before.tail_records
      << ",\"scan_tail_records_after\":" << scan_after.tail_records << ",\"update_ms\":" << update_ms
      << ",\"update_ms_per_op\":" << update_ms / kUpdates << ",\"compaction_drain_ms\":" << compaction_ms
      << ",\"checkpoint_ms\":" << checkpoint_ms << ",\"compaction_cycles\":" << after_compaction.compaction_cycle_count
      << ",\"checkpoints_before_final_checkpoint\":" << after_compaction.checkpoint_count
      << ",\"t1_splits\":" << after_compaction.t1_split_count
      << ",\"dirty_shards_after\":" << after_compaction.t2_dirty_shard_count
      << ",\"append_entries_after_compaction\":" << append_entries_after_compaction
      << ",\"append_entries_after_checkpoint\":" << append_entries_after_checkpoint
      << ",\"dirty_shards_after_checkpoint\":" << after_checkpoint.t2_dirty_shard_count
      << ",\"pending_segments_after_compaction\":" << pending_segments_after_compaction
      << ",\"compaction_running_after_compaction\":" << (compaction_running_after_compaction ? "true" : "false")
      << ",\"probe_segment_visits\":" << compaction_metrics.probe_segment_visits.load()
      << ",\"probe_source_candidates\":" << compaction_metrics.probe_source_candidates.load()
      << ",\"probe_tracked_keys\":" << compaction_metrics.probe_tracked_keys.load()
      << ",\"probe_tracked_live\":" << compaction_metrics.probe_tracked_live.load()
      << ",\"probe_lookup_stale\":" << compaction_metrics.probe_lookup_stale.load()
      << ",\"probe_candidates_deduplicated\":" << compaction_metrics.probe_candidates_deduplicated.load()
      << ",\"probe_relocated\":" << compaction_metrics.probe_relocated.load()
      << ",\"probe_stale\":" << compaction_metrics.probe_stale.load()
      << ",\"probe_moved_bytes\":" << compaction_metrics.probe_moved_bytes.load()
      << ",\"probe_collect_ms\":" << compaction_metrics.probe_collect_ns.load() / 1e6
      << ",\"probe_scan_ms\":" << compaction_metrics.probe_scan_ns.load() / 1e6
      << ",\"probe_scan_entries\":" << compaction_metrics.probe_scan_entries.load()
      << ",\"probe_scan_matched\":" << compaction_metrics.probe_scan_matched.load()
      << ",\"probe_copy_bytes\":" << compaction_metrics.probe_copy_bytes.load()
      << ",\"probe_lookup_ms\":" << compaction_metrics.probe_lookup_ns.load() / 1e6
      << ",\"probe_sort_ms\":" << compaction_metrics.probe_sort_ns.load() / 1e6
      << ",\"probe_sort_candidates\":" << compaction_metrics.probe_sort_candidates.load()
      << ",\"probe_t2_key_reads\":" << compaction_metrics.probe_t2_key_reads.load()
      << ",\"probe_punched_segments\":" << compaction_metrics.probe_punched_segments.load()
      << ",\"probe_punched_bytes\":" << compaction_metrics.probe_punched_bytes.load()
      << ",\"probe_relocate_ms\":" << compaction_metrics.probe_relocate_ns.load() / 1e6
      << ",\"probe_validate_ms\":" << compaction_metrics.probe_validate_ns.load() / 1e6
      << ",\"probe_movedata_ms\":" << compaction_metrics.probe_movedata_ns.load() / 1e6
      << ",\"probe_publish_ms\":" << compaction_metrics.probe_publish_ns.load() / 1e6
      << ",\"scan_t2_records_before\":" << layout_before.t2_records
      << ",\"scan_t2_records_after\":" << layout_after.t2_records
      << ",\"scan_backward_before\":" << layout_before.backward << ",\"scan_backward_after\":" << layout_after.backward
      << ",\"scan_forward_page_gap_before\":" << layout_before.forward_page_gap
      << ",\"scan_forward_page_gap_after\":" << layout_after.forward_page_gap
      << ",\"scan_base_records_layout_before\":" << layout_before.base_records
      << ",\"scan_base_records_layout_after\":" << layout_after.base_records
      << ",\"scan_tail_records_layout_before\":" << layout_before.tail_records
      << ",\"scan_tail_records_layout_after\":" << layout_after.tail_records
      << ",\"t2_bytes_before\":" << t2_bytes_before << ",\"t2_bytes_after_updates\":" << t2_bytes_after_updates
      << ",\"t2_bytes_after_compaction\":" << store->impl().t2().bytes_used()
      << ",\"offset_inversions_before\":" << before_offset_inversions
      << ",\"offset_inversions_after\":" << after_offset_inversions << ",\"scan_starts\":[";
  for (size_t i = 0; i < scan_starts.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << scan_starts[i];
  }
  std::cout << "],\"scan_window_backward_before\":[";
  for (size_t i = 0; i < layout_before.window_backward.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << layout_before.window_backward[i];
  }
  std::cout << "],\"scan_window_backward_after\":[";
  for (size_t i = 0; i < layout_after.window_backward.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << layout_after.window_backward[i];
  }
  const auto print_doubles = [](const std::vector<double> &values) {
    for (size_t i = 0; i < values.size(); ++i) {
      if (i != 0) {
        std::cout << ",";
      }
      std::cout << values[i];
    }
  };
  const auto print_longs = [](const std::vector<long> &values) {
    for (size_t i = 0; i < values.size(); ++i) {
      if (i != 0) {
        std::cout << ",";
      }
      std::cout << values[i];
    }
  };
  std::cout << "],\"scan_window_ms_before\":[";
  print_doubles(scan_before.window_milliseconds);
  std::cout << "],\"scan_window_ms_after\":[";
  print_doubles(scan_after.window_milliseconds);
  std::cout << "],\"scan_window_majflt_before\":[";
  print_longs(scan_before.window_major_faults);
  std::cout << "],\"scan_window_majflt_after\":[";
  print_longs(scan_after.window_major_faults);
  std::cout << "],\"scan_window_minflt_before\":[";
  print_longs(scan_before.window_minor_faults);
  std::cout << "],\"scan_window_minflt_after\":[";
  print_longs(scan_after.window_minor_faults);
  std::cout << "],\"scan_window_inblock_before\":[";
  print_longs(scan_before.window_input_blocks);
  std::cout << "],\"scan_window_inblock_after\":[";
  print_longs(scan_after.window_input_blocks);
  std::cout << "],\"scan_window_offsets_before\":[";
  for (size_t i = 0; i < layout_before.window_offsets.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << "[";
    const auto &offsets = layout_before.window_offsets[i];
    for (size_t j = 0; j < offsets.size(); ++j) {
      if (j != 0) {
        std::cout << ",";
      }
      std::cout << offsets[j];
    }
    std::cout << "]";
  }
  std::cout << "],\"scan_window_offsets_after\":[";
  for (size_t i = 0; i < layout_after.window_offsets.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << "[";
    const auto &offsets = layout_after.window_offsets[i];
    for (size_t j = 0; j < offsets.size(); ++j) {
      if (j != 0) {
        std::cout << ",";
      }
      std::cout << offsets[j];
    }
    std::cout << "]";
  }
  std::cout << "]}" << std::endl;

  store.reset();
  vmemkv::remove_store_files(clone_path);
  std::_Exit(0);
}

[[noreturn]] void run_compaction_churn_probe(const ProbeArgs &args) {
  using Store = vmemkv::variants::VMemKV_Default;
  const std::string path = reorg_probe_path(args, "compaction_churn");
  const std::size_t writer_threads = resolve_writer_threads();
  const std::size_t key_count = args.key_count > 0 ? args.key_count : 500'000;
  const std::size_t val_size = args.val_size;
  const double duration_sec = args.duration_sec > 0 ? args.duration_sec : 60.0;

  auto store = make_vmemkv_fresh(
      path, [&path]() { return std::make_unique<Store>(path, Store::ConfigType::kDefaultCapacityBytes); });
  bench_help::populate(*store, {key_count, val_size}, /*shuffled=*/true);
  store->checkpoint();

  std::atomic<bool> stop{false};
  std::atomic<std::size_t> updated_count{0};
  std::vector<std::thread> workers;
  workers.reserve(writer_threads);
  for (std::size_t t = 0; t < writer_threads; ++t) {
    workers.emplace_back([&, t]() {
      std::mt19937_64 rng(bench_help::kBenchmarkSeed + t);
      bench_help::ZipfDistribution zipf({key_count, 1.0});
      while (!stop.load(std::memory_order_relaxed)) {
        const std::size_t idx = zipf(rng);
        store->update(bench_help::make_key(idx), bench_help::make_value_for_key(idx, val_size));
        updated_count.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  const auto t0 = std::chrono::steady_clock::now();
  auto elapsed_sec = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
  std::size_t last_updated = 0;
  double last_t = 0.0;

  const std::string t2_file = path;
  const std::string wal_file = path + ".wal";

  while (elapsed_sec() < duration_sec) {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    const double now = elapsed_sec();
    const std::size_t cur_updated = updated_count.load(std::memory_order_relaxed);
    const double dt = now - last_t;
    const double qps = dt > 0 ? static_cast<double>(cur_updated - last_updated) / dt : 0.0;
    last_t = now;
    last_updated = cur_updated;

    struct stat st_t2 {
    }, st_wal{};
    uint64_t t2_blocks = 0;
    uint64_t wal_blocks = 0;
    if (::stat(t2_file.c_str(), &st_t2) == 0) {
      t2_blocks = st_t2.st_blocks;
    }
    if (::stat(wal_file.c_str(), &st_wal) == 0) {
      wal_blocks = st_wal.st_blocks;
    }
    const uint64_t physical_bytes = (t2_blocks + wal_blocks) * 512;
    const auto stats = store->get_statistics();
    const uint64_t punch_backlog = store->impl().punch_backlog_segments();

    std::cout << "{\"probe\":\"compaction_churn\"," << "\"scenario\":\"" << (args.is_ltm ? "ltm" : "in_memory") << "\","
              << "\"value_size\":" << val_size << "," << "\"key_count\":" << key_count << ","
              << "\"writer_threads\":" << writer_threads << "," << "\"t_sec\":" << now << "," << "\"qps\":" << qps
              << "," << "\"checkpoint_count\":" << stats.checkpoint_count << ","
              << "\"compaction_count\":" << stats.compaction_cycle_count << "," << "\"punch_backlog\":" << punch_backlog
              << "," << "\"physical_bytes\":" << physical_bytes << ","
              << "\"dirty_shards\":" << stats.t2_dirty_shard_count << "}" << std::endl;
  }

  stop.store(true, std::memory_order_relaxed);
  for (auto &w : workers) {
    w.join();
  }

  store.reset();
  vmemkv::remove_store_files(path);
  std::_Exit(0);
}

[[noreturn]] void run(const ProbeArgs &args) {
  if (args.is_ltm) {
    // Matches run_bench.sh's real scenario_env_prefix() for "ltm" -- corpus_size_for_value()
    // below only scales with VMEMKV_BENCH_TARGET_RATIO when VMEMKV_BENCH_LTM is set (see that
    // function's in-memory fixed-constant branches), so both must be set before it's first called.
    // overwrite=0 on the ratio: this is a default matching run_bench.sh's own "ltm"
    // scenario, not a mandate -- a caller (e.g. a ratio sweep) that already exported
    // VMEMKV_BENCH_TARGET_RATIO before invoking this binary must win.
    ::setenv("VMEMKV_BENCH_LTM", "1", 1);
    ::setenv("VMEMKV_BENCH_TARGET_RATIO", "8.0", 0);
  }
  if (args.mode == ProbeMode::kBackgroundJobProbe) {
    run_background_job_probe(args);
  } else if (args.mode == ProbeMode::kCheckpointScaleProbe) {
    run_checkpoint_scale_probe(args);
  } else if (args.mode == ProbeMode::kScanUpdateScanProbe) {
    run_scan_update_scan_probe(args);
  } else if (args.mode == ProbeMode::kCompactionChurnProbe) {
    run_compaction_churn_probe(args);
  } else {
    run_organic_split_probe(args);
  }
}

}  // namespace reorg_probe

int main(int argc, char **argv) {
  std::cout.setf(std::ios::unitbuf);
  for (int i = 1; i < argc; ++i) {
    if (std::string_view(argv[i]) == "--reorg-probe") {
      reorg_probe::run(reorg_probe::parse_args(argc, argv));
    }
  }
  if (!should_skip_cleanup()) {
    cleanup_stale_benchmark_files();
  }
  register_benchmark_context();
  benchmark::Initialize(&argc, argv);
  register_all_benchmarks();
  benchmark::RunSpecifiedBenchmarks();
  {
    std::lock_guard<std::mutex> lock(g_active_store_mutex);
    if (g_active_store_holder) {
      g_active_store_holder->reset_store();
      g_active_store_holder = nullptr;
    }
  }
  benchmark::Shutdown();
  return 0;
}
// NOLINTEND
