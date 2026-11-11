#pragma once

#include <sys/mman.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <utility>
#include <vector>
#include <vmemkv/api/utils.hpp>
#include <vmemkv/core/reference_tracker.hpp>
#include <vmemkv/core/spin_backoff.hpp>

// alloc_len != value_len: fits-within-alloc updates overwrite in place without appending.
struct ValueRecordHeader {
  uint32_t key_len;
  uint32_t value_len;
  uint32_t alloc_len;
  uint64_t version;
};

static_assert(std::is_standard_layout_v<ValueRecordHeader>);
static_assert(alignof(ValueRecordHeader) == alignof(uint64_t));

// Zero-copy view into the T2 mapping. Valid while `mem` is alive.
struct T2RecordView {
  const ValueRecordHeader *header = nullptr;
  std::span<const std::byte> key;
  std::span<const std::byte> value;
};

inline auto make_record_view(const ValueRecordHeader *header) noexcept -> T2RecordView {
  const auto *key_begin = reinterpret_cast<const std::byte *>(header + 1);
  std::span<const std::byte> key(key_begin, header->key_len);
  std::span<const std::byte> value(key.data() + header->key_len, header->value_len);
  return T2RecordView{header, key, value};
}

namespace vmemkv {

// Four handles over the same [0, capacity) bytes; the per-record choice is made by
// try_read_base_record(). madvise is per mapping, hence one handle per readahead policy.
struct T2ReadHandles {
  // Primary read-write mapping, MADV_RANDOM. All appends/in-place updates land here;
  // small gets read here (1-op-1-record random access never pays for readahead).
  std::byte *random_mmap = nullptr;
  // Read-only, MADV_SEQUENTIAL. Small scans: the wide window folds many small faults
  // into few major faults.
  std::byte *sequential_mmap = nullptr;
  // Read-only, kernel default readahead. Large scans, plus large gets whose pages
  // mincore() found resident.
  std::byte *normal_mmap = nullptr;
  // dup()'d fd for one bounded pread() of a non-resident large get. -1 when the
  // best-effort setup failed; readers then fall back to random_mmap + seqlock.
  int pread_fd = -1;
};

// Single T2 mapping for the store's lifetime. Readers use the raw pointer directly;
// writers register via WriterGate below. See docs/specification/low_level_design.md 7.4-7.5 for the
// read-path rationale.
struct T2Memory {
  T2ReadHandles reads;
  uint64_t capacity = 0;

  // Monotonic append frontier. Mutable: mutated through the const pointer from get_memory().
  mutable std::atomic<uint64_t> bytes_used{0};
  // Durable + immutable prefix; reads below it skip the seqlock. Only grows.
  mutable std::atomic<uint64_t> base_boundary{0};

  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  T2Memory(std::byte *base_ptr, uint64_t capacity_bytes, uint64_t initial_bytes_used) noexcept
      : capacity(capacity_bytes), bytes_used(initial_bytes_used), base_boundary(initial_bytes_used) {
    reads.random_mmap = base_ptr;
  }

  ~T2Memory() noexcept;

  T2Memory(const T2Memory &) = delete;
  auto operator=(const T2Memory &) -> T2Memory & = delete;
  T2Memory(T2Memory &&) = delete;
  auto operator=(T2Memory &&) -> T2Memory & = delete;

  // Best-effort, never throws, at most once before publication.
  void setup_read_handles(int file_descriptor) noexcept;
};

struct NoOpAcquireWriteHandleHook {
  void operator()() const noexcept {}
};

// Groups the writer registry with its stop flag so register-then-check cannot drift apart.
class WriterGate {
 public:
  using Handle = ThreadReferenceTracker<const T2Memory *>::Guard;

  WriterGate() noexcept = default;

  WriterGate(const WriterGate &) = delete;
  auto operator=(const WriterGate &) -> WriterGate & = delete;

  // Registers first, then checks stop: checking first would miss a writer that slips
  // past wait_until_retired()'s single ordered pass after the stop flag went up.
  // The outer wait loop stays out of the registry so a rejected slot cannot flap and
  // starve stop_and_wait()'s drain check under sustained write load.
  template <typename GetMemory, typename Hook>
  auto acquire(GetMemory &&get_memory, Hook &&hook) const noexcept -> Handle {
    while (true) {
      vmemkv::SpinBackoff backoff;
      while (stop_.load(std::memory_order_seq_cst)) {
        backoff.wait();
      }
      const T2Memory *mem = get_memory();
      writers_.acquire(mem);
      hook();
      if (!stop_.load(std::memory_order_seq_cst)) {
        return {writers_, mem};
      }
      writers_.release();
    }
  }

  void stop_and_wait(const T2Memory *mem) const noexcept {
    stop_.store(true, std::memory_order_seq_cst);
    wait_until_retired(mem);
  }

  void wait_until_retired(const T2Memory *mem) const noexcept { writers_.wait_until_retired(mem); }

  void resume() const noexcept { stop_.store(false, std::memory_order_release); }

 private:
  // In-flight writers only; plain readers never register and never block quiesce.
  mutable ThreadReferenceTracker<const T2Memory *> writers_;
  mutable std::atomic<bool> stop_{false};
};

class T2FlatFile {
 public:
  using T2MemoryHandle = typename ThreadReferenceTracker<const T2Memory *>::Guard;

  // Always creates or adopts vmemkv::derive_t2_chk_path(path). nullopt discards any
  // untrusted bytes there; a value adopts them (caller must have validated the length
  // against a committed manifest).
  T2FlatFile(const std::filesystem::path &path,
             uint64_t bytes_capacity,
             std::optional<uint64_t> initial_bytes_used = std::nullopt);
  ~T2FlatFile() noexcept;

  T2FlatFile(const T2FlatFile &) = delete;
  auto operator=(const T2FlatFile &) -> T2FlatFile & = delete;

  // Stable for the store's lifetime; no handle needed. Does not register with the gate.
  auto get_memory() const noexcept -> const T2Memory * { return t2_mem_.load(std::memory_order_acquire); }

  // Hold until the T1 publish attempt returns, otherwise stop_writers_and_wait() can
  // consider the write quiesced before its T1 entry exists.
  template <typename Hook = NoOpAcquireWriteHandleHook>
  auto acquire_write_handle(Hook &&hook = {}) const noexcept -> T2MemoryHandle {
    return gate_.acquire([this]() noexcept { return get_memory(); }, std::forward<Hook>(hook));
  }

  void stop_writers_and_wait(const T2Memory *mem) const noexcept { gate_.stop_and_wait(mem); }
  void resume_writers() const noexcept { gate_.resume(); }

  static auto at(uint64_t payload, const T2Memory *mem) noexcept -> T2RecordView;

  // `mem` must come from acquire_write_handle().
  static auto append_default(const T2Memory *mem,
                             std::span<const std::byte> key,
                             std::span<const std::byte> value) -> uint64_t;
  // True when the new value fits alloc_len. Safe for distinct keys under the caller's stripe lock.
  static auto update_value_at(uint64_t payload, std::span<const std::byte> value, const T2Memory *mem) noexcept -> bool;

  // Hole-punches [offset, offset + len) after compaction evacuated it. Caller serializes
  // each segment and guarantees no live record starts in or spans into the range.
  enum class PunchOutcome : std::uint8_t { Punched, AlreadyHollow, Failed };
  auto punch_if_occupied(uint64_t offset, uint64_t len) const noexcept -> PunchOutcome;

  auto bytes_used() const noexcept -> uint64_t { return get_memory()->bytes_used.load(std::memory_order_acquire); }
  auto path() const noexcept -> const std::filesystem::path & { return path_; }

 private:
  static auto resolve_record(uint64_t payload, const T2Memory *mem) noexcept -> const std::byte * {
    assert(payload < mem->capacity);
    return mem->reads.random_mmap + payload;
  }

  static void create_empty_file(const std::filesystem::path &path, uint64_t bytes_capacity);

  void map_file(const std::filesystem::path &path, uint64_t bytes_capacity, uint64_t initial_bytes_used);

  // Identity path for deriving sibling paths; the mapped file is derive_t2_chk_path(path_).
  std::filesystem::path path_;

  std::atomic<const T2Memory *> t2_mem_{nullptr};

  mutable WriterGate gate_;
};

}  // namespace vmemkv
