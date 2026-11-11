// wal.hpp - Write-Ahead Log for vmemkv crash recovery.
#pragma once

#include <fcntl.h>

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <functional>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>
#include <vmemkv/api/utils.hpp>
#include <vmemkv/checkpoint/paths.hpp>
#include <vmemkv/core/io.hpp>

namespace vmemkv {

enum class WalRecordType : uint8_t { Insert = 1, Update = 2, Delete = 3 };

inline constexpr uint32_t kWalRecordMagic = 0x574C4B31;  // ASCII "1KLW", sentinel for torn/corrupt detection.

// Discriminates on-disk record layout, independent of kWalRecordMagic (which only flags "looks
// like a header"). Bump on format changes rather than overloading magic.
inline constexpr uint8_t kWalFormatVersion = 1;

// 8+8+4+4+4+1+1+2 = 32B, naturally aligned, no implicit padding.
struct WalRecordHeader {
  uint64_t lsn = 0;
  uint64_t checksum = 0;  // FNV-1a64 over header(checksum zeroed)+key+value.
  uint32_t magic = kWalRecordMagic;
  uint32_t key_len = 0;
  uint32_t value_len = 0;  // 0 for Delete.
  uint8_t type = 0;
  uint8_t format_version = kWalFormatVersion;
  // NOLINTNEXTLINE(modernize-avoid-c-arrays)
  uint8_t reserved[2] = {};  // Explicit so the checksum input has no UB padding bytes.
};
inline constexpr size_t kWalRecordHeaderBytes = 32;
static_assert(sizeof(WalRecordHeader) == kWalRecordHeaderBytes);
static_assert(std::is_standard_layout_v<WalRecordHeader>);

using WalReplayCallback = std::function<void(
    WalRecordType type, std::span<const std::byte> key, std::span<const std::byte> value, uint64_t lsn)>;

namespace wal_debug {
inline std::atomic<uint64_t> g_fsync_ns{0};
inline std::atomic<uint64_t> g_fsync_count{0};
inline std::atomic<uint64_t> g_batch_records{0};

inline void reset() noexcept {
  g_fsync_ns.store(0, std::memory_order_relaxed);
  g_fsync_count.store(0, std::memory_order_relaxed);
  g_batch_records.store(0, std::memory_order_relaxed);
}
}  // namespace wal_debug

// Sequential Write-Ahead Log with a split reserve/await API and lock-free group commit.
//
// reserve_insert()/reserve_update()/reserve_delete() + await_durable() together append one
// record. reserve_*() fixes the LSN order (via atomic fetch_add) and CAS-publishes into a
// fixed ring -- no blocking wait, so a caller can reserve while holding its per-key lock and
// release it before the durability wait. await_durable() elects a leader per round (atomic
// exchange); the leader writes the round via chunked writev() plus one shared fdatasync()
// (fdatasync skips syncing file metadata that doesn't affect retrieval). A round failure
// fails every record in it uniformly; all participants wake on one shared
// highest_settled_lsn_ counter.
//
// Recovery: the constructor discovers every segment on disk, replays in generation order,
// and truncates at the first invalid record (torn header/payload, bad magic or version,
// checksum mismatch) in the newest segment only -- older segments retired under
// rotate_segment()'s exclusive leadership, so only the active one can have a torn tail.
class Wal {
 public:
  // One caller's serialized record queued for a group-commit round. Intrusively refcounted
  // with exactly two owners (producer + ring/flusher); last release deletes. Not
  // producer-owned (the flusher could touch freed memory) and not atomic<shared_ptr>
  // (libstdc++ spinlocks its refcount, defeating lock-freedom).
  //
  // Public only so reserve_*()/await_durable() can name it; treat as opaque.
  struct PendingRecord {
    // Plain vector rather than an inline small-buffer-optimized array: glibc's malloc is fast
    // enough here that allocation count isn't the bottleneck.
    std::vector<std::byte> buffer;  // fully serialized header(lsn/checksum patched)+key+value
    uint64_t lsn = 0;
    std::exception_ptr error;
    std::atomic<int> refcount{2};  // producer's ref + ring-slot/flusher's ref

    void release() {
      if (refcount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        delete this;
      }
    }
  };

  explicit Wal(std::filesystem::path path);
  ~Wal() noexcept;

  Wal(const Wal &) = delete;
  auto operator=(const Wal &) -> Wal & = delete;
  Wal(Wal &&) = delete;
  auto operator=(Wal &&) -> Wal & = delete;

  // Each reserve_*() value must reach exactly one await_durable(): skipping it leaks the
  // record and hides write failures ([[nodiscard]] catches that at compile time).
  [[nodiscard]] auto reserve_insert(std::span<const std::byte> key,
                                    std::span<const std::byte> value) -> PendingRecord *;
  [[nodiscard]] auto reserve_update(std::span<const std::byte> key,
                                    std::span<const std::byte> value) -> PendingRecord *;
  [[nodiscard]] auto reserve_delete(std::span<const std::byte> key) -> PendingRecord *;
  auto await_durable(PendingRecord *rec) -> uint64_t;

  // Reads the whole log from the beginning and invokes callback for each valid record in order.
  // Returns the number of records replayed.
  // NOLINTNEXTLINE(modernize-use-nodiscard) -- recover_from_wal() legitimately ignores the count.
  auto replay(const WalReplayCallback &callback) const -> uint64_t;

  // Every generation with a segment file on disk, ascending. For tests/tooling (inspection,
  // corruption injection, cleanup) without a live Wal; production never needs it.
  static auto discover_segments(const std::filesystem::path &wal_path) -> std::vector<uint64_t>;

  // Rolls to a fresh segment (docs/specification/low_level_design.md 5.5) and deletes the segment two
  // generations back -- safe by construction, see the .cpp retention argument. Takes no LSN:
  // no per-record scan. Briefly takes leadership only to swap fd_ without racing an
  // in-flight write_and_fsync_batch(); does not drain first (a record landing in either
  // segment is equally correct).
  void rotate_segment();

  [[nodiscard]] auto next_lsn() const noexcept -> uint64_t;

  // Active segment's on-disk size: "bytes since the last checkpoint" for the WAL-size
  // trigger, since each cycle starts exactly one fresh empty segment.
  [[nodiscard]] auto size_bytes() const -> uint64_t;

  // Leader-wait time of the most recent rotate_segment(). Dominates checkpoint wall-clock
  // under sustained concurrent writers; scales with leader busyness, not data volume.
  [[nodiscard]] auto last_rotate_leader_wait_us() const noexcept -> uint64_t {
    return last_rotate_leader_wait_us_.load(std::memory_order_relaxed);
  }

 private:
  // Above the 256-way stripe lock's max concurrent callers (throughput tuning only; wraparound
  // safety is collect_batch()'s bound plus its slot/LSN check).
  static constexpr size_t kWalRingCapacity = 4096;
  static_assert((kWalRingCapacity & (kWalRingCapacity - 1)) == 0, "must be a power of two");

  // Fast half of the split API: serialize, fetch_add an LSN, CAS-publish into the ring.
  // Shared by reserve_insert()/reserve_update()/reserve_delete().
  [[nodiscard]] auto reserve_record(WalRecordType type,
                                    std::span<const std::byte> key,
                                    std::span<const std::byte> value) -> PendingRecord *;

  // Validates one segment, returning its last valid lsn (0 if none). allow_truncate only for
  // the active (newest) segment -- the only one writable at crash time. A torn tail elsewhere
  // is an integrity failure and throws instead of truncating.
  static auto scan_and_validate(int segment_fd, bool allow_truncate) -> uint64_t;

  // Writes+fsyncs everything up to the entry-time next_lsn_ snapshot, absorbing mid-flush
  // backlog. Funnels any failure (I/O or otherwise) through
  // fail_all_pending_and_release_leadership(), so nobody is ever left stranded.
  auto drain_pending() -> uint64_t;

  // Collects up to kWalRingCapacity pending records into `batch`. The bound is what makes
  // slot reuse safe; a slot/LSN mismatch throws rather than being trusted.
  void collect_batch(uint64_t target, std::vector<PendingRecord *> &batch);

  // Chunked writev() plus one shared fsync() per batch; notifies and releases every record
  // regardless of outcome. See wal.cpp for failure semantics and the ftruncate reasoning.
  void write_and_fsync_batch(const std::vector<PendingRecord *> &batch);

  // Fails everything outstanding, poisons the Wal, releases leadership. From
  // drain_pending()'s catch block only; never throws.
  void fail_all_pending_and_release_leadership(const std::exception_ptr &err) noexcept;

  // Releases leadership; reclaims it (returns true: drain again) when an LSN was reserved in
  // the release window, instead of stranding that record.
  auto release_leadership(uint64_t observed_target) -> bool;

  // pread_exact wrappers for scan_and_validate()/replay(), which read arbitrary segments
  // (not just the active fd_).
  [[nodiscard]] static auto read_header_at(int segment_fd, uint64_t offset) -> WalRecordHeader;
  [[nodiscard]] static auto read_payload_at(int segment_fd,
                                            uint64_t offset,
                                            uint64_t payload_len) -> std::vector<std::byte>;

  // Atomic: replay()/size_bytes() race rotate_segment()'s close-then-reassign on a plain int
  // (data race plus a close-then-fd-reused hazard).
  std::atomic<int> fd_{-1};
  // Serializes rotate_segment()'s close(old_fd) against size_bytes()'s fstat(fd_).
  mutable std::mutex fd_close_mu_;
  // Published by rotate_segment(); see last_rotate_leader_wait_us() above.
  std::atomic<uint64_t> last_rotate_leader_wait_us_{0};
  std::filesystem::path path_;
  // Written only under flushing_ leadership (rotate_segment() is the only writer).
  uint64_t active_generation_ = 1;

  // Slots hold pointers only; lifetime is PendingRecord::refcount's, independent of reuse.
  std::array<std::atomic<PendingRecord *>, kWalRingCapacity> ring_{};
  std::atomic<uint64_t> next_lsn_{1};  // fetch_add assigns both the LSN and its ring slot.
  // Leader-only (flushing_ == true); no atomic needed.
  uint64_t next_to_flush_ = 1;
  std::atomic<bool> flushing_{false};  // Exchange-elected leader flag.
  // Highest LSN safe to read PendingRecord::error for ("settled", not necessarily durable).
  // Followers check this atomically before waiting on settled_cv_.
  mutable std::mutex settled_mutex_;
  mutable std::condition_variable settled_cv_;
  std::atomic<uint64_t> highest_settled_lsn_{0};
  // Sticky after a round failure: later reserves fail fast. Treat as process-fatal.
  std::atomic<bool> poisoned_{false};
};

// Test/tooling: active (highest-generation) segment path, or nullopt if none exist.
// Bypasses Wal for filesystem-level inspection/injection; production never needs it.
inline auto find_active_wal_segment(const std::filesystem::path &wal_path) -> std::optional<std::filesystem::path> {
  const auto generations = Wal::discover_segments(wal_path);
  if (generations.empty()) {
    return std::nullopt;
  }
  return derive_wal_segment_path(wal_path, generations.back());
}

// Test/tooling helper: removes every existing segment file for this WAL identity.
inline void remove_wal_segments(const std::filesystem::path &wal_path) {
  for (const uint64_t generation : Wal::discover_segments(wal_path)) {
    remove_quiet(derive_wal_segment_path(wal_path, generation));
  }
}

// Removes every file a VMemKVImpl store could have created at `t2_path` (data file, WAL
// segments, manifest, checkpoints). Shared by tests and benchmark tools.
inline void remove_store_files(const std::filesystem::path &t2_path) {
  remove_quiet(t2_path);
  remove_wal_segments(derive_wal_path(t2_path));
  remove_quiet(derive_manifest_path(t2_path));
  remove_quiet(derive_t1_chk_path(t2_path));
  remove_quiet(derive_t2_chk_path(t2_path));
}

// Test helper: appends a header declaring `payload_len` payload bytes with only `partial`
// following, simulating a torn mid-append crash. Recovery truncates it.
inline void append_truncated_record_for_test(const std::filesystem::path &seg_path,
                                             uint64_t payload_len,
                                             std::string_view partial) {
  WalRecordHeader header{};
  constexpr uint32_t kTruncatedKeyLen = 5;  // arbitrary short key: only truncation matters here.
  header.key_len = kTruncatedKeyLen;
  header.value_len = static_cast<uint32_t>(payload_len - kTruncatedKeyLen);
  header.type = static_cast<uint8_t>(WalRecordType::Insert);
  header.magic = kWalRecordMagic;
  header.format_version = kWalFormatVersion;
  FdGuard guard{open_file(seg_path.c_str(), O_WRONLY | O_APPEND, "append truncated record for test")};
  write_all_exact(guard.fd, &header, sizeof(header), "append truncated record for test");
  write_all_exact(guard.fd, partial.data(), partial.size(), "append truncated record for test");
}

}  // namespace vmemkv
