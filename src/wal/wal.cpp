#include <fcntl.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <climits>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <system_error>
#include <thread>
#include <vector>
#include <vmemkv/api/utils.hpp>
#include <vmemkv/core/io.hpp>
#include <vmemkv/core/spin_backoff.hpp>
#include <vmemkv/wal/wal.hpp>

namespace vmemkv {

namespace {

auto compute_checksum(const WalRecordHeader &header,
                      // header/key/value order is the on-disk record order.
                      // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                      std::span<const std::byte> key,
                      std::span<const std::byte> value) noexcept -> uint64_t {
  // Same FNV chain as checksum_header_plus_spans did (header-zeroed, then spans in order),
  // folded through the single shared accumulator.
  FoldingChecksum sum;
  sum.add_header(header);
  sum.add(key);
  sum.add(value);
  return sum.value();
}

// Spin count at which a wait loop logs a stall tripwire (diagnostic only): unreachable in
// normal operation, fast enough to surface a genuine hang within seconds. Fires once per stall.
constexpr uint64_t kStallWarnThreshold = 20'000'000;
constexpr uint64_t kIterationsPerMillionForLog = 1'000'000;

inline void log_wal_stall(const char *role, uint64_t worker_id, size_t slot, uint64_t next_lsn) {
  // <print>/std::println needs GCC 14+; this toolchain (GCC 13) doesn't have it yet.
  std::fprintf(  // NOLINT(modernize-use-std-print)
      stderr,
      "wal: %s stalled >%luM spins waiting (id=%lu slot=%zu next_lsn_=%lu)\n",
      role,
      static_cast<unsigned long>(kStallWarnThreshold / kIterationsPerMillionForLog),
      static_cast<unsigned long>(worker_id),
      slot,
      static_cast<unsigned long>(next_lsn));
}

// Backoff plus one stall tripwire, shared by the three wait loops below. The tripwire reads
// SpinBackoff's own spin counter: one count, not two.
struct StallBackoff {
  SpinBackoff backoff;
  void wait() { backoff.wait(); }
  [[nodiscard]] auto should_warn() const -> bool { return backoff.spin_count == kStallWarnThreshold; }
};

}  // namespace

Wal::Wal(std::filesystem::path path) : path_(std::move(path)) {
  const std::vector<uint64_t> generations = discover_segments(path_);

  if (generations.empty()) {
    active_generation_ = 1;
    const auto seg_path = derive_wal_segment_path(path_, active_generation_);
    fd_.store(open_create_0600(seg_path.c_str(), O_RDWR | O_CREAT | O_APPEND, "open wal segment"),
              std::memory_order_relaxed);
  } else {
    active_generation_ = generations.back();
    uint64_t last_valid_lsn = 0;
    for (const uint64_t generation : generations) {
      const bool is_active = (generation == active_generation_);
      const auto seg_path = derive_wal_segment_path(path_, generation);
      FdGuard seg_fd{open_file(seg_path.c_str(), is_active ? (O_RDWR | O_APPEND) : O_RDONLY, "open wal segment")};
      // 0 means "this segment contributed nothing" (lsn 0 never exists) -- must not overwrite
      // a real value already found in an earlier generation, e.g. when the active segment is
      // still empty right after a rollover.
      const uint64_t found = scan_and_validate(seg_fd.fd, is_active);
      if (found > 0) {
        last_valid_lsn = found;
      }
      if (is_active) {
        fd_.store(seg_fd.release(), std::memory_order_relaxed);
      }
    }
    next_lsn_.store(last_valid_lsn + 1, std::memory_order_relaxed);
  }

  // Without this, a reopened WAL's next_to_flush_ would stay at 1 while next_lsn_ is far
  // ahead, and the first leader would spin forever on never-published slots.
  next_to_flush_ = next_lsn_.load(std::memory_order_relaxed);
}

Wal::~Wal() noexcept {
  const int local_fd = fd_.load(std::memory_order_relaxed);
  if (local_fd >= 0) {
    ::close(local_fd);
  }
}

auto Wal::discover_segments(const std::filesystem::path &wal_path) -> std::vector<uint64_t> {
  std::vector<uint64_t> generations;
  const std::filesystem::path dir = wal_path.has_parent_path() ? wal_path.parent_path() : ".";
  if (!std::filesystem::exists(dir)) {
    return generations;
  }
  const std::string prefix = wal_path.filename().string() + ".";
  for (const auto &entry : std::filesystem::directory_iterator(dir)) {
    const std::string name = entry.path().filename().string();
    if (name.size() <= prefix.size() || name.compare(0, prefix.size(), prefix) != 0) {
      continue;
    }
    const std::string suffix = name.substr(prefix.size());
    // Pure-digit suffixes only: guards against unrelated files sharing the prefix.
    if (!suffix.empty()) {
      uint64_t generation = 0;
      const auto conv = std::from_chars(suffix.data(), suffix.data() + suffix.size(), generation);
      if (conv.ec == std::errc{} && conv.ptr == suffix.data() + suffix.size()) {
        generations.push_back(generation);
      }
    }
  }
  std::sort(generations.begin(), generations.end());
  return generations;
}

auto Wal::scan_and_validate(int segment_fd, bool allow_truncate) -> uint64_t {
  const auto file_size = fstat_size(segment_fd, "fstat wal segment");

  uint64_t offset = 0;
  uint64_t last_valid_lsn = 0;

  while (file_size - offset >= sizeof(WalRecordHeader)) {
    const WalRecordHeader header = read_header_at(segment_fd, offset);

    if (header.magic != kWalRecordMagic || header.format_version != kWalFormatVersion) {
      break;  // Corrupt, or a record layout this build doesn't understand: stop here.
    }

    const uint64_t payload_len = static_cast<uint64_t>(header.key_len) + header.value_len;
    if (file_size - offset - sizeof(WalRecordHeader) < payload_len) {
      break;  // Torn payload: stop here.
    }

    const std::vector<std::byte> payload = read_payload_at(segment_fd, offset + sizeof(WalRecordHeader), payload_len);

    const std::span<const std::byte> key_span(payload.data(), header.key_len);
    const std::span<const std::byte> value_span(payload.data() + header.key_len, header.value_len);
    if (compute_checksum(header, key_span, value_span) != header.checksum) {
      break;  // Corrupt: stop here.
    }

    offset += sizeof(WalRecordHeader) + payload_len;
    last_valid_lsn = header.lsn;
  }

  if (offset < file_size) {
    if (!allow_truncate) {
      throw std::runtime_error("wal: corrupt tail found in a retired (non-active) segment");
    }
    if (::ftruncate(segment_fd, static_cast<off_t>(offset)) != 0) {
      throw std::system_error(errno, std::generic_category(), "ftruncate wal");
    }
  }

  return last_valid_lsn;
}

auto Wal::read_header_at(int segment_fd, uint64_t offset) -> WalRecordHeader {
  WalRecordHeader header;
  pread_exact(segment_fd, &header, sizeof(header), static_cast<off_t>(offset), "pread wal header");
  return header;
}

auto Wal::read_payload_at(int segment_fd, uint64_t offset, uint64_t payload_len) -> std::vector<std::byte> {
  std::vector<std::byte> payload(payload_len);
  pread_exact(segment_fd, payload.data(), payload_len, static_cast<off_t>(offset), "pread wal payload");
  return payload;
}

// See declaration in wal.hpp. noexcept: must never throw while already unwinding another
// exception.
void Wal::fail_all_pending_and_release_leadership(const std::exception_ptr &err) noexcept {
  poisoned_.store(true, std::memory_order_release);
  while (true) {
    const uint64_t target = next_lsn_.load(std::memory_order_acquire);
    while (next_to_flush_ != target) {
      const size_t slot = next_to_flush_ % kWalRingCapacity;
      PendingRecord *rec = ring_[slot].load(std::memory_order_acquire);
      StallBackoff wait_backoff;
      while (rec == nullptr) {
        wait_backoff.wait();
        rec = ring_[slot].load(std::memory_order_acquire);
      }
      rec->error = err;
      ring_[slot].store(nullptr, std::memory_order_release);
      rec->release();
      ++next_to_flush_;
    }
    if (next_lsn_.load(std::memory_order_acquire) == target) {
      break;
    }
  }
  // One shared wake-up for everything just failed. next_to_flush_ - 1 is the highest lsn
  // settled so far (next_to_flush_ itself was never reserved); re-notify is harmless when
  // there was nothing new (next_to_flush_ >= 1, so no underflow).
  highest_settled_lsn_.store(next_to_flush_ - 1, std::memory_order_release);
  settled_cv_.notify_all();
  flushing_.store(false, std::memory_order_release);
  flushing_.notify_all();
}

// Collects up to kWalRingCapacity records into `batch`. The bound is what makes slot reuse
// safe; a slot/LSN mismatch throws rather than being trusted.
void Wal::collect_batch(uint64_t target, std::vector<PendingRecord *> &batch) {
  while (next_to_flush_ != target && batch.size() < kWalRingCapacity) {
    const size_t slot = next_to_flush_ % kWalRingCapacity;
    PendingRecord *rec = ring_[slot].load(std::memory_order_acquire);
    StallBackoff publish_backoff;
    while (rec == nullptr) {
      // fetch_add for this LSN already happened; the producer's CAS-publish just hasn't landed.
      publish_backoff.wait();
      rec = ring_[slot].load(std::memory_order_acquire);
      if (publish_backoff.should_warn()) {
        log_wal_stall("leader", next_to_flush_, slot, next_lsn_.load(std::memory_order_relaxed));
      }
    }
    if (rec->lsn != next_to_flush_) {
      throw std::logic_error("wal ring buffer invariant violated: slot lsn mismatch");
    }
    batch.push_back(rec);
    ++next_to_flush_;
  }
}

// Chunked writev() plus one shared fdatasync(). A failure fails the whole batch, truncates
// back to the batch's starting length (a failed record must never resurface on replay), and
// poisons the Wal. Notifies and releases every record regardless of outcome.
void Wal::write_and_fsync_batch(const std::vector<PendingRecord *> &batch) {
  const int local_fd = fd_.load(std::memory_order_acquire);
  struct stat file_stat {};
  const uint64_t saved_length = (::fstat(local_fd, &file_stat) == 0) ? static_cast<uint64_t>(file_stat.st_size) : 0;

  std::exception_ptr io_err;
  try {
    std::vector<iovec> iov;
    iov.reserve(std::min(batch.size(), static_cast<size_t>(IOV_MAX)));
    size_t offset = 0;
    while (offset < batch.size()) {
      const size_t chunk = std::min(batch.size() - offset, static_cast<size_t>(IOV_MAX));
      iov.clear();
      size_t chunk_bytes = 0;
      for (size_t i = 0; i < chunk; ++i) {
        PendingRecord *rec = batch[offset + i];
        // writev() takes non-const iovecs by convention only; it never writes through them.
        iov.push_back(iovec{const_cast<std::byte *>(rec->buffer.data()), rec->buffer.size()});
        chunk_bytes += rec->buffer.size();
      }
      const ssize_t written = ::writev(local_fd, iov.data(), static_cast<int>(iov.size()));
      if (written != static_cast<ssize_t>(chunk_bytes)) {
        throw std::system_error(errno, std::generic_category(), "writev wal batch");
      }
      offset += chunk;
    }
    // fdatasync() rather than fsync() -- see class contract in wal.hpp for why.
    const auto fsync_start = std::chrono::steady_clock::now();
    if (::fdatasync(local_fd) != 0) {
      throw std::system_error(errno, std::generic_category(), "fdatasync wal");
    }
    const auto fsync_dur = std::chrono::steady_clock::now() - fsync_start;
    wal_debug::g_fsync_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(fsync_dur).count(),
                                    std::memory_order_relaxed);
    wal_debug::g_fsync_count.fetch_add(1, std::memory_order_relaxed);
    wal_debug::g_batch_records.fetch_add(batch.size(), std::memory_order_relaxed);
  } catch (...) {
    io_err = std::current_exception();
    const int truncate_result = ::ftruncate(local_fd, static_cast<off_t>(saved_length));
    (void)truncate_result;  // Best-effort; nothing more we can do if this also fails.
    poisoned_.store(true, std::memory_order_release);
  }

  // back() is this round's highest: collect_batch() appends in next_to_flush_ order, and
  // nothing below has released a record yet.
  const uint64_t round_last_lsn = batch.back()->lsn;

  for (auto *rec : batch) {
    rec->error = io_err;
  }

  // One shared wake-up for the whole round instead of a per-record notify_one() loop.
  highest_settled_lsn_.store(round_last_lsn, std::memory_order_release);
  settled_cv_.notify_all();

  for (auto *rec : batch) {
    ring_[rec->lsn % kWalRingCapacity].store(nullptr, std::memory_order_release);
    rec->release();
  }
}

auto Wal::drain_pending() -> uint64_t {
  // Snapshot the target once: re-reading next_lsn_ each pass would chase a moving target under
  // sustained writers and never return. Rounds still repeat (collect_batch() caps each at
  // kWalRingCapacity), bounded by the entry-time backlog instead.
  uint64_t target = next_lsn_.load(std::memory_order_acquire);

  // Adaptive group commit: if only 1 record is pending in this round, briefly pause
  // (up to ~5 microseconds) to coalesce back-to-back concurrent writes into a single fsync.
  // Breaks early the moment next_lsn_ advances past target.
  if (target > next_to_flush_ && (target - next_to_flush_) == 1) {
    constexpr uint64_t kMaxAdaptiveSpins = 100;
    for (uint64_t spin = 0; spin < kMaxAdaptiveSpins; ++spin) {
      if (next_lsn_.load(std::memory_order_relaxed) != target) {
        break;
      }
#if defined(__x86_64__) || defined(_M_X64)
      _mm_pause();
#elif defined(__aarch64__)
      asm volatile("yield" ::: "memory");
#else
      std::this_thread::yield();
#endif
    }
    target = next_lsn_.load(std::memory_order_acquire);
  }

  while (next_to_flush_ != target) {
    try {
      std::vector<PendingRecord *> batch;
      collect_batch(target, batch);
      write_and_fsync_batch(batch);
    } catch (...) {
      fail_all_pending_and_release_leadership(std::current_exception());
      throw;
    }
  }
  return target;
}

auto Wal::release_leadership(uint64_t observed_target) -> bool {
  flushing_.store(false, std::memory_order_release);
  flushing_.notify_all();
  if (next_lsn_.load(std::memory_order_acquire) == observed_target) {
    return false;
  }
  // An LSN landed in the release window: reclaim leadership rather than stranding it -- unless
  // someone else already did.
  if (flushing_.exchange(true, std::memory_order_acq_rel)) {
    return false;
  }
  return true;
}

auto Wal::reserve_record(WalRecordType type,
                         std::span<const std::byte> key,
                         std::span<const std::byte> value) -> PendingRecord * {
  if (poisoned_.load(std::memory_order_acquire)) {
    throw std::runtime_error("WAL is poisoned after an earlier durability failure");
  }

  WalRecordHeader header;
  header.key_len = static_cast<uint32_t>(key.size());
  header.value_len = static_cast<uint32_t>(value.size());
  header.type = static_cast<uint8_t>(type);

  auto owned_rec = std::make_unique<PendingRecord>();
  owned_rec->buffer.resize(sizeof(header) + key.size() + value.size());
  if (!key.empty()) {
    std::memcpy(owned_rec->buffer.data() + sizeof(header), key.data(), key.size());
  }
  if (!value.empty()) {
    std::memcpy(owned_rec->buffer.data() + sizeof(header) + key.size(), value.data(), value.size());
  }

  const uint64_t lsn = next_lsn_.fetch_add(1, std::memory_order_acq_rel);
  owned_rec->lsn = lsn;
  header.lsn = lsn;
  header.checksum = compute_checksum(header, key, value);
  std::memcpy(owned_rec->buffer.data(), &header, sizeof(header));

  // Ownership transfers to the ring (PendingRecord::refcount already accounts for both the
  // producer's and the ring/flusher's reference -- see its declaration).
  PendingRecord *rec = owned_rec.release();

  const size_t slot = lsn % kWalRingCapacity;
  PendingRecord *expected = nullptr;
  StallBackoff slot_backoff;
  while (!ring_[slot].compare_exchange_weak(expected, rec, std::memory_order_acq_rel)) {
    expected = nullptr;
    // Backpressure: the prior occupant hasn't retired yet. Rare given the ring's margin over
    // max concurrent callers.
    slot_backoff.wait();
    if (slot_backoff.should_warn()) {
      log_wal_stall("producer", lsn, slot, next_lsn_.load(std::memory_order_relaxed));
    }
  }

  return rec;
}

auto Wal::await_durable(PendingRecord *rec) -> uint64_t {
  const uint64_t lsn = rec->lsn;
  bool is_leader = !flushing_.exchange(true, std::memory_order_acq_rel);

  if (!is_leader) {
    if (highest_settled_lsn_.load(std::memory_order_acquire) < lsn) {
      // Bounded wait_for(), not a spin (spins steal cycles from concurrent work) and not an
      // unbounded wait: a leader handoff in the release-check gap can strand this follower on a
      // notify that never names it. On timeout with the LSN still unsettled and no active
      // leader, self-elect instead of waiting forever.
      constexpr auto kFollowerWaitTimeout = std::chrono::milliseconds(5);
      std::unique_lock<std::mutex> lock(settled_mutex_);
      while (highest_settled_lsn_.load(std::memory_order_acquire) < lsn) {
        const bool timed_out = settled_cv_.wait_for(lock, kFollowerWaitTimeout) == std::cv_status::timeout;
        if (timed_out && highest_settled_lsn_.load(std::memory_order_acquire) < lsn) {
          lock.unlock();
          if (!flushing_.exchange(true, std::memory_order_acq_rel)) {
            is_leader = true;
            break;
          }
          lock.lock();
        }
      }
    }
  }

  if (!is_leader) {
    const std::exception_ptr err = rec->error;
    rec->release();
    if (err) {
      std::rethrow_exception(err);
    }
    return lsn;
  }

  while (true) {
    const uint64_t target = drain_pending();
    if (!release_leadership(target)) {
      break;
    }
  }

  const std::exception_ptr err = rec->error;
  rec->release();
  if (err) {
    std::rethrow_exception(err);
  }
  return lsn;
}

auto Wal::reserve_insert(std::span<const std::byte> key, std::span<const std::byte> value) -> PendingRecord * {
  return reserve_record(WalRecordType::Insert, key, value);
}

auto Wal::reserve_update(std::span<const std::byte> key, std::span<const std::byte> value) -> PendingRecord * {
  return reserve_record(WalRecordType::Update, key, value);
}

auto Wal::reserve_delete(std::span<const std::byte> key) -> PendingRecord * {
  return reserve_record(WalRecordType::Delete, key, std::span<const std::byte>{});
}

auto Wal::replay(const WalReplayCallback &callback) const -> uint64_t {
  uint64_t count = 0;

  for (const uint64_t generation : discover_segments(path_)) {
    const auto seg_path = derive_wal_segment_path(path_, generation);
    const vmemkv::FdGuard fd_guard{open_file(seg_path.c_str(), O_RDONLY, "open wal segment for replay")};
    const int seg_fd = fd_guard.fd;
    const auto file_size = fstat_size(seg_fd, "fstat wal segment replay");

    uint64_t offset = 0;
    while (file_size - offset >= sizeof(WalRecordHeader)) {
      const WalRecordHeader header = read_header_at(seg_fd, offset);
      const uint64_t payload_len = static_cast<uint64_t>(header.key_len) + header.value_len;
      const std::vector<std::byte> payload = read_payload_at(seg_fd, offset + sizeof(WalRecordHeader), payload_len);

      const std::span<const std::byte> key_span(payload.data(), header.key_len);
      const std::span<const std::byte> value_span(payload.data() + header.key_len, header.value_len);

      callback(static_cast<WalRecordType>(header.type), key_span, value_span, header.lsn);
      ++count;

      offset += sizeof(WalRecordHeader) + payload_len;
    }
  }

  return count;
}

// Deleting generation (new_generation - 2) is safe: cycles run single-flight, so this call
// retires exactly the generation active during the *previous* cycle. That generation holds the
// previous checkpoint's covered records plus spillover reserved mid-cycle -- but this cycle's
// checkpoint_lsn (captured before this call) already covers everything reserved as of the
// previous rollover, since that rollover preceded this cycle. The one-generation slack covers
// the in-between spillover with no per-record bookkeeping.
void Wal::rotate_segment() {
  if (poisoned_.load(std::memory_order_acquire)) {
    throw std::runtime_error("WAL is poisoned, refusing to rotate");
  }

  // Take leadership rather than "wait until flushing_ is false" (TOCTOU: another thread could
  // grab it between wake and act). No drain first: a record reserved just before this swap may
  // land in either segment -- both replay correctly. Leadership exists only so no
  // write_and_fsync_batch() is mid-flight against the fd closed below.
  //
  // Timed: under sustained writers this wait dominates rotate_segment()'s (and therefore the
  // checkpoint's) wall-clock cost; open/close/unlink are consistently sub-millisecond.
  const auto leader_wait_start = std::chrono::steady_clock::now();
  while (flushing_.exchange(true, std::memory_order_acq_rel)) {
    flushing_.wait(true, std::memory_order_acquire);
  }
  last_rotate_leader_wait_us_.store(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                                              std::chrono::steady_clock::now() - leader_wait_start)
                                                              .count()),
                                    std::memory_order_relaxed);

  const int old_fd = fd_.load(std::memory_order_acquire);
  const uint64_t new_generation = active_generation_ + 1;
  const auto new_path = derive_wal_segment_path(path_, new_generation);
  int new_fd = -1;
  try {
    new_fd = open_create_0600(new_path.c_str(), O_RDWR | O_CREAT | O_EXCL | O_APPEND, "open new wal segment");
  } catch (...) {
    const auto err = std::current_exception();
    fail_all_pending_and_release_leadership(err);
    std::rethrow_exception(err);
  }

  fd_.store(new_fd, std::memory_order_release);
  active_generation_ = new_generation;
  flushing_.store(false, std::memory_order_release);
  flushing_.notify_all();

  // fd_close_mu_ pairs with rotate_segment()'s lock around close(old_fd): the fd snapshotted
  // here can still be the outgoing one even though fd_ already points at the new fd.
  {
    std::lock_guard<std::mutex> lock(fd_close_mu_);
    ::close(old_fd);
  }

  // Safe when generation (new_generation - 2) is missing (first cycles, interrupted priors).
  if (new_generation >= 2) {
    remove_quiet(derive_wal_segment_path(path_, new_generation - 2));
  }
}

auto Wal::next_lsn() const noexcept -> uint64_t { return next_lsn_.load(std::memory_order_relaxed); }

auto Wal::size_bytes() const -> uint64_t {
  // Without the lock this fstat() can race close() of the snapshotted outgoing fd: EBADF, or
  // worse a reused fd number silently sizing an unrelated file. Lock-hold is tiny (close only)
  // and sampling is strided, not per-write.
  std::lock_guard<std::mutex> lock(fd_close_mu_);
  const int current_fd = fd_.load(std::memory_order_acquire);
  return fstat_size(current_fd, "fstat wal (size_bytes)");
}

}  // namespace vmemkv
