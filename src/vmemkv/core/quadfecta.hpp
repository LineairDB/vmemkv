// quadfecta.hpp - Standalone Linux four-handle adaptive file reader ("Quadfecta").
//
// Maintains up to four OS views over the same backing file, selecting one per read by
// (Access, size <= page_size, Policy):
// - Random          + <= page: MADV_RANDOM mmap (zero-copy, no readahead pollution)
// - Random          +  > page: bounded pread() via independent fd + POSIX_FADV_RANDOM
// - Sequential      +     any: MADV_SEQUENTIAL mmap (wide kernel readahead + dropbehind, lazy VMA)
// - SequentialReuse +     any: MADV_NORMAL mmap (adaptive readahead + page retention, lazy VMA)
//
// Dependencies: C++20/23 STL + Linux POSIX (<fcntl.h>, <sys/mman.h>, <sys/stat.h>, <unistd.h>).
// Thread safety: all const read methods are lock-free on the hot path and thread-safe when callers
// supply their own `scratch` buffer or `read_into()` destination (when `scratch == nullptr`, both
// Policy::PreadOnly and Policy::Adaptive fall back to a thread_local buffer). Any concurrent writes
// to the backing file must be externally bounded below `valid_end`.
#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <system_error>
#include <utility>
#include <vector>

namespace quadfecta {

static_assert(sizeof(off_t) >= sizeof(int64_t), "quadfecta requires 64-bit off_t");
static_assert(sizeof(size_t) >= sizeof(uint64_t), "quadfecta requires 64-bit size_t");

enum class Access : std::uint8_t {
  Random,           // Single-record point access (avoid readahead).
  Sequential,       // One-shot contiguous range scan (MADV_SEQUENTIAL across all sizes).
  SequentialReuse,  // Repeatedly scanned contiguous working set (MADV_NORMAL to retain pages).
};

enum class Policy : std::uint8_t {
  Adaptive,    // Selects handle per (Access, size <= page_size).
  RandomOnly,  // Pins all reads to MADV_RANDOM mmap.
  SeqOnly,     // Pins all reads to MADV_SEQUENTIAL mmap (falls back to random_mmap).
  NormalOnly,  // Pins all reads to normal-advice mmap (falls back to random_mmap).
  PreadOnly,   // Pins all reads to bounded pread().
};

class MappedReader {
 public:
  MappedReader() noexcept = default;

  // Adopts an existing primary MADV_RANDOM mapping of `capacity` bytes (owned and unmapped on
  // destruction). Call open_secondary_handles(fd) to attach lazy scan mmaps and pread_fd.
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  explicit MappedReader(std::byte *primary_random_mmap, uint64_t capacity) noexcept
      : random_mmap_(primary_random_mmap), capacity_(capacity) {}

  ~MappedReader() noexcept { reset(); }

  MappedReader(const MappedReader &) = delete;
  auto operator=(const MappedReader &) -> MappedReader & = delete;

  MappedReader(MappedReader &&other) noexcept { swap(*this, other); }

  auto operator=(MappedReader &&other) noexcept -> MappedReader & {
    if (this != &other) {
      reset();
      swap(*this, other);
    }
    return *this;
  }

  friend void swap(MappedReader &lhs, MappedReader &rhs) noexcept {
    using std::swap;
    swap(lhs.random_mmap_, rhs.random_mmap_);
    swap(lhs.sec_, rhs.sec_);
    swap(lhs.capacity_, rhs.capacity_);
  }

  // Wraps an already-open read-only regular file descriptor into a MappedReader.
  // Does not take ownership of `file_descriptor` (caller keeps responsibility for closing it).
  // Throws std::system_error if fstat or mapping the primary random view fails; secondary views are best-effort.
  [[nodiscard]] static auto open_readonly_fd(int file_descriptor) -> MappedReader {
    return open_readonly_fd_impl(file_descriptor, nullptr);
  }

  // Opens an immutable/read-only file and sets up primary MADV_RANDOM mmap + independent
  // POSIX_FADV_RANDOM pread_fd (secondary scan VMAs are materialized lazily on first use).
  [[nodiscard]] static auto open_readonly(const std::filesystem::path &path) -> MappedReader {
    const int file_descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (file_descriptor < 0) {
      throw std::system_error(errno, std::generic_category(), "quadfecta open");
    }
    try {
      MappedReader reader = open_readonly_fd_impl(file_descriptor, &path);
      ::close(file_descriptor);
      return reader;
    } catch (...) {
      ::close(file_descriptor);
      throw;
    }
  }

  // Best-effort setup of secondary read handles: attaches an independent POSIX_FADV_RANDOM
  // pread_fd and retains a read-only fd to lazily map normal_mmap/sequential_mmap on demand.
  // Idempotent when called repeatedly; safe across threads.
  void open_secondary_handles(int file_descriptor) noexcept {
    open_secondary_handles_impl(file_descriptor, nullptr);
  }

  [[nodiscard]] static auto page_size() noexcept -> uint64_t {
    static const auto kCached = static_cast<uint64_t>(::sysconf(_SC_PAGESIZE));
    return kCached;
  }

  [[nodiscard]] auto mapping_for(Access access,
                                 bool is_small,
                                 Policy policy = Policy::Adaptive) const noexcept -> std::byte * {
    switch (policy) {
      case Policy::RandomOnly:
        return random_mmap_;
      case Policy::SeqOnly: {
        std::byte *seq = ensure_sequential_mmap();
        return seq != nullptr ? seq : random_mmap_;
      }
      case Policy::NormalOnly: {
        std::byte *norm = ensure_normal_mmap();
        return norm != nullptr ? norm : random_mmap_;
      }
      case Policy::PreadOnly:
        return nullptr;
      case Policy::Adaptive:
        break;
    }
    switch (access) {
      case Access::Random:
        return is_small ? random_mmap_ : nullptr;
      case Access::Sequential: {
        std::byte *seq = ensure_sequential_mmap();
        return seq != nullptr ? seq : random_mmap_;
      }
      case Access::SequentialReuse: {
        std::byte *norm = ensure_normal_mmap();
        return norm != nullptr ? norm : random_mmap_;
      }
    }
    return nullptr;
  }

  // Reads exactly `size` bytes at `offset`. Uses zero-copy mmap when selected, or copies into
  // `scratch` via pread(). Returns std::nullopt on out-of-bounds or I/O error.
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  [[nodiscard]] auto read(uint64_t offset,
                          uint64_t size,
                          Access access,
                          std::vector<std::byte> *scratch,
                          Policy policy = Policy::Adaptive) const -> std::optional<std::span<const std::byte>> {
    if (offset > capacity_ || size > capacity_ - offset) {
      return std::nullopt;
    }
    if (size == 0) {
      return std::span<const std::byte>{};
    }
    const auto window = view_at_most(offset, size, offset + size, access, scratch, policy);
    if (!window.has_value() || window->size() < size) {
      return std::nullopt;
    }
    return window->first(static_cast<size_t>(size));
  }

  // Reads exactly `size` bytes at `offset` directly into caller-owned `dst` without scratch allocation.
  // Uses direct ::pread() for Q2 / Policy::PreadOnly, or memcpy() from the selected zero-copy VMA.
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  [[nodiscard]] auto read_into(uint64_t offset,
                               uint64_t size,
                               void *dst,
                               Access access,
                               Policy policy = Policy::Adaptive) const noexcept -> bool {
    if (offset > capacity_ || size > capacity_ - offset) {
      return false;
    }
    if (size == 0) {
      return true;
    }
    if (dst == nullptr) {
      return false;
    }
    const bool is_small = size <= page_size();
    if (policy != Policy::PreadOnly) {
      if (std::byte *mapping = mapping_for(access, is_small, policy); mapping != nullptr) {
        std::memcpy(dst, mapping + offset, static_cast<size_t>(size));
        return true;
      }
    }
    const int read_fd = pread_fd();
    if (read_fd < 0) {
      return false;
    }
    const auto got = pread_some(read_fd, dst, static_cast<size_t>(size), offset);
    return got.has_value() && *got == static_cast<size_t>(size);
  }

  // Indexed fixed-length record read at `base_offset + index * stride`.
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  [[nodiscard]] auto read_at_index(uint64_t index,
                                   uint64_t stride,
                                   Access access,
                                   std::vector<std::byte> *scratch,
                                   Policy policy = Policy::Adaptive,
                                   uint64_t base_offset = 0) const -> std::optional<std::span<const std::byte>> {
    if (stride == 0 || base_offset > capacity_ || index > (capacity_ - base_offset) / stride) {
      return std::nullopt;
    }
    return read(base_offset + (index * stride), stride, access, scratch, policy);
  }

  // Variable-length read helper: returns an mmap view up to `min(valid_end, capacity())` bytes, or
  // preads up to `min(max_bytes, valid_end - offset)` bytes into `scratch` (or a thread_local
  // fallback vector when `scratch == nullptr`).
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  [[nodiscard]] auto view_at_most(uint64_t offset,
                                  uint64_t max_bytes,
                                  uint64_t valid_end,
                                  Access access,
                                  std::vector<std::byte> *scratch,
                                  Policy policy = Policy::Adaptive) const -> std::optional<std::span<const std::byte>> {
    const uint64_t limit = std::min(valid_end, capacity_);
    if (offset >= limit) {
      return std::nullopt;
    }
    const bool is_small = max_bytes <= page_size();

    if (policy != Policy::PreadOnly) {
      if (std::byte *mapping = mapping_for(access, is_small, policy); mapping != nullptr) {
        return std::span<const std::byte>(mapping + offset, static_cast<size_t>(limit - offset));
      }
    }

    thread_local std::vector<std::byte> tl_pread_scratch;
    std::vector<std::byte> *target = scratch != nullptr ? scratch : &tl_pread_scratch;
    return pread_window(offset, std::min(max_bytes, limit - offset), *target);
  }

  // Issues MADV_WILLNEED covering `[offset, offset + len)` clamped to capacity(), rounding
  // start down to a page boundary as required by Linux madvise(2). Skips large random Q2 ranges
  // and PreadOnly policy so pread-targeted windows do not populate mmap page tables.
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  void prefetch(uint64_t offset,
                uint64_t len,
                Access access = Access::Sequential,
                Policy policy = Policy::Adaptive) const noexcept {
    if (random_mmap_ == nullptr || offset >= capacity_ || len == 0) {
      return;
    }
    if (policy == Policy::PreadOnly || (policy == Policy::Adaptive && access == Access::Random && len > page_size())) {
      return;
    }
    const uint64_t page_mask = page_size() - 1;
    const uint64_t aligned_start = offset & ~page_mask;
    const uint64_t clamped_len = std::min(len, capacity_ - offset);
    const uint64_t aligned_len = (offset - aligned_start) + clamped_len;
    (void)::madvise(random_mmap_ + aligned_start, static_cast<size_t>(aligned_len), MADV_WILLNEED);
  }

  [[nodiscard]] auto random_mmap() const noexcept -> std::byte * { return random_mmap_; }
  [[nodiscard]] auto sequential_mmap() const noexcept -> std::byte * { return ensure_sequential_mmap(); }
  [[nodiscard]] auto normal_mmap() const noexcept -> std::byte * { return ensure_normal_mmap(); }
  [[nodiscard]] auto peek_sequential_mmap() const noexcept -> std::byte * {
    return sec_ ? sec_->sequential_mmap.load(std::memory_order_acquire) : nullptr;
  }
  [[nodiscard]] auto peek_normal_mmap() const noexcept -> std::byte * {
    return sec_ ? sec_->normal_mmap.load(std::memory_order_acquire) : nullptr;
  }
  [[nodiscard]] auto pread_fd() const noexcept -> int { return sec_ ? sec_->pread_fd : -1; }
  [[nodiscard]] auto has_independent_pread_fd() const noexcept -> bool {
    return sec_ != nullptr && sec_->pread_fd_independent;
  }
  [[nodiscard]] auto capacity() const noexcept -> uint64_t { return capacity_; }

 private:
  struct SecondaryState {
    std::atomic<std::byte *> sequential_mmap{nullptr};
    std::atomic<std::byte *> normal_mmap{nullptr};
    int mmap_fd = -1;
    int pread_fd = -1;
    bool pread_fd_independent = false;
    std::mutex mu;
  };

  [[nodiscard]] static auto open_readonly_fd_impl(int file_descriptor,
                                                  const std::filesystem::path *opt_path) -> MappedReader {
    if (file_descriptor < 0) {
      throw std::system_error(EBADF, std::generic_category(), "quadfecta invalid file descriptor");
    }
    struct stat file_stat {};
    if (::fstat(file_descriptor, &file_stat) != 0) {
      throw std::system_error(errno, std::generic_category(), "quadfecta fstat");
    }
    if (!S_ISREG(file_stat.st_mode) || file_stat.st_size < 0) {
      throw std::system_error(EINVAL, std::generic_category(), "quadfecta requires regular non-negative file");
    }
    const auto file_bytes = static_cast<uint64_t>(file_stat.st_size);
    std::byte *primary = nullptr;
    if (file_bytes > 0) {
      void *mapped = ::mmap(nullptr, static_cast<size_t>(file_bytes), PROT_READ, MAP_SHARED, file_descriptor, 0);
      if (mapped == MAP_FAILED) {
        throw std::system_error(errno, std::generic_category(), "quadfecta mmap");
      }
      if (::madvise(mapped, static_cast<size_t>(file_bytes), MADV_RANDOM) != 0) {
        const int err = errno;
        ::munmap(mapped, static_cast<size_t>(file_bytes));
        throw std::system_error(err, std::generic_category(), "quadfecta madvise MADV_RANDOM");
      }
      primary = static_cast<std::byte *>(mapped);
    }
    MappedReader reader(primary, file_bytes);
    reader.open_secondary_handles_impl(file_descriptor, opt_path);
    return reader;
  }

  void open_secondary_handles_impl(int file_descriptor, const std::filesystem::path *opt_path) noexcept {
    if (file_descriptor < 0) {
      return;
    }
    if (!sec_) {
      sec_ = std::make_unique<SecondaryState>();
    }
    if (sec_->pread_fd < 0) {
      int indep = -1;
      if (opt_path != nullptr) {
        indep = ::open(opt_path->c_str(), O_RDONLY | O_CLOEXEC);
      } else {
        constexpr size_t kProcFdPathBufferBytes = 64;
        std::array<char, kProcFdPathBufferBytes> proc_path{};
        std::snprintf(proc_path.data(), proc_path.size(), "/proc/self/fd/%d", file_descriptor);
        indep = ::open(proc_path.data(), O_RDONLY | O_CLOEXEC);
      }
      if (indep >= 0) {
        (void)::posix_fadvise(indep, 0, 0, POSIX_FADV_RANDOM);
        sec_->pread_fd = indep;
        sec_->pread_fd_independent = true;
      } else {
        sec_->pread_fd = ::fcntl(file_descriptor, F_DUPFD_CLOEXEC, 0);
        sec_->pread_fd_independent = false;
      }
    }
    if (capacity_ > 0 && sec_->mmap_fd < 0 &&
        (sec_->sequential_mmap.load(std::memory_order_acquire) == nullptr ||
         sec_->normal_mmap.load(std::memory_order_acquire) == nullptr)) {
      sec_->mmap_fd = ::fcntl(file_descriptor, F_DUPFD_CLOEXEC, 0);
    }
  }

  [[nodiscard]] auto ensure_lazy_mmap(std::atomic<std::byte *> SecondaryState::*field, int advice) const noexcept
      -> std::byte * {
    if (!sec_ || capacity_ == 0) {
      return nullptr;
    }
    std::byte *existing = (sec_.get()->*field).load(std::memory_order_acquire);
    if (existing != nullptr) {
      return existing;
    }
    std::lock_guard<std::mutex> lock(sec_->mu);
    existing = (sec_.get()->*field).load(std::memory_order_relaxed);
    if (existing != nullptr) {
      return existing;
    }
    if (sec_->mmap_fd < 0) {
      return nullptr;
    }
    std::byte *mapped = best_effort_mmap_ro(static_cast<size_t>(capacity_), sec_->mmap_fd);
    if (mapped != nullptr && advice != MADV_NORMAL) {
      if (::madvise(mapped, static_cast<size_t>(capacity_), advice) != 0) {
        ::munmap(mapped, static_cast<size_t>(capacity_));
        mapped = nullptr;
      }
    }
    if (mapped != nullptr) {
      (sec_.get()->*field).store(mapped, std::memory_order_release);
      if (sec_->sequential_mmap.load(std::memory_order_relaxed) != nullptr &&
          sec_->normal_mmap.load(std::memory_order_relaxed) != nullptr) {
        ::close(sec_->mmap_fd);
        sec_->mmap_fd = -1;
      }
    }
    return mapped;
  }

  [[nodiscard]] auto ensure_sequential_mmap() const noexcept -> std::byte * {
    return ensure_lazy_mmap(&SecondaryState::sequential_mmap, MADV_SEQUENTIAL);
  }

  [[nodiscard]] auto ensure_normal_mmap() const noexcept -> std::byte * {
    return ensure_lazy_mmap(&SecondaryState::normal_mmap, MADV_NORMAL);
  }

  static auto best_effort_mmap_ro(size_t size, int file_descriptor) noexcept -> std::byte * {
    void *mapped = ::mmap(nullptr, size, PROT_READ, MAP_SHARED, file_descriptor, 0);
    return mapped == MAP_FAILED ? nullptr : static_cast<std::byte *>(mapped);
  }

  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  static auto pread_some(int file_descriptor, void *dst, size_t len, uint64_t offset) noexcept
      -> std::optional<size_t> {
    auto *out = static_cast<std::byte *>(dst);
    size_t total = 0;
    while (total < len) {
      const ssize_t bytes_read =
          ::pread(file_descriptor, out + total, len - total, static_cast<off_t>(offset + total));
      if (bytes_read > 0) {
        total += static_cast<size_t>(bytes_read);
        continue;
      }
      if (bytes_read == 0) {
        break;
      }
      if (errno == EINTR) {
        continue;
      }
      return std::nullopt;
    }
    return total;
  }

  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  auto pread_window(uint64_t offset,
                    uint64_t len,
                    std::vector<std::byte> &scratch) const -> std::optional<std::span<const std::byte>> {
    const int read_fd = pread_fd();
    if (read_fd < 0) {
      return std::nullopt;
    }
    scratch.resize(static_cast<size_t>(len));
    const auto total = pread_some(read_fd, scratch.data(), static_cast<size_t>(len), offset);
    if (!total.has_value() || (*total == 0 && len > 0)) {
      return std::nullopt;
    }
    return std::span<const std::byte>(scratch.data(), *total);
  }

  void reset() noexcept {
    if (capacity_ > 0) {
      if (random_mmap_ != nullptr) {
        ::munmap(random_mmap_, static_cast<size_t>(capacity_));
      }
      if (sec_) {
        for (std::byte *mapping : {sec_->normal_mmap.load(std::memory_order_relaxed),
                                   sec_->sequential_mmap.load(std::memory_order_relaxed)}) {
          if (mapping != nullptr) {
            ::munmap(mapping, static_cast<size_t>(capacity_));
          }
        }
      }
    }
    if (sec_) {
      if (sec_->pread_fd >= 0) {
        ::close(sec_->pread_fd);
      }
      if (sec_->mmap_fd >= 0) {
        ::close(sec_->mmap_fd);
      }
      sec_.reset();
    }
    random_mmap_ = nullptr;
    capacity_ = 0;
  }

  std::byte *random_mmap_ = nullptr;
  mutable std::unique_ptr<SecondaryState> sec_;
  uint64_t capacity_ = 0;
};

// Positional sequential stream cursor over a MappedReader (Access::Sequential).
// Fits cursor-oriented scans (e.g. Lucene IndexInput, RocksDB BlobLogSequentialReader, Git packs).
class Cursor {
 public:
  Cursor() noexcept = default;

  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  explicit Cursor(const MappedReader &reader,
                  uint64_t begin_offset = 0,
                  uint64_t end_offset = std::numeric_limits<uint64_t>::max()) noexcept
      : reader_(&reader), end_(std::min(end_offset, reader.capacity())), pos_(std::min(begin_offset, end_)) {}

  [[nodiscard]] auto pos() const noexcept -> uint64_t { return pos_; }
  [[nodiscard]] auto end() const noexcept -> uint64_t { return end_; }
  [[nodiscard]] auto remaining() const noexcept -> uint64_t { return end_ - pos_; }
  [[nodiscard]] auto at_end() const noexcept -> bool { return pos_ >= end_; }

  void seek(uint64_t offset) noexcept { pos_ = std::min(offset, end_); }
  void advance(uint64_t bytes) noexcept { pos_ += std::min(bytes, end_ - pos_); }

  // Reads `size` sequential bytes from pos() and advances pos() on success.
  [[nodiscard]] auto read_next(uint64_t size,
                               std::vector<std::byte> *scratch,
                               Policy policy = Policy::Adaptive) -> std::optional<std::span<const std::byte>> {
    if (reader_ == nullptr || size > remaining()) {
      return std::nullopt;
    }
    const auto span = reader_->read(pos_, size, Access::Sequential, scratch, policy);
    if (span.has_value()) {
      pos_ += size;
    }
    return span;
  }

  // Variable-length sequential read helper at pos() bounded by end_. Does not advance pos();
  // call advance(consumed_bytes) after parsing the record length.
  [[nodiscard]] auto peek_at_most(uint64_t max_bytes,
                                  std::vector<std::byte> *scratch,
                                  Policy policy = Policy::Adaptive) const -> std::optional<std::span<const std::byte>> {
    if (reader_ == nullptr) {
      return std::nullopt;
    }
    return reader_->view_at_most(pos_, max_bytes, end_, Access::Sequential, scratch, policy);
  }

 private:
  const MappedReader *reader_ = nullptr;
  uint64_t end_ = 0;
  uint64_t pos_ = 0;
};

// C++20 input range yielding sequential fixed-stride records or chunks via Quadfecta.
class RecordRange {
 public:
  enum class TailMode : std::uint8_t { ExactOnly, AllowPartial };

  struct Spec {
    uint64_t stride = 0;
    std::vector<std::byte> *scratch = nullptr;
    Policy policy = Policy::Adaptive;
    TailMode tail_mode = TailMode::ExactOnly;
  };

  class Iterator {
   public:
    using iterator_concept = std::input_iterator_tag;
    using iterator_category = std::input_iterator_tag;
    using value_type = std::span<const std::byte>;
    using difference_type = std::ptrdiff_t;

    Iterator() noexcept = default;
    [[nodiscard]] auto operator*() const noexcept -> value_type { return current_.value_or(value_type{}); }
    auto operator++() -> Iterator & {
      fetch_next();
      return *this;
    }
    void operator++(int) { ++(*this); }
    [[nodiscard]] friend auto operator==(const Iterator &iter, std::default_sentinel_t /*unused*/) noexcept -> bool {
      return !iter.current_.has_value();
    }

   private:
    friend class RecordRange;
    Iterator(Cursor cursor, Spec spec) : cursor_(cursor), spec_(spec) { fetch_next(); }

    void fetch_next() {
      if (spec_.stride == 0 || cursor_.at_end()) {
        current_ = std::nullopt;
        return;
      }
      const uint64_t want =
          spec_.tail_mode == TailMode::AllowPartial ? std::min(spec_.stride, cursor_.remaining()) : spec_.stride;
      current_ = cursor_.read_next(want, spec_.scratch, spec_.policy);
    }

    Cursor cursor_;
    Spec spec_{};
    std::optional<value_type> current_;
  };

  RecordRange(Cursor cursor, Spec spec) noexcept : cursor_(cursor), spec_(spec) {}

  [[nodiscard]] auto begin() const -> Iterator { return {cursor_, spec_}; }
  [[nodiscard]] static auto end() noexcept -> std::default_sentinel_t { return std::default_sentinel; }

 private:
  Cursor cursor_;
  Spec spec_{};
};

// Iterates exact `stride`-byte sequential records over `[begin_offset, min(end_offset, reader.capacity()))`.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
[[nodiscard]] inline auto records(const MappedReader &reader,
                                  uint64_t stride,
                                  std::vector<std::byte> *scratch,
                                  Policy policy = Policy::Adaptive,
                                  uint64_t begin_offset = 0,
                                  uint64_t end_offset = std::numeric_limits<uint64_t>::max()) noexcept -> RecordRange {
  return {Cursor(reader, begin_offset, end_offset), {stride, scratch, policy, RecordRange::TailMode::ExactOnly}};
}

// Iterates sequential chunks up to `chunk_bytes` (including any shorter trailing chunk).
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
[[nodiscard]] inline auto chunks(const MappedReader &reader,
                                 uint64_t chunk_bytes,
                                 std::vector<std::byte> *scratch,
                                 Policy policy = Policy::Adaptive,
                                 uint64_t begin_offset = 0,
                                 uint64_t end_offset = std::numeric_limits<uint64_t>::max()) noexcept -> RecordRange {
  return {Cursor(reader, begin_offset, end_offset),
          {chunk_bytes, scratch, policy, RecordRange::TailMode::AllowPartial}};
}

}  // namespace quadfecta
