// quadfecta.hpp - Standalone Linux four-handle adaptive file reader ("Quadfecta").
//
// Maintains up to four OS views over the same backing file, selecting one per read by
// (Access, size <= page_size, Policy):
// - Random     + <= page: MADV_RANDOM mmap (zero-copy, no readahead pollution)
// - Random     +  > page: bounded pread() (avoids minor-fault / VM overhead on large gets)
// - Sequential + <= page: MADV_SEQUENTIAL mmap (wide kernel readahead)
// - Sequential +  > page: normal mmap (default kernel readahead)
//
// Dependencies: C++20/23 STL + Linux POSIX (<fcntl.h>, <sys/mman.h>, <sys/stat.h>, <unistd.h>).
// Thread safety: all const read methods are lock-free and thread-safe when callers supply their
// own `scratch` buffer (for Policy::PreadOnly with scratch == nullptr, a thread_local buffer is
// used). Any concurrent writes to the backing file must be externally bounded below `valid_end`.
#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <span>
#include <system_error>
#include <utility>
#include <vector>

namespace quadfecta {

static_assert(sizeof(off_t) >= sizeof(int64_t), "quadfecta requires 64-bit off_t");

enum class Access : std::uint8_t {
  Random,      // Single-record point access (avoid readahead).
  Sequential,  // Contiguous range scan (benefit from readahead).
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
  // destruction). Call open_secondary_handles(fd) to attach scan mmaps and pread_fd.
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  explicit MappedReader(std::byte *primary_random_mmap, uint64_t capacity) noexcept
      : random_mmap_(primary_random_mmap), capacity_(capacity) {}

  ~MappedReader() noexcept { reset(); }

  MappedReader(const MappedReader &) = delete;
  auto operator=(const MappedReader &) -> MappedReader & = delete;

  MappedReader(MappedReader &&other) noexcept
      : random_mmap_(std::exchange(other.random_mmap_, nullptr)),
        sequential_mmap_(std::exchange(other.sequential_mmap_, nullptr)),
        normal_mmap_(std::exchange(other.normal_mmap_, nullptr)),
        pread_fd_(std::exchange(other.pread_fd_, -1)),
        capacity_(std::exchange(other.capacity_, 0)) {}

  auto operator=(MappedReader &&other) noexcept -> MappedReader & {
    if (this != &other) {
      reset();
      random_mmap_ = std::exchange(other.random_mmap_, nullptr);
      sequential_mmap_ = std::exchange(other.sequential_mmap_, nullptr);
      normal_mmap_ = std::exchange(other.normal_mmap_, nullptr);
      pread_fd_ = std::exchange(other.pread_fd_, -1);
      capacity_ = std::exchange(other.capacity_, 0);
    }
    return *this;
  }

  // Opens an immutable/read-only file and maps all four Quadfecta handles. Throws std::system_error
  // if opening, stat'ing, or mapping the primary random view fails; secondary views are best-effort.
  [[nodiscard]] static auto open_readonly(const std::filesystem::path &path) -> MappedReader {
    const int file_descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (file_descriptor < 0) {
      throw std::system_error(errno, std::generic_category(), "quadfecta open");
    }
    try {
      struct stat file_stat {};
      if (::fstat(file_descriptor, &file_stat) != 0) {
        throw std::system_error(errno, std::generic_category(), "quadfecta fstat");
      }
      if (!S_ISREG(file_stat.st_mode) || file_stat.st_size < 0) {
        throw std::system_error(EINVAL, std::generic_category(), "quadfecta requires regular non-negative file");
      }
      const auto file_bytes = static_cast<uint64_t>(file_stat.st_size);
      if constexpr (sizeof(uint64_t) > sizeof(size_t)) {
        if (file_bytes > std::numeric_limits<size_t>::max()) {
          throw std::system_error(EOVERFLOW, std::generic_category(), "quadfecta file exceeds address space");
        }
      }
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
      reader.open_secondary_handles(file_descriptor);
      ::close(file_descriptor);
      return reader;
    } catch (...) {
      ::close(file_descriptor);
      throw;
    }
  }

  // Best-effort setup of secondary read views (normal_mmap, sequential_mmap, dup'd pread_fd).
  // Idempotent when called repeatedly; should complete before publishing this reader across threads.
  void open_secondary_handles(int file_descriptor) noexcept {
    if (capacity_ > 0) {
      if (normal_mmap_ == nullptr) {
        normal_mmap_ = best_effort_mmap_ro(static_cast<size_t>(capacity_), file_descriptor);
      }
      if (sequential_mmap_ == nullptr) {
        std::byte *seq = best_effort_mmap_ro(static_cast<size_t>(capacity_), file_descriptor);
        if (seq != nullptr) {
          if (::madvise(seq, static_cast<size_t>(capacity_), MADV_SEQUENTIAL) == 0) {
            sequential_mmap_ = seq;
          } else {
            ::munmap(seq, static_cast<size_t>(capacity_));
          }
        }
      }
    }
    if (pread_fd_ < 0) {
      pread_fd_ = ::fcntl(file_descriptor, F_DUPFD_CLOEXEC, 0);
    }
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
      case Policy::SeqOnly:
        return sequential_mmap_ != nullptr ? sequential_mmap_ : random_mmap_;
      case Policy::NormalOnly:
        return normal_mmap_ != nullptr ? normal_mmap_ : random_mmap_;
      case Policy::PreadOnly:
        return nullptr;
      case Policy::Adaptive:
        break;
    }
    if (access == Access::Sequential) {
      std::byte *preferred = is_small ? sequential_mmap_ : normal_mmap_;
      return preferred != nullptr ? preferred : random_mmap_;
    }
    return is_small ? random_mmap_ : nullptr;
  }

  // Reads exactly `size` bytes at `offset`. Uses zero-copy mmap when selected, or copies into
  // `scratch` via pread(). Returns std::nullopt on out-of-bounds, missing handle/scratch, or I/O error.
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

  // Variable-length read helper: returns an mmap view up to `min(valid_end, capacity())` bytes, or
  // preads up to `min(max_bytes, valid_end - offset)` bytes into `scratch`. Callers parse record
  // headers inside the returned span and verify total record length <= returned span size.
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

    if (policy == Policy::PreadOnly) {
      thread_local std::vector<std::byte> tl_pread_scratch;
      std::vector<std::byte> *target = scratch != nullptr ? scratch : &tl_pread_scratch;
      return pread_window(offset, std::min(max_bytes, limit - offset), *target);
    }

    if (std::byte *mapping = mapping_for(access, is_small, policy); mapping != nullptr) {
      return std::span<const std::byte>(mapping + offset, static_cast<size_t>(limit - offset));
    }

    if (access == Access::Random && scratch != nullptr) {
      return pread_window(offset, std::min(max_bytes, limit - offset), *scratch);
    }
    return std::nullopt;
  }

  // Issues MADV_WILLNEED covering `[offset, offset + len)` clamped to capacity(), rounding
  // start down to a page boundary as required by Linux madvise(2). Best-effort no-op.
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  void prefetch(uint64_t offset, uint64_t len) const noexcept {
    if (random_mmap_ == nullptr || offset >= capacity_ || len == 0) {
      return;
    }
    const uint64_t page_mask = page_size() - 1;
    const uint64_t aligned_start = offset & ~page_mask;
    const uint64_t clamped_len = std::min(len, capacity_ - offset);
    const uint64_t aligned_len = (offset - aligned_start) + clamped_len;
    (void)::madvise(random_mmap_ + aligned_start, static_cast<size_t>(aligned_len), MADV_WILLNEED);
  }

  [[nodiscard]] auto random_mmap() const noexcept -> std::byte * { return random_mmap_; }
  [[nodiscard]] auto sequential_mmap() const noexcept -> std::byte * { return sequential_mmap_; }
  [[nodiscard]] auto normal_mmap() const noexcept -> std::byte * { return normal_mmap_; }
  [[nodiscard]] auto pread_fd() const noexcept -> int { return pread_fd_; }
  [[nodiscard]] auto capacity() const noexcept -> uint64_t { return capacity_; }

 private:
  static auto best_effort_mmap_ro(size_t size, int file_descriptor) noexcept -> std::byte * {
    void *mapped = ::mmap(nullptr, size, PROT_READ, MAP_SHARED, file_descriptor, 0);
    return mapped == MAP_FAILED ? nullptr : static_cast<std::byte *>(mapped);
  }

  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  auto pread_window(uint64_t offset,
                    uint64_t len,
                    std::vector<std::byte> &scratch) const -> std::optional<std::span<const std::byte>> {
    if (pread_fd_ < 0) {
      return std::nullopt;
    }
    scratch.resize(static_cast<size_t>(len));
    size_t total = 0;
    while (total < len) {
      const ssize_t bytes_read = ::pread(
          pread_fd_, scratch.data() + total, static_cast<size_t>(len) - total, static_cast<off_t>(offset + total));
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
    if (total == 0 && len > 0) {
      return std::nullopt;
    }
    return std::span<const std::byte>(scratch.data(), total);
  }

  void reset() noexcept {
    if (capacity_ > 0) {
      if (normal_mmap_ != nullptr) {
        ::munmap(normal_mmap_, static_cast<size_t>(capacity_));
      }
      if (sequential_mmap_ != nullptr) {
        ::munmap(sequential_mmap_, static_cast<size_t>(capacity_));
      }
      if (random_mmap_ != nullptr) {
        ::munmap(random_mmap_, static_cast<size_t>(capacity_));
      }
    }
    if (pread_fd_ >= 0) {
      ::close(pread_fd_);
    }
    random_mmap_ = nullptr;
    sequential_mmap_ = nullptr;
    normal_mmap_ = nullptr;
    pread_fd_ = -1;
    capacity_ = 0;
  }

  std::byte *random_mmap_ = nullptr;
  std::byte *sequential_mmap_ = nullptr;
  std::byte *normal_mmap_ = nullptr;
  int pread_fd_ = -1;
  uint64_t capacity_ = 0;
};

}  // namespace quadfecta
