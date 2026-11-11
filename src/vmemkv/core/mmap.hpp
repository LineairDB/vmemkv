// mmap.hpp - mmap/madvise helpers plus an RAII mapping guard.
#pragma once

#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <system_error>
#include <utility>

namespace vmemkv {

inline auto mmap_or_throw(size_t size, int prot, int flags, int file_descriptor, off_t offset, const char *what)
    -> void * {
  void *mapped = ::mmap(nullptr, size, prot, flags, file_descriptor, offset);
  if (mapped == MAP_FAILED) {
    throw std::system_error(errno, std::generic_category(), what);
  }
  return mapped;
}

inline auto mmap_anon_or_throw(size_t size, const char *what = "mmap") -> void * {
  return mmap_or_throw(size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0, what);
}

inline auto mmap_shared_or_throw(size_t size,
                                 int file_descriptor,
                                 int prot,
                                 const char *what,
                                 int flags = MAP_SHARED | MAP_NORESERVE,
                                 off_t offset = 0) -> void * {
  return mmap_or_throw(size, prot, flags, file_descriptor, offset, what);
}

inline void madvise_or_throw(void *address, size_t size, int advice, const char *what) {
  if (::madvise(address, size, advice) != 0) {
    throw std::system_error(errno, std::generic_category(), what);
  }
}

// Best-effort shared mapping for read-only scan optimizations. Null on failure, never throws.
// Deliberately without MAP_NORESERVE (unlike mmap_shared_or_throw's default): file-backed
// read-only pages need no swap reservation, so the flag would change nothing.
inline auto best_effort_mmap_shared(size_t size, int file_descriptor, int prot, off_t offset = 0) noexcept -> void * {
  void *mapped = ::mmap(nullptr, size, prot, MAP_SHARED, file_descriptor, offset);
  if (mapped == MAP_FAILED) {
    return nullptr;
  }
  return mapped;
}

// RAII owner of one mmap region. Unmaps on destruction, movable, non-copyable.
class MmapGuard {
 public:
  MmapGuard() noexcept = default;
  MmapGuard(void *address, size_t size) noexcept : address_(address), size_(size) {}

  ~MmapGuard() noexcept { reset(); }

  MmapGuard(const MmapGuard &) = delete;
  auto operator=(const MmapGuard &) -> MmapGuard & = delete;

  MmapGuard(MmapGuard &&other) noexcept : address_(other.address_), size_(other.size_) {
    other.address_ = nullptr;
    other.size_ = 0;
  }
  auto operator=(MmapGuard &&other) noexcept -> MmapGuard & {
    if (this != &other) {
      reset();
      address_ = other.address_;
      size_ = other.size_;
      other.address_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  void reset() noexcept {
    if (address_ != nullptr) {
      ::munmap(address_, size_);
      address_ = nullptr;
      size_ = 0;
    }
  }

  [[nodiscard]] auto get() const noexcept -> void * { return address_; }

 private:
  void *address_ = nullptr;
  size_t size_ = 0;
};

}  // namespace vmemkv
