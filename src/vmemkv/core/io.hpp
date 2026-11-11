// io.hpp - Exact-length fd read/write helpers.
#pragma once

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <system_error>

namespace vmemkv {

// RAII owner of one file descriptor. Closes on destruction, movable, non-copyable.
struct FdGuard {
  int fd = -1;
  FdGuard() noexcept = default;
  explicit FdGuard(int fd_in) noexcept : fd(fd_in) {}
  ~FdGuard() noexcept { reset(); }
  FdGuard(const FdGuard &) = delete;
  auto operator=(const FdGuard &) -> FdGuard & = delete;
  FdGuard(FdGuard &&other) noexcept : fd(other.fd) { other.fd = -1; }
  auto operator=(FdGuard &&other) noexcept -> FdGuard & {
    if (this != &other) {
      reset();
      fd = other.fd;
      other.fd = -1;
    }
    return *this;
  }
  void reset() noexcept {
    if (fd >= 0) {
      ::close(fd);
      fd = -1;
    }
  }
  // Releases ownership without closing; the caller closes the returned fd.
  auto release() noexcept -> int {
    const int out = fd;
    fd = -1;
    return out;
  }
};

// Opens or throws. Wrap the result in FdGuard unless ownership transfers immediately.
inline auto open_file(const char *path, int flags, const char *what) -> int {
  const int file_descriptor = ::open(path, flags);
  if (file_descriptor < 0) {
    throw std::system_error(errno, std::generic_category(), what);
  }
  return file_descriptor;
}

// Opens-or-creates with 0600 or throws. Single place for the permission bits.
inline auto open_create_0600(const char *path, int flags, const char *what) -> int {
  const int file_descriptor = ::open(path, flags, 0600);
  if (file_descriptor < 0) {
    throw std::system_error(errno, std::generic_category(), what);
  }
  return file_descriptor;
}

// fstat size or throws.
inline auto fstat_size(int file_descriptor, const char *what) -> uint64_t {
  struct stat file_stat {};
  if (::fstat(file_descriptor, &file_stat) != 0) {
    throw std::system_error(errno, std::generic_category(), what);
  }
  if (file_stat.st_size < 0) {
    throw std::system_error(EIO, std::generic_category(), what);
  }
  return static_cast<uint64_t>(file_stat.st_size);
}

// One partial syscall with EINTR retry; throws on a real error. Short results are the caller's to interpret.
template <typename Op>
inline auto syscall_once(Op &&operation, const char *what) -> ssize_t {
  while (true) {
    const ssize_t result_code = operation();
    if (result_code >= 0) {
      return result_code;
    }
    if (errno == EINTR) {
      continue;
    }
    throw std::system_error(errno, std::generic_category(), what);
  }
}

// Writes exactly `size` bytes or throws. Leaves `file_descriptor` open.
inline void write_all_exact(int file_descriptor, const void *data, size_t size, const char *what) {
  const auto *bytes = static_cast<const std::byte *>(data);
  size_t offset = 0;
  while (offset < size) {
    const ssize_t written = syscall_once([&] { return ::write(file_descriptor, bytes + offset, size - offset); }, what);
    if (written == 0) {
      throw std::system_error(EIO, std::generic_category(), what);
    }
    offset += static_cast<size_t>(written);
  }
}

// Reads exactly `size` bytes at `offset` or throws. Leaves `file_descriptor` open.
inline void pread_exact(int file_descriptor, void *data, size_t size, off_t offset, const char *what) {
  auto *bytes = static_cast<std::byte *>(data);
  size_t done = 0;
  while (done < size) {
    const ssize_t got = syscall_once(
        [&] { return ::pread(file_descriptor, bytes + done, size - done, offset + static_cast<off_t>(done)); }, what);
    if (got == 0) {
      throw std::system_error(EIO, std::generic_category(), what);
    }
    done += static_cast<size_t>(got);
  }
}

}  // namespace vmemkv
