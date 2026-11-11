#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cassert>
#include <cstring>
#include <limits>
#include <optional>
#include <system_error>
#include <vmemkv/checkpoint/checkpoint.hpp>
#include <vmemkv/core/io.hpp>
#include <vmemkv/core/mmap.hpp>
#include <vmemkv/t2_flat_file/t2_flat_file.hpp>

namespace vmemkv {

namespace {

// One cold get faults whole pages, so a <=1-page record must not straddle a boundary.
// Sub-page skips stay zero-filled and are never reused.
constexpr uint64_t kPageBytes = 4096;

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
auto page_skip_bytes(uint64_t frontier, uint64_t required) noexcept -> uint64_t {
  const uint64_t misalignment = frontier % kPageBytes;
  if (misalignment == 0) {
    return 0;
  }
  if (required <= kPageBytes) {
    return (misalignment + required > kPageBytes) ? (kPageBytes - misalignment) : 0;
  }
  return 0;
}

void write_record(std::byte *record_base, std::span<const std::byte> key, std::span<const std::byte> value) noexcept {
  const uint64_t raw_len = sizeof(ValueRecordHeader) + key.size() + value.size();
  const uint64_t aligned_len = vmemkv::record_aligned_len(sizeof(ValueRecordHeader), key.size(), value.size());

  auto *header = reinterpret_cast<ValueRecordHeader *>(record_base);
  header->key_len = static_cast<uint32_t>(key.size());
  header->value_len = static_cast<uint32_t>(value.size());
  header->alloc_len = static_cast<uint32_t>(value.size());
  header->version = 0;

  auto *cursor = reinterpret_cast<std::byte *>(header + 1);
  std::memcpy(cursor, key.data(), key.size());
  cursor += key.size();

  if (!value.empty()) {
    std::memcpy(cursor, value.data(), value.size());
    cursor += value.size();
  }

  const uint64_t padding = aligned_len - raw_len;
  if (padding > 0) {
    std::memset(cursor, 0, padding);
  }
}

}  // namespace

T2FlatFile::T2FlatFile(const std::filesystem::path &path,
                       uint64_t bytes_capacity,
                       std::optional<uint64_t> initial_bytes_used)
    : path_(path) {
  const std::filesystem::path data_path = vmemkv::derive_t2_chk_path(path);
  if (!initial_bytes_used.has_value()) {
    remove_quiet(data_path);
    create_empty_file(data_path, bytes_capacity);
  }
  map_file(data_path, bytes_capacity, initial_bytes_used.value_or(0));
}

T2FlatFile::~T2FlatFile() noexcept {
  const T2Memory *mem = t2_mem_.load(std::memory_order_relaxed);
  if (mem != nullptr) {
    gate_.wait_until_retired(mem);
    delete mem;
  }
}

auto T2FlatFile::at(uint64_t payload, const T2Memory *mem) noexcept -> T2RecordView {
  const std::byte *record_base = resolve_record(payload, mem);
  return make_record_view(reinterpret_cast<const ValueRecordHeader *>(record_base));
}

auto T2FlatFile::append_default(const T2Memory *mem,
                                std::span<const std::byte> key,
                                std::span<const std::byte> value) -> uint64_t {
  const uint64_t required = vmemkv::record_aligned_len(sizeof(ValueRecordHeader), key.size(), value.size());

  uint64_t frontier = mem->bytes_used.load(std::memory_order_relaxed);
  while (true) {
    const uint64_t skip = page_skip_bytes(frontier, required);
    if (frontier + skip + required > mem->capacity) {
      throw std::runtime_error("T2 storage capacity exceeded");
    }
    if (mem->bytes_used.compare_exchange_weak(frontier, frontier + skip + required, std::memory_order_relaxed)) {
      const uint64_t offset = frontier + skip;
      write_record(mem->reads.random_mmap + offset, key, value);
      return offset;
    }
  }
}

auto T2FlatFile::update_value_at(uint64_t payload,
                                 std::span<const std::byte> value,
                                 const T2Memory *mem) noexcept -> bool {
  auto *header = reinterpret_cast<ValueRecordHeader *>(const_cast<std::byte *>(resolve_record(payload, mem)));

  if (value.size() > header->alloc_len) {
    return false;
  }

  std::byte *value_begin = reinterpret_cast<std::byte *>(header + 1) + header->key_len;

  // Odd version blocks readers; the racing plain memcpy below is discarded by the version recheck.
  auto atomic_version = std::atomic_ref<uint64_t>(header->version);
  uint64_t ver = atomic_version.load(std::memory_order_relaxed);
  atomic_version.store(ver + 1, std::memory_order_release);
  std::atomic_thread_fence(std::memory_order_release);

  if (!value.empty()) {
    std::memcpy(value_begin, value.data(), value.size());
  }

  header->value_len = static_cast<uint32_t>(value.size());

  std::atomic_thread_fence(std::memory_order_release);
  atomic_version.store(ver + 2, std::memory_order_release);
  return true;
}

void T2FlatFile::map_file(const std::filesystem::path &path, uint64_t bytes_capacity, uint64_t initial_bytes_used) {
  FdGuard file_descriptor{open_file(path.c_str(), O_RDWR, "open")};

  if (fstat_size(file_descriptor.fd, "fstat") < bytes_capacity) {
    throw std::invalid_argument("T2 file is smaller than capacity");
  }

  // MAP_SHARED keeps every mapping coherent so msync() durabilizes what readers already see.
  // MAP_NORESERVE allows address spaces larger than RAM + swap. See docs/specification/low_level_design.md 4.3/5.1.
  void *mapped = mmap_shared_or_throw(static_cast<size_t>(bytes_capacity),
                                      file_descriptor.fd,
                                      PROT_READ | PROT_WRITE,
                                      "mmap",
                                      MAP_SHARED | MAP_NORESERVE,
                                      0);
  try {
    madvise_or_throw(mapped, static_cast<size_t>(bytes_capacity), MADV_RANDOM, "madvise MADV_RANDOM");
  } catch (...) {
    ::munmap(mapped, static_cast<size_t>(bytes_capacity));
    throw;
  }

  auto *mem = new T2Memory(static_cast<std::byte *>(mapped), bytes_capacity, initial_bytes_used);
  mem->setup_read_handles(file_descriptor.fd);

  file_descriptor.reset();
  t2_mem_.store(mem, std::memory_order_release);
}

void T2FlatFile::create_empty_file(const std::filesystem::path &path, uint64_t bytes_capacity) {
  if (bytes_capacity == 0) {
    throw std::invalid_argument("T2Store capacity must be greater than zero");
  }
  if (bytes_capacity > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    throw std::invalid_argument("T2Store capacity is too large for mmap");
  }
  if (bytes_capacity > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
    throw std::invalid_argument("T2Store capacity is too large for ftruncate");
  }

  const FdGuard file_descriptor{open_create_0600(path.c_str(), O_RDWR | O_CREAT | O_EXCL, "open")};

  if (::ftruncate(file_descriptor.fd, static_cast<off_t>(bytes_capacity)) != 0) {
    const int err = errno;
    remove_quiet(path);
    throw std::system_error(err, std::generic_category(), "ftruncate");
  }
}
auto T2FlatFile::punch_if_occupied(uint64_t offset, uint64_t len) const noexcept -> PunchOutcome {
  if (len == 0) {
    return PunchOutcome::AlreadyHollow;
  }
  const T2Memory *mem = get_memory();
  if (offset + len > mem->capacity) {
    return PunchOutcome::Failed;
  }
  const int file_descriptor = ::open(vmemkv::derive_t2_chk_path(path_).c_str(), O_RDWR);
  if (file_descriptor < 0) {
    return PunchOutcome::Failed;
  }
  const off_t data_at = ::lseek(file_descriptor, static_cast<off_t>(offset), SEEK_DATA);
  const int seek_errno = errno;
  if (data_at < 0) {
    ::close(file_descriptor);
    // ENXIO: no data at or after offset within this file. Other errors mean
    // SEEK_DATA is unsupported; fall through and let fallocate report support.
    if (seek_errno == ENXIO) {
      return PunchOutcome::AlreadyHollow;
    }
  } else if (static_cast<uint64_t>(data_at) >= offset + len) {
    ::close(file_descriptor);
    return PunchOutcome::AlreadyHollow;
  }
  const int punch_rc = ::fallocate(
      file_descriptor, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE, static_cast<off_t>(offset), static_cast<off_t>(len));
  ::close(file_descriptor);
  if (punch_rc != 0) {
    return PunchOutcome::Failed;
  }
  // Punch zeroes file blocks but resident pages would keep serving stale bytes.
  ::madvise(mem->reads.random_mmap + offset, static_cast<size_t>(len), MADV_DONTNEED);
  return PunchOutcome::Punched;
}

T2Memory::~T2Memory() noexcept {
  if (reads.normal_mmap != nullptr) {
    ::munmap(reads.normal_mmap, static_cast<size_t>(capacity));
  }
  if (reads.sequential_mmap != nullptr) {
    ::munmap(reads.sequential_mmap, static_cast<size_t>(capacity));
  }
  if (reads.pread_fd >= 0) {
    ::close(reads.pread_fd);
  }
  if ((reads.random_mmap != nullptr) && capacity > 0) {
    ::munmap(reads.random_mmap, static_cast<size_t>(capacity));
  }
}

void T2Memory::setup_read_handles(int file_descriptor) noexcept {
  void *scan = best_effort_mmap_shared(static_cast<size_t>(capacity), file_descriptor, PROT_READ, 0);
  if (scan != nullptr) {
    reads.normal_mmap = static_cast<std::byte *>(scan);
  }

  void *scan_seq = best_effort_mmap_shared(static_cast<size_t>(capacity), file_descriptor, PROT_READ, 0);
  if (scan_seq != nullptr) {
    if (::madvise(scan_seq, static_cast<size_t>(capacity), MADV_SEQUENTIAL) == 0) {
      reads.sequential_mmap = static_cast<std::byte *>(scan_seq);
    } else {
      ::munmap(scan_seq, static_cast<size_t>(capacity));
    }
  }

  // Dup before the caller closes file_descriptor so the lifetime is self-contained.
  reads.pread_fd = ::fcntl(file_descriptor, F_DUPFD_CLOEXEC, 0);
}

}  // namespace vmemkv
