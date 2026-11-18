// read_path.hpp - T2 immutable-region read path for VMemKVImpl.
//
// Once a record's offset is below `read_only_offset`, its bytes are immutable forever (an
// in-place update targeting the immutable region redirects out-of-place instead), so a reader
// doesn't need the seqlock protecting the mutable region and can read straight out of a
// mapping. Get and Scan want *different* readahead policies for that read though (madvise
// is a property of the whole mapping, not of one read), so four handles over the identical
// bytes exist (`reads.random_mmap`, `reads.sequential_mmap`, `reads.normal_mmap`, plus
// `reads.pread_fd` -- see T2ReadHandles for what each serves) -- which one a given
// call uses is decided per record by mapping_for() below. That function is
// the single place this decision is made, and callers never choose a mapping themselves.
// Span construction itself is ::make_record_view (t2_flat_file.hpp), shared with
// T2FlatFile::at().
#pragma once

#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cassert>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <vector>
#include <vmemkv/config.hpp>
#include <vmemkv/core/bytes.hpp>
#include <vmemkv/core/spin_backoff.hpp>
#include <vmemkv/impl/hooks.hpp>
#include <vmemkv/impl/t2_ownership.hpp>
#include <vmemkv/t1_index/t1_index.hpp>
#include <vmemkv/t2_flat_file/t2_flat_file.hpp>

namespace vmemkv {

// AtFunc: () -> T2RecordView, called fresh on every retry attempt. Required because
// T2FlatFile::at() reads key_len/value_len unsynchronized to size the key/value spans; a view
// built once and reused across retries can carry a stale span size that the version check
// afterward can't catch, since the size was already wrong before the loop started.
//
// CopyFunc's plain (non-atomic) reads of a record's key/value bytes race, in the C++
// abstract-machine sense, with T2FlatFile::update_value_at()'s plain memcpy of the same bytes.
// It's benign by construction: the version check below discards any read that overlapped a
// concurrent write, so a torn read here is never actually used, only retried (same principle
// as a Linux kernel seqlock). See tsan_suppressions.txt for the suppression.
template <typename AtFunc, typename CopyFunc>
inline auto read_t2_record_seqlock(AtFunc &&at_func, CopyFunc &&copy_func) {
  SpinBackoff backoff;
  while (true) {
    const T2RecordView record = at_func();
    auto atomic_version = std::atomic_ref<const uint64_t>(record.header->version);
    uint64_t version_before = atomic_version.load(std::memory_order_acquire);
    if (version_before % 2 != 0) {
      backoff.wait();
      continue;
    }

    auto result = copy_func(record);

    std::atomic_thread_fence(std::memory_order_acquire);
    uint64_t version_after = atomic_version.load(std::memory_order_acquire);
    if (version_before == version_after) {
      return result;
    }
  }
}

// Which caller is asking try_read_immutable_record() below for an immutable-region record.
enum class ImmutableReader : uint8_t {
  kGet,   // One record per call, effectively random access -- never wants readahead.
  kScan,  // A contiguous key range per call -- benefits from readahead.
};

// Single table for every immutable-region mapping choice. Picking the "wrong" mapping for a
// record's size is only ever a readahead-policy mismatch, never incorrect data: every
// mapping here covers the identical underlying bytes.
//
// Pinned policies ignore the reader and size. The default (adaptive) policy picks per
// record; null means "no direct mapping: take the residency-check path instead"
// (large, non-resident get only).
inline auto mapping_for(const T2Memory *mem,
                        ImmutableReader reader,
                        bool is_small,
                        ReadPolicy policy = ReadPolicy::Adaptive) noexcept -> std::byte * {
  switch (policy) {
    case ReadPolicy::RandomOnly:
      return mem->reads.random_mmap;
    case ReadPolicy::SeqOnly:
      return mem->reads.sequential_mmap != nullptr ? mem->reads.sequential_mmap : mem->reads.random_mmap;
    case ReadPolicy::NormalOnly:
      return mem->reads.normal_mmap != nullptr ? mem->reads.normal_mmap : mem->reads.random_mmap;
    case ReadPolicy::PreadOnly:
      // Handled by try_read_immutable_record()'s pread branch below, never by mapping selection.
      return nullptr;
    case ReadPolicy::Adaptive:
      break;
  }
  if (reader == ImmutableReader::kScan) {
    return is_small ? mem->reads.sequential_mmap : mem->reads.normal_mmap;
  }
  if (is_small) {
    return mem->reads.random_mmap;
  }
  return nullptr;
}

// Defense-in-depth backstop, not a routine path: the record must end within the immutable
// prefix. False falls through to the always-correct mmap+seqlock path.
inline auto immutable_bounds_ok(uint64_t offset, uint64_t needed, uint64_t read_only_offset) noexcept -> bool {
  return offset + needed <= read_only_offset;
}

// Bounds-checked view of the immutable-region record at `mapping + offset`: the record's bytes
// must end within the immutable prefix, otherwise std::nullopt.
inline auto view_immutable_record_if_in_bounds(const std::byte *mapping,
                                               uint64_t offset,
                                               uint64_t read_only_offset) -> std::optional<T2RecordView> {
  const auto *header = reinterpret_cast<const ValueRecordHeader *>(mapping + offset);
  const uint64_t needed = sizeof(ValueRecordHeader) + header->key_len + header->value_len;
  if (!immutable_bounds_ok(offset, needed, read_only_offset)) {
    return std::nullopt;
  }
  return make_record_view(header);
}

// Blind direct read through a given immutable-region mapping -- the common tail end of mmap
// branches in try_read_immutable_record() below.
inline auto read_immutable_record_via(std::byte *mapping,
                                      uint64_t offset,
                                      uint64_t read_only_offset) -> std::optional<T2RecordView> {
  if (mapping == nullptr) {
    return std::nullopt;
  }
  return view_immutable_record_if_in_bounds(mapping, offset, read_only_offset);
}

// Embedded block-count size hint for a T1 payload, plus margin covering 8-byte alignment.
inline auto immutable_size_hint(uint64_t payload_bits) noexcept -> uint64_t {
  return t2_detail::payload_hint_bytes(payload_bits, t2_detail::kImmutableSizeHintMargin);
}

// Bounded pread() + header-bounds verification into `buf`, shared by ReadPolicyPreadOnly and
// large Gets. Short reads/errors return std::nullopt (caller falls back to seqlock).
inline auto read_immutable_via_pread(const T2Memory *mem,
                                     uint64_t offset,
                                     uint64_t read_only_offset,
                                     uint64_t size_hint,
                                     std::vector<std::byte> *buf) -> std::optional<T2RecordView> {
  if (mem->reads.pread_fd < 0 || buf == nullptr) {
    return std::nullopt;
  }
  const uint64_t read_len = std::min(size_hint, read_only_offset - offset);
  buf->resize(read_len);
  const ssize_t bytes_read = ::pread(mem->reads.pread_fd, buf->data(), read_len, static_cast<off_t>(offset));
  if (bytes_read < static_cast<ssize_t>(sizeof(ValueRecordHeader))) {
    return std::nullopt;
  }
  const auto *header = reinterpret_cast<const ValueRecordHeader *>(buf->data());
  const uint64_t needed = sizeof(ValueRecordHeader) + header->key_len + header->value_len;
  if (needed > static_cast<uint64_t>(bytes_read)) {
    return std::nullopt;
  }
  return make_record_view(header);
}

// Pread-pinned immutable-region read (ReadPolicyPreadOnly) using thread-local scratch.
inline auto read_pread_pinned(const T2Memory *mem,
                              uint64_t offset,
                              uint64_t read_only_offset,
                              uint64_t size_hint) -> std::optional<T2RecordView> {
  thread_local static std::vector<std::byte> tl_pread_scratch;
  return read_immutable_via_pread(mem, offset, read_only_offset, size_hint, &tl_pread_scratch);
}

// Single entry point for every T2 immutable-region read. Returns std::nullopt when this fast
// path isn't available (offset still in the mutable region, mapping/fd missing, pread failed,
// or bounds check failed) -- callers fall back to T2FlatFile::at() + seqlock.
inline auto try_read_immutable_record(const T2Memory *mem,
                                      uint64_t payload_bits,
                                      ImmutableReader reader,
                                      std::vector<std::byte> *cold_buf,
                                      ReadPolicy policy = ReadPolicy::Adaptive) -> std::optional<T2RecordView> {
  const uint64_t offset = t2_detail::decode_offset(payload_bits);
  const uint64_t read_only_offset = mem->read_only_offset.load(std::memory_order_acquire);
  if (offset >= read_only_offset) {
    return std::nullopt;
  }

  const uint64_t size_hint = immutable_size_hint(payload_bits);
  const bool is_small = size_hint <= t2_detail::os_page_size();

  if (policy == ReadPolicy::PreadOnly) {
    return read_pread_pinned(mem, offset, read_only_offset, size_hint);
  }

  if (std::byte *mapping = mapping_for(mem, reader, is_small, policy); mapping != nullptr) {
    return read_immutable_record_via(mapping, offset, read_only_offset);
  }
  if (reader == ImmutableReader::kGet) {
    return read_immutable_via_pread(mem, offset, read_only_offset, size_hint, cold_buf);
  }
  return std::nullopt;
}

// Recovers an inlined key's length from its zero-padded 16-byte prefix by trimming trailing
// zero bytes. Only unambiguous because inlining refuses keys whose own last byte is 0x00.
inline auto inline_key_len(std::span<const std::byte> index_key) noexcept -> size_t {
  static_assert(t1_detail::kStoreKeyBytes == 2 * sizeof(uint64_t));
  static_assert(std::endian::native == std::endian::little);
  uint64_t lo_word;
  uint64_t hi_word;
  std::memcpy(&lo_word, index_key.data(), sizeof(lo_word));
  std::memcpy(&hi_word, index_key.data() + sizeof(lo_word), sizeof(hi_word));
  return hi_word != 0 ? sizeof(lo_word) + (std::bit_width(hi_word) + CHAR_BIT - 1) / CHAR_BIT
                      : (std::bit_width(lo_word) + CHAR_BIT - 1) / CHAR_BIT;
}

// Copies a T1-inlined value's bytes out of its payload word into the caller's buffer.
inline void copy_inline_value(uint64_t payload_bits, size_t size, std::byte *out) noexcept {
  std::memcpy(out, &payload_bits, size);
}

// Decodes a T1-inlined (payload_bits, raw_hash) and invokes callback(value), returning true.
// Stack-backed: the span lives only inside the callback.
template <typename Callback>
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
inline auto try_read_inline_value(uint64_t payload_bits, uint64_t raw_hash, Callback &&callback) -> bool {
  if (t1_detail::is_inline(raw_hash)) {
    const size_t size = t1_detail::decode_size(raw_hash);
    std::array<std::byte, t1_detail::kInlineValueByteCount> stack_value;
    copy_inline_value(payload_bits, size, stack_value.data());
    std::forward<Callback>(callback)(std::span<const std::byte>(stack_value.data(), size));
    return true;
  }
  return false;
}

// Whether both scan bounds decide range membership identically on 16-byte padded
// prefixes and on trimmed keys: no 0x00 in the first 16 bytes and length at most 16.
// Inlined keys are at most 16 bytes and never end in 0x00, so under clean bounds the
// sorted walk's prefix containment implies full-key containment and the per-entry
// recheck can be skipped. Evaluated once per scan call -- the bounds are constant
// across all its entries.
inline auto scan_bounds_clean(std::span<const std::byte> lower_bound,
                              std::span<const std::byte> upper_bound) noexcept -> bool {
  auto bound_is_clean = [](std::span<const std::byte> bound) noexcept {
    return bound.size() <= t1_detail::kStoreKeyBytes &&
           std::ranges::all_of(bound, [](std::byte part) { return part != std::byte{0}; });
  };
  return bound_is_clean(lower_bound) && bound_is_clean(upper_bound);
}

}  // namespace vmemkv
