#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <unordered_set>
#include <vmemkv/checkpoint/checkpoint.hpp>
#include <vmemkv/core/io.hpp>
#include <vmemkv/core/mmap.hpp>

namespace vmemkv {

namespace {

// Writes `size` bytes fully, fsyncs, and closes `file_descriptor` -- shared tail for the T1
// checkpoint file and the manifest. Chunked with periodic fdatasync()+DONTNEED so the
// checkpoint doesn't evict T2's resident pages from page cache; only the final fsync is
// a hard durability requirement.
void write_fsync_close(int file_descriptor, const void *data, size_t size, const char *what) {
  constexpr size_t kSyncIntervalBytes = 512ULL * 1024 * 1024;
  const auto *bytes = static_cast<const std::byte *>(data);
  size_t offset = 0;
  size_t synced = 0;
  while (offset < size) {
    const size_t chunk = std::min(kSyncIntervalBytes, size - offset);
    try {
      write_all_exact(file_descriptor, bytes + offset, chunk, what);
    } catch (...) {
      ::close(file_descriptor);
      throw;
    }
    offset += chunk;
    if (offset - synced >= kSyncIntervalBytes && ::fdatasync(file_descriptor) == 0) {
      ::posix_fadvise(
          file_descriptor, static_cast<off_t>(synced), static_cast<off_t>(offset - synced), POSIX_FADV_DONTNEED);
      synced = offset;
    }
  }
  if (::fsync(file_descriptor) != 0) {
    const int err = errno;
    ::close(file_descriptor);
    throw std::system_error(err, std::generic_category(), what);
  }
  ::posix_fadvise(file_descriptor, 0, 0, POSIX_FADV_DONTNEED);
  ::close(file_descriptor);
}

// Atomically publishes a temp file onto its final path. Shared by write_via_temp_then_rename
// and ShardedT1CheckpointWriter::finish().
void rename_temp_to_final(const std::filesystem::path &temp_path,
                          const std::filesystem::path &final_path,
                          const char *what) {
  std::error_code rename_ec;
  std::filesystem::rename(temp_path, final_path, rename_ec);
  if (rename_ec) {
    throw std::system_error(rename_ec, what);
  }
}

// Opens a fresh temp file beside `final_path`, invokes `write_body(file_descriptor)`, then
// renames it atomically onto `final_path`: the "write to temp, fsync, rename over"
// crash-safety pattern in exactly one place.
template <typename WriteBody>
void write_via_temp_then_rename(const std::filesystem::path &final_path, WriteBody write_body) {
  const std::filesystem::path temp_path = final_path.string() + ".tmp";
  // Ownership transfers to write_body, which closes it on all paths (see write_fsync_close).
  const int file_descriptor =
      open_create_0600(temp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, "open checkpoint temp file");
  write_body(file_descriptor);

  rename_temp_to_final(temp_path, final_path, "rename checkpoint file");
}

// Shard sections: entry spans plus the offset where boundaries begin.
struct ParsedShardSections {
  std::vector<std::span<const T1ChkEntry>> entries;
  size_t end_offset = 0;
  uint64_t seen_entries = 0;
};

// Walks the sections, folding counts-then-spans into the checksum. Bounds-checked: a corrupt
// entry_count fails here, not at the final checksum. Nullopt on any layout violation.
auto parse_shard_sections(const std::byte *base,
                          size_t payload_limit,
                          const ShardedT1ChkFileHeader &header,
                          FoldingChecksum &checksum) -> std::optional<ParsedShardSections> {
  ParsedShardSections parsed;
  parsed.entries.reserve(header.shard_count);
  size_t offset = 0;
  for (uint64_t shard_index = 0; shard_index < header.shard_count; ++shard_index) {
    if (offset + sizeof(uint64_t) > payload_limit) {
      return std::nullopt;
    }
    uint64_t entry_count = 0;
    std::memcpy(&entry_count, base + offset, sizeof(entry_count));
    checksum.add(base + offset, sizeof(entry_count));
    offset += sizeof(entry_count);
    const size_t entry_bytes = entry_count * sizeof(T1ChkEntry);
    if (offset + entry_bytes > payload_limit) {
      return std::nullopt;
    }
    parsed.entries.emplace_back(reinterpret_cast<const T1ChkEntry *>(base + offset), static_cast<size_t>(entry_count));
    if (entry_bytes > 0) {
      checksum.add(base + offset, entry_bytes);
    }
    offset += entry_bytes;
    parsed.seen_entries += entry_count;
  }
  parsed.end_offset = offset;
  return parsed;
}

// Validates trailing boundaries against the sections (exact fit, entry totals, count
// relation) and folds them in. Boundary span start, or nullptr on violation.
auto parse_shard_boundaries(const std::byte *base,
                            size_t offset,
                            size_t payload_limit,
                            const ShardedT1ChkFileHeader &header,
                            uint64_t seen_entries,
                            FoldingChecksum &checksum) -> const T1ChkKeyPrefix * {
  const size_t boundary_bytes = header.boundary_count * sizeof(T1ChkKeyPrefix);
  if (offset + boundary_bytes != payload_limit || seen_entries != header.total_entry_count ||
      header.boundary_count + 1 != header.shard_count) {
    return nullptr;
  }
  if (boundary_bytes > 0) {
    checksum.add(base + offset, boundary_bytes);
  }
  return reinterpret_cast<const T1ChkKeyPrefix *>(base + offset);
}

}  // namespace

ShardedT1CheckpointWriter::ShardedT1CheckpointWriter(const std::filesystem::path &path)
    : final_path_(path), temp_path_(path.string() + ".tmp") {
  file_descriptor_ = FdGuard{
      open_create_0600(temp_path_.c_str(), O_WRONLY | O_CREAT | O_TRUNC, "open sharded t1 checkpoint temp file")};
}

ShardedT1CheckpointWriter::~ShardedT1CheckpointWriter() {
  // Unfinished: the temp file never became reachable, so discarding is tidiness, not
  // correctness -- best-effort, matching the temp+rename crash-safety contract.
  if (!finished_) {
    remove_quiet(temp_path_);
  }
}

void ShardedT1CheckpointWriter::write_bytes_internal(const void *data, size_t size) {
  write_all_exact(file_descriptor_.fd, data, size, "write sharded t1 checkpoint");
  checksum_.add(data, size);
  bytes_written_ += size;
}

void ShardedT1CheckpointWriter::begin_shard(uint64_t entry_count) {
  write_bytes_internal(&entry_count, sizeof(entry_count));
  ++shard_count_;
  total_entry_count_ += entry_count;
}

void ShardedT1CheckpointWriter::write_shard_chunk(const T1ChkEntry *entries, size_t chunk_count) {
  if (chunk_count > 0) {
    write_bytes_internal(entries, chunk_count * sizeof(T1ChkEntry));
  }
}

void ShardedT1CheckpointWriter::add_shard_entries(const T1ChkEntry *entries, size_t entry_count) {
  begin_shard(entry_count);
  write_shard_chunk(entries, entry_count);
}

void ShardedT1CheckpointWriter::finish(std::span<const T1ChkKeyPrefix> boundaries) {
  if (!boundaries.empty()) {
    const size_t boundary_bytes = boundaries.size() * sizeof(T1ChkKeyPrefix);
    write_bytes_internal(boundaries.data(), boundary_bytes);
  }

  ShardedT1ChkFileHeader header;
  header.shard_count = shard_count_;
  header.boundary_count = boundaries.size();
  header.total_entry_count = total_entry_count_;
  // Checksum covers the header zeroed, folded last (every other format folds first).
  checksum_.add_header(header);
  header.checksum = checksum_.value();

  // write_fsync_close() closes the fd itself on all paths -- release our copy first so a
  // throw doesn't double-close in the destructor.
  const int trailer_fd = file_descriptor_.release();
  write_fsync_close(trailer_fd, &header, sizeof(header), "write sharded t1 checkpoint trailer");
  bytes_written_ += sizeof(header);
  finished_ = true;

  rename_temp_to_final(temp_path_, final_path_, "rename sharded t1 checkpoint file");
}

ShardedT1CheckpointFile::ShardedT1CheckpointFile(const std::filesystem::path &path) {
  const FdGuard file_descriptor{open_file(path.c_str(), O_RDONLY, "open sharded t1 checkpoint")};

  const auto file_size = fstat_size(file_descriptor.fd, "fstat sharded t1 checkpoint");
  if (file_size < kShardedT1ChkFileHeaderBytes) {
    throw std::runtime_error("sharded t1 checkpoint file smaller than its trailer");
  }

  void *mapped = mmap_shared_or_throw(
      static_cast<size_t>(file_size), file_descriptor.fd, PROT_READ, "mmap sharded t1 checkpoint", MAP_PRIVATE, 0);

  const auto *base = static_cast<const std::byte *>(mapped);
  ShardedT1ChkFileHeader header;
  std::memcpy(&header, base + (file_size - kShardedT1ChkFileHeaderBytes), sizeof(header));

  const bool magic_ok = header.magic == kShardedT1ChkMagic && header.format_version == kShardedT1ChkFormatVersion;

  // Walk sections and boundaries in finish()'s checksum convention, recording each shard's
  // span. Bounds-checked: corrupt counts fail here, not at the checksum.
  bool layout_ok = true;
  FoldingChecksum checksum;
  uint64_t seen_entries = 0;
  std::vector<std::span<const T1ChkEntry>> shard_entries;
  const T1ChkKeyPrefix *boundaries = nullptr;
  const size_t payload_limit = file_size - kShardedT1ChkFileHeaderBytes;
  if (magic_ok) {
    if (auto sections = parse_shard_sections(base, payload_limit, header, checksum)) {
      shard_entries = std::move(sections->entries);
      seen_entries = sections->seen_entries;
      boundaries = parse_shard_boundaries(base, sections->end_offset, payload_limit, header, seen_entries, checksum);
      layout_ok = (boundaries != nullptr);
    } else {
      layout_ok = false;
    }
  }

  bool checksum_ok = false;
  if (magic_ok && layout_ok) {
    ShardedT1ChkFileHeader header_for_hash = header;
    header_for_hash.checksum = 0;
    const auto header_bytes =
        std::span<const std::byte>(reinterpret_cast<const std::byte *>(&header_for_hash), sizeof(header_for_hash));
    checksum_ok = fnv1a64_range(checksum.value(), header_bytes) == header.checksum;
  }

  if (!magic_ok || !layout_ok || !checksum_ok) {
    throw std::runtime_error("sharded t1 checkpoint file failed validation (magic/version/layout/checksum)");
  }

  mapping_ = MmapGuard(mapped, static_cast<size_t>(file_size));
  shard_entries_ = std::move(shard_entries);
  boundaries_ = boundaries;
  boundary_count_ = header.boundary_count;
}

ShardedT1CheckpointFile::~ShardedT1CheckpointFile() noexcept = default;

auto ShardedT1CheckpointFile::boundaries() const noexcept -> std::span<const T1ChkKeyPrefix> {
  return {boundaries_, boundary_count_};
}

namespace {

auto validate_manifest_chunks(std::span<const T1ChunkMeta> chunks, uint64_t next_chunk_id) -> bool {
  for (size_t i = 0; i < chunks.size(); ++i) {
    if (chunks[i].chunk_id == 0 || chunks[i].chunk_id >= next_chunk_id) {
      return false;
    }
    const bool expect_bound = (i + 1 < chunks.size());
    if ((chunks[i].has_upper_bound != 0) != expect_bound) {
      return false;
    }
    if (i > 0 && !(chunks[i - 1].upper_bound < chunks[i].upper_bound) && expect_bound) {
      return false;
    }
  }
  return true;
}

auto compute_manifest_checksum(ManifestHeader header, std::span<const T1ChunkMeta> chunks) noexcept -> uint64_t {
  header.checksum = 0;
  FoldingChecksum checksum;
  checksum.add_header(header);
  if (!chunks.empty()) {
    checksum.add(chunks.data(), chunks.size_bytes());
  }
  return checksum.value();
}

}  // namespace

void write_manifest(const std::filesystem::path &manifest_path,
                    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                    uint64_t generation,
                    uint64_t t2_bytes_used,
                    std::span<const T1ChunkMeta> t1_chunks,
                    uint64_t next_t1_chunk_id) {
  ManifestHeader header;
  header.generation = generation;
  header.t2_bytes_used = t2_bytes_used;
  header.next_t1_chunk_id = next_t1_chunk_id;
  header.t1_chunk_count = t1_chunks.size();
  header.checksum = compute_manifest_checksum(header, t1_chunks);

  std::vector<std::byte> payload(sizeof(header) + t1_chunks.size_bytes());
  std::memcpy(payload.data(), &header, sizeof(header));
  if (!t1_chunks.empty()) {
    std::memcpy(payload.data() + sizeof(header), t1_chunks.data(), t1_chunks.size_bytes());
  }

  write_via_temp_then_rename(manifest_path, [&](int file_descriptor) {
    write_fsync_close(file_descriptor, payload.data(), payload.size(), "write manifest");
  });
}

auto read_manifest(const std::filesystem::path &manifest_path) -> std::optional<ManifestData> {
  const int file_descriptor = ::open(manifest_path.c_str(), O_RDONLY);
  if (file_descriptor < 0) {
    return std::nullopt;  // Missing manifest == "no checkpoint exists yet", not an error.
  }
  const FdGuard fd_guard{file_descriptor};

  const auto file_size = fstat_size(fd_guard.fd, "fstat manifest");
  if (file_size < sizeof(ManifestHeader)) {
    return std::nullopt;  // Torn/truncated manifest: treat as absent, fall back safely.
  }

  ManifestHeader header;
  if (::pread(fd_guard.fd, &header, sizeof(header), 0) != static_cast<ssize_t>(sizeof(header))) {
    return std::nullopt;
  }
  if (header.magic != kManifestMagic || header.format_version != kManifestFormatVersion ||
      header.next_t1_chunk_id == 0) {
    return std::nullopt;
  }
  if (header.t1_chunk_count > (std::numeric_limits<size_t>::max() - sizeof(header)) / sizeof(T1ChunkMeta)) {
    return std::nullopt;
  }
  const size_t chunks_bytes = static_cast<size_t>(header.t1_chunk_count) * sizeof(T1ChunkMeta);
  if (file_size != sizeof(header) + chunks_bytes) {
    return std::nullopt;
  }

  ManifestData data;
  data.generation = header.generation;
  data.t2_bytes_used = header.t2_bytes_used;
  data.next_t1_chunk_id = header.next_t1_chunk_id;
  data.t1_chunks.resize(header.t1_chunk_count);
  if (chunks_bytes > 0 &&
      ::pread(fd_guard.fd, data.t1_chunks.data(), chunks_bytes, sizeof(header)) != static_cast<ssize_t>(chunks_bytes)) {
    return std::nullopt;
  }
  if (!validate_manifest_chunks(data.t1_chunks, data.next_t1_chunk_id) ||
      compute_manifest_checksum(header, data.t1_chunks) != header.checksum) {
    return std::nullopt;
  }
  return data;
}

void fsync_directory(const std::filesystem::path &dir_path) {
  const FdGuard dir_fd{open_file(dir_path.c_str(), O_RDONLY | O_DIRECTORY, "open t1 checkpoint directory")};
  if (::fsync(dir_fd.fd) != 0) {
    throw std::system_error(errno, std::generic_category(), "fsync t1 checkpoint directory");
  }
}

void cleanup_stale_t1_chunks(const std::filesystem::path &t1_chk_dir, std::span<const T1ChunkMeta> active_chunks) {
  std::error_code dir_error;
  if (!std::filesystem::exists(t1_chk_dir, dir_error)) {
    return;
  }
  std::unordered_set<std::string> keep;
  keep.reserve(active_chunks.size());
  for (const auto &meta : active_chunks) {
    keep.insert(derive_t1_chunk_filename(meta.chunk_id));
  }
  for (const auto &entry : std::filesystem::directory_iterator(t1_chk_dir, dir_error)) {
    if (dir_error) {
      break;
    }
    if (!keep.contains(entry.path().filename().string())) {
      remove_quiet(entry.path());
    }
  }
}

}  // namespace vmemkv
