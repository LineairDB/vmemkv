// checkpoint.hpp - T1 checkpoint file and manifest formats for vmemkv checkpoint reload.
// See docs/specification/low_level_design.md 5.2-5.5.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <type_traits>
#include <vector>
#include <vmemkv/core/io.hpp>
#include <vmemkv/core/mmap.hpp>
#include <vmemkv/core/utils.hpp>

namespace vmemkv {

// ─── T1 Checkpoint Entry Layout ─────────────────────────────────────────────

// Must match t1_index.hpp's kStoreKeyBytes, without depending on T1Index internals.
inline constexpr size_t kT1ChkKeyPrefixBytes = 16;
using T1ChkKeyPrefix = std::array<std::byte, kT1ChkKeyPrefixBytes>;

// On-disk entry: (key_prefix, hash, payload_bits), 32B fixed POD. Deliberately distinct from
// T1Index's EntrySnapshot (different member order) -- callers convert on write.
struct T1ChkEntry {
  T1ChkKeyPrefix key_prefix{};
  uint64_t hash = 0;
  uint64_t payload_bits = 0;
};
inline constexpr size_t kT1ChkEntryBytes = 32;
static_assert(sizeof(T1ChkEntry) == kT1ChkEntryBytes);
static_assert(std::is_standard_layout_v<T1ChkEntry>);

// ─── Sharded T1 Checkpoint File (t1_index.chk, multi-shard layout) ─────────
//
// The header is a *trailer*, not a prefix: shard_count/boundaries aren't known until every
// shard is written, so the layout is `shard_count` sections of (entry_count, entries), then
// boundary keys, then the header last. The writer stays purely sequential (no seek-back);
// the reader locates the header at `file_size - sizeof(header)`. Written to temp, fsynced,
// reachable only once the manifest commits its generation.

inline constexpr uint32_t kShardedT1ChkMagic = 0x324B4C56;  // ASCII "VLK2" -- distinct from "VLK1"
inline constexpr uint8_t kShardedT1ChkFormatVersion = 1;

struct ShardedT1ChkFileHeader {
  uint32_t magic = kShardedT1ChkMagic;
  uint8_t format_version = kShardedT1ChkFormatVersion;
  // NOLINTNEXTLINE(modernize-avoid-c-arrays)
  uint8_t reserved[3] = {};
  uint64_t shard_count = 0;
  uint64_t boundary_count = 0;     // == shard_count - 1.
  uint64_t total_entry_count = 0;  // Cross-checks that the walk consumed exactly this much.
  // FNV-1a64, folded in write order: every shard section (entry_count then entries) + boundaries
  // + this header with checksum zeroed *last*.
  uint64_t checksum = 0;
};
inline constexpr size_t kShardedT1ChkFileHeaderBytes = 40;
static_assert(sizeof(ShardedT1ChkFileHeader) == kShardedT1ChkFileHeaderBytes);
static_assert(std::is_standard_layout_v<ShardedT1ChkFileHeader>);

// Incremental writer: open once, add_shard() per shard in ascending order, then
// finish(boundaries) once. Only running counts/checksum persist across calls, so shards never
// coexist in memory. Destroying before finish() discards the temp file (the final path is
// untouched until finish()'s atomic rename).
class ShardedT1CheckpointWriter {
 public:
  explicit ShardedT1CheckpointWriter(const std::filesystem::path &path);
  ~ShardedT1CheckpointWriter();

  ShardedT1CheckpointWriter(const ShardedT1CheckpointWriter &) = delete;
  auto operator=(const ShardedT1CheckpointWriter &) -> ShardedT1CheckpointWriter & = delete;
  ShardedT1CheckpointWriter(ShardedT1CheckpointWriter &&) = delete;
  auto operator=(ShardedT1CheckpointWriter &&) -> ShardedT1CheckpointWriter & = delete;

  // Starts a new shard section with `entry_count` entries.
  void begin_shard(uint64_t entry_count);

  // Writes a chunk of `chunk_count` entries for the current shard.
  void write_shard_chunk(const T1ChkEntry *entries, size_t chunk_count);

  void add_shard_entries(const T1ChkEntry *entries, size_t entry_count);

  // `entries` must already be sorted ascending (reorganize()'s merge output satisfies this).
  // Streams entries using a stack chunk buffer without dynamic allocations.
  template <typename EntrySnapshotLike>
  void add_shard(std::span<const EntrySnapshotLike> entries) {
    begin_shard(entries.size());
    constexpr size_t kChunkSize = 2048;
    std::array<T1ChkEntry, kChunkSize> chunk{};
    size_t chunk_len = 0;
    for (const auto &entry : entries) {
      chunk[chunk_len++] = T1ChkEntry{entry.key, entry.hash, entry.payload_bits};
      if (chunk_len == kChunkSize) {
        write_shard_chunk(chunk.data(), chunk_len);
        chunk_len = 0;
      }
    }
    if (chunk_len > 0) {
      write_shard_chunk(chunk.data(), chunk_len);
    }
  }

  // Call once, after all add_shard() calls. Writes boundaries, then the trailer header,
  // fsyncs, and renames atomically onto the final path.
  void finish(std::span<const T1ChkKeyPrefix> boundaries);

  [[nodiscard]] auto bytes_written() const noexcept -> uint64_t { return bytes_written_; }
  [[nodiscard]] auto shard_count() const noexcept -> uint64_t { return shard_count_; }

 private:
  void write_bytes_internal(const void *data, size_t size);

  std::filesystem::path final_path_;
  std::filesystem::path temp_path_;
  FdGuard file_descriptor_;
  uint64_t shard_count_ = 0;
  uint64_t total_entry_count_ = 0;
  uint64_t bytes_written_ = 0;
  FoldingChecksum checksum_;  // Running FNV-1a64 state, folded incrementally as bytes are written.
  bool finished_ = false;
};

// Reads and validates a sharded T1 checkpoint file via read-only MAP_PRIVATE mmap. Throws on
// missing file or bad magic/version/size/checksum. Views stay valid for this object's lifetime.
class ShardedT1CheckpointFile {
 public:
  explicit ShardedT1CheckpointFile(const std::filesystem::path &path);
  ~ShardedT1CheckpointFile() noexcept;

  ShardedT1CheckpointFile(const ShardedT1CheckpointFile &) = delete;
  auto operator=(const ShardedT1CheckpointFile &) -> ShardedT1CheckpointFile & = delete;
  ShardedT1CheckpointFile(ShardedT1CheckpointFile &&) = delete;
  auto operator=(ShardedT1CheckpointFile &&) -> ShardedT1CheckpointFile & = delete;

  [[nodiscard]] auto boundaries() const noexcept -> std::span<const T1ChkKeyPrefix>;
  [[nodiscard]] auto shard_count() const noexcept -> size_t { return shard_entries_.size(); }
  [[nodiscard]] auto shard_entries(size_t index) const noexcept -> std::span<const T1ChkEntry> {
    return shard_entries_[index];
  }

 private:
  MmapGuard mapping_;
  std::vector<std::span<const T1ChkEntry>> shard_entries_;
  const T1ChkKeyPrefix *boundaries_ = nullptr;
  size_t boundary_count_ = 0;
};

// ─── Manifest (<t2_path>.manifest) ─────────────────────────────────────────
//
// The single source of truth for how much of the T1/T2 files is durable and trustworthy.
// `generation` is the checkpoint LSN: the WAL replay boundary up to which both files are
// mutually consistent. Neither file is trusted beyond the manifest's promise until the
// manifest itself commits (temp write, fsync, atomic rename).

inline constexpr uint32_t kManifestMagic = 0x314B4D56;  // ASCII "VMK1"
// A non-v5 manifest reads as "no checkpoint exists yet" (same as corrupt/missing); no on-disk
// migration path exists or is needed.
inline constexpr uint8_t kManifestFormatVersion = 5;

inline constexpr size_t kT1ChunkMetaReservedBytes = 7;
// Describes one active VLK2 key-range chunk file inside `<t2_path>.t1chk/`.
struct T1ChunkMeta {
  uint64_t chunk_id = 0;
  uint64_t approx_bytes = 0;
  T1ChkKeyPrefix upper_bound{};
  uint8_t has_upper_bound = 0;
  // NOLINTNEXTLINE(modernize-avoid-c-arrays)
  uint8_t reserved[kT1ChunkMetaReservedBytes] = {};
};
inline constexpr size_t kT1ChunkMetaBytes = 40;
static_assert(sizeof(T1ChunkMeta) == kT1ChunkMetaBytes);
static_assert(std::is_standard_layout_v<T1ChunkMeta>);

struct ManifestHeader {
  uint32_t magic = kManifestMagic;
  uint8_t format_version = kManifestFormatVersion;
  // NOLINTNEXTLINE(modernize-avoid-c-arrays)
  uint8_t reserved[3] = {};
  // == checkpoint_lsn of the committing cycle: the WAL replay boundary.
  uint64_t generation = 0;
  uint64_t t2_bytes_used = 0;  // Logical durable T2 bytes -- not recoverable from file size
                               // (capacity-truncated files are larger than their live data).
  uint64_t next_t1_chunk_id = 1;
  uint64_t t1_chunk_count = 0;
  uint64_t checksum = 0;  // FNV-1a64 over header (checksum zeroed) followed by t1_chunk_count T1ChunkMeta records.
};
inline constexpr size_t kManifestHeaderBytes = 48;
static_assert(sizeof(ManifestHeader) == kManifestHeaderBytes);
static_assert(std::is_standard_layout_v<ManifestHeader>);

struct ManifestData {
  uint64_t generation = 0;
  uint64_t t2_bytes_used = 0;
  uint64_t next_t1_chunk_id = 1;
  std::vector<T1ChunkMeta> t1_chunks;
};

// Temp-write, fsync, atomic rename. Call only after `generation`'s T1 and T2 files are fully
// written and fsynced.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void write_manifest(const std::filesystem::path &manifest_path,
                    uint64_t generation,
                    uint64_t t2_bytes_used,
                    std::span<const T1ChunkMeta> t1_chunks = {},
                    uint64_t next_t1_chunk_id = 1);

// Returns manifest state, or nullopt when missing or invalid (bad magic, version, layout,
// checksum) -- callers treat that as "no checkpoint exists yet".
auto read_manifest(const std::filesystem::path &manifest_path) -> std::optional<ManifestData>;

// Persists newly created or renamed directory entries inside `dir_path`.
void fsync_directory(const std::filesystem::path &dir_path);

// Removes obsolete chunk files or orphaned temps inside `t1_chk_dir` not listed in `active_chunks`.
void cleanup_stale_t1_chunks(const std::filesystem::path &t1_chk_dir, std::span<const T1ChunkMeta> active_chunks);

}  // namespace vmemkv
