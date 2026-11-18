# VMemKV Terminology and Taxonomy

Declarative glossary unifying VMemKV concepts with **FASTER** and **LSM-tree** terminology, and isolating where VMemKV diverges.

## 1. Glossary & Unified Naming

### Architecture & Records

| Unified Term | Current Code Symbol | Standard Reference | Concise Definition |
| --- | --- | --- | --- |
| **Tier 1 (T1)** | `T1Index`, `ShardedT1Index` | In-RAM LSM / Index | Range-sharded, RAM-resident 32B-slot primary key index. |
| **Tier 2 (T2)** | `T2FlatFile`, `T2Memory` | WiscKey vLog / FASTER Log | Single `MAP_SHARED` value-log file addressed by byte offset. |
| **Index Entry** | `SortedSlot`, `MemTableSlot` | Key-Value Index Slot | 16B key prefix, 60-bit hash, and 64-bit T2 offset or inline value. |
| **Tombstone** | `STORE_NOT_FOUND` (`~0ULL`) | LSM Tombstone | Sentinel payload marking a logically deleted key. |
| **Value Record** | `ValueRecordHeader` + payload | HybridLog / vLog Record | Variable-length T2 record with `alloc_len` headroom and seqlock. |
| **Embedded Size Hint** | `t2_detail::payload_hint_bytes`, `encode_offset_payload` | Novel (T1-encoded length) | 16B-granular T2 record size in T1 for sizing `pread()` and segments. |
| **Adaptive Covering** | `UseT1InlineValue` | Inline K-V Payload | Stores `<=8B` values (`<=16B` non-zero-end keys) directly in T1. |

### Tier 1 (Aligned with LSM-tree)

| Unified Term | Current Code Symbol | LSM-tree Equivalent | Concise Definition |
| --- | --- | --- | --- |
| **Active MemTable** | `MemTable`, `memtable_active_`, `memtable_live_count` | Active MemTable | Unsorted mutable per-shard buffer paired with `MemTableIndex`. |
| **Immutable MemTable** | `memtable_immutable_` | Immutable MemTable | Frozen write buffer snapshot queued for in-memory merge. |
| **In-Memory Sorted Run** | `SortedRun`, `sorted_run_` | Sorted Run / SSTable + Bloom | Single binary-searchable per-shard entry array with a Bloom filter. |
| **In-Memory Compaction** | `T1Index::reorganize()`, `ShardedT1Index::reorganize()` | MemTable Flush / Compaction | Merges non-empty MemTables into each shard's single `SortedRun`. |
| **Freeze Guard** | `FreezableRegion` | Fuzzy-Region Bypass | Forces updates onto `Active MemTable` while a snapshot merges. |
| **T1 Checkpoint Chunks** | `<t2>.t1chk/chunk_*.chk` (`VLK2`) | Partitioned Index SSTables | 32 MiB key-range checkpoint files holding serialized sorted shards. |
| **MANIFEST** | `<t2>.manifest` (v5) | LSM `MANIFEST` | Atomic metadata tracking checkpoint LSN, `t2_bytes_used`, and chunks. |

### Tier 2 (Aligned with Microsoft FASTER)

| Unified Term | Current Code Symbol | FASTER Equivalent | Concise Definition |
| --- | --- | --- | --- |
| **Immutable Region** | `immutable_region` (`[0, read_only_offset)`) | Read-Only / Stable Region | Durable read-only T2 prefix; never edited in place. |
| **Mutable Region** | `mutable_region` (`[read_only_offset, tail_offset)`) | Mutable Region | Uncheckpointed T2 suffix; updatable in place within `alloc_len`. |
| **Read-Only Offset** | `T2Memory::read_only_offset` | `ReadOnlyOffset` | Monotonic offset dividing T2 `Immutable` and `Mutable` regions. |
| **Tail Offset** | `T2Memory::tail_offset` | `TailOffset` | Monotonic frontier where new records and copy-on-write updates append. |
| **Fuzzy Watermark** | `T2Ownership::watermark_`, `InPlaceUpdateBarrier` | Fuzzy Region / `SafeReadOnlyOffset` | Pre-flush boundary draining active in-place writes to stop lost updates. |
| **Segment** | `T2Config::kSegmentBytes` (`8 MiB`) | Fixed Compaction Unit | 8 MiB T2 accounting unit bounding shard size and hole-punch reclamation. |

### Sharding, Concurrency & Compaction

| Unified Term | Current Code Symbol | Standard Reference | Concise Definition |
| --- | --- | --- | --- |
| **Range Shard** | `Shard`, `ShardSlot`, `Directory` | B-link Range Partition | Disjoint key-interval `T1Index` capped at 8 MiB of live T2 bytes. |
| **B-link Forwarding** | `kClosingSentinel` -> `Split{boundary,low,high}` | B-link Forwarding Pointer | Non-blocking two-phase shard split via atomic `superseded` pointer. |
| **Epoch Domain** | `EpochDomain`, `ThreadReferenceTracker` | Epoch-Based Reclamation (EBR) | Store-wide EBR registry guarding T1 directory and shard buffers. |
| **Writer Gate** | `WriterGate`, `stop_writers_and_wait` | EBR Quiesce Barrier | Drains in-flight T2 writers at checkpoint frontier capture. |
| **Key Stripe Lock** | `write_stripes_` (256 mutexes) | Striped Key Latching | Hash-partitioned mutex serializing same-key T1/T2 mutations. |
| **Seqlock** | `SlotBase::version`, `ValueRecordHeader::version` | Linux Seqlock | Odd/even sequence guard enabling lock-free reads of mutable slots. |
| **Quadfecta Read Path** | `T2ReadHandles` (3 mmaps + `pread_fd`) | Novel (VM Alias + `pread`) | Routes `random`/`seq`/`normal` mmap or `pread` by op (`Get`/`Scan`) and size. |
| **Unified Shard Repack** | `repack_dirty_shard()`, `compact_cycle()` | Novel (T1/T2 Co-Compaction) | Rewrites one shard key-ordered to fix 4 KiB scan locality and sparse segments. |
| **Dirty Shard Queue** | `dirty_queue_`, `t2_dirty` | Write-Dirtied Work Set | O(1) FIFO set tracking T1 shards needing scan-locality check or repack. |
| **Garbage Segment** | `garbage_head_` (`<50%` live) | Sparse Log Segment | Frozen `< read_only_offset` segment below 50% live bytes awaiting evacuation. |
| **Epoch-Gated Hole Punch** | `punch_head_`, `FALLOC_FL_PUNCH_HOLE` | Novel (Zero-WAL Reclaim) | Deallocates empty 8 MiB segments after two checkpoints cover relocation. |
| **WAL & Group Commit** | `Wal`, `reserve_*` / `await_durable` | ARIES WAL + Group Commit | Redo log reserving LSNs under stripe lock and batching `fdatasync()`. |

---

## 2. Structural Alignment vs. Novelty Extraction

### 2.1 Concepts Identical to Standard Systems

- **From FASTER (Microsoft)**: Dividing an append-oriented log into a **Mutable Region** (in-place updates allowed) and an **Immutable / Read-Only Region** (updates trigger read-copy-update to tail) at a `ReadOnlyOffset`, using EBR (`EpochDomain`) for safe memory reclamation and a fuzzy-region barrier (`watermark_` + `InPlaceUpdateBarrier`) to avoid lost updates during flush.
- **From LSM-tree / WiscKey**: Key-value separation into a RAM index (`Tier 1`) plus value log (`Tier 2`), buffering writes in a lock-free `Active MemTable` -> `Immutable MemTable` -> sorted array + Bloom filter (`In-Memory Sorted Run`), `Tombstone` deletion markers, `MANIFEST` atomic metadata commits, and partitioned sorted checkpoint files.
- **From B-link Trees & Traditional Systems**: Atomic range-split forwarding pointers (`Closing` -> `Split`), hash-striped per-key locks, seqlocks, and group-committed WAL.

### 2.2 Where VMemKV Diverges (Core Novelties)

1. **OS Page-Cache HybridLog Without User-Space Buffer Pool**: FASTER bounds a circular RAM ring `[HeadOffset, TailOffset)` and manages eviction (`HeadOffset`) in user space. VMemKV maps the entire T2 file via `MAP_SHARED | MAP_NORESERVE`, leaving page residency/eviction to the Linux kernel and using `read_only_offset` strictly for *immutability/durability* rather than RAM residency.
2. **Single In-RAM Sorted Run (Zero Disk Read Amplification in T1)**: Unlike disk-backed multi-level LSM-trees (`L0..Ln`), each shard merges immediately into **one** RAM-resident sorted run, keeping negative lookups at O(1) Bloom checks without disk I/O.
3. **Immutable-Region Multi-VMA + `pread()` Quadfecta**: Exploiting `read_only_offset` immutability (no seqlock needed), VMemKV aliases one physical file across three VMAs (`MADV_RANDOM` for small `Get`, `MADV_SEQUENTIAL` for small `Scan`, `MADV_NORMAL` for large `Scan > 4 KiB`) plus `pread_fd` directly for large random `Get > 4 KiB` (no `mincore()`), avoiding both multi-page fault amplification and `mm->mmap_lock` overhead.
4. **Segment-Coupled Byte-Bounded Sharding & Unified Repack**: Splitting T1 shards by *referenced T2 bytes* (`8 MiB = 1 Segment`) unifies WiscKey-style 4 KiB range-scan locality repair and Bitcask/FASTER-style sparse segment reclamation into a single `repack_dirty_shard()` primitive capped at one segment per cycle.
5. **Zero-WAL Compaction via Two-Checkpoint Epoch Gate**: Because immutable copies stay reachable through checkpointed T1 until two subsequent checkpoints persist post-relocation pointers and `msync()`ed T2 data, compaction writes **zero WAL records**, eliminating WAL amplification during space reclamation.
6. **Hybrid In-Place + Adaptive Inline Slot Format**: Combines entry-level `<=8B` value inlining in T1 (`0` T2 touch) with `alloc_len`-bounded in-place overwrites in T2's Mutable Region and in-place T1 slot updates via seqlock instead of stacking delta versions.

