# vmemkv High Level Design

## 1. Overview

vmemkv is a larger-than-memory KVS that delegates data management to the OS virtual-memory subsystem (`mmap`, `fork`, `mincore`, `madvise`) as much as possible. It consists of the following components.

- Tier 1: RAM-resident index layer
- Tier 2: large-capacity data layer on file-backed mmap
- WAL
- Checkpoint
- Background Jobs

vmemkv's goal is to leave the buffer pool, page replacement algorithms, complex compactions, and the like to the OS, keeping the implementation simple while securing production-usable performance. It aims for LSM-tree (RocksDB, LevelDB) class performance with Bitcask-like design simplicity, handling larger-than-memory data beyond LMDB's reach.

## 2. Features and Limitations

### 2.1 What It Does

- Larger-than-memory data operation
- Get / Insert / Update / Delete / Scan KVS API
- Durability via WAL and checkpoint

### 2.2 What It Does Not Do

- Multi-operation transactions
- Phantom avoidance

## 3. Common Architecture: Reorganizing Two-Region

vmemkv's core concept is the `Reorganizing Two-Region` structure with `sorted_region` and `append_region`, where periodic `reorganize` absorbs `append_region` into `sorted_region`.

- `sorted_region`: sorted region. Lookups are O(log N).
- `append_region`: unsorted region receiving inserts. With a combined hash index, lookups are O(1) expected and writes are O(1).

Shape language: `[box]` = component/region, `([rounded])` = instruction, `{diamond}` = decision, `[[double box]]` = background job.

```mermaid
flowchart TB
    subgraph R["one shard"]
        direction LR
        S["sorted region\nO(log N) binary search"]
        A["append region\nO(1) hash lookup"]
    end
    G([Get / Scan / Update / Delete]) --> S & A
    I([Insert]) --> A
    A --> RG[[reorganize:\nmerge + atomic swap]]
```

The basic behavior of this structure is as follows.

- `insert` adds a new entry to `append_region`
- reads look at both `sorted_region` and `append_region`
- `update` and `delete` basically update in place once the target entry is found
- `reorganize` merges `append_region`'s contents into `sorted_region`, dropping old versions and unneeded entries

This structure makes the steady-state insert path a simple append, and the update path an in-place update of the existing entry. It is therefore not an immutable design that stacks a new version per update like an LSM-tree.

## 4. Two-Tier Architecture

vmemkv consists of a Tier 1 holding a `Reorganizing Two-Region` plus a Tier 2 value store referenced from Tier 1 offsets.

### 4.1 Tier 1

Tier 1 is a fixed-size index layer. An `IndexEntry` holds `key prefix + key hash + payload` and anchors point lookups and range scans. The key hash is the full key hashed. In the base design the payload is the `offset` pointing at Tier 2, described below.

Tier 1 specifics:

- fixed-size entries, easy to keep as a dense array
- `sorted_region` is a key-ordered IndexEntry array, `append_region` an unordered IndexEntry array.

Additionally, as an opt-in optimization, entries with values of 8 bytes or less support **entry-level adaptive covering**, interpreting the payload as a 64-bit value instead of an `offset`. For such entries `Get()` completes on the Tier 1 hit alone, skipping Tier 2 access. See low_level_design.md 2.1.1.

### 4.2 Tier 2

Tier 2 is the variable-length value layer. Referenced by `offset` from Tier 1, it holds the actual key/value records.

Tier 2 specifics:

- handles variable-size records
- kept as a single byte array on file-backed mmap; `offset` is the byte offset from its start
- inserts, and updates whose new value doesn't fit the existing record's allocated length, append a new record at the tail and store its offset in Tier 1
- `update` rewrites the existing record in place when possible, otherwise appends a new record
- carries larger-than-memory operation and durability

Tier 1/Tier 2 responsibility split and the `offset`-connected layout:

```mermaid
flowchart TB
    subgraph OPS["client operations"]
        direction LR
        R([Get / Scan]) --- W([Insert / Update / Delete])
    end
    subgraph T1["Tier 1: sharded index (RAM)"]
        direction LR
        S1[shard 0\nsorted + append] --- S2[shard 1\nsorted + append] --- SD[... K shards]
    end
    subgraph T2["Tier 2: file-backed mmap (MAP_SHARED)"]
        direction LR
        BA[base region\nimmutable] --- TA[tail region\nmutable]
    end
    subgraph JOBS["background jobs"]
        direction LR
        RG[[reorganize]] --- CK[[checkpoint\nmsync + manifest]]
    end
    WAL[WAL file]
    OPS --> T1
    OPS -.-> T2
    T1 -- offset --> T2
    T1 -.->|append growth| JOBS
    WAL -.->|WAL bytes| JOBS
    T2 -.->|space overhead| JOBS
```

## 5. Operation Examples

### 5.1 Get

1. Search Tier 1's `sorted_region` / `append_region` for a candidate `IndexEntry`.

- Complexity is O(1) on an `append_region` hit, plus O(log N) on a `sorted_region` hit. A miss costs O(1) + O(log N).

2. Verify the hash.

- Verified to distinguish keys sharing a prefix but differing after it.

3. Reference the Tier 2 record via `offset`.

- Tier 2 is a single byte array, referenced from `base + offset`.

4. Confirm the full-key match and return.

- Verified to distinguish actually-different keys with matching prefix and hash.

For inlined entries step 3. is unnecessary; the Tier 1 payload is returned directly.

### 5.2 Insert

1. `Get()` to check whether the entry already exists
2. Append a value record at Tier 2's tail, obtaining the offset
3. Add an `IndexEntry` to Tier 1's `append_region`, writing its offset
4. WAL append + `fsync`. Success is returned to the caller only after this `fsync` completes.

For inline-eligible entries of 8 bytes or less, step 2. is skipped and the 64-bit value is written directly into the Tier 1 entry's payload.

### 5.3 Update / Delete / Scan

- Update: locate the entry with `Get()`; rewrite the Tier 2 tail record in place when it fits. When the existing record resides in the immutable base region (`offset < base_boundary`), updates to keys $\le$ 16 bytes immediately redirect to out-of-place append without reading SSD base data (prefix containment guarantees full-key identity). Keys $> 16$ bytes verify the full key against the base record before appending.
- Delete: locate the entry with `Get()`, tombstone the offset on Tier 1, then write the WAL. Tier 2 is untouched.
- Scan: narrow the range on Tier 1 first, then collect value records on Tier 2 in key order. When non-ascending physical offsets (`curr_offset < prev_offset`) are encountered, the disordered segment is enqueued for background compaction on demand.

For inlined entries, Update / Delete / Scan complete on the Tier 1 payload alone.

For offset-payload indexes, Update and Delete leave old Tier 2 records unreferenced. Physically reclaiming unreferenced Tier 2 segments via hole punching, and keeping frozen Tier 2 key-ordered, is handled by background jobs (see `background_jobs.md` and `t2_compaction_design.md`).

## 6. Background Jobs (Reorganize, Checkpoint, Compaction)

vmemkv coordinates three decoupled background maintenance mechanisms to balance foreground latency with memory/disk reclamation:

- **T1 Shard Reorganize**: Per-shard worker pool merges `append_region` into `sorted_region`, eliminating ordering fragmentation in memory and splitting shards as they grow.
- **Checkpoint Coordinator**: Pacing-gated coordinator synchronizes Tier 2's dirty tail to disk via chunked `msync(MS_SYNC)` with cooperative yielding, persists a T1 snapshot, advances `base_boundary`, and rotates the WAL.
- **T2 Compaction & Autonomous Defragmentation**: Single-flight background worker repacks disordered T1 shards, evacuates live records from sparse base segments, and hole-punches empty 8MiB segments after an epoch gate.

See [`background_jobs.md`](background_jobs.md) for the unified design and lifecycle guarantees.

## 7. Fault Tolerance and WAL

vmemkv guarantees durability through the following.

- Tier 1 is a pure in-RAM structure with no path writing itself back to disk. It is rebuilt on every boot from the T1 checkpoint file (when present) plus WAL replay.
- Updates apply to Tier 2 / Tier 1 first, and only on success WAL-append + `fsync`. Success is returned to the caller only after the WAL `fsync` completes. This ordering keeps failed applies out of the WAL; see low_level_design.md 3.2 Failure Rule for the detailed reason.
- On failure, recover by checkpoint load + WAL replay
- After checkpoint completes, the WAL rotates per low_level_design.md 5.5

## 8. Optimization Overview

Several optimizations exist (the latter two are opt-in and operate correctly disabled). See [low_level_design.md](./low_level_design.md).

- Tier 2 `madvise(MADV_RANDOM)` (always on)
- Recommended paging regime for LTM operation (classic LRU; see low_level_design.md 7.6)
- WAL Group Commit (implemented; lock-free fixed-length ring-buffer scheme; see low_level_design.md 7.1)
- entry-level adaptive covering
- O(1) misses via Bloom filter for `sorted_region` negative lookups

## Appendix A. Relationship with LineairDB

vmemkv is designed to replace LineairDB's KVS portion.

- As-is: KVS composed of multiple elements including lock-free hashtable, PLI, WAL, CPR
- To-be: vmemkv alone as the KVS core

LineairDB is treated as a wrapper providing concurrency control and table/secondary-index features around vmemkv.
