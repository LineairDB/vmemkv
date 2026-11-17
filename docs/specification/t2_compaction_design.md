# T2 Compaction and Hole-Punch Specification

Tier 2 is an append-only value log. Checkpoint persists the new log tail in
place; background compaction restores 4KiB page locality for scans and reclaims
dead disk space from overwrites/deletes using a single unified primitive:
**whole-shard key-ordered repack (`repack_dirty_shard`)**, followed by
epoch-gated hole punching of empty 8MiB segments.

## Shard Sizing and $O(1)$ Queueing

Each T1 shard splits when its live entries reach either the entry-count threshold
or one T2 segment (`kSegmentBytes` = 8 MiB) of referenced T2 live bytes. Because
every T1 shard is bounded by one T2 segment, repacking a T1 shard is the single
unit of work for both scan-locality repair and segment defragmentation.

### 1. Dirty Shard Queue (`dirty_queue_` / `t2_dirty`)

Each T1 shard slot holds an atomic `t2_dirty` flag and participates in `dirty_queue_`
(`ShardQueue<&ShardSlot::t2_dirty>`):

- On a foreground T2 offset write (`put`), if `t2_dirty` is `false`, it transitions
  `false -> true` via CAS and appends the shard to `dirty_queue_` in $O(1)$.
- Subsequent writes to an already-dirty shard observe `t2_dirty == true` with a
  single relaxed load and return immediately with zero lock or atomic overhead.
- When a dirty shard splits, `continue_split()` purges the parent shard and
  enqueues both child shards into `dirty_queue_`.

### 2. Garbage Segment Queue (`garbage_head_` / `dirty_flag_`)

T2 live-byte counters (`seg_live_`) are tracked per 8MiB segment (`kSegmentBytes`).
An append contributes its aligned size hint to every segment it touches; an
overwrite or deletion subtracts the old record's size hint via `subtract_segment()`:

- **Below 50% Live Bytes**: When a segment's live bytes drop below 50% of
  `kSegmentBytes` (`< 4 MiB`), `mark_segment_dirty(seg)` sets `dirty_flag_[seg]`
  and pushes the segment onto the lock-free `garbage_head_` stack in $O(1)$.
- **Zero Live Bytes**: When a segment's live bytes reach zero (`seg_live == 0`),
  `dirty_flag_[seg]` is cleared and the segment is pushed onto the lock-free
  `punch_head_` stack tagged with the current completed-checkpoint epoch.

## Unified Compaction Cycle (`compact_cycle`)

The background compaction worker runs in a single-flight loop:

1. **Hole Punching (`punch_evacuated_segments`)**:
   Punches eligible empty segments (`seg_live == 0`) from `punch_head_`.
2. **Dirty-Shard Repack (`for_each_dirty_shard`)**:
   Pops shards from `dirty_queue_`. For each shard, `repack_dirty_shard()`
   collects the shard's live T2 payloads in full-key order (`collect_shard_payloads`,
   reading T2 only when consecutive entries share an identical 16-byte prefix) and
   checks in RAM whether:
   - any adjacent pair of entries is not 4KiB-page-local (`!are_t2_payloads_page_local`), or
   - any entry overlaps a frozen garbage segment (`payload_overlaps_garbage_segment`).
   If neither condition holds (e.g. sequential inserts), no T2 writes occur.
   Otherwise, the entire shard is rewritten contiguously in key order to the T2
   append frontier (`relocate_one`), updating T1 via stripe-locked, offset-validated
   CAS (`mark_t2_dirty = false`) without emitting WAL records.
3. **Garbage-Segment Evacuation (`for_each_shard_for_compaction`)**:
   If movement budget remains (`moved_bytes < kCompactionMaxMoveBytesPerCycle`) and
   `garbage_head_` holds any frozen segment below `read_only_offset` (`has_garbage_segment`),
   the worker visits T1 shards and runs the same `repack_dirty_shard()` on any shard
   referencing a frozen garbage segment. Evacuating those shards drops the source
   segments' `seg_live_` to zero, moving them into `punch_head_`. Once all shards
   are visited, `clear_garbage_segments()` clears the processed frozen segments.

Work stops between shards when `kCompactionMaxMoveBytesPerCycle` (8 MiB) is reached.

## Hole Punching and Epoch Gate

Hole punching reclaims space by deallocating 8MiB segments on the underlying
filesystem via `fallocate(FALLOC_FL_PUNCH_HOLE)`:

1. When a segment's live-byte count reaches zero during compaction or delete, it
   is enqueued as a punch candidate tagged with the current completed checkpoint
   count.
2. **Epoch Gate**: A candidate segment is eligible for hole-punching only after
   two subsequent Checkpoint-mode cycles have completed since its evacuation.
   This ensures that no active readers or adopted T1 checkpoints can access the
   unpunched physical offsets without WAL replay.
3. If hole-punching is unsupported by the filesystem (e.g. tmpfs), candidates are
   retained without punching and repacking continues safely.

## Checkpoint Coordination and Pacing

Checkpoint durabilizes T2's Mutable Region and updates the T1 checkpoint chunks
and `MANIFEST`.

### Subdivided Synchronization

Checkpoint synchronizes `[old_read_only_offset, target)` using `msync(MS_SYNC)`
subdivided into chunks of `kMsyncChunkBytes` (64 MiB). The coordinator yields
execution (`std::this_thread::yield()`) between chunks to prevent writeback
monopolization and reduce foreground latency spikes.

### Automatic Triggering and Emergency Cap

The background reorganize worker executes checkpoints based on two conditions:

1. **WAL-Size Trigger**: Fires immediately when WAL appended bytes exceed
   `kMaxBytesSinceCheckpoint` (64 MiB).
2. **Punch-Backlog Emergency Cap**: If the punch backlog reaches
   `kPunchBacklogCheckpointCapSegments` (1,058 segments), a checkpoint fires
   to bound unreclaimed disk space.

Explicit `checkpoint()` calls execute immediately.

## Files

- `src/vmemkv/impl/compaction.hpp`: unified `repack_dirty_shard()`, `compact_cycle()`,
  and `punch_evacuated_segments()`.
- `src/vmemkv/impl/t2_ownership.hpp`: segment live-byte counters, lock-free
  `garbage_head_` and `punch_head_` stacks, and epoch gating.
- `src/vmemkv/t1_index/sharded_t1_index.hpp`: byte-bounded shard splitting and
  `dirty_queue_` (`ShardQueue<&ShardSlot::t2_dirty>`).
- `src/vmemkv/impl/checkpoint_coordinator.hpp`: chunked msync, manifest
  publication, and reorg trigger pacing.
- `src/vmemkv/vmemkv_impl.hpp`: compaction and reorganize background worker loops.
