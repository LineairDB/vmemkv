# T2 Compaction and Hole-Punch Specification

Tier 2 is an append-only value log. Checkpoint persists the new log tail in
place; background compaction repacks dirty T1 key-range shards, evacuates sparse
dirty segments to the log frontier, and hole-punches eligible empty 8MiB segments.

## Shard and Segment Tracking

Each T1 shard maintains a `t2_dirty` flag indicating that the shard contains T2
appends whose physical page locality has not yet been verified.

T2 live-byte counters are tracked per 8MiB segment (`kSegmentBytes`). An append
or update contributes its size to every segment it overlaps. An overwrite or
deletion subtracts the record size from the corresponding segment counter and
marks the segment dirty (`dirty_flag_`). When a segment's live-byte count
reaches zero, the segment enters the punch candidate queue.

## Compaction Cycle

The background compaction worker runs in a single-flight loop:

1. **Hole Punching**: Punch eligible empty segments from the candidate queue.
2. **Dirty-Shard Repack**: Select dirty T1 shards in ascending key-range order.
   For each entry whose T2 record is not 4KiB-page-local with its predecessor or
   successor (`are_t2_payloads_page_local`), copy the live record to the T2
   append frontier in key order and repoint its T1 entry using stripe-locked,
   offset-validated CAS. Relocations do not emit WAL records.
3. **Dirty-Segment Evacuation**: When dirty shards are absent, the worker scans
   for a segment below `base_boundary` that has unreferenced garbage bytes
   (`find_garbage_segment`) and evacuates its remaining live records to the
   append frontier, reducing the source segment's live-byte count to zero.

Work stops between shards or segments when `kCompactionMaxMoveBytesPerCycle` is
reached.

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

Checkpoint durabilizes T2's tail region and updates the T1 index snapshot and
manifest.

### Subdivided Synchronization

Checkpoint synchronizes `[old_base_boundary, target)` using `msync(MS_SYNC)`
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

- `src/vmemkv/impl/compaction.hpp`: dirty-shard repack, segment evacuation, and
  punch passes.
- `src/vmemkv/impl/t2_ownership.hpp`: segment live-byte counters and punch
  candidate epoch gating.
- `src/vmemkv/impl/checkpoint_coordinator.hpp`: chunked msync, manifest
  publication, and reorg trigger pacing.
- `src/vmemkv/vmemkv_impl.hpp`: compaction and reorganize background worker loops.
