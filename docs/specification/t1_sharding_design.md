#T1 Range - Sharding Design

Eliminates the O(corpus) cost of `reorganize()`/`checkpoint()` (see [`t1_index_structure_decision.md`](t1_index_structure_decision.md)) by splitting a single `T1Index` into K independent shards. Each shard uses `src/vmemkv/t1_index/t1_index.hpp` with sharding hooks (freeze/bypass, offset-mapped reorganize, sorted-region dump/load).

## Overall Structure

T1 becomes two layers:

- **Directory**: an array sorted by boundary keys, mapping each interval to one shard.
- **Shard**: K instances of `T1Index<Config>`. Each shard holds only data in its own key range, an independent sorted region + append region pair.

The key space is range-partitioned (not hashed): shards stay disjoint and ordered at all times, so cross-shard Scan needs no k-way merge — concatenating each intersecting shard's `scan()` results in shard order already yields a fully sorted result.

The initial state is K=1 (identical behavior to the existing single T1Index), splitting dynamically as data grows.

## Directory Concurrency

The directory applies one level up the RCU-style pattern T1Index itself already uses (`AppendGeneration`/`SortedRegion` atomic pointer swaps + epoch-based reclamation). The directory is an immutable object, atomically swapped for a new instance only on split/merge. Readers load the directory once at operation start and keep using that shard reference until the operation completes.

## Splitting

`reorganize()` already computes a merged, deduplicated sorted array (`merged`) of all live entries. Split reuses this result, but using `merged` as-is loses writes, hence a 2-phase protocol.

**Hazard**: `reorganize()` freezes the old active region to immutable and publishes a new active region (the existing double buffering) *before* computing `merged`. By the time `merged` is finalized, the target shard has started accepting writes to the new (empty) active region. Entries written there are not in `merged`, so splitting `merged` in two as S1/S2 would drop writes that landed on that shard after the split decision.

**2-phase protocol (Closing → Split)**:

1. A normal `reorganize()` completes;
if the resulting (`merged`) size exceeds the target shard size threshold (e.g. 2x), the shard qualifies for split.
2. Set the target shard's `ShardSlot.superseded` to `Closing` (S1/S2 don't exist yet; merely a mark of "accept no further new writes"). put/remove check `superseded` and spin-wait on `Closing` without writing (get/scan may keep reading the target shard directly — data is intact, only writes stop).
3. Call the target shard's `reorganize()` once more. It folds writes that were in flight on the target shard when `Closing` was set into `merged`. New writes arriving after `Closing` are stopped by (2), so this second `merged` is that shard's complete final state.
4. Split this final `merged` in the middle, building new shards from each half (empty append region + split sorted region).
5. Update `superseded` from `Closing` to `Split{
  S1, S2, branch key
}
` (puts / removes spin - waiting on `Closing` wake here and redirect to S1 or S2).6. Replace the directory with
    new entries(boundaries to S1, S2) .7. Free the shard only after epoch
    - based reclamation confirms every operation that could directly reference the target shard via the old directory
          has completed.

          Cost is amortized O(1) /
          insert; worst-case one-shot cost is O(shard size), independent of total corpus size (the second `reorganize()` is also O(shard size), adding only a constant factor). Even with monotonically increasing keys like `bench_helpers.hpp`'s `ikey()` (`"k"` + 15 hex digits), only the rightmost shard keeps growing, so this property holds.

## APPEND_CAP Handling

Each shard's `APPEND_CAP` is fixed and bears no proportional relation to the shard target size. New shards inherit the parent shard's `append_capacity()`. The default is `Config::kAppendCapacityEntries` (16K entries); the soft maintenance threshold (half of it) equals the 8K-entry split threshold of the default shard target (`kShardTargetSizeEntries` = 4K entries), so a shard spans about one 8 MiB T2 segment of 1KB values.

Split decisions use live entry counts of `kShardTargetSizeEntries` × `kShardSplitThresholdPercent`, independent of `APPEND_CAP`. Tuning knobs remain shard target size, split threshold, and `APPEND_CAP` individually — they are not aggregated into a single "shard target size".

Splits all run independently and concurrently per shard via the mechanism in the "background worker thread pool" section. Only directory updates serialize through CAS on the shared object.

## Cross-Shard Concurrency Algorithms

### Forwarding Pointer (B-link Tree Style)

Each shard holds a one-word atomic `superseded` field (initially null). States are `null` (normal), `Closing` (splitting, rejecting new writes, S1/S2 undetermined), and `Split{S1, S2, branch key}` (split done, redirect target fixed), transitioning one way `null → Closing → Split{...}`. See the "Splitting" section for the concrete procedure.

Every operation (get/put/remove/scan) checks `superseded` before entering a shard:
on `Split { ... }` compare keys, forward to S1 or S2, and retry. On `Closing` only put/remove spin-wait (get/scan may keep reading S directly — within the per-key freshness scope of the "Scan consistency model scope" section), waiting for `Closing` to transition to `Split{ ... }` before forwarding.

### The `null → Closing` CAS Runs Guarded or Pinned

The `null → Closing` CAS starting a split is itself a dereference of the target `ShardSlot`, so it must run inside the routing epoch guard (like get / put / scan) or with the target pinned by `outside_refs` (which blocks reclamation the same way); only after a successful CAS may it leave the guard for the remaining work (second reorganize onward, equivalent to `continue_split`). Between resolving the target (observing `superseded == null`) and attempting the CAS unprotected, another thread's split of the same shard may have completed and the target may already be freed. After a successful CAS only the CAS winner can free that shard, so the remaining heavy work (second reorganize, S1/S2 construction, directory update, epoch drain, freeing) runs guard-free — holding the guard would turn `wait_until_epoch`'s self epoch drain into waiting on its own guard release, i.e. deadlock.

This split — "only the CAS attempt under guard, heavy post-success work outside it" — is an invariant every split-capable path (explicit calls and background-worker auto-fire alike) honors. Likewise, read-only methods that merely dereference the `Directory` (e.g. accessors returning the shard count) read `directory_` inside the routing epoch guard, like get/put/scan.

### Get/Scan Reaching a Splitting Shard

S's memory is not freed until the "Splitting" step 7 (epoch reclamation) completes, so direct reads don't corrupt. Read operations also follow forwarding pointers, so they don't miss latest data written to S1/S2 post-split.

### Scan Consistency Model Scope

What T1 (including the sharding layer) guarantees for Scan is only these two:

- **Structural completeness**: intersecting key ranges return exactly once each, with no leaks and no duplicates, even with concurrent split/merge (SMO) running.
- **Per-key freshness**: each returned key's value actually held at some point between the Scan's start and end.

Deliberately out of scope (not guaranteed) is **real-time ordering across writes to different keys**. For example, when two unrelated clients run `put(A)` → `put(B)` sequentially in real time, Scan returning "B new, A old" is acceptable.

These two (structural completeness + per-key freshness) hold on the forwarding-pointer mechanism alone: directly reading old shard S mid-split (uncorrected) still sees S holding all freeze-time data intact until discard. Following forwarding pointers to read the S1/S2 side additionally reflects post-freeze S1/S2 writes within per-key freshness. Either reading keeps structural completeness intact. **No global epoch/sequence counter is needed, adding zero cost.**

### Serializing Background Maintenance Operations

Each shard independently holds the equivalent of the existing T1Index's own `reorg_in_progress_` CAS guard; multiple shards' `reorganize()`/split run concurrently. Only shared-directory-object updates need serialization, resolved by CAS on the shared directory pointer (concurrent split declarations from multiple shards retry-loop: reread the latest directory and reapply on CAS failure). The concurrency to reason about is "one shard's background maintenance vs foreground read/write to that shard"; "shard A/B background maintenance vs each other" are mutually independent and never race.

## Background Worker Thread Pool

### Model

No dedicated thread per shard (OS threads would explode at large K). Instead, a fixed-size shared worker thread pool plus a queue (set, no duplicate insertion) holding "shard IDs needing reorganize". Every merge completion site re-drives this queue for shards still above the split threshold: put paths on soft-threshold crossing, dirty-shard visits and checkpoints after capturing a big shard. Splits themselves run only on the pool workers; the other sites merely enqueue. Each pool worker takes a shard ID and runs that shard's `reorganize()` (including split decisions). The hard threshold (a shard's append region actually full) stays scoped per shard: only threads attempting writes to that shard block and wait. Other shards' writes are unaffected. Having the accessing thread itself perform reorganize is not adopted.

The queue is a set, so its depth caps at K (total shards) and never grows unbounded. FIFO + dedup means a processed shard immediately re-crossing the threshold is simply re-enqueued at the tail — no starvation of specific shards.

### Avoiding Nested Parallelism

Shard `reorganize()` uses sequential sort (`std::execution::seq`). Parallelism comes from many workers processing many small reorganizes concurrently.

### Pool Size

No configuration knob;
the default is `max(1, hardware_concurrency / 4)`.

### Degradation When the Queue Falls Behind

Under sustained write load exceeding pool throughput, multiple shards reach the hard threshold together and only writes to those shards block. Total throughput caps at pool capacity, but the impact stays on those shards — other shards keep serving writes, unlike a single unsharded T1Index where the whole store stalls at hard threshold. The exception is workloads like monotonically increasing keys where only one shard keeps updating: only one shard parallelizes at all, so growing the pool changes nothing. This limit shares its cause with the same limit in the split cost analysis in the "Splitting" section.

## Checkpoint / Recovery

T2 and WAL are global (shard-independent). `checkpoint()` forces `reorganize()`s (including split decisions) of each of the K shards, serializing boundary keys plus each shard's sorted entries into the checkpoint file. This repeats K times with "T1 checkpoint before WAL rotation" ordering. Recovery rebuilds directory + K shards from this file, then replays the WAL remainder with directory-based routing. Forwarding pointers protect during checkpoint, so no split-specific new hazards arise.

## Related Documents

- [`t1_index_structure_decision.md`](t1_index_structure_decision.md): the decision to keep a flat array per shard, and the range-sharding definition.
