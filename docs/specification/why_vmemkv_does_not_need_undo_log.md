# Why vmemkv Does Not Need an Undo Log

vmemkv's live T2 mmap is `MAP_SHARED` (low_level_design.md 5.1). In classical buffer management terms this design tolerates **STEAL**: the OS page cache may write dirty pages back to disk at arbitrary times unrelated to application intent. Buffer management that tolerates STEAL generally requires an undo log (see below). This document shows why vmemkv needs none despite STEAL.

## 1. STEAL/FORCE Definitions and the Classical STEAL→UNDO Requirement

- **STEAL**: dirty pages carrying a transaction's changes may be written back to disk before the transaction commits.
- **FORCE**: a transaction's commit completes only after every page it modified is written back to disk.

vmemkv is STEAL (above) and **NO-FORCE**: each write (insert/update/delete) counts as committed once its WAL record is `fsync`ed and success returned to the caller (low_level_design.md 3.2). Whether the corresponding T2 physical pages have reached disk is not part of the commit condition.

The reason classical ARIES-style buffer management demands an undo log under STEAL is its **physical page-level redo**: recovery compares each page's LSN against log records' LSNs to decide whether the page already reflects a log's contents, reapplying physical diffs to pages as needed. In this scheme the on-disk pages' contents are themselves recovery inputs, so a page carrying an uncommitted transaction's changes that stands as an official page needs a separate means to take it back (undo).

## 2. vmemkv's Redo Is Logical (Operation-Level)

vmemkv recovery never takes T2's raw bytes as direct input. It trusts only bytes below the manifest's `t2_bytes_used` (the base region) and never parses anything physically present beyond it (the tail region), no matter what (low_level_design.md 5.3). The tail region's contents are **rebuilt from scratch at new offsets** by replaying the WAL's corresponding range through `write_entry_lockfree()` — the ordinary write path itself (low_level_design.md 4.4, 5.2).

In other words, vmemkv's redo is not "diffing physical bytes onto pages" but "re-executing the operations (insert/update/delete) themselves". Whatever remains at old offsets is never referenced by what replay reproduces. This property eliminates the very problem of "tail-region bytes stolen onto disk ahead of time lingering regardless of transaction outcome".

## 3. Why Each Torn-Write Pattern Is Harmless

**(a) A tail-region record stolen to disk in a torn state**
Its offset is not below the manifest boundary, so recovery never references it. If the corresponding WAL record is fsync'ed, it is replayed through the ordinary write path and correctly rebuilt at a new offset. With no WAL record (crashed before fsync), the operation is treated as "never happened" — consistent with an operation whose success was never returned to the caller.

**(b) A torn WAL record itself**
WAL replay detects and discards incomplete trailing records (via checksum / length validation). A torn WAL record is equivalent to "no record" and reduces to case (a)'s no-record situation.

**(c) A torn record in the base region (below the manifest boundary)**
This case cannot structurally occur. The boundary only advances after `checkpoint_internal()` briefly stops new appends (`stop_writers_and_wait()`), targets only fully-written records, and `msync()` succeeds (low_level_design.md 4.3). So the published range leaves no room for half-written records.

## 4. Conclusion

vmemkv is STEAL and NO-FORCE but needs no undo log, because the tail region's raw T2 bytes carry no authority whatsoever and are always rebuilt at new offsets by logical redo (operation re-execution) from the WAL. The base region, by design, advances its boundary only after completeness is guaranteed pre-publication, so it is never published torn. Unlike classical designs with physical page-level redo, the STEAL–UNDO coupling does not hold.
