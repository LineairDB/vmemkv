---
name: Feature request
about: Suggest an idea for vmemkv
labels: enhancement
---

## Problem

What can vmemkv not do today, and who needs it?

## Proposal

What should change (API, behavior, scope)?

## Notes

- vmemkv is Linux-only (`mmap` / `fork` / `mincore` based) and
  single-operation transactions are out of scope by design.
- Performance claims need measurements; include numbers if relevant.
