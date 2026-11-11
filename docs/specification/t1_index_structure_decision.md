# T1 Index Structure

T1 is the flat array in `src/vmemkv/t1_index/t1_index.hpp` (sorted region plus append region, merged by `reorganize()`), range-sharded by `src/vmemkv/t1_index/sharded_t1_index.hpp`. See `docs/specification/low_level_design.md` 2.1 and `docs/specification/t1_sharding_design.md`.
