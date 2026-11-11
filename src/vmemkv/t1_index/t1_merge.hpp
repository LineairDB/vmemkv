// t1_merge.hpp - T1 merge dedup helpers for T1Index::reorganize().
// See docs/specification/low_level_design.md 7.5.
#pragma once

#include <cstdint>
#include <vmemkv/t1_index/t1_codec.hpp>

namespace t1_detail {

// Logical entry identity used by merge dedup and its invariant check: key plus
// clean hash (raw hash embeds transient inline metadata that must not affect identity).
inline auto entry_identity_equal(const StoreKey &a_key,
                                 uint64_t a_hash,
                                 const StoreKey &b_key,
                                 uint64_t b_hash) noexcept -> bool {
  return a_key == b_key && (a_hash & kCleanHashMask) == (b_hash & kCleanHashMask);
}

inline auto entry_less(const StoreKey &a_key,
                       uint64_t a_hash,
                       const StoreKey &b_key,
                       uint64_t b_hash) noexcept -> bool {
  if (a_key != b_key) {
    return a_key < b_key;
  }
  return (a_hash & kCleanHashMask) < (b_hash & kCleanHashMask);
}

}  // namespace t1_detail
