// rival_common.hpp — Shared helpers for the rival (non-vmemkv) backend wrappers.
#pragma once

#include <algorithm>
#include <atomic>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vmemkv/core/bytes.hpp>

namespace vmemkv::rivals {

// Shared tag selecting the clone-from-master constructor.
struct CloneFromMasterTag {};

// Existence-gated mutation shared by rival backends (LMDB shape as model):
// runs mutate() only when exists() matches required, otherwise reports false.
template <typename ExistsFn, typename MutateFn>
inline auto require_exists(ExistsFn &&exists, bool required, MutateFn &&mutate) -> bool {
  if (std::forward<ExistsFn>(exists)() != required) {
    return false;
  }
  return std::forward<MutateFn>(mutate)();
}

// Unique per-instance path: appends "_<n><ext>" with a process-wide counter.
inline auto make_unique_instance_path(std::string base, std::string_view ext) -> std::string {
  static std::atomic<uint64_t> instance_counter{0};
  base += "_";
  base += std::to_string(instance_counter.fetch_add(1, std::memory_order_relaxed));
  base.append(ext.data(), ext.size());
  return base;
}

// Sibling path holding a master under construction.
inline auto building_path(const std::string &master) -> std::string { return master + ".building"; }

// Renames src onto dst, throwing on failure.
inline void atomic_rename(const std::string &src, const std::string &dst, const std::string &failure_prefix) {
  std::error_code ec;
  std::filesystem::rename(src, dst, ec);
  if (ec) {
    throw std::runtime_error(failure_prefix + ": " + ec.message());
  }
}

}  // namespace vmemkv::rivals
