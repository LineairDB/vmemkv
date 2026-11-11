// recovery.hpp - Recovery helpers for VMemKVImpl construction.
// See docs/specification/low_level_design.md 5.3/5.4.
#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <vmemkv/checkpoint/checkpoint.hpp>
#include <vmemkv/checkpoint/paths.hpp>

namespace vmemkv::recovery_detail {

// Manifest-named live T2 bytes for T2FlatFile's adopt-constructor. Only the manifest is
// consulted; T2's own construction validates the data file against this value.
inline auto adopted_t2_bytes_used(const std::filesystem::path &t2_path) -> std::optional<uint64_t> {
  const auto manifest = vmemkv::read_manifest(vmemkv::derive_manifest_path(t2_path));
  if (!manifest.has_value()) {
    return std::nullopt;
  }
  return manifest->t2_bytes_used;
}

}  // namespace vmemkv::recovery_detail
