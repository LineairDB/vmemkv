// rival_store_disabled_stub.hpp — Shared "backend not compiled in" stub surface for
// rivals/*.hpp store classes.
//
// Every rival store class exposes the same shape regardless of whether its backend was actually
// compiled in (kIsEnabled=false): a single-arg constructor and a CloneFromMasterTag constructor
// that both throw, a no-op reorganize(), and impl methods that report "not found"/empty rather
// than touching any backend state. This macro expands to that whole surface so it is written
// once instead of once per rival.
#pragma once

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

#include "rival_common.hpp"

namespace vmemkv::rivals {}  // namespace vmemkv::rivals

// clang-format off
// EngineLabel accepts any string expression (literals and Policy::kLabel alike).
#define VMEMKV_RIVAL_DISABLED_STUB(ClassName, EngineLabel)                                            \
  explicit ClassName(const std::string &unused_path) {                                                \
    (void)unused_path;                                                                                 \
    throw std::runtime_error(std::string(EngineLabel) + " not enabled in this build");                  \
  }                                                                                                       \
  ~ClassName() = default;                                                                                  \
  ClassName(const ClassName &) = delete;                                                                    \
  auto operator=(const ClassName &)->ClassName & = delete;                                                   \
  using CloneFromMasterTag = ::vmemkv::rivals::CloneFromMasterTag;                                           \
  template <typename KeyFn, typename ValueFn>                                                                   \
  ClassName(CloneFromMasterTag /*tag*/,                                                                          \
           const std::string &master_path,                                                                       \
           std::size_t key_count,                                                                                 \
           KeyFn &&make_key,                                                                                       \
           ValueFn &&make_value) {                                                                                  \
    (void)master_path;                                                                                               \
    (void)key_count;                                                                                                   \
    (void)make_key;                                                                                                     \
    (void)make_value;                                                                                                    \
    throw std::runtime_error(std::string(EngineLabel) + " not enabled in this build");                                                   \
  }                                                                                                                         \
  void reorganize() {}                                                                                                       \
  template <typename Callback>                                                                                                \
  auto get_impl(std::span<const std::byte> key, Callback callback) const->bool {                                              \
    (void)key;                                                                                                                  \
    (void)callback;                                                                                                              \
    return false;                                                                                                                 \
  }                                                                                                                                 \
  auto insert_impl(std::span<const std::byte> key, std::span<const std::byte> value)->bool {                                        \
    (void)key;                                                                                                                        \
    (void)value;                                                                                                                       \
    return false;                                                                                                                       \
  }                                                                                                                                       \
  auto update_impl(std::span<const std::byte> key, std::span<const std::byte> value)->bool {                                              \
    (void)key;                                                                                                                              \
    (void)value;                                                                                                                             \
    return false;                                                                                                                             \
  }                                                                                                                                             \
  auto remove_impl(std::span<const std::byte> key)->bool {                                                                                       \
    (void)key;                                                                                                                                     \
    return false;                                                                                                                                    \
  }                                                                                                                                                     \
  template <typename KeyFn, typename ValueFn>                                                                                                           \
  void bulk_load_impl(std::size_t key_count, KeyFn &&make_key, ValueFn &&make_value) {                                                                    \
    (void)key_count;                                                                                                                                        \
    (void)make_key;                                                                                                                                          \
    (void)make_value;                                                                                                                                         \
    throw std::runtime_error(std::string(EngineLabel) + " not enabled in this build");                                                                                        \
  }                                                                                                                                                               \
  template <typename Cb>                                                                                                                                           \
  auto scan_impl(std::span<const std::byte> low, std::span<const std::byte> high, Cb callback) const->size_t {                                                       \
    (void)low;                                                                                                                                                         \
    (void)high;                                                                                                                                                         \
    (void)callback;                                                                                                                                                      \
    return 0;                                                                                                                                                             \
  }
// clang-format on
