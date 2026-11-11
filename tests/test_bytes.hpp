// test_bytes.hpp — Byte/span/key/value helpers for tests.
#pragma once

#include <doctest/doctest.h>

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace vmemkv_test {

// to_span: borrow from std::string (no copy). bytes_of+as_span: owning copy for
// APIs holding the span past the call. Prefer to_span unless ownership is needed.
inline auto to_span(const std::string &text) -> std::span<const std::byte> {
  return std::span<const std::byte>(reinterpret_cast<const std::byte *>(text.data()), text.size());
}

inline auto bytes_of(std::string_view value) -> std::vector<std::byte> {
  std::vector<std::byte> out(value.size());
  std::memcpy(out.data(), value.data(), value.size());
  return out;
}

inline auto as_span(const std::vector<std::byte> &value) -> std::span<const std::byte> {
  return std::span<const std::byte>(value.data(), value.size());
}

inline auto span_to_string(std::span<const std::byte> value) -> std::string {
  return {reinterpret_cast<const char *>(value.data()), value.size()};
}

template <typename StorePtr, typename Key>
auto get_optional_bytes(const StorePtr &store, const Key &key) -> std::optional<std::vector<std::byte>> {
  std::optional<std::vector<std::byte>> result;
  const bool found =
      store->get(key, [&](std::span<const std::byte> val) { result = std::vector<std::byte>(val.begin(), val.end()); });
  if (found) {
    return result;
  }
  return std::nullopt;
}

// Fixed-width decimal key maker: prefix followed by the zero-padded index.
inline auto padded_key(std::string_view prefix, long long index, int width) -> std::string {
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.*s%0*lld", static_cast<int>(prefix.size()), prefix.data(), width, index);
  return {buf};
}

// Fixed-width decimal key maker with the default "k" prefix.
inline auto padded_key(long long index, int width) -> std::string { return padded_key("k", index, width); }

// Value maker used by crash-recovery tests: always >= 9 bytes, forcing every write through T2.
inline auto indexed_value(long long index) -> std::string { return "value_" + std::to_string(index) + "_pad"; }

template <typename Store>
void insert_range(Store &store, int key_count) {
  for (int i = 0; i < key_count; ++i) {
    REQUIRE(store.insert("k" + std::to_string(i), indexed_value(i)));
  }
}

template <typename Store>
void check_range(Store &store, int key_count) {
  for (int i = 0; i < key_count; ++i) {
    const auto bytes = get_optional_bytes(&store, "k" + std::to_string(i));
    REQUIRE(bytes.has_value());
    CHECK(span_to_string(as_span(*bytes)) == indexed_value(i));
  }
}

}  // namespace vmemkv_test
