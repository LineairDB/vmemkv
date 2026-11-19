// serializer.hpp -- Span-lifetime wrappers around vmemkv::detail's key/value codecs.
#pragma once

#include <concepts>
#include <cstddef>
#include <span>
#include <type_traits>
#include <utility>
#include <vmemkv/vmemkv.hpp>

namespace kvs_detail {

// Invokes the callback with the serialized bytes, keeping stack-owned buffers alive across the call.
template <typename Serialized, typename Callback>
inline auto invoke_with_span(Serialized &&serialized, Callback &&callback) -> decltype(auto) {
  if constexpr (std::same_as<std::decay_t<Serialized>, std::span<const std::byte>>) {
    return std::forward<Callback>(callback)(serialized);
  } else {
    return std::forward<Callback>(callback)(std::span<const std::byte>(serialized.data(), serialized.size()));
  }
}

template <typename T, typename Callback>
auto with_key_serialized(const T &value, Callback &&callback) -> decltype(auto) {
  auto serialized = vmemkv::detail::serialize_key(value);
  return invoke_with_span(serialized, std::forward<Callback>(callback));
}

template <typename T, typename Callback>
auto with_val_serialized(const T &value, Callback &&callback) -> decltype(auto) {
  auto serialized = vmemkv::detail::serialize_val(value);
  return invoke_with_span(serialized, std::forward<Callback>(callback));
}

}  // namespace kvs_detail
