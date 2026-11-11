// serializer.hpp -- Common serialization and encoding utilities.
// Canonical codec twins with include/vmemkv/vmemkv.hpp detail:: -- keep in sync
#pragma once

#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace kvs_detail {

inline constexpr std::size_t kEncodedValueBytes = 8;

// Encode integral keys as ordered big-endian bytes using CPU-intrinsic byte swap.
// Sign-flips signed values so lexicographic order matches numeric order, then byteswaps
// to big-endian on little-endian hosts. Twin of vmemkv::detail::encode_key in
// include/vmemkv/vmemkv.hpp; keep in sync.
template <typename Int>
inline auto encode_integral_key(Int value) noexcept -> std::array<std::byte, sizeof(Int)> {
  using BareInt = std::remove_cv_t<Int>;
  static_assert(std::is_integral_v<BareInt>);
  static_assert(sizeof(BareInt) <= sizeof(uint64_t));

  using Unsigned = std::make_unsigned_t<BareInt>;
  auto encoded = static_cast<Unsigned>(value);
  if constexpr (std::is_signed_v<BareInt>) {
    constexpr Unsigned sign_bit = Unsigned{1} << (sizeof(BareInt) * 8 - 1);
    encoded ^= sign_bit;
  }

  if constexpr (std::endian::native == std::endian::little) {
    encoded = std::byteswap(encoded);
  }

  std::array<std::byte, sizeof(BareInt)> out{};
  std::memcpy(out.data(), &encoded, sizeof(BareInt));
  return out;
}

// Raw bytes need no byte-order handling, so Key and Value share this base; only their
// integral specializations differ (sign-flipped big-endian keys, little-endian values).
template <typename T>
inline constexpr bool is_byte_container_v = std::is_same_v<std::decay_t<T>, std::span<const std::byte>> ||
                                            std::is_same_v<std::decay_t<T>, std::vector<std::byte>>;

template <typename T>
inline constexpr bool is_string_like_v = std::is_convertible_v<std::decay_t<T>, std::string_view>;

template <typename T>
inline constexpr bool is_raw_bytes_like_v = is_byte_container_v<T> || is_string_like_v<T>;

template <typename T>
inline auto serialize_byte_container(const T& value) noexcept -> std::span<const std::byte> {
  return std::as_bytes(std::span(value));
}

template <typename T>
inline auto serialize_string_like(const T& value) noexcept -> std::span<const std::byte> {
  std::string_view view(value);
  return {reinterpret_cast<const std::byte*>(view.data()), view.size()};
}

// Common base for the raw-bytes specializations of KeySerializer and ValueSerializer.
template <typename T>
struct RawBytesSerializer {
  static auto serialize(const T& value) noexcept -> std::span<const std::byte> {
    if constexpr (is_byte_container_v<T>) {
      return serialize_byte_container(value);
    } else {
      return serialize_string_like(value);
    }
  }
};

// KeySerializer (Big Endian for lexicographical ordering)
// Integral keys use sign-flipped big-endian encoding above; raw bytes pass through.
template <typename T, typename Enable = void>
struct KeySerializer {
  static auto serialize(const T& value) {
    return serialize_kvs(value);  // ADL
  }
};

template <typename T>
struct KeySerializer<T, std::enable_if_t<is_raw_bytes_like_v<T>>> : RawBytesSerializer<T> {};

template <typename T>
struct KeySerializer<T, std::enable_if_t<std::is_integral_v<std::decay_t<T>>>> {
  static auto serialize(T val) noexcept { return encode_integral_key(val); }
};

// ValueSerializer (Little Endian for raw data compatibility, no sign bit flipping)
// Integral values store as 8-byte little-endian opaque bytes; raw bytes pass through.
template <typename T, typename Enable = void>
struct ValueSerializer {
  static auto serialize(const T& value) {
    return serialize_kvs(value);  // ADL
  }
};

template <typename T>
struct ValueSerializer<T, std::enable_if_t<is_raw_bytes_like_v<T>>> : RawBytesSerializer<T> {};

template <typename T>
struct ValueSerializer<T, std::enable_if_t<std::is_integral_v<std::decay_t<T>>>> {
  static auto serialize(T value) noexcept {
    auto encoded = static_cast<uint64_t>(value);
    if constexpr (std::endian::native == std::endian::big) {
      encoded = std::byteswap(encoded);
    }
    std::array<std::byte, kEncodedValueBytes> out{};
    std::memcpy(out.data(), &encoded, sizeof(encoded));
    return out;
  }
};

// Invokes the callback with the serialized bytes, extending temporaries' lifetime.
template <typename Serialized, typename Callback>
inline auto invoke_with_span(Serialized&& serialized, Callback&& callback) -> decltype(auto) {
  if constexpr (std::same_as<std::decay_t<Serialized>, std::span<const std::byte>>) {
    return std::forward<Callback>(callback)(serialized);
  } else {
    return std::forward<Callback>(callback)(std::span<const std::byte>(serialized.data(), serialized.size()));
  }
}

template <typename T, typename Callback>
auto with_key_serialized(T&& value, Callback&& callback) -> decltype(auto) {
  auto serialized = KeySerializer<std::decay_t<T>>::serialize(std::forward<T>(value));
  return invoke_with_span(serialized, std::forward<Callback>(callback));
}

template <typename T, typename Callback>
auto with_val_serialized(T&& value, Callback&& callback) -> decltype(auto) {
  auto serialized = ValueSerializer<std::decay_t<T>>::serialize(std::forward<T>(value));
  return invoke_with_span(serialized, std::forward<Callback>(callback));
}

}  // namespace kvs_detail

// Consistency pin: the internal codec above twins include/vmemkv/vmemkv.hpp
// detail::encode_key/to_span. Additions here must update both sides together.
#include <vmemkv/vmemkv.hpp>

static_assert(kvs_detail::kEncodedValueBytes == vmemkv::detail::kEncodedValueBytes);

namespace kvs_detail {
// Runtime twin check (called by tests): key/value codecs must agree byte-for-byte.
inline void check_public_internal_codec_parity() {
  const std::array<int32_t, 5> samples = {0, -1, 1, INT32_MIN, INT32_MAX};
  for (int32_t sample : samples) {
    if (encode_integral_key(sample) != vmemkv::detail::encode_key(sample)) {
      throw std::logic_error("key codec parity broken");
    }
  }
}
}  // namespace kvs_detail
