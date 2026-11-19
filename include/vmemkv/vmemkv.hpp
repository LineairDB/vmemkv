#pragma once
#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>
#include <vmemkv/scan_control.hpp>
#include <vmemkv/stats.hpp>

namespace vmemkv {

// Ordered byte-key / opaque-value KVS interface. Keys/values accept string-like types,
// integrals (sign-flipped big-endian keys, 8-byte little-endian values), byte spans/vectors,
// or ADL `serialize_kvs(const T&)`. Get/scan callback spans remain valid only for the call.
template <typename Store>
concept KVStore = requires(Store store, std::span<const std::byte> key, std::span<const std::byte> value) {
  { store.insert(key, value) } -> std::same_as<bool>;
  { store.update(key, value) } -> std::same_as<bool>;
  { store.remove(key) } -> std::same_as<bool>;
  store.get(key, [](std::span<const std::byte>) {});
  {
    store.scan(key, key, [](std::span<const std::byte>, std::span<const std::byte>) {})
  } -> std::same_as<std::size_t>;
  {
    store.scan(key, key, [](std::span<const std::byte>, std::span<const std::byte>) { return ScanControl::Continue; })
  } -> std::same_as<std::size_t>;
  store.bulk_load(std::size_t{0}, [](std::size_t) { return std::string{}; }, [](std::size_t) { return std::string{}; });
  { store.reorganize() } -> std::same_as<void>;
  { store.checkpoint() } -> std::same_as<void>;
  { store.get_statistics() } -> std::same_as<VMemKVStatistics>;
};

// Thread-safe durable ordered KV store backed by a T2 file, WAL, and T1 checkpoint files.
class VMemKVStore {
 public:
  VMemKVStore(const std::filesystem::path &path, uint64_t capacity);
  ~VMemKVStore();
  VMemKVStore(VMemKVStore &&) noexcept;
  VMemKVStore(const VMemKVStore &) = delete;
  auto operator=(const VMemKVStore &) -> VMemKVStore & = delete;
  auto operator=(VMemKVStore &&) noexcept -> VMemKVStore &;

  template <typename K, typename V>
  [[nodiscard]] auto insert(const K &key, const V &value) -> bool;
  template <typename K, typename V>
  [[nodiscard]] auto update(const K &key, const V &value) -> bool;
  template <typename K>
  [[nodiscard]] auto remove(const K &key) -> bool;
  template <typename K, typename C>
  [[nodiscard]] auto get(const K &key, C callback) const -> bool;
  // Visits ascending entries in [lower, upper); returns live entries visited.
  template <typename L, typename H, typename C>
  [[nodiscard]] auto scan(const L &lower, const H &upper, C callback) const -> size_t;
  // Fast non-durable ingestion with no concurrent callers; follow with checkpoint().
  template <typename KF, typename VF>
  void bulk_load(std::size_t count, KF &&make_key, VF &&make_value);

  void reorganize();
  void checkpoint();
  [[nodiscard]] auto get_statistics() const noexcept -> VMemKVStatistics;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  auto insert_bytes(std::span<const std::byte> key, std::span<const std::byte> value) -> bool;
  auto update_bytes(std::span<const std::byte> key, std::span<const std::byte> value) -> bool;
  auto remove_bytes(std::span<const std::byte> key) -> bool;
  auto get_bytes(std::span<const std::byte> key,
                 void *context,
                 void (*callback)(void *, std::span<const std::byte>)) const -> bool;
  auto scan_bytes(std::span<const std::byte> lower,
                  std::span<const std::byte> upper,
                  void *context,
                  ScanControl (*callback)(void *,
                                          std::span<const std::byte>,
                                          std::span<const std::byte>)) const -> size_t;
  void bulk_load_erased(std::size_t count,
                        const std::function<std::string(std::size_t)> &make_key,
                        const std::function<std::string(std::size_t)> &make_value);
};
static_assert(KVStore<VMemKVStore>);

namespace detail {
inline constexpr std::size_t kBitsPerByte = 8, kEncodedValueBytes = 8;
template <typename T>
inline constexpr bool is_bytes_v = std::is_same_v<std::decay_t<T>, std::span<const std::byte>> ||
                                   std::is_same_v<std::decay_t<T>, std::vector<std::byte>>;
template <typename T>
inline constexpr bool is_raw_v = is_bytes_v<T> || std::convertible_to<std::decay_t<T>, std::string_view>;

template <typename I>
inline auto encode_key(I value) noexcept -> std::array<std::byte, sizeof(I)> {
  using Unsigned = std::make_unsigned_t<std::remove_cv_t<I>>;
  static_assert(sizeof(I) <= sizeof(uint64_t));
  auto encoded = static_cast<Unsigned>(value);
  if constexpr (std::is_signed_v<I>) {
    encoded ^= Unsigned{1} << (sizeof(I) * kBitsPerByte - 1);
  }
  if constexpr (std::endian::native == std::endian::little) {
    encoded = std::byteswap(encoded);
  }
  std::array<std::byte, sizeof(I)> out{};
  std::memcpy(out.data(), &encoded, sizeof(I));
  return out;
}

template <typename I>
inline auto encode_val(I value) noexcept -> std::array<std::byte, kEncodedValueBytes> {
  auto encoded = static_cast<uint64_t>(value);
  if constexpr (std::endian::native == std::endian::big) {
    encoded = std::byteswap(encoded);
  }
  std::array<std::byte, kEncodedValueBytes> out{};
  std::memcpy(out.data(), &encoded, kEncodedValueBytes);
  return out;
}

template <typename T>
inline auto to_span(const T &value) noexcept -> std::span<const std::byte> {
  if constexpr (is_bytes_v<T>) {
    return std::as_bytes(std::span(value));
  } else {
    std::string_view view(value);
    return {reinterpret_cast<const std::byte *>(view.data()), view.size()};
  }
}

template <bool IsKey, typename T>
inline auto serialize(const T &value) {
  if constexpr (std::is_integral_v<std::decay_t<T>>) {
    if constexpr (IsKey) {
      return encode_key(value);
    } else {
      return encode_val(value);
    }
  } else if constexpr (is_raw_v<T>) {
    return to_span(value);
  } else {
    return serialize_kvs(value);  // ADL
  }
}

template <typename T>
inline auto serialize_key(const T &value) {
  return serialize<true>(value);
}

template <typename T>
inline auto serialize_val(const T &value) {
  return serialize<false>(value);
}
}  // namespace detail

template <typename K, typename V>
auto VMemKVStore::insert(const K &key, const V &value) -> bool {
  return insert_bytes(detail::serialize_key(key), detail::serialize_val(value));
}

template <typename K, typename V>
auto VMemKVStore::update(const K &key, const V &value) -> bool {
  return update_bytes(detail::serialize_key(key), detail::serialize_val(value));
}

template <typename K>
auto VMemKVStore::remove(const K &key) -> bool {
  return remove_bytes(detail::serialize_key(key));
}

template <typename K, typename C>
auto VMemKVStore::get(const K &key, C callback) const -> bool {
  auto key_span = detail::serialize_key(key);
  return get_bytes(key_span, &callback, [](void *context, std::span<const std::byte> value) {
    (*static_cast<C *>(context))(value);
  });
}

template <typename L, typename H, typename C>
auto VMemKVStore::scan(const L &lower, const H &upper, C callback) const -> size_t {
  auto lower_span = detail::serialize_key(lower);
  auto upper_span = detail::serialize_key(upper);
  return scan_bytes(lower_span,
                    upper_span,
                    &callback,
                    [](void *context, std::span<const std::byte> key, std::span<const std::byte> value) {
                      return scan_detail::invoke_callback(*static_cast<C *>(context), key, value);
                    });
}

template <typename KF, typename VF>
void VMemKVStore::bulk_load(std::size_t count, KF &&make_key, VF &&make_value) {
  bulk_load_erased(
      count,
      [key_factory = std::forward<KF>(make_key)](std::size_t index) { return std::string(key_factory(index)); },
      [value_factory = std::forward<VF>(make_value)](std::size_t index) { return std::string(value_factory(index)); });
}

}  // namespace vmemkv
