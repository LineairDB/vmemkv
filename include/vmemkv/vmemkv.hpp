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
// KVStore: the uniform interface every backend in this repository provides. Tests and
// benchmarks are written generically against this contract, so all backends behave alike.
//
// Data model: keys are ordered byte strings, values are opaque bytes. Both sides accept
// string-like arguments (anything convertible to std::string_view, including literals),
// integral arguments (keys ordered numerically via big-endian encoding, values stored as
// 8-byte little-endian), raw byte spans/vectors, and user types with an ADL serialize_kvs()
// overload. A missing key reports false (get/remove/update) or an empty scan, never a
// sentinel value.
//
// Callbacks observe borrowed spans: get()'s value and scan()'s key/value live only for the
// duration of the callback. Copy the bytes out if they must outlive the call; a later
// get()/scan() on the same thread may reuse the buffer.
//
// Operations: insert() fails (false) when the key exists; update()/remove() fail when it is
// absent. scan() invokes its callback in ascending key order over the half-open [lower,
// upper) range and returns the number of live entries visited. A callback may return
// ScanControl::Continue to keep scanning or ScanControl::Stop to end after the current entry;
// void callbacks continue. bulk_load() ingests generated entries fast: no per-write durability
// and no concurrent access while it runs, so follow it with checkpoint() before relying on the
// data.
//
// Maintenance: reorganize() merges only the in-memory index (no I/O); checkpoint() makes the
// log durable and compacts it into a manifest.
// get_statistics() snapshots the counters described in <vmemkv/stats.hpp>.
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
// The remainder of this header is machinery for the templates above; callers implementing or
// using KVStore need not read further.
namespace detail {
inline constexpr std::size_t kBitsPerByte = 8, kEncodedValueBytes = 8;
template <typename T>
inline constexpr bool is_bytes_v = std::is_same_v<std::decay_t<T>, std::span<const std::byte>> ||
                                   std::is_same_v<std::decay_t<T>, std::vector<std::byte>>;
template <typename T>
inline constexpr bool is_str_v = std::convertible_to<std::decay_t<T>, std::string_view>;
template <typename T>
inline constexpr bool is_raw_v = is_bytes_v<T> || is_str_v<T>;
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
template <typename T>
inline auto to_span(const T &value) noexcept -> std::span<const std::byte> {
  if constexpr (is_bytes_v<T>) {
    return std::as_bytes(std::span(value));
  } else {
    return {reinterpret_cast<const std::byte *>(std::string_view(value).data()), std::string_view(value).size()};
  }
}
}  // namespace detail
// VMemKVStore: durable ordered KV store on a T2 file plus WAL/checkpoint files.
// Data model is ordered byte-string keys with opaque byte-string values (2.1, 2.2).
// Integral keys encode as sign-flipped big-endian (ordered); integral values as 8-byte
// little-endian (opaque). Custom types use ADL:
// e.g. std::span<const std::byte> serialize_kvs(const MyType&);
// See docs/specification/low_level_design.md 2.1, 2.2, 3.2, 4.3.
class VMemKVStore {
 public:
  // Opens path (T2 file to create or reopen); capacity is T2 bytes fixed at creation.
  // Throws std::system_error on I/O failure.
  VMemKVStore(const std::filesystem::path &path, uint64_t capacity);
  ~VMemKVStore();
  VMemKVStore(VMemKVStore &&) noexcept;
  VMemKVStore(const VMemKVStore &) = delete;
  auto operator=(const VMemKVStore &) -> VMemKVStore & = delete;
  auto operator=(VMemKVStore &&) noexcept -> VMemKVStore &;

  // Inserts key/value; false when the key exists. See docs/specification/low_level_design.md 3.2.
  template <typename K, typename V>
  [[nodiscard]] auto insert(const K &key, V &&value) -> bool {
    return insert_bytes(key_bytes(key), val_bytes(std::forward<V>(value)));
  }
  // Updates value in place or by append; false when the key is absent.
  // See docs/specification/low_level_design.md 3.3.
  template <typename K, typename V>
  [[nodiscard]] auto update(const K &key, V &&value) -> bool {
    return update_bytes(key_bytes(key), val_bytes(std::forward<V>(value)));
  }
  // Removes key via tombstone; false when absent. See docs/specification/low_level_design.md 3.4.
  template <typename K>
  [[nodiscard]] auto remove(const K &key) -> bool {
    return remove_bytes(key_bytes(key));
  }
  // Gets value via callback; false when absent. Span lives for the callback only.
  // See docs/specification/low_level_design.md 3.1.
  template <typename K, typename C>
  [[nodiscard]] auto get(const K &key, C callback) const -> bool {
    auto key_span = key_bytes(key);
    return get_bytes(key_span, &callback, [](void *context, std::span<const std::byte> value) {
      (*static_cast<C *>(context))(value);
    });
  }
  // Scans ascending over half-open [lower, upper); STOP includes the current entry in the live count.
  // See docs/specification/low_level_design.md 3.5.
  template <typename L, typename H, typename C>
  [[nodiscard]] auto scan(const L &lower, const H &upper, C callback) const -> size_t {
    auto lower_span = key_bytes(lower);
    auto upper_span = key_bytes(upper);
    return scan_bytes(lower_span,
                      upper_span,
                      &callback,
                      [](void *context, std::span<const std::byte> key, std::span<const std::byte> value) {
                        return scan_detail::invoke_callback(*static_cast<C *>(context), key, value);
                      });
  }
  // Ingests generated entries with no per-write durability; follow with checkpoint().
  // See docs/specification/low_level_design.md 3.6.
  template <typename KF, typename VF>
  void bulk_load(std::size_t count, KF &&make_key, VF &&make_value) {
    bulk_load_erased(
        count,
        [key_factory = std::forward<KF>(make_key)](std::size_t index) { return std::string(key_factory(index)); },
        [value_factory = std::forward<VF>(make_value)](std::size_t index) {
          return std::string(value_factory(index));
        });
  }

  // Merges only the in-memory index; no I/O. See docs/specification/low_level_design.md 4.2.
  void reorganize();
  // Makes the log durable and compacts it into a manifest. See docs/specification/low_level_design.md 4.3, 5.3.
  void checkpoint();
  // Snapshots the counters described in <vmemkv/stats.hpp>.
  [[nodiscard]] auto get_statistics() const noexcept -> VMemKVStatistics;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  template <typename T>
  static auto key_bytes(const T &value) {
    if constexpr (std::is_integral_v<std::decay_t<T>>) {
      return detail::encode_key(value);
    } else if constexpr (detail::is_raw_v<T>) {
      return detail::to_span(value);
    } else {
      return serialize_kvs(value);
    }
  }
  template <typename T>
  static auto val_bytes(const T &value) {
    if constexpr (std::is_integral_v<std::decay_t<T>>) {
      auto encoded = static_cast<uint64_t>(value);
      if constexpr (std::endian::native == std::endian::big) {
        encoded = std::byteswap(encoded);
      }
      std::array<std::byte, detail::kEncodedValueBytes> out{};
      std::memcpy(out.data(), &encoded, detail::kEncodedValueBytes);
      return out;
    } else if constexpr (detail::is_raw_v<T>) {
      return detail::to_span(value);
    } else {
      return serialize_kvs(value);
    }
  }

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
}  // namespace vmemkv
