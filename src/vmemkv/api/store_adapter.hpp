// store_adapter.hpp -- KVStore high-level facade / adapter template.
#pragma once

#include <algorithm>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <vmemkv/api/serializer.hpp>
#include <vmemkv/config.hpp>

namespace vmemkv {

// Codec constant parity: public header and Config must agree.
static_assert(vmemkv::detail::kBitsPerByte == vmemkv::Config<>::kBitsPerByte);

namespace detail {
template <typename T, typename = void>
struct get_config_type {
  using type = vmemkv::Config<>;
};

template <typename T>
struct get_config_type<T, std::void_t<typename T::ConfigType>> {
  using type = typename T::ConfigType;
};

template <typename T, typename = void>
struct is_rival_store : std::false_type {};

template <typename T>
struct is_rival_store<T, std::void_t<decltype(T::kIsRival)>> : std::bool_constant<T::kIsRival> {};

// Rival (non-vmemkv) backends have no T1/T2 concept: they self-manage storage
// layout, so reorganize()/get_statistics() are no-ops/empty for all of them.
// New backends opt in via `static constexpr bool kIsRival = true`.
template <typename T>
inline constexpr bool is_rival_store_v = is_rival_store<T>::value;

// Name hook: rivals specialize store_name<T>() in src/rivals/.
template <typename T>
auto store_name() -> std::string {
  return T::name();
}
}  // namespace detail

template <typename KVSImpl>
class StoreAdapter {
 public:
  static constexpr bool kIsEnabled = KVSImpl::kIsEnabled;
  using ConfigType = typename detail::get_config_type<KVSImpl>::type;
  using Impl = KVSImpl;

  static auto name() -> std::string { return detail::store_name<KVSImpl>(); }

  template <typename... Args>
  explicit StoreAdapter(Args &&...args) : impl_(std::forward<Args>(args)...) {}

  ~StoreAdapter() noexcept = default;

  StoreAdapter(const StoreAdapter &) = delete;
  auto operator=(const StoreAdapter &) -> StoreAdapter & = delete;
  StoreAdapter(StoreAdapter &&) = delete;
  auto operator=(StoreAdapter &&) -> StoreAdapter & = delete;

  template <typename Key, typename Callback>
  auto get(const Key &key, Callback callback) const -> bool {
    return kvs_detail::with_key_serialized(key, [this, &callback](std::span<const std::byte> key_bytes) -> bool {
      return impl_.get_impl(key_bytes, std::move(callback));
    });
  }

  template <auto ImplMethod, typename Key, typename Value>
  auto insert_or_update(const Key &key, Value &&value) -> bool {
    return kvs_detail::with_key_serialized(key, [this, &value](std::span<const std::byte> key_bytes) -> bool {
      return kvs_detail::with_val_serialized(std::forward<Value>(value),
                                             [this, key_bytes](std::span<const std::byte> val_bytes) -> bool {
                                               return (impl_.*ImplMethod)(key_bytes, val_bytes);
                                             });
    });
  }

  template <typename Key, typename Value>
  auto insert(const Key &key, Value &&value) -> bool {
    return insert_or_update<&KVSImpl::insert_impl>(key, std::forward<Value>(value));
  }

  template <typename Key, typename Value>
  auto update(const Key &key, Value &&value) -> bool {
    return insert_or_update<&KVSImpl::update_impl>(key, std::forward<Value>(value));
  }

  template <typename Key>
  auto remove(const Key &key) -> bool {
    return kvs_detail::with_key_serialized(
        key, [this](std::span<const std::byte> key_bytes) -> bool { return impl_.remove_impl(key_bytes); });
  }

  // Upsert semantics, no durability guarantee, no concurrent access during the call.
  template <typename KeyFn, typename ValueFn>
  void bulk_load(std::size_t count, KeyFn &&make_key, ValueFn &&make_value) {
    impl_.bulk_load_impl(count, std::forward<KeyFn>(make_key), std::forward<ValueFn>(make_value));
  }

  template <typename LoKey, typename HiKey, typename Callback>
  [[nodiscard]] auto scan(const LoKey &lower_bound, const HiKey &upper_bound, Callback callback) const -> size_t {
    return kvs_detail::with_key_serialized(
        lower_bound, [this, &upper_bound, &callback](std::span<const std::byte> lower_bound_bytes) -> size_t {
          return kvs_detail::with_key_serialized(
              upper_bound,
              [this, lower_bound_bytes, &callback](std::span<const std::byte> upper_bound_bytes) -> size_t {
                return impl_.scan_impl(lower_bound_bytes, upper_bound_bytes, std::move(callback));
              });
        });
  }

  auto impl() noexcept -> KVSImpl & { return impl_; }
  [[nodiscard]] auto impl() const noexcept -> const KVSImpl & { return impl_; }

  // T1-only in-memory merge; rival backends define their own no-op.
  void reorganize() { impl_.reorganize(); }

  // Cheapest available checkpoint path; no-op for rivals.
  void checkpoint() {
    if constexpr (detail::is_rival_store_v<KVSImpl>) {
      // no-op
    } else {
      impl_.checkpoint();
    }
  }

  // Drains background compaction until clean; no-op for rivals.
  void drain_compaction() {
    if constexpr (detail::is_rival_store_v<KVSImpl>) {
      // no-op
    } else {
      impl_.drain_compaction();
    }
  }

  [[nodiscard]] auto get_statistics() const noexcept -> ::vmemkv::VMemKVStatistics {
    if constexpr (detail::is_rival_store_v<KVSImpl>) {
      return ::vmemkv::VMemKVStatistics{};
    } else {
      return impl_.get_statistics();
    }
  }

  auto t2() -> auto & { return impl_.t2(); }
  [[nodiscard]] auto t2() const -> const auto & { return impl_.t2(); }

 private:
  KVSImpl impl_;
};

}  // namespace vmemkv
