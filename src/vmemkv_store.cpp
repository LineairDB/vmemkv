// Public facade over StoreAdapter<VMemKVImpl<...>>; no logic here.
#include <vmemkv/api/store_adapter.hpp>
#include <vmemkv/stats.hpp>
#include <vmemkv/vmemkv.hpp>
#include <vmemkv/vmemkv_impl.hpp>
namespace vmemkv {
struct VMemKVStore::Impl {
  StoreAdapter<VMemKVImpl<>> inner;
  Impl(const std::filesystem::path &path, uint64_t capacity) : inner(path, capacity) {}
};
VMemKVStore::VMemKVStore(const std::filesystem::path &path, uint64_t capacity)
    : impl_(std::make_unique<Impl>(path, capacity)) {}
VMemKVStore::~VMemKVStore() = default;
VMemKVStore::VMemKVStore(VMemKVStore &&) noexcept = default;
auto VMemKVStore::operator=(VMemKVStore &&) noexcept -> VMemKVStore & = default;
auto VMemKVStore::insert_bytes(std::span<const std::byte> key, std::span<const std::byte> value) -> bool {
  return impl_->inner.impl().insert_impl(key, value);
}
auto VMemKVStore::update_bytes(std::span<const std::byte> key, std::span<const std::byte> value) -> bool {
  return impl_->inner.impl().update_impl(key, value);
}
auto VMemKVStore::remove_bytes(std::span<const std::byte> key) -> bool { return impl_->inner.impl().remove_impl(key); }
auto VMemKVStore::get_bytes(std::span<const std::byte> key,
                            void *context,
                            void (*callback)(void *, std::span<const std::byte>)) const -> bool {
  return impl_->inner.impl().get_impl(
      key, [context, callback](std::span<const std::byte> value) { callback(context, value); });
}
auto VMemKVStore::scan_bytes(std::span<const std::byte> lower,
                             std::span<const std::byte> upper,
                             void *context,
                             ScanControl (*callback)(void *,
                                                     std::span<const std::byte>,
                                                     std::span<const std::byte>)) const -> size_t {
  return impl_->inner.impl().scan_impl(
      lower, upper, [context, callback](std::span<const std::byte> key, std::span<const std::byte> value) {
        return callback(context, key, value);
      });
}
void VMemKVStore::bulk_load_erased(std::size_t count,
                                   const std::function<std::string(std::size_t)> &make_key,
                                   const std::function<std::string(std::size_t)> &make_value) {
  impl_->inner.impl().bulk_load_impl(count, make_key, make_value);
}
void VMemKVStore::reorganize() { impl_->inner.reorganize(); }
void VMemKVStore::checkpoint() { impl_->inner.checkpoint(); }
auto VMemKVStore::get_statistics() const noexcept -> VMemKVStatistics { return impl_->inner.get_statistics(); }
}  // namespace vmemkv
