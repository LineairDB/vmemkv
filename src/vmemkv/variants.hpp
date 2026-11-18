// Default and read-policy configurations for tests and benchmarks.
#pragma once
#include <cstdint>
#include <filesystem>
#include <string>
#include <vmemkv/api/store_adapter.hpp>
#include <vmemkv/config.hpp>
#include <vmemkv/vmemkv_impl.hpp>

namespace vmemkv::variants {

template <ReadPolicy Policy>
class ReadPolicyVariant : public VMemKVImpl {
 public:
  static auto name() -> std::string { return VMemKVImpl::name(Policy); }

  ReadPolicyVariant(const std::filesystem::path &t2_path, uint64_t t2_bytes_capacity)
      : VMemKVImpl(t2_path, t2_bytes_capacity, StoreOptions{.read_policy = Policy}) {}
};

using VMemKV_Default = StoreAdapter<VMemKVImpl>;
using VMemKV_ReadRandom = StoreAdapter<ReadPolicyVariant<ReadPolicy::RandomOnly>>;
using VMemKV_ReadSeq = StoreAdapter<ReadPolicyVariant<ReadPolicy::SeqOnly>>;
using VMemKV_ReadNormal = StoreAdapter<ReadPolicyVariant<ReadPolicy::NormalOnly>>;
using VMemKV_ReadPread = StoreAdapter<ReadPolicyVariant<ReadPolicy::PreadOnly>>;

}  // namespace vmemkv::variants
