// Default and read-policy configurations for tests and benchmarks.
#pragma once
#include <vmemkv/api/store_adapter.hpp>
#include <vmemkv/config.hpp>
#include <vmemkv/vmemkv_impl.hpp>
namespace vmemkv::variants {
using VMemKV_Default = StoreAdapter<VMemKVImpl>;
using VMemKV_ReadRandom = StoreAdapter<ConfiguredVMemKV<Config<ReadPolicyRandomOnly>>>;
using VMemKV_ReadSeq = StoreAdapter<ConfiguredVMemKV<Config<ReadPolicySeqOnly>>>;
using VMemKV_ReadNormal = StoreAdapter<ConfiguredVMemKV<Config<ReadPolicyNormalOnly>>>;
using VMemKV_ReadPread = StoreAdapter<ConfiguredVMemKV<Config<ReadPolicyPreadOnly>>>;
}  // namespace vmemkv::variants
