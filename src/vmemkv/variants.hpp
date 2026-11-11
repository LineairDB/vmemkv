// Default and read-policy configurations for tests and benchmarks.
#pragma once
#include <vmemkv/api/store_adapter.hpp>
#include <vmemkv/config.hpp>
#include <vmemkv/vmemkv_impl.hpp>
namespace vmemkv::variants {
using VMemKV_Default = StoreAdapter<VMemKVImpl<>>;
using VMemKV_ReadRandom = StoreAdapter<VMemKVImpl<Config<ReadPolicyRandomOnly>>>;
using VMemKV_ReadSeq = StoreAdapter<VMemKVImpl<Config<ReadPolicySeqOnly>>>;
using VMemKV_ReadNormal = StoreAdapter<VMemKVImpl<Config<ReadPolicyNormalOnly>>>;
using VMemKV_ReadPread = StoreAdapter<VMemKVImpl<Config<ReadPolicyPreadOnly>>>;
}  // namespace vmemkv::variants
