#include <filesystem>
#include <iostream>
#include <span>
#include <string>
#include <vmemkv/vmemkv.hpp>

int main() {
  static_assert(vmemkv::KVStore<vmemkv::VMemKVStore>);
  auto dir = std::filesystem::temp_directory_path() / "vmemkv-example";
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
  std::filesystem::create_directories(dir, ec);
  // Capacity 64 MiB: small T2 file for the example; fixed at creation, not resizable.
  vmemkv::VMemKVStore store(dir / "test.bin", 1ULL << 26);
  if (!store.insert("k", "v")) {
    std::cerr << "insert failed\n";
    return 1;
  }
  std::cout << "inserted k=v\n";
  if (!store.update("k", "v2")) {
    std::cerr << "update failed\n";
    return 1;
  }
  std::cout << "updated k=v2\n";
  bool found = false;
  if (!store.get("k", [&](std::span<const std::byte> value) { found = !value.empty(); })) {
    std::cerr << "get failed\n";
    return 1;
  }
  std::cout << "got k found=" << found << "\n";
  std::size_t scanned = store.scan("a", "z", [](std::span<const std::byte>, std::span<const std::byte>) {});
  std::cout << "scanned " << scanned << " entries\n";
  if (!store.remove("k")) {
    std::cerr << "remove failed\n";
    return 1;
  }
  std::cout << "removed k\n";
  store.bulk_load(
      100,
      [](std::size_t i) { return "bulk-" + std::to_string(i); },
      [](std::size_t i) { return "v" + std::to_string(i); });
  std::cout << "bulk-loaded 100 entries\n";
  store.checkpoint();
  std::cout << "checkpoint done\n";
  auto stats = store.get_statistics();
  std::cout << "checkpoints=" << stats.checkpoint_count << "\n";
  return 0;
}
