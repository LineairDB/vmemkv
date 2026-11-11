cmake_minimum_required(VERSION 3.16)

# One binary per test TU so ctest -j runs areas in parallel (see timings in the
# commit message). test_main.cpp supplies the doctest entry point for each.
set(VMEMKV_TEST_CASES
  test_kv_store.cpp
  test_kv_store_vmemkv.cpp
  test_public_api.cpp
  test_wal.cpp
  test_checkpoint.cpp
  test_crash_recovery.cpp
  test_t1_index.cpp
  test_sharded_t1_index.cpp
  test_t2_flat_file.cpp
  test_cgroup_throttle.cpp
  test_swap_check.cpp
  test_compaction.cpp
)

foreach(case_file IN LISTS VMEMKV_TEST_CASES)
  get_filename_component(case_name "${case_file}" NAME_WE)
  add_executable(${case_name} ${case_file} test_main.cpp)
  target_link_libraries(${case_name} PRIVATE vmemkv vmemkv_headers doctest::doctest)
  # vmemkv_headers exposes only include/; src/ (private rivals/) reaches tests only
  # via this PRIVATE dir so <rivals/...> keeps resolving without leaking into installs.
  target_include_directories(${case_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)

  vmemkv_link_rival_backends(${case_name})
  # --duration=true: doctest prints each TEST CASE's file/line/name as it *starts*, not just a
  # final summary -- combined with test_main.cpp's forced line-buffering (see its comment), this is
  # what makes a hung test identifiable in CI at all: CTest pipes the binary's stdout (non-TTY), so
  # without the buffering fix any output sitting unflushed at kill time is lost, and without
  # --duration doctest doesn't print per-test progress in the first place, only a final summary.
  add_test(NAME ${case_name} COMMAND ${case_name} --duration=true)
  # 1200s: the compaction area runs multi-megabyte corpora through real
  # checkpoints/compactions, so this is a generous ceiling for a healthy run,
  # not a budget to fill -- CI runner time isn't free. It's also comfortably
  # past the 300s mark where ci.yml's watchdog dumps thread backtraces if the process is still
  # alive, so a genuine hang still gets diagnosed (via the watchdog) well before this kills it.
  # LEANSTORE_DRAM_GIB=1: unit-test corpora are tiny; the LeanStore adapter defaults
  # to a 16GiB buffer pool that small runners cannot allocate.
  set_tests_properties(${case_name} PROPERTIES TIMEOUT 1200 ENVIRONMENT "LEANSTORE_DRAM_GIB=1")
endforeach()
