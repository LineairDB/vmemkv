cmake_minimum_required(VERSION 3.16)
project(VMemKV VERSION 0.1.0 DESCRIPTION "Larger-than-memory key-value store on OS virtual memory" LANGUAGES CXX)

# vmemkv delegates paging to the OS virtual-memory subsystem (mmap/fork/mincore).
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "vmemkv supports Linux only (mmap/fork/mincore based).")
endif()

# Allow older 3rd-party CMake files that use cmake_minimum_required(VERSION 3.0)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

# ── Interface library for headers ─────────────────────────────────────────────
# Rival comparison backends. Default OFF so a plain clone builds the library and its
# tests without third-party stores; CI enables all three explicitly.
option(ENABLE_ROCKSDB "Build with RocksDB backend" OFF)
option(ENABLE_LMDB "Build with LMDB backend" OFF)
option(ENABLE_LEANSTORE "Build with LeanStore backend" OFF)
add_library(vmemkv_headers INTERFACE)
target_compile_features(vmemkv_headers INTERFACE cxx_std_23)
target_include_directories(vmemkv_headers INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

add_library(vmemkv_warnings INTERFACE)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(vmemkv_warnings INTERFACE -Wall -Wextra -Wpedantic -Werror)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(vmemkv_warnings INTERFACE -Wall -Wextra -Wpedantic -Werror)
endif()

# ── Static library for source implementation ─────────────────────────────────
add_library(vmemkv STATIC
  src/vmemkv_store.cpp
  src/t2_flat_file/t2_flat_file.cpp
  src/wal/wal.cpp
  src/checkpoint/checkpoint.cpp
)
target_link_libraries(vmemkv PUBLIC vmemkv_headers PRIVATE $<BUILD_INTERFACE:vmemkv_warnings>)
# Private implementation headers live under src/ (src/vmemkv/...); kept off the
# installed INTERFACE so downstream consumers see only include/.
target_include_directories(vmemkv PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# ── doctest (single-header, header-only test framework) ─────────────────────
include(FetchContent)
FetchContent_Declare(
  doctest
  GIT_REPOSITORY https://github.com/doctest/doctest.git
  GIT_TAG        v2.4.11
  GIT_SHALLOW    TRUE
)
set(DOCTEST_NO_INSTALL ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(doctest)

# Rival comparison backends (RocksDB/LMDB/LeanStore probing + link helper).
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Rivals.cmake)

include(CTest)

# ── Tests ─────────────────────────────────────────────────────────────────────
# ENABLE_TESTS is canonical; BUILD_TESTING comes from include(CTest) above.
# The tidy test below additionally requires ENABLE_TESTS so an install-check
# with ENABLE_TESTS=OFF registers no tests.
option(ENABLE_TESTS "Build tests" ON)
if(ENABLE_TESTS)
  add_subdirectory(tests)
endif()

# ── Benchmarks ────────────────────────────────────────────────────────────────
option(ENABLE_BENCHMARK "Build Google Benchmark suite" ON)
if(ENABLE_BENCHMARK)
  add_subdirectory(benchmark)
endif()

# example/ is intentionally not added here: it is a standalone downstream
# consumer test (find_package(VMemKV)) exercised via the CI install-check,
# not part of this build.

# ── clang-tidy Static Analysis Target (CI-only) ──────────────────────────────
find_program(CLANG_TIDY_PROGRAM clang-tidy)
if(CLANG_TIDY_PROGRAM)
  set(CLANG_TIDY_HEADER_FILTER "(^|.*/)(src|include/vmemkv)/.*\\.(hpp|cpp)$")

  # Parallelism is fixed at configure time: make would swallow a runtime `$(nproc)` inside the
  # VERBATIM rule text.
  cmake_host_system_information(RESULT CLANG_TIDY_JOBS QUERY NUMBER_OF_LOGICAL_CORES)

  # Per-file clang-tidy processes in parallel across files (clang-tidy has no built-in -j);
  # header-filter findings repeat per including TU single- or multi-process alike, and `sort`
  # keeps the interleaved output deterministic.
  add_custom_target(clang-tidy-check
    COMMAND bash -lc "set -o pipefail; find ${CMAKE_CURRENT_SOURCE_DIR}/src -name '*.cpp' -print0 | xargs -0 -P ${CLANG_TIDY_JOBS} -n 1 ${CLANG_TIDY_PROGRAM} -quiet --extra-arg=-w --header-filter='${CLANG_TIDY_HEADER_FILTER}' -p ${CMAKE_BINARY_DIR} 2>&1 | sort | sed -E '/^[0-9]+ warnings generated\\.$/d; /Suppressed [0-9]+ warnings/d; /Use -header-filter=\\.\\*/d; /Use -system-headers/d'"
    COMMENT "Running clang-tidy static analysis..."
    VERBATIM
  )

  # Compatibility alias used by CI and local scripts (cmake-tidy-check forwards to clang-tidy-check).
  add_custom_target(cmake-tidy-check DEPENDS clang-tidy-check)

  if(BUILD_TESTING AND ENABLE_TESTS)
    add_test(
      NAME cmake_tidy_check
      COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target cmake-tidy-check --config $<CONFIG>
    )
    set_tests_properties(cmake_tidy_check PROPERTIES LABELS "lint;tidy")
  endif()
endif()

# ── Install ───────────────────────────────────────────────────────────────────
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS vmemkv vmemkv_headers EXPORT VMemKVTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(FILES include/vmemkv/vmemkv.hpp include/vmemkv/scan_control.hpp include/vmemkv/stats.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/vmemkv)
install(EXPORT VMemKVTargets NAMESPACE VMemKV:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/VMemKV)
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/VMemKVConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/VMemKVConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/VMemKV
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/VMemKVConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/VMemKVConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/VMemKVConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/VMemKV
)
