# Rivals.cmake — Optional rival comparison-backend probing + link helper.
#
# Included by the top-level CMakeLists after the ENABLE_ROCKSDB/ENABLE_LMDB/
# ENABLE_LEANSTORE options are defined. Keeps third-party probing out of the
# main file; behavior is identical to the former inline blocks.
#
# Defines per-backend *_FOUND when located and vmemkv_link_rival_backends(target),
# which links the found rival backends into `target` and defines
# ENABLE_ROCKSDB/ENABLE_LMDB/ENABLE_LEANSTORE for it. Shared by tests/ and
# benchmark/, which both build against the same optional rival backends.

# ── RocksDB (optional) ────────────────────────────────────────────────────────
if(ENABLE_ROCKSDB)
  find_package(RocksDB QUIET)
  if(NOT RocksDB_FOUND)
    find_library(ROCKSDB_LIB rocksdb
      HINTS /opt/homebrew/lib /usr/local/lib /usr/lib)
    find_path(ROCKSDB_INCLUDE_DIR rocksdb/db.h
      HINTS /opt/homebrew/include /usr/local/include /usr/include)
    if(ROCKSDB_LIB AND ROCKSDB_INCLUDE_DIR)
      add_library(RocksDB::rocksdb UNKNOWN IMPORTED)
      set_target_properties(RocksDB::rocksdb PROPERTIES
        IMPORTED_LOCATION "${ROCKSDB_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${ROCKSDB_INCLUDE_DIR}")
      set(RocksDB_FOUND TRUE)
      message(STATUS "Found RocksDB: ${ROCKSDB_LIB}")
    else()
      message(FATAL_ERROR "ENABLE_ROCKSDB=ON but RocksDB not found.")
    endif()
  endif()
endif()

# ── LMDB (optional) ───────────────────────────────────────────────────────────
if(ENABLE_LMDB)
  find_package(LMDB QUIET)
  if(NOT LMDB_FOUND)
    find_library(LMDB_LIB lmdb
      HINTS /opt/homebrew/lib /usr/local/lib /usr/lib)
    find_path(LMDB_INCLUDE_DIR lmdb.h
      HINTS /opt/homebrew/include /usr/local/include /usr/include)
    if(LMDB_LIB AND LMDB_INCLUDE_DIR)
      add_library(LMDB::lmdb UNKNOWN IMPORTED)
      set_target_properties(LMDB::lmdb PROPERTIES
        IMPORTED_LOCATION "${LMDB_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${LMDB_INCLUDE_DIR}")
      set(LMDB_FOUND TRUE)
      message(STATUS "Found LMDB: ${LMDB_LIB}")
    else()
      message(FATAL_ERROR "ENABLE_LMDB=ON but LMDB not found.")
    endif()
  endif()
endif()

# ── LeanStore (optional) ────────────────────────────────────────────────────
# Pinned by commit: master moves fast and carries no stable tags. Only backend/ is
# configured (EXCLUDE_FROM_ALL, built on demand); frontend/ executables additionally need
# rocksdb/wiredtiger and are never configured. backend's own CMakeLists links tbb/aio by
# bare name, so their -I/-L are appended to the leanstore target below (PUBLIC: consumers
# including leanstore headers need them too).
if(ENABLE_LEANSTORE)
  FetchContent_Declare(leanstore_probe
    GIT_REPOSITORY https://github.com/leanstore/leanstore.git
    GIT_TAG 90fcf185c1c8506344a7aa779928787d494348f4)
  FetchContent_GetProperties(leanstore_probe)
  if(NOT leanstore_probe_POPULATED)
    FetchContent_Populate(leanstore_probe)
    find_package(Threads REQUIRED)
    include(${leanstore_probe_SOURCE_DIR}/libs/tabluate.cmake)
    include(${leanstore_probe_SOURCE_DIR}/libs/rapidjson.cmake)
    add_subdirectory(${leanstore_probe_SOURCE_DIR}/backend ${leanstore_probe_BINARY_DIR}/backend EXCLUDE_FROM_ALL)
  endif()
  find_path(LEANSTORE_TBB_INCLUDE_DIR tbb/enumerable_thread_specific.h
    HINTS $ENV{LEANSTORE_SYSROOT}/usr/include /usr/include)
  find_library(LEANSTORE_TBB_LIB tbb
    HINTS $ENV{LEANSTORE_SYSROOT}/usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu /usr/lib)
  find_path(LEANSTORE_AIO_INCLUDE_DIR libaio.h
    HINTS $ENV{LEANSTORE_SYSROOT}/usr/include /usr/include)
  find_library(LEANSTORE_AIO_LIB aio
    HINTS $ENV{LEANSTORE_SYSROOT}/usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu /usr/lib)
  if(LEANSTORE_TBB_INCLUDE_DIR AND LEANSTORE_TBB_LIB AND LEANSTORE_AIO_INCLUDE_DIR AND LEANSTORE_AIO_LIB)
    set(LeanStore_FOUND TRUE)
    message(STATUS "Found LeanStore deps: ${LEANSTORE_TBB_LIB} ${LEANSTORE_AIO_LIB}")
    # SYSTEM: third-party headers carry GNU extensions our -Wpedantic -Werror build
    # would otherwise reject; -isystem keeps our warnings enforced on our own code only.
    target_include_directories(leanstore SYSTEM PUBLIC ${LEANSTORE_TBB_INCLUDE_DIR} ${LEANSTORE_AIO_INCLUDE_DIR})
    target_include_directories(leanstore SYSTEM PUBLIC ${leanstore_probe_SOURCE_DIR}/backend
                                                       ${leanstore_probe_SOURCE_DIR}/shared-headers)
    target_include_directories(tabluate SYSTEM INTERFACE ${TABLUATE_INCLUDE_DIR})
    target_include_directories(rapidjson SYSTEM INTERFACE ${RAPIDJSON_INCLUDE_DIR})
    get_filename_component(LEANSTORE_TBB_LIB_DIR "${LEANSTORE_TBB_LIB}" DIRECTORY)
    get_filename_component(LEANSTORE_AIO_LIB_DIR "${LEANSTORE_AIO_LIB}" DIRECTORY)
    target_link_directories(leanstore PUBLIC ${LEANSTORE_TBB_LIB_DIR} ${LEANSTORE_AIO_LIB_DIR})
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
      target_compile_options(leanstore PRIVATE -mavx2 -mcx16)
    endif()
  else()
    message(FATAL_ERROR "ENABLE_LEANSTORE=ON but tbb/aio not found (apt install libtbb-dev libaio-dev, or point LEANSTORE_SYSROOT at an extracted sysroot).")
  endif()
endif()

# Links the RocksDB/LMDB/LeanStore rival-store backends into `target` and defines
# ENABLE_ROCKSDB/ENABLE_LMDB/ENABLE_LEANSTORE for it, whichever of the three were actually
# found above. Shared by tests/ and benchmark/, which both build against the same optional
# rival backends.
function(vmemkv_link_rival_backends target)
  if(ENABLE_ROCKSDB AND RocksDB_FOUND)
    target_compile_definitions(${target} PRIVATE ENABLE_ROCKSDB)
    target_link_libraries(${target} PRIVATE RocksDB::rocksdb)
  endif()
  if(ENABLE_LMDB AND LMDB_FOUND)
    target_compile_definitions(${target} PRIVATE ENABLE_LMDB)
    target_link_libraries(${target} PRIVATE LMDB::lmdb)
  endif()
  if(ENABLE_LEANSTORE AND LeanStore_FOUND)
    target_compile_definitions(${target} PRIVATE ENABLE_LEANSTORE)
    target_link_libraries(${target} PRIVATE leanstore)
  endif()
endfunction()
