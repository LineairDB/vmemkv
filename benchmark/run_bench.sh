#!/usr/bin/env bash
# run_bench.sh - Unified vmemkv benchmark runner (local + AWS spot).
#
# Single entry point for measuring the CRUD/Scan/YCSB-E matrix and the
# background-maintenance probes. Subcommands share one
# scenario-matrix definition, so local and AWS runs cannot drift apart on what
# gets measured -- only on how a scenario step runs (in-process here vs. over
# SSH, optionally inside a cgroup, there).
#
# Results are written as Google Benchmark JSON under benchmark/logs/.
# The measured binary is $IMPL_ROOT/build*/benchmark/bench_kv.
# Subcommands: local | aws | parallel | probe | run-scenario | clean.
#   local        In-process run on this machine (default when argv holds flags).
#   aws          Full-matrix run on an EC2 spot instance.
#   parallel     4 CRUD-matrix combos + 1 probe instance, each on its own spot VM.
#   probe        One background-maintenance probe point set (bg-jobs | organic-split).
#   run-scenario One Google Benchmark scenario step (same code local and remote).
#   clean        Delete temporary AWS resources from a previous run.

set -euo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_SELF="$BENCH_DIR/run_bench.sh"
# vmemkv source root (measured binary lives under $IMPL_ROOT/build*/benchmark).
# Layouts: monorepo checkout, standalone repository, and remote
# single-file deployment (~/run_bench.sh with the repo root synced to ~/vmemkv).
# The fallback derives from the script's own directory, not $HOME: probe/LTM passes
# run this script under `sudo systemd-run -- bash -lc` where HOME is unset (and would
# be /root even if set), while the script itself is always invoked by absolute path
# (/home/ubuntu/run_bench.sh) so BENCH_DIR stays /home/ubuntu either way.
if [[ -d "$BENCH_DIR/../implementation/vmemkv" ]]; then
  IMPL_ROOT="$(cd "$BENCH_DIR/../implementation/vmemkv" && pwd)"
elif [[ -f "$BENCH_DIR/../CMakeLists.txt" && -d "$BENCH_DIR/../src/vmemkv" ]]; then
  IMPL_ROOT="$(cd "$BENCH_DIR/.." && pwd)"
else
  IMPL_ROOT="$BENCH_DIR/vmemkv"
fi

# ── scenario matrix (filters, env prefixes, result paths) ──
# Source of truth for benchmark filters and result paths.
# Keep the data here so both local and AWS runners stay in sync without
# duplicating filter strings.

# Read-policy pins only diverge from the default read policy under memory pressure.
# In-memory matrices measure only the default configuration; LTM also measures read policies.
vmemkv_matrix::vmemkv_variant_filter() {
  case "${1:-}" in
    in_memory) printf '%s\n' 'Bloom-T1InlineValue' ;;
    *) printf '%s\n' 'Bloom-T1InlineValue|Bloom-T1InlineValue-ReadRandom|Bloom-T1InlineValue-ReadSeq|Bloom-T1InlineValue-ReadNormal|Bloom-T1InlineValue-ReadPread' ;;
  esac
}

vmemkv_matrix::_store_filter() {
  # Shared body for scenario_filter() / scenario_filter_no_leanstore(): include_leanstore
  # ("true"/"1") keeps the LeanStore alternative in the rival branch.
  local scenario_key="${1:-}"
  local vmemkv_only="${2:-}"
  local include_leanstore="${3:-true}"
  local variants
  variants="$(vmemkv_matrix::vmemkv_variant_filter "$scenario_key")"
  if [[ "$vmemkv_only" == "true" || "$vmemkv_only" == "1" ]]; then
    printf '(^Store=VMemKV/Variant=(%s)/)\n' "$variants"
  elif [[ "$include_leanstore" == "true" || "$include_leanstore" == "1" ]]; then
    printf '(^Store=VMemKV/Variant=(%s)/|^Store=RocksDB/|^Store=RocksDB-BlobDB/|^Store=LMDB/|^Store=LeanStore/)\n' "$variants"
  else
    printf '(^Store=VMemKV/Variant=(%s)/|^Store=RocksDB/|^Store=RocksDB-BlobDB/|^Store=LMDB/)\n' "$variants"
  fi
}

vmemkv_matrix::scenario_filter() {
  # vmemkv_only ("true"/"1"): drops the four rival backends
  # (RocksDB/RocksDB-BlobDB/LMDB/LeanStore) from
  # the filter -- their numbers are unaffected by a vmemkv-internal-only code change, so
  # re-measuring them is unnecessary for a VMemKV-only regression-check run.
  vmemkv_matrix::_store_filter "${1:-}" "${2:-}" true
}

vmemkv_matrix::scenario_filter_no_leanstore() {
  # Same as scenario_filter() without LeanStore. LeanStore's BTreeVI key/value lengths are u16,
  # so the 64KB corpus (65536-byte values) is unstorable there; benchmark_filter_for_case()
  # substitutes this for 64KB cases (RE2 has no negative lookahead, hence the explicit
  # alternation instead of an exclusion pattern).
  vmemkv_matrix::_store_filter "${1:-}" "${2:-}" false
}

# LTM scenario memory parameters: a small *declared* budget (fed to bench_kv as
# VMEMKV_CONTEXT_memory_budget_bytes) combined with a large target_ratio (8.0) yields a
# corpus a few times bigger than the budget. Actually forcing that corpus to spill into
# swap requires constraining the process's real RSS to roughly this budget (e.g. via
# `systemd-run ... -p MemoryHigh=... -p MemoryMax=...` as the AWS runner does); without
# that cgroup wrap, the numbers here only size the corpus consistently across
# environments, they do not by themselves recreate memory pressure.
vmemkv_matrix::ltm_memory_budget_bytes() {
  printf '%s\n' "$((1 * 1024 * 1024 * 1024))"
}

vmemkv_matrix::ltm_swap_budget_bytes() {
  printf '%s\n' "$((1024 * 1024 * 1024 * 1024))"
}

vmemkv_matrix::scenario_env_prefix() {
  # Opt-in diagnostic passthrough (unset by default): suppresses VMemKV's background
  # compaction worker for deterministic builds/measurements.
  local suppress_compact=""
  if [[ -n "${VMEMKV_SUPPRESS_AUTO_COMPACT:-}" ]]; then
    suppress_compact=" VMEMKV_SUPPRESS_AUTO_COMPACT=${VMEMKV_SUPPRESS_AUTO_COMPACT}"
  fi
  case "$1" in
    in_memory) printf '%s\n' "VMEMKV_BENCH_TARGET_RATIO=0.5${suppress_compact}" ;;
    ltm) printf '%s\n' "VMEMKV_BENCH_LTM=1 VMEMKV_BENCH_TARGET_RATIO=8.0${suppress_compact}" ;;
    *) return 1 ;;
  esac
}

vmemkv_matrix::scenario_result_path() {
  case "$1" in
    in_memory) printf '%s\n' "/tmp/results_in_memory.json" ;;
    ltm) printf '%s\n' "/tmp/results_ltm.json" ;;
    *) return 1 ;;
  esac
}

vmemkv_matrix::scenario_order_keys_from_flag() {
  case "$1" in
    A_FIRST|"") printf '%s\n' "in_memory ltm" ;;
    B_FIRST) printf '%s\n' "ltm in_memory" ;;
    *) return 1 ;;
  esac
}

vmemkv_matrix::scenario_value_order_keys_from_flag() {
  local scenario_key="$1"
  local large_value_first="${2:-0}"

  case "$scenario_key" in
    in_memory)
      case "$large_value_first" in
        true|1) printf '%s\n' "1kb 8b" ;;
        false|0|"") printf '%s\n' "8b 1kb" ;;
        *) return 1 ;;
      esac
      ;;
    ltm)
      case "$large_value_first" in
        true|1) printf '%s\n' "64kb 1kb" ;;
        false|0|"") printf '%s\n' "1kb 64kb" ;;
        *) return 1 ;;
      esac
      ;;
    *)
      return 1
      ;;
  esac
}

vmemkv_matrix::value_filter_fragment() {
  case "$1" in
    8b) printf '%s\n' "Value=8B" ;;
    1kb) printf '%s\n' "Value=1KB" ;;
    64kb) printf '%s\n' "Value=64KB" ;;
    *) return 1 ;;
  esac
}

vmemkv_matrix::benchmark_filter_for_case() {
  local scenario_key="$1"
  local value_key="$2"
  local vmemkv_only="${3:-}"
  local leanstore_only="${4:-}"
  local scenario_regex

  if [[ "$leanstore_only" == "true" || "$leanstore_only" == "1" ]]; then
    # LeanStore-only re-measurement (e.g. after a rival-backend change): output contains
    # LeanStore cells only.
    # LeanStore's BTreeVI key/value lengths are u16, so the 64KB corpus is unstorable there:
    # match nothing (deliberately, so callers fail visibly instead of throwing mid-run).
    if [[ "$value_key" == "64kb" ]]; then
      printf '%s\n' '(^Store=NoSuchStore/)'
      return 0
    fi
    scenario_regex='(^Store=LeanStore/)'
  elif [[ "$value_key" == "64kb" ]]; then
    # LeanStore's BTreeVI lengths are u16: registration (see value_sizes_for_mode() in
    # bench_kv.cpp) never creates its 64KB cells, and the matrix filter below mirrors that
    # so the 64KB case resolves to the other backends only.
    scenario_regex="$(vmemkv_matrix::scenario_filter_no_leanstore "$scenario_key" "$vmemkv_only")"
  else
    scenario_regex="$(vmemkv_matrix::scenario_filter "$scenario_key" "$vmemkv_only")"
  fi
  printf '%s\n' "(${scenario_regex}).*$(vmemkv_matrix::value_filter_fragment "$value_key")"
}

vmemkv_matrix::scenario_run_filter() {
  local scenario_key="$1"
  local value_order="$2"
  local vmemkv_only="${3:-}"
  local leanstore_only="${4:-}"
  local -a values=()
  local -a case_filters=()
  local value_key
  local joined=""

  read -r -a values <<<"$value_order"
  for value_key in "${values[@]}"; do
    case_filters+=("$(vmemkv_matrix::benchmark_filter_for_case "$scenario_key" "$value_key" "$vmemkv_only" "$leanstore_only")")
  done

  for value_key in "${case_filters[@]}"; do
    if [[ -z "$joined" ]]; then
      joined="$value_key"
    else
      joined+="|$value_key"
    fi
  done
  printf '(%s)\n' "$joined"
}

vmemkv_matrix::scenario_effective_filter() {
  local scenario_key="$1"
  local large_value_first="${2:-0}"
  local vmemkv_only="${3:-}"
  local leanstore_only="${4:-}"

  vmemkv_matrix::scenario_run_filter \
    "$scenario_key" \
    "$(vmemkv_matrix::scenario_value_order_keys_from_flag "$scenario_key" "$large_value_first")" \
    "$vmemkv_only" \
    "$leanstore_only"
}

vmemkv_matrix::ltm_priming_filter() {
  # Triggers the minimum necessary single-thread cells (threads:1) to pre-populate shared
  # master corpora at unconstrained speed before entering the cgroup. Respects YCSB_ONLY,
  # VALUE_SIZE_LIMIT, WITHOUT_RIVALS, LEANSTORE_ONLY, and ROCKSDB_ONLY to avoid building
  # unused masters.
  local vmemkv_only="${1:-false}"
  local leanstore_only="${2:-false}"
  local rocksdb_only="${3:-false}"

  local val_fragment=""
  if [[ -n "${VALUE_SIZE_LIMIT:-}" ]]; then
    val_fragment="$(vmemkv_matrix::value_filter_fragment "${VALUE_SIZE_LIMIT,,}")"
  fi
  local val_filter=""
  if [[ -n "$val_fragment" ]]; then
    val_filter="/${val_fragment}"
  fi

  # YCSB_ONLY: YCSB-E uses Bloom-T1InlineValue and RocksDB only, ordered load only.
  if [[ "${YCSB_ONLY:-false}" == "true" ]]; then
    local store_pattern="Store=(VMemKV|RocksDB)/Variant=(Bloom-T1InlineValue|RocksDB)"
    if [[ "$vmemkv_only" == "true" || "$vmemkv_only" == "1" || "${WITHOUT_RIVALS:-false}" == "true" ]]; then
      store_pattern="Store=VMemKV/Variant=Bloom-T1InlineValue"
    elif [[ "$rocksdb_only" == "true" || "$rocksdb_only" == "1" || "${ROCKSDB_ONLY:-false}" == "true" ]]; then
      store_pattern="Store=RocksDB/Variant=RocksDB"
    fi
    printf '((^%s/).*Op=Get/Mode=Hit/Dist=Zipf%s.*threads:1$)\n' "$store_pattern" "$val_filter"
    return 0
  fi

  # ROCKSDB_ONLY
  if [[ "$rocksdb_only" == "true" || "$rocksdb_only" == "1" || "${ROCKSDB_ONLY:-false}" == "true" ]]; then
    printf '((^Store=RocksDB/).*Op=Get/Mode=Hit/Dist=Zipf%s.*threads:1$|(^Store=RocksDB/).*Op=Scan/Dist=Zipf.*%s.*Load=Shuffled.*threads:1$)\n' \
      "$val_filter" "${val_fragment:-}"
    return 0
  fi

  # LEANSTORE_ONLY (BTreeVI key/value length u16 ceiling: 64KB not supported)
  if [[ "$leanstore_only" == "true" || "$leanstore_only" == "1" || "${LEANSTORE_ONLY:-false}" == "true" ]]; then
    if [[ "${VALUE_SIZE_LIMIT,,}" == "64kb" ]]; then
      printf '(^Store=NoSuchStore/)\n'
      return 0
    fi
    local ls_val="${val_filter:-/Value=(8B|1KB)}"
    local ls_shuf_val="${val_fragment:-Value=(8B|1KB)}"
    printf '((^Store=LeanStore/).*Op=Get/Mode=Hit/Dist=Zipf%s.*threads:1$|(^Store=LeanStore/).*Op=Scan/Dist=Zipf.*%s.*Load=Shuffled.*threads:1$)\n' \
      "$ls_val" "$ls_shuf_val"
    return 0
  fi

  # General matrix path
  local scenario_regex
  local shuffled_prime
  if [[ -n "$val_filter" ]]; then
    shuffled_prime=".*Op=Scan/Dist=Zipf.*%s.*Load=Shuffled.*threads:1\$"
  else
    shuffled_prime='.*Op=Scan/Dist=Zipf/.*Load=Shuffled.*threads:1$'
  fi

  if [[ "$vmemkv_only" == "true" || "$vmemkv_only" == "1" || "${WITHOUT_RIVALS:-false}" == "true" ]]; then
    scenario_regex="$(vmemkv_matrix::scenario_filter "ltm" "true")"
    if [[ -n "$val_filter" ]]; then
      printf '((%s).*Op=Get/Mode=Hit/Dist=Zipf%s.*threads:1$|(%s)%s)\n' \
        "$scenario_regex" "$val_filter" "$scenario_regex" "$(printf "$shuffled_prime" "$val_fragment")"
    else
      printf '((%s).*Op=Get/Mode=Hit/Dist=Zipf/.*threads:1$|(%s)%s)\n' \
        "$scenario_regex" "$scenario_regex" "$shuffled_prime"
    fi
  else
    local no_leanstore_regex
    no_leanstore_regex="$(vmemkv_matrix::scenario_filter_no_leanstore "ltm" "false")"
    if [[ "${VALUE_SIZE_LIMIT,,}" == "64kb" ]]; then
      printf '((%s).*Op=Get/Mode=Hit/Dist=Zipf%s.*threads:1$|(%s)%s)\n' \
        "$no_leanstore_regex" "$val_filter" "$no_leanstore_regex" "$(printf "$shuffled_prime" "$val_fragment")"
    else
      local ls_val="${val_filter:-/Value=(8B|1KB)}"
      local ls_shuf_val="${val_fragment:-Value=(8B|1KB)}"
      local leanstore_prime="((^Store=LeanStore/).*Op=Get/Mode=Hit/Dist=Zipf${ls_val}.*threads:1\$)"
      local leanstore_shuffled_prime="((^Store=LeanStore/).*Op=Scan/Dist=Zipf.*${ls_shuf_val}.*Load=Shuffled.*threads:1\$)"
      if [[ -n "$val_filter" ]]; then
        printf '((%s).*Op=Get/Mode=Hit/Dist=Zipf%s.*threads:1$|%s|(%s)%s|%s)\n' \
          "$no_leanstore_regex" "$val_filter" "$leanstore_prime" "$no_leanstore_regex" "$(printf "$shuffled_prime" "$val_fragment")" "$leanstore_shuffled_prime"
      else
        printf '((%s).*Op=Get/Mode=Hit/Dist=Zipf/.*threads:1$|%s|(%s)%s|%s)\n' \
          "$no_leanstore_regex" "$leanstore_prime" "$no_leanstore_regex" "$shuffled_prime" "$leanstore_shuffled_prime"
      fi
    fi
  fi
}

vmemkv_matrix::ltm_priming_filter_from_names() {
  # Maps a newline-separated list of matched benchmark test names on stdin to the minimal
  # threads:1 priming filter for the shared master corpora those cells require.
  awk -F'/' '
    NF == 0 || /\/Op=Insert\// { next }
    {
      val = ""
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^Value=/) {
          val = $i
          break
        }
      }
      if (val == "") next
      if ($0 ~ /\/Load=Shuffled\//) {
        print $1 "/" $2 "/Op=Scan/Dist=Zipf/" val "/Load=Shuffled/real_time/threads:1"
      } else {
        print $1 "/" $2 "/Op=Get/Mode=Hit/Dist=Zipf/" val "/real_time/threads:1"
      }
    }
  ' | sort -u | sed -E 's/[][(){}.^$*+?\\|]/\\&/g' | paste -sd'|' - | awk 'NF { printf "^(%s)$\n", $0 } END { if (NR == 0) print "^Store=NoSuchStore/$" }'
}

# ── shared shell helpers ──
# Shared helper functions for vmemkv benchmark runners.
# Keep this file shell-only and dependency-light so both local and AWS runners
# can source it without pulling in extra runtime requirements.

vmemkv_git_revision() {
  local revision
  if revision="$(git -C "$IMPL_ROOT" rev-parse --short=12 HEAD 2>/dev/null)" && [[ -n "$revision" ]]; then
    printf '%s\n' "$revision"
  elif [[ -n "${VMEMKV_CONTEXT_git_revision:-}" ]]; then
    printf '%s\n' "$VMEMKV_CONTEXT_git_revision"
  else
    printf '%s\n' "uncommitted"
  fi
}

vmemkv_git_dirty() {
  if git -C "$IMPL_ROOT" rev-parse --verify HEAD >/dev/null 2>&1; then
    if git -C "$IMPL_ROOT" status --porcelain --untracked-files=no | awk 'NF { exit 1 }'; then
      printf '%s\n' "0"
    else
      printf '%s\n' "1"
    fi
  elif [[ -n "${VMEMKV_CONTEXT_git_dirty:-}" ]]; then
    printf '%s\n' "$VMEMKV_CONTEXT_git_dirty"
  else
    printf '%s\n' "1"
  fi
}

vmemkv_join_arrow() {
  local joined=""
  local item

  for item in "$@"; do
    if [[ -z "$joined" ]]; then
      joined="$item"
    else
      joined+="->$item"
    fi
  done
  printf '%s\n' "$joined"
}

vmemkv_scenario_order_label() {
  vmemkv_join_arrow "$@"
}

vmemkv_value_order_label() {
  local mapped=()
  local value_key

  for value_key in "$@"; do
    case "$value_key" in
      64kb) mapped+=("64KB") ;;
      1kb) mapped+=("1KB") ;;
      *) mapped+=("8B") ;;
    esac
  done
  if ((${#mapped[@]})); then
    vmemkv_join_arrow "${mapped[@]}"
  else
    printf '\n'
  fi
}

vmemkv_target_profile_label() {
  local mapped=()
  local scenario_key
  local ratio

  for scenario_key in "$@"; do
    case "$scenario_key" in
      in_memory) ratio="0.5" ;;
      ltm) ratio="8.0" ;;
      *) ratio="?" ;;
    esac
    mapped+=("$scenario_key=$ratio")
  done
  if ((${#mapped[@]})); then
    vmemkv_join_arrow "${mapped[@]}"
  else
    printf '\n'
  fi
}

vmemkv_benchmark_memo() {
  local run_kind="$1"
  shift
  local has_ltm=false
  local scenario_key

  for scenario_key in "$@"; do
    if [[ "$scenario_key" == "ltm" ]]; then
      has_ltm=true
      break
    fi
  done

  if [[ "$has_ltm" == "true" && "$run_kind" == aws_* ]]; then
    printf '%s\n' "AWS LTM runs use a 1GiB cgroup budget so corpus sizes stay bounded by the effective memory limit."
  fi
}

vmemkv_mem_total_bytes() {
  awk '/MemTotal:/ { print $2 * 1024; exit }' /proc/meminfo
}

vmemkv_swap_total_bytes() {
  awk '/SwapTotal:/ { print $2 * 1024; exit }' /proc/meminfo
}

vmemkv_cpu_model() {
  awk -F': ' '/model name/ { print $2; exit }' /proc/cpuinfo
}

vmemkv_cgroup_memory_limit_bytes() {
  local cgroup_v2_limit="/sys/fs/cgroup/memory.max"
  local cgroup_v1_limit="/sys/fs/cgroup/memory/memory.limit_in_bytes"
  local limit

  if [[ -r "$cgroup_v2_limit" ]]; then
    limit="$(cat "$cgroup_v2_limit")"
    if [[ "$limit" != "max" ]]; then
      printf '%s\n' "$limit"
      return 0
    fi
  fi

  if [[ -r "$cgroup_v1_limit" ]]; then
    limit="$(cat "$cgroup_v1_limit")"
    # cgroup v1 limit can be a very large number (e.g. 9223372036854771712) when unlimited.
    if [[ -n "$limit" && "$limit" -lt 9000000000000000000 ]]; then
      printf '%s\n' "$limit"
      return 0
    fi
  fi

  printf '%s\n' "-1"
  return 0
}

vmemkv_context_env_array() {
  local -n env_ref="$1"
  shift
  local run_kind="$1"
  local build_type="$2"
  local build_dir="$3"
  local enable_rocksdb="$4"
  local scenario_order="$5"
  local benchmark_flags="$6"
  local memory_budget_source="${7:-}"
  local memory_budget_bytes="${8:-}"
  local instance_type="${9:-}"
  local aws_region="${10:-}"
  local memo="${11:-}"
  local memory_high_bytes="${12:-}"
  local memory_max_bytes="${13:-}"
  local memory_swap_max_bytes="${14:-}"
  local swap_storage_media="${15:-}"
  local cgroup_limit=""

  env_ref=()
  env_ref+=("VMEMKV_CONTEXT_git_revision=$(vmemkv_git_revision)")
  env_ref+=("VMEMKV_CONTEXT_git_dirty=$(vmemkv_git_dirty)")
  env_ref+=("VMEMKV_CONTEXT_run_kind=$run_kind")
  env_ref+=("VMEMKV_CONTEXT_build_type=$build_type")
  env_ref+=("VMEMKV_CONTEXT_build_dir=$build_dir")
  env_ref+=("VMEMKV_CONTEXT_enable_rocksdb=$enable_rocksdb")
  env_ref+=("VMEMKV_CONTEXT_scenario_order=$scenario_order")
  env_ref+=("VMEMKV_CONTEXT_flags=$benchmark_flags")
  if [[ -n "$memory_budget_source" ]]; then
    env_ref+=("VMEMKV_CONTEXT_memory_budget_source=$memory_budget_source")
  fi
  if [[ -n "$memory_budget_bytes" ]]; then
    env_ref+=("VMEMKV_CONTEXT_memory_budget_bytes=$memory_budget_bytes")
  fi
  if [[ -n "$memory_high_bytes" ]]; then
    env_ref+=("VMEMKV_CONTEXT_memory_high_bytes=$memory_high_bytes")
  fi
  if [[ -n "$memory_max_bytes" ]]; then
    env_ref+=("VMEMKV_CONTEXT_memory_max_bytes=$memory_max_bytes")
  fi
  if [[ -n "$memory_swap_max_bytes" ]]; then
    env_ref+=("VMEMKV_CONTEXT_memory_swap_max_bytes=$memory_swap_max_bytes")
  fi
  if [[ -n "$swap_storage_media" ]]; then
    env_ref+=("VMEMKV_CONTEXT_swap_storage_media=$swap_storage_media")
  fi
  env_ref+=("VMEMKV_CONTEXT_kernel_release=$(uname -r)")
  env_ref+=("VMEMKV_CONTEXT_cpu_model=$(vmemkv_cpu_model)")
  env_ref+=("VMEMKV_CONTEXT_cpu_count=$(nproc --all)")
  env_ref+=("VMEMKV_CONTEXT_mem_total_bytes=$(vmemkv_mem_total_bytes)")
  cgroup_limit="$(vmemkv_cgroup_memory_limit_bytes 2>/dev/null || true)"
  env_ref+=("VMEMKV_CONTEXT_cgroup_memory_limit_bytes=$cgroup_limit")
  env_ref+=("VMEMKV_CONTEXT_swap_total_bytes=$(vmemkv_swap_total_bytes)")
  if [[ -n "$instance_type" ]]; then
    env_ref+=("VMEMKV_CONTEXT_instance_type=$instance_type")
  fi
  if [[ -n "$aws_region" ]]; then
    env_ref+=("VMEMKV_CONTEXT_aws_region=$aws_region")
  fi
  if [[ -n "$memo" ]]; then
    env_ref+=("VMEMKV_CONTEXT_memo=$memo")
  fi
}

vmemkv_context_env_prefix() {
  local -a env_pairs=()
  local -a quoted=()
  local pair

  vmemkv_context_env_array env_pairs "$@"
  for pair in "${env_pairs[@]}"; do
    printf -v pair '%q' "$pair"
    quoted+=("$pair")
  done
  printf '%s ' "${quoted[@]}"
}

vmemkv_build_bench() {
  local source_dir="$1"
  local build_dir="$2"
  local build_type="$3"
  local enable_rocksdb="$4"
  shift 4

  local -a configure_args=(
    -S "$source_dir"
    -B "$build_dir"
    -DCMAKE_BUILD_TYPE="$build_type"
    -DENABLE_ROCKSDB="$enable_rocksdb"
    -DENABLE_BENCHMARK=ON
  )
  if (($# > 0)); then
    configure_args+=("$@")
  fi

  cmake "${configure_args[@]}"
  cmake --build "$build_dir" --target bench_kv --parallel
}

# ── measurement definition (what gets measured) ──
# Measurement definition shared by the local and aws subcommands: both measure
# the same thing and differ only in *how* a scenario runs (in-process here vs.
# over SSH, optionally inside a cgroup, there). These functions own the two
# decisions that define "what gets measured": filter selection per scenario,
# and scenario iteration order. Backend execution (cmake build, SSH
# provisioning/execution, result download, probes, auto-log files) stays in
# the subcommands.
#
# Globals read (subcommands set what they own; anything unset defaults neutral):
#   LARGE_VALUE_FIRST, WITHOUT_RIVALS, LEANSTORE_ONLY, VALUE_SIZE_LIMIT,
#   YCSB_ONLY, SCENARIO_LIMIT, SCENARIO_ORDER_KEYS.

# Prints the benchmark filter for one scenario key (in_memory|ltm).
run_bench::scenario_filter() {
  local scenario_key="$1"
  if [[ "${YCSB_ONLY:-false}" == "true" ]]; then
    local store_pattern="Store=(VMemKV|RocksDB)/Variant=(Bloom-T1InlineValue|RocksDB)"
    if [[ "${WITHOUT_RIVALS:-false}" == "true" ]]; then
      store_pattern="Store=VMemKV/Variant=Bloom-T1InlineValue"
    fi
    if [[ "${ROCKSDB_ONLY:-false}" == "true" ]]; then
      store_pattern="Store=RocksDB/Variant=RocksDB"
    fi
    local val_pattern=""
    if [[ -n "${VALUE_SIZE_LIMIT:-}" ]]; then
      val_pattern="/$(vmemkv_matrix::value_filter_fragment "${VALUE_SIZE_LIMIT,,}")"
    fi
    printf '%s/Op=YCSB-E/Dist=Zipf%s\n' "$store_pattern" "$val_pattern"
    return 0
  fi
  if [[ -n "${VALUE_SIZE_LIMIT:-}" ]]; then
    vmemkv_matrix::benchmark_filter_for_case \
      "$scenario_key" "${VALUE_SIZE_LIMIT,,}" "${WITHOUT_RIVALS:-false}" "${LEANSTORE_ONLY:-false}"
    return 0
  fi
  vmemkv_matrix::scenario_effective_filter \
    "$scenario_key" "${LARGE_VALUE_FIRST:-false}" \
    "${WITHOUT_RIVALS:-false}" "${LEANSTORE_ONLY:-false}"
}

# Calls <matrix_fn> once per active scenario, in ORDER (SCENARIO_ORDER_KEYS
# filtered by SCENARIO_LIMIT). Replaces each launcher's own scenario if/elif chain
# so the two cannot drift apart on iteration order.
run_bench::each_scenario() {
  local matrix_fn="$1" key
  for key in "${SCENARIO_ORDER_KEYS[@]}"; do
    if [[ "${SCENARIO_LIMIT:-all}" != "all" && "${SCENARIO_LIMIT}" != "$key" ]]; then
      continue
    fi
    "$matrix_fn" "$key"
  done
}

write_progress_awk() {
  cat > "$1" <<'PROGRESS_AWK_EOF'
# Prefix Google Benchmark output with wall clock time, elapsed time, and
# progress counters.
#
# Expected state file format: a single bare number (global_total).

function color_enabled() {
  return ("VMEMKV_COLOR" in ENVIRON) && ENVIRON["VMEMKV_COLOR"] != "" && ENVIRON["VMEMKV_COLOR"] != "0" && tolower(ENVIRON["VMEMKV_COLOR"]) != "false";
}

function load_state(   line) {
  global_total = 0;

  if ((getline line < state_file) > 0) {
    global_total = line + 0;
  }
  close(state_file);
}

function make_prefix(global_done, global_total,   now, wall, elapsed, hrs, mins, secs, percent) {
  now = systime();
  wall = strftime("%H:%M:%S", now);
  elapsed = now - start;
  hrs = int(elapsed / 3600);
  mins = int((elapsed % 3600) / 60);
  secs = elapsed % 60;
  if (global_total > 0) {
    percent = int((global_done * 100) / global_total);
    return sprintf("[%s][%02d:%02d:%02d][Total %d/%d (%d%%)]", wall, hrs, mins, secs, global_done, global_total, percent);
  } else {
    return sprintf("[%s][%02d:%02d:%02d][Total %d]", wall, hrs, mins, secs, global_done);
  }
}

BEGIN {
  start = systime();
  last_emit = start;
  global_done = 0;
  use_color = color_enabled();
  reset = use_color ? "\033[0m" : "";
  prefix_color = use_color ? "\033[36m" : "";
  alert_color = use_color ? "\033[1;4m" : "";
  line_color = use_color ? "\033[32m" : "";
  header_color = use_color ? "\033[1m" : "";
  load_state();
}

{
  load_state();
  if ($0 ~ /Elapsed_Sec=/) {
    global_done++;
  }

  prefix = make_prefix(global_done, global_total);
  now = systime();
  style = "";
  marker = "";
  if (use_color && (now - last_emit) >= 60) {
    style = alert_color;
    marker = "[!slow!] ";
  }

  if ($0 ~ /^Store=VMemKV\//) {
    printf "%s%s%s%s %s%s%s\n", prefix_color, style, marker, prefix, reset, line_color, $0, reset;
  } else if (use_color) {
    printf "%s%s%s%s %s%s\n", prefix_color, style, marker, prefix, reset, $0;
  } else {
    printf "%s%s %s\n", marker, prefix, $0;
  }
  last_emit = now;
  fflush();
}
PROGRESS_AWK_EOF
}

# run-scenario: runs one Google Benchmark scenario step (bench_kv invocation).
# Single code path for local runs and remote (SSH) runs: the local runner calls
# this function in-process (via `bash run_bench.sh run-scenario`), the AWS runner
# invokes the same subcommand over SSH, optionally inside a systemd-run cgroup.
bench_run_scenario() {
  local BENCH_KV_BIN="$1"
  local SCENARIO_RUN_FILTER="$2"
  local MIN_TIME="$3"
  local SCENARIO_RESULT_PATH="$4"
  # Optional Google Benchmark repetition count for error bars (default 1 = single
  # run, identical to the old invocation shape).
  local REPETITIONS="${5:-1}"

  # Host hardware and OS metadata, resolved wherever this runs (local dev
  # machine or AWS instance).
  export VMEMKV_CONTEXT_kernel_release="$(uname -r)"
  export VMEMKV_CONTEXT_cpu_model="$(vmemkv_cpu_model)"
  export VMEMKV_CONTEXT_cpu_count="$(nproc --all)"
  export VMEMKV_CONTEXT_mem_total_bytes="$(vmemkv_mem_total_bytes)"
  export VMEMKV_CONTEXT_swap_total_bytes="$(vmemkv_swap_total_bytes)"
  # Measurement protocol: what a plotted point means. Single-run cells report one
  # window; repetition runs report per-run values plus mean/median/stddev aggregates.
  export VMEMKV_CONTEXT_min_time_seconds="${MIN_TIME%s}"
  export VMEMKV_CONTEXT_repetitions="${REPETITIONS}"
  # Comparison-backend versions (best effort; empty when not detectable).
  export VMEMKV_CONTEXT_rocksdb_version="$(dpkg-query -W -f='${Version}' librocksdb-dev 2>/dev/null || echo "")"
  export VMEMKV_CONTEXT_lmdb_version="$(dpkg-query -W -f='${Version}' liblmdb-dev 2>/dev/null || echo "")"
  # LeanStore is pinned by commit in the vmemkv CMakeLists; the file's location depends on
  # whether this runs from a checkout (IMPL_ROOT) or the remote repo root (cwd).
  leanstore_pin=""
  for cmake_lists in "$IMPL_ROOT/CMakeLists.txt" "./implementation/vmemkv/CMakeLists.txt"; do
    if [[ -f "$cmake_lists" ]]; then
      leanstore_pin="$(grep -A1 'leanstore/leanstore' "$cmake_lists" 2>/dev/null | grep -oE 'GIT_TAG [^ )]+' | awk '{print $2}' || echo "")"
      [[ -n "$leanstore_pin" ]] && break
    fi
  done
  export VMEMKV_CONTEXT_leanstore_pinned_commit="$leanstore_pin"
  export VMEMKV_CONTEXT_compiler_version="$(c++ --version 2>/dev/null | head -n 1 || echo "")"
  # Paging regime this run was measured under (MGLRU generations on recent kernels,
  # classic LRU otherwise). Read-only observability: never enforced here, so this
  # cannot change behavior. Unknown where the sysfs knob is absent.
  export VMEMKV_CONTEXT_lru_gen_enabled="$(cat /sys/kernel/mm/lru_gen/enabled 2>/dev/null || echo unknown)"

  local run_status=0
  # Unbuffered stdout to prevent lag in systemd-run/SSH pipes.
  if command -v stdbuf >/dev/null 2>&1; then
    stdbuf -oL -eL "$BENCH_KV_BIN" \
      --benchmark_min_time="${MIN_TIME}" \
      --benchmark_filter="${SCENARIO_RUN_FILTER}" \
      --benchmark_repetitions="${REPETITIONS}" \
      --benchmark_out="${SCENARIO_RESULT_PATH}" \
      --benchmark_out_format=json || run_status=$?
  else
    "$BENCH_KV_BIN" \
      --benchmark_min_time="${MIN_TIME}" \
      --benchmark_filter="${SCENARIO_RUN_FILTER}" \
      --benchmark_repetitions="${REPETITIONS}" \
      --benchmark_out="${SCENARIO_RESULT_PATH}" \
      --benchmark_out_format=json || run_status=$?
  fi
  if [[ "$run_status" -ne 0 ]]; then
    return "$run_status"
  fi

  local sanitizer="$BENCH_DIR/sanitize_results.py"
  if [[ ! -f "$sanitizer" ]]; then
    sanitizer="$IMPL_ROOT/benchmark/sanitize_results.py"
  fi
  python3 "$sanitizer" "$SCENARIO_RESULT_PATH"
}

# Shared driver logic for bench_kv's standalone `--reorg-probe` CLI mode, used
# by the probe subcommands below (bg-jobs, organic-split). Holds the
# per-data-point execution/timeout-handling and driver-setup boilerplate.

# Shared probe setup: parses the common <bench_kv_bin> <output_jsonl_path>
# [db_dir] [combo_filter] positional-arg convention, truncates $OUTPUT_PATH,
# and sets $PROBE_LOG_PREFIX (consumed by log() below).
# Sets (global, not local -- callers need these after this function returns): BENCH_KV_BIN,
# OUTPUT_PATH, DB_DIR, COMBO_FILTER, ALL_COMBOS, PROBE_LOG_PREFIX.
#
# Args: $1 = log prefix (e.g. "background-jobs-probe"), $2.. = the probe
# subcommand's own $1..$4 (bench_kv_bin, output_jsonl_path, [db_dir], [combo_filter]).
init_probe_driver() {
  PROBE_LOG_PREFIX="$1"
  shift
  BENCH_KV_BIN="${1:?usage: run_bench.sh probe <name> <bench_kv_binary> <output_jsonl_path> [db_dir] [combo_filter]}"
  OUTPUT_PATH="${2:?usage: run_bench.sh probe <name> <bench_kv_binary> <output_jsonl_path> [db_dir] [combo_filter]}"
  DB_DIR="${3:-/tmp}"
  # Optional 4th arg: restrict which combos run. Two forms:
  #   "in_memory" / "ltm"          -- every combo for that scenario (both its value sizes)
  #   "in_memory:8B" / "ltm:64KB"  -- exactly one (scenario, value_size) combo
  # The scenario-only form exists so a caller that must wrap only the ltm half in a memory cgroup
  # (see the aws subcommand's run_scenario(), which does exactly this for the normal benchmark
  # matrix) can invoke a probe twice -- once unconstrained for in_memory, once
  # systemd-run-wrapped for ltm. The exact-combo form exists so a caller that's already scoped to
  # one (scenario, value_size) pair -- e.g. one of parallel's 4 CRUD-matrix tasks, which each
  # own exactly one such pair -- can run only its own combo instead of duplicating its sibling's.
  # Empty (the default) means "all 4 combos", for local testing.
  COMBO_FILTER="${4:-}"

  ALL_COMBOS=("in_memory:8B" "in_memory:1KB" "ltm:1KB" "ltm:64KB")

  : > "$OUTPUT_PATH"
}

log() { echo "[${PROBE_LOG_PREFIX}] $*" >&2; }

# Runs one --reorg-probe data point, applies two-tier timeout handling, and appends its JSONL
# result line to $OUTPUT_PATH:
#   - bench_kv's own internal kReorgTimeoutSecondsDefault cap on the reorganize()/checkpoint()
#     call itself (overridable via VMEMKV_BENCH_REORG_TIMEOUT_SECONDS, see bench_kv.cpp) -- this
#     is what actually bounds the interesting measurement.
#   - this function's outer OUTER_TIMEOUT_SECONDS, a generous backstop covering setup too (which
#     the internal cap deliberately excludes), in case populate/checkpoint/churn itself hangs.
# Expects $BENCH_KV_BIN, $DB_DIR, and $OUTPUT_PATH (set by init_probe_driver() above) and
# $OUTER_TIMEOUT_SECONDS (a driver-specific constant the caller sets itself) to already be set.
# Extra args (e.g. --churn-ratio=.../--sweep-tag=...) are passed through as $6+. Returns 0 if the
# point completed cleanly (caller should keep escalating), 1 otherwise (caller should stop
# escalating for this sweep).
run_probe_point() {
  local scenario="$1" value_size="$2" mode="$3" ratio="$4" label="$5"
  shift 5
  local stdout_file stderr_file exit_code line reason
  stdout_file="$(mktemp)"
  stderr_file="$(mktemp)"
  VMEMKV_DB_DIR="$DB_DIR" timeout "${OUTER_TIMEOUT_SECONDS}s" "$BENCH_KV_BIN" \
    --reorg-probe --scenario="$scenario" --value-size="$value_size" --mode="$mode" --ratio="$ratio" "$@" \
    >"$stdout_file" 2>"$stderr_file"
  exit_code=$?

  line="$(tail -n 1 "$stdout_file")"
  if [[ $exit_code -eq 0 && -n "$line" ]]; then
    log "${label}: $line"
    if [[ -s "$stderr_file" ]]; then
      log "${label}: stderr: $(cat "$stderr_file")"
    fi
    echo "$line" >> "$OUTPUT_PATH"
    rm -f "$stdout_file" "$stderr_file"
    return 0
  fi

  if [[ -n "$line" ]] && echo "$line" | python3 -c "import json,sys; json.load(sys.stdin)" >/dev/null 2>&1; then
    # The probe's own internal reorganize()/checkpoint()-timeout fired: it still printed a JSON
    # line (timed_out:true) before exiting 124. Trust that over synthesizing our own record.
    log "${label}: internal timeout -- $line"
    if [[ -s "$stderr_file" ]]; then
      log "${label}: stderr: $(cat "$stderr_file")"
    fi
    echo "$line" >> "$OUTPUT_PATH"
  else
    # The outer backstop killed the process before it could print anything (e.g. setup itself
    # never finished), or it crashed/errored some other way.
    reason="outer_timeout"
    if [[ $exit_code -ne 124 ]]; then
      reason="error_exit_${exit_code}"
    fi
    log "${label}: ${reason} (exit=${exit_code}); stderr: $(tail -n 5 "$stderr_file")"
    printf '{"scenario":"%s","value_size":"%s","mode":"%s","ratio":%s,"key_count":null,"elapsed_sec":%s,"timed_out":true,"failure_reason":"%s"}\n' \
      "$scenario" "$value_size" "$mode" "$ratio" "$OUTER_TIMEOUT_SECONDS" "$reason" >> "$OUTPUT_PATH"
  fi
  rm -f "$stdout_file" "$stderr_file"
  return 1
}

probe_bg_jobs() {
init_probe_driver "background-jobs-probe" "$@"
SCENARIO_FILTER="$COMBO_FILTER"

# Generous: populating 10,000,000 records plus a full checkpoint()/reorganize() cycle under LTM
# memory pressure can legitimately take longer than the smaller sweeps the other (retired) probes
# ran -- this is a fixed reference point measured once, not many points in a loop, so there's
# no per-point time budget to protect.
OUTER_TIMEOUT_SECONDS=900

SCENARIOS=("in_memory" "ltm")
if [[ -n "$SCENARIO_FILTER" ]]; then
  # Matches run_remote_probe()'s convention of passing "in_memory"/"ltm" or "in_memory:1KB"/
  # "ltm:1KB" (value-size suffix, if a caller ever adds one -- this probe is fixed at 1KB
  # regardless, so only the scenario half of the filter is meaningful here).
  SCENARIOS=("${SCENARIO_FILTER%%:*}")
fi

for scenario in "${SCENARIOS[@]}"; do
  for job in reorganize checkpoint; do
    log "=== ${scenario}/${job} ==="
    run_probe_point "$scenario" "1KB" "background_job_probe" "1.0" "${scenario}/${job}" --job="$job"
  done
done

log "done. Results written to $OUTPUT_PATH"
}

probe_checkpoint_scale() {
init_probe_driver "checkpoint-scale-probe" "$@"
SCENARIO_FILTER="$COMBO_FILTER"

# Prime (interleaved bulk gens + settle) dominates: 10M keys at 1KB plus per-cycle
# checkpoints take an order of magnitude longer than one bg-jobs point.
OUTER_TIMEOUT_SECONDS=3600

SCENARIOS=("in_memory" "ltm")
if [[ -n "$SCENARIO_FILTER" ]]; then
  SCENARIOS=("${SCENARIO_FILTER%%:*}")
fi

for scenario in "${SCENARIOS[@]}"; do
  # Key-count sweep per scenario (fixed 1KB values, 10 interleaved prime gens,
  # 5 measured cycles each): per-cycle checkpoint phases vs corpus bytes.
  # NOTE: unlike run_probe_point (single-line result), the scale probe prints
  # one JSON line per measured cycle, so this loop appends full stdout.
  key_counts="500000 1000000 2000000"
  chunk_keys="100000"
  if [[ "$scenario" == "ltm" ]]; then
    key_counts="2000000 5000000 10000000"
    chunk_keys="500000"
  fi
  for key_count in $key_counts; do
    log "=== $scenario key_count=$key_count ==="
    stdout_file="$(mktemp)"
    stderr_file="$(mktemp)"
    if VMEMKV_DB_DIR="$DB_DIR" timeout "${OUTER_TIMEOUT_SECONDS}s" "$BENCH_KV_BIN" \
      --reorg-probe --scenario="$scenario" --value-size="1KB" --mode=checkpoint_scale_probe --ratio="1.0" \
      --key-count="$key_count" --gens=10 --cycles=5 --chunk-keys="$chunk_keys" \
      >"$stdout_file" 2>"$stderr_file"; then
      grep -E '^\{"probe":"checkpoint_scale"' "$stdout_file" >> "$OUTPUT_PATH"
      log "${scenario}/keys=${key_count}: $(wc -l < "$stdout_file") cycle lines recorded"
    else
      log "point failed: ${scenario} keys=${key_count} (rc=$?) -- see $stderr_file"
    fi
    rm -f "$stdout_file" "$stderr_file"
  done
done

log "done. Results written to $OUTPUT_PATH"
}

probe_segment_scan() {
  init_probe_driver "segment-scan-probe" "$@"
  if [[ "$COMBO_FILTER" != "ltm:1KB" ]]; then
    log "segment-scan-probe requires the ltm:1KB combo"
    return 1
  fi
  local reverse=""
  if [[ "${5:-}" == *reverse* || -n "${VMEMKV_SEGMENT_SCAN_REVERSE:-}" ]]; then
    reverse=1
  fi
  local abab=""
  if [[ "${5:-}" == *abab* || -n "${VMEMKV_SEGMENT_SCAN_ABAB:-}" ]]; then
    abab=1
  fi
  OUTER_TIMEOUT_SECONDS=3600
  # Counterbalanced order: reverse runs the replacement-key
  # run first, so first-run/second-run effects can be separated from mode effects.
  # ABAB repeats both modes twice alternating, giving within-session A/A controls.
  if [[ -n "$abab" ]]; then
    if [[ -n "$reverse" ]]; then
      local labels=("with_replaced_keys" "baseline" "with_replaced_keys" "baseline")
    else
      local labels=("baseline" "with_replaced_keys" "baseline" "with_replaced_keys")
    fi
    local i=0
    for label in "${labels[@]}"; do
      i=$((i + 1))
      if [[ "$label" == "baseline" ]]; then
        if ! VMEMKV_SEGMENT_PROBE_BASELINE=1 run_probe_point "ltm" "1KB" "scan_update_scan_probe" "1.0" "ltm/baseline_rep${i}"; then
          return 1
        fi
      else
        if ! run_probe_point "ltm" "1KB" "scan_update_scan_probe" "1.0" "ltm/with_replaced_keys_rep${i}"; then
          return 1
        fi
      fi
    done
  elif [[ -n "$reverse" ]]; then
    if ! run_probe_point "ltm" "1KB" "scan_update_scan_probe" "1.0" "ltm/with_replaced_keys"; then
      return 1
    fi
    if ! VMEMKV_SEGMENT_PROBE_BASELINE=1 run_probe_point "ltm" "1KB" "scan_update_scan_probe" "1.0" "ltm/baseline"; then
      return 1
    fi
  else
    if ! VMEMKV_SEGMENT_PROBE_BASELINE=1 run_probe_point "ltm" "1KB" "scan_update_scan_probe" "1.0" "ltm/baseline"; then
      return 1
    fi
    if ! run_probe_point "ltm" "1KB" "scan_update_scan_probe" "1.0" "ltm/with_replaced_keys"; then
      return 1
    fi
  fi
  if ! jq -se '((length == 2 and .[0].mode != .[1].mode) or (length == 4 and ([.[].mode] | sort) == ["baseline","baseline","with_replaced_keys","with_replaced_keys"])) and all(.[]; (.mode == "baseline" or .mode == "with_replaced_keys") and .cache_dropped_before == true and .cache_dropped_after == true and .dirty_shards_after == 0 and .dirty_shards_after_checkpoint == 0 and .pending_segments_after_compaction == 0 and .compaction_running_after_compaction == false and .append_entries_after_checkpoint == 0 and .scan_records_before == .scan_records_after)' \
    "$OUTPUT_PATH" >/dev/null; then
    log "invalid result: cache drop, clean compaction state, or scan record count check failed"
    return 1
  fi
  log "done. Results written to $OUTPUT_PATH"
}

probe_organic_split() {
init_probe_driver "organic-split-probe" "$@"
SCENARIO_FILTER="$COMBO_FILTER"
KEY_PATTERN="${5:-monotonic}"
# Accept both the bare pattern ("random") and the bench_kv flag form ("--key-pattern=random")
# so remote callers can forward the flag verbatim.
KEY_PATTERN="${KEY_PATTERN#--key-pattern=}"

# Generous: the probe itself runs a fixed ~90s (bench_kv.cpp's kOrganicSplitProbeDurationSec) of
# sustained inserts, so there's real headroom needed beyond that alone.
OUTER_TIMEOUT_SECONDS=300

SCENARIOS=("in_memory" "ltm")
if [[ -n "$SCENARIO_FILTER" ]]; then
  # Matches run_remote_probe()'s convention of passing "in_memory"/"ltm" (see its own comment).
  SCENARIOS=("${SCENARIO_FILTER%%:*}")
fi

for scenario in "${SCENARIOS[@]}"; do
  log "=== ${scenario} (${KEY_PATTERN}) ==="
  run_probe_point "$scenario" "1KB" "organic_split_probe" "1.0" "${scenario}/organic_split" \
    --key-pattern="$KEY_PATTERN"
done

log "done. Results written to $OUTPUT_PATH"
}

probe_compaction_churn() {
init_probe_driver "compaction-churn-probe" "$@"
SCENARIO_FILTER="$COMBO_FILTER"
OUTER_TIMEOUT_SECONDS=600

SCENARIOS=("in_memory" "ltm")
if [[ -n "$SCENARIO_FILTER" ]]; then
  SCENARIOS=("${SCENARIO_FILTER%%:*}")
fi

for scenario in "${SCENARIOS[@]}"; do
  log "=== ${scenario} ==="
  stdout_file="$(mktemp)"
  stderr_file="$(mktemp)"
  if VMEMKV_DB_DIR="$DB_DIR" timeout "${OUTER_TIMEOUT_SECONDS}s" "$BENCH_KV_BIN" \
    --reorg-probe --scenario="$scenario" --value-size="1KB" --mode=compaction_churn_probe --ratio="1.0" \
    --duration-sec=60 \
    >"$stdout_file" 2>"$stderr_file"; then
    grep -E '^\{"probe":"compaction_churn"' "$stdout_file" >> "$OUTPUT_PATH"
    log "${scenario}: $(wc -l < "$stdout_file") sample lines recorded"
  else
    log "point failed: ${scenario} (rc=$?) -- see $stderr_file"
  fi
  rm -f "$stdout_file" "$stderr_file"
done

log "done. Results written to $OUTPUT_PATH"
}

aws_clean_resources() {
TARGET_KEY="${1:-}"
# Must match the aws subcommand's KEY_NAME prefix ("vmemkv-i4i-key-$$"), or the no-TARGET_KEY
# fallback below can't find or clean up any current run's resources.
KEY_PREFIX="vmemkv-i4i-key-"

echo "==========================================================="
echo "   Cleaning up vmemkv Temporary AWS Benchmark Resources"
if [[ -n "$TARGET_KEY" ]]; then
  echo "   Target Key: $TARGET_KEY"
else
  echo "   Target Key: ALL temporary keys"
fi
echo "==========================================================="

# 1. Terminate EC2 instances associated with the benchmark keypair
# Cleanup requires an authenticated AWS API session. The local state file remains available
# when authentication is unavailable.
if ! aws sts get-caller-identity --query Account --output text >/dev/null 2>&1; then
  echo "[WARN] AWS auth expired -- skipping cleanup, resources left running." >&2
  if [[ -n "$TARGET_KEY" && -f "/tmp/vmemkv_aws_state_${TARGET_KEY}.json" ]]; then
    echo "[WARN] Resume after re-auth: aws login && benchmark/run_bench.sh clean ${TARGET_KEY}" >&2
  else
    echo "[WARN] Resume after re-auth: aws login && benchmark/run_bench.sh clean" >&2
  fi
  return 1
fi

# The launch state file identifies the instance without a key-name search.
STATE_FILE=""
if [[ -n "$TARGET_KEY" && -f "/tmp/vmemkv_aws_state_${TARGET_KEY}.json" ]]; then
  STATE_FILE="/tmp/vmemkv_aws_state_${TARGET_KEY}.json"
fi
if [[ -n "$STATE_FILE" ]]; then
  INSTANCE_ID=$(jq -r '.instance_id' "$STATE_FILE")
  INSTANCES=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --filters "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query "Reservations[*].Instances[*].InstanceId" \
    --output text)
  [[ "$INSTANCES" == "None" ]] && INSTANCES=""
else
  if [[ -n "$TARGET_KEY" ]]; then
    INSTANCES=$(aws ec2 describe-instances \
      --filters "Name=key-name,Values=${TARGET_KEY}" "Name=instance-state-name,Values=pending,running,stopping,stopped" \
      --query "Reservations[*].Instances[*].InstanceId" \
      --output text)
  else
    INSTANCES=$(aws ec2 describe-instances \
      --filters "Name=key-name,Values=${KEY_PREFIX}*" "Name=instance-state-name,Values=pending,running,stopping,stopped" \
      --query "Reservations[*].Instances[*].InstanceId" \
      --output text)
  fi
fi

if [ -n "$INSTANCES" ]; then
  echo "Found running benchmark instances: $INSTANCES"
  echo "Terminating instances..."
  aws ec2 terminate-instances --instance-ids $INSTANCES > /dev/null
  echo "Waiting for instances to be terminated..."
  aws ec2 wait instance-terminated --instance-ids $INSTANCES
  echo "Instances terminated successfully."
else
  echo "No active benchmark EC2 instances found."
fi

# 2. Delete temporary security groups
if [[ -n "$TARGET_KEY" ]]; then
  SGS=$(aws ec2 describe-security-groups \
    --filters "Name=tag:VMemKV-Temp-Spot,Values=${TARGET_KEY}" \
    --query "SecurityGroups[*].GroupId" \
    --output text)
else
  SGS=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=vmemkv-i4i-sg-*" \
    --query "SecurityGroups[*].GroupId" \
    --output text)
fi

if [ -n "$SGS" ]; then
  echo "Found temporary security groups: $SGS"
  for sg in $SGS; do
    echo "Deleting security group $sg..."
    # Retry with a generous budget: the ENI attached to this SG normally releases within
    # seconds of `aws ec2 wait instance-terminated` returning, but that wait is only in the
    # code path where *we* call terminate-instances above. When the instance instead vanished
    # on its own (e.g. a spot capacity reclaim -- "instance-terminated-no-capacity" -- observed
    # in practice to leave orphaned SGs behind with the previous 5x3s=15s budget), this is the
    # only wait covering ENI teardown, so it needs real margin, not a token retry.
    deleted=false
    last_err=""
    for i in {1..20}; do
      if last_err=$(aws ec2 delete-security-group --group-id "$sg" 2>&1); then
        echo "Deleted security group $sg."
        deleted=true
        break
      fi
      echo "Waiting for security group dependencies to clear (retry $i/20)..."
      sleep 5
    done
    if [[ "$deleted" == "false" ]]; then
      echo "[WARNING] Failed to delete security group $sg after 20 retries (100s) -- leaving it" >&2
      echo "[WARNING] behind as an orphaned resource. Last error: $last_err" >&2
      echo "[WARNING] Clean it up manually: aws ec2 delete-security-group --group-id $sg" >&2
    fi
  done
else
  echo "No temporary security groups found."
fi

# 3. Delete key pairs
if [[ -n "$TARGET_KEY" ]]; then
  KEY_PAIRS=""
  if aws ec2 describe-key-pairs --key-names "$TARGET_KEY" >/dev/null 2>&1; then
    KEY_PAIRS="$TARGET_KEY"
  fi
else
  KEY_PAIRS=$(aws ec2 describe-key-pairs \
    --query "KeyPairs[?starts_with(KeyName, '${KEY_PREFIX}')].KeyName" \
    --output text)
fi

if [ -n "$KEY_PAIRS" ]; then
  echo "Found temporary key pairs: $KEY_PAIRS"
  for kp in $KEY_PAIRS; do
    echo "Deleting AWS Key Pair $kp..."
    aws ec2 delete-key-pair --key-name "$kp"
  done
else
  echo "No temporary AWS key pairs found."
fi

if [[ -n "${STATE_FILE:-}" ]]; then
  rm -f "$STATE_FILE"
fi

# 4. Clean up local key files in /tmp
echo "Cleaning up local temporary key files..."
if [[ -n "$TARGET_KEY" ]]; then
  rm -f "/tmp/${TARGET_KEY}.pem"
else
  rm -f /tmp/vmemkv-i4i-key-*.pem
fi

echo "==========================================================="
echo "   AWS Clean Up Completed Successfully!"
echo "==========================================================="
}

# local: in-process run on this machine (build, measure, save JSON).

cmd_local() {
# ── Defaults ────────────────────────────────────────────────────────────
ENABLE_ROCKSDB=ON
ENABLE_LMDB=ON
ENABLE_LEANSTORE=ON
BUILD_TYPE=Release
BUILD_DIR=build
OUTPUT_FILE=""
AUTO_LOG=true
MIN_TIME="5.0s"
# Three Google Benchmark repetitions produce mean/median/stddev aggregates in the result JSON.
# Priming and probe passes use one repetition.
MATRIX_REPETITIONS=3
LARGE_VALUE_FIRST=false
SCENARIO_LIMIT="in_memory"
ORDER="A_FIRST"
LTM_MEMORY_BUDGET_BYTES="$(vmemkv_matrix::ltm_memory_budget_bytes)"
USE_CGROUP=false

# ── Argument Parsing ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-rocksdb)   ENABLE_ROCKSDB=OFF; shift ;;
    --no-lmdb)      ENABLE_LMDB=OFF;    shift ;;
    --no-leanstore) ENABLE_LEANSTORE=OFF; shift ;;
    --build-type)   BUILD_TYPE="$2";    shift 2 ;;
    --build-dir)    BUILD_DIR="$2";     shift 2 ;;
    --min-time)     MIN_TIME="$2";      shift 2 ;;
    --repetitions)  MATRIX_REPETITIONS="$2"; shift 2 ;;
    --output)       OUTPUT_FILE="$2";   shift 2 ;;
    --no-log)       AUTO_LOG=false;     shift ;;
    --scenario)     SCENARIO_LIMIT="$2"; shift 2 ;;
    --ltm-first)    ORDER="B_FIRST";    shift ;;
    --ltm-memory-budget-bytes) LTM_MEMORY_BUDGET_BYTES="$2"; shift 2 ;;
    --cgroup)       USE_CGROUP=true;    shift ;;
    --large-value-first|--large_value-first)
      LARGE_VALUE_FIRST=true
      shift
      ;;
    -h|--help)
      echo "Usage: run_bench.sh local [options]"
      echo "  --scenario <S>    in_memory | ltm | all (default: in_memory)"
      echo "  --ltm-first       With --scenario all, run LTM before in-memory"
      echo "  --no-rocksdb      Disable RocksDB build"
      echo "  --no-lmdb         Disable LMDB build"
      echo "  --no-leanstore    Disable LeanStore build"
      echo "  --build-type <T>  CMake build type (default: Release)"
      echo "  --build-dir <D>   Build directory (default: build)"
      echo "  --min-time <S>    Minimum measurement time (default: 5.0s)"
      echo "  --repetitions <N> Google Benchmark repetitions per matrix cell (default: 3)"
      echo "  --large-value-first  Run larger-value benchmarks before smaller-value benchmarks"
      echo "  --ltm-memory-budget-bytes <N>  Declared LTM memory budget fed to bench_kv"
      echo "                    (default: same 1GiB the AWS runner uses; combined with"
      echo "                    the fixed LTM target_ratio this sizes the LTM corpus the"
      echo "                    same way regardless of this machine's real RAM)"
      echo "  --cgroup          Best-effort: wrap the LTM scenario in 'systemd-run --user'"
      echo "                    with the same MemoryHigh/MemoryMax as the AWS runner, to"
      echo "                    also reproduce real memory pressure (not just corpus"
      echo "                    sizing). Requires a systemd user session with working"
      echo "                    cgroup delegation -- silently has no effect otherwise, so"
      echo "                    don't rely on it to prove swap behavior without checking."
      echo "  --output <F>      Save raw JSON output to file"
      echo "  --no-log          Disable automatic logging"
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

case "$SCENARIO_LIMIT" in
  in_memory|ltm|all) ;;
  *) echo "Invalid --scenario: $SCENARIO_LIMIT (expected in_memory|ltm|all)" >&2; exit 1 ;;
esac

cd "$IMPL_ROOT"

if [[ -t 1 ]]; then
  export VMEMKV_COLOR=1
fi

if [[ "$LARGE_VALUE_FIRST" == "true" ]]; then
  export VMEMKV_BENCH_LARGE_VALUE_FIRST=1
fi

# ── RocksDB Check ────────────────────────────────────────────────────────
if [[ "$ENABLE_ROCKSDB" == "ON" ]]; then
  if ! (pkg-config --exists rocksdb 2>/dev/null || \
        [[ -f /opt/homebrew/lib/librocksdb.dylib ]] || \
        [[ -f /usr/local/lib/librocksdb.so ]] || \
        [[ -f /usr/lib/librocksdb.so ]]); then
    ENABLE_ROCKSDB=OFF
  fi
fi

# ── LMDB Check ───────────────────────────────────────────────────────────
if [[ "$ENABLE_LMDB" == "ON" ]]; then
  if ! (pkg-config --exists lmdb 2>/dev/null || \
        [[ -f /opt/homebrew/lib/liblmdb.dylib ]] || \
        [[ -f /usr/local/lib/liblmdb.so ]] || \
        [[ -f /usr/lib/x86_64-linux-gnu/liblmdb.so ]] || \
        [[ -f /usr/lib/liblmdb.so ]]); then
    ENABLE_LMDB=OFF
  fi
fi

# ── CMake Configure & Build ──────────────────────────────────────────────
echo "Building benchmark binary..."
vmemkv_build_bench "$IMPL_ROOT" "$BUILD_DIR" "$BUILD_TYPE" "$ENABLE_ROCKSDB" -DENABLE_LMDB="$ENABLE_LMDB" -DENABLE_LEANSTORE="$ENABLE_LEANSTORE" >/dev/null
echo "Build completed."

BENCH_BIN="$IMPL_ROOT/$BUILD_DIR/benchmark/bench_kv"

# ── Auto log file setup ──────────────────────────────────────────────────
LOG_DIR="$BENCH_DIR/logs"
if [[ "$AUTO_LOG" == "true" ]]; then
  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  ROCKSDB_SUFFIX="$([ "$ENABLE_ROCKSDB" == "ON" ] && echo "rocksdb" || echo "no_rocksdb")"
  mkdir -p "$LOG_DIR"
fi

read -r -a SCENARIO_ORDER_KEYS <<<"$(vmemkv_matrix::scenario_order_keys_from_flag "$ORDER")"

run_local_scenario() {
  local scenario_key="$1"
  local scenario_env_prefix
  local scenario_runtime_env=""
  local filter
  local temp_json
  local progress_state
  local auto_log_file=""

  scenario_env_prefix="$(vmemkv_matrix::scenario_env_prefix "$scenario_key")"

  if [[ "$scenario_key" == "in_memory" ]]; then
    # Same reasoning as the AWS runner: guarantee the in-memory corpus is sized
    # off real host RAM regardless of any ambient cgroup the shell happens to
    # be running under.
    scenario_runtime_env="VMEMKV_BENCH_FORCE_HOST_MEMORY=1"
  fi

  local memory_budget_source=""
  local memory_budget_bytes=""
  local memo=""
  if [[ "$scenario_key" == "ltm" ]]; then
    memory_budget_source="declared"
    memory_budget_bytes="$LTM_MEMORY_BUDGET_BYTES"
    memo="Local LTM run uses the same declared memory budget as the AWS runner ($LTM_MEMORY_BUDGET_BYTES bytes) to size the corpus consistently; real memory pressure additionally requires --cgroup with working cgroup delegation."
    scenario_runtime_env="VMEMKV_CONTEXT_memory_budget_bytes=$memory_budget_bytes"
  fi

  filter="$(run_bench::scenario_filter "$scenario_key")"

  local scenario_order_label value_order_keys value_order_label target_profile
  scenario_order_label="$(vmemkv_scenario_order_label "$scenario_key")"
  read -r -a value_order_keys <<<"$(vmemkv_matrix::scenario_value_order_keys_from_flag "$scenario_key" "$LARGE_VALUE_FIRST")"
  value_order_label="$(vmemkv_value_order_label "${value_order_keys[@]}")"
  target_profile="$(vmemkv_target_profile_label "$scenario_key")"

  local benchmark_flags="large_value_first=$([[ "$LARGE_VALUE_FIRST" == "true" ]] && printf '1' || printf '0')"
  declare -a context_env=()
  vmemkv_context_env_array context_env \
    "local" \
    "$BUILD_TYPE" \
    "$BUILD_DIR" \
    "$ENABLE_ROCKSDB" \
    "$scenario_order_label" \
    "$benchmark_flags" \
    "$memory_budget_source" \
    "$memory_budget_bytes" \
    "host" \
    "" \
    "$memo"

  echo "=== Scenario: $scenario_key (target_ratio=$target_profile) ==="

  progress_state="$(mktemp /tmp/vmemkv_local_progress.XXXXXX)"
  temp_json="$(mktemp /tmp/vmemkv_bench_results.XXXXXX.json)"
  # shellcheck disable=SC2064
  trap "rm -f '$progress_state' '$temp_json'" RETURN

  local -a bench_stdout_wrapper=()
  if command -v stdbuf >/dev/null 2>&1; then
    bench_stdout_wrapper=(stdbuf -oL -eL)
  fi

  echo "Counting benchmarks..."
  local total_benchmarks
  total_benchmarks="$(env $scenario_env_prefix ${scenario_runtime_env:+$scenario_runtime_env} "$BENCH_BIN" --benchmark_list_tests --benchmark_filter="${filter}" 2>/dev/null | awk 'NF { c++ } END { print c + 0 }')"
  if [[ "$total_benchmarks" -eq 0 ]]; then
    echo "Failed to match any benchmarks against regex: ${filter}" >&2
    return 1
  fi
  echo "Running ${total_benchmarks} benchmark(s)..."
  echo "Filter: ${filter}"
  echo "$total_benchmarks" >"$progress_state"

  local -a scenario_cmd=(bash "$BENCH_SELF" run-scenario "$BENCH_BIN" "$filter" "$MIN_TIME" "$temp_json" "$MATRIX_REPETITIONS")
  if [[ "$USE_CGROUP" == "true" && "$scenario_key" == "ltm" ]]; then
    if command -v systemd-run >/dev/null 2>&1; then
      # Both the priming pass below and the cgroup-wrapped measurement pass after it run with
      # VMEMKV_BENCH_SKIP_CLEANUP=1 (see its comment in bench_kv.cpp), so bench_kv's own
      # automatic sweeps don't fight a deliberate multi-store, multi-master-generation priming
      # pass. That means *this* script owns the "start from a clean slate" step neither of those
      # invocations will do on their own -- do it once, here, unconditionally, exactly like
      # bench_kv's own startup sweep normally would (same "bench_" prefix, same directory: this
      # script never sets VMEMKV_DB_DIR, so bench_kv defaults to ".", i.e. $IMPL_ROOT after the
      # `cd "$IMPL_ROOT"` above).
      find "$IMPL_ROOT" -maxdepth 1 -name 'bench_*' -exec rm -rf {} +

      # Prime every shared master corpus this run will need at full, unconstrained disk speed
      # *before* entering the cgroup below. Without this, a real per-master populate under this
      # exact cgroup was measured at 200-1200s (vs. ~20-30s unconstrained) -- paying that cost
      # here means the cgroup-wrapped measurement pass only ever needs to clone, not rebuild,
      # those masters. See vmemkv_matrix::ltm_priming_filter()'s comment for what this
      # single-cell-per-(store,value size) filter does and does not cover.
      echo "Priming LTM master corpora (unconstrained) before entering the cgroup..."
      local priming_filter priming_json
      priming_filter="$(vmemkv_matrix::ltm_priming_filter "${WITHOUT_RIVALS:-false}" "${LEANSTORE_ONLY:-false}" "${ROCKSDB_ONLY:-false}")"
      if [[ -n "${VMEMKV_RUN_FILTER_OVERRIDE:-}" ]]; then
        priming_filter="$(env $scenario_env_prefix "$BENCH_BIN" --benchmark_list_tests --benchmark_filter="$VMEMKV_RUN_FILTER_OVERRIDE" 2>/dev/null | vmemkv_matrix::ltm_priming_filter_from_names)"
      fi
      priming_json="$(mktemp /tmp/vmemkv_ltm_priming.XXXXXX.json)"
      env $scenario_env_prefix ${scenario_runtime_env:+$scenario_runtime_env} VMEMKV_BENCH_SKIP_CLEANUP=1 \
        bash "$BENCH_SELF" run-scenario "$BENCH_BIN" "$priming_filter" "1s" "$priming_json" \
        >/dev/null
      rm -f "$priming_json"
      echo "Priming complete."

      # The measurement pass below reuses the masters priming just built -- it must not let
      # bench_kv's own startup cleanup_stale_benchmark_files() wipe them the instant it starts.
      scenario_runtime_env="${scenario_runtime_env:+$scenario_runtime_env }VMEMKV_BENCH_SKIP_CLEANUP=1"

      local ltm_swap_budget
      ltm_swap_budget="$(vmemkv_matrix::ltm_swap_budget_bytes)"
      scenario_cmd=(systemd-run --user --scope --quiet \
        -p "MemoryHigh=${LTM_MEMORY_BUDGET_BYTES}" \
        -p "MemoryMax=$((LTM_MEMORY_BUDGET_BYTES * 2))" \
        -p "MemorySwapMax=${ltm_swap_budget}" \
        -- "${scenario_cmd[@]}")
      echo "NOTE: --cgroup requested systemd-run --user with MemoryHigh=${LTM_MEMORY_BUDGET_BYTES}," \
           "MemoryMax=$((LTM_MEMORY_BUDGET_BYTES * 2)). Whether this actually constrains RSS" \
           "depends on this host's cgroup delegation (verify with e.g. 'systemctl --user status');" \
           "in sandboxed/nested containers it is commonly a no-op."
    else
      echo "NOTE: --cgroup requested but systemd-run is not available; running without a memory cgroup." >&2
    fi
  fi

  set +e
  env "${context_env[@]}" $scenario_env_prefix ${scenario_runtime_env:+$scenario_runtime_env} "${bench_stdout_wrapper[@]}" \
    "${scenario_cmd[@]}" 2>&1 \
    | awk -v state_file="$progress_state" -f "$PROGRESS_AWK"
  local bench_status=${PIPESTATUS[0]}
  set -e

  if [[ "$bench_status" -ne 0 ]]; then
    return "$bench_status"
  fi

  if [[ ! -s "$temp_json" ]]; then
    echo "Benchmark produced no JSON output. Check the benchmark filter: ${filter}" >&2
    return 1
  fi

  if [[ "$AUTO_LOG" == "true" ]]; then
    auto_log_file="$LOG_DIR/${TIMESTAMP}_${ROCKSDB_SUFFIX}_${scenario_key}.json"
    cp "$temp_json" "$auto_log_file"
    echo "JSON saved to: $auto_log_file"
  fi

  if [[ -n "$OUTPUT_FILE" ]]; then
    local scenario_output="$OUTPUT_FILE"
    if [[ "$SCENARIO_LIMIT" == "all" ]]; then
      scenario_output="${OUTPUT_FILE%.json}_${scenario_key}.json"
    fi
    mkdir -p "$(dirname "$scenario_output")"
    cp "$temp_json" "$scenario_output"
    echo "JSON saved to: $scenario_output"
  fi
}

run_bench::each_scenario run_local_scenario
}

# aws: full-matrix run on an EC2 spot instance (provision, build, measure, download).


cmd_aws() {
# ── Parse Command Line Arguments ──────────────────────────────────────────
ORDER="A_FIRST"
LARGE_VALUE_FIRST=false

show_help() {
  echo "Usage: run_bench.sh aws [options]"
  echo "Options:"
  echo "  -h, --help       Show this help message and exit"
  echo "  --ltm-first      Run Scenario B (Larger-than-Memory) before Scenario A (In-Memory)"
  echo "  --large-value-first  Run larger-value benchmarks before smaller-value benchmarks"
  echo "  --scenario in_memory|ltm|all  Limit which scenario(s) run (default: all)"
  echo "  --value-size SIZE  Limit to one value-size combo within the active scenario(s)"
  echo "                   (e.g. 8B, 1KB, 64KB)"
  echo "  --ltm-mglru-mode MODE  LTM paging policy: on (default), split, off, or both"
  echo "                   both runs counterbalanced ON/OFF pairs on one instance"
  echo "  --ltm-mglru-pairs N  ON/OFF pairs for mode=both (default 1, range 1..5)"
  echo "  --repetitions N  Google Benchmark repetitions per matrix cell (default: 3)."
  echo "                   3 gives mean/median/stddev aggregates in the result JSON."
  echo "  --background-jobs-probe  After the normal matrix, additionally measure"
  echo "                   reorganize()/checkpoint() (fixed 1KB/10,000,000-record corpus): each"
  echo "                   job's own duration plus the QPS degradation it causes to concurrent"
  echo "                   Insert/Update/Scan on the same instance via"
  echo "                   run_bench.sh probe bg-jobs and download its JSONL output. This is a"
  echo "                   forced, whole-store operation -- not what happens during ordinary"
  echo "                   operation; see --organic-split-probe for that."
  echo "  --checkpoint-scale-probe  After the normal matrix, additionally measure per-cycle"
  echo "                   checkpoint/flush/merge cost vs corpus size (checkpoint_scale_probe):"
  echo "                   interleaved-prime steady-state store, 5 measured insert+checkpoint+"
  echo "                   compact cycles at 3 corpus sizes per scenario (1KB fixed), JSONL output."
  echo "                   Same non-fatal-failure convention as --background-jobs-probe."
  echo "  --segment-scan-probe  After the normal matrix, run Scan -> Zipf Update -> Scan"
  echo "                   on one 1KB LTM store, dropping page cache before both Scan phases."
  echo "  --compaction-churn-probe  After the normal matrix, measure punch backlog and physical"
  echo "                   disk convergence under sustained Zipf churn."
  echo "  --organic-split-probe  After the normal matrix, additionally measure the Insert-QPS"
  echo "                   impact of ShardedT1Index's own automatic per-shard background"
  echo "                   splitting under sustained write load (the maintenance path that"
  echo "                   actually runs during ordinary operation) via"
  echo "                   run_bench.sh probe organic-split and download its JSONL output."
  echo "  --organic-split-probe-random"
  echo "                   Same as --organic-split-probe but with random keys (writes spread"
  echo "                   over all shards), showing whether one split's throughput impact"
  echo "                   shrinks as the shard count grows. Downloaded separately as"
  echo "                   organic_split_random_{in_memory,ltm}.jsonl."
  echo "  --skip-matrix    Skip the main Google Benchmark-registered CRUD/Scan/YCSB-E matrix"
  echo "                   entirely (and its results download) -- provision/build the instance"
  echo "                   and run only the *-probe flags passed alongside this one."
  echo "                   For a standalone probe-only run when the matrix has already been"
  echo "                   measured separately and only a probe (e.g. --background-jobs-probe)"
  echo "                   is still needed."
  echo "  --filter PATTERN  Override benchmark filter (e.g. 'Store=VMemKV.*1024B.*Op=Insert.*threads:32')"
  echo "                   for targeted micro-benchmarks or fast iteration"
  echo "  --without-rivals  Drop RocksDB/RocksDB-BlobDB/LMDB/LeanStore from the benchmark filter, running"
  echo "                   vmemkv variants only. Use for regression-check runs after a"
  echo "                   vmemkv-internal-only code change, where the rivals' numbers are"
  echo "                   unaffected. The output contains only VMemKV variants."
  echo "  --leanstore-only  Run LeanStore cells only (for re-measuring the rival backend after a"
  echo "                   rival-side change). The output contains only LeanStore cells. Cannot be"
  echo "                   combined with --without-rivals."
  echo "  --ycsb-only      Run YCSB-E (Zipf) cells only, vmemkv + RocksDB variants."
  echo "                   Same MIN_TIME handling as a normal run (no implicit 30s bump)."
  exit 0
}

SCENARIO_LIMIT="all"
VALUE_SIZE_LIMIT=""
# Google Benchmark repetitions per matrix cell.
MATRIX_REPETITIONS=3
BACKGROUND_JOBS_PROBE=false
ORGANIC_SPLIT_PROBE=false
ORGANIC_SPLIT_PROBE_RANDOM=false
CHECKPOINT_SCALE_PROBE=false
SEGMENT_SCAN_PROBE=false
COMPACTION_CHURN_PROBE=false
ABLATION_PROBE=false
SKIP_MATRIX=false
WITHOUT_RIVALS=false
LEANSTORE_ONLY=false
YCSB_ONLY=false
ROCKSDB_ONLY=false
LTM_MGLRU_MODE="on"
LTM_MGLRU_PAIRS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      show_help
      ;;
    --ltm-first)
      ORDER="B_FIRST"
      shift
      ;;
    --large-value-first|--large_value-first)
      LARGE_VALUE_FIRST=true
      shift
      ;;
    --scenario)
      SCENARIO_LIMIT="$2"
      shift 2
      ;;
    --value-size)
      VALUE_SIZE_LIMIT="$2"
      shift 2
      ;;
    --ltm-mglru-mode)
      LTM_MGLRU_MODE="$2"
      case "$LTM_MGLRU_MODE" in
        split|on|off|both) ;;
        *) echo "[ERROR] Invalid --ltm-mglru-mode: $LTM_MGLRU_MODE (expected split|on|off|both)" >&2; exit 1 ;;
      esac
      shift 2
      ;;
    --ltm-mglru-pairs)
      if ! [[ "$2" =~ ^[1-5]$ ]]; then
        echo "[ERROR] Invalid LTM MGLRU pair count: $2 (expected 1..5)" >&2
        exit 1
      fi
      LTM_MGLRU_PAIRS="$2"
      shift 2
      ;;
    --repetitions)
      MATRIX_REPETITIONS="$2"
      shift 2
      ;;
    --background-jobs-probe)
      BACKGROUND_JOBS_PROBE=true
      shift
      ;;
    --organic-split-probe)
      ORGANIC_SPLIT_PROBE=true
      shift
      ;;
    --organic-split-probe-random)
      ORGANIC_SPLIT_PROBE_RANDOM=true
      shift
      ;;
    --checkpoint-scale-probe)
      CHECKPOINT_SCALE_PROBE=true
      shift
      ;;
    --segment-scan-probe)
      SEGMENT_SCAN_PROBE=true
      shift
      ;;
    --compaction-churn-probe)
      COMPACTION_CHURN_PROBE=true
      shift
      ;;
    --ablation-probe)
      ABLATION_PROBE=true
      shift
      ;;
    --skip-matrix)
      SKIP_MATRIX=true
      shift
      ;;
    --without-rivals)
      WITHOUT_RIVALS=true
      shift
      ;;
    --leanstore-only)
      LEANSTORE_ONLY=true
      shift
      ;;
    --filter)
      VMEMKV_RUN_FILTER_OVERRIDE="$2"
      export VMEMKV_RUN_FILTER_OVERRIDE
      shift 2
      ;;
    --partition)
      VMEMKV_GROUP_PARTITION="$2"
      export VMEMKV_GROUP_PARTITION
      shift 2
      ;;
    --rocksdb-only)
      ROCKSDB_ONLY=true
      shift
      ;;
    --ycsb-only)
      YCSB_ONLY=true
      shift
      ;;
    *)
      echo "[ERROR] Unknown option: $1" >&2
      echo "Use -h or --help to see available options." >&2
      exit 1
      ;;
  esac
done

if [[ "$WITHOUT_RIVALS" == "true" && "$LEANSTORE_ONLY" == "true" ]]; then
  echo "[ERROR] --without-rivals and --leanstore-only are mutually exclusive." >&2
  exit 1
fi

if [[ "$LEANSTORE_ONLY" == "true" && "${VALUE_SIZE_LIMIT,,}" == "64kb" ]]; then
  echo "[ERROR] --leanstore-only cannot run 64KB: LeanStore's BTreeVI key/value lengths are u16." >&2
  exit 1
fi

# Same strictness as run_bench.sh: an unknown --scenario used to silently run the
# whole matrix here (no branch matched, the else ran everything). Fail fast instead.
case "$SCENARIO_LIMIT" in
  in_memory|ltm|all) ;;
  *) echo "[ERROR] Invalid --scenario: $SCENARIO_LIMIT (expected in_memory|ltm|all)" >&2; exit 1 ;;
esac

# Preserve the original terminal fds before redirecting everything through the
# timestamping pipe. Signal handlers must not depend on that pipe because it
# can be broken precisely when Ctrl+C is delivered.
exec 3>&1 4>&2

if [[ -t 1 ]]; then
  export VMEMKV_COLOR=1
fi

if [[ "$LARGE_VALUE_FIRST" == "true" ]]; then
  export VMEMKV_BENCH_LARGE_VALUE_FIRST=1
fi

# ── Configuration Parameters ──────────────────────────────────────────────
AWS_REGION="ap-northeast-1"
# i4i.8xlarge (32 vCPU, local NVMe): storage-optimized family sized for the 1/4/16/32-thread sweep and the LTM/swap workload.
INSTANCE_TYPE="i4i.8xlarge"
# Pin the spot request to one AZ; re-check placement scores and spot-price history before assuming this choice still holds.
PREFERRED_AZ="ap-northeast-1d"
# MIN_TIME honors the environment; default 5.0s.
MIN_TIME="${MIN_TIME:-5.0s}"
LTM_MEMORY_BUDGET_BYTES="$(vmemkv_matrix::ltm_memory_budget_bytes)"
LTM_SWAP_BUDGET_BYTES="$(vmemkv_matrix::ltm_swap_budget_bytes)"
KEY_NAME="vmemkv-i4i-key-$$"
PEM_FILE="/tmp/${KEY_NAME}.pem"
SG_NAME="vmemkv-i4i-sg-$$"
PORT=22
SIGNAL_LOG="/tmp/vmemkv_signal_${KEY_NAME}.log"
PROGRESS_STATE="/tmp/vmemkv_progress_state_${KEY_NAME}.txt"
exec > >(awk -v state_file="$PROGRESS_STATE" -f "$PROGRESS_AWK") 2>&1

# ── SSH Security Settings ─────────────────────────────────────────────────
export AWS_DEFAULT_REGION="$AWS_REGION"
MY_IP=$(curl -s https://checkip.amazonaws.com)
if [[ -z "$MY_IP" ]]; then
  echo "[ERROR] Could not determine your local IP address." >&2
  exit 1
fi
echo "Your Local IP: $MY_IP"

# ── Cleanup Hook for Temporary Resources ──────────────────────────────────
AWS_CLEANUP_RAN=false

write_progress_state() {
  local global_total="$1"
  echo "$global_total" >"$PROGRESS_STATE"
}

count_remote_benchmarks() {
  local benchmark_filter="$1"
  local env_prefix="${2:-}"
  local runtime_prefix="${3:-}"
  ssh $SSH_OPTS "ubuntu@$PUBLIC_IP" "bash -lc 'cd ~/vmemkv && ${env_prefix:+${env_prefix} }${runtime_prefix:+${runtime_prefix} }./build-rel/benchmark/bench_kv --benchmark_list_tests --benchmark_filter=\"${benchmark_filter}\" 2>/dev/null | awk \"NF { c++ } END { print c + 0 }\"'" | tail -n 1
}

signal_log() {
  local message="$1"
  local line="[signal] $message"
  printf '%s\n' "$line" >>"$SIGNAL_LOG" || true
  printf '%s\n' "$line" >&4 || true
}

run_aws_cleanup() {
  local signal_name="${1:-EXIT}"

  if [[ "$AWS_CLEANUP_RAN" == "false" ]]; then
    AWS_CLEANUP_RAN=true
    signal_log "starting aws_clean.sh signal=$signal_name pid=$$ bashtid=$BASHPID"
    if [[ "$signal_name" == "EXIT" ]]; then
      aws_clean_resources "$KEY_NAME" || true
    else
      local cleanup_log="/tmp/vmemkv_aws_clean_${KEY_NAME}.log"
      nohup bash "$BENCH_SELF" clean "$KEY_NAME" >"$cleanup_log" 2>&1 < /dev/null &
      local cleanup_pid=$!
      signal_log "cleanup continued in background pid=$cleanup_pid log=$cleanup_log"
      wait "$cleanup_pid" || true
      signal_log "cleanup helper finished pid=$cleanup_pid signal=$signal_name"
    fi
  fi
}

on_signal() {
  set +e
  local signal_name="$1"
  local exit_code=130
  if [[ "$signal_name" == "TERM" ]]; then
    exit_code=143
  fi
  # Ignore further interrupts while cleanup is running; otherwise Ctrl+C can
  # interrupt the cleanup path itself before resources are released.
  signal_log "caught signal=$signal_name pid=$$ bashtid=$BASHPID exit_code=$exit_code"
  trap '' INT TERM
  trap - EXIT
  run_aws_cleanup "$signal_name"
  exit "$exit_code"
}

trap 'run_aws_cleanup EXIT' EXIT
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM

create_ssh_key_pair() {
  echo "Creating SSH Key Pair..."
  if [[ -e "$PEM_FILE" ]]; then
    echo "[ERROR] $PEM_FILE already exists -- refusing to overwrite (stale state from a previous run, or re-entrant provisioning)" >&2
    exit 1
  fi
  aws ec2 create-key-pair --key-name "$KEY_NAME" --query "KeyMaterial" --output text > "$PEM_FILE"
  chmod 400 "$PEM_FILE"
}

resolve_ubuntu_ami() {
  echo "Locating latest Ubuntu 24.04 LTS AMI..."
  AMI_ID=$(aws ec2 describe-images \
    --owners 099720109477 \
    --filters "Name=name,Values=ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*" \
              "Name=state,Values=available" \
              "Name=architecture,Values=x86_64" \
    --query "reverse(sort_by(Images, &CreationDate))[0].ImageId" \
    --output text)
  echo "Using AMI ID: $AMI_ID"
}

create_security_group() {
  echo "Creating temporary Security Group..."
  VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query "Vpcs[0].VpcId" --output text)
  SG_ID=$(aws ec2 create-security-group \
    --group-name "$SG_NAME" \
    --description "vmemkv Temporary Benchmark SG" \
    --vpc-id "$VPC_ID" \
    --query "GroupId" \
    --output text)

  aws ec2 create-tags --resources "$SG_ID" --tags "Key=VMemKV-Temp-Spot,Value=$KEY_NAME" >/dev/null

  echo "Authorizing SSH ingress access from your IP ($MY_IP)..."
  aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" \
    --protocol tcp \
    --port "$PORT" \
    --cidr "${MY_IP}/32" >/dev/null

  echo "Locating default subnet in preferred AZ ($PREFERRED_AZ)..."
  SUBNET_ID=$(aws ec2 describe-subnets \
    --filters "Name=vpc-id,Values=$VPC_ID" "Name=availability-zone,Values=$PREFERRED_AZ" \
    --query "Subnets[0].SubnetId" \
    --output text)
  if [[ -z "$SUBNET_ID" || "$SUBNET_ID" == "None" ]]; then
    echo "[ERROR] No default subnet found in $PREFERRED_AZ (VPC $VPC_ID)." >&2
    exit 1
  fi
  echo "Using subnet $SUBNET_ID in $PREFERRED_AZ"
}

launch_spot_instance() {
  echo "Requesting EC2 Spot Instance ($INSTANCE_TYPE)..."
  local user_data_script="#!/bin/bash
nohup bash -c 'sleep 36000 && sudo poweroff' >/dev/null 2>&1 &"

  INSTANCE_JSON=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --subnet-id "$SUBNET_ID" \
    --instance-market-options "MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}" \
    --user-data "$user_data_script" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=VMemKV-i4i-Bench},{Key=VMemKV-Temp-Spot,Value=$KEY_NAME}]" \
    --query "Instances[0]" \
    --output json)

  INSTANCE_ID=$(echo "$INSTANCE_JSON" | jq -r '.InstanceId')
  echo "Launched Spot Instance ID: $INSTANCE_ID"

  echo "Waiting for instance to start running..."
  aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"

  PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --query "Reservations[0].Instances[0].PublicIpAddress" \
    --output text)
  echo "Public IP Address: $PUBLIC_IP"

  # The local state file stores instance identity and connection details.
  jq -n --arg iid "$INSTANCE_ID" --arg ip "$PUBLIC_IP" --arg key "$KEY_NAME" \
    --arg sg "$SG_ID" '{instance_id:$iid, public_ip:$ip, key_name:$key, sg_id:$sg}' \
    >"/tmp/vmemkv_aws_state_${KEY_NAME}.json"
  echo "Launch state saved: /tmp/vmemkv_aws_state_${KEY_NAME}.json"
}

wait_for_ssh_ready() {
  echo "Waiting for SSH to become available..."
  # ServerAliveInterval/CountMax: the LTM priming step's ssh command blocks for many minutes
  # while the remote side silently builds large corpora on NVMe -- no output flows back over the
  # SSH channel for most of that time. A NAT/firewall on the local side (WSL2's own NAT is a
  # candidate) can silently drop a connection it sees as idle well before the remote work is done,
  # independent of whether the remote instance is actually still healthy. Periodic keepalives keep
  # the channel visibly active even when the remote command itself has nothing to print for a while.
  SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $PEM_FILE -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=6"
  for i in {1..30}; do
    if ssh $SSH_OPTS "ubuntu@$PUBLIC_IP" "echo 'SSH Ready'" >/dev/null 2>&1; then
      echo "Connection established successfully."
      return 0
    fi
    sleep 4
  done
  echo "[ERROR] SSH did not become ready in time." >&2
  return 1
}

install_remote_dependencies() {
  echo "Preparing package locks & Installing runtime dependencies..."
  ssh $SSH_OPTS "ubuntu@$PUBLIC_IP" "
    for i in {1..30}; do
      if ! sudo fuser /var/lib/dpkg/lock-frontend /var/lib/apt/lists/lock /var/lib/dpkg/lock >/dev/null 2>&1; then
        break
      fi
      sleep 4
    done
    sudo apt-get update -y >/tmp/setup.log 2>&1 && sudo apt-get install -y build-essential cmake ninja-build libgoogle-perftools-dev librocksdb-dev liblmdb-dev libtbb-dev libaio-dev jq rsync >>/tmp/setup.log 2>&1
  "
}

prepare_remote_storage() {
  echo "Locating and mounting local NVMe SSD..."
  ssh $SSH_OPTS "ubuntu@$PUBLIC_IP" "
    # Select the nvme*n1 device that is both unpartitioned and not itself mounted (root EBS always carries a partition or a whole-disk mount; the local instance-store NVMe is raw at boot).
    DEV=\"\"
    for CAND in \$(lsblk -dbno NAME,SIZE | grep -E '^nvme[0-9]+n1' | sort -k2 -n -r | awk '{print \$1}'); do
      PART_COUNT=\$(lsblk -ln -o NAME \"/dev/\$CAND\" | wc -l)
      if [ \"\$PART_COUNT\" -gt 1 ]; then
        continue  # has at least one partition -- almost certainly the root EBS volume
      fi
      if findmnt -rn -S \"/dev/\$CAND\" >/dev/null 2>&1; then
        continue  # mounted directly (whole-disk root, no partition table)
      fi
      DEV=\"/dev/\$CAND\"
      break
    done
    if [ -z \"\$DEV\" ]; then
      echo \"[ERROR] Local NVMe SSD not found! A physical local NVMe is strictly required to run LTM / Swap benchmarks.\" >&2
      exit 1
    fi

    sudo mkdir -p /mnt/nvme
    # By construction DEV is unpartitioned and unmounted, so this should always be empty --
    # checked anyway (rather than assuming) in case some other process mounted it between the
    # selection above and here.
    MOUNT_TARGET=\$(findmnt -rn -S \"\$DEV\" -o TARGET 2>/dev/null | head -n 1 || true)

    if [ -n \"\$MOUNT_TARGET\" ]; then
      echo \"NVMe SSD \$DEV is already mounted at \$MOUNT_TARGET\"
      if [ \"\$MOUNT_TARGET\" != \"/mnt/nvme\" ]; then
        sudo mount --bind \"\$MOUNT_TARGET\" /mnt/nvme || true
        echo \"Bind-mounted \$MOUNT_TARGET to /mnt/nvme\"
      fi
    else
      if ! sudo blkid \"\$DEV\" >/dev/null 2>&1; then
        echo \"Formatting NVMe SSD \$DEV as ext4...\"
        sudo mkfs.ext4 -F -O mmp,dir_index,sparse_super \"\$DEV\" >/dev/null 2>&1
      else
        echo \"NVMe SSD \$DEV already has a filesystem; mounting without reformatting...\"
      fi
      sudo mount -o noatime,data=ordered \"\$DEV\" /mnt/nvme || true
    fi
    if ! findmnt -rn /mnt/nvme >/dev/null 2>&1; then
      echo \"[ERROR] Failed to mount NVMe SSD to /mnt/nvme\" >&2
      exit 1
    fi
    sudo chown -R ubuntu:ubuntu /mnt/nvme
    echo \"Mounted NVMe SSD to /mnt/nvme\"

    # Single-device topology for every backend: WAL and data share /mnt/nvme. A second
    # NVMe, if present, is deliberately left unmounted -- a separated WAL device for
    # vmemkv alone would flatter its write path against rivals whose WAL shares the
    # data device, and not every engine can separate (LMDB is a single file).

    SWAPFILE_SIZE_GB=\${AWS_SWAPFILE_SIZE_GB:-32}
    SWAPFILE_SIZE_BYTES=\$((SWAPFILE_SIZE_GB * 1024 * 1024 * 1024))
    AVAILABLE_BYTES=\$(df -B1 --output=avail /mnt/nvme | tail -n 1 | tr -d ' ')
    RESERVED_BYTES=\$((2 * 1024 * 1024 * 1024))
    MAX_SWAP_BYTES=\$((AVAILABLE_BYTES > RESERVED_BYTES ? AVAILABLE_BYTES - RESERVED_BYTES : 0))
    if [ \"\$MAX_SWAP_BYTES\" -le 0 ]; then
      echo \"[ERROR] NVMe filesystem does not have enough free space for a swapfile.\" >&2
      exit 1
    fi
    if [ \"\$SWAPFILE_SIZE_BYTES\" -gt \"\$MAX_SWAP_BYTES\" ]; then
      SWAPFILE_SIZE_BYTES=\"\$MAX_SWAP_BYTES\"
    fi
    echo \"Creating swap file of \$SWAPFILE_SIZE_BYTES bytes on NVMe SSD to prevent OOM Killer...\"
    sudo rm -f /mnt/nvme/swapfile
    sudo fallocate -l \"\$SWAPFILE_SIZE_BYTES\" /mnt/nvme/swapfile
    sudo chmod 600 /mnt/nvme/swapfile
    sudo mkswap /mnt/nvme/swapfile >/dev/null 2>&1
    sudo swapon /mnt/nvme/swapfile
    sudo sysctl -w vm.swappiness=100
    echo \"Swap file created and enabled successfully.\"

    # Transparent Huge Pages: vmemkv's T2 is a large MAP_PRIVATE region that gets randomly
    # accessed and, under the LTM scenario, swapped -- exactly the pattern THP hurts most (COW
    # amplification to a full 2MB on any write, khugepaged scan/lock overhead, synchronous
    # compaction stalls under the memory pressure this scenario deliberately creates, and 2MB-
    # granularity swap I/O that drags cold data in with hot data). vmemkv never opts in via
    # MADV_HUGEPAGE anywhere, so there is no upside to THP being enabled for it, regardless of
    # this AMI's actual default. 'madvise' rather than 'never': functionally equivalent for vmemkv
    # (which never advises for it) while remaining the standard, less invasive system-wide choice.
    THP_ENABLED_PATH=/sys/kernel/mm/transparent_hugepage/enabled
    THP_DEFRAG_PATH=/sys/kernel/mm/transparent_hugepage/defrag
    if [ -f \"\$THP_ENABLED_PATH\" ]; then
      CURRENT_THP=\$(grep -o '\\[[a-z]*\\]' \"\$THP_ENABLED_PATH\" | tr -d '[]')
      echo \"Transparent Huge Pages: currently '\$CURRENT_THP' (raw: \$(cat \"\$THP_ENABLED_PATH\"))\"
      if [ \"\$CURRENT_THP\" != \"madvise\" ] && [ \"\$CURRENT_THP\" != \"never\" ]; then
        echo \"  -> not madvise/never; setting to madvise (vmemkv never opts in via MADV_HUGEPAGE, so this is effectively 'never' for it).\"
        echo madvise | sudo tee \"\$THP_ENABLED_PATH\" >/dev/null
      fi
      if [ -f \"\$THP_DEFRAG_PATH\" ]; then
        CURRENT_DEFRAG=\$(grep -o '\\[[a-z+]*\\]' \"\$THP_DEFRAG_PATH\" | tr -d '[]')
        if [ \"\$CURRENT_DEFRAG\" != \"madvise\" ] && [ \"\$CURRENT_DEFRAG\" != \"never\" ]; then
          echo madvise | sudo tee \"\$THP_DEFRAG_PATH\" >/dev/null
        fi
      fi
      echo \"Transparent Huge Pages: now '\$(grep -o '\\[[a-z]*\\]' \"\$THP_ENABLED_PATH\" | tr -d '[]')'\"
    else
      echo \"Transparent Huge Pages: sysfs knob not present on this kernel, skipping.\"
    fi
  "
}

sync_repo_to_remote() {
  echo "Syncing codebase..."
  git -C "$IMPL_ROOT" ls-files | rsync -az -e "ssh $SSH_OPTS" --files-from=- "$IMPL_ROOT/" "ubuntu@$PUBLIC_IP:~/vmemkv/"
  scp $SSH_OPTS "$BENCH_SELF" "ubuntu@$PUBLIC_IP:~/run_bench.sh"
  scp $SSH_OPTS "${BENCH_DIR}/measure_ltm_ablation.sh" "ubuntu@$PUBLIC_IP:~/measure_ltm_ablation.sh"
}

build_remote_benchmark() {
  echo "Building bench_kv on AWS instance..."
  ssh $SSH_OPTS "ubuntu@$PUBLIC_IP" "
    cd ~/vmemkv &&
    cmake -S . -B build-rel -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS_RELEASE='-O3 -DNDEBUG -march=native' \
      -DENABLE_ROCKSDB=ON \
      -DENABLE_LMDB=ON \
      -DENABLE_LEANSTORE=ON \
      -DENABLE_BENCHMARK=ON &&
    cmake --build build-rel --target bench_kv --clean-first -j\"\$(nproc)\"
  "
  echo "AWS build completed successfully."
}

create_ssh_key_pair
resolve_ubuntu_ami
create_security_group
launch_spot_instance
wait_for_ssh_ready
install_remote_dependencies
prepare_remote_storage
sync_repo_to_remote
build_remote_benchmark

echo "Counting benchmark instances..."

read -r -a SCENARIO_ORDER_KEYS <<<"$(vmemkv_matrix::scenario_order_keys_from_flag "$ORDER")"

SCENARIO_ORDER_LABEL="$(vmemkv_scenario_order_label "${SCENARIO_ORDER_KEYS[@]}")"
RUNNER_FLAGS="ltm_first=$([[ "$ORDER" == "B_FIRST" ]] && printf '1' || printf '0');large_value_first=$([[ "$LARGE_VALUE_FIRST" == "true" ]] && printf '1' || printf '0')"

scenario_context_env_prefix() {
  local scenario_key="$1"
  local memory_budget_source
  local memory_budget_bytes
  local memo
  local memory_swap_max_bytes
  local swap_storage_media=""

  case "$scenario_key" in
    ltm)
      memory_budget_source="cgroup"
      memory_budget_bytes="$LTM_MEMORY_BUDGET_BYTES"
      memory_swap_max_bytes="$LTM_SWAP_BUDGET_BYTES"
      swap_storage_media="Local NVMe SSD"
      memo="$(vmemkv_benchmark_memo "aws_i4i" "ltm")"
      ;;
    in_memory)
      memory_budget_source="host"
      memory_budget_bytes=""
      memory_swap_max_bytes=""
      memo="In-memory runs execute without an enclosing cgroup."
      ;;
    *)
      return 1
      ;;
  esac

  vmemkv_context_env_prefix \
    "aws_i4i" \
    "Release" \
    "build-rel" \
    "ON" \
    "$SCENARIO_ORDER_LABEL" \
    "$RUNNER_FLAGS" \
    "$memory_budget_source" \
    "$memory_budget_bytes" \
    "$INSTANCE_TYPE" \
    "$AWS_REGION" \
    "$memo" \
    "" \
    "" \
    "$memory_swap_max_bytes" \
    "$swap_storage_media"
}

declare -A SCENARIO_TOTALS=(
  [in_memory]=0
  [ltm]=0
)

# Appends "VAR=value " pairs to the named accumulator for every listed env var that
# is set and non-empty, so normal runs stay byte-identical. Local-only helper for
# set_ltm_pass_env() below.
# Usage: vmemkv_forward_env ACCUMULATOR VAR...
vmemkv_forward_env() {
  local acc_name="$1"
  shift
  local current="${!acc_name:-}" var_name var_value
  for var_name in "$@"; do
    var_value="${!var_name:-}"
    if [[ -n "$var_value" ]]; then
      current+="${var_name}=${var_value} "
    fi
  done
  printf -v "$acc_name" '%s' "$current"
}

# Shared remote-command environment for LTM passes. Sets LTM_PASS_CTX_PREFIX,
# LTM_PASS_RT_PREFIX, LTM_PASS_ENV_PREFIX, LTM_PASS_LVF_PREFIX, LTM_PASS_YPOP_PREFIX,
# LTM_PASS_QUIESCE_PREFIX globals for a scenario, so repeated passes build
# byte-identical remote environments (same cgroup/memory posture, DB dir,
# SKIP_CLEANUP, and populate flags). Clone quiesce is standard here: the per-group T2
# sparse copy streams gigabytes through the page cache, and measuring on its residue
# makes throughput run-dependent (see quiesce_after_clone() in bench_kv.cpp).
set_ltm_pass_env() {
  local scenario_key="$1"
  local waldir_probe
  LTM_PASS_CTX_PREFIX="$(scenario_context_env_prefix "$scenario_key")"
  LTM_PASS_RT_PREFIX=""
  if [[ "$scenario_key" == "in_memory" ]]; then
    LTM_PASS_RT_PREFIX="VMEMKV_BENCH_FORCE_HOST_MEMORY=1"
  fi
  LTM_PASS_ENV_PREFIX="$(vmemkv_matrix::scenario_env_prefix "$scenario_key")"
  LTM_PASS_LVF_PREFIX=""
  if [[ "$LARGE_VALUE_FIRST" == "true" ]]; then
    LTM_PASS_LVF_PREFIX="VMEMKV_BENCH_LARGE_VALUE_FIRST=1"
  fi
  LTM_PASS_YPOP_PREFIX=""
  if [[ -n "${YCSB_E_POPULATE:-}" ]]; then
    LTM_PASS_YPOP_PREFIX="YCSB_E_POPULATE=${YCSB_E_POPULATE}"
  fi
  LTM_PASS_QUIESCE_PREFIX="VMEMKV_BENCH_QUIESCE_AFTER_CLONE=1"
  if [[ "${VMEMKV_BENCH_QUIESCE_AFTER_CLONE:-1}" == "0" ]]; then
    LTM_PASS_QUIESCE_PREFIX=""
  fi
  # WAL and data stay colocated on one device for every backend (see the storage setup
  # above); no per-backend WAL dir is forwarded here.
  LTM_PASS_WALDIR_PREFIX=""
  # Measurement-window overrides (forwarded only when set, so normal runs stay
  # byte-identical with the YPOP knob above). bench_kv honors these natively;
  # Update follows the runner's MIN_TIME.
  vmemkv_forward_env LTM_PASS_WINTIME_PREFIX VMEMKV_BENCH_INSERT_TIME_BUDGET_SECONDS VMEMKV_BENCH_DELETE_TIME_BUDGET_SECONDS VMEMKV_YCSB_DURATION_SEC YCSB_DURATION_SEC VMEMKV_YCSB_DISABLE_FORCED_TRIGGERS VMEMKV_YCSB_WARMUP_SEC YCSB_WARMUP_SEC
  # Cyclic-prime generation count for the Load=Cyclic steady-state Scan twin.
  vmemkv_forward_env LTM_PASS_CYCLIC_PREFIX VMEMKV_BENCH_CYCLIC_GENS
}

# Retried scp for result downloads: a single dropped SSH connection after hours of
# measurement must not strand the data. Fails loudly only after N attempts (so a
# genuinely dead instance still fails the run instead of hanging forever).
scp_retry() {
  local attempts=3 delay_seconds=15 attempt
  for ((attempt = 1; attempt <= attempts; attempt++)); do
    if scp "$@"; then
      return 0
    fi
    if [[ "$attempt" -lt "$attempts" ]]; then
      echo "[runner] scp attempt $attempt/$attempts failed -- retrying in ${delay_seconds}s ..." >&2
      sleep "$delay_seconds"
    fi
  done
  return 1
}

# Resume helper for run_ltm_isolated_groups(): if $1 (local per-group stage file
# from an earlier attempt) holds a valid result, copy it to $2 (this run's remote
# per-group result path) so the end-of-run merge sees a complete set, and return
# 0 -- the caller then skips the measurement. $3/$4 carry the scp transcript.
# Relies on the stage directory holding the same partition's group_idx mapping
# (group order is deterministic); point VMEMKV_STAGE_DIR at the old directory.
try_resume_group() {
  local stage_file="$1" remote_path="$2" stdout_log="$3" stderr_log="$4"
  [[ -s "$stage_file" ]] || return 1
  jq -e '.benchmarks | length > 0' "$stage_file" >/dev/null 2>&1 || return 1
  scp $SSH_OPTS "$stage_file" "ubuntu@$PUBLIC_IP:${remote_path}" >>"$stdout_log" 2>>"$stderr_log"
}

# Runs one ssh invocation with stdout/stderr teed to the given logs (mirrored to this
# shell's own stdout/stderr as well). Returns ssh's exit status, so callers under
# `set -e` must capture it with `|| status=$?`. Centralizes the set +e/PIPESTATUS
# juggling so the call sites cannot drift apart.
ssh_tee_logs() {
  local stdout_log="$1"
  local stderr_log="$2"
  shift 2
  local restore_errexit=""
  case $- in
    *e*) restore_errexit=1 ;;
  esac
  set +e
  { ssh "$@" 2> >(tee -a "$stderr_log" >&2); } | tee -a "$stdout_log"
  local ssh_status=${PIPESTATUS[0]}
  if [[ -n "$restore_errexit" ]]; then
    set -e
  fi
  return "$ssh_status"
}

# Runs LTM per-cell isolation groups for the benchmarks a filter matches.
# Each cell (one benchmark name) gets drop_caches + its OWN systemd-run cgroup scope
# and a pristine master clone. Sharing one scope across cells of a store/variant was
# found to let later-measured cells reuse the earlier cells' warm page cache (level
# shifts up to 5x on 64KB-Zipf), and sharing one store holder across cells lets later
# cells observe earlier cells' writes -- both make results depend on execution order
# Per-cell
# isolation removes both confounds; repetitions within one cell still share its scope
# (correlated error bars -- documented, not fixed here).
# Paging mode is split by default. An explicit mode can apply MGLRU on or off to all
# stores; in-memory scenarios do not use this LTM path. Split mode runs rivals first
# so no rival follows a knob toggle. Toggle state is tracked so a split run flips at
# most twice; the merged context records the selected mode. See
# Discovers exact benchmark names remotely and runs each as its own anchored filter
# (never a re-derived Value=/Op= regex), so this stays correct under every filter
# shape (plain combo, --scenario/--value-size limits, --ycsb-only).
# Args: $1 = scenario key (only used for env + log labels; the filter fully selects cells),
#       $2 = benchmark filter to discover and group,
#       $3 = remote result-path prefix; group files land at "<prefix>_group_<N>_<KEY>.json",
#       $4 = repetitions (1 = single run),
#       $5 = merged destination remote path,
#       $6 = stdout log, $7 = stderr log, $8 = progress label infix.
# Returns nonzero on the first failing group (remaining groups skipped); merges only on full
# success, exactly as if all groups had come from a single run (context from the first pass).
run_ltm_isolated_groups() {
  local scenario_key="$1" discover_filter="$2" out_prefix="$3" repetitions="$4"
  local merge_dest="$5" stdout_log="$6" stderr_log="$7" label="$8"
  local ltm_mglru_mode="${VMEMKV_LTM_MGLRU_ACTIVE_MODE:-on}"

  case "$ltm_mglru_mode" in
    split|on|off) ;;
    *)
      echo "[ERROR] Invalid active LTM MGLRU mode: $ltm_mglru_mode" >&2
      return 1
      ;;
  esac

  set_ltm_pass_env "$scenario_key"
  echo "[runner] discovering per-cell isolation groups for scenario=$scenario_key ($label) ..."
  local all_bench_names
  all_bench_names=$(ssh $SSH_OPTS "ubuntu@$PUBLIC_IP" \
    "bash -lc 'cd ~/vmemkv && ${LTM_PASS_ENV_PREFIX:+${LTM_PASS_ENV_PREFIX} }${LTM_PASS_CYCLIC_PREFIX:+${LTM_PASS_CYCLIC_PREFIX} }./build-rel/benchmark/bench_kv --benchmark_list_tests --benchmark_filter=\"${discover_filter}\" 2>/dev/null'")

  local -a identities=()
  while IFS= read -r identity_line; do
    [[ -z "$identity_line" ]] && continue
    identities+=("$identity_line")
  done < <(printf '%s\n' "$all_bench_names" | sort -u)

  if [[ "${#identities[@]}" -eq 0 ]]; then
    echo "[ERROR] No benchmarks matched ($label) -- nothing to isolate/run." >&2
    return 1
  fi
  echo "[runner] ${#identities[@]} isolation groups (one per cell)"

  # Split mode runs rival cells with MGLRU on and VMemKV cells with classic LRU.
  # The knob is machine-global, so split mode orders rivals before VMemKV.
  local -a ordered_identities=()
  local -a rival_identities=()
  local -a vmemkv_identities=()
  for identity in "${identities[@]}"; do
    if [[ "$identity" == Store=VMemKV/* ]]; then
      vmemkv_identities+=("$identity")
    else
      rival_identities+=("$identity")
    fi
  done
  if [[ "$ltm_mglru_mode" == "split" || "${VMEMKV_LTM_STORE_ORDER:-rivals-first}" != "vmemkv-first" ]]; then
    ordered_identities=("${rival_identities[@]}" "${vmemkv_identities[@]}")
  else
    ordered_identities=("${vmemkv_identities[@]}" "${rival_identities[@]}")
  fi

  # Partitioned runs: VMEMKV_GROUP_PARTITION="i/N" restricts this task to a
  # contiguous slice of the ordered identities (ceil(n/N) each), so N spot
  # instances can cover the matrix in parallel. Each partition still primes the
  # full masters and toggles MGLRU the same way; concatenating the per-partition
  # result files afterward reproduces the single-task file. Unset = whole matrix.
  if [[ -n "${VMEMKV_GROUP_PARTITION:-}" ]]; then
    local part_i="${VMEMKV_GROUP_PARTITION%/*}" part_n="${VMEMKV_GROUP_PARTITION#*/}"
    if ! [[ "$part_i" =~ ^[0-9]+$ && "$part_n" =~ ^[0-9]+$ ]] || ((part_i < 1 || part_i > part_n)); then
      echo "[ERROR] Invalid VMEMKV_GROUP_PARTITION=${VMEMKV_GROUP_PARTITION} (expected i/N, 1 <= i <= N)" >&2
      return 1
    fi
    local part_total="${#ordered_identities[@]}"
    local part_chunk=$(((part_total + part_n - 1) / part_n))
    local part_start=$(((part_i - 1) * part_chunk))
    ordered_identities=("${ordered_identities[@]:$part_start:$part_chunk}")
    if [[ "${#ordered_identities[@]}" -eq 0 ]]; then
      echo "[ERROR] Partition ${part_i}/${part_n} is empty (${part_total} groups total)" >&2
      return 1
    fi
    echo "[runner] partition ${part_i}/${part_n}: groups $((part_start + 1))..$((part_start + ${#ordered_identities[@]})) of ${part_total}"
  fi

  # Current knob state, read once (distro default is on; verified on every box used
  # so far, but never assume it here). Kernels without the knob run ambient.
  local mg_state mg_has_knob=true
  mg_state="$(ssh $SSH_OPTS "ubuntu@$PUBLIC_IP" "cat /sys/kernel/mm/lru_gen/enabled 2>/dev/null || echo unknown")"
  if [[ "$mg_state" == "unknown" ]]; then
    if [[ "$ltm_mglru_mode" != "split" ]]; then
      echo "[ERROR] MGLRU mode $ltm_mglru_mode requires /sys/kernel/mm/lru_gen/enabled" >&2
      return 1
    fi
    echo "[runner] no lru_gen knob on this kernel -- running ambient, no split" >&2
    mg_has_knob=false
  fi
  echo "[runner] MGLRU initial state: $mg_state"
  local saw_vmemkv=false saw_rival=false

  local -a group_result_paths=()
  local group_idx=0 identity group_names group_filter group_result_path group_remote_cmd group_remote_cmd_quoted group_status want_off
  # Per-group stage downloads: each group's JSON is scp'd locally right after it
  # succeeds, so a later Spot interruption loses at most the in-flight group. The
  # end-of-run merge below still reads the remote files when the instance survives;
  # when it does not, merge these stages by hand (one jq -s call, same expression)
  # and rerun only the missing identities via a custom filter. Best-effort: a failed
  # stage download must never fail the measurement itself. Point VMEMKV_STAGE_DIR at
  # a previous attempt's directory to resume it (same partition => same group_idx
  # mapping); already-staged groups are re-uploaded and skipped below.
  local stage_dir="${VMEMKV_STAGE_DIR:-${RESULTS_DIR:-/tmp}/groups_${KEY_NAME}}"
  mkdir -p "$stage_dir"
  local stage_label="${label//[^a-zA-Z0-9_]/_}"
  for identity in "${ordered_identities[@]}"; do
    group_idx=$((group_idx + 1))
    group_names="$(printf '%s\n' "$identity" | sed -E 's/[][(){}.^$*+?\\|]/\\&/g')"
    group_filter="^(${group_names})\$"
    group_result_path="${out_prefix}_group_${group_idx}_${KEY_NAME}.json"
    group_result_paths+=("$group_result_path")

    echo "[runner] LTM ${label} ${group_idx}/${#identities[@]}: ${identity}"
    if try_resume_group "${stage_dir}/group_${stage_label}_${group_idx}.json" \
        "$group_result_path" "$stdout_log" "$stderr_log"; then
      echo "[runner] LTM ${label} ${group_idx}/${#identities[@]}: already staged -- skipping"
      continue
    fi
    case "$ltm_mglru_mode" in
      split)
        if [[ "$identity" == Store=VMemKV/* ]]; then
          want_off=true
        else
          want_off=false
        fi
        ;;
      on) want_off=false ;;
      off) want_off=true ;;
    esac
    if [[ "$identity" == Store=VMemKV/* ]]; then
      saw_vmemkv=true
    else
      saw_rival=true
    fi
    local wanted_mg_state="0x0007"
    if [[ "$want_off" == true ]]; then
      wanted_mg_state="0x0000"
    fi
    if [[ "$mg_has_knob" == true && "$mg_state" != "$wanted_mg_state" ]]; then
      if [[ "$ltm_mglru_mode" == "split" ]]; then
        if [[ "$want_off" == true ]]; then
          echo "[runner] MGLRU off for vmemkv cells"
        else
          echo "[runner] MGLRU on for rival cells"
        fi
      else
        echo "[runner] MGLRU $([[ "$want_off" == true ]] && printf 'off' || printf 'on') for $identity"
      fi
      local set_mg_value="y"
      if [[ "$want_off" == true ]]; then
        set_mg_value="n"
      fi
      mg_state="$(ssh $SSH_OPTS "ubuntu@$PUBLIC_IP" \
        "echo $set_mg_value | sudo tee /sys/kernel/mm/lru_gen/enabled >/dev/null && cat /sys/kernel/mm/lru_gen/enabled" \
        2>>"$stderr_log")"
      printf '%s\n' "$mg_state" >>"$stdout_log"
      if [[ "$mg_state" != "$wanted_mg_state" ]]; then
        echo "[ERROR] MGLRU knob is $mg_state, expected $wanted_mg_state" >&2
        return 1
      fi
    fi
    ssh $SSH_OPTS "ubuntu@$PUBLIC_IP" "sync && echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null" \
      >>"$stdout_log" 2>>"$stderr_log"

    group_remote_cmd="
cd /home/ubuntu/vmemkv &&
${LTM_PASS_CTX_PREFIX} ${LTM_PASS_RT_PREFIX:+${LTM_PASS_RT_PREFIX} }VMEMKV_DB_DIR=/mnt/nvme ${LTM_PASS_ENV_PREFIX:+${LTM_PASS_ENV_PREFIX} }${LTM_PASS_LVF_PREFIX:+${LTM_PASS_LVF_PREFIX} }VMEMKV_BENCH_SKIP_CLEANUP=1 \
${LTM_PASS_YPOP_PREFIX:+${LTM_PASS_YPOP_PREFIX} }${LTM_PASS_QUIESCE_PREFIX:+${LTM_PASS_QUIESCE_PREFIX} }${LTM_PASS_WINTIME_PREFIX:+${LTM_PASS_WINTIME_PREFIX} }${LTM_PASS_CYCLIC_PREFIX:+${LTM_PASS_CYCLIC_PREFIX} }${LTM_PASS_WALDIR_PREFIX:+${LTM_PASS_WALDIR_PREFIX} }\
bash /home/ubuntu/run_bench.sh run-scenario \
  './build-rel/benchmark/bench_kv' \
  '$group_filter' \
  '$MIN_TIME' \
  '$group_result_path' \
  '$repetitions'
    "
    printf -v group_remote_cmd_quoted '%q' "$group_remote_cmd"

    group_status=0
    ssh_tee_logs "$stdout_log" "$stderr_log" $SSH_OPTS "ubuntu@$PUBLIC_IP" \
      "sudo systemd-run --wait --pipe --quiet -p MemoryAccounting=yes -p MemoryHigh=${LTM_MEMORY_BUDGET_BYTES} -p MemoryMax=$((LTM_MEMORY_BUDGET_BYTES * 2)) -p MemorySwapMax=${LTM_SWAP_BUDGET_BYTES} -- bash -lc ${group_remote_cmd_quoted}" \
      || group_status=$?
    if [[ "$group_status" -ne 0 ]]; then
      echo "[ERROR] LTM ${label} for ${identity} failed with exit code $group_status" >&2
      return "$group_status"
    fi
    scp $SSH_OPTS "ubuntu@$PUBLIC_IP:${group_result_path}" "${stage_dir}/group_${stage_label}_${group_idx}.json" \
      >>"$stdout_log" 2>>"$stderr_log" || echo "[runner] WARNING: stage download failed for group ${group_idx} (${identity}) -- continuing" >&2
  done

  local merge_paths_quoted="" p
  for p in "${group_result_paths[@]}"; do
    merge_paths_quoted+=" $(printf '%q' "$p")"
  done
  ssh $SSH_OPTS "ubuntu@$PUBLIC_IP" \
    "jq -s '{context: .[0].context, benchmarks: (map(.benchmarks) | add)}'${merge_paths_quoted} > $(printf '%q' "$merge_dest")" \
    >>"$stdout_log" 2>>"$stderr_log"
  if [[ "$mg_has_knob" == true && "$saw_vmemkv" == true && "$saw_rival" == true ]]; then
    local mglru_memo
    case "$ltm_mglru_mode" in
      split) mglru_memo="split-MGLRU: vmemkv cells classic-LRU, rival cells MGLRU-on" ;;
      on) mglru_memo="MGLRU-on for all stores" ;;
      off) mglru_memo="MGLRU-off for all stores" ;;
    esac
    ssh $SSH_OPTS "ubuntu@$PUBLIC_IP" \
      "jq --arg memo ' [$mglru_memo]' '.context.memo = ((.context.memo // \"\") + \$memo)' $(printf '%q' "$merge_dest") > $(printf '%q' "${merge_dest}.tmp") && mv $(printf '%q' "${merge_dest}.tmp") $(printf '%q' "$merge_dest")" \
      >>"$stdout_log" 2>>"$stderr_log"
  fi
  if [[ "$mg_has_knob" == true && "$mg_state" == "0x0000" ]]; then
    echo "[runner] restoring MGLRU on for subsequent steps"
    mg_state="$(ssh $SSH_OPTS "ubuntu@$PUBLIC_IP" \
      "echo y | sudo tee /sys/kernel/mm/lru_gen/enabled >/dev/null && cat /sys/kernel/mm/lru_gen/enabled" \
      2>>"$stderr_log")"
    printf '%s\n' "$mg_state" >>"$stdout_log"
    if [[ "$mg_state" != "0x0007" ]]; then
      echo "[ERROR] MGLRU restore returned $mg_state, expected 0x0007" >&2
      return 1
    fi
  fi
}

run_scenario_once() {
  local scenario_key="$1"
  local scenario_env_prefix
  local scenario_context_prefix
  local scenario_runtime_env_prefix=""
  local ycsb_populate_env_prefix=""
  local scenario_result_path
  local scenario_stdout_log
  local scenario_stderr_log
  local scenario_total
  local scenario_run_filter
  local remote_cmd
  local remote_cmd_quoted
  local result_mode_suffix=""
  local scenario_repetitions="$MATRIX_REPETITIONS"

  read -r -a scenario_value_order_keys <<<"$(vmemkv_matrix::scenario_value_order_keys_from_flag "$scenario_key" "$LARGE_VALUE_FIRST")"
  # The run filter comes from the main flow's pre-resolved inmem/ltm globals
  # (same run_bench::scenario_filter() rule the local launcher uses), so an
  # inmem_filter/ltm_filter env override applies here too.
  if [[ "$scenario_key" == "in_memory" ]]; then
    scenario_run_filter="$inmem_filter"
  else
    scenario_run_filter="$ltm_filter"
  fi
  scenario_env_prefix="$(vmemkv_matrix::scenario_env_prefix "$scenario_key")"
  scenario_context_prefix="$(scenario_context_env_prefix "$scenario_key")"
  scenario_result_path="$(vmemkv_matrix::scenario_result_path "$scenario_key")"
  if [[ "$scenario_key" == "ltm" && ( "${VMEMKV_LTM_MGLRU_ACTIVE_MODE:-on}" != "on" || -n "${VMEMKV_LTM_MGLRU_PASS_ID:-}" ) ]]; then
    result_mode_suffix="_mglru_${VMEMKV_LTM_MGLRU_ACTIVE_MODE}"
    if [[ -n "${VMEMKV_LTM_MGLRU_PASS_ID:-}" ]]; then
      result_mode_suffix+="_pass${VMEMKV_LTM_MGLRU_PASS_ID}"
    fi
    scenario_result_path="${scenario_result_path%.json}${result_mode_suffix}.json"
  fi
  LAST_SCENARIO_RESULT_PATH="$scenario_result_path"
  scenario_stdout_log="/tmp/vmemkv_${scenario_key}_${KEY_NAME}${result_mode_suffix}.stdout.log"
  scenario_stderr_log="/tmp/vmemkv_${scenario_key}_${KEY_NAME}${result_mode_suffix}.stderr.log"
  scenario_total="${SCENARIO_TOTALS[$scenario_key]}"

  : >"$scenario_stdout_log"
  : >"$scenario_stderr_log"

  if [[ "$scenario_key" == "in_memory" ]]; then
    scenario_runtime_env_prefix="VMEMKV_BENCH_FORCE_HOST_MEMORY=1"
  fi

  if [[ -n "${YCSB_E_POPULATE:-}" ]]; then
    ycsb_populate_env_prefix="YCSB_E_POPULATE=${YCSB_E_POPULATE}"
  fi
  if [[ -n "${VMEMKV_YCSB_DURATION_SEC:-}" ]]; then
    ycsb_populate_env_prefix="${ycsb_populate_env_prefix:+${ycsb_populate_env_prefix} }VMEMKV_YCSB_DURATION_SEC=${VMEMKV_YCSB_DURATION_SEC}"
  fi
  if [[ -n "${YCSB_DURATION_SEC:-}" ]]; then
    ycsb_populate_env_prefix="${ycsb_populate_env_prefix:+${ycsb_populate_env_prefix} }YCSB_DURATION_SEC=${YCSB_DURATION_SEC}"
  fi
  if [[ -n "${VMEMKV_YCSB_DISABLE_FORCED_TRIGGERS:-}" ]]; then
    ycsb_populate_env_prefix="${ycsb_populate_env_prefix:+${ycsb_populate_env_prefix} }VMEMKV_YCSB_DISABLE_FORCED_TRIGGERS=${VMEMKV_YCSB_DISABLE_FORCED_TRIGGERS}"
  fi
  if [[ -n "${VMEMKV_YCSB_WARMUP_SEC:-}" ]]; then
    ycsb_populate_env_prefix="${ycsb_populate_env_prefix:+${ycsb_populate_env_prefix} }VMEMKV_YCSB_WARMUP_SEC=${VMEMKV_YCSB_WARMUP_SEC}"
  fi
  if [[ -n "${YCSB_WARMUP_SEC:-}" ]]; then
    ycsb_populate_env_prefix="${ycsb_populate_env_prefix:+${ycsb_populate_env_prefix} }YCSB_WARMUP_SEC=${YCSB_WARMUP_SEC}"
  fi



  local large_value_first_env=""
  if [[ "$LARGE_VALUE_FIRST" == "true" ]]; then
    large_value_first_env="VMEMKV_BENCH_LARGE_VALUE_FIRST=1"
  fi

  local skip_cleanup_env_prefix=""
  if [[ "$scenario_key" == "ltm" ]]; then
    # Both the priming pass below and the cgroup-wrapped measurement pass after it run with
    # VMEMKV_BENCH_SKIP_CLEANUP=1 (see its comment in bench_kv.cpp), so bench_kv's own automatic
    # sweeps don't fight a deliberate multi-store, multi-master-generation priming pass. That
    # means this script owns the "start from a clean slate" step neither of those invocations
    # will do on their own -- do it once, here, unconditionally, exactly like bench_kv's own
    # startup sweep normally would (same "bench_" prefix, same directory: /mnt/nvme).
    ssh $SSH_OPTS "ubuntu@$PUBLIC_IP" "find /mnt/nvme -maxdepth 1 -name 'bench_*' -exec rm -rf {} +" \
      >>"$scenario_stdout_log" 2>>"$scenario_stderr_log"

    # Prime every shared master corpus this run will need at full, unconstrained NVMe speed
    # *before* entering the cgroup below. Without this, a real per-master populate under this
    # exact cgroup was measured locally at 200-1200s (vs. ~20-30s unconstrained) -- paying that
    # cost here means the cgroup-wrapped measurement pass only ever needs to clone, not rebuild,
    # those masters. See vmemkv_matrix::ltm_priming_filter()'s comment: one Get/Hit/Zipf cell per
    # (store, value size) is enough for the ordered masters, plus one shuffled Scan cell per
    # (store, value size) for the _shuffled masters the Load=Shuffled cells clone from.
    local priming_filter priming_cmd priming_cmd_quoted
    priming_filter="$(vmemkv_matrix::ltm_priming_filter "$WITHOUT_RIVALS" "$LEANSTORE_ONLY" "$ROCKSDB_ONLY")"
    # Subset runs prime only the master corpora required by the cells matching the filter.
    if [[ -n "${VMEMKV_RUN_FILTER_OVERRIDE:-}" ]]; then
      priming_filter="$(ssh $SSH_OPTS "ubuntu@$PUBLIC_IP" "bash -lc 'cd ~/vmemkv && ${scenario_env_prefix:+${scenario_env_prefix} }./build-rel/benchmark/bench_kv --benchmark_list_tests --benchmark_filter=\"${VMEMKV_RUN_FILTER_OVERRIDE}\" 2>/dev/null'" | vmemkv_matrix::ltm_priming_filter_from_names)"
    fi
    # VMEMKV_CONTEXT_memory_budget_bytes must be set here too, not just on the measurement pass
    # below (scenario_context_prefix) -- bench_kv's detect_machine_memory_bytes() falls back to
    # the host's raw /proc/meminfo MemTotal whenever this is unset, and priming runs unconstrained
    # (no cgroup, so no memory.max to fall back to either). Without it, the "small declared budget
    # x large target_ratio" corpus-sizing scheme (see vmemkv_matrix::ltm_memory_budget_bytes()'s
    # comment: 1GB budget x 8.0 ratio should be an ~8GB corpus) instead sizes against this
    # instance's full ~62GB RAM, producing a ~530GB corpus -- large enough to reliably OOM-kill
    # bench_kv partway through populate. Observed directly: every LTM priming attempt without this
    # fix died to the OOM killer, never a script or SSH problem.
    priming_cmd="
cd /home/ubuntu/vmemkv &&
VMEMKV_CONTEXT_memory_budget_bytes=$LTM_MEMORY_BUDGET_BYTES ${scenario_runtime_env_prefix:+${scenario_runtime_env_prefix} }VMEMKV_DB_DIR=/mnt/nvme ${scenario_env_prefix:+${scenario_env_prefix} } VMEMKV_BENCH_SKIP_CLEANUP=1 \
bash ~/run_bench.sh run-scenario \
  './build-rel/benchmark/bench_kv' \
  '$priming_filter' \
  '1s' \
  '/mnt/nvme/ltm_priming_discard.json'
    "
    printf -v priming_cmd_quoted '%q' "$priming_cmd"
    echo "[runner] priming LTM master corpora (unconstrained) for scenario=$scenario_key ..."
    ssh $SSH_OPTS "ubuntu@$PUBLIC_IP" "bash -lc ${priming_cmd_quoted}" \
      >>"$scenario_stdout_log" 2>>"$scenario_stderr_log"
    echo "[runner] priming complete for scenario=$scenario_key"

    # The measurement pass below reuses the masters priming just built -- it must not let
    # bench_kv's own startup cleanup_stale_benchmark_files() wipe them the instant it starts.
    skip_cleanup_env_prefix="VMEMKV_BENCH_SKIP_CLEANUP=1"
  fi

  remote_cmd="
cd /home/ubuntu/vmemkv &&
${scenario_context_prefix} ${scenario_runtime_env_prefix:+${scenario_runtime_env_prefix} }VMEMKV_DB_DIR=/mnt/nvme ${scenario_env_prefix:+${scenario_env_prefix} }${large_value_first_env:+${large_value_first_env} }${skip_cleanup_env_prefix:+${skip_cleanup_env_prefix} } \
${ycsb_populate_env_prefix:+${ycsb_populate_env_prefix} }\
bash ~/run_bench.sh run-scenario \
  './build-rel/benchmark/bench_kv' \
  '$scenario_run_filter' \
  '$MIN_TIME' \
  '$scenario_result_path' \
  '$MATRIX_REPETITIONS'
  "
  printf -v remote_cmd_quoted '%q' "$remote_cmd"

  echo "Scenario $scenario_key stdout log: $scenario_stdout_log"
  echo "Scenario $scenario_key stderr log: $scenario_stderr_log"
  echo "[runner] start scenario=$scenario_key"

  local ssh_status=0
  set +e
  if [[ "$scenario_key" == "ltm" ]]; then
    local saved_stage_dir="${VMEMKV_STAGE_DIR:-}"
    if [[ -n "$result_mode_suffix" ]]; then
      VMEMKV_STAGE_DIR="/tmp/groups_${KEY_NAME}${result_mode_suffix}"
    fi
    if [[ "$LTM_MGLRU_MODE" == "both" ]]; then
      scenario_repetitions=1
    fi
    run_ltm_isolated_groups "$scenario_key" "$scenario_run_filter" "/tmp/vmemkv_ltm" "$scenario_repetitions" \
      "$scenario_result_path" "$scenario_stdout_log" "$scenario_stderr_log" "isolated pass"
    ssh_status=$?
    VMEMKV_STAGE_DIR="$saved_stage_dir"
  else
    ssh_tee_logs "$scenario_stdout_log" "$scenario_stderr_log" $SSH_OPTS "ubuntu@$PUBLIC_IP" \
      "bash -lc ${remote_cmd_quoted}" \
      || ssh_status=$?
  fi
  set -e
  echo "[runner] end scenario=$scenario_key status=$ssh_status"

  if [[ "$ssh_status" -ne 0 ]]; then
    echo "[ERROR] Scenario $scenario_key failed with exit code $ssh_status" >&2
    echo "[ERROR] Scenario $scenario_key stdout log: $scenario_stdout_log" >&2
    echo "[ERROR] Scenario $scenario_key stderr log: $scenario_stderr_log" >&2
    # Scenario failures copy available result and YCSB-E timeline files over SSH.
    echo "[runner] best-effort partial result recovery (scp only) ..." >&2
    mkdir -p "${BENCH_DIR}/logs"
    scp $SSH_OPTS "ubuntu@$PUBLIC_IP:$scenario_result_path" \
      "${BENCH_DIR}/logs/$(basename "$scenario_result_path" .json).partial.json" >/dev/null 2>&1 || true
    scp $SSH_OPTS "ubuntu@$PUBLIC_IP:/tmp/ycsb_e_timeline_*.json" "${BENCH_DIR}/logs/" >/dev/null 2>&1 || true
    return "$ssh_status"
  fi
}

run_scenario() {
  local scenario_key="$1"
  if [[ "$scenario_key" != "ltm" || "$LTM_MGLRU_MODE" != "both" ]]; then
    VMEMKV_LTM_MGLRU_ACTIVE_MODE="$LTM_MGLRU_MODE" \
      run_scenario_once "$scenario_key"
    return
  fi

  local -a modes=(on off off on on off off on on off)
  local -a store_orders=(rivals-first vmemkv-first rivals-first vmemkv-first vmemkv-first rivals-first vmemkv-first rivals-first rivals-first vmemkv-first)
  local pass mode store_order pass_count value_suffix subset_suffix without_rivals_suffix leanstore_only_suffix result_name
  pass_count=$((LTM_MGLRU_PAIRS * 2))
  local results_dir="${BENCH_DIR}/logs"
  mkdir -p "$results_dir"
  value_suffix=""
  if [[ -n "$VALUE_SIZE_LIMIT" ]]; then
    value_suffix="_${VALUE_SIZE_LIMIT}"
  fi
  subset_suffix=""
  if [[ -n "${VMEMKV_RUN_FILTER_OVERRIDE:-}" ]]; then
    subset_suffix="_subset"
  fi
  without_rivals_suffix=""
  if [[ "$WITHOUT_RIVALS" == "true" ]]; then
    without_rivals_suffix="_vmemkv_only"
  fi
  leanstore_only_suffix=""
  if [[ "$LEANSTORE_ONLY" == "true" ]]; then
    leanstore_only_suffix="_leanstore_only"
  fi

  for ((pass = 0; pass < pass_count; pass++)); do
    mode="${modes[$pass]}"
    store_order="${store_orders[$pass]}"
    printf -v VMEMKV_LTM_MGLRU_PASS_ID '%02d' "$((pass + 1))"
    echo "[runner] LTM MGLRU pass ${VMEMKV_LTM_MGLRU_PASS_ID}/${pass_count}: mode=$mode store_order=$store_order"
    VMEMKV_LTM_MGLRU_ACTIVE_MODE="$mode" \
      VMEMKV_LTM_STORE_ORDER="$store_order" \
      run_scenario_once "$scenario_key"

    result_name="results_ltm${value_suffix}${without_rivals_suffix}${leanstore_only_suffix}_mglru_${mode}_pass${VMEMKV_LTM_MGLRU_PASS_ID}${subset_suffix}.json"
    scp_retry $SSH_OPTS "ubuntu@$PUBLIC_IP:${LAST_SCENARIO_RESULT_PATH}" "${results_dir}/${result_name}"
  done
  unset VMEMKV_LTM_MGLRU_ACTIVE_MODE VMEMKV_LTM_MGLRU_PASS_ID VMEMKV_LTM_STORE_ORDER
}


# Runs one "additive extra measurement" probe script against the already-provisioned instance.
# BACKGROUND_JOBS_PROBE below uses this orchestration -- up to two invocations (in_memory
# unconstrained, ltm systemd-run-wrapped with the same MemoryHigh/MemoryMax/MemorySwapMax as
# run_scenario()'s own ltm wrap), respecting SCENARIO_LIMIT/VALUE_SIZE_LIMIT exactly like the
# matrix above.
#
# Args: $1 = probe subcommand name (e.g. bg-jobs)
#       $2 = output basename (e.g. background_jobs -> background_jobs_in_memory.jsonl)
#       $3 = log-file prefix (e.g. vmemkv_background_jobs_probe)
#       $4 = human-readable label for [runner]/[WARN] messages (e.g. background-jobs-probe)
#       $5 = optional extra args forwarded verbatim to the probe script (e.g. --key-pattern=random)
#
# A failure here is logged as [WARN], not [ERROR], and does not fail the whole run: unlike the
# matrix above, this is a supplementary measurement, not the main deliverable.
run_remote_probe() {
  local probe_script="$1"
  local output_basename="$2"
  local log_prefix="$3"
  local label="$4"
  local extra_probe_args="${5:-}"

  local probe_failed=0

  local scenario
  for scenario in in_memory ltm; do
    if [[ "$SCENARIO_LIMIT" != "all" && "$SCENARIO_LIMIT" != "$scenario" ]]; then
      continue
    fi
    # in_memory logs use the historic "inmem" infix; ltm keeps its scenario name.
    local short="${scenario/in_memory/inmem}"
    local combo_filter="$scenario"
    local dst_name="${output_basename}_${scenario}.jsonl"
    if [[ -n "$VALUE_SIZE_LIMIT" ]]; then
      combo_filter+=":${VALUE_SIZE_LIMIT}"
      dst_name="${output_basename}_${scenario}_${VALUE_SIZE_LIMIT}.jsonl"
    fi
    local stdout_log="/tmp/${log_prefix}_${short}_${KEY_NAME}.stdout.log"
    local stderr_log="/tmp/${log_prefix}_${short}_${KEY_NAME}.stderr.log"
    : >"$stdout_log"
    : >"$stderr_log"
    local context_env_prefix=""
    local ssh_wrap="bash -lc"
    if [[ "$scenario" == "ltm" ]]; then
      # VMEMKV_CONTEXT_memory_budget_bytes must be set explicitly here, not left to
      # detect_machine_memory_bytes()'s cgroup-file fallback: that would read this systemd-run scope's
      # MemoryMax (2x LTM_MEMORY_BUDGET_BYTES, see below), not the declared budget itself, mis-sizing
      # every corpus by 2x -- same reasoning as run_scenario()'s priming/measurement passes above.
      context_env_prefix="VMEMKV_CONTEXT_memory_budget_bytes=${LTM_MEMORY_BUDGET_BYTES} "
      ssh_wrap="sudo systemd-run --wait --pipe --quiet -p MemoryAccounting=yes -p MemoryHigh=${LTM_MEMORY_BUDGET_BYTES} -p MemoryMax=$((LTM_MEMORY_BUDGET_BYTES * 2)) -p MemorySwapMax=${LTM_SWAP_BUDGET_BYTES} -- bash -lc"
    fi
    local remote_cmd="
cd /home/ubuntu/vmemkv &&
${context_env_prefix}bash /home/ubuntu/run_bench.sh probe ${probe_script} './build-rel/benchmark/bench_kv' '/mnt/nvme/${output_basename}_${scenario}.jsonl' '/mnt/nvme' '$combo_filter' ${extra_probe_args}
    "
    local remote_cmd_quoted
    printf -v remote_cmd_quoted '%q' "$remote_cmd"
    echo "[runner] start ${label} scenario=$scenario combo_filter=$combo_filter"
    local pass_status=0
    ssh_tee_logs "$stdout_log" "$stderr_log" $SSH_OPTS "ubuntu@$PUBLIC_IP" "${ssh_wrap} ${remote_cmd_quoted}" \
      || pass_status=$?
    echo "[runner] end ${label} scenario=$scenario status=$pass_status"
    if [[ "$pass_status" -ne 0 ]]; then
      # [WARN], deliberately not [ERROR]: this run's own on-error logging (and any external
      # watcher pattern-matching "[ERROR]" to decide the whole run failed, see this session's
      # AWS-monitoring convention) must not treat a supplementary-measurement hiccup as fatal
      # when the main benchmark matrix already succeeded.
      echo "[WARN] ${label} (${scenario}) failed with exit code $pass_status -- logs: $stdout_log $stderr_log" >&2
      probe_failed=1
    fi
    scp $SSH_OPTS "ubuntu@$PUBLIC_IP:/mnt/nvme/${output_basename}_${scenario}.jsonl" "${RESULTS_DIR}/${dst_name}" || true
  done

  if [[ "$probe_failed" -ne 0 ]]; then
    echo "[WARN] ${label} had failures -- main benchmark matrix results above are still valid" >&2
  fi
}


inmem_filter="${inmem_filter:-$(run_bench::scenario_filter in_memory)}"
ltm_filter="${ltm_filter:-$(run_bench::scenario_filter ltm)}"

# One-off subset runs: VMEMKV_RUN_FILTER_OVERRIDE replaces both scenario filters (and LTM
# priming below), so only matching cells are measured. Priming then builds exactly the
# masters the subset needs. Downloads are suffixed, never overwriting full results; merge
# the subset file back by hand afterward. Not honored by cmd_parallel (full matrix only).
if [[ -n "${VMEMKV_RUN_FILTER_OVERRIDE:-}" ]]; then
  echo "[runner] subset run: filter override active"
  inmem_filter="$VMEMKV_RUN_FILTER_OVERRIDE"
  ltm_filter="$VMEMKV_RUN_FILTER_OVERRIDE"
fi

echo "Counting remote benchmarks concurrently..."
if [[ "$SCENARIO_LIMIT" == "in_memory" || "$SCENARIO_LIMIT" == "all" ]]; then
  count_remote_benchmarks "$inmem_filter" "$(vmemkv_matrix::scenario_env_prefix in_memory)" "$([[ "$LARGE_VALUE_FIRST" == "true" ]] && printf 'VMEMKV_BENCH_LARGE_VALUE_FIRST=1')${VMEMKV_BENCH_CYCLIC_GENS:+ VMEMKV_BENCH_CYCLIC_GENS=${VMEMKV_BENCH_CYCLIC_GENS}}" > "/tmp/vmemkv_inmem_count_${KEY_NAME}.txt" &
  pid1=$!
else
  pid1=""
fi

if [[ "$SCENARIO_LIMIT" == "ltm" || "$SCENARIO_LIMIT" == "all" ]]; then
  count_remote_benchmarks "$ltm_filter" "$(vmemkv_matrix::scenario_env_prefix ltm)" "$([[ "$LARGE_VALUE_FIRST" == "true" ]] && printf 'VMEMKV_BENCH_LARGE_VALUE_FIRST=1')${VMEMKV_BENCH_CYCLIC_GENS:+ VMEMKV_BENCH_CYCLIC_GENS=${VMEMKV_BENCH_CYCLIC_GENS}}" > "/tmp/vmemkv_ltm_count_${KEY_NAME}.txt" &
  pid2=$!
else
  pid2=""
fi

if [[ -n "$pid1" ]]; then wait "$pid1"; fi
if [[ -n "$pid2" ]]; then wait "$pid2"; fi

total_inmem=0
total_ltm=0
if [[ "$SCENARIO_LIMIT" == "in_memory" || "$SCENARIO_LIMIT" == "all" ]]; then
  total_inmem=$(cat "/tmp/vmemkv_inmem_count_${KEY_NAME}.txt")
  rm -f "/tmp/vmemkv_inmem_count_${KEY_NAME}.txt"
fi
if [[ "$SCENARIO_LIMIT" == "ltm" || "$SCENARIO_LIMIT" == "all" ]]; then
  total_ltm=$(cat "/tmp/vmemkv_ltm_count_${KEY_NAME}.txt")
  rm -f "/tmp/vmemkv_ltm_count_${KEY_NAME}.txt"
fi

if [[ "$LTM_MGLRU_MODE" == "both" && ( "$SCENARIO_LIMIT" == "ltm" || "$SCENARIO_LIMIT" == "all" ) ]]; then
  GLOBAL_TOTAL=$((total_inmem + total_ltm * LTM_MGLRU_PAIRS * 2))
else
  GLOBAL_TOTAL=$((total_inmem + total_ltm))
fi

if [[ "$SKIP_MATRIX" != "true" ]]; then
  if [[ "$SCENARIO_LIMIT" == "in_memory" || "$SCENARIO_LIMIT" == "all" ]] && ((total_inmem == 0)); then
    echo "[ERROR] No in-memory benchmarks matched the selected filters." >&2
    return 1
  fi
  if [[ "$SCENARIO_LIMIT" == "ltm" || "$SCENARIO_LIMIT" == "all" ]] && ((total_ltm == 0)); then
    echo "[ERROR] No LTM benchmarks matched the selected filters." >&2
    return 1
  fi
fi

write_progress_state "$GLOBAL_TOTAL"
echo "Total benchmarks to run: $GLOBAL_TOTAL"

if [[ "$SKIP_MATRIX" != "true" ]]; then
  run_bench::each_scenario run_scenario
fi

# ── Retrieve Results & Save ───────────────────────────────────────────────
RESULTS_DIR="${BENCH_DIR}/logs"
mkdir -p "$RESULTS_DIR"

if [[ "$SKIP_MATRIX" != "true" ]]; then
  echo "Downloading results..."

  # With --without-rivals, the downloaded JSON only covers VMemKV variants; tag the filename so it
  # is not mistaken for or silently overwritten by a full-matrix result.
  without_rivals_suffix=""
  if [[ "$WITHOUT_RIVALS" == "true" ]]; then
    without_rivals_suffix="_vmemkv_only"
  fi
  leanstore_only_suffix=""
  if [[ "$LEANSTORE_ONLY" == "true" ]]; then
    leanstore_only_suffix="_leanstore_only"
  fi
  rocksdb_only_suffix=""
  if [[ "$ROCKSDB_ONLY" == "true" ]]; then
    rocksdb_only_suffix="_rocksdb_only"
  fi
  # Partitioned LTM runs (VMEMKV_GROUP_PARTITION=i/N) download side by side;
  # the caller merges them afterward. In-memory runs never set the knob.
  part_suffix=""
  if [[ -n "${VMEMKV_GROUP_PARTITION:-}" ]]; then
    part_suffix="_part${VMEMKV_GROUP_PARTITION%/*}of${VMEMKV_GROUP_PARTITION#*/}"
  fi
  mglru_suffix=""
  if [[ "$LTM_MGLRU_MODE" == "off" || "$LTM_MGLRU_MODE" == "split" ]]; then
    mglru_suffix="_mglru_${LTM_MGLRU_MODE}"
  fi
  # Subset runs (VMEMKV_RUN_FILTER_OVERRIDE) download side by side for the same reason:
  # a partial matrix must never overwrite a full result. Merge back by hand afterward.
  subset_suffix=""
  if [[ -n "${VMEMKV_RUN_FILTER_OVERRIDE:-}" ]]; then
    subset_suffix="_subset"
  fi

  if [[ "$SCENARIO_LIMIT" == "in_memory" || "$SCENARIO_LIMIT" == "all" ]]; then
    dst_name="results_in_memory${without_rivals_suffix}${leanstore_only_suffix}${rocksdb_only_suffix}${subset_suffix}.json"
    if [[ -n "$VALUE_SIZE_LIMIT" ]]; then
      dst_name="results_in_memory_${VALUE_SIZE_LIMIT}${without_rivals_suffix}${leanstore_only_suffix}${rocksdb_only_suffix}${subset_suffix}.json"
    fi
    scp_retry $SSH_OPTS "ubuntu@$PUBLIC_IP:$(vmemkv_matrix::scenario_result_path in_memory)" "${RESULTS_DIR}/${dst_name}"
  fi

  if [[ "$SCENARIO_LIMIT" == "ltm" || "$SCENARIO_LIMIT" == "all" ]]; then
    if [[ "$LTM_MGLRU_MODE" == "both" ]]; then
      echo "[runner] all-on/off result files were downloaded after each pass"
    else
      dst_name="results_ltm${without_rivals_suffix}${leanstore_only_suffix}${rocksdb_only_suffix}${part_suffix}${mglru_suffix}${subset_suffix}.json"
      if [[ -n "$VALUE_SIZE_LIMIT" ]]; then
        dst_name="results_ltm_${VALUE_SIZE_LIMIT}${without_rivals_suffix}${leanstore_only_suffix}${rocksdb_only_suffix}${part_suffix}${mglru_suffix}${subset_suffix}.json"
      fi
      remote_ltm_result="$(vmemkv_matrix::scenario_result_path ltm)"
      if [[ -n "$mglru_suffix" ]]; then
        remote_ltm_result="${remote_ltm_result%.json}${mglru_suffix}.json"
      fi
      scp_retry $SSH_OPTS "ubuntu@$PUBLIC_IP:${remote_ltm_result}" "${RESULTS_DIR}/${dst_name}"
    fi
  fi

  # Retrieve YCSB-E timeline results if they exist
  echo "Downloading YCSB-E timeline logs..."
  scp $SSH_OPTS "ubuntu@$PUBLIC_IP:/tmp/ycsb_e_timeline_*.json" "${RESULTS_DIR}/" || true
fi

if [[ "$BACKGROUND_JOBS_PROBE" == "true" ]]; then
  # Additive extra measurement (reorganize()/checkpoint() own duration + Insert/Update/Scan QPS
  # degradation while each runs, fixed 1KB/10,000,000-record corpus) on top of the normal matrix
  # just run above -- not part of the Google Benchmark-registered matrix itself, see
  # bg-jobs's own comment for why. Supersedes the three retired probes
  # (reorg-scaling, checkpoint-throughput, maintenance-contention). Up to two invocations, exactly
  # mirroring run_scenario()'s ltm-only cgroup wrap: unconstrained for in_memory, systemd-run-
  # wrapped (same MemoryHigh/MemoryMax/MemorySwapMax as the real ltm scenario) for ltm -- mixing
  # both under one wrap would starve in_memory of memory it was never meant to be constrained by.
  # Ignores VALUE_SIZE_LIMIT (this probe's corpus is always 1KB, fixed) -- parallel's
  # dedicated 5th instance passes no --value-size, so this always covers both scenarios here.
  # A failure here is logged but does not fail the whole run -- unlike the matrix above, this is a
  # supplementary measurement, not the main deliverable.
  run_remote_probe "bg-jobs" "background_jobs" "vmemkv_background_jobs_probe" "background-jobs-probe"
fi

if [[ "$ORGANIC_SPLIT_PROBE" == "true" ]]; then
  # Additive extra measurement (Insert-QPS impact of ShardedT1Index's own automatic per-shard
  # background splitting under sustained write load) on top of the normal matrix just run above --
  # complements background-jobs-probe above (which measures a forced, whole-store call, not what
  # happens during ordinary operation) rather than replacing it. Same ltm-cgroup-wrap/
  # VALUE_SIZE_LIMIT-ignoring/non-fatal-failure conventions as that call, see its own comment.
  run_remote_probe "organic-split" "organic_split" "vmemkv_organic_split_probe" "organic-split-probe"
fi

if [[ "$ORGANIC_SPLIT_PROBE_RANDOM" == "true" ]]; then
  # Random-key twin of the probe above: same sustained-insert shape, but writes spread over all
  # shards, showing whether one split's throughput impact shrinks as the shard count grows.
  # Separate output basenames so both patterns coexist in the logs directory; same
  # non-fatal-failure convention as every other probe.
  run_remote_probe "organic-split" "organic_split_random" "vmemkv_organic_split_probe_random" \
    "organic-split-probe-random" "--key-pattern=random"
fi

if [[ "$CHECKPOINT_SCALE_PROBE" == "true" ]]; then
  # Checkpoint/flush/merge cost vs corpus size (checkpoint_scale_probe): interleaved-prime
  # steady-state store, 5 measured insert+checkpoint+compact cycles at 3 corpus sizes per
  # scenario (1KB fixed). Same non-fatal-failure convention as every other probe.
  run_remote_probe "checkpoint-scale" "checkpoint_scale" "vmemkv_checkpoint_scale_probe" "checkpoint-scale-probe"
fi

if [[ "$SEGMENT_SCAN_PROBE" == "true" ]]; then
  if [[ "$SCENARIO_LIMIT" != "ltm" || "$VALUE_SIZE_LIMIT" != "1KB" ]]; then
    echo "[ERROR] --segment-scan-probe requires --scenario ltm --value-size 1KB" >&2
    return 1
  fi

  # Build the shared LTM master before the cgroup-wrapped probe; the probe itself clones it.
  local prime_filter='^Store=VMemKV/Variant=Bloom-T1InlineValue/Op=Get/Mode=Hit/Dist=Zipf/Value=1KB\(20% 8B\)/real_time/threads:1$'
  local prime_cmd prime_cmd_quoted
  prime_cmd="cd /home/ubuntu/vmemkv && VMEMKV_CONTEXT_memory_budget_bytes=$LTM_MEMORY_BUDGET_BYTES VMEMKV_BENCH_LTM=1 VMEMKV_BENCH_TARGET_RATIO=8.0 VMEMKV_DB_DIR=/mnt/nvme VMEMKV_BENCH_SKIP_CLEANUP=1 bash ~/run_bench.sh run-scenario './build-rel/benchmark/bench_kv' '$prime_filter' 1s /mnt/nvme/segment_scan_prime.json 1"
  printf -v prime_cmd_quoted '%q' "$prime_cmd"
  echo "[runner] priming the 1KB LTM master for Scan -> Update -> Scan ..."
  ssh $SSH_OPTS "ubuntu@$PUBLIC_IP" "bash -lc ${prime_cmd_quoted}"

  run_remote_probe "segment-scan" "segment_scan" "vmemkv_segment_scan_probe" "segment-scan-probe" "${VMEMKV_SEGMENT_SCAN_REVERSE:+--reverse} ${VMEMKV_SEGMENT_SCAN_ABAB:+--abab}"
fi

if [[ "$COMPACTION_CHURN_PROBE" == "true" ]]; then
  run_remote_probe "compaction-churn" "compaction_churn" "vmemkv_compaction_churn_probe" "compaction-churn-probe"
fi

if [[ "$ABLATION_PROBE" == "true" ]]; then
  echo "[runner] start ablation-probe (Compaction ON vs OFF)"
  ssh $SSH_OPTS "ubuntu@$PUBLIC_IP" "bash ~/measure_ltm_ablation.sh ~/vmemkv/build-rel/benchmark/bench_kv /mnt/nvme/ablation"
  echo "[runner] end ablation-probe"
fi
}

cmd_parallel() {
# parallel: 4 CRUD-matrix combos + 1 background-probe instance, each on its own spot VM.

# Pass all incoming flags (e.g. --without-rivals, --repetitions, --filter) through to each task
local extra_flags=("$@")
if [[ " ${extra_flags[*]} " =~ " --without-rivals " ]]; then
  echo "[without-rivals] Running vmemkv variants only -- reuse the last full-bench run's rival numbers."
fi

echo "========================================================"
echo " Starting 7-Parallel AWS Spot Benchmarks for vmemkv     "
echo "========================================================"

# Define the CRUD-matrix tasks. Format: "scenario value_size log_prefix [partition]"
declare -a TASKS=(
  "in_memory 8B inmem_8b"
  "in_memory 1KB inmem_1kb"
  "ltm 1KB ltm_1kb_p1 1/2"
  "ltm 1KB ltm_1kb_p2 2/2"
  "ltm 64KB ltm_64kb_p1 1/2"
  "ltm 64KB ltm_64kb_p2 2/2"
)

pids=()
declare -A task_logs

for task in "${TASKS[@]}"; do
  read -r scenario val_size log_prefix part <<< "$task"
  log_file="/tmp/vmemkv_parallel_${log_prefix}.log"
  task_logs["$log_prefix"]="$log_file"

  echo "Launching Instance for: Scenario=$scenario, ValueSize=$val_size${part:+ Partition=$part} (Log: $log_file)"

  local -a part_flag=()
  if [[ -n "$part" ]]; then
    part_flag=(--partition "$part")
  fi

  "${BENCH_SELF}" aws \
    --scenario "$scenario" \
    --value-size "$val_size" \
    "${part_flag[@]}" \
    "${extra_flags[@]}" \
    > "$log_file" 2>&1 &
  pids+=($!)

  # Stagger instance launches to prevent AWS OAuth/token 429 Rate Limit Errors
  sleep 12
done

# 5th task: dedicated background-maintenance-probes instance (each probe's own scenario loop
# covers both in_memory and ltm internally -- see probe bg-jobs /
# probe organic-split / probe checkpoint-scale / probe compaction-churn), no CRUD matrix.
echo "Launching Instance for: Background Maintenance Probes (Log: /tmp/vmemkv_parallel_bgjobs.log)"
"${BENCH_SELF}" aws \
  --skip-matrix \
  --background-jobs-probe \
  --organic-split-probe \
  --checkpoint-scale-probe \
  --compaction-churn-probe \
  > "/tmp/vmemkv_parallel_bgjobs.log" 2>&1 &
pids+=($!)
TASKS+=("- - bgjobs")
task_logs["bgjobs"]="/tmp/vmemkv_parallel_bgjobs.log"

echo "--------------------------------------------------------"
echo "All 5 Spot Instances requested. Waiting for execution..."
echo "You can tail the logs in /tmp/vmemkv_parallel_*.log"
echo "--------------------------------------------------------"

# Monitor loop
failed=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  read -r scenario val_size log_prefix part <<< "${TASKS[$i]}"

  set +e
  wait "$pid"
  status=$?
  set -e

  if [[ "$status" -ne 0 ]]; then
    echo "[ERROR] Task Scenario=$scenario ValSize=$val_size failed with exit code $status" >&2
    echo "Check log: ${task_logs[$log_prefix]}" >&2
    failed=$((failed + 1))
  else
    echo "[SUCCESS] Task Scenario=$scenario ValSize=$val_size finished successfully."
  fi
done

if [[ "$failed" -eq 0 ]]; then
  # Merge partitioned LTM results if any
  local results_dir="${BENCH_DIR}/logs"
  local without_rivals_suffix=""
  if [[ " ${extra_flags[*]} " =~ " --without-rivals " ]]; then
    without_rivals_suffix="_vmemkv_only"
  fi
  local mglru_suffix=""
  if [[ " ${extra_flags[*]} " =~ " --ltm-mglru-mode off " ]]; then
    mglru_suffix="_mglru_off"
  elif [[ " ${extra_flags[*]} " =~ " --ltm-mglru-mode split " ]]; then
    mglru_suffix="_mglru_split"
  fi
  for sc in ltm; do
    for vs in 1KB 64KB; do
      local p1="${results_dir}/results_${sc}_${vs}${without_rivals_suffix}_part1of2${mglru_suffix}.json"
      local p2="${results_dir}/results_${sc}_${vs}${without_rivals_suffix}_part2of2${mglru_suffix}.json"
      local target="${results_dir}/results_${sc}_${vs}${without_rivals_suffix}${mglru_suffix}.json"
      if [[ -f "$p1" && -f "$p2" ]]; then
        echo "Merging partitioned results for $sc $vs into $(basename "$target") ..."
        jq -s '{context: .[0].context, benchmarks: ([.[].benchmarks] | add)}' "$p1" "$p2" > "$target"
      fi
    done
  done

  echo "========================================================"
  echo " All 7 Parallel Benchmarks Finished SUCCESSFULLY!       "
  echo " Results downloaded to benchmark/logs/   "
  echo "========================================================"
else
  echo "========================================================"
  echo " Parallel Benchmarks Finished with $failed Errors.       "
  echo "========================================================"
  exit 1
fi
}


usage() {
  echo "Usage: run_bench.sh [local [options]] | aws [options] | parallel [--without-rivals]"
  echo "                    | probe <bg-jobs|organic-split|checkpoint-scale|segment-scan|compaction-churn> <bench_kv_bin> <output> [db_dir] [combo] [extra...]"
  echo "                    | run-scenario <bench_kv_bin> <filter> <min_time> <out_json> [repetitions]"
  echo "                    | clean [key-name]"
  echo "Bare flags (e.g. --scenario ltm) mean 'local'. 'local --help' / 'aws --help' list options."
}

cmd_probe() {
  if [[ $# -lt 1 ]]; then
    echo "[ERROR] probe needs a name: bg-jobs | organic-split | checkpoint-scale | segment-scan | compaction-churn" >&2
    return 1
  fi
  local name="$1"
  shift
  local had_errexit=false
  case $- in
    *e*) had_errexit=true ;;
  esac
  set +e
  "probe_${name//-/_}" "$@"
  local status=$?
  if [[ "$had_errexit" == "true" ]]; then
    set -e
  fi
  return "$status"
}

PROGRESS_AWK="$(mktemp /tmp/vmemkv_progress_awk.XXXXXX.awk)"
write_progress_awk "$PROGRESS_AWK"

main() {
  local cmd="${1:-local}"
  case "$cmd" in
    local) shift; cmd_local "$@" ;;
    aws) shift; cmd_aws "$@" ;;
    parallel) shift; cmd_parallel "$@" ;;
    probe) shift; cmd_probe "$@" ;;
    run-scenario) shift; bench_run_scenario "$@" ;;
    clean) shift; aws_clean_resources "$@" ;;
    -h|--help|help) usage ;;
    --*) cmd_local "$@" ;;
    *) echo "[ERROR] Unknown subcommand: $cmd" >&2; usage >&2; return 1 ;;
  esac
}

main "$@"
