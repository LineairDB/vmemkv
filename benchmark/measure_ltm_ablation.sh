#!/usr/bin/env bash
set -euo pipefail

BENCH_KV="${1:-./build-rel/benchmark/bench_kv}"
OUT_DIR="${2:-/mnt/nvme/ablation}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SANITIZER="$SCRIPT_DIR/sanitize_results.py"
if [[ ! -f "$SANITIZER" && -f "$HOME/vmemkv/benchmark/sanitize_results.py" ]]; then
  SANITIZER="$HOME/vmemkv/benchmark/sanitize_results.py"
fi
mkdir -p "$OUT_DIR"

LTM_MEMORY_BUDGET_BYTES=$((1024 * 1024 * 1024))
LTM_SWAP_BUDGET_BYTES=$((32 * 1024 * 1024 * 1024))

export VMEMKV_CONTEXT_memory_budget_bytes=$LTM_MEMORY_BUDGET_BYTES
export VMEMKV_BENCH_LTM=1
export VMEMKV_BENCH_TARGET_RATIO=8.0
export VMEMKV_DB_DIR=/mnt/nvme
export VMEMKV_BENCH_SKIP_CLEANUP=1

SYSTEMD_WRAP="sudo systemd-run --wait --pipe --quiet \
  -p MemoryAccounting=yes \
  -p MemoryHigh=${LTM_MEMORY_BUDGET_BYTES} \
  -p MemoryMax=$((LTM_MEMORY_BUDGET_BYTES * 2)) \
  -p MemorySwapMax=${LTM_SWAP_BUDGET_BYTES} \
  -- "

OPS=("Insert" "Update" "YCSB-E")

run_pass() {
  local mode="$1" # "on" or "off"
  echo "=================================================="
  echo "=== Running Pass: Compaction ${mode^^} ==="
  echo "=================================================="
  
  if [[ "$mode" == "off" ]]; then
    export VMEMKV_SUPPRESS_AUTO_COMPACT=1
  else
    unset VMEMKV_SUPPRESS_AUTO_COMPACT || true
  fi

  for op in "${OPS[@]}"; do
    echo "--- Measuring Op=$op (Compaction $mode) ---"
    local filter="^Store=VMemKV/Variant=Bloom-T1InlineValue/Op=${op}.*/Value=1KB.*/real_time/threads:32$"
    local out_json="$OUT_DIR/${op}_compaction_${mode}.json"
    
    if $SYSTEMD_WRAP "$BENCH_KV" \
      --benchmark_filter="$filter" \
      --benchmark_min_time=5s \
      --benchmark_repetitions=1 \
      --benchmark_out="$out_json" \
      --benchmark_out_format=json; then
      python3 "$SANITIZER" "$out_json"
    else
      echo "Warning: Op=$op failed with status $?"
    fi
  done
}

run_pass "on"
run_pass "off"

echo "=================================================="
echo "=== Results Summary (Compaction ON vs OFF) ==="
echo "=================================================="
python3 -c "
import json
import glob

def get_qps(filepath):
    try:
        with open(filepath) as f:
            data = json.load(f)
        for b in data.get('benchmarks', []):
            if 'items_per_second' in b:
                return b['items_per_second']
            if 'real_time' in b and b['real_time'] > 0:
                return 1e9 / b['real_time']
    except Exception as e:
        pass
    return 0.0

ops = ['Insert', 'Update', 'YCSB-E']
print(f'| Op | Compaction ON (ops/s) | Compaction OFF (ops/s) | Diff |')
print(f'|---|---|---|---|')
for op in ops:
    on_qps = get_qps(f'$OUT_DIR/{op}_compaction_on.json')
    off_qps = get_qps(f'$OUT_DIR/{op}_compaction_off.json')
    diff_str = 'N/A'
    if on_qps > 0:
        pct = (off_qps - on_qps) / on_qps * 100.0
        diff_str = f'{pct:+.1f}%'
    print(f'| {op} (32 thr) | {on_qps:,.0f} | {off_qps:,.0f} | {diff_str} |')
"
