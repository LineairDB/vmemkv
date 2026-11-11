# Benchmark runner

This directory contains the C++ benchmark target and the local/AWS matrix runner. The measured binary is `build/benchmark/bench_kv`.

## Local runs

Install the root README's build requirements, plus Python 3 and `jq`. Optional rival backends require their corresponding development packages.

```bash
benchmark/run_bench.sh local --scenario in_memory --value-size 1KB
benchmark/run_bench.sh local --scenario ltm --value-size 1KB --without-rivals
```

Run `benchmark/run_bench.sh local --help` for all filters and settings. LTM corpus sizing uses a 1 GiB budget. To apply real memory pressure locally, `--cgroup` requests a systemd user scope with delegated cgroups; verify that the scope constrains RSS before treating a run as an LTM measurement.

## AWS Spot runs

AWS mode creates a temporary Spot instance, key pair, and security group. It requires AWS CLI credentials with the EC2 permissions used by the runner, a default VPC/subnet in the selected availability zone, outbound access to AWS, SSH, `rsync`, `curl`, `jq`, and available Spot capacity. LTM runs also require local NVMe. SSH ingress is restricted to the caller's current public IP. The runner attempts cleanup on exit and signals; verify AWS resources after interrupted or failed runs. `benchmark/run_bench.sh clean [key-name]` removes matching resources.

The remote sync uses `git ls-files`, so commit the repository before starting an AWS run. Local runs from a checkout without a commit record `git_revision=uncommitted` and `git_dirty=1`. Do not add AWS credentials, generated key files, state files, or raw run logs to this repository.

```bash
VMEMKV_YCSB_WARMUP_SEC=90 \
VMEMKV_YCSB_DURATION_SEC=600 \
YCSB_DURATION_SEC=600 \
VMEMKV_YCSB_DISABLE_FORCED_TRIGGERS=1 \
benchmark/run_bench.sh aws --scenario ltm --value-size 1KB --without-rivals --ycsb-only
```

Use `--rocksdb-only --ycsb-only` for the RocksDB run. Archived run data are in `benchmark/results/ycsb_e.json`.

## Results

Runner outputs are written to `benchmark/logs/`, which is ignored by Git. `run_bench.sh` passes Google Benchmark JSON through `sanitize_results.py` to remove host and cloud-resource identifiers while retaining measurement and build metadata. The published experiment archives are `benchmark/results/10_23.json` and `benchmark/results/ycsb_e.json`.
