#!/usr/bin/env bash
# P3b mempool remap microbenchmark incl. the workspace sizes cuBLAS requests per square n.
# Usage: bench/run_p3b.sh GPU MODEL CPUS
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/env/toolchain.sh"   # CUDA_DEVICE_ORDER=PCI_BUS_ID: without it GPU indices follow CUDA's fastest-first order
G=${1:?gpu}; MODEL=${2:?model}; CPUS=${3:?cpus}
OUT=$ROOT/results/final/P3b_mempool/P3b_mempool_gpu${G}_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
echo "commit $(git -C "$ROOT" rev-parse HEAD)" > "$OUT/git.txt"
"$ROOT/env/capture.sh" "$G" > "$OUT/env_snapshot_dir.txt"
sha256sum "$ROOT/bench/bin/mempool_bench" > "$OUT/binary.sha256"
# workspace sizes from the P1a grid (eager_ws lines), minus the 32 MiB margin -> what cuBLAS asks for
WS=$(grep '^sq:eager_ws:' "$ROOT/bench/grids/final/P1a_$MODEL.txt" | grep -o -- '--workspace-mb [0-9]*' | awk '{print $2 - 32}')
CUDA_VISIBLE_DEVICES=$G taskset -c "$CPUS" "$ROOT/bench/bin/mempool_bench" 30 $WS > "$OUT/mempool.csv" 2> "$OUT/err.log"
echo "$OUT"
