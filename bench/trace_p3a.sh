#!/usr/bin/env bash
# P3a mechanism traces: nsys on one point each, per-phase kernel tables + API summaries.
# Workspace sizes are taken from the generated final grids (same shape args -> same probe).
# Usage: bench/trace_p3a.sh GPU MODEL CPUS       MODEL: 3090 | 4090 | 5060ti
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/env/toolchain.sh"
G=${1:?gpu}; MODEL=${2:?model}; CPUS=${3:?cpus}
BIN=$ROOT/bench/bin/dgemm_bench_v5
GR=$ROOT/bench/grids/final
COMMIT=$(git -C "$ROOT" rev-parse HEAD)
OUT=$ROOT/results/final/P3a_nsys/P3a_nsys_gpu${G}_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
echo "commit $COMMIT" > "$OUT/git.txt"
"$ROOT/env/capture.sh" "$G" > "$OUT/env_snapshot_dir.txt"
sha256sum "$BIN" > "$OUT/binary.sha256"
EMU="CUBLAS_EMULATE_DOUBLE_PRECISION=1"
EAGER="$EMU CUBLAS_EMULATION_STRATEGY=EAGER"
ws() {  # workspace MiB used by grid line <name> in <phase grid>
  grep -m1 "^$2|" "$GR/${1}_$MODEL.txt" | grep -o -- '--workspace-mb [0-9]*' | cut -d' ' -f2
}
# zero-code strategy: the 3090 only emulates under EAGER
if [ "$MODEL" = 3090 ]; then ZC=$EAGER; ZCN=eager; else ZC=$EMU; ZCN=default; fi
trace() {  # name traces env args
  local name=$1 tr=$2 e=$3 a=$4
  env $e CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=12 taskset -c "$CPUS" nsys profile -t "$tr" --sample=none \
    --cpuctxsw=none -f true --export sqlite -o "$OUT/$name" "$BIN" $a --no-ref --no-cupti --warmup 1 --warmup-s 0.5 \
    --reps 5 --samples 64 --tag "p3a:$name" --out "$OUT/bench.csv" > /dev/null 2>> "$OUT/err.log" \
    || echo "FAIL $name" >> "$OUT/err.log"
  nsys stats --force-export=true -r cuda_gpu_kern_sum,cuda_api_sum -f csv -o "$OUT/$name" "$OUT/$name.nsys-rep" > /dev/null 2>> "$OUT/err.log"
  env $e CUDA_VISIBLE_DEVICES=$G CUBLASLT_LOG_LEVEL=5 CUBLASLT_LOG_FILE="$OUT/$name.ltlog" taskset -c "$CPUS" \
    "$BIN" $a --no-ref --no-cupti --no-wall --warmup 0 --warmup-s 0 --reps 1 --samples 16 > /dev/null 2>&1 || true
}
sq() { echo "--m $1 --n $1 --k $1 --api dgemm"; }
# 1. Zero-code memory penalty timeline: zero code vs workspace, n = 1024 and 4096 (osrt for the host side of the remap)
for n in 1024 4096; do
  trace "sq$n.${ZCN}_ws0" cuda,osrt "$ZC" "$(sq $n)"
  trace "sq$n.${ZCN}_ws" cuda,osrt "$ZC" "$(sq $n) --workspace-mb $(ws P1a sq:${ZCN}_ws:n$n)"
done
# 2. per-phase breakdown at n = 8192
trace "sq8192.eager_ws" cuda "$EAGER" "$(sq 8192) --workspace-mb $(ws P1a sq:eager_ws:n8192)"
# 3. Gram s = 128 over k; 4. panel / skinny around the break-even
for k in 8192 32768 131072; do
  trace "gram128_k$k.eager_ws" cuda "$EAGER" "--m 128 --n 128 --k $k --api dgemm --workspace-mb $(ws P1b shape:eager_ws:gram:m128_n128_k$k)"
done
for k in 32 128; do
  trace "panel_k$k.eager_ws" cuda "$EAGER" "--m 8192 --n 8192 --k $k --api dgemm --workspace-mb $(ws P1b shape:eager_ws:panel:m8192_n8192_k$k)"
done
for n in 64 512; do
  trace "skinny_n$n.eager_ws" cuda "$EAGER" "--m 8192 --n $n --k 8192 --api dgemm --workspace-mb $(ws P1b shape:eager_ws:skinny:m8192_n${n}_k8192)"
done
# 5. the two kinds of fallback: runtime (inner phi 8) and heuristic decline (default strategy, n = 4096)
trace "inner8.eager_ws" cuda "$EAGER" "$(sq 4096) --dist inner --phi 8 --workspace-mb $(ws P1c range:eager_ws:inner:phi8:k4096)"
trace "sq4096.default_ws" cuda "$EMU" "$(sq 4096) --workspace-mb $(ws P1a sq:default_ws:n4096)"
echo "$(date -u +%FT%TZ) DONE" >> "$OUT/err.log"
echo "$OUT"
