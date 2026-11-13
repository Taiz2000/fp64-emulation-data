#!/usr/bin/env bash
# Documented vs observed: which memory pool holds cuBLAS's FP64-emulation workspace (workloads/pool_probe.py).
# Pool cases through CuPy's matmul and through a direct cublasDgemm, plus nsys traces (CUDA + cuBLAS API), one GPU.
# Usage: workloads/run_pool_probe.sh GPU CPUS [N]     Output: results/final/P2f_pool/pool_gpu<G>_<ts>/
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/env/toolchain.sh"
G=${1:?gpu}; CPUS=${2:?cpus}; N=${3:-4096}
PY=$ROOT/.venv/bin/python
DIRTY=$(git -C "$ROOT" status --porcelain --untracked-files=no -- bench env workloads)
if [ -n "$DIRTY" ] && [ "${ALLOW_DIRTY:-0}" != 1 ]; then echo "uncommitted code changes" >&2; exit 1; fi
OUT=$ROOT/results/final/P2f_pool/pool_gpu${G}_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
{ echo "commit $(git -C "$ROOT" rev-parse HEAD)"; echo "dirty ${DIRTY:+yes}"; } > "$OUT/git.txt"
"$ROOT/env/capture.sh" "$G" > "$OUT/env_snapshot_dir.txt"
EMU="CUBLAS_EMULATE_DOUBLE_PRECISION=1"
EAGER="$EMU CUBLAS_EMULATION_STRATEGY=EAGER"
WS=2048  # MiB, above the emulation workspace at n=4096 (about 1.2 GiB)
run() {  # label env args...
  local label=$1 e=$2; shift 2
  env $e CUDA_VISIBLE_DEVICES=$G LD_LIBRARY_PATH=$CUDA_LIB taskset -c "$CPUS" "$PY" "$ROOT/workloads/pool_probe.py" \
    --label "$label" --n "$N" --out "$OUT/pool.csv" "$@" >> "$OUT/pool.log" 2>&1 || echo "FAIL $label rc=$?" >> "$OUT/pool.log"
}
run native             ""       --api matmul --pool default --threshold zero
run default_zc         "$EMU"   --api matmul --pool default --threshold zero   # PERFORMANT (declines on the 3090)
run eager_zc           "$EAGER" --api matmul --pool default --threshold zero   # zero code changes
run eager_kp           "$EAGER" --api matmul --pool default --threshold max    # default pool kept (--keep-pool fix)
run eager_custom_zero  "$EAGER" --api matmul --pool custom --threshold zero    # new pool installed as current pool
run eager_custom_max   "$EAGER" --api matmul --pool custom --threshold max
run eager_ws_matmul    "$EAGER" --api matmul --pool default --threshold zero --ws-mib $WS
run eager_zc_dgemm     "$EAGER" --api dgemm  --pool default --threshold zero
run eager_ws_dgemm     "$EAGER" --api dgemm  --pool default --threshold zero --ws-mib $WS
run eager_zc_ctypes    "$EAGER" --api ctypes --pool default --threshold zero               # direct libcublas call
run eager_ws_ctypes    "$EAGER" --api ctypes --pool default --threshold zero --ws-mib $WS  # no CuPy setStream in between
# nsys: every CUDA API call (runtime + driver), cuBLAS API calls, and kernel names (emulated path or not)
for t in "eager_zc_matmul|--api matmul" "eager_ws_matmul|--api matmul --ws-mib $WS" "eager_ws_dgemm|--api dgemm --ws-mib $WS" \
         "eager_ws_ctypes|--api ctypes --ws-mib $WS"; do
  name=${t%%|*}; a=${t#*|}
  env $EAGER CUDA_VISIBLE_DEVICES=$G LD_LIBRARY_PATH=$CUDA_LIB taskset -c "$CPUS" nsys profile -t cuda,cublas \
    --sample=none --cpuctxsw=none -f true -o "$OUT/nsys_$name" "$PY" "$ROOT/workloads/pool_probe.py" \
    --label "nsys_$name" --n "$N" --pool default --threshold zero --out "$OUT/pool_nsys.csv" $a \
    > /dev/null 2>> "$OUT/nsys.log" || echo "FAIL nsys $name" >> "$OUT/nsys.log"
  nsys stats --force-export=true -r cuda_api_sum,cuda_gpu_kern_sum,nvtx_sum -f csv -o "$OUT/nsys_$name" \
    "$OUT/nsys_$name.nsys-rep" > /dev/null 2>> "$OUT/nsys.log"
done
echo "rows: $(grep -c "'call'" "$OUT/pool.log")"; grep FAIL "$OUT"/*.log
echo "$OUT"
