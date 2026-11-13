#!/usr/bin/env bash
# P4 randomized-SVD collection on one GPU.
# Usage: workloads/run_rsvd.sh GPU CPUS        env: REPEATS (3), MATS (exp poly lrnoise scaled4 scaled8), PATHCHECK (1)
# Output: results/final/P4_rsvd/P4_rsvd_gpu<G>_<ts>/ with rsvd.csv, svals/, log, nsys path check, provenance.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/env/toolchain.sh"
G=${1:?gpu}; CPUS=${2:?cpus}
REPEATS=${REPEATS:-3}
MATS=${MATS:-"exp poly lrnoise scaled4 scaled8"}
PY=$ROOT/.venv/bin/python
COMMIT=$(git -C "$ROOT" rev-parse HEAD)
DIRTY=$(git -C "$ROOT" status --porcelain --untracked-files=no -- bench env workloads)
if [ -n "$DIRTY" ] && [ "${ALLOW_DIRTY:-0}" != 1 ]; then echo "uncommitted code changes" >&2; exit 1; fi
OUT=$ROOT/results/final/P4_rsvd/P4_rsvd_gpu${G}_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT/svals" "$OUT/nsys"
{ echo "commit $COMMIT"; echo "dirty ${DIRTY:+yes}"; } > "$OUT/git.txt"
"$ROOT/env/capture.sh" "$G" > "$OUT/env_snapshot_dir.txt"
sha256sum "$ROOT/workloads/rsvd.py" "$ROOT"/data/*.npy > "$OUT/inputs.sha256" 2>/dev/null
echo "REPEATS=$REPEATS CPUS=$CPUS MATS=$MATS" > "$OUT/runner.txt"
LOG=$OUT/rsvd.log
EMU="CUBLAS_EMULATE_DOUBLE_PRECISION=1"
EAGER="$EMU CUBLAS_EMULATION_STRATEGY=EAGER"
MODES=(native zc zc_eager fix fix_eager fp32 tf32)
menv() { case $1 in native|fp32) echo "";; tf32) echo "CUPY_TF32=1";; zc|fix) echo "$EMU";; zc_eager|fix_eager) echo "$EAGER";; esac; }
margs() { case $1 in fix|fix_eager) echo "--keep-pool";; fp32|tf32) echo "--dtype float32";; *) echo "";; esac; }
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,clocks.sm,power.draw,temperature.gpu,clocks_event_reasons.active \
  --format=csv -i "$G" -l 1 > "$OUT/smi_audit.csv" &
SMI=$!; trap 'kill $SMI 2>/dev/null' EXIT
NM=${#MODES[@]}
for rep in $(seq 1 "$REPEATS"); do
  for mat in $MATS; do
    for i in $(seq 0 $((NM - 1))); do
      mode=${MODES[$(((i + rep - 1) % NM))]}
      echo "$(date -u +%FT%TZ) rep=$rep $mat $mode" >> "$LOG"
      env $(menv $mode) CUDA_VISIBLE_DEVICES=$G LD_LIBRARY_PATH=$CUDA_LIB OPENBLAS_NUM_THREADS=12 \
        taskset -c "$CPUS" "$PY" "$ROOT/workloads/rsvd.py" --matrix "$ROOT/data/$mat.npy" --mode $mode $(margs $mode) \
        --seed $rep --out "$OUT/rsvd.csv" --sdir "$OUT/svals" --tag "rep$rep" >> "$LOG" 2>&1 \
        || echo "$(date -u +%FT%TZ) FAIL rc=$? rep=$rep $mat $mode" >> "$LOG"
    done
  done
done
# ---- path check: which kernels the GEMMs (and cuSOLVER's QR) actually ran, via nsys kernel names
if [ "${PATHCHECK:-1}" = 1 ]; then
  for mat in exp scaled8; do for mode in zc zc_eager; do
    env $(menv $mode) CUDA_VISIBLE_DEVICES=$G LD_LIBRARY_PATH=$CUDA_LIB OPENBLAS_NUM_THREADS=12 taskset -c "$CPUS" \
      nsys profile -t cuda -o "$OUT/nsys/$mat.$mode" -f true "$PY" "$ROOT/workloads/rsvd.py" --matrix "$ROOT/data/$mat.npy" \
      --mode $mode --warm 0 --reps 1 --out "$OUT/nsys/pathcheck.csv" --sdir "$OUT/nsys/svals" --tag pathcheck >> "$LOG" 2>&1
    nsys stats -q -r cuda_gpu_kern_sum -f csv -o "$OUT/nsys/$mat.$mode" "$OUT/nsys/$mat.$mode.nsys-rep" >> "$LOG" 2>&1
  done; done
fi
echo "$(date -u +%FT%TZ) DONE" >> "$LOG"
echo "$OUT"
