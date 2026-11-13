#!/usr/bin/env bash
# P2e follow-up: non-finite class of the overflowing entry (workloads/overflow_class.py), all modes, one GPU.
# Usage: workloads/run_overflow.sh GPU CPUS     Output: results/final/P2e_followup/overflow_gpu<G>_<ts>/
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/env/toolchain.sh"
G=${1:?gpu}; CPUS=${2:?cpus}
PY=$ROOT/.venv/bin/python
DIRTY=$(git -C "$ROOT" status --porcelain --untracked-files=no -- bench env workloads)
if [ -n "$DIRTY" ] && [ "${ALLOW_DIRTY:-0}" != 1 ]; then echo "uncommitted code changes" >&2; exit 1; fi
OUT=$ROOT/results/final/P2e_followup/overflow_gpu${G}_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
{ echo "commit $(git -C "$ROOT" rev-parse HEAD)"; echo "dirty ${DIRTY:+yes}"; } > "$OUT/git.txt"
"$ROOT/env/capture.sh" "$G" > "$OUT/env_snapshot_dir.txt"
EMU="CUBLAS_EMULATE_DOUBLE_PRECISION=1"
EAGER="$EMU CUBLAS_EMULATION_STRATEGY=EAGER"
for mode in native default eager eager_mask0; do
  case $mode in native) e="";; default) e="$EMU";; eager) e="$EAGER";;
    eager_mask0) e="$EAGER CUBLAS_EMULATION_SPECIAL_VALUES_SUPPORT_MASK=0";; esac
  env $e CUDA_VISIBLE_DEVICES=$G LD_LIBRARY_PATH=$CUDA_LIB taskset -c "$CPUS" "$PY" "$ROOT/workloads/overflow_class.py" \
    --mode $mode --out "$OUT/overflow.csv" >> "$OUT/overflow.log" 2>&1 || echo "FAIL $mode rc=$?" >> "$OUT/overflow.log"
done
cat "$OUT/overflow.csv"
