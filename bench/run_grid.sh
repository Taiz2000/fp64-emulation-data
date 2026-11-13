#!/usr/bin/env bash
# Generic grid runner. Each grid line: name|ENV=val ...|bench args (must include --m/--n/--k).
# Lines run in rotated order per repeat; idle check before each; nvidia-smi audit log.
# Usage: bench/run_grid.sh GPU GRIDFILE LABEL
#   env: REPEATS (3), BIN (bench/bin/dgemm_bench_v5), CPUS (taskset list, e.g. 16-31; empty = no pinning),
#        OMP_NUM_THREADS (12), OUTBASE (results/final), ALLOW_DIRTY=1 to run with uncommitted code changes
# Output: $OUTBASE/<LABEL>/<LABEL>_gpu<G>_<ts>/ ; one line appended to $OUTBASE/MANIFEST.tsv when done.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(dirname "$HERE")
source "$ROOT/env/toolchain.sh"
G=${1:?gpu}; GRID=${2:?gridfile}; LABEL=${3:?label}
BIN=${BIN:-$HERE/bin/dgemm_bench_v5}
REPEATS=${REPEATS:-3}
OUTBASE=${OUTBASE:-$ROOT/results/final}
CPUS=${CPUS:-}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-12}
# provenance: code must be committed (results/ and docs may change freely)
COMMIT=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo none)
DIRTY=$(git -C "$ROOT" status --porcelain --untracked-files=no -- bench env workloads 2>/dev/null)
if [ -n "$DIRTY" ] && [ "${ALLOW_DIRTY:-0}" != 1 ]; then
  echo "uncommitted changes in bench/env/workloads; commit first or set ALLOW_DIRTY=1:" >&2; echo "$DIRTY" >&2; exit 1
fi
OUT=$OUTBASE/$LABEL/${LABEL}_gpu${G}_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
cp "$GRID" "$OUT/grid.txt"
{ echo "commit $COMMIT"; echo "dirty ${DIRTY:+yes}"; echo "$DIRTY"; } > "$OUT/git.txt"
"$ROOT/env/capture.sh" "$G" > "$OUT/env_snapshot_dir.txt"
sha256sum "$BIN" "$BIN.cu" > "$OUT/binary.sha256" 2>/dev/null || sha256sum "$BIN" > "$OUT/binary.sha256"
echo "REPEATS=$REPEATS CPUS=$CPUS OMP_NUM_THREADS=$OMP_NUM_THREADS BIN=$BIN" > "$OUT/runner.txt"
PIN=(); [ -n "$CPUS" ] && PIN=(taskset -c "$CPUS")
mapfile -t LINES < <(grep -v -E '^\s*(#|$)' "$GRID")
N=${#LINES[@]}
CSV=$OUT/gpu$G.csv; LOG=$OUT/gpu$G.log

wait_idle() {  # no compute process on the GPU and mem<500 MiB (utilization.gpu lags, not used)
  local bus nproc m i
  bus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "$G")
  for i in $(seq 1 120); do
    nproc=$(nvidia-smi --query-compute-apps=gpu_bus_id --format=csv,noheader | grep -c "$bus")
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$G")
    if [ "$nproc" -eq 0 ] && [ "$m" -lt 500 ]; then return 0; fi
    echo "$(date -u +%FT%TZ) busy procs=$nproc mem=$m, waiting" >> "$LOG"
    sleep 5
  done
  return 1
}

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active \
  --format=csv -i "$G" -l 1 > "$OUT/smi_audit.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null' EXIT

for rep in $(seq 1 "$REPEATS"); do
  for idx in $(seq 0 $((N - 1))); do
    IFS='|' read -r name envs args <<< "${LINES[$(((idx + rep - 1) % N))]}"
    # placeholders: @OUT@ = this run's directory, @REP@ = repeat number (for per-line Lt logs / kernel dumps)
    envs=${envs//@OUT@/$OUT}; envs=${envs//@REP@/$rep}; args=${args//@OUT@/$OUT}; args=${args//@REP@/$rep}
    mkdir -p "$OUT/lt" "$OUT/kern"
    if ! wait_idle; then echo "$(date -u +%FT%TZ) SKIP rep=$rep $name" >> "$LOG"; continue; fi
    echo "$(date -u +%FT%TZ) rep=$rep $name: env[$envs] $args" >> "$LOG"
    env $envs CUDA_VISIBLE_DEVICES=$G "${PIN[@]}" "$BIN" $args \
      --tag "$LABEL:$name:rep$rep" --out "$CSV" > /dev/null 2>> "$LOG" \
      || echo "$(date -u +%FT%TZ) FAIL rc=$? rep=$rep $name" >> "$LOG"
  done
done
echo "$(date -u +%FT%TZ) DONE" >> "$LOG"
NFAIL=$(grep -c -E ' (FAIL|SKIP) ' "$LOG")
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$LABEL" "$G" "${OUT#$ROOT/}" "${COMMIT:0:12}" \
  "$(cut -c1-12 "$OUT/binary.sha256" | head -1)" "$((N * REPEATS))" "$NFAIL" >> "$OUTBASE/MANIFEST.tsv"
echo "$OUT"
