#!/usr/bin/env bash
# Run final-collection phases back to back on one GPU, QA after each.
# Usage: bench/collect.sh GPU MODEL CPUS PHASE...      MODEL: 3090 | 4090 | 5060ti (selects grid files)
#   PHASE: P1a P1b P1c P2a P2b P2c P2d  (repeats: P2a/P2b 1, others 3)
# Log: results/final/collect_gpu<G>.log
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
G=$1; MODEL=$2; CPUS=$3; shift 3
LOG=$ROOT/results/final/collect_gpu$G.log
mkdir -p "$ROOT/results/final"
declare -A LABEL=([P1a]=P1a_square [P1b]=P1b_shapes [P1c]=P1c_ranges [P2a]=P2a_decision [P2b]=P2b_version
                  [P2c]=P2c_bits [P2d]=P2d_edges)
declare -A REPS=([P1a]=3 [P1b]=3 [P1c]=3 [P2a]=1 [P2b]=1 [P2c]=3 [P2d]=3)
for P in "$@"; do
  GRID=$ROOT/bench/grids/final/${P}_$MODEL.txt
  echo "$(date -u +%FT%TZ) start $P ($(grep -c . "$GRID") lines x ${REPS[$P]})" >> "$LOG"
  OUT=$(REPEATS=${REPS[$P]} CPUS=$CPUS "$ROOT/bench/run_grid.sh" "$G" "$GRID" "${LABEL[$P]}" 2>> "$LOG" | tail -1)
  echo "$(date -u +%FT%TZ) end $P -> $OUT" >> "$LOG"
  python3 "$ROOT/bench/qa.py" "$OUT" >> "$LOG" 2>&1
done
echo "$(date -u +%FT%TZ) ALL DONE" >> "$LOG"
