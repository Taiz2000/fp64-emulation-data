#!/usr/bin/env bash
# Generate every final grid for one GPU model on that GPU. Usage: bench/grids/gen_all.sh GPU NAME CPUS
set -uo pipefail
G=$1; NAME=$2; CPUS=$3
D=$(cd "$(dirname "$0")" && pwd)/final
for P in P1a P1b P1c P2a P2b P2c P2d; do
  taskset -c "$CPUS" "$D/../make_final.sh" "$G" "$P" > "$D/${P}_$NAME.txt" 2> "$D/${P}_$NAME.probe.log"
  echo "$P $(grep -c . "$D/${P}_$NAME.txt") lines, $(grep -c SKIP "$D/${P}_$NAME.probe.log") skipped"
done
