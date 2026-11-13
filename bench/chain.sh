#!/usr/bin/env bash
# Whole final collection for one GPU, back to back: P1 -> P4 -> P2 -> P3.
# Usage: bench/chain.sh GPU MODEL CPUS "P1 phases" P4(0|1) "P2 phases" P3(0|1)
#   e.g. bench/chain.sh 8 4090 36-47 "P1a P1b P1c" 1 "P2a P2b P2c P2d" 1
# Log: results/final/chain_gpu<G>.log
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
G=$1; MODEL=$2; CPUS=$3; P1=$4; P4=$5; P2=$6; P3=$7
LOG=$ROOT/results/final/chain_gpu$G.log
mkdir -p "$ROOT/results/final"
say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
say "chain start: gpu=$G model=$MODEL cpus=$CPUS P1=[$P1] P4=$P4 P2=[$P2] P3=$P3 commit=$(git -C "$ROOT" rev-parse --short HEAD)"
[ -n "$P1" ] && "$ROOT/bench/collect.sh" "$G" "$MODEL" "$CPUS" $P1 && say "P1 done"
if [ "$P4" = 1 ]; then
  until [ -f "$ROOT/data/scaled8.sigma_ref.npy" ]; do sleep 60; done   # inputs from workloads/make_inputs.py
  say "P4 start"; say "P4 -> $("$ROOT/workloads/run_rsvd.sh" "$G" "$CPUS" 2>> "$LOG" | tail -1)"
fi
[ -n "$P2" ] && "$ROOT/bench/collect.sh" "$G" "$MODEL" "$CPUS" $P2 && say "P2 done"
if [ "$P3" = 1 ]; then
  say "P3a -> $("$ROOT/bench/trace_p3a.sh" "$G" "$MODEL" "$CPUS" 2>> "$LOG" | tail -1)"
  say "P3b -> $("$ROOT/bench/run_p3b.sh" "$G" "$MODEL" "$CPUS" 2>> "$LOG" | tail -1)"
fi
say "CHAIN DONE"
