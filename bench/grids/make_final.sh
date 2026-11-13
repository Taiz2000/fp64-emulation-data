#!/usr/bin/env bash
# Generate a final-collection grid for one GPU and one phase.
# Workspace for the *_ws modes is probed per shape ON THE TARGET GPU: cuBLASLt-requested size under EAGER + 32 MiB.
# Points whose operands + workspace don't fit (total memory - 1 GiB) are skipped (listed on stderr).
# Usage: bench/grids/make_final.sh GPU PHASE > grid.txt     PHASE: P1a P1b P1c P2a P2b P2c P2d P2e
#   env: BIN (bench/bin/dgemm_bench_v5)
set -uo pipefail
G=${1:?gpu}; PHASE=${2:?phase}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/env/toolchain.sh"
BIN=${BIN:-$ROOT/bench/bin/dgemm_bench_v5}
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$G")
LIMIT_MIB=$((MEM_MIB - 1024))
EMU="CUBLAS_EMULATE_DOUBLE_PRECISION=1"
EAGER="CUBLAS_EMULATE_DOUBLE_PRECISION=1 CUBLAS_EMULATION_STRATEGY=EAGER"

declare -A WSCACHE

# probe "<bench args>" -> workspace MiB (requested + 32), cached per args string
probe() {
  local a="$1" key="$EXTRA_ENV|$1" f b
  [ -n "${WSCACHE[$key]:-}" ] && { echo "${WSCACHE[$key]}"; return; }
  f=$T/p$RANDOM.log
  # strip per-line output flags from the probe (they reference @OUT@)
  a=$(sed -E 's/--kern-out [^ ]+//' <<< "$a")
  env CUDA_VISIBLE_DEVICES=$G $EXTRA_ENV $EAGER CUBLASLT_LOG_LEVEL=5 CUBLASLT_LOG_FILE=$f OMP_NUM_THREADS=16 "$BIN" $a \
    --no-ref --no-cupti --no-wall --samples 16 --warmup 0 --warmup-s 0 --reps 1 > /dev/null 2>&1
  b=$(grep Trace "$f" 2>/dev/null | grep -m1 -o 'workSpaceSizeInBytes=[0-9]*' | cut -d= -f2)
  WSCACHE[$key]=$(((${b:-0} + 1048575) / 1048576 + 32))
  echo "# probe [$EXTRA_ENV $a] requested=${b:-unknown} -> ${WSCACHE[$key]} MiB" >&2
  echo "${WSCACHE[$key]}"
}
# Per-line options, set by the phase: EXTRA_ENV (prepended to every line's env, e.g. LD_LIBRARY_PATH),
# LTLOG=1 (cuBLASLt trace log per line), KERN=1 (CUPTI kernel dump per line). Files are named after the line.
EXTRA_ENV=""; LTLOG=0; KERN=0
line() {  # name env args
  local name=$1 e=$2 a=$3
  [ "$LTLOG" = 1 ] && e="$e CUBLASLT_LOG_LEVEL=5 CUBLASLT_LOG_FILE=@OUT@/lt/$name.r@REP@.log"
  [ "$KERN" = 1 ] && a="$a --kern-out @OUT@/kern/$name.r@REP@.tsv"
  e="$EXTRA_ENV $e"
  echo "$name|$(sed -E 's/^ +| +$//g; s/ +/ /g' <<< "$e")|$a"
}
# fits m n k batch ws_mib -> 0 if operands (+ C0 when beta) + workspace fit
fits() {
  local need=$(( (8 * ($1 * $3 + $3 * $2 + 2 * $1 * $2) * $4) / 1048576 + $5 ))
  [ "$need" -le "$LIMIT_MIB" ]
}
# emit PREFIX ID "<shape args>" "<extra args>" MODE...   modes: native default_ws0 default_kp default_ws eager_ws0 eager_kp eager_ws
#   plus any "name=ENV" custom mode handled by the caller via emit_raw
emit() {
  local pre=$1 id=$2 S=$3 X=$4; shift 4
  local m n k nb ws
  m=$(sed -E 's/.*--m ([0-9]+).*/\1/' <<< "$S"); n=$(sed -E 's/.*--n ([0-9]+).*/\1/' <<< "$S")
  k=$(sed -E 's/.*--k ([0-9]+).*/\1/' <<< "$S"); nb=1
  [[ "$S $X" =~ --batch\ ([0-9]+) ]] && nb=${BASH_REMATCH[1]}
  ws=$(probe "$S $X")
  if ! fits "$m" "$n" "$k" "$nb" "$ws"; then echo "# SKIP $pre:$id (needs > $LIMIT_MIB MiB)" >&2; return; fi
  for mode in "$@"; do
    case $mode in
      native)        line "$pre:native:$id" "" "$S $X" ;;
      default_ws0)   line "$pre:default_ws0:$id" "$EMU" "$S $X" ;;
      default_kp)    line "$pre:default_kp:$id" "$EMU" "$S $X --keep-pool" ;;
      default_ws)    line "$pre:default_ws:$id" "$EMU" "$S $X --workspace-mb $ws" ;;
      eager_ws0)     line "$pre:eager_ws0:$id" "$EAGER" "$S $X" ;;
      eager_kp)      line "$pre:eager_kp:$id" "$EAGER" "$S $X --keep-pool" ;;
      eager_ws)      line "$pre:eager_ws:$id" "$EAGER" "$S $X --workspace-mb $ws" ;;
      eager_ws_mask0) line "$pre:eager_ws_mask0:$id" "$EAGER CUBLAS_EMULATION_SPECIAL_VALUES_SUPPORT_MASK=0" \
                       "$S $X --workspace-mb $ws" ;;
      *) echo "bad mode $mode" >&2; exit 1 ;;
    esac
  done
}
sh3() { echo "--m $1 --n $2 --k $3 --api dgemm"; }
ALL7="native default_ws0 default_kp default_ws eager_ws0 eager_kp eager_ws"
SH5="native default_ws0 default_ws eager_ws0 eager_ws"
R3="native default_ws eager_ws"

case $PHASE in
P1a)  # square, U[-1,1)
  for n in 512 1024 1536 2048 3072 4096 6144 8192 12288 16384; do emit sq "n$n" "$(sh3 $n $n $n)" "" $ALL7; done ;;
P1b)  # extended shapes
  for s in "square 4096 4096 4096" "square 8192 8192 8192" "odd 4097 4097 4097" "odd 4000 4000 4000" "odd 3001 5003 2999"; do
    set -- $s; emit shape "$1:m$2_n$3_k$4" "$(sh3 $2 $3 $4)" "" $SH5; done
  for k in 16 32 48 64 96 128 192 256 512 1024 2048; do emit shape "panel:m8192_n8192_k$k" "$(sh3 8192 8192 $k)" "" $SH5; done
  for n in 8 16 32 64 96 128 192 256 384 512 1024 2048; do emit shape "skinny:m8192_n${n}_k8192" "$(sh3 8192 $n 8192)" "" $SH5; done
  for m in 8 32 64 128 256 512; do emit shape "shortwide:m${m}_n8192_k8192" "$(sh3 $m 8192 8192)" "" $SH5; done
  for s in 32 128 512; do for k in 8192 32768 131072; do emit shape "gram:m${s}_n${s}_k$k" "$(sh3 $s $s $k)" "" $SH5; done; done
  for l in 64 256; do  # randomized-SVD GEMMs for A (16384 x 8192): A*Omega, A^T*Q, Q^T*A
    emit shape "rsvd_AO:m16384_n${l}_k8192" "$(sh3 16384 $l 8192)" "" $SH5
    emit shape "rsvd_AtQ:m8192_n${l}_k16384_TN" "$(sh3 8192 $l 16384)" "--opa T" $SH5
    emit shape "rsvd_QtA:m${l}_n8192_k16384_TN" "$(sh3 $l 8192 16384)" "--opa T" $SH5
  done ;;
P1c)  # value ranges and the fallback cliff
  for phi in 0 0.5 1 2 3 4 6 8; do emit range "ozaki:phi$phi:k4096" "$(sh3 4096 4096 4096)" "--dist ozaki --phi $phi" $R3; done
  for phi in 1 2 4 6 7 7.5 7.75 8 12; do emit range "inner:phi$phi:k4096" "$(sh3 4096 4096 4096)" "--dist inner --phi $phi" $R3; done
  for phi in 2 4 8 16; do emit range "rowcol:phi$phi:k4096" "$(sh3 4096 4096 4096)" "--dist rowcol --phi $phi" $R3; done
  for k in 256 1024 16384; do for phi in 4 6 7 7.5 7.75 8; do
    emit range "inner:phi$phi:k$k" "$(sh3 4096 4096 $k)" "--dist inner --phi $phi" $R3; done; done ;;
P2a)  # heuristic decision map (path only; Lt log per point). Run with REPEATS=1.
  Q="--no-ref --no-wall --reps 3 --warmup-s 0.2"
  pts() {
    for m in 16 64 256 1024 4096 16384; do for n in 16 64 256 1024 4096 16384; do for k in 16 64 256 1024 4096 16384; do
      echo "cube $m $n $k"; done; done; done
    for n in 256 362 512 724 1024 1448 2048 2896 4096 5793 8192 11585 16384; do echo "sq $n $n $n"; done
    for k in 16 23 32 45 64 91 128 181 256 362 512 724 1024 1448 2048 2896 4096 5793 8192; do echo "panel 8192 8192 $k"; done
    for n in 8 11 16 23 32 45 64 91 128 181 256 362 512 724 1024 1448 2048 2896 4096 5793 8192; do echo "skinny 8192 $n 8192"; done
    for m in 8 11 16 23 32 45 64 91 128 181 256 362 512 724 1024 1448 2048 2896 4096 5793 8192; do echo "shortwide $m 8192 8192"; done
  }
  LTLOG=1; KERN=1
  while read -r fam m n k; do
    emit dec "$fam:m${m}_n${n}_k$k" "$(sh3 $m $n $k)" "$Q" default_ws0 default_ws eager_ws
  done < <(pts) ;;
P2b)  # cuBLAS version axis (LD_LIBRARY_PATH; binary uses RUNPATH). Run with REPEATS=1.
  LTLOG=1; KERN=1
  for tk in 13.1 13.2 13.3 13.4; do
    EXTRA_ENV="LD_LIBRARY_PATH=/usr/local/cuda-$tk/targets/x86_64-linux/lib"
    v() { local id=$1 S=$2 X=$3; shift 3; emit "ver$tk" "$id" "$S" "$X" "$@"; }
    for n in 2048 4096 8192; do v "sq:n$n" "$(sh3 $n $n $n)" "" native default_ws0 default_ws eager_ws; done
    v "panel:k128" "$(sh3 8192 8192 128)" "" default_ws0 default_ws
    for n in 64 512; do v "skinny:n$n" "$(sh3 8192 $n 8192)" "" default_ws0 default_ws; done
    for phi in 7.5 7.75 8; do v "inner:phi$phi" "$(sh3 4096 4096 4096)" "--dist inner --phi $phi" default_ws eager_ws; done
    Q="--no-ref --no-wall --reps 3 --warmup-s 0.2"
    for n in 256 512 1024 2048 4096 8192 16384; do v "dec_sq:n$n" "$(sh3 $n $n $n)" "$Q" default_ws; done
    for k in 16 32 64 128 256 512 1024 2048 4096 8192; do v "dec_panel:k$k" "$(sh3 8192 8192 $k)" "$Q" default_ws; done
    for n in 8 16 32 64 128 256 512 1024 2048 4096 8192; do v "dec_skinny:n$n" "$(sh3 8192 $n 8192)" "$Q" default_ws; done
  done ;;
P2c)  # precision knob at 4096^3, EAGER + workspace
  for d in "u11" "inner --phi 6"; do
    did=$(tr ' ' '_' <<< "${d//--phi /phi}")
    emit bits "$did:ref" "$(sh3 4096 4096 4096)" "--dist $d" native eager_ws
    for b in 24 32 40 48 53 56 64 72 80 96; do
      emit bits "$did:fixed$b" "$(sh3 4096 4096 4096)" "--dist $d --mcontrol fixed --max-bits $b" eager_ws; done
    for b in 64 79; do emit bits "$did:dyncap$b" "$(sh3 4096 4096 4096)" "--dist $d --mcontrol dynamic --max-bits $b" eager_ws; done
  done ;;
P2d)  # correctness edges and API variants
  KERN=1
  for inj in nan inf ninf subnormal huge overflow zeroblock; do
    emit edge "inject:$inj" "$(sh3 2048 2048 2048)" "--inject $inj" $R3 eager_ws_mask0
  done
  KERN=0
  for op in NN NT TN TT; do
    emit edge "op:sq4096:$op" "$(sh3 4096 4096 4096)" "--opa ${op:0:1} --opb ${op:1:1}" native default_ws0 default_ws eager_ws
    emit edge "op:skinny256:$op" "$(sh3 8192 256 8192)" "--opa ${op:0:1} --opb ${op:1:1}" native default_ws0 default_ws eager_ws
  done
  emit edge "beta:panel256" "$(sh3 8192 8192 256)" "--alpha -1 --beta 1" native default_ws0 default_ws eager_ws
  for n in 512 1024; do emit edge "batch64:n$n" "$(sh3 $n $n $n)" "--batch 64" native default_ws0 default_ws eager_ws; done ;;
P2e)  # follow-ups from the off-node review (2026-11-08). Run with REPEATS=1 (paths and bits are deterministic).
  # (1) where exactly the ADP runtime fallback starts: fine phi steps around the cliff, across k
  KERN=1
  for k in 256 512 1024 2048 4096 8192 16384; do for phi in 7.25 7.5 7.6 7.7 7.75 7.8 7.9 8; do
    emit cliff "inner:phi$phi:k$k" "$(sh3 4096 4096 $k)" "--dist inner --phi $phi --no-wall" eager_ws; done; done
  # (2) why the dynamic cap changes speed at equal bits used: per-kernel times (u11, 4096^3)
  emit cap "u11:ref" "$(sh3 4096 4096 4096)" "--dist u11" eager_ws
  for b in 64 79; do emit cap "u11:dyncap$b" "$(sh3 4096 4096 4096)" "--dist u11 --mcontrol dynamic --max-bits $b" eager_ws; done
  for b in 53 64 96; do emit cap "u11:fixed$b" "$(sh3 4096 4096 4096)" "--dist u11 --mcontrol fixed --max-bits $b" eager_ws; done ;;
*) echo "unknown phase $PHASE" >&2; exit 1 ;;
esac
