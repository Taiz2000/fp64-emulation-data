#!/usr/bin/env bash
# Step 0: capture the full environment into env/<timestamp>/ .
# Usage: env/capture.sh [gpu indices...]   (default: $PATHB_GPUS from toolchain.sh)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/toolchain.sh"
GPUS=${*:-$PATHB_GPUS}
TS=$(date -u +%Y%m%dT%H%M%SZ)
OUT=$HERE/$TS
mkdir -p "$OUT"

# Host
{
  echo "timestamp_utc=$TS"; echo "hostname=$(hostname)"; echo "kernel=$(uname -r)"
  grep PRETTY_NAME /etc/os-release
  lscpu | grep -E "Model name|^CPU\(s\)|Socket|NUMA node\(s\)"
  free -g | head -2
} > "$OUT/host.txt"

# Toolchain
{
  echo "CUDA_ROOT=$CUDA_ROOT"
  echo "/usr/local/cuda -> $(readlink -f /usr/local/cuda)"
  "$NVCC" --version
  python3 -c "import json;d=json.load(open('$CUDA_ROOT/version.json'));[print(k,d[k]['version']) for k in sorted(d)]"
  ls -l "$CUDA_LIB"/libcublas.so* "$CUDA_LIB"/libcublasLt.so*
  sha256sum "$CUDA_LIB"/libcublas.so.13.* "$CUDA_LIB"/libcublasLt.so.13.*
  nsys --version; "$CUDA_ROOT"/bin/ncu --version | tail -1
} > "$OUT/toolchain.txt" 2>&1

# GPUs: whole node summary + full -q for the GPUs under test
nvidia-smi > "$OUT/nvidia-smi.txt"
nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,compute_cap,vbios_version,pcie.link.gen.max,pcie.link.width.max,pcie.link.width.current,memory.total,memory.used,utilization.gpu,pstate,clocks.sm,clocks.max.sm,clocks.mem,clocks.max.mem,power.limit,power.default_limit,power.max_limit,temperature.gpu,persistence_mode,compute_mode \
  --format=csv > "$OUT/gpus.csv"
nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv > "$OUT/gpu_procs.csv"
nvidia-smi topo -m > "$OUT/topo.txt"
for g in $GPUS; do nvidia-smi -q -i "$g" > "$OUT/nvidia-smi-q.gpu$g.txt"; done

# Build probe
"$NVCC" $NVCC_FLAGS "$HERE/probe.cu" -o "$HERE/probe" -lcublas
ldd "$HERE/probe" > "$OUT/probe.ldd.txt"

# Probe each GPU under each env-var configuration
for g in $GPUS; do
  for cfg in none emu emu_eager; do
    case $cfg in
      none)      envs=() ;;
      emu)       envs=(CUBLAS_EMULATE_DOUBLE_PRECISION=1) ;;
      emu_eager) envs=(CUBLAS_EMULATE_DOUBLE_PRECISION=1 CUBLAS_EMULATION_STRATEGY=EAGER) ;;
    esac
    env "${envs[@]}" CUDA_VISIBLE_DEVICES=$g "$HERE/probe" > "$OUT/probe.gpu$g.$cfg.txt" 2>&1
  done
done

echo "$OUT"
