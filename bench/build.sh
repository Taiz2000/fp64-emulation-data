#!/usr/bin/env bash
# Usage: bench/build.sh [output]   (default bench/dgemm_bench). Never rebuild a binary a running job uses:
# pass a new output name instead (e.g. bench/bin/dgemm_bench_v2).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../env/toolchain.sh"
OUTBIN=${1:-$HERE/dgemm_bench}
mkdir -p "$(dirname "$OUTBIN")"
"$NVCC" $NVCC_FLAGS -Xcompiler -fopenmp -Xcompiler -Wall -Xcompiler -Wno-deprecated-declarations \
  -L"$CUDA_LIB/stubs" "$HERE/dgemm_bench.cu" -o "$OUTBIN" -lcublas -lnvidia-ml -lcupti -lgomp
cp "$HERE/dgemm_bench.cu" "$OUTBIN.cu"   # keep the exact source next to each binary
ldd "$OUTBIN" | grep -E "cublas|nvidia-ml"
