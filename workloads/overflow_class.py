"""Which non-finite value does the overflowing entry get, natively and under emulation? (follow-up to P2d)

Reproduces the bench's `--inject overflow` case (bench/dgemm_bench.cu, inject()): A (m x k) and B (k x n) uniform in
[-1, 1); row r = m/3 of A and column c = n/2 of B are scaled by 1e200, so every product in C[r, c] overflows and
only that entry of C is non-finite. The random values differ from the bench's (NumPy, not the bench's generator);
the class of C[r, c] depends on the mechanism, which this script reports.

cuBLAS reads the emulation env vars at init, so one process = one mode; the caller sets the env.
Prints one CSV row: mode, time per call, class of C[r, c], the sign of the exact sum (computed from the unscaled
values in float64; the scaled exact value is 1e400 times it), and whether every other entry is finite.

Usage: python workloads/overflow_class.py --mode LABEL [--n 2048] [--out file.csv]
"""
import argparse
import csv
import os
import time

import cupy as cp
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True)
ap.add_argument("--n", type=int, default=2048)
ap.add_argument("--seed", type=int, default=20261108)
ap.add_argument("--out", default="")
args = ap.parse_args()

# keep the cuBLAS workspace mapped, so the timing shows the path, not the mempool remap (C3)
pool = cp.cuda.runtime.deviceGetDefaultMemPool(0)
cp.cuda.runtime.memPoolSetAttribute(pool, cp.cuda.runtime.cudaMemPoolAttrReleaseThreshold, 2**63 - 1)

m = n = k = args.n
r, c = m // 3, n // 2
rng = np.random.default_rng(args.seed)
A0 = rng.uniform(-1, 1, (m, k))
B0 = rng.uniform(-1, 1, (k, n))
exact_sign = np.sign(np.dot(A0[r, :], B0[:, c]))  # sign of the true (scaled) C[r, c]
A, B = A0.copy(), B0.copy()
A[r, :] *= 1e200
B[:, c] *= 1e200
dA, dB = cp.asarray(A), cp.asarray(B)


def cls(x):
    return "nan" if np.isnan(x) else ("+inf" if x > 0 else "-inf") if np.isinf(x) else "finite"


C = dA @ dB  # warm-up (also loads cuBLAS)
cp.cuda.Device().synchronize()
ts = []
for _ in range(5):
    t0 = time.perf_counter()
    C = dA @ dB
    cp.cuda.Device().synchronize()
    ts.append(time.perf_counter() - t0)
Ch = cp.asnumpy(C)
others = np.isfinite(np.delete(Ch.ravel(), r * n + c)).all()
row = dict(mode=args.mode, gpu=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
           env_emulate=os.environ.get("CUBLAS_EMULATE_DOUBLE_PRECISION", ""),
           env_strategy=os.environ.get("CUBLAS_EMULATION_STRATEGY", ""),
           env_svmask=os.environ.get("CUBLAS_EMULATION_SPECIAL_VALUES_SUPPORT_MASK", ""),
           n=n, ms_per_call=f"{1e3 * float(np.median(ts)):.3f}", C_rc=repr(float(Ch[r, c])), class_rc=cls(Ch[r, c]),
           exact_sign=int(exact_sign), other_entries_finite=bool(others))
print(row)
if args.out:
    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
