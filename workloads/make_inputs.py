"""Build the randomized-SVD test matrices. Run WITHOUT any CUBLAS_EMULATE* env var.

A = U diag(sigma) V^T, U (m x n) and V (n x n) orthonormal from QR of Gaussian matrices (GPU, native FP64).
Families (Tropp et al. 2017 style):
  exp      sigma_i = 1 for i < 10, then 10^(-0.25 (i - 9))
  poly     sigma_i = 1 for i < 10, then (i - 8)^-2
  lrnoise  rank 50 (sigma = 1) plus 1e-10 * G / sqrt(m)
  scaled4 / scaled8   exp times column scaling diag(10^(phi t_j)), t_j ~ U[-1, 1]: exponent range along the
           inner dimension of A @ Omega (stresses the emulation's bit requirement / fallback)
Writes data/<name>.npy (float64, C-order m x n), data/<name>.sigma_exact.npy (construction; NaN where unknown),
data/<name>.sigma_ref.npy (CPU LAPACK dgesdd on the stored matrix: the accuracy reference).

Usage: python workloads/make_inputs.py [--m 16384] [--n 8192] [--only name,...]
"""
import argparse
import os
import sys
import time

if any(k.startswith("CUBLAS_EMULAT") for k in os.environ):
    sys.exit("refusing to build inputs with cuBLAS emulation env vars set")

import cupy as cp
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--m", type=int, default=16384)
ap.add_argument("--n", type=int, default=8192)
ap.add_argument("--only", default="")
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data"))
args = ap.parse_args()
m, n = args.m, args.n
os.makedirs(args.out, exist_ok=True)
rng = np.random.default_rng(20261107)


def orth(rows, cols):
    g = cp.asarray(rng.standard_normal((rows, cols)))
    q, r = cp.linalg.qr(g, mode="reduced")
    return q * cp.sign(cp.diag(r))[None, :]  # make the factorization unique


i = np.arange(n, dtype=np.float64)
s_exp = np.where(i < 10, 1.0, 10.0 ** (-0.25 * (i - 9)))
s_poly = np.where(i < 10, 1.0, (i - 8.0) ** -2)

t0 = time.time()
U = orth(m, n)
V = orth(n, n)
print(f"orthonormal factors: {time.time() - t0:.1f} s", flush=True)

col_t = rng.uniform(-1, 1, n)
mats = {}
mats["exp"] = (lambda: (U * cp.asarray(s_exp)[None, :]) @ V.T, s_exp)
mats["poly"] = (lambda: (U * cp.asarray(s_poly)[None, :]) @ V.T, s_poly)


def lrnoise():
    g = cp.asarray(rng.standard_normal((m, n))) * (1e-10 / np.sqrt(m))
    return U[:, :50] @ V[:, :50].T + g


mats["lrnoise"] = (lrnoise, np.full(n, np.nan))
for phi in (4, 8):
    d = 10.0 ** (phi * col_t)
    mats[f"scaled{phi}"] = (lambda d=d: ((U * cp.asarray(s_exp)[None, :]) @ V.T) * cp.asarray(d)[None, :],
                            np.full(n, np.nan))

only = set(filter(None, args.only.split(",")))
for name, (build, s_exact) in mats.items():
    if only and name not in only:
        continue
    t0 = time.time()
    A = cp.asnumpy(build())
    np.save(os.path.join(args.out, f"{name}.npy"), np.ascontiguousarray(A))
    np.save(os.path.join(args.out, f"{name}.sigma_exact.npy"), s_exact)
    t1 = time.time()
    s_ref = np.linalg.svd(A, compute_uv=False)
    np.save(os.path.join(args.out, f"{name}.sigma_ref.npy"), s_ref)
    k = ~np.isnan(s_exact)
    dev = np.max(np.abs(s_ref[k] - s_exact[k])) if k.any() else float("nan")
    print(f"{name}: build {t1 - t0:.1f} s, dgesdd {time.time() - t1:.1f} s, sigma_1={s_ref[0]:.6g}, "
          f"sigma_200={s_ref[200]:.3e}, max|ref-exact|={dev:.2e}", flush=True)
