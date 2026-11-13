"""Randomized SVD (Halko-Martinsson-Tropp, Alg. 4.4 with QR re-orthonormalisation) in CuPy.

One process = one (matrix, mode). cuBLAS emulation env vars are set by the caller (read at library init).
The process loops over all (l, q) configs in a seeded shuffled order; per config: WARM warm-up runs, then REPS timed
runs (wall clock, device sync at the end of each run). Accuracy metrics are computed on the CPU in NumPy float64,
so emulated GEMMs never enter the reference arithmetic. Appends one CSV row per config; saves the singular values.

GEMMs per run (A: m x n):  Y = A @ Om (skinny, m x l x n);  q x [Z = A.T @ Q (opT, n x l x m), Y = A @ Qz];
B = Q.T @ A (short-wide, l x n x m);  U = Q @ Ub.  QR via cuSOLVER (cupy.linalg.qr), small SVD of B on the GPU.

Usage: python workloads/rsvd.py --matrix data/exp.npy --mode LABEL [--dtype float64|float32] [--keep-pool]
         [--ls 64,128,256,512,1024] [--qs 0,2] [--warm 2] [--reps 5] --out results.csv --sdir svals/ --tag TAG
"""
import argparse
import csv
import os
import time

import cupy as cp
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--matrix", required=True)
ap.add_argument("--mode", required=True)
ap.add_argument("--dtype", default="float64")
ap.add_argument("--keep-pool", action="store_true")
ap.add_argument("--ls", default="64,128,256,512,1024")
ap.add_argument("--qs", default="0,2")
ap.add_argument("--warm", type=int, default=2)
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--out", required=True)
ap.add_argument("--sdir", required=True)
ap.add_argument("--tag", default="")
args = ap.parse_args()

if args.keep_pool:  # the one-line fix: keep cuBLAS's cudaMallocAsync workspace mapped across syncs
    pool = cp.cuda.runtime.deviceGetDefaultMemPool(0)
    cp.cuda.runtime.memPoolSetAttribute(pool, cp.cuda.runtime.cudaMemPoolAttrReleaseThreshold, 2**63 - 1)

name = os.path.basename(args.matrix).removesuffix(".npy")
A_h = np.load(args.matrix)  # float64 reference copy on the host
s_ref = np.load(args.matrix.replace(".npy", ".sigma_ref.npy"))
dt = np.dtype(args.dtype)
A = cp.asarray(A_h.astype(dt))
m, n = A.shape
normA = np.linalg.norm(s_ref)  # ||A||_F from the reference singular values


def rsvd(Om, q):
    Y = A @ Om
    Q, _ = cp.linalg.qr(Y)
    for _ in range(q):
        Z = A.T @ Q
        Qz, _ = cp.linalg.qr(Z)
        Y = A @ Qz
        Q, _ = cp.linalg.qr(Y)
    B = Q.T @ A
    Ub, s, Vt = cp.linalg.svd(B, full_matrices=False)
    return Q, B, Q @ Ub, s


cfgs = [(l, q) for l in map(int, args.ls.split(",")) for q in map(int, args.qs.split(","))]
np.random.default_rng(args.seed).shuffle(cfgs)
os.makedirs(args.sdir, exist_ok=True)
hdr = ["timestamp", "tag", "gpu", "matrix", "m", "n", "mode", "dtype", "keep_pool", "env_emulate", "env_strategy",
       "l", "q", "warm", "reps", "t_median_s", "t_min_s", "t_max_s", "sv_relerr_top10", "sv_abserr_max_over_s1",
       "resid_fro_rel", "resid_opt_rel", "orth_err_max", "libcublas"]
dev = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
for l, q in cfgs:
    Om = cp.asarray(np.random.default_rng(1000 + l).standard_normal((n, l)).astype(dt))
    for _ in range(args.warm):
        rsvd(Om, q)
    cp.cuda.Device().synchronize()
    # CuPy loads cuBLAS lazily: read the mapped libraries after the first GEMMs
    libs = sorted({x.split()[-1] for x in open("/proc/self/maps") if "libcublas" in x and ".so" in x})
    ts = []
    for _ in range(args.reps):
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        Q, B, U, s = rsvd(Om, q)
        cp.cuda.Device().synchronize()
        ts.append(time.perf_counter() - t0)
    # ---- accuracy on the CPU (float64), from the last timed run
    Qh, Bh, sh = (cp.asnumpy(x).astype(np.float64) for x in (Q, B, s))
    k = min(10, l)
    sv_rel = float(np.max(np.abs(sh[:k] - s_ref[:k]) / s_ref[:k]))
    sv_abs = float(np.max(np.abs(sh - s_ref[:l])) / s_ref[0])
    resid = float(np.linalg.norm(A_h - Qh @ Bh) / normA)       # ||A - Q Q^T A||_F / ||A||_F
    resid_opt = float(np.linalg.norm(s_ref[l:]) / normA)        # best rank-l (Eckart-Young)
    orth = float(np.max(np.abs(Qh.T @ Qh - np.eye(l))))
    np.save(os.path.join(args.sdir, f"{name}.{args.mode}.l{l}.q{q}.{args.tag}.npy"), sh)
    row = [time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), args.tag, dev, name, m, n, args.mode, args.dtype,
           int(args.keep_pool), os.environ.get("CUBLAS_EMULATE_DOUBLE_PRECISION", ""),
           os.environ.get("CUBLAS_EMULATION_STRATEGY", ""), l, q, args.warm, args.reps, f"{np.median(ts):.5f}",
           f"{min(ts):.5f}", f"{max(ts):.5f}", f"{sv_rel:.3e}", f"{sv_abs:.3e}", f"{resid:.3e}", f"{resid_opt:.3e}",
           f"{orth:.3e}", ";".join(libs)]
    new = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    with open(args.out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(hdr)
        w.writerow(row)
    print(f"{name} {args.mode} l={l} q={q} t={np.median(ts) * 1e3:.1f} ms sv_rel={sv_rel:.2e} resid={resid:.3e} "
          f"(opt {resid_opt:.3e}) orth={orth:.1e}", flush=True)
