"""CPU reference for the overflow case (P2e): what NumPy's BLAS returns for the overflowing entry C[r, c].

Builds exactly the same inputs as workloads/overflow_class.py (same generator, seed, scaling) for each seed and
reports the class of C[r, c] from NumPy's matmul (the BLAS NumPy links, OpenBLAS in our venv), the sign of the exact
result, and how many of the k products overflow to +Inf / -Inf on their own.

Usage: python workloads/overflow_cpu.py --seeds 20261108,1,2,3,4,5,6 [--n 2048] --out cpu_openblas.csv
"""
import argparse
import csv

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", required=True)
ap.add_argument("--n", type=int, default=2048)
ap.add_argument("--out", required=True)
args = ap.parse_args()

n = args.n
r, c = n // 3, n // 2
cfg = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
with open(args.out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["seed", "n", "blas", "C_rc", "class_rc", "exact_sign", "products_pos_inf", "products_neg_inf",
                "products_finite"])
    for s in (int(x) for x in args.seeds.split(",")):
        rng = np.random.default_rng(s)
        A = rng.uniform(-1, 1, (n, n))
        B = rng.uniform(-1, 1, (n, n))
        sign = int(np.sign(np.dot(A[r, :], B[:, c])))
        A[r, :] *= 1e200
        B[:, c] *= 1e200
        with np.errstate(all="ignore"):
            C = A @ B
            p = A[r, :] * B[:, c]
        x = C[r, c]
        cls = "nan" if np.isnan(x) else ("+inf" if x > 0 else "-inf") if np.isinf(x) else "finite"
        w.writerow([s, n, f"{cfg.get('name')} {cfg.get('version')}", repr(float(x)), cls, sign,
                    int(np.sum(p == np.inf)), int(np.sum(p == -np.inf)), int(np.sum(np.isfinite(p)))])
        print(s, cls, sign, flush=True)
