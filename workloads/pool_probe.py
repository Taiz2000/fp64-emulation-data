"""Which CUDA memory pool does cuBLAS take its emulation workspace from? (documented vs observed)

The CUDA 13.4 Update 1 release notes (cuBLAS 13.8.0.4) say that, when the per-handle workspace is too small, FP64
emulation "allocates its temporary workspace from a cuBLAS-managed, per-device CUDA memory pool that retains memory
across stream synchronizations". Our nsys traces instead showed one cudaMallocAsync per call. This probe measures
it directly with the pools' high-water marks:

  * CuPy allocates its own arrays with cudaMalloc (its own MemoryPool), so the CUDA async pools hold only what
    cuBLAS (or something else in the process) allocates with cudaMallocAsync / cudaMallocFromPoolAsync.
  * Before every GEMM call the probe resets UsedMemHigh / ReservedMemHigh of the device's default pool (and of an
    optional custom pool) to 0, runs one GEMM, synchronizes, and reads them back.
  * If cuBLAS used its own private pool, the default pool's high-water marks would stay 0; if it allocates from the
    device's current pool (cudaMallocAsync), they rise by the workspace size on every call.
  * --pool custom installs a new pool as the device's *current* pool (cudaDeviceSetMemPool). If the allocations move
    there, cuBLAS allocates from whatever pool is current, i.e. plain cudaMallocAsync.
  * --ws-mib N gives the handle a workspace with cublasSetWorkspace (ctypes on CuPy's handle; CuPy has no wrapper):
    the control case, where no pool allocation is expected.
  * --api matmul (default) runs C = A @ B through CuPy; --api dgemm calls cublasDgemm directly through CuPy's
    low-level binding on the same handle. cuBLAS docs: cublasSetStream "unconditionally resets the cuBLAS library
    workspace back to the default workspace pool", so if CuPy sets the stream per call, a workspace set by the user
    is lost on every call. CuPy's low-level wrappers (cupy_backends ... cublas._setStream, which imports
    cublasSetStream_v2) appear to set the stream before each call, so --api ctypes calls cublasDgemm_v2 through ctypes
    on the same handle, bypassing CuPy: there a workspace set with cublasSetWorkspace should hold.

cuBLAS reads the emulation env vars at init, so one process = one mode; the caller sets the env.
One CSV row per call (call 0 includes cuBLAS initialisation and is reported, not dropped).

Usage: python workloads/pool_probe.py --label L [--n 4096] [--calls 6] [--pool default|custom]
                                      [--threshold zero|max] [--ws-mib 0] [--api matmul|dgemm|ctypes] --out file.csv
"""
import argparse
import csv
import ctypes
import os
import time

import cupy as cp
import numpy as np
import cupy.cuda.runtime as rt
from cupy_backends.cuda.libs import cublas

ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True)
ap.add_argument("--n", type=int, default=4096)
ap.add_argument("--calls", type=int, default=6)
ap.add_argument("--pool", choices=["default", "custom"], default="default")
ap.add_argument("--threshold", choices=["zero", "max"], default="zero")
ap.add_argument("--ws-mib", type=int, default=0)
ap.add_argument("--api", choices=["matmul", "dgemm", "ctypes"], default="matmul")
ap.add_argument("--seed", type=int, default=20261108)
ap.add_argument("--out", required=True)
args = ap.parse_args()

MiB = 2**20
dev = 0
thr = 0 if args.threshold == "zero" else 2**63 - 1
default_pool = rt.deviceGetDefaultMemPool(dev)
custom_pool = None
if args.pool == "custom":
    props = rt.MemPoolProps(rt.cudaMemAllocationTypePinned, rt.cudaMemHandleTypeNone, rt.cudaMemLocationTypeDevice, dev)
    custom_pool = rt.memPoolCreate(props)
    rt.memPoolSetAttribute(custom_pool, rt.cudaMemPoolAttrReleaseThreshold, thr)
    rt.deviceSetMemPool(dev, custom_pool)
else:
    rt.memPoolSetAttribute(default_pool, rt.cudaMemPoolAttrReleaseThreshold, thr)

handle = cp.cuda.device.get_cublas_handle()
rs = cp.random.RandomState(args.seed)
n = args.n
A = rs.uniform(-1, 1, (n, n))  # float64
B = rs.uniform(-1, 1, (n, n))
C = cp.empty((n, n))
alpha, beta = np.array(1.0), np.array(0.0)  # host pointer mode (cuBLAS default)
cp.cuda.Device().synchronize()
lib = ctypes.CDLL("libcublas.so.13")  # same soname CuPy loaded, so the same library instance (checked)
lib.cublasDgemm_v2.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                               ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                               ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
ws_buf = None
if args.ws_mib:
    ws_buf = cp.empty(args.ws_mib * MiB, dtype=cp.uint8)  # cudaMalloc via CuPy's pool, not an async pool
    lib.cublasSetWorkspace_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    st = lib.cublasSetWorkspace_v2(ctypes.c_void_p(handle), ctypes.c_void_p(ws_buf.data.ptr),
                                   ctypes.c_size_t(args.ws_mib * MiB))
    assert st == 0, f"cublasSetWorkspace_v2 status {st}"


def gemm():
    global C
    if args.api == "matmul":
        C = A @ B
    elif args.api == "dgemm":  # column-major view of row-major arrays: C^T = B^T A^T, square so lda = ldb = ldc = n
        cublas.dgemm(handle, cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_N, n, n, n, alpha.ctypes.data, B.data.ptr, n,
                     A.data.ptr, n, beta.ctypes.data, C.data.ptr, n)
    else:  # same call, straight into libcublas (no CuPy wrapper, so no cublasSetStream before it)
        st = lib.cublasDgemm_v2(handle, 0, 0, n, n, n, alpha.ctypes.data, B.data.ptr, n, A.data.ptr, n,
                                beta.ctypes.data, C.data.ptr, n)
        assert st == 0, f"cublasDgemm_v2 status {st}"


pools = {"default": default_pool, "custom": custom_pool}
attrs = {"used_high": rt.cudaMemPoolAttrUsedMemHigh, "reserved_high": rt.cudaMemPoolAttrReservedMemHigh,
         "reserved_cur": rt.cudaMemPoolAttrReservedMemCurrent}


def reset():
    for p in pools.values():
        if p is not None:
            rt.memPoolSetAttribute(p, rt.cudaMemPoolAttrUsedMemHigh, 0)
            rt.memPoolSetAttribute(p, rt.cudaMemPoolAttrReservedMemHigh, 0)


def read():
    out = {}
    for name, p in pools.items():
        for a, code in attrs.items():
            out[f"{name}_{a}_mib"] = "" if p is None else f"{rt.memPoolGetAttribute(p, code) / MiB:.1f}"
    return out


base = dict(label=args.label, gpu=rt.getDeviceProperties(dev)["name"].decode(),
            cublas_version=cublas.getVersion(handle), cupy=cp.__version__,
            env_emulate=os.environ.get("CUBLAS_EMULATE_DOUBLE_PRECISION", ""),
            env_strategy=os.environ.get("CUBLAS_EMULATION_STRATEGY", ""),
            n=n, api=args.api, pool=args.pool, threshold=args.threshold, ws_mib=args.ws_mib)
rows = []
for i in range(args.calls):
    reset()
    free0 = rt.memGetInfo()[0]
    t0 = time.perf_counter()
    gemm()
    cp.cuda.Device().synchronize()
    ms = 1e3 * (time.perf_counter() - t0)
    row = dict(base, call=i, wall_ms=f"{ms:.3f}", **read(),
               devfree_drop_mib=f"{(free0 - rt.memGetInfo()[0]) / MiB:.1f}",
               current_pool=("custom" if rt.deviceGetMemPool(dev) == custom_pool else
                             "default" if rt.deviceGetMemPool(dev) == default_pool else "other"))
    # sanity check that the call computed A @ B: one entry against a host dot product
    i0, j0 = (7 * i) % n, (7 * i + 3) % n  # off-diagonal, so a transposed result would show
    ref = float(np.dot(cp.asnumpy(A[i0, :]), cp.asnumpy(B[:, j0])))
    row["check_relerr"] = f"{abs(float(C[i0, j0]) - ref) / abs(ref):.1e}"
    rows.append(row)
    print(row, flush=True)

new = not os.path.exists(args.out)
with open(args.out, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    if new:
        w.writeheader()
    w.writerows(rows)
