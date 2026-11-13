# FP64 emulation in cuBLAS on consumer GPUs: data and code

This repository holds the benchmark, the workload, the raw results, the environment snapshots, and the analysis
scripts behind the paper *FP64 Emulation in cuBLAS on Consumer GPUs: Analysing What a Single Switch Delivers*. The
study measures cuBLAS's automatic FP64 emulation (`CUBLAS_EMULATE_DOUBLE_PRECISION=1`) on an RTX 3090, an RTX 4090,
and an RTX 5060 Ti, using cuBLAS 13.8.0.4 from CUDA 13.4 Update 1.

Every number in the paper appears in `summary/LEDGER.md` together with the files it comes from, and the scripts in
`analysis/` regenerate that ledger, the summary tables, and the data behind every figure from `results/final/`.

## Platform

- **GPUs:** RTX 3090 (two identical boards; each run used one), RTX 4090 (modified to 48 GB of memory), and RTX 5060 Ti
  (8 GB, in a PCIe Gen3 x8 slot), in one server with an AMD EPYC 7C13 processor, 512 GB of memory, and Ubuntu 22.04.
- **Software:** driver 615.71.09 and CUDA 13.4 Update 1 with cuBLAS 13.8.0.4. For the comparison of versions (P2b),
  the grids load cuBLAS 13.2.2.2, 13.4.1.3, and 13.6.0.2 from CUDA 13.1, 13.2, and 13.3 installations through
  `LD_LIBRARY_PATH`. Nsight Systems 2026.3.2. The workload ran under Python 3.11 with CuPy 14.2.0 and NumPy 2.4.6,
  as pinned in `uv.lock`.

The environment snapshots in `env/` record the driver, toolkit, and library versions at the start of each run.

## Layout

| Path | Contents |
|---|---|
| `bench/bin/dgemm_bench_v5`, `.cu` | The GEMM benchmark binary used for every GEMM measurement, and its exact source (their SHA-256 hashes match each run's `binary.sha256`) |
| `bench/bin/mempool_bench`, `.cu` | Microbenchmark of the allocation, release, and remapping cost of the CUDA memory pool (phase P3b) |
| `bench/dgemm_bench.cu`, `build.sh` | The source that `build.sh` compiles, identical to `dgemm_bench_v5.cu` |
| `bench/run_grid.sh`, `collect.sh`, `chain.sh` | Grid runner (one process per configuration), phase driver, and driver for a whole collection |
| `bench/qa.py`, `select_reruns.py`, `summarize_grid.py` | Quality check of a run, selection of the lines to rerun, and per-run summaries |
| `bench/trace_p3a.sh`, `run_p3b.sh` | Nsight Systems traces (P3a) and the memory-pool microbenchmark (P3b) |
| `bench/grids/` | The exact grids that were run (`final/`, `rerun/`) and the scripts that generate them |
| `workloads/` | Randomised SVD (`rsvd.py`, `run_rsvd.sh`), its test matrices (`make_inputs.py`), the memory-pool probe (`pool_probe.py`), and the overflow tests (`overflow_class.py`, `overflow_cpu.py`) |
| `env/toolchain.sh`, `capture.sh` | Toolchain settings, and the script that records an environment snapshot at the start of each run |
| `env/<timestamp>/` | Environment snapshots: driver, toolkit, and library versions, GPU list and properties, and topology |
| `data/` | Reference singular values of the randomised-SVD test matrices, and the checksums of the matrices |
| `results/final/` | Raw results of every phase, one directory per run (see below) |
| `analysis/` | `headline.py` (summary tables), `ledger.py` (every number in the paper with its source), and `figures.py` (Figs. 1 to 4) |
| `summary/` | `LEDGER.md`, `HEADLINE.md`, and a summary of each GEMM run |
| `figs/` | Figs. 1 to 4 as PDF and PNG, with the data of each as CSV |
| `pyproject.toml`, `uv.lock` | Pinned Python environment |
| `unpack.sh` | Restores the per-run logs and kernel records from their archives |

## Phases

| Phase | What it measures | Paper |
|---|---|---|
| P1a square | Square GEMMs, n = 512 to 16384, seven modes | VI-A, VI-D |
| P1b shapes | 49 shapes: skinny, panel, short-wide, Gram, odd sizes, and those of the randomised SVD | VI-B, VI-D |
| P1c ranges | Exponent-range sweeps with *ozaki*, *inner*, and *rowcol* inputs | VI-C |
| P2a decision | Path taken by the default strategy and by EAGER on 290 shapes | VI-B |
| P2b version | cuBLAS 13.2.2.2, 13.4.1.3, 13.6.0.2, and 13.8.0.4 | VI-C |
| P2c bits | Fixed mantissa bits and dynamic caps | VI-C, Table II |
| P2d edges | Special values, transposes, β = 1, and batched calls | VI-A, VI-C |
| P2e follow-up | Fine grid around the fallback threshold, overflow tests, and kernel timings of the precision settings | VI-C |
| P2f pool | Which memory pool the emulation workspace comes from, called directly and through CuPy | VI-D |
| P3a nsys | Nsight Systems traces of selected configurations | VI-D |
| P3b mempool | Allocation and remapping cost of the memory pool against size | VI-D |
| P4 rsvd | End-to-end randomised SVD | VI-E, Table III |

Directories ending in `_rerun` hold the configurations that the quality check flagged, run three more times, and the
analysis pools them with the original run. In directory and file names, `gpu2` is the RTX 5060 Ti, `gpu4` and `gpu5`
are the two RTX 3090 boards, and `gpu8` is the RTX 4090.

## A run directory

Each GEMM run directory contains:

- `gpu<G>.csv`, one row per configuration and repeat (columns below), and `gpu<G>.log`, the runner's log;
- `grid.txt`, the grid that was run, and `runner.txt`, the repeats, CPU pinning, and binary;
- `binary.sha256`, the hashes of the benchmark binary and its source, and `git.txt`, the commit of the working repository;
- `env_snapshot_dir.txt`, the environment snapshot taken at the start of the run (under `env/`);
- `smi_audit.csv`, the GPU's clocks, power, temperature, and memory use, sampled once per second;
- `qa.txt`, the result of the quality check;
- `lt.tar.xz`, the cuBLASLt API log of every configuration, which includes the workspace that cuBLAS requested;
- `kern.tar.xz`, the kernels that each call launched, recorded with CUPTI.

The columns of `gpu<G>.csv` used most often are:

| Column | Meaning |
|---|---|
| `tag` | Label of the configuration and repeat, for example `P1a_square:sq:default_ws0:n4096:rep1`; the mode is `native`, `default_*`, or `eager_*`, with the suffix `ws0` (zero code), `kp` (memory pool kept), or `ws` (workspace set) |
| `median_ms` | Median GPU time per call, from CUDA events (the analysis takes the median of each column over the repeats) |
| `wall_sync_ms` | Mean wall-clock time per call over a series of calls, each followed by a host synchronisation |
| `wall_pipe_ms` | Mean wall-clock time per call over the same number of calls run back to back, with one synchronisation at the end |
| `kern_n`, `kern_emu_frac` | Number of kernels a call launched, and the share of kernel time spent in emulation kernels, from which the path is classified |
| `bits_last` | Value of the mantissa-bit counter after the last timed call; it is preset to −12345, so an unwritten counter is recognisable |
| `err_comp_max`, `nat_err_comp_max` | Componentwise error of the product, and of native DGEMM on the same inputs, against the extended-precision reference |
| `hash_C` | Hash of the output, for bitwise comparisons across repeats and GPUs |
| `workspace_mb` | Workspace in MiB passed with `cublasSetWorkspace` (the size cuBLAS requested plus a 32 MiB margin), or −1 when none was set |
| `clk_reasons_or`, `other_procs_pre` | Throttle reasons during the timed window, and the number of other processes on the GPU before the configuration ran |

The other phases write their own files: `rsvd.csv` (P4), `pool.csv` (P2f), `mempool.csv` (P3b), `overflow.csv`
(P2e), and the kernel and API summaries of Nsight Systems (`*_cuda_gpu_kern_sum.csv`, `*_cuda_api_sum.csv`).

## Regenerating the tables and figures

```sh
./unpack.sh                                              # restores lt/ and kern/ (about 2.3 GB)
uv sync --only-group analysis                            # matplotlib and NumPy
uv run python analysis/headline.py > summary/HEADLINE.md
uv run python analysis/ledger.py   > summary/LEDGER.md   # NumPy rebuilds the overflow test's inputs (row C5.15)
uv run python analysis/figures.py --out figs             # uses the TeX Gyre Termes font if it is installed
```

Run from the root of the repository, these commands reproduce `summary/HEADLINE.md`, `summary/LEDGER.md`, and the
CSV files in `figs/` exactly.

## Rerunning the measurements

The measurements need a Linux machine with CUDA 13.4 and a GPU of compute capability 8.x to 12.x.

1. Edit `env/toolchain.sh` for the local CUDA installation, and `source env/toolchain.sh`.
2. Use the included binary, or build one with `bench/build.sh bench/bin/<name>` and pass `BIN=bench/bin/<name>` to
   the runner.
3. Run a grid, for example `REPEATS=3 CPUS=0-11 bench/run_grid.sh <gpu> bench/grids/final/P1a_4090.txt P1a_square`.
   The grids embed the workspace sizes requested on our GPUs, and `bench/grids/make_final.sh` regenerates them for
   other hardware.
4. For the randomised SVD, run `uv sync`, build the five test matrices with `uv run python workloads/make_inputs.py`
   (seed 20261107, about 1.1 GB each), and run `workloads/run_rsvd.sh <gpu> <cpus>`. `data/inputs.sha256` holds the
   checksums of our matrices. A different BLAS build may not reproduce them bit for bit, but the reference singular
   values in `data/` are included either way.

Recorded absolute paths of the form `/home/taiz/fp64/pathB/...` refer to the root of this repository.

## What is not included

- **Nsight Systems reports** (`*.nsys-rep`) and their SQLite exports. The reports embed identifiers of the host
  machine, so only their CSV summaries, which the analysis reads, are included.
- **The randomised-SVD input matrices**, which `workloads/make_inputs.py` regenerates as described above.
- **Identifying details in the environment snapshots.** The host name, the GPU UUIDs and persistent device
  identifiers, and the lists of processes running on the host have been removed. Library versions, GPU models, and
  topology are unchanged.
- **The environment snapshots of five runs**, `env/20261108T040414Z`, `041053Z`, `041120Z`, `041156Z`, and
  `065615Z`, which belong to the P2e follow-up and P2f pool runs and were not archived.
