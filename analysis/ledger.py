#!/usr/bin/env python3
"""Claims ledger: every number the paper may quote, computed from results/final (stdlib only).

Each entry: ID, the claim as it may be written, the value(s), and where they come from. Numbers in the paper must be
copied from summary/LEDGER.md (or regenerated), never from memory or older notes. Helpers (loading, pooling of
reruns, path classification, medians) are shared with analysis/headline.py.

Usage: python3 analysis/ledger.py > summary/LEDGER.md
"""
import csv
import glob
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from headline import (FINAL, MODELS, NOBITS, WS_PAD_MIB, by_config, load, med, model, natkey, p3b_slopes,  # noqa: E402
                      path_of, req_mib, split)

KEYS = [k for k, _ in MODELS]
FIXED = {"3090": "eager", "4090": "default", "5060 Ti": "default"}  # the mode that emulates "for real" per GPU
OUT = []


def claim(cid, text, value, source):
    OUT.append((cid, text, value, source))


def section(title):
    OUT.append((None, title, None, None))


def spd(nat, rs, col="median_ms"):
    return med(nat, col) / med(rs, col)


def fmt(x, nd=2):
    return f"{x:.{nd}f}"


def rng(xs, nd=2):
    return f"{min(xs):.{nd}f}–{max(xs):.{nd}f}" if xs else "-"


def e(x):
    return f"{x:.1e}"


# ---------------------------------------------------------------------------------------------------------- setup
def setup_and_quality():
    section("Q. Setup and data quality")
    rows, notok = 0, 0
    bad_bits, foreign, nondet = 0, 0, 0
    swtherm = 0
    for p in glob.glob(str(FINAL / "P*" / "*" / "gpu*.csv")):
        for r in csv.DictReader(open(p)):
            rows += 1
            notok += r["status"] != "ok"
            b = int(r["clk_reasons_or"], 16)
            bad_bits += bool(b & (0x8 | 0x40 | 0x80))
            swtherm += bool(b & 0x20)
            foreign += r["other_procs_pre"] != "0"
            nondet += r["deterministic"] != "1"
    claim("Q1", "All benchmark rows completed", f"{rows - notok} of {rows} rows status ok",
          "all results/final/P*/*/gpu*.csv (incl. reruns, P2e)")
    claim("Q2", "No hardware slowdown / HW thermal / power-brake throttling in any row",
          f"{bad_bits} rows with NVML bits 0x8|0x40|0x80; {swtherm} rows with SW-thermal 0x20",
          "clk_reasons_or (OR over the timed window); bit meanings as in bench/qa.py BAD_REASONS")
    claim("Q3", "No foreign process on the GPU before any run; every output deterministic on rerun",
          f"{foreign} rows with other_procs_pre≠0; {nondet} rows not deterministic", "other_procs_pre, deterministic")
    g = load("P1a_square")
    for k in KEYS:
        nat = [r for (m, name), rs in g.items() if m == k and name.startswith("sq:native:")
               and int(name.split(":n")[-1]) >= 4096 for r in rs]
        tf = [2 * int(r["m"]) * int(r["n"]) * int(r["k"]) / (float(r["median_ms"]) * 1e-3) / 1e12 for r in nat]
        claim(f"Q4.{k}", f"Measured native DGEMM throughput, RTX {k}", f"{st.median(tf):.3f} TFLOPS (median, n≥4096; "
              f"range {rng(tf, 3)})", "P1a native rows, 2mnk/median_ms")
    # Accuracy reference: 80-bit long double sum vs Dot2 on the same sampled entries (bench/dgemm_bench.cu), in the
    # componentwise-error metric. Bounds how finely an emulated error can be resolved per input family.
    dis = defaultdict(list)
    for p in glob.glob(str(FINAL / "P*" / "*" / "gpu*.csv")):
        for r in csv.DictReader(open(p)):
            x = r.get("ref_disagree", "")
            if r["status"] != "ok" or x in ("", "-1") or float(x) < 0:
                continue
            grp = "subnormal injection" if "subnormal" in r["tag"] else (
                "uniform" if r["dist"] == "u11" else f"wide-range ({r['dist']})")
            dis[grp].append(float(x))
    claim("Q5", "Extended-precision reference vs Dot2: largest disagreement in the componentwise-error metric",
          "; ".join(f"{g_} {e(max(v))} (median {e(st.median(v))}, {len(v)} rows)" for g_, v in sorted(dis.items())),
          "ref_disagree, all ok rows; subnormal: Dot2's products underflow in double")
    # Peak memory in use on the modified 48 GB RTX 4090 (GPU index 8), from the once-per-second nvidia-smi log.
    peak, nlog = 0, 0
    for p in glob.glob(str(FINAL / "P*" / "*gpu8*" / "smi_audit.csv")):
        nlog += 1
        for r in csv.reader(open(p)):
            if len(r) > 3 and r[1].strip() == "8" and r[3].strip().endswith("MiB"):
                peak = max(peak, int(r[3].split()[0]))
    claim("Q6", "Peak GPU memory in use on the RTX 4090 (48 GB board) in any run",
          f"{peak} MiB ({peak / 1024:.1f} GiB) over {nlog} run logs; stock capacity 24 GB",
          "results/final/P*/*gpu8*/smi_audit.csv memory.used (1 Hz)")


# ------------------------------------------------------------------------------------------------------------- C1
def c1():
    section("C1. Large well-scaled square GEMMs: speed and accuracy")
    g = load("P1a_square")
    for k in KEYS:
        rows = by_config(g, k)
        pre = FIXED[k]
        sp = {}
        for cfg, v in rows.items():
            if f"{pre}_ws" in v and "native" in v and path_of(v[f"{pre}_ws"][0]) == "emu":
                sp[int(cfg.split(":n")[-1])] = spd(v["native"], v[f"{pre}_ws"])
        best = max(sp, key=sp.get)
        claim(f"C1.1.{k}", f"Peak speedup over native, square, {pre.upper()} + workspace (event time), RTX {k}",
              f"{sp[best]:.2f}× at n={best}; n=4096 {sp.get(4096, float('nan')):.2f}×, n=8192 "
              f"{sp.get(8192, float('nan')):.2f}×; emulated sizes n={min(sp)}…{max(sp)}", f"P1a {pre}_ws vs native")
    emu_err, nat_err, bits = [], [], set()
    for (m, name), rs in g.items():
        for r in rs:
            if path_of(r) == "emu":
                emu_err.append(float(r["err_comp_max"])); bits.add(int(r["bits_last"]))
            nat_err.append(float(r["nat_err_comp_max"]))
    claim("C1.2", "Componentwise error, square u11 inputs: emulated vs native (all GPUs, all n)",
          f"emulated {e(min(emu_err))}–{e(max(emu_err))}; native {e(min(nat_err))}–{e(max(nat_err))}; "
          f"bits used {min(bits)}–{max(bits)}", "P1a err_comp_max (emu rows), nat_err_comp_max")
    h = defaultdict(lambda: defaultdict(set))
    for ph in ("P1a_square", "P1b_shapes", "P1c_ranges"):
        for (m, name), rs in load(ph).items():
            for r in rs:
                if path_of(r) == "emu":
                    h[name][m].add(r["hash_C"])
    multi = {n: v for n, v in h.items() if len(v) >= 2}
    same = sum(1 for v in multi.values() if len(set().union(*v.values())) == 1 and all(len(s) == 1 for s in v.values()))
    claim("C1.3", "Emulated results are bit-identical across the three GPUs",
          f"{same} of {len(multi)} emulated configs run on ≥2 GPUs have one identical output hash", "P1a/P1b/P1c hash_C")
    spread = []
    for ph in ("P2d_edges",):
        g2 = load(ph)
        for k in KEYS:
            for fam in ("sq4096", "skinny256"):
                s = [spd(v["native"], v["eager_ws"]) for cfg, v in by_config(g2, k).items()
                     if f"op:{fam}:" in cfg and "eager_ws" in v and "native" in v]
                if s:
                    spread.append((100 * (max(s) / min(s) - 1), k, fam))
                    claim(f"C1.4.{k}.{fam}", f"Transposes NN/NT/TN/TT change little, {fam}, RTX {k} (EAGER+ws)",
                          f"speedups {rng(s, 3)} (max/min − 1 = {spread[-1][0]:.1f} %)", "P2d op rows")
    w = max(spread)
    claim("C1.4", "Largest change of speedup from transposing an operand, all GPUs and shapes (unrounded speedups)",
          f"{w[0]:.1f} % (RTX {w[1]}, {w[2]})", "P2d op rows, max/min − 1")


# ------------------------------------------------------------------------------------------------------------- C2
def c2():
    section("C2. What the default (zero-code) strategy decides")
    g = load("P2a_decision")
    for k in KEYS:
        pts = by_config(g, k)
        n = em = dec = f1 = f15 = slow = edec = 0
        for cfg, v in pts.items():
            if "default_ws" not in v or "eager_ws" not in v:
                continue
            n += 1
            if path_of(v["default_ws"][0]) == "emu":
                em += 1
            else:
                dec += 1
                if path_of(v["eager_ws"][0]) == "emu":
                    s = spd(v["default_ws"], v["eager_ws"]); f1 += s > 1; f15 += s > 1.5; slow += s < 1
                else:
                    edec += 1
        claim(f"C2.1.{k}", f"Decision map (m,n,k cube + √2 slices), RTX {k}",
              f"{n} points: default emulates {em}, declines {dec}; where it declines EAGER is faster at {f1} "
              f"(>1.5× at {f15}), slower at {slow}, also declines at {edec}", "P2a, 1 repeat, path from CUPTI")
    cnt = Counter()
    for ph in ("P1a_square", "P1b_shapes", "P1c_ranges", "P2a_decision", "P2b_version", "P2d_edges"):
        for (m, name), rs in load(ph).items():
            if m == "3090" and split(name)[0].startswith("default"):
                for r in rs:
                    cnt[path_of(r)] += 1
    claim("C2.2", "On the RTX 3090 the default strategy never emulated, in any phase or cuBLAS version",
          f"3090 default_* rows by path: {dict(cnt)}", "P1a,P1b,P1c,P2a,P2b,P2d (P2b = 4 cuBLAS releases)")
    g = load("P1a_square")
    v = by_config(g, "5060 Ti").get("sq:n12288", {})
    paths = {md: path_of(rs[0]) for md, rs in v.items()}
    req = lambda k, n: req_mib(by_config(g, k)[f"sq:n{n}"]["eager_ws"][0])
    mats = 3 * 12288 ** 2 * 8 / 2 ** 30
    claim("C2.3", "RTX 5060 Ti (8 GB), n=12288: every mode incl. EAGER runs native, silently",
          f"paths {paths}; cuBLAS-requested workspace (whole MiB, rounded down): 5060 Ti n=12288 "
          f"{req('5060 Ti', 12288)} MiB, n=8192 {req('5060 Ti', 8192)} MiB; RTX 3090 / 4090 at n=12288 "
          f"{req('3090', 12288)} / {req('4090', 12288)} MiB ({req('3090', 12288) / 1024:.1f} GiB); "
          f"A, B, C at n=12288 occupy {mats:.2f} GiB; 5060 Ti total memory 8151 MiB",
          "P1a eager_ws workspace_mb via req_mib(); env/20261107T194918Z/gpus.csv (memory.total)")


# ------------------------------------------------------------------------------------------------------------- C3
def c3():
    section("C3. Zero-code penalty (mempool) and the one-line fix")
    g = load("P1a_square")
    rates = p3b_slopes()
    for k in KEYS:
        pre = FIXED[k]
        rows = by_config(g, k)
        kept, below, ratio, evshare, kpws = [], [], [], [], []
        pipe_lo, pipe_hi, pipe_ex, runs = [], [], [], []
        for cfg in sorted(rows, key=natkey):
            v = rows[cfg]
            ws, ws0, kp, nat = v.get(f"{pre}_ws"), v.get(f"{pre}_ws0"), v.get(f"{pre}_kp"), v.get("native")
            if not ws or not ws0 or path_of(ws[0]) != "emu":
                continue
            n = int(cfg.split(":n")[-1])
            s_fix, s_zc = spd(nat, ws), spd(nat, ws0, "wall_sync_ms")
            kept.append(s_zc / s_fix)
            if s_zc < 1:
                below.append(f"n={n} {s_zc:.2f}×")
            # calls back to back, one final sync (wall_pipe_ms): zero code vs the fix, both on that clock
            pm = lambda rs: st.median(float(r["wall_pipe_ms"]) for r in rs if float(r["wall_pipe_ms"]) > 0)
            (pipe_lo if n <= 4096 else pipe_hi).append(pm(ws) / pm(ws0))
            if k == "4090" and n in (1024, 4096):
                pipe_ex.append(f"n={n}: back to back {pm(nat) / pm(ws0):.2f}× vs sync after every call {s_zc:.2f}×")
            # per-run spread of the zero-code speedup (each run vs the native median), where it is near 1
            per = sorted(med(nat, "wall_sync_ms") / float(r["wall_sync_ms"]) for r in ws0)
            if per[0] < 1 <= per[-1] or 0.9 <= s_zc < 1:
                runs.append(f"n={n} median {s_zc:.2f}×, runs {per[0]:.2f}–{per[-1]:.2f}× ({len(per)})")
            fe, we, ww = med(ws, "median_ms"), med(ws0, "median_ms"), med(ws0, "wall_sync_ms")
            pred = req_mib(ws[0]) / 1024 * rates[k][1]
            ratio.append((ww - fe) / pred); evshare.append((we - fe) / (ww - fe))
            if kp:
                kpws.append(abs(med(kp, "median_ms") / fe - 1))
        claim(f"C3.1.{k}", f"Share of the fixed speedup kept with zero code changes (wall, sync after every call), "
              f"square, RTX {k} ({pre})",
              f"{rng([100 * x for x in kept], 0)} %", "P1a ws0 wall vs ws event (fixed wall = event within C1/check)")
        claim(f"C3.13.{k}", f"Zero code with calls back to back (one final sync) as a share of the fix on the same "
              f"clock, square, RTX {k} ({pre})",
              f"n ≤ 4096: {rng([100 * x for x in pipe_lo], 1)} %; n > 4096: {rng([100 * x for x in pipe_hi], 1)} %"
              + (f"; {'; '.join(pipe_ex)}" if pipe_ex else ""),
              "P1a wall_pipe_ms (ws0 vs ws; examples vs native wall_pipe_ms)")
        claim(f"C3.14.{k}", f"Zero-code speedups near 1× (wall, sync after every call): per-run range, square, RTX {k}",
              "; ".join(runs) or "none", "P1a ws0 wall_sync_ms per run vs native median")
        claim(f"C3.2.{k}", f"Square sizes where zero code runs below native (wall), RTX {k}",
              ", ".join(below) or "none", "P1a")
        claim(f"C3.3.{k}", f"Penalty model: measured / (requested workspace × remap rate), RTX {k}",
              f"{rng(ratio)} (rate {rates[k][1]:.1f} ms/GiB)", "P1a + P3b")
        claim(f"C3.4.{k}", f"Share of the zero-code penalty visible in GPU (event) time, RTX {k}",
              f"{rng([100 * x for x in evshare], 0)} %", "P1a ws0 event vs wall")
        claim(f"C3.5.{k}", f"The two fixes agree: pool threshold (kp) vs workspace (ws), event time, RTX {k}",
              f"max |kp/ws − 1| = {100 * max(kpws):.1f} %", "P1a")
    for k, (run, slope, last, tmax) in sorted(p3b_slopes().items()):
        claim(f"C3.6.{k}", f"Remap cost of cudaMallocAsync+free+sync, release threshold 0, RTX {k}",
              f"{slope:.1f} ms/GiB; with threshold raised ≤ {tmax:.3f} ms", f"P3b {run}")
    f = glob.glob(str(FINAL / "P3a_nsys" / "P3a_nsys_gpu8_*" / "sq4096.default_ws0_cuda_api_sum.csv"))
    if f:
        r = [x for x in csv.DictReader(open(f[0])) if x["Name"] == "cudaMallocAsync"]
        f2 = f[0].replace("default_ws0", "default_ws")
        r2 = [x for x in csv.DictReader(open(f2)) if "Malloc" in x["Name"] and "Async" in x["Name"]]
        api = lambda path: {x["Name"]: x for x in csv.DictReader(open(path))}
        a0, a1 = api(f[0]), api(f2)
        tot = lambda a, name: float(a[name]["Total Time (ns)"]) / 1e6 if name in a else 0.0
        cnt = lambda a, name: int(a[name]["Num Calls"]) if name in a else 0
        syncs = ("cudaEventSynchronize", "cudaDeviceSynchronize", "cudaStreamSynchronize")
        # GEMM calls per trace: two cuLaunchKernelEx per emulated GEMM call (= cudaMallocAsync count with zero code)
        n0, n1 = cnt(a0, "cuLaunchKernelEx") // 2, cnt(a1, "cuLaunchKernelEx") // 2
        s0, s1 = sum(tot(a0, s) for s in syncs) / n0, sum(tot(a1, s) for s in syncs) / n1
        m0 = tot(a0, "cudaMallocAsync") / cnt(a0, "cudaMallocAsync")
        claim("C3.7", "Mechanism in the trace (RTX 4090, 4096³, default strategy): one allocation per call",
              f"zero code: {r[0]['Num Calls']} cudaMallocAsync, {float(r[0]['Total Time (ns)']) / 1e6:.0f} ms total "
              f"({m0:.1f} ms per call; median {float(r[0]['Med (ns)']) / 1e6:.1f}, max {float(r[0]['Max (ns)']) / 1e6:.1f}); "
              f"{cnt(a0, 'cudaFreeAsync')} cudaFreeAsync, {tot(a0, 'cudaFreeAsync') / cnt(a0, 'cudaFreeAsync') * 1e3:.1f} µs "
              f"each; with workspace: {len(r2)} async allocations; no memory-pool API calls in either. Synchronisation "
              f"per GEMM call: zero code {s0:.1f} ms ({n0} calls), workspace {s1:.1f} ms ({n1} calls), difference "
              f"{s0 - s1:.1f} ms; allocation + extra synchronisation {m0 + s0 - s1:.1f} ms per call (under nsys)",
              "P3a sq4096.default_ws0 / default_ws cuda_api_sum.csv (calls = cuLaunchKernelEx / 2)")
    # session-to-session variability of the zero-code wall time (original vs rerun)
    var = []
    for ph in ("P1a_square", "P1b_shapes"):
        orig, rer = defaultdict(list), defaultdict(list)
        for p in glob.glob(str(FINAL / ph / "*" / "gpu*.csv")):
            for r in csv.DictReader(open(p)):
                orig[(model(r["gpu"]), ":".join(r["tag"].split(":")[1:-1]))].append(r)
        for p in glob.glob(str(FINAL / f"{ph}_rerun" / "*" / "gpu*.csv")):
            for r in csv.DictReader(open(p)):
                rer[(model(r["gpu"]), ":".join(r["tag"].split(":")[1:-1]))].append(r)
        for key in rer:
            if key in orig and key[1].split(":")[1].endswith("_ws0"):
                var.append(abs(med(rer[key], "wall_sync_ms") / med(orig[key], "wall_sync_ms") - 1))
    claim("C3.8", "Zero-code wall time varies between sessions (original vs rerun of the same line)",
          f"{len(var)} lines; |Δ| median {100 * st.median(var):.0f} %, max {100 * max(var):.0f} %", "P1a/P1b + *_rerun")
    # Spread between the repeats of one configuration, (max - min) / median, original runs (not reruns).
    spr = defaultdict(list)
    t4 = {}
    for ph in ("P1a_square", "P1b_shapes"):
        reps = defaultdict(list)
        for p in glob.glob(str(FINAL / ph / "*" / "gpu*.csv")):
            for r in csv.DictReader(open(p)):
                if r["status"] == "ok":
                    reps[(model(r["gpu"]), ":".join(r["tag"].split(":")[1:-1]))].append(r)
        for (m, name), rs in reps.items():
            mode = name.split(":")[1]
            grp = "zero code" if mode.endswith("_ws0") else "native" if mode == "native" else "fix"
            col = "wall_sync_ms" if grp == "zero code" else "median_ms"
            v = [float(r[col]) for r in rs if float(r[col]) > 0]
            if len(v) > 1:
                s = (max(v) - min(v)) / st.median(v)
                spr[grp].append(s)
                cfg = ":".join([name.split(":")[0]] + name.split(":")[2:])
                if cfg in ("sq:n1024", "sq:n4096", "shape:skinny:m8192_n256_k8192"):
                    t4[(grp, m, mode, cfg)] = s
    parts = []
    for grp in ("zero code", "fix", "native"):
        xs = sorted(spr[grp])
        parts.append(f"{grp}: {len(xs)} configs, median {100 * st.median(xs):.1f} %, max {100 * xs[-1]:.1f} %, "
                     f"> 20 % in {sum(x > 0.2 for x in xs)}")
    w = max((s, k) for k, s in t4.items() if k[0] == "zero code")
    f = max(s for k, s in t4.items() if k[0] == "fix")
    parts.append(f"Table IV GEMM cells: zero code max {100 * w[0]:.0f} % ({w[1][1]} {w[1][2]} {w[1][3]}), "
                 f"fix max {100 * f:.1f} %")
    claim("C3.15", "Spread between repeats of one configuration, (max − min)/median, GEMM P1a/P1b original runs "
          "(zero code: wall with sync after every call; fix and native: GPU time)", "; ".join(parts),
          "P1a/P1b gpu*.csv, per repeat")
    # Requested workspace (bench pad removed) per square size and GPU, where that GPU emulates under EAGER.
    g = load("P1a_square")
    ws = defaultdict(dict)
    for k in KEYS:
        for cfg, v in by_config(g, k).items():
            if "eager_ws" in v and path_of(v["eager_ws"][0]) == "emu":
                ws[int(cfg.split(":n")[-1])][k] = req_mib(v["eager_ws"][0])
    same = all(len(set(d.values())) == 1 for d in ws.values())
    # Exact byte counts where a cuBLASLt trace log of an emulating cuBLAS 13.8 square call exists (P2a cube, P2b 13.4).
    exact = {}
    for p in glob.glob(str(FINAL / "P2*" / "*" / "lt" / "*.log")):
        m = re.search(r"(?:ver13\.4:eager_ws:sq:n(\d+)|:eager_ws:cube:m(\d+)_n\2_k\2)\.r\d\.log$", p)
        if not m:
            continue
        n = int(m.group(1) or m.group(2))
        b = Counter(int(x) for x in re.findall(r"workSpaceSizeInBytes=([1-9]\d*)", open(p, errors="replace").read()))
        if b:
            exact[n] = b.most_common(1)[0][0] / 2 ** 20
    claim("C3.12", "Workspace cuBLAS requests, square, per size (emulating GPUs only): identical across GPUs",
          f"identical at every size: {same}; " + "; ".join(
              f"n={n}: {next(iter(d.values()))} MiB ({len(d)} GPUs)" for n, d in sorted(ws.items()))
          + "; exact from cuBLASLt logs: " + ", ".join(f"n={n} {x:.3f} MiB" for n, x in sorted(exact.items())),
          "P1a eager_ws workspace_mb via req_mib() (whole MiB, rounded down); P2a/P2b lt/*.log workSpaceSizeInBytes")
    c3_pool()


def c3_pool():
    """P2f: which pool holds the emulation workspace (workloads/pool_probe.py). Call 0 (cuBLAS init) excluded."""
    cases = ["native", "default_zc", "eager_zc", "eager_kp", "eager_custom_zero", "eager_custom_max",
             "eager_ws_matmul", "eager_zc_dgemm", "eager_ws_dgemm", "eager_zc_ctypes", "eager_ws_ctypes"]
    runs = sorted(glob.glob(str(FINAL / "P2f_pool" / "pool_gpu*" / "pool.csv")))
    relerr = []
    for p in runs:
        rows = [r for r in csv.DictReader(open(p)) if r["call"] != "0"]
        k = model(rows[0]["gpu"])
        by = defaultdict(list)
        for r in rows:
            by[r["label"]].append(r)
            relerr.append(float(r["check_relerr"]))

        def mm(rs, col):
            v = [float(r[col]) for r in rs if r[col] != ""]
            return f"{st.median(v):.0f}" if v else "-"
        parts = []
        for c in cases:
            rs = by.get(c)
            if not rs:
                parts.append(f"{c}: missing")
                continue
            cu = f" / custom {mm(rs, 'custom_used_high_mib')}" if rs[0]["pool"] == "custom" else ""
            parts.append(f"{c}: default {mm(rs, 'default_used_high_mib')}{cu} MiB, {mm(rs, 'wall_ms')} ms")
        claim(f"C3.9.{k}", f"Async-pool memory used per GEMM call (high-water mark), {rows[0]['n']}³, RTX {k} "
              f"(cuBLAS {rows[0]['cublas_version']}); median wall ms per call (shared host, 3 GPUs in parallel, single-call "
              f"spikes up to ~4×: not a timing source, see C3.1-C3.6)", "; ".join(parts),
              f"P2f {Path(p).parent.name}/pool.csv, calls 1–{max(int(r['call']) for r in rows)}")
    if relerr:
        claim("C3.10", "Probe sanity check: one entry of C vs a host dot product, every call, every case",
              f"max relative error {max(relerr):.1e}", "P2f pool.csv check_relerr")
    names, mallocs = set(), defaultdict(dict)
    for p in sorted(glob.glob(str(FINAL / "P2f_pool" / "pool_gpu*" / "nsys_*_cuda_api_sum.csv"))):
        k = model(open(Path(p).parent / "pool.csv").readlines()[1])
        case = Path(p).name[len("nsys_"):-len("_cuda_api_sum.csv")]
        api = {r["Name"]: int(r["Num Calls"]) for r in csv.DictReader(open(p))}
        names |= set(api)
        mallocs[case][k] = api.get("cudaMallocAsync", 0)
    if mallocs:
        # the probe itself calls these (reading/resetting the default pool); anything else pool-related is cuBLAS
        probe_own = {"cudaDeviceGetDefaultMemPool", "cudaDeviceGetMemPool", "cudaMemPoolSetAttribute",
                     "cudaMemPoolGetAttribute"}
        pool_api = sorted(n for n in names if re.search(r"Pool", n) and n not in probe_own)
        claim("C3.11", "nsys, 6 GEMM calls per trace (EAGER, 4096³): cudaMallocAsync calls per trace by case and GPU; "
              "pool-related CUDA API names in any trace other than the probe's own pool queries",
              "; ".join(f"{c}: " + ", ".join(f"{k} {v}" for k, v in sorted(d.items())) for c, d in sorted(mallocs.items()))
              + f". Other pool APIs: {', '.join(pool_api) or 'none'}",
              "P2f nsys_*_cuda_api_sum.csv (CUDA runtime + driver API)")


# ------------------------------------------------------------------------------------------------------------- C4
def c4():
    section("C4. Shape sensitivity and the default heuristic")
    g = load("P1b_shapes")
    fams = {"skinny": "n", "panel": "k", "shortwide": "m"}
    for k in KEYS:
        for fam, dim in fams.items():
            pts = defaultdict(dict)
            for (m, name), rs in g.items():
                if m == k and f":{fam}:" in name:
                    mode, cfg = split(name)
                    pts[int(re.search(rf"{dim}(\d+)", cfg.split(":")[-1]).group(1))][mode] = rs

            def first(test):
                return next((str(d) for d in sorted(pts) if "native" in pts[d] and test(pts[d])), "never")
            a = first(lambda v: spd(v["native"], v["eager_ws"]) > 1)
            b = first(lambda v: path_of(v["default_ws"][0]) == "emu")
            c = first(lambda v: path_of(v["default_ws0"][0]) == "emu" and spd(v["native"], v["default_ws0"], "wall_sync_ms") > 1)
            claim(f"C4.1.{k}.{fam}", f"Break-even, {fam} ({dim} varies, others 8192), RTX {k}",
                  f"EAGER+ws beats native from {dim}={a}; default emulates from {b}; zero-code default beats native from {c}",
                  "P1b, event (ws) / wall (ws0) vs native mode")
    for k in KEYS:
        rows = by_config(g, k)
        w = {c: spd(v["native"], v["eager_ws"]) for c, v in rows.items() if "eager_ws" in v}
        pick = ["shape:skinny:m8192_n8_k8192", "shape:panel:m8192_n8192_k16", "shape:gram:m32_n32_k8192"]
        claim(f"C4.2.{k}", f"Worst EAGER cases (event), RTX {k}",
              "; ".join(f"{c.split(':', 1)[1]} {w[c]:.2f}×" for c in pick if c in w), "P1b")
    n, slow = 0, 0
    for ph in ("P1a_square", "P1b_shapes", "P1c_ranges", "P2d_edges"):
        for k in KEYS:
            for cfg, v in by_config(load(ph), k).items():
                if "native" in v and "default_ws" in v and path_of(v["default_ws"][0]) == "emu":
                    n += 1; slow += spd(v["native"], v["default_ws"]) < 1
    claim("C4.3", "Where the default emulates and native was also measured, it is never slower than native in GPU "
          "time (the decision map P2a has no native run of default-emulated shapes and is not included)",
          f"{slow} of {n} default-emulated configs slower than native", "P1a/P1b/P1c/P2d default_ws vs native")
    rows = by_config(g, "4090")
    lst = [f"{c.split(':', 1)[1]} {spd(v['native'], v['default_ws0'], 'wall_sync_ms'):.2f}×"
           for c, v in sorted(rows.items(), key=lambda x: natkey(x[0]))
           if "default_ws0" in v and path_of(v["default_ws0"][0]) == "emu"
           and spd(v["native"], v["default_ws0"], "wall_sync_ms") < 1]
    claim("C4.4", "RTX 4090: shapes the default emulates but that run below native with zero code changes (wall)",
          "; ".join(lst), "P1b default_ws0 wall vs native wall")
    # Why small dimensions lose: below break-even the emulated time hardly moves with the small dimension, while the
    # native time shrinks with it (the speedup curve then copies native's shape, incl. its kernel switches).
    for k in KEYS:
        parts = []
        for fam, dim in fams.items():
            pts = {}
            for (m, name), rs in g.items():
                if m == k and f":{fam}:" in name:
                    mode, cfg = split(name)
                    d = int(re.search(rf"{dim}(\d+)", cfg.split(":")[-1]).group(1))
                    pts.setdefault(d, {})[mode] = rs
            small = [d for d in sorted(pts) if d <= 128 and "eager_ws" in pts[d] and "native" in pts[d]]
            em = [med(pts[d]["eager_ws"], "median_ms") for d in small]
            na = [med(pts[d]["native"], "median_ms") for d in small]
            parts.append(f"{fam} {dim}={small[0]}…{small[-1]}: EAGER {min(em):.1f}–{max(em):.1f} ms, "
                         f"native {min(na):.2f}–{max(na):.2f} ms")
        claim(f"C4.5.{k}", f"Below break-even the EAGER time is nearly flat in the small dimension (event, ws), RTX {k}",
              "; ".join(parts), "P1b eager_ws / native median_ms, varying dimension ≤ 128")


# ------------------------------------------------------------------------------------------------------------- C5
def c5():
    section("C5. Exponent range: the runtime fallback (cliff) and special values")
    emu_b, fb_b, both = [], [], defaultdict(set)
    per_gpu = defaultdict(set)
    for ph in ("P1c_ranges", "P2e_followup"):
        for (m, name), rs in load(ph).items():
            mode, cfg = split(name)
            if mode != "eager_ws" or "inner" not in cfg:
                continue
            for r in rs:
                b = int(r["bits_last"]); p = path_of(r)
                (emu_b if p == "emu" else fb_b).append(b)
                both[b].add(p)
                per_gpu[(cfg, int(r["k"]))].add((m, r["bits_last"], p))
    amb = sorted(b for b, s in both.items() if len(s) > 1)
    claim("C5.1", "Runtime fallback threshold (inner-range inputs, EAGER)",
          f"emulated up to {max(emu_b)} bits, fell back from {min(fb_b)} bits; bit counts that went both ways: {amb}; "
          f"always emulated ≤{min(fb_b) - 1}, always fell back ≥{max(emu_b) + 1}", "P1c + P2e fine grid")
    dis = sum(1 for v in per_gpu.values() if len({(b, p) for _, b, p in v}) > 1)
    claim("C5.2", "Bits and fallback decisions are identical on all three GPUs",
          f"{dis} of {len(per_gpu)} inner configs disagree across GPUs", "P1c + P2e")
    g = load("P1c_ranges")
    for k in KEYS:
        v = by_config(g, k)
        a = v["range:inner:phi7.75:k4096"]; b = v["range:inner:phi8:k4096"]
        claim(f"C5.3.{k}", f"Either side of the cliff at 4096³ (EAGER+ws), RTX {k}",
              f"φ7.75: {spd(a['native'], a['eager_ws']):.2f}×, bits {a['eager_ws'][0]['bits_last']}, err "
              f"{e(float(a['eager_ws'][0]['err_comp_max']))} vs native {e(float(a['native'][0]['nat_err_comp_max']))}; "
              f"φ8: {spd(b['native'], b['eager_ws']):.2f}× ({path_of(b['eager_ws'][0])})", "P1c")
    rc = [int(r["bits_last"]) for (m, name), rs in g.items() if ":rowcol:" in name and "eager_ws" in name for r in rs]
    claim("C5.4", "Row/column scaling (φ up to 16) does not raise the bits needed", f"bits {min(rc)}–{max(rc)}",
          "P1c rowcol")
    oz = {}
    for (m, name), rs in g.items():
        if ":ozaki:" in name and split(name)[0] == "eager_ws":
            oz[float(re.search(r"phi([\d.]+)", name).group(1))] = (rs[0]["bits_last"], path_of(rs[0]))
    claim("C5.5", "φ-generated (`ozaki`) inputs", "; ".join(f"φ{p:g}: {b} {pa}" for p, (b, pa) in sorted(oz.items())),
          "P1c ozaki, EAGER")
    g = load("P2c_bits")
    for k in KEYS:
        v = by_config(g, k)
        ref = v["bits:u11:ref"]["native"]

        def s(c):
            return med(ref, "median_ms") / med(v[c]["eager_ws"], "median_ms"), float(v[c]["eager_ws"][0]["err_comp_max"])
        cs = ("bits:u11:fixed24", "bits:u11:fixed48", "bits:u11:fixed53", "bits:u11:ref", "bits:u11:dyncap64", "bits:u11:fixed96")
        parts = [f"{c.split(':')[-1]} {s(c)[0]:.1f}× err {e(s(c)[1])}" for c in cs]
        claim(f"C5.6.{k}", f"Precision knob, u11 4096³ (EAGER), RTX {k}", "; ".join(parts), "P2c (baseline = native mode of `ref`)")
        rel = [f"{c.split(':')[-1]} {s(c)[0] / s('bits:u11:ref')[0]:.2f}" for c in cs]
        claim(f"C5.6r.{k}", f"Precision knob, speed relative to the default dynamic setting (same run, Table II), RTX {k}",
              "; ".join(rel), "P2c eager_ws median_ms ratios")
    fbs = [path_of(v["eager_ws"][0]) for k in KEYS for c, v in by_config(g, k).items() if "inner_phi6:dyncap" in c]
    claim("C5.7", "A dynamic cap below the needed bits falls back (no silent accuracy loss)",
          f"inner φ6 (94 bits) with cap 64/79: paths {Counter(fbs)}", "P2c")
    # Fixed mantissa control on an input that needs more bits: emulated (no fallback), error vs native, per setting.
    # Errors are taken per GPU and must agree (bit-identical emulation, C1.3); report one value if they do.
    fx = {}
    for c in ("fixed24", "fixed48", "fixed53", "fixed56", "fixed64", "ref"):
        errs, nats, paths = set(), set(), set()
        for k in KEYS:
            r = by_config(g, k)[f"bits:inner_phi6:{c}"]["eager_ws"][0]
            errs.add(e(float(r["err_comp_max"]))); nats.add(e(float(r["nat_err_comp_max"]))); paths.add(path_of(r))
        fx[c] = (errs, nats, paths)
    rat = []
    for ph in ("P1c_ranges", "P2c_bits", "P2e_followup"):
        for _, rs in load(ph).items():
            for r in rs:
                if path_of(r) == "fallback" and float(r["nat_err_comp_max"]) > 0 and float(r["err_comp_max"]) >= 0:
                    rat.append(float(r["err_comp_max"]) / float(r["nat_err_comp_max"]))
    claim("C5.12", "A call that falls back has native accuracy (error / in-process native error), exponent-range inputs",
          f"{len(rat)} fallback rows: ratio {min(rat):.2f}–{max(rat):.2f}", "P1c/P2c/P2e err_comp_max / nat_err_comp_max")
    # The opt-in mantissa-bit counter (cublasSetFixedPointEmulationMantissaBitCountPointer) against the CUPTI path,
    # over every GEMM row with emulation enabled. The bench presets it to NOBITS before the last timed call.
    gem = ("P1a_square", "P1b_shapes", "P1c_ranges", "P2a_decision", "P2b_version", "P2c_bits", "P2d_edges", "P2e_followup")
    tab, fbv, emv = Counter(), Counter(), []
    acc_ok, acc_bad, acc_noref, acc_max = 0, 0, 0, 0.0
    for ph in gem:
        for _, rs in load(ph).items():
            for r in rs:
                if r.get("env_emulate") != "1" or r.get("kern_n") in (None, "", "-1"):
                    continue
                p = path_of(r)
                written = r["bits_last"] != NOBITS
                tab[(p, written)] += 1
                if p == "fallback" and written:
                    fbv[int(r["bits_last"])] += 1
                if p == "emu" and written:
                    emv.append(int(r["bits_last"]))
                if p == "emu" and r["mcontrol"] != "fixed":  # dynamic precision (default or user cap)
                    er, na = float(r["err_comp_max"]), float(r["nat_err_comp_max"])
                    if er < 0 or na < 0:
                        acc_noref += 1
                    elif er > na:
                        acc_bad += 1
                    else:
                        acc_ok += 1
                        acc_max = max(acc_max, er / na if na > 0 else 0.0)
    nrow = sum(tab.values())
    claim("C5.13", "Opt-in mantissa-bit counter vs path (all GEMM phases, all GPUs and releases, emulation enabled)",
          f"{nrow} rows; declined {tab[('native', False)]} unwritten / {tab[('native', True)]} written; emulated "
          f"{tab[('emu', True)]} written / {tab[('emu', False)]} unwritten (bits {min(emv)}–{max(emv)}); fell back "
          f"{tab[('fallback', True)]} written / {tab[('fallback', False)]} unwritten, values {dict(sorted(fbv.items()))} "
          f"(−1 = NaN/Inf inputs; 94 = user caps 64/79; 104/106 = CUDA 13.1–13.3)",
          "bits_last vs CUPTI path; bench/bin/dgemm_bench_v5.cu presets the counter to -12345")
    claim("C5.14", "Dynamic precision (default or user cap) never less accurate than native, all emulated GEMM rows",
          f"{acc_ok} rows err ≤ native (largest ratio {acc_max:.2f}); {acc_bad} rows worse; {acc_noref} decision-only "
          f"rows without a reference", "err_comp_max vs nat_err_comp_max, mcontrol ≠ fixed, path emu")
    claim("C5.11", "Fixed mantissa bits on a wide-range input (inner φ6, needs 94 bits), 4096³ EAGER: emulated "
          "without fallback; error vs native (all three GPUs)",
          "; ".join(f"{c} {'/'.join(sorted(p))} err {'/'.join(sorted(er))} (native {'/'.join(sorted(na))})"
                    for c, (er, na, p) in fx.items()), "P2c inner_phi6 err_comp_max, nat_err_comp_max")
    g = load("P2d_edges")
    for k in KEYS:
        v = by_config(g, k)
        parts = []
        for inj in ("nan", "inf", "ninf", "overflow", "subnormal", "huge", "zeroblock"):
            x = v[f"edge:inject:{inj}"]
            parts.append(f"{inj}: {path_of(x['eager_ws'][0])} {spd(x['native'], x['eager_ws']):.2f}×")
        claim(f"C5.8.{k}", f"Special-value inputs, 2048³, EAGER+ws, RTX {k}", "; ".join(parts), "P2d inject")
    ov = []
    for p in glob.glob(str(FINAL / "P2e_followup" / "overflow*" / "overflow.csv")):
        ov += list(csv.DictReader(open(p)))
    emu_ok = [r for r in ov if r["mode"] != "native" and r["env_strategy"] == "EAGER"]
    nat = [r for r in ov if r["mode"] == "native"]
    sign = lambda r: 1 if r["class_rc"] == "+inf" else -1 if r["class_rc"] == "-inf" else 0
    cpu = []
    for p in glob.glob(str(FINAL / "P2e_followup" / "overflow*" / "cpu_openblas.csv")):
        cpu += list(csv.DictReader(open(p)))
    seeds = glob.glob(str(FINAL / "P2e_followup" / "overflow_seeds*" / "overflow.csv"))
    inputs = 1 + sum(1 for p in seeds for r in csv.DictReader(open(p)) if r["mode"] == "native")  # default seed + seeds
    # Per input: native GPU DGEMM is deterministic, so the default seed run on three GPUs counts once.
    by_in = defaultdict(lambda: defaultdict(set))  # (n, exact_sign, input id) -> impl -> correct?
    for p in glob.glob(str(FINAL / "P2e_followup" / "overflow*" / "overflow.csv")):
        rows = list(csv.DictReader(open(p)))
        seeded = "seeds" in p
        for i, r in enumerate(rows):
            inp = f"seed{i // 2 + 1}" if seeded else "default"
            impl = "native" if r["mode"] == "native" else ("eager" if r["env_strategy"] == "EAGER" else "default")
            by_in[inp][impl].add(sign(r) == int(r["exact_sign"]))
    per = {impl: sum(1 for d in by_in.values() if impl in d and all(d[impl])) for impl in ("eager", "native")}
    gpus_default = sum(1 for p in glob.glob(str(FINAL / "P2e_followup" / "overflow_gpu*" / "overflow.csv")))
    claim("C5.9i", "Overflowing entry, counted per input",
          f"{len(by_in)} inputs (default seed on {gpus_default} GPUs, {len(by_in) - 1} further seeds on the RTX 4090); "
          f"EAGER emulation correct-signed Inf for {per['eager']}/{len(by_in)}; native GPU DGEMM for "
          f"{per['native']}/{len(by_in)}; CPU OpenBLAS NaN for {sum(r['class_rc'] == 'nan' for r in cpu)}/{len(cpu)}",
          "P2e overflow*/overflow.csv (seeds file: native/eager pairs per seed), cpu_openblas.csv")
    claim("C5.9", "Overflowing entry (about 1e400 × a random dot product): what each implementation returns",
          f"{inputs} distinct inputs; EAGER emulation correct-signed Inf in {sum(sign(r) == int(r['exact_sign']) for r in emu_ok)}"
          f"/{len(emu_ok)} rows; native GPU DGEMM correct-signed Inf in {sum(sign(r) == int(r['exact_sign']) for r in nat)}"
          f"/{len(nat)} rows; CPU OpenBLAS: {Counter(r['class_rc'] for r in cpu)}", "P2e overflow*/overflow.csv, cpu_openblas.csv")
    # Exact value of the overflowing entry: rebuild each input from its seed exactly as workloads/overflow_class.py
    # does (NumPy generator), scale row r of A and column c of B by 1e200 in FP64, and sum the products of the scaled
    # inputs in exact rational arithmetic. Needs NumPy (uv sync --only-group analysis).
    try:
        import numpy as np
        from fractions import Fraction
        vals, agree = [], 0
        for r0 in cpu:
            s, n = int(r0["seed"]), int(r0["n"])
            rg = np.random.default_rng(s)
            A0, B0 = rg.uniform(-1, 1, (n, n)), rg.uniform(-1, 1, (n, n))
            a, b = A0[n // 3, :] * 1e200, B0[:, n // 2] * 1e200
            x = sum(Fraction(u) * Fraction(v) for u, v in zip(a.tolist(), b.tolist())) / Fraction(10) ** 400
            vals.append(float(x)); agree += (x > 0) - (x < 0) == int(r0["exact_sign"])
        claim("C5.15", "Overflowing entry: exact value from the scaled inputs (rational arithmetic), and the recorded "
              "reference sign (FP64 dot product of the unscaled values)",
              f"{len(vals)} inputs; |exact| {min(abs(v) for v in vals):.2f}e400 – {max(abs(v) for v in vals):.2f}e400; "
              f"recorded sign correct for {agree}/{len(vals)}", "P2e overflow_seeds*/cpu_openblas.csv seeds; overflow_class.py generator")
    except ImportError:
        claim("C5.15", "Overflowing entry: exact value from the scaled inputs", "needs NumPy", "-")
    g = load("P2b_version")
    for k in KEYS:
        v = by_config(g, k)
        parts = []
        for ver in ("13.1", "13.2", "13.3", "13.4"):
            x = v.get(f"ver{ver}:inner:phi7.5", {}).get("eager_ws")
            y = v.get(f"ver{ver}:sq:n4096", {})
            if x and y:
                parts.append(f"CUDA {ver}: φ7.5 {path_of(x[0])}, sq4096 EAGER {spd(y['native'], y['eager_ws']):.1f}×")
        y4 = v.get("ver13.4:sq:n4096", {}).get("eager_ws")
        rel = [med(y4, "median_ms") / med(v[f"ver{ver}:sq:n4096"]["eager_ws"], "median_ms")
               for ver in ("13.1", "13.2", "13.3") if f"ver{ver}:sq:n4096" in v]
        parts.append(f"older releases' EAGER speed / cuBLAS 13.8 (13.4) in the same run: {rng(rel, 3)} "
                     f"(≈{100 * st.mean(rel):.0f} %)")
        claim(f"C5.10.{k}", f"cuBLAS version axis (CUDA 13.1–13.3 = Ozaki-I pipeline, 13.4 = Ozaki-II), RTX {k}",
              "; ".join(parts), "P2b")


# ------------------------------------------------------------------------------------------------------------- C6
def c6():
    section("C6. End-to-end randomized SVD")
    g = defaultdict(list)
    for p in glob.glob(str(FINAL / "P4_rsvd" / "*" / "rsvd.csv")):
        for r in csv.DictReader(open(p)):
            g[(model(r["gpu"]), r["matrix"], r["mode"], int(r["l"]), int(r["q"]))].append(r)
    t = lambda key: st.median(float(r["t_median_s"]) for r in g[key])
    for k in KEYS:
        for (l, q) in ((64, 0), (256, 0), (512, 2), (1024, 2)):
            parts = []
            for md in ("zc", "zc_eager", "fix", "fix_eager", "fp32"):
                if (k, "exp", md, l, q) in g:
                    parts.append(f"{md} {t((k, 'exp', 'native', l, q)) / t((k, 'exp', md, l, q)):.2f}")
            claim(f"C6.1.{k}.l{l}q{q}", f"rSVD speedup vs native, `exp`, l={l} q={q}, RTX {k}", ", ".join(parts), "P4 rsvd.csv")
        below = sorted({(mat, l, q, round(t((k, mat, "native", l, q)) / t(key), 2)) for key in g
                        for (m, mat, md, l, q) in [key] if m == k and md == "zc" and t((k, mat, "native", l, q)) / t(key) < 0.99})
        claim(f"C6.2.{k}", f"rSVD cells where zero code (env var only) is slower than native, RTX {k}",
              "; ".join(f"{a} l={b} q={c} {d}×" for a, b, c, d in below) or "none", "P4")
        sc = [t((k, "scaled8", "native", l, 2)) / t((k, "scaled8", "fix_eager", l, 2)) for l in (256, 512, 1024)]
        ex = [t((k, "exp", "native", l, 2)) / t((k, "exp", "fix_eager", l, 2)) for l in (256, 512, 1024)]
        claim(f"C6.3.{k}", f"Badly scaled input gains less (fix_eager, q=2, l=256–1024), RTX {k}",
              f"scaled8 {rng(sc)}× vs exp {rng(ex)}×", "P4")
    acc = defaultdict(list)
    for (m, mat, md, l, q), rs in g.items():
        if mat == "poly":
            continue  # algorithm-limited (σ error 2.3e-7 in every FP64 mode)
        grp = "fp64 native" if md == "native" else "fp64 emulated/zero-code" if md in ("zc", "zc_eager", "fix", "fix_eager") else md
        acc[grp] += [float(r["sv_relerr_top10"]) for r in rs]
    claim("C6.4", "Top-10 singular-value relative error by arithmetic (all GPUs, exp/lrnoise/scaled4/scaled8)",
          "; ".join(f"{k} {e(min(v))}–{e(max(v))}" for k, v in sorted(acc.items())), "P4 sv_relerr_top10")
    pacc = defaultdict(list)
    for (m, mat, md, l, q), rs in g.items():
        if mat == "poly":
            pacc[md] += [float(r["sv_relerr_top10"]) for r in rs]
    claim("C6.7", "`poly` is algorithm-limited: the same top-10 singular-value error range in every FP64 mode, native included",
          "; ".join(f"{k} {e(min(v))}–{e(max(v))}" for k, v in sorted(pacc.items())), "P4 sv_relerr_top10, matrix poly")
    # Per setting: top-10 singular-value error of each FP64 mode vs native at the same (GPU, matrix, l, q), poly excluded.
    sv = defaultdict(list)
    for p in glob.glob(str(FINAL / "P4_rsvd" / "*" / "rsvd.csv")):
        for r in csv.DictReader(open(p)):
            if r["matrix"] != "poly":
                sv[(r["gpu"], r["matrix"], r["l"], r["q"], r["mode"])].append(float(r["sv_relerr_top10"]))
    parts = []
    for md in ("fix_eager", "zc_eager", "fix", "zc"):
        rat = [st.median(v) / st.median(sv[k[:4] + ("native",)]) for k, v in sv.items()
               if k[4] == md and k[:4] + ("native",) in sv]
        top = max(st.median(v) for k, v in sv.items() if k[4] == md)
        parts.append(f"{md}: {len(rat)} settings, higher than native in {sum(x > 1 for x in rat)}, ratio "
                     f"{min(rat):.2f}–{max(rat):.2f}, largest error {e(top)}")
    claim("C6.8", "Top-10 singular-value error per setting vs native (same GPU, matrix, l, q; poly excluded): at the "
          "level of FP64, though often above native", "; ".join(parts), "P4 sv_relerr_top10, median over repeats")
    above, floor, floor_abs = [], [], []
    for (m, mat, md, l, q), rs in g.items():
        if md in ("fix_eager",) and (m, mat, "native", l, q) in g:
            a = st.median(float(r["resid_fro_rel"]) for r in rs)
            b = st.median(float(r["resid_fro_rel"]) for r in g[(m, mat, "native", l, q)])
            if b < 1e-14:  # both at the FP64 rounding floor of the reconstruction: ratios are rounding noise
                floor.append(a / b); floor_abs.append(a)
            else:
                above.append(a / b)
    claim("C6.5", "Residual ‖A−QB‖/‖A‖, emulated (fix_eager) / native, all GPUs, matrices, l, q",
          f"where native residual ≥ 1e-14 ({len(above)} cells): ratio {rng(above, 3)}; at the rounding floor "
          f"(native < 1e-14, {len(floor)} cells): ratio {rng(floor, 2)}, emulated lower in "
          f"{sum(x < 0.9 for x in floor)}, higher in {sum(x > 1.1 for x in floor)}, emulated ≤ {e(max(floor_abs))}",
          "P4 resid_fro_rel (median over 3 repeats; repeats agree)")
    for k in KEYS:
        f32 = [t((k, mat, "native", l, q)) / t((k, mat, "fp32", l, q)) for (m, mat, md, l, q) in g if m == k and md == "fp32"]
        claim(f"C6.6.{k}", f"FP32 rSVD speedup vs FP64 native (not an FP64 substitute: see C6.4), RTX {k}", f"{rng(f32)}×", "P4")
    # Fallback in the workload: share of the gain over native (S - 1) lost on scaled8 vs exp, q=2, l=256..1024.
    sp = lambda k, mat, md, l, q: t((k, mat, "native", l, q)) / t((k, mat, md, l, q))
    lost, dev4, below = defaultdict(list), [], []
    for k in KEYS:
        for md in ("zc", "zc_eager", "fix", "fix_eager"):
            for l in (256, 512, 1024):
                se, s4, s8 = (sp(k, mat, md, l, 2) for mat in ("exp", "scaled4", "scaled8"))
                if se > 1.02:  # the mode emulates there
                    dev4.append(abs(s4 / se - 1))
                    (lost[k] if s8 >= 1 else below).append(1 - (s8 - 1) / (se - 1))
                    if s8 < 1:
                        below[-1] = f"{k} {md} l={l}: exp {se:.2f}× → scaled8 {s8:.2f}× (lost {100 * below[-1]:.0f} %)"
    claim("C6.9", "Fallback in the workload: share of the gain over native (S − 1) lost on scaled8 vs exp, q=2, "
          "l=256–1024, every mode that emulates; ranges over the cases where scaled8 stays above native, the cases "
          "below native listed separately; scaled4 vs exp",
          "; ".join(f"RTX {k} {rng([100 * x for x in v], 0)} %" for k, v in lost.items())
          + f"; all {rng([100 * x for x in sum(lost.values(), [])], 0)} %; below native: {'; '.join(below)}"
          + f"; scaled4 within {100 * max(dev4):.1f} % of exp", "P4 rsvd.csv (median over repeats)")
    # Spread between repeats, (max - min) / median of t_median_s, per (GPU, matrix, mode, l, q).
    spr, t3 = defaultdict(list), defaultdict(list)
    for (m, mat, md, l, q), rs in g.items():
        v = [float(r["t_median_s"]) for r in rs]
        if len(v) > 1:
            s = (max(v) - min(v)) / st.median(v)
            spr[md].append(s)
            if mat == "exp" and (l, q) in ((64, 0), (256, 0), (512, 2)):
                t3[md].append(s)
    claim("C6.10", "rSVD spread between repeats, (max − min)/median, by mode: all settings; Table III cells (exp, "
          "l=64 q=0, l=256 q=0, l=512 q=2)",
          "; ".join(f"{md}: max {100 * max(v):.1f} % (> 10 % in {sum(x > 0.1 for x in v)} of {len(v)}), Table III "
                    f"max {100 * max(t3[md]):.1f} %" for md, v in sorted(spr.items())), "P4 rsvd.csv t_median_s per repeat")


# ------------------------------------------------------------------------------------------------------------- C7
def c7():
    section("C7. Precision-cap mechanism (P2e kernel dumps)")
    for d in sorted(glob.glob(str(FINAL / "P2e_followup" / "P2e_followup_gpu*"))):
        k = model(next(csv.DictReader(open(glob.glob(d + "/gpu*.csv")[0])))["gpu"])
        res = {}
        for s in ("ref", "dyncap64", "dyncap79", "fixed53", "fixed96"):
            f = glob.glob(f"{d}/kern/cap:eager_ws:u11:{s}.r*.tsv")
            kern = [l.rstrip("\n").split("\t", 1) for l in open(f[0]) if "\t" in l]
            tot = sum(int(t) for t, _ in kern)
            gemm = sum(int(t) for t, n in kern if "biu" in n)
            prol = sum(int(t) for t, n in kern if "prolog_mod_dfma" in n)
            inv = [n for _, n in kern if "inverse_scaling" in n]
            res[s] = (len(re.findall(r"Lj\d+E", inv[0])), tot / 1e6, gemm / 1e6, prol / 1e6)
        claim(f"C7.1.{k}", f"Per-call kernel time by precision setting, u11 4096³ EAGER, RTX {k} "
              "(moduli; total / INT8 GEMM / modular prolog, ms)",
              "; ".join(f"{s}: {m} moduli, {t:.1f} / {gm:.1f} / {p:.1f}" for s, (m, t, gm, p) in res.items())
              + f"; default prolog share {100 * res['ref'][3] / res['ref'][1]:.0f} %", "P2e cap:* CUPTI kernel dumps (1 call)")


# ------------------------------------------------------------------------------------------------------------- C8
def c8():
    """Table IV, 'what a user gets': speedup over native per level of effort, at representative points.
    zero code = default_ws0, wall clock vs native wall; one-line fix = default_ws, event vs native event;
    EAGER zero code = eager_ws0, wall; EAGER + fix = eager_ws, event. rSVD (exp, l=512 q=2): zc / fix / fix_eager, wall clock (as measured)."""
    section("C8. What a user gets (Table IV)")
    pts = [("P1a_square", "sq:n1024", "square n=1024"), ("P1a_square", "sq:n4096", "square n=4096"),
           ("P1b_shapes", "shape:skinny:m8192_n256_k8192", "skinny n=256")]
    g4 = defaultdict(list)
    for p in glob.glob(str(FINAL / "P4_rsvd" / "*" / "rsvd.csv")):
        for r in csv.DictReader(open(p)):
            g4[(model(r["gpu"]), r["matrix"], r["mode"], int(r["l"]), int(r["q"]))].append(r)
    t4 = lambda key: st.median(float(r["t_median_s"]) for r in g4[key])
    for k in KEYS:
        parts = []
        for ph, cfg, label in pts:
            v = by_config(load(ph), k)[cfg]
            zc = spd(v["native"], v["default_ws0"], "wall_sync_ms")
            ez = spd(v["native"], v["eager_ws0"], "wall_sync_ms")
            fx, eg = spd(v["native"], v["default_ws"]), spd(v["native"], v["eager_ws"])
            pa = "/".join(path_of(v[m][0]) for m in ("default_ws0", "default_ws", "eager_ws0", "eager_ws"))
            parts.append(f"{label}: zero code {zc:.2f}, fix {fx:.2f}, EAGER zero code {ez:.2f}, EAGER+fix {eg:.2f} "
                         f"(paths {pa})")
        nat = t4((k, "exp", "native", 512, 2))
        parts.append("rSVD exp l=512 q=2: zero code {:.2f}, fix {:.2f}, EAGER zero code {:.2f}, EAGER+fix {:.2f}".format(
            *(nat / t4((k, "exp", m, 512, 2)) for m in ("zc", "fix", "zc_eager", "fix_eager"))))
        claim(f"C8.{k}", f"Speedup over native by level of effort, RTX {k}", "; ".join(parts),
              "P1a/P1b (default_ws0, eager_ws0 wall; default_ws, eager_ws event), P4 rsvd.csv (wall)")


if __name__ == "__main__":
    setup_and_quality(); c1(); c2(); c3(); c4(); c5(); c6(); c7(); c8()
    print("# Claims ledger (generated by analysis/ledger.py; do not edit by hand)\n")
    print("Every number the paper quotes must appear here (or in HEADLINE.md) with its source. Speedup = native "
          "time / mode time, same config, native *mode* as baseline. Event = GPU time (CUDA events); wall = per call "
          "with a host sync. 'Fixed' = workspace set (`*_ws`). Zero code = env var only (`*_ws0`, wall clock).\n")
    for cid, text, value, src in OUT:
        if cid is None:
            print(f"\n## {text}\n\n| id | claim | value | source |\n|---|---|---|---|")
        else:
            print(f"| {cid} | {text} | {value} | {src} |")
