"""Acceptance checks for one run_grid.sh output directory.
Usage: python3 bench/qa.py RUN_DIR [RUN_DIR ...]      writes RUN_DIR/qa.txt, prints a verdict per dir.

FAIL : missing / failed rows, foreign processes, path or output hash changing between repeats, nondeterminism.
WARN : rep-median spread > 5% (rerun the line), thermal/HW throttling, native TFLOPS off the PoC value by > 3%,
       emulated error worse than native on well-scaled inputs.
"""
import csv
import glob
import os
import re
import statistics as st
import sys
from collections import defaultdict

NATIVE_TFLOPS = {"RTX 3090": 0.54, "RTX 4090": 1.23, "RTX 5060 Ti": 0.36}
BAD_REASONS = 0x8 | 0x20 | 0x40 | 0x80  # HW slowdown, SW thermal, HW thermal, HW power brake


def path_of(r):
    if r.get("kern_n") in (None, "", "-1"):
        return "?"
    if float(r["kern_emu_frac"]) > 0.5:
        return "emu"
    return "fallback" if int(r["kern_n"]) > 3 else "native"


def check(d):
    fails, warns = [], []
    grid = [l.split("|")[0] for l in open(f"{d}/grid.txt") if l.strip() and not l.lstrip().startswith("#")]
    reps = int(re.search(r"REPEATS=(\d+)", open(f"{d}/runner.txt").read()).group(1))
    csvs = glob.glob(f"{d}/gpu*.csv")
    rows = [r for p in csvs for r in csv.DictReader(open(p))]
    log = "".join(open(p).read() for p in glob.glob(f"{d}/gpu*.log"))
    if "DONE" not in log:
        fails.append("run not finished (no DONE in log)")
    nfs = len(re.findall(r" (FAIL|SKIP) ", log))
    if nfs:
        fails.append(f"{nfs} FAIL/SKIP lines in log")
    by = defaultdict(list)
    for r in rows:
        by[":".join(r["tag"].split(":")[1:-1])].append(r)
    for name in grid:
        rs = by.get(name, [])
        ok = [r for r in rs if r["status"] == "ok"]
        if len(ok) < reps:
            fails.append(f"{name}: {len(ok)}/{reps} ok rows ({','.join(sorted({r['status'] for r in rs})) or 'none'})")
        if not ok:
            continue
        if any(int(r["other_procs_pre"]) for r in ok):
            fails.append(f"{name}: foreign process on the GPU")
        if len({path_of(r) for r in ok}) > 1:
            fails.append(f"{name}: path differs between repeats {[path_of(r) for r in ok]}")
        if len({r["hash_C"] for r in ok}) > 1:
            fails.append(f"{name}: output hash differs between repeats")
        if any(r["deterministic"] != "1" for r in ok):
            fails.append(f"{name}: nondeterministic within a process")
        # *_ws0 (zero code change, emulated) has intrinsic per-call jitter from the mempool remap (the zero-code memory penalty),
        # so it gets a looser limit; everything else must repeat within 5%
        lim = 0.15 if name.split(":")[1].endswith("_ws0") else 0.05
        t = [float(r["median_ms"]) for r in ok]
        if len(t) >= 3 and (max(t) - min(t)) / st.median(t) > lim:
            warns.append(f"{name}: rep-median spread {(max(t) - min(t)) / st.median(t) * 100:.1f}% -> rerun")
        w = [float(r["wall_sync_ms"]) for r in ok if float(r["wall_sync_ms"]) > 0]
        if len(w) >= 3 and (max(w) - min(w)) / st.median(w) > lim:
            warns.append(f"{name}: wall_sync spread {(max(w) - min(w)) / st.median(w) * 100:.1f}%")
        bad = [r for r in ok if int(r["clk_reasons_or"], 16) & BAD_REASONS]
        if bad:
            warns.append(f"{name}: throttle reasons {sorted({r['clk_reasons_or'] for r in bad})}")
        r0 = ok[0]
        mode = name.split(":")[1]
        sq = r0["m"] == r0["n"] == r0["k"] and int(r0["m"]) >= 2048 and r0.get("batch", "1") == "1"
        if mode == "native" and sq and r0["dist"] == "u11":
            ref = next((v for kk, v in NATIVE_TFLOPS.items() if kk in r0["gpu"]), None)
            tf = st.median(float(r["tflops"]) for r in ok)
            if ref and abs(tf / ref - 1) > 0.03:
                warns.append(f"{name}: native {tf:.3f} TFLOPS vs PoC {ref}")
        if (path_of(r0) == "emu" and r0["dist"] == "u11" and r0.get("inject", "none") == "none"
                and float(r0["nat_err_comp_max"]) > 0
                and max(float(r["err_comp_max"]) for r in ok) > float(r0["nat_err_comp_max"])):
            warns.append(f"{name}: emulated error {r0['err_comp_max']} > native {r0['nat_err_comp_max']}")
    verdict = "FAIL" if fails else "WARN" if warns else "PASS"
    txt = [f"verdict {verdict}", f"lines {len(grid)} x reps {reps}, rows {len(rows)}"]
    txt += [f"FAIL {x}" for x in fails] + [f"WARN {x}" for x in warns]
    open(f"{d}/qa.txt", "w").write("\n".join(txt) + "\n")
    return verdict, fails, warns


for d in sys.argv[1:]:
    v, f, w = check(d.rstrip("/"))
    print(f"{v:5} {os.path.relpath(d)}  ({len(f)} fail, {len(w)} warn)")
    for x in (f + w)[:15]:
        print("   ", x)
