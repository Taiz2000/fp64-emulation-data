"""Summarise run_grid.sh CSVs. Groups rows by the config name in the tag (LABEL:<name...>:repN),
medians over repeats. Columns: path from the bit-count pointer (emu/native), speedup vs the
'native' config with the same suffix (e.g. ws:native:n4096 for ws:eager_ws:n4096).

Usage: python3 bench/summarize_grid.py results/raw/<run>/gpu*.csv
"""
import csv
import statistics as st
import sys
from collections import defaultdict

groups = defaultdict(list)
for p in sys.argv[1:]:
    for r in csv.DictReader(open(p)):
        parts = r["tag"].split(":")
        name = ":".join(parts[1:-1])  # drop label and repN
        groups[(r["gpu"], name)].append(r)


def med(rs, col):
    return st.median(float(r[col]) for r in rs)


def path_of(r):
    """emu | fallback (ADP runtime fallback inside emulation) | native (incl. heuristic decline).
    Uses the CUPTI kernel probe when present (v3+), else the bit-count pointer (valid only for easy inputs)."""
    if r.get("kern_n") not in (None, "", "-1"):
        if float(r["kern_emu_frac"]) > 0.5:
            return "emu"
        return "fallback" if int(r["kern_n"]) > 3 else "native"
    return "emu?" if r["bits_last"] != "-12345" else "native"


def native_key(name):
    p = name.split(":")
    return ":".join([p[0], "native"] + p[2:])


def wall(r):
    v = float(r.get("wall_sync_ms") or -1)
    return v if v > 0 else float("nan")


nat_rows = {k: [r for r in v if r["status"] == "ok"] for k, v in groups.items()
            if k[1].split(":")[1:2] == ["native"] and any(r["status"] == "ok" for r in v)}
nat = {k: med(v, "median_ms") for k, v in nat_rows.items()}
nat_wall = {k: st.median(wall(r) for r in v) for k, v in nat_rows.items()}

cols = ("gpu", "config", "ok/runs", "med_ms", "spread%", "p90/med", "TFLOPS", "x_native", "path", "bits",
        "err_comp", "nat_err", "err_rel", "J/call", "wall_sync", "x_nat_wall")
print("\t".join(cols))
for (gpu, name) in sorted(groups, key=lambda k: (k[0], k[1].split(":")[0], k[1].split(":")[2:], k[1])):
    rs = groups[(gpu, name)]
    ok = [r for r in rs if r["status"] == "ok"]
    if not ok:
        print("\t".join([gpu[-11:], name, f"0/{len(rs)}"] + [""] * 13 + [rs[0]["status"]]))
        continue
    t = [float(r["median_ms"]) for r in ok]
    tm = st.median(t)
    p90 = st.median(float(r["p90_ms"]) / float(r["median_ms"]) for r in ok) if "p90_ms" in ok[0] else float("nan")
    bits = sorted({r["bits_last"] for r in ok})
    path = "/".join(sorted({path_of(r) for r in ok}))
    xn = nat.get((gpu, native_key(name)), float("nan")) / tm
    print("\t".join(map(str, (
        gpu[-11:], name, f"{len(ok)}/{len(rs)}", f"{tm:.3f}", f"{(max(t) - min(t)) / tm * 100:.1f}", f"{p90:.2f}",
        f"{med(ok, 'tflops'):.3f}", f"{xn:.2f}", path, "/".join(b if b != "-12345" else "-" for b in bits),
        f"{max(float(r['err_comp_max']) for r in ok):.1e}", f"{max(float(r['nat_err_comp_max']) for r in ok):.1e}",
        f"{max(float(r['err_rel_max']) for r in ok):.1e}",
        f"{st.median(float(r['energy_j_timed']) / int(r['reps']) for r in ok):.3f}",
        f"{st.median(wall(r) for r in ok):.3f}",
        f"{nat_wall.get((gpu, native_key(name)), float('nan')) / st.median(wall(r) for r in ok):.2f}"))))
