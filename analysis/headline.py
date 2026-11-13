#!/usr/bin/env python3
"""Headline tables for the paper from results/final (stdlib only; run with system python3).

Pools the original run and its *_rerun for each phase/GPU.

Baseline: the `native` mode of the same config (median over all its rows). Only where a grid has no native mode
(P2b decision slices, P2c) does a cell fall back to the bench's per-row PEDANTIC native reference (ref_native_ms),
marked `*`; that reference can be slower than the native mode on small shapes. P2a/P2b decision rows are path-only
(`--no-ref`), so they have no speed at all and show `path`.

Timing: non-ws0 modes use the event-timed median (cudaEvent, GPU time); a separate section checks that their
wall-clock speedups agree. *_ws0 modes (zero code changes) use wall-clock per call (wall_sync_ms), because that is
what a zero-code user sees; their cells are marked 'w'.

Paths: emu = >50 % of kernel time in emulation kernels; fallback = emulation entered but the time went to a native
FP64 GEMM (ADP runtime fallback); native = heuristic declined (or emulation disabled).
Bits: mantissa bits reported by cuBLAS for emulated rows; `fb:N` = bits reported on rows that fell back at runtime
(the bit-count pointer is written even when ADP falls back).

Usage: python3 analysis/headline.py > summary/HEADLINE.md
"""
import csv
import glob
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / "results" / "final"
MODELS = [("3090", "RTX 3090 (sm_86)"), ("4090", "RTX 4090 (sm_89)"), ("5060 Ti", "RTX 5060 Ti (sm_120)")]
WS_PAD_MIB = 32  # bench/grids/make_final.sh: *_ws workspace = cuBLASLt-requested size (EAGER probe) + 32 MiB
NOBITS = "-12345"


def model(gpu):
    return next(k for k, _ in MODELS if k in gpu)


def req_mib(r):
    """Workspace cuBLAS requested, in whole MiB rounded down. The grid probe stored ceil(bytes / MiB) + WS_PAD_MIB;
    the cuBLASLt logs show the true requests are a whole MiB plus 0.008-0.25 MiB (claim C3.12), so ceil - 1 = floor.
    A probe value of exactly WS_PAD_MIB means no workspace was requested."""
    v = int(r["workspace_mb"]) - WS_PAD_MIB
    return max(v - 1, 0)


def path_of(r):
    if r.get("kern_n") not in (None, "", "-1"):
        if float(r["kern_emu_frac"]) > 0.5:
            return "emu"
        return "fallback" if int(r["kern_n"]) > 3 else "native"
    return "emu?" if r["bits_last"] != NOBITS else "native"


def load(phase):
    """{(model, name): [rows]} for phase + phase_rerun, ok rows only. name = tag without label and repeat."""
    g = defaultdict(list)
    for p in sorted(glob.glob(str(FINAL / phase / "*" / "gpu*.csv")) + glob.glob(str(FINAL / f"{phase}_rerun" / "*" / "gpu*.csv"))):
        for r in csv.DictReader(open(p)):
            if r["status"] != "ok":
                continue
            g[(model(r["gpu"]), ":".join(r["tag"].split(":")[1:-1]))].append(r)
    return g


def med(rs, col):
    if not rs:
        return float("nan")
    v = [float(r[col]) for r in rs if r.get(col) not in (None, "", "nan")]
    v = [x for x in v if x > 0] if col in ("wall_sync_ms", "ref_native_ms", "median_ms") else v
    return st.median(v) if v else float("nan")


def split(name):
    p = name.split(":")
    return p[1], ":".join([p[0]] + p[2:])  # mode, config


def by_config(g, key):
    """{config: {mode: rows}} for one GPU model."""
    rows = defaultdict(dict)
    for (m, name), rs in g.items():
        if m == key:
            mode, cfg = split(name)
            rows[cfg][mode] = rs
    return rows


def cell(rs, mode, nat=None):
    """nat = rows of the native mode for the same config (preferred baseline). Without it, fall back to the
    per-row PEDANTIC reference (ref_native_ms), marked '*'. Path-only rows (no reference, no native) show 'path'."""
    paths = "/".join(sorted({path_of(r) for r in rs}))
    if mode == "native":
        return f"{med(rs, 'median_ms'):.3g} ms"
    ev = med(nat, "median_ms") if nat else med(rs, "ref_native_ms")
    if ev != ev:
        return f"path {paths}"
    star = "" if nat else "*"
    if mode.endswith("_ws0"):
        wl = med(nat, "wall_sync_ms") if nat else ev
        wl = wl if wl == wl else ev
        return f"{wl / med(rs, 'wall_sync_ms'):.2f}w{star} {paths}"
    return f"{ev / med(rs, 'median_ms'):.2f}{star} {paths}"


def bits_col(v):
    emu = [r for md, rs in v.items() if md != "native" for r in rs if path_of(r) == "emu"]
    fb = [r for md, rs in v.items() if md != "native" for r in rs if path_of(r) == "fallback" and r["bits_last"] != NOBITS]
    out = "/".join(sorted({r["bits_last"] for r in emu}))
    fbb = "/".join(sorted({r["bits_last"] for r in fb}))
    if fbb:
        out = (out + " " if out else "") + f"fb:{fbb}"
    return out or "-"


def pivot(phase, title, modes=None, extra=None):
    g = load(phase)
    print(f"\n## {title}\n")
    for key, mname in MODELS:
        rows = by_config(g, key)
        if not rows:
            continue
        ms = modes or sorted({md for v in rows.values() for md in v}, key=lambda x: (x != "native", x))
        ms = [x for x in ms if any(x in v for v in rows.values())]
        hdr = ["config"] + ms + ["bits", "err emu", "err native"] + ([extra[0]] if extra else [])
        print(f"### {mname}\n")
        print("| " + " | ".join(hdr) + " |")
        print("|" + "---|" * len(hdr))
        for cfg in sorted(rows, key=natkey):
            v = rows[cfg]
            emu = [r for md, rs in v.items() if md != "native" for r in rs if path_of(r) == "emu"]
            anyr = [r for rs in v.values() for r in rs]
            e_emu = f"{max(float(r['err_comp_max']) for r in emu):.1e}" if emu else "-"
            nat_errs = [float(r["nat_err_comp_max"]) for r in anyr if float(r["nat_err_comp_max"]) >= 0]
            e_nat = f"{max(nat_errs):.1e}" if nat_errs else "-"
            cells = [cell(v[md], md, v.get("native")) if md in v else "" for md in ms]
            ex = [extra[1](v)] if extra else []
            print("| " + " | ".join([cfg] + cells + [bits_col(v), e_emu, e_nat] + ex) + " |")
        print()


def natkey(s):
    return [float(t) if re.fullmatch(r"\d+(\.\d+)?", t) else t for t in re.split(r"([0-9.]+)", s) if t]


def p2a():
    g = load("P2a_decision")
    print("\n## P2a decision map: which path does each strategy take?\n")
    print("P2a is path-only by design (`--no-ref --no-wall`, 1 repeat): no native reference per point. Where the "
          "default heuristic declines, `default_ws` *is* native, so EAGER/native = default_ws/eager_ws measures what "
          "the heuristic leaves on the table. Speedups where the default emulates come from P1b.\n")
    print("| GPU | points | default emulates | default declines | of those EAGER emulates | EAGER faster (s>1) | "
          "s>1.5 | EAGER slower (s<1) | EAGER declines too | default fallback | smallest min(m,n,k) default emulates |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for key, mname in MODELS:
        pts = by_config(g, key)
        n = em = dec = eem = f1 = f15 = slow = edec = fb = 0
        mins = []
        for cfg, v in pts.items():
            if not all(x in v for x in ("default_ws", "eager_ws")):
                continue
            n += 1
            d, e = v["default_ws"], v["eager_ws"]
            pd, pe = path_of(d[0]), path_of(e[0])
            if pd == "emu":
                em += 1
                mins.append(min(int(d[0][c]) for c in ("m", "n", "k")))
            else:
                dec += 1
                fb += pd == "fallback"
                if pe == "emu":
                    eem += 1
                    s = med(d, "median_ms") / med(e, "median_ms")
                    f1 += s > 1; f15 += s > 1.5; slow += s < 1
                else:
                    edec += 1
        if n:
            print(f"| {mname} | {n} | {em} | {dec} | {eem} | {f1} | {f15} | {slow} | {edec} | {fb} | "
                  f"{min(mins) if mins else '-'} |")


def breakeven():
    """P1b: per shape family, the varying dimension at which (a) EAGER with workspace beats native (event),
    (b) the default heuristic starts emulating, (c) zero-code default (wall) beats native."""
    g = load("P1b_shapes")
    fams = {"skinny": "n", "panel": "k", "shortwide": "m"}
    print("\n## P1b break-even (smallest varying dimension; baseline = native mode)\n")
    print("| GPU | family (varying dim) | EAGER ws > native | default starts emulating | zero-code default wall > native |")
    print("|---|---|---|---|---|")
    for key, mname in MODELS:
        for fam, dim in fams.items():
            pts = defaultdict(dict)
            for (m, name), rs in g.items():
                if m == key and name.startswith("shape:") and f":{fam}:" in name:
                    mode, cfg = split(name)
                    d = int(re.search(rf"{dim}(\d+)", cfg.split(":")[-1]).group(1))
                    pts[d][mode] = rs

            def first(test):
                for d in sorted(pts):
                    v = pts[d]
                    if "native" in v and test(v):
                        return str(d)
                return "never"
            nat = lambda v: med(v["native"], "median_ms")
            natw = lambda v: med(v["native"], "wall_sync_ms")
            a = first(lambda v: "eager_ws" in v and nat(v) / med(v["eager_ws"], "median_ms") > 1)
            b = first(lambda v: "default_ws" in v and path_of(v["default_ws"][0]) == "emu")
            c = first(lambda v: "default_ws0" in v and path_of(v["default_ws0"][0]) == "emu"
                      and natw(v) / med(v["default_ws0"], "wall_sync_ms") > 1)
            print(f"| {mname} | {fam} ({dim}) | {a} | {b} | {c} |")


def wall_check():
    """Fixed modes (*_ws, *_kp): does the wall-clock speedup (native wall / mode wall) match the event speedup?"""
    print("\n## Check: wall-clock vs event speedup for the fixed modes (`*_ws`, `*_kp`)\n")
    print("Wall speedup = native wall_sync / mode wall_sync; event speedup = native median / mode median. "
          "Rows where the two differ by more than 2 % are listed.\n")
    print("| phase | GPU | rows compared | max abs. difference | rows > 2 % |")
    print("|---|---|---|---|---|")
    outl = []
    for phase in ("P1a_square", "P1b_shapes", "P1c_ranges", "P2d_edges"):
        g = load(phase)
        for key, _ in MODELS:
            n, mx = 0, 0.0
            for cfg, v in by_config(g, key).items():
                nat = v.get("native")
                if not nat or med(nat, "wall_sync_ms") != med(nat, "wall_sync_ms"):
                    continue
                for md, rs in v.items():
                    if md == "native" or md.endswith("_ws0") or path_of(rs[0]) == "native":
                        continue
                    w = med(rs, "wall_sync_ms")
                    if w != w:
                        continue
                    se = med(nat, "median_ms") / med(rs, "median_ms")
                    sw = med(nat, "wall_sync_ms") / w
                    rel = sw / se - 1
                    n += 1
                    mx = max(mx, abs(rel))
                    if abs(rel) > 0.02:
                        outl.append((phase, key, cfg, md, se, sw))
            if n:
                k = sum(1 for o in outl if o[0] == phase and o[1] == key)
                print(f"| {phase} | {key} | {n} | {100 * mx:.1f} % | {k} |")
    if outl:
        print("\n| phase | GPU | config | mode | event speedup | wall speedup |\n|---|---|---|---|---|---|")
        for o in outl:
            print(f"| {o[0]} | {o[1]} | {o[2]} | {o[3]} | {o[4]:.2f} | {o[5]:.2f} |")


def default_below_native():
    """Does the default strategy ever emulate a config that is slower than native (event time, workspace set)?"""
    print("\n## Check: default strategy emulating below native (event time, `default_ws` vs native mode)\n")
    print("| phase | GPU | configs where default_ws emulates | of those slower than native | slowest (speedup) |")
    print("|---|---|---|---|---|")
    for phase in ("P1a_square", "P1b_shapes", "P1c_ranges", "P2d_edges"):
        g = load(phase)
        for key, _ in MODELS:
            n, slow, worst = 0, 0, (float("inf"), "-")
            for cfg, v in by_config(g, key).items():
                if "native" not in v or "default_ws" not in v or path_of(v["default_ws"][0]) != "emu":
                    continue
                n += 1
                s = med(v["native"], "median_ms") / med(v["default_ws"], "median_ms")
                slow += s < 1
                worst = min(worst, (s, cfg))
            if n:
                print(f"| {phase} | {key} | {n} | {slow} | {worst[1]} ({worst[0]:.2f}) |")


def p2d_extra(v):
    out = []
    for md, rs in sorted(v.items()):
        r = rs[0]
        if r.get("inject", "none") != "none":
            out.append(f"{md}: nf={r['n_nonfinite']}/{r['n_nonfinite_native']} cls≠{r['n_class_mismatch_native']}")
    return "; ".join(out)


def p3b_slopes():
    """{model: (run, ms/GiB at threshold 0, last row, max ms at threshold max)} from the valid P3b runs."""
    out = {}
    for p in sorted(glob.glob(str(FINAL / "P3b_mempool" / "*" / "mempool.csv"))):
        if (Path(p).parent / "INVALID.txt").exists():
            continue
        rows = [r for r in csv.DictReader(open(p)) if r["variant"] == "alloc" and r["touch"] == "0"]
        if not rows:
            continue
        t0 = sorted([r for r in rows if r["threshold"] == "0"], key=lambda r: int(r["size_mib"]))
        tm = [r for r in rows if r["threshold"] != "0"]
        big = [r for r in t0 if int(r["size_mib"]) >= 512]
        xs = [int(r["size_mib"]) / 1024 for r in big]; ys = [float(r["median_ms"]) for r in big]
        mx, my = st.mean(xs), st.mean(ys)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        out[model(rows[0]["gpu"])] = (Path(p).parent.name, slope, t0[-1], max(float(r["median_ms"]) for r in tm))
    return out


def p3b():
    print("\n## P3b mempool remap cost (cudaMallocAsync+cudaFreeAsync+sync, no touch)\n")
    print("| GPU | run | ms/GiB, release threshold 0 (cuBLAS default) | ms at largest size | threshold max: ms |")
    print("|---|---|---|---|---|")
    for key, (run, slope, last, tmax) in sorted(p3b_slopes().items(), key=lambda kv: kv[1][0]):
        print(f"| {key} | {run} | {slope:.1f} | {last['median_ms']} ms @ {last['size_mib']} MiB | {tmax:.3f} |")


def zero_code_penalty():
    """P1a: zero-code penalty vs the fixed mode of the same strategy, and the prediction workspace x remap rate."""
    rates = p3b_slopes()
    g = load("P1a_square")
    print("\n## Zero-code penalty model (P1a; EAGER on the 3090, default strategy elsewhere)\n")
    print("Penalty = zero-code time per call − fixed (`*_ws`) event time per call. Predicted = cuBLASLt-requested "
          f"workspace (the `*_ws` size − {WS_PAD_MIB} MiB pad, in whole MiB rounded down: req_mib) × the P3b remap rate (ms/GiB). `ws0 event` shows how much "
          "of the penalty is visible in GPU (event) time.\n")
    print("| GPU | n | fixed event ms | ws0 event ms | ws0 wall ms | penalty wall ms | penalty event ms | "
          "requested MiB | predicted ms | wall penalty / predicted |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for key, _ in MODELS:
        pre = "eager" if key == "3090" else "default"
        rows = by_config(g, key)
        for cfg in sorted(rows, key=natkey):
            v = rows[cfg]
            ws, ws0 = v.get(f"{pre}_ws"), v.get(f"{pre}_ws0")
            if not ws or not ws0 or path_of(ws[0]) != "emu":
                continue
            req = req_mib(ws[0])
            pred = req / 1024 * rates[key][1] if key in rates else float("nan")
            fe, we, ww = med(ws, "median_ms"), med(ws0, "median_ms"), med(ws0, "wall_sync_ms")
            print(f"| {key} | {cfg.split(':n')[-1]} | {fe:.3g} | {we:.3g} | {ww:.3g} | {ww - fe:.3g} | {we - fe:.3g} | "
                  f"{req:.0f} | {pred:.3g} | {(ww - fe) / pred:.2f} |")


def p2e_cliff():
    """P2e (1): fine grid around the ADP runtime-fallback cliff. Cell = bits reported + path (E emulated,
    F fell back), one table, plus a check that all GPUs agree."""
    g = load("P2e_followup")
    cells = defaultdict(dict)  # (k, phi) -> {model: (bits, path)}
    for (m, name), rs in g.items():
        mode, cfg = split(name)
        if not cfg.startswith("cliff:"):
            continue
        r = rs[0]
        cells[(int(r["k"]), float(r["phi"]))][m] = (r["bits_last"], path_of(r))
    if not cells:
        return
    ks = sorted({k for k, _ in cells}); phis = sorted({p for _, p in cells})
    print("\n## P2e cliff fine grid (inner inputs, m=n=4096, EAGER + workspace)\n")
    print("Cell = mantissa bits reported by cuBLAS, then E (emulated) or F (ADP runtime fallback). One table because "
          "all GPUs agree unless listed below.\n")
    print("| k \\ φ | " + " | ".join(f"{p:g}" for p in phis) + " |")
    print("|---|" + "---|" * len(phis))
    disagree = []
    for k in ks:
        row = []
        for p in phis:
            v = cells.get((k, p), {})
            if len(set(v.values())) > 1:
                disagree.append((k, p, v))
            b, pa = next(iter(v.values())) if v else ("", "")
            row.append(f"{b} {'E' if pa == 'emu' else 'F' if pa == 'fallback' else pa}" if v else "")
        print(f"| {k} | " + " | ".join(row) + " |")
    gpus = sorted({m for v in cells.values() for m in v})
    print(f"\nGPUs: {', '.join(gpus)}. Disagreements: {len(disagree)}")
    for k, p, v in disagree:
        print(f"- k={k} φ={p:g}: {v}")
    emu_bits = [int(b) for v in cells.values() for b, pa in v.values() if pa == "emu"]
    fb_bits = [int(b) for v in cells.values() for b, pa in v.values() if pa == "fallback"]
    if emu_bits and fb_bits:
        print(f"\nLargest bit count emulated: {max(emu_bits)}; smallest that fell back: {min(fb_bits)}.")


def p2e_cap():
    """P2e (2): per-kernel time for the u11 4096^3 precision settings (CUPTI dump of one call). Moduli = length of
    the index pack in inverse_scaling_mrc_kernel's template (the CRT moduli the reconstruction handles)."""
    g = load("P2e_followup")
    cats = [("int8 GEMM", "biu"), ("modular prolog", "prolog_mod_dfma"),
            ("quantize/scale", r"prolog_quantize|sft_fast|get_minmax|reduce_nmoduli"),
            ("reconstruction", "inverse_scaling")]
    rows = []
    for (m, name), rs in sorted(g.items()):
        mode, cfg = split(name)
        if not cfg.startswith("cap:"):
            continue
        r = rs[0]
        kern = None
        for p in glob.glob(str(FINAL / "P2e_followup" / "*" / "kern" / f"{name}.r*.tsv")):
            gcsv = glob.glob(str(Path(p).parents[1] / "gpu*.csv"))
            if gcsv and model(next(csv.DictReader(open(gcsv[0])))["gpu"]) == m:
                kern = [l.rstrip("\n").split("\t", 1) for l in open(p) if "\t" in l]
        if not kern:
            continue
        tot = sum(int(t) for t, _ in kern)
        by = {c: sum(int(t) for t, n in kern if re.search(pat, n)) for c, pat in cats}
        inv = [n for _, n in kern if "inverse_scaling" in n]
        moduli = len(re.findall(r"Lj\d+E", inv[0])) if inv else 0
        nbiu = sum(1 for _, n in kern if "biu" in n)
        rows.append((m, cfg.split(":", 1)[1], r["bits_last"], r["workspace_mb"], moduli, nbiu, tot, by))
    if not rows:
        return
    print("\n## P2e dynamic cap vs fixed bits: per-kernel time of one call (u11, 4096³, EAGER + workspace)\n")
    print("Moduli = CRT moduli in the reconstruction kernel's template. Times in ms (one CUPTI-traced call).\n")
    print("| GPU | setting | bits | workspace MiB | moduli | int8 GEMM launches | total | " +
          " | ".join(c for c, _ in cats) + " | other |")
    print("|---|---|---|---|---|---|---|" + "---|" * (len(cats) + 1))
    for m, s, b, ws, mod, nb, tot, by in sorted(rows, key=lambda x: (x[0], natkey(x[1]))):
        oth = tot - sum(by.values())
        print(f"| {m} | {s} | {b} | {ws} | {mod} | {nb} | {tot / 1e6:.2f} | " +
              " | ".join(f"{by[c] / 1e6:.2f}" for c, _ in cats) + f" | {oth / 1e6:.2f} |")


def p2e_overflow():
    """P2e (3): class of the overflowing entry (workloads/overflow_class.py)."""
    rows = []
    for p in sorted(glob.glob(str(FINAL / "P2e_followup" / "overflow_*" / "overflow.csv"))):
        rows += list(csv.DictReader(open(p)))
    if not rows:
        return
    print("\n## P2e overflow: class of the one overflowing entry C[r, c] (CuPy, 2048³, keep-pool)\n")
    print("Exact value = 1e400 × (sign shown), i.e. +inf or -inf in double. ms/call shows the path (emulation is "
          "several times faster than native at this size).\n")
    print("| GPU | mode | ms/call | C[r,c] | class | exact sign | all other entries finite |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {model(r['gpu'])} | {r['mode']} | {r['ms_per_call']} | {r['C_rc']} | {r['class_rc']} | "
              f"{r['exact_sign']} | {r['other_entries_finite']} |")


def p4():
    rows = []
    for p in sorted(glob.glob(str(FINAL / "P4_rsvd" / "*" / "rsvd.csv"))):
        rows += [r for r in csv.DictReader(open(p))]
    g = defaultdict(list)
    for r in rows:
        g[(model(r["gpu"]), r["matrix"], r["mode"], int(r["l"]), int(r["q"]))].append(r)
    modes = ["native", "zc", "zc_eager", "fix", "fix_eager", "fp32", "tf32"]
    print("\n## P4 randomized SVD (16384×8192, Halko-Martinsson-Tropp + QR re-orth.)\n")
    print("Speedup vs native FP64 (median over 3 repeats of the 5-rep median). zc = zero code changes "
          "(env var only), fix = pool release threshold raised (--keep-pool), _eager = EAGER strategy.\n")
    for key, mname in MODELS:
        print(f"### {mname}: speedup\n")
        print("| matrix | l | q | native s | " + " | ".join(modes[1:]) + " |")
        print("|---|---|---|---|" + "---|" * (len(modes) - 1))
        for (m, mat, md, l, q) in sorted(k for k in g if k[0] == key and k[2] == "native"):
            nat = st.median(float(r["t_median_s"]) for r in g[(m, mat, "native", l, q)])
            cs = []
            for md2 in modes[1:]:
                rs = g.get((m, mat, md2, l, q))
                cs.append(f"{nat / st.median(float(r['t_median_s']) for r in rs):.2f}" if rs else "")
            print(f"| {mat} | {l} | {q} | {nat:.3f} | " + " | ".join(cs) + " |")
        print(f"\n### {mname}: accuracy (max over l,q and repeats)\n")
        print("`resid_fro_rel` = ‖A − QB‖_F/‖A‖_F of this run; `resid_opt_rel` = the optimal rank-l residual "
              "(a property of the input, identical in every mode, shown for reference).\n")
        print("| matrix | mode | sv_relerr_top10 | resid_fro_rel | resid_opt_rel (ref.) | orth_err_max |")
        print("|---|---|---|---|---|---|")
        for mat in sorted({k[1] for k in g if k[0] == key}):
            for md in modes:
                rs = [r for k, v in g.items() if k[0] == key and k[1] == mat and k[2] == md for r in v]
                if rs:
                    f = lambda c: f"{max(float(r[c]) for r in rs):.2e}"
                    print(f"| {mat} | {md} | {f('sv_relerr_top10')} | {f('resid_fro_rel')} | {f('resid_opt_rel')} | "
                          f"{f('orth_err_max')} |")
        print()


def p4_paths():
    """nsys path check (one profiled pass over all l, q per matrix and mode). Every GEMM that enters emulation
    launches one cublasLt_imma_emu::...d884gemm kernel; it exits in ~2 us when emulation proceeds and runs the whole
    product natively when ADP falls back. So: entered = its instances, fell back = instances that ran long."""
    print("\n## P4 nsys path check: GEMM calls that entered emulation, and how many fell back at runtime\n")
    print("One profiled pass over all (l, q) per matrix and mode. Entered = instances of the "
          "`cublasLt_imma_emu::…d884gemm` kernel (launched once per emulated call; exits in ~2 µs unless ADP falls "
          "back). Fell back = instances of a variant whose minimum duration is > 100 µs (`mixed` if a variant has "
          "both short and long instances). Emulation share = GPU time in emulation kernels.\n")
    print("| GPU | matrix | mode | entered emulation | fell back | fallback share of GPU time | emulation share of GPU time | "
          "total GPU s |")
    print("|---|---|---|---|---|---|---|---|")
    emu_pat = re.compile(r"oz2|biu|emu_batch|fused_imma|get_minmax|prolog_quantize|sft_fast|inverse_scaling|reduce_nmoduli")
    for p in sorted(glob.glob(str(FINAL / "P4_rsvd" / "*" / "nsys" / "*_cuda_gpu_kern_sum.csv"))):
        gpu = re.search(r"_gpu(\d+)_", p).group(1)
        key = {"2": "5060 Ti", "4": "3090", "5": "3090", "8": "4090"}[gpu]
        mat, mode = Path(p).name.replace("_cuda_gpu_kern_sum.csv", "").split(".")
        rs = list(csv.DictReader(open(p)))
        tot = sum(float(r["Total Time (ns)"]) for r in rs)
        imma = [r for r in rs if "cublasLt_imma_emu" in r["Name"]]
        entered = sum(int(r["Instances"]) for r in imma)
        fb, fbt, mixed = 0, 0.0, False
        for r in imma:
            lo, hi = float(r["Min (ns)"]), float(r["Max (ns)"])
            if lo > 1e5:
                fb += int(r["Instances"]); fbt += float(r["Total Time (ns)"])
            elif hi > 1e5:
                mixed = True; fbt += float(r["Total Time (ns)"])
        emu_t = sum(float(r["Total Time (ns)"]) for r in rs if emu_pat.search(r["Name"]) and "cublasLt_imma_emu" not in r["Name"])
        print(f"| {key} (gpu{gpu}) | {mat} | {mode} | {entered} | {fb}{' + mixed' if mixed else ''} | "
              f"{100 * fbt / tot:.0f} % | {100 * emu_t / tot:.0f} % | {tot / 1e9:.2f} |")


if __name__ == "__main__":
    print("# Headline tables (generated by analysis/headline.py; do not edit by hand)\n")
    print("Cells: speedup vs native FP64 and path. Plain number = event time with workspace set (`*_ws`) or pool "
          "kept (`*_kp`); `w` = zero-code wall clock (`*_ws0`) vs native wall clock. Native column = native time. "
          "Baseline = the native mode of the same config; `*` = no native mode in that grid, baseline is the "
          "bench's per-row PEDANTIC reference instead (can be slower than the native mode on small shapes); "
          "`path` = path-only row (no timing reference). Bits: emulated rows; `fb:N` = bits reported on rows "
          "that fell back at runtime.")
    pivot("P1a_square", "P1a square n×n×n")
    pivot("P1b_shapes", "P1b shapes")
    breakeven()
    pivot("P1c_ranges", "P1c exponent ranges (ozaki: φ-generated; inner: inner-product range; rowcol: row/col scaling)")
    p2a()
    pivot("P2b_version", "P2b cuBLAS version axis (ver13.1 = CUDA 13.1 = cuBLAS 13.2.2 … ver13.4 = cuBLAS 13.8.0.4)")
    pivot("P2c_bits", "P2c precision knob (EAGER, fixed mantissa bits / dynamic cap)")
    pivot("P2d_edges", "P2d edge cases (special values, transposes, beta, batched)",
          extra=("non-finite emu/native, class mismatch", p2d_extra))
    wall_check()
    default_below_native()
    p3b()
    zero_code_penalty()
    p2e_cliff()
    p2e_cap()
    p2e_overflow()
    p4()
    p4_paths()
