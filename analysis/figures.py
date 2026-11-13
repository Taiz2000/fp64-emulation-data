#!/usr/bin/env python3
"""Paper figures from results/final, using the same loaders and baselines as analysis/headline.py / ledger.py.

Every figure is written as PDF (vector, fonts embedded) plus a PNG preview, and the plotted values are written to a
CSV next to it, so each point traces back to the ledger. Speedup = native-mode time / mode time of the same config.

  fig1_square    P1a: speedup vs n per GPU; workspace set (GPU time) vs zero code (wall clock)       ledger C1.1, C3
  fig2_shapes    P1b: EAGER speedup vs the varying dimension; whether the default heuristic emulates  ledger C4.1-C4.3
  fig3_cliff     P2e grid (bits, emulated / fell back) + P1c speedup vs bits at 4096^3             ledger C5.1-C5.5
  fig4_penalty   P1a: measured zero-code penalty vs requested workspace x P3b remap rate            ledger C3.3, C3.6

Encoding (all figures): colour + marker = GPU (categorical slots 1-3, validated all-pairs, light mode); filled marker =
emulated, hollow = ran native (heuristic declined or runtime fallback). Aqua is below 3:1 on white, so every series
is also direct-labelled or in a legend with its marker.

Usage: .venv/bin/python analysis/figures.py [--out paper/figs] [--font "TeX Gyre Termes"] [--full 7.14] [--col 3.49]
(install with `uv sync --only-group analysis`)
"""
import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from headline import WS_PAD_MIB, by_config, req_mib, load, med, natkey, p3b_slopes, path_of  # noqa: E402

GPUS = [  # key, label, colour (categorical slot), marker
    ("3090", "RTX 3090", "#2a78d6", "o"),
    ("4090", "RTX 4090", "#eb6834", "s"),
    ("5060 Ti", "RTX 5060 Ti", "#1baf7a", "^"),
]
FIXED = {"3090": "eager", "4090": "default", "5060 Ti": "default"}  # same as ledger.py
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(font):
    plt.rcParams.update({
        "font.family": "serif", "font.serif": [font, "Nimbus Roman", "DejaVu Serif"], "font.size": 8,
        "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "lines.linewidth": 1.1, "lines.markersize": 4, "legend.frameon": False, "pdf.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def fit_width(fig, target, **tight):
    """tight_layout, then resize until the tight-cropped output (what savefig writes) is exactly `target` inches
    wide, so LaTeX includes it at scale 1 and the 8 pt figure text matches the caption size."""
    pad = plt.rcParams["savefig.pad_inches"]
    for _ in range(6):
        fig.tight_layout(**tight)
        w = fig.get_tightbbox(fig.canvas.get_renderer()).width + 2 * pad  # inches
        if abs(w - target) < 0.003:
            return
        W, H = fig.get_size_inches()
        fig.set_size_inches(W + target - w, H)
    raise RuntimeError(f"could not fit figure to {target} in (got {w:.3f})")


def save(fig, out, name, rows, header):
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=200)
    plt.close(fig)
    with open(out / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def log2_axis(ax, ticks):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" if t < 1024 else f"{t // 1024:g}k" for t in ticks])
    ax.minorticks_off()


def speed_axis(ax, lo, hi):
    ax.set_yscale("log")
    ticks = [t for t in (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20) if lo <= t <= hi]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}×" for t in ticks])
    ax.set_ylim(lo, hi)
    ax.minorticks_off()
    ax.axhline(1, color=INK2, lw=0.7, zorder=1)


def marker_kw(color, marker, filled):
    return dict(marker=marker, color=color, markerfacecolor=color if filled else "white", markeredgecolor=color,
                markeredgewidth=0.9)


# ------------------------------------------------------------------------------------------------------------ fig 1
def fig1(out, width):
    g = load("P1a_square")
    fig, axes = plt.subplots(1, 3, figsize=(width, 2.0), sharey=True)
    rows = []
    for ax, (k, label, col, mk) in zip(axes, GPUS):
        pre = FIXED[k]
        cfgs = by_config(g, k)
        series = {"fixed": [], "zc": []}
        for cfg in sorted(cfgs, key=natkey):
            v = cfgs[cfg]
            n = int(cfg.split(":n")[-1])
            nat = v.get("native")
            if not nat:
                continue
            for s, mode, col_t, col_n in (("fixed", f"{pre}_ws", "median_ms", "median_ms"),
                                          ("zc", f"{pre}_ws0", "wall_sync_ms", "wall_sync_ms")):
                rs = v.get(mode)
                if not rs:
                    continue
                sp = med(nat, col_n) / med(rs, col_t)
                per = [med(nat, col_n) / float(r[col_t]) for r in rs if float(r[col_t]) > 0]  # each run vs native
                pa = path_of(rs[0])
                series[s].append((n, sp, pa, min(per), max(per)))
                rows.append([k, n, mode, "event" if s == "fixed" else "wall", f"{sp:.4f}", f"{min(per):.4f}",
                             f"{max(per):.4f}", len(per), pa])
        for s, ls in (("fixed", "-"), ("zc", "--")):
            pts = series[s]
            run = []  # draw the line only through consecutive emulated points (a native point is not a trend)
            for n, sp, pa, _, _ in pts + [(None, None, None, None, None)]:
                if pa == "emu":
                    run.append((n, sp))
                    continue
                if len(run) > 1:
                    ax.plot([p[0] for p in run], [p[1] for p in run], ls=ls, color=col, zorder=2)
                run = []
            for n, sp, pa, lo, hi in pts:
                ax.vlines(n, lo, hi, color=col, lw=0.8, zorder=2.5)  # range over the runs, with end caps
                ax.hlines([lo, hi], n / 2 ** 0.09, n * 2 ** 0.09, color=col, lw=0.8, zorder=2.5)
                ax.plot([n], [sp], ls="none", zorder=3, **marker_kw(col, mk, pa == "emu"))
        nat_pts = sorted({n for s in series.values() for n, _, pa, _, _ in s if pa != "emu"})
        for n in nat_pts:
            if n >= 12288:
                ax.annotate(f"n = {n}: workspace too\nlarge for 8 GB, ran native", xy=(n, 1), xytext=(n / 1.6, 0.25),
                            fontsize=6.5, color=INK2, ha="right", va="center",
                            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5))
            else:
                ax.annotate("default\ndeclines", xy=(n, 1), xytext=(n * 1.15, 0.45), fontsize=6.5, color=INK2,
                            ha="left", va="center", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5))
        strat = "EAGER" if pre == "eager" else "default strategy"
        ax.set_title(f"{label} ({strat})", color=INK)
        log2_axis(ax, [512, 1024, 2048, 4096, 8192, 16384])
        ax.set_xlabel("n (m = n = k)")
        speed_axis(ax, 0.05, 25)
        if k == "3090":
            ax.text(0.03, 0.97, "default strategy\nnever emulates", transform=ax.transAxes, va="top", fontsize=6.5,
                    color=INK2)
    axes[0].set_ylabel("speedup over native FP64")
    handles = [Line2D([], [], color=INK2, ls="-", label="workspace set (GPU time)"),
               Line2D([], [], color=INK2, ls="--", label="zero code (wall clock)"),
               Line2D([], [], color=INK2, ls="none", marker="$\\mathsf{I}$", markersize=6, markeredgewidth=0.3,
                      label="range over runs"),
               Line2D([], [], ls="none", **marker_kw(INK2, "o", True), label="emulated"),
               Line2D([], [], ls="none", **marker_kw(INK2, "o", False), label="ran native")]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 1.0))
    fit_width(fig, width)
    save(fig, out, "fig1_square", rows, ["gpu", "n", "mode", "time", "speedup", "run_min", "run_max", "runs", "path"])


# ------------------------------------------------------------------------------------------------------------ fig 2
def fig2(out, width):
    g = load("P1b_shapes")
    fams = [("skinny", "n", "skinny: n varies (m = k = 8192)"), ("panel", "k", "panel: k varies (m = n = 8192)"),
            ("shortwide", "m", "short-wide: m varies (n = k = 8192)")]
    fig, grid = plt.subplots(2, 3, figsize=(width, 2.55), sharex="col",
                             gridspec_kw={"height_ratios": [4, 1.05], "hspace": 0.08})
    axes, strips = grid[0], grid[1]
    rows = []
    for col_i, (ax, strip, (fam, dim, title)) in enumerate(zip(axes, strips, fams)):
        allx = set()
        for k, label, col, mk in GPUS:
            pts = []
            for cfg, v in by_config(g, k).items():
                if f":{fam}:" not in cfg or "native" not in v or "eager_ws" not in v:
                    continue
                d = int(re.search(rf"{dim}(\d+)", cfg.split(":")[-1]).group(1))
                sp = med(v["native"], "median_ms") / med(v["eager_ws"], "median_ms")
                de = path_of(v["default_ws"][0]) if "default_ws" in v else "-"
                pe = path_of(v["eager_ws"][0])
                pts.append((d, sp, de, pe))
                rows.append([k, fam, d, f"{sp:.4f}", pe, de])
            pts.sort()
            allx |= {p[0] for p in pts}
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, zorder=2)
            for d, sp, de, pe in pts:  # EAGER path: filled = emulated
                ax.plot([d], [sp], ls="none", zorder=3, markersize=3.2, **marker_kw(col, mk, pe == "emu"))
            y = [g_[0] for g_ in GPUS].index(k)
            for d, sp, de, pe in pts:  # decision strip: filled = the default strategy emulates this shape
                strip.plot([d], [2 - y], ls="none", markersize=3.4, **marker_kw(col, mk, de == "emu"))
        ax.set_title(title, color=INK)
        speed_axis(ax, 0.04, 25)
        ticks = [t for t in (8, 32, 128, 512, 2048, 8192) if min(allx) <= t <= max(allx)]
        log2_axis(strip, ticks)
        strip.set_xlabel(f"{dim}")
        strip.set_ylim(-0.6, 2.6); strip.set_yticks([2, 1, 0]); strip.grid(axis="y", visible=False)
        strip.set_yticklabels([l.replace("RTX ", "") for _, l, _, _ in GPUS] if col_i == 0 else [])
        strip.tick_params(axis="y", length=0)
        if col_i:
            ax.tick_params(labelleft=False)
    axes[0].set_ylabel("EAGER speedup over native\n(workspace set, GPU time)")
    strips[0].set_ylabel("default\nemulates?", rotation=0, ha="right", va="center", labelpad=22)
    handles = [Line2D([], [], **marker_kw(c, m, True), label=l) for _, l, c, m in GPUS]
    handles += [Line2D([], [], ls="none", **marker_kw(INK2, "o", True), label="strip: default heuristic emulates"),
                Line2D([], [], ls="none", **marker_kw(INK2, "o", False), label="strip: default heuristic declines")]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.97))
    fit_width(fig, width, h_pad=0.3)
    save(fig, out, "fig2_shapes", rows, ["gpu", "family", "dim", "eager_speedup", "eager_path", "default_path"])


# ------------------------------------------------------------------------------------------------------------ fig 3
def fig3(out, width):
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(width, 2.3), gridspec_kw={"width_ratios": [1.15, 1]})
    rows = []
    # (a) P2e cliff grid: one cell per (k, phi); all GPUs must agree (checked, else the cell is marked)
    g = load("P2e_followup")
    cells = defaultdict(dict)
    for (m, name), rs in g.items():
        if ":cliff:" not in f":{name}":
            continue
        r = rs[0]
        cells[(int(r["k"]), float(r["phi"]))][m] = (int(r["bits_last"]), path_of(r))
    ks = sorted({k for k, _ in cells}); phis = sorted({p for _, p in cells})
    for i, k in enumerate(ks):
        for j, p in enumerate(phis):
            v = cells.get((k, p), {})
            if not v:
                continue
            agree = len(set(v.values())) == 1
            bits, pa = next(iter(v.values()))
            fb = pa == "fallback"
            axa.add_patch(plt.Rectangle((j, i), 1, 1, facecolor="#d9d8d3" if fb else "white",
                                        edgecolor="#9a9994", lw=0.4, hatch="////" if fb else None))
            axa.text(j + 0.5, i + 0.5, f"{bits}" + ("" if agree else "*"), ha="center", va="center", fontsize=6.5,
                     color=INK, fontweight="bold" if bits == 107 else "normal")
            rows.append(["a", k, p, bits, pa, "/".join(sorted(v)), agree])
    axa.set_xlim(0, len(phis)); axa.set_ylim(0, len(ks))
    axa.set_xticks([j + 0.5 for j in range(len(phis))]); axa.set_xticklabels([f"{p:g}" for p in phis])
    axa.set_yticks([i + 0.5 for i in range(len(ks))])
    axa.set_yticklabels([f"{k}" if k < 1024 else f"{k // 1024}k" for k in ks])
    axa.tick_params(length=0); axa.grid(False)
    for s in ("left", "bottom"):
        axa.spines[s].set_visible(False)
    axa.set_xlabel("φ (exponent range inside the dot products)")
    axa.set_ylabel("k (m = n = 4096)")
    axa.set_title("(a) bits needed; shaded = runtime fallback to native", color=INK)
    axa.legend(handles=[Patch(facecolor="white", edgecolor="#9a9994", label="emulated"),
                        Patch(facecolor="#d9d8d3", edgecolor="#9a9994", hatch="////", label="fell back")],
               loc="upper left", bbox_to_anchor=(1.0, 1.0), handlelength=1.2)
    # (b) P1c at 4096^3: speedup vs bits needed, all input families, per GPU
    g = load("P1c_ranges")
    for k, label, col, mk in GPUS:
        pts = []
        for cfg, v in by_config(g, k).items():
            if not cfg.endswith(":k4096") or "native" not in v or "eager_ws" not in v:
                continue
            r = v["eager_ws"][0]
            sp = med(v["native"], "median_ms") / med(v["eager_ws"], "median_ms")
            pts.append((int(r["bits_last"]), sp, path_of(r), cfg))
        pts.sort()
        for b, sp, pa, cfg in pts:
            axb.plot([b], [sp], ls="none", zorder=3, **marker_kw(col, mk, pa == "emu"))
            rows.append(["b", k, cfg, b, f"{sp:.4f}", pa])
    axb.axvspan(106.5, 107.5, color="#d9d8d3", lw=0, zorder=0)
    axb.text(107.9, 15, "107 bits:\neither way", fontsize=6.5, color=INK2, va="center")
    axb.set_xlabel("mantissa bits needed (reported by cuBLAS)")
    axb.set_ylabel("EAGER speedup over native")
    speed_axis(axb, 0.5, 20)
    axb.set_title("(b) speedup at 4096³, all input families", color=INK)
    axb.legend(handles=[Line2D([], [], ls="none", **marker_kw(c, m, True), label=l) for _, l, c, m in GPUS]
               + [Line2D([], [], ls="none", **marker_kw(INK2, "o", False), label="fell back")],
               loc="lower left")
    fit_width(fig, width)
    save(fig, out, "fig3_cliff", rows, ["panel", "gpu_or_k", "cfg_or_phi", "bits", "speedup_or_path", "path_or_gpus",
                                        "agree"])


# ------------------------------------------------------------------------------------------------------------ fig 4
def fig4(out, width):
    rates = p3b_slopes()
    g = load("P1a_square")
    fig, ax = plt.subplots(figsize=(width, 2.5))
    rows, allv = [], []
    for k, label, col, mk in GPUS:
        pre = FIXED[k]
        xs, ys = [], []
        for cfg, v in sorted(by_config(g, k).items(), key=lambda t: natkey(t[0])):
            ws, ws0 = v.get(f"{pre}_ws"), v.get(f"{pre}_ws0")
            if not ws or not ws0 or path_of(ws[0]) != "emu":
                continue
            req = req_mib(ws[0])
            pred = req / 1024 * rates[k][1]
            meas = med(ws0, "wall_sync_ms") - med(ws, "median_ms")
            xs.append(pred); ys.append(meas)
            rows.append([k, cfg.split(":n")[-1], f"{req:.0f}", f"{rates[k][1]:.2f}", f"{pred:.3f}", f"{meas:.3f}"])
        allv += xs + ys
        ax.plot(xs, ys, ls="none", label=f"{label} ({rates[k][1]:.1f} ms/GiB)", **marker_kw(col, mk, True))
    lo, hi = min(allv) / 1.5, max(allv) * 1.5
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.7, zorder=1)
    ax.text(hi / 1.6, hi / 1.25, "y = x", fontsize=6.5, color=INK2, ha="right")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("predicted: requested workspace × remap rate (ms)")
    ax.set_ylabel("measured zero-code penalty\n(wall − fixed GPU time, ms)")
    ax.legend(loc="upper left")
    fit_width(fig, width)
    save(fig, out, "fig4_penalty", rows, ["gpu", "n", "requested_mib", "rate_ms_per_gib", "predicted_ms", "measured_ms"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/figs")
    ap.add_argument("--font", default="TeX Gyre Termes")
    ap.add_argument("--full", type=float, default=516 / 72.27, help="double-column figure width, inches (IEEEtran \\textwidth 516 pt)")
    ap.add_argument("--col", type=float, default=252 / 72.27, help="single-column figure width, inches (IEEEtran \\columnwidth 252 pt)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    style(a.font)
    fig1(out, a.full); fig2(out, a.full); fig3(out, a.full); fig4(out, a.col)
    print(f"wrote {', '.join(sorted(p.name for p in out.glob('fig*.pdf')))} to {out}")
