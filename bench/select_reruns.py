#!/usr/bin/env python3
"""Pick grid lines to rerun from a run directory's qa.txt.

Rerun: rep-median spread ("-> rerun"), within-rep wall_sync spread on non-*_ws0 lines, throttle flags.
Not rerun: *_ws0 within-rep spread (intrinsic zero-code jitter, the measured effect), "emulated error > native"
(expected for reduced-bit P2c settings), "native TFLOPS vs PoC" (shape/size effect, stable across versions).
Usage: select_reruns.py RUN_DIR  -> prints the selected grid.txt lines (stdout), reasons (stderr)
"""
import re
import sys
from pathlib import Path

run = Path(sys.argv[1])
want = {}
for ln in (run / "qa.txt").read_text().splitlines():
    m = re.match(r"\s*WARN (\S+): (.*)", ln)
    if not m:
        continue
    name, why = m.groups()
    mode = name.split(":")[1] if name.count(":") >= 1 else ""
    if "-> rerun" in why or "throttle" in why or ("wall_sync spread" in why and not mode.endswith("_ws0")):
        want.setdefault(name, []).append(why)

for ln in (run / "grid.txt").read_text().splitlines():
    if ln.strip() and not ln.lstrip().startswith("#") and ln.split("|", 1)[0] in want:
        print(ln)
        print(f"{ln.split('|', 1)[0]}: {'; '.join(want.pop(ln.split('|', 1)[0]))}", file=sys.stderr)
for name in want:
    print(f"NOT FOUND in grid: {name}", file=sys.stderr)
