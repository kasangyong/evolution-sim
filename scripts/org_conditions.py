"""What lets an organism last? Reads the organism records a gate saved (runs/genomes/<name>/*.npz,
`org_log`: birth conditions and fate of every organism that ended) and compares organisms that
lived long (≥ LONG ticks) with those that died young (< SHORT): where they were born, the food and
light there, how crowded the cell was with other cells, what they fed on, how big they grew, and
the rank correlation of lifespan with each condition. Pure numpy.

    PYTHONPATH=. uv run python scripts/org_conditions.py s4_gate_cond
"""
import sys
from pathlib import Path

import numpy as np

from evo.life.organisms import LOG_FIELDS

LONG, SHORT = 1024, 256
KINDS = ("chemosynthesis", "anoxygenic photo", "oxygenic photo", "predation")
CAUSES = ("unknown", "eaten", "starved", "background")
root = Path(__file__).resolve().parents[1]
col = {f: k for k, f in enumerate(LOG_FIELDS)}


def rank(x):
    r = np.empty(len(x))
    r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return r


for name in sys.argv[1:]:
    runs = []
    for f in sorted((root / "runs" / "genomes" / name).glob("*.npz")):
        d = np.load(f)
        if "org_log" in d and len(d["org_log"]):
            runs.append((f.stem, d["org_log"]))
    print(f"== {name}: {sum(len(r) for _, r in runs):,} organisms in {len(runs)} runs")
    for stem, r in runs:
        lived = r[:, col["lived"]]
        print(f"   {stem:<14} {len(r):>8,} organisms, lived median {np.median(lived):>6.0f} "
              f"max {lived.max():>7.0f}, long-lived {int((lived >= LONG).sum()):>6,}")
    if not runs:
        continue
    rows = np.concatenate([r for _, r in runs])
    lived = rows[:, col["lived"]]
    long_, short = lived >= LONG, lived < SHORT
    print(f"long-lived (≥ {LONG}): {int(long_.sum()):,}   died young (< {SHORT}): {int(short.sum()):,}")
    print(f"{'condition at birth':<22} {'long-lived':>12} {'died young':>12} {'rank corr. with lifespan':>26}")
    for f in ("depth_above_floor", "h2", "light", "cells_here", "cells_around", "max_segments",
              "neurons"):
        x = rows[:, col[f]]
        rho = float(np.corrcoef(rank(x), rank(lived))[0, 1]) if x.std() > 0 else float("nan")
        med = lambda m: float(np.median(x[m])) if m.any() else float("nan")  # noqa: B023
        print(f"{f:<22} {med(long_):>12.3g} {med(short):>12.3g} {rho:>26.2f}")
    for label, idx, names in (("fed on", col["kind"], KINDS), ("cause of end", col["cause"], CAUSES)):
        print(label + ":")
        for k, n in enumerate(names):
            m = rows[:, idx] == k
            if m.any():
                print(f"   {n:<18} {int(m.sum()):>8,}  long-lived share {long_[m].mean():.3f}  "
                      f"median life {np.median(lived[m]):.0f}")
