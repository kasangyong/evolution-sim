"""Chemotaxis across runs (stage 7a evaluation, Major 1): for each saved gate genome set
(runs/genomes/<name>/*.npz, written by s4_gate.py), the cell-weighted net share of swimming
cells whose real brain tumbles when a sense falls, for the food sense and the other senses.
Within one run a spreading lineage tilts every sense alike; across runs drift scatters around 0
for every sense, while selection keeps the food sense positive. Prints one row per run and a sign
count. Pure numpy (no GPU).

    PYTHONPATH=. uv run python scripts/taxis_check.py s4_gate_7a_base [more names ...]
"""
import sys
from pathlib import Path

import numpy as np

from evo.life.genome import ETA_H2, ETA_PHOTO_AN, ETA_PHOTO_OX, SPEED
from evo.ui.web import N_SENSES, sense_bias

NAMES = ("H2", "sulfide", "O2", "light")
root = Path(__file__).resolve().parents[1]
for name in sys.argv[1:]:
    print(f"== {name}  (food = H2 for chemosynthesis, light for photosynthesis; weighted by cells)")
    print(f"{'run':<14} {'cells':>8} {'food':>6} " + " ".join(f"{n:>7}" for n in NAMES))
    food_runs, ctrl_runs = [], []
    for f in sorted((root / "runs" / "genomes" / name).glob("*.npz")):
        d = np.load(f)
        g, w = d["genome"], d["ncell"].astype(np.float64)
        if not len(g):
            print(f"{f.stem:<14} extinct")
            continue
        swims = g[:, SPEED] > 0.1
        bias = np.stack([sense_bias(g, s) * swims for s in range(N_SENSES)], axis=1)
        per_sense = (bias * w[:, None]).sum(axis=0) / w.sum()
        photo = g[:, ETA_PHOTO_OX] + g[:, ETA_PHOTO_AN] > g[:, ETA_H2]
        food_s = np.where(photo, 3, 0)
        food = float((bias[np.arange(len(g)), food_s] * w).sum() / w.sum())
        food_runs.append(food)
        main = int(np.bincount(food_s, weights=w).argmax())  # the run's main food sense
        ctrl_runs += [per_sense[s] for s in (1, 2)]  # sulfide and O2 feed nobody directly
        print(f"{f.stem:<14} {w.sum():>8.0f} {food:>+6.2f} "
              + " ".join(f"{v:>+7.2f}" for v in per_sense) + ("  (photo)" if main == 3 else ""))
    if food_runs:
        fr, cr = np.array(food_runs), np.array(ctrl_runs)
        print(f"food   : mean {fr.mean():+.3f}, positive in {(fr > 0).sum()}/{len(fr)}, "
              f"|x|>0.1 positive {(fr > 0.1).sum()} negative {(fr < -0.1).sum()}")
        print(f"control: mean {cr.mean():+.3f}, positive in {(cr > 0).sum()}/{len(cr)}, "
              f"|x|>0.1 positive {(cr > 0.1).sum()} negative {(cr < -0.1).sum()}  (sulfide, O2)")
