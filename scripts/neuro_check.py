"""Do neuron genes behave differently where they can act? (stage 7c, design decision 9.) For each
saved gate genome set (runs/genomes/<name>/*.npz), compare the neuron-count genes of slots that
can become organisms (differentiation and adhesion ≥ 0.5: their genes reach a body) with the other
slots, in which the same genes are neutral and only drift. Cell-weighted. Pure numpy.

    PYTHONPATH=. uv run python scripts/neuro_check.py s4_gate_7c
"""
import sys
from pathlib import Path

import numpy as np

from evo.life.genome import ADHESION, DIFF, N_TYPES, NEURO, NT_COUNT

COUNT = [NEURO + 8 * k + NT_COUNT for k in range(N_TYPES)]
root = Path(__file__).resolve().parents[1]
for name in sys.argv[1:]:
    print(f"== {name}: neurons per type (floor of count genes, summed over 8 types), cell-weighted")
    print(f"{'run':<14} {'able cells':>10} {'able':>6} {'neutral':>8}")
    able_runs, neutral_runs = [], []
    for f in sorted((root / "runs" / "genomes" / name).glob("*.npz")):
        d = np.load(f)
        g, w = d["genome"], d["ncell"].astype(np.float64)
        if not len(g):
            continue
        neurons = np.floor(g[:, COUNT]).sum(axis=1)
        able = (g[:, DIFF] >= 0.5) & (g[:, ADHESION] >= 0.5)
        a = float(np.average(neurons[able], weights=w[able])) if able.any() else float("nan")
        n = float(np.average(neurons[~able], weights=w[~able])) if (~able).any() else float("nan")
        if able.any():
            able_runs.append(a)
        neutral_runs.append(n)
        print(f"{f.stem:<14} {w[able].sum():>10.0f} {a:>6.2f} {n:>8.2f}")
    print(f"mean neurons: able {np.nanmean(able_runs):.2f} in {len(able_runs)} runs, "
          f"neutral {np.mean(neutral_runs):.2f} in {len(neutral_runs)} runs")
