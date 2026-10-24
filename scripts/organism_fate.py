"""Why do organisms not last? (after stage 7d.) Runs a world with the default settings and reports,
every `every` ticks: organisms, and the stored-energy fill (energy / capacity) of organism
segments vs colonies (≥ 16 cells, not organisms) vs single cells, the organisms' metabolic tissue
share; at the end, how organisms ended (cause of the head's death), why bodies were cut, and the
lifespan histogram.

Organisms rarely arise on their own, so `plan=chemo|neural` gives the 40 best chemosynthesizers a
body plan at tick 5,000 (as organism_check.py does; checking only, never part of an observed run).

    PYTHONPATH=. uv run python scripts/organism_fate.py <seed> [ticks] [every] [plan=none|chemo|neural]
        [sect.key=val ...]
"""
import sys

import numpy as np
import taichi as ti

args = [a for a in sys.argv[1:] if "=" not in a]
opts = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
seed = int(args[0])
ticks = int(args[1]) if len(args) > 1 else 100_000
every = int(args[2]) if len(args) > 2 else 10_000
plan_name = opts.get("plan", "none")
ti.init(arch=ti.cuda, random_seed=seed, log_level=ti.WARN)

from evo.config import Config
from evo.life.agents import ALIVE
from evo.life.genome import (
    ADHESION,
    CONN,
    CPG_AMP,
    CPG_FREQ,
    CPG_LAG,
    DIFF,
    ETA_H2,
    N_SEG,
    N_TYPES,
    NEURO,
    SEG_CELLS,
    TISSUE,
)
from evo.sim import Sim

cfg = Config()
cfg.world.seed = seed
for path, val in opts.items():
    if "." in path:  # config override, e.g. life.seg_len=0.5
        sect, key = path.split(".", 1)
        setattr(getattr(cfg, sect), key, type(getattr(getattr(cfg, sect), key))(val))
sim = Sim(cfg)
a, o = sim.life, sim.life.orgs


def give_plan():
    alive = np.nonzero(a.state.to_numpy() == ALIVE)[0]
    g = a.genome.to_numpy()
    best = alive[np.argsort(-g[alive, ETA_H2])][:40]
    plan = {ADHESION: 0.9, DIFF: 0.9, N_SEG: 6.0, SEG_CELLS: 8.0, TISSUE: -2.0, TISSUE + 1: 4.0,
            TISSUE + 2: -6.0, TISSUE + 3: 0.0, TISSUE + 4: -8.0, TISSUE + 5: 11.0, CPG_AMP: 0.6,
            CPG_FREQ: 0.05, CPG_LAG: 0.6}
    if plan_name == "neural":
        def nt(k, p):
            return NEURO + 8 * k + p
        plan |= {CPG_AMP: 0.0, nt(0, 0): 1.0, nt(0, 2): 6.0, nt(0, 6): 0.7, nt(0, 7): 24.0,
                 nt(1, 0): 1.0, nt(1, 2): 6.0, nt(1, 7): 24.0, nt(2, 0): 1.0, nt(2, 1): 1.0,
                 nt(2, 2): 3.0, nt(2, 4): 6.5, nt(2, 5): 1.0, nt(2, 6): 1.0,
                 CONN + 0: 1.5, CONN + 1: -2.0, CONN + N_TYPES: 2.0, CONN + N_TYPES + 1: 1.5}
    for i in best:
        for k, v in plan.items():
            a.genome[i, k] = v
    print(f"tick {sim.tick}: {len(best)} cells given the {plan_name} body plan", flush=True)


def fill(mask, e, b):
    return float(np.mean(e[mask] / np.maximum(cfg.life.energy_cap * b[mask], 1e-12))) if mask.any() else float("nan")


print(f"seed {seed} plan {plan_name}")
print(f"{'tick':>8} {'orgs':>6} {'mean seg':>8}  fill: {'organism':>8} {'colony':>7} {'single':>7}  "
      f"organism metabolic share")
while sim.tick < ticks:
    step = every if not (plan_name != "none" and sim.tick < 5_000 <= sim.tick + every) else 5_000 - sim.tick
    sim.step(step)
    if plan_name != "none" and sim.tick == 5_000:
        give_plan()
    alive = a.state.to_numpy() == ALIVE
    org, nc = a.org.to_numpy(), a.ncell.to_numpy()
    e, b = a.energy.to_numpy(), a.body.to_numpy()
    seg = alive & (org >= 0)
    colony = alive & (org < 0) & (nc >= 16)
    single = alive & (nc <= 1.0)
    t = o.table()
    print(f"{sim.tick:>8} {len(t['id']):>6} {t['n'].mean() if len(t['n']) else 0:>8.2f}        "
          f"{fill(seg, e, b):>8.3f} {fill(colony, e, b):>7.3f} {fill(single, e, b):>7.3f}  "
          f"{a.metab_frac.to_numpy()[seg].mean() if seg.any() else float('nan'):.2f}", flush=True)
r = o.death_report()
print("organism ends by cause of the head's death:", r["ends"])
print("body cuts by cause:", r["cuts"])
hist = r["lifespan_log2"]
print("lifespan (ticks lived, log2 bins):", {f"<{2 ** (k + 1)}": c for k, c in enumerate(hist) if c})
