"""Check the organism life cycle in a full world (stage 7b; checking only, never part of an observed
run): hand-made organisms are placed at the vents, then the world runs on its own. Prints, every
`every` ticks: organisms, longest body, organisms born from released cells (generations), how far
they swam, and the element budget.

    PYTHONPATH=. uv run python scripts/organism_check.py <seed> [ticks] [every] [plan] [sect.key=val ...]

plan "chemo" (default): chemosynthetic colonies with a body. plan "predator": swimming hunters
(the head's brain holds the hunt output on, methanogenesis burns the eaten body for energy).
"""
import sys

import numpy as np
import taichi as ti

seed = int(sys.argv[1])
ticks = int(sys.argv[2]) if len(sys.argv) > 2 else 50_000
every = int(sys.argv[3]) if len(sys.argv) > 3 else 5_000
plan_name = sys.argv[4] if len(sys.argv) > 4 else "chemo"
overrides = [kv.split("=", 1) for kv in sys.argv[5:]]
ti.init(arch=ti.cuda, random_seed=seed, log_level=ti.WARN)

from evo.config import Config
from evo.life.agents import ALIVE
from evo.life.genome import (
    ADHESION,
    BRAIN,
    CPG_AMP,
    CPG_FREQ,
    CPG_LAG,
    DIFF,
    ETA_H2,
    ETA_METH,
    ETA_PRED,
    N_HID,
    N_IN,
    N_SEG,
    PREY_RATIO,
    SEG_CELLS,
    TISSUE,
)
from evo.sim import Sim

cfg = Config()
cfg.world.seed = seed
for path, val in overrides:  # e.g. world.vent_h=8.0: a richer vent, so food is not what limits
    sect, key = path.split(".", 1)
    setattr(getattr(cfg, sect), key, type(getattr(getattr(cfg, sect), key))(val))
sim = Sim(cfg)
sim.step(5_000)  # let the vents and the founders settle first
a, o = sim.life, sim.life.orgs
alive = np.nonzero(a.state.to_numpy() == ALIVE)[0]
g = a.genome.to_numpy()
best = alive[np.argsort(-g[alive, ETA_H2])][:40]  # the best-adapted chemosynthesizers
# an animal plan: metabolic head, muscle in the middle, germ tissue at the tail
# (tissue logits vs metabolism 0 at s = 0 head .. 1 tail: muscle −2 → +2, germ −8 → +3)
plan = {ADHESION: 0.9, DIFF: 0.9, N_SEG: 6.0, SEG_CELLS: 8.0, TISSUE: -2.0, TISSUE + 1: 4.0,
        TISSUE + 2: -6.0, TISSUE + 3: 0.0, TISSUE + 4: -8.0, TISSUE + 5: 11.0, CPG_AMP: 0.6,
        CPG_FREQ: 0.05, CPG_LAG: 0.6}
if plan_name == "predator":
    # hidden unit 0 always on (bias 2), wired to the hunt output (ACT_HUNT = 2)
    plan |= {ETA_PRED: 0.9, PREY_RATIO: 1.0, ETA_METH: 0.9, BRAIN + N_IN * N_HID: 2.0,
             BRAIN + N_IN * N_HID + N_HID + N_HID * 2: 2.0}
for i in best:
    for k, v in plan.items():
        a.genome[i, k] = v
# they grow into colonies by dividing (adhesion keeps daughters) and develop at org_min_cells;
# organisms of later uids grew from cells that organisms released (the next generation)
print(f"seed {seed}: {len(best)} cells given the {plan_name} body plan at tick {sim.tick}")
while sim.tick < ticks:
    sim.step(every)
    t = o.table()
    n, travel, gen = t["n"], t["travel"], t["gen"]
    print(f"tick {sim.tick:>7} organisms {len(n):>5} longest {n.max() if len(n) else 0:>2} "
          f"mean seg {n.mean() if len(n) else 0:4.1f} generation max {gen.max() if len(gen) else 0:>3} "
          f"max swum {travel.max() if len(travel) else 0:6.1f} cells {a.cells():.3g} "
          f"err {sim.budget_error().worst():.1e}", flush=True)
