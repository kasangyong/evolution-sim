"""One S4 survival run: world seed and RNG seed = argv[1]; prints a single summary line."""
import sys
import time

import numpy as np
import taichi as ti

seed, ticks = int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 100_000
ti.init(arch=ti.cuda, random_seed=seed, log_level=ti.WARN)

from evo.config import Config
from evo.life.genome import GENE_NAMES
from evo.sim import Sim

cfg = Config()
cfg.world.seed = seed
sim = Sim(cfg)
t, worst, min_pop = time.perf_counter(), 0.0, None
while sim.tick < ticks and sim.life.count() > 0:
    sim.step(1000)
    pop = sim.life.count()
    min_pop = pop if min_pop is None else min(min_pop, pop)
    worst = max(worst, sim.budget_error().worst())
snap = sim.life.snapshot()
means = snap["genome"].mean(axis=0) if len(snap["body"]) else np.full(len(GENE_NAMES), np.nan)
genes = " ".join(f"{g}={m:.3g}" for g, m in zip(GENE_NAMES[:6], means[:6]))
status = "ALIVE" if sim.life.count() > 0 else f"EXTINCT@{sim.tick}"
print(f"seed {seed:>2} {status:<14} pop {sim.life.count():>6} min_pop {min_pop:>6} "
      f"worst_err {worst:.1e} {time.perf_counter() - t:.0f}s | {genes}", flush=True)
