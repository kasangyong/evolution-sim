"""One S4 survival run: world seed and RNG seed = argv[1]; prints a single summary line."""
import sys
import time

import numpy as np
import taichi as ti

seed, ticks = int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 100_000
ti.init(arch=ti.cuda, random_seed=seed, log_level=ti.WARN)

from evo.config import ATM_O, Config, O
from evo.life.agents import ALIVE
from evo.life.genome import ADHESION, ETA_PHOTO_AN, ETA_PHOTO_OX, ETA_PRED, GENE_NAMES, SPEED
from evo.sim import Sim

cfg = Config()
cfg.world.seed = seed
sim = Sim(cfg)
t, worst, min_pop, first_photo = time.perf_counter(), 0.0, None, None
while sim.tick < ticks and sim.life.count() > 0:
    sim.step(1000)
    pop = sim.life.count()
    min_pop = pop if min_pop is None else min(min_pop, pop)
    worst = max(worst, sim.budget_error().worst())
    alive = sim.life.state.to_numpy() == ALIVE
    pos = sim.life.pos.to_numpy()[alive].astype(int)
    photic = int((sim.world.light.to_numpy()[pos[:, 0], pos[:, 1]] > 0.01).sum())
    if first_photo is None and photic > 1000:
        first_photo = sim.tick  # first checkpoint with > 1000 cells living in light
snap = sim.life.snapshot()
means = snap["genome"].mean(axis=0) if len(snap["body"]) else np.full(len(GENE_NAMES), np.nan)
genes = " ".join(f"{g}={m:.3g}" for g, m in zip(GENE_NAMES[:6], means[:6]))
status = "ALIVE" if sim.life.count() > 0 else f"EXTINCT@{sim.tick}"
g = snap["genome"]
photo = (f"photo_an max {g[:, ETA_PHOTO_AN].max():.2f} photo_ox max {g[:, ETA_PHOTO_OX].max():.2f} "
         f"pred>0.1 {(g[:, ETA_PRED] > 0.1).mean():.2f} speed {g[:, SPEED].mean():.2f} "
         f"adh {g[:, ADHESION].mean():.2f} colony {sim.life.colonies()[alive].mean():.2f}"
         if len(g) else "photo -")
o2 = sim.world.chem.to_numpy()[O].sum() + sim.world.atm[ATM_O]
body = snap["body"].mean() if len(snap["body"]) else 0.0
print(f"seed {seed:>2} {status:<14} pop {sim.life.count():>6} min_pop {min_pop:>6} "
      f"worst_err {worst:.1e} {time.perf_counter() - t:.0f}s | photic {photic} first_photo "
      f"{first_photo} dark {sim.life.count() - photic} cells {sim.life.cells():.3g} body {body:.3f} {photo} O2 {o2:.3g} | "
      f"{genes}", flush=True)
