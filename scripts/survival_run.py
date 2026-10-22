"""One S4 survival run: world seed and RNG seed = argv[1], ticks = argv[2]; then optional
sect.key=val config overrides and save=PATH (final genomes and cell counts, .npz). Prints a
single summary line."""
import sys
import time

import numpy as np
import taichi as ti

seed, ticks = int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 100_000
opts = dict(a.split("=", 1) for a in sys.argv[3:] if "=" in a)
ti.init(arch=ti.cuda, random_seed=seed, log_level=ti.WARN)

from evo.config import ATM_O, Config, O
from evo.life.agents import ALIVE
from evo.life.genome import ADHESION, ETA_PHOTO_AN, ETA_PHOTO_OX, ETA_PRED, GENE_NAMES, SPEED
from evo.sim import Sim

cfg = Config()
cfg.world.seed = seed
for path, val in opts.items():
    if path != "save":
        sect, key = path.split(".", 1)
        setattr(getattr(cfg, sect), key, type(getattr(getattr(cfg, sect), key))(val))
sim = Sim(cfg)
t, worst, min_pop, first_photo = time.perf_counter(), 0.0, None, None
first_org, max_orgs, max_seg, max_swum = None, 0, 0, 0.0  # stage 7b
max_neurons, neural_orgs = 0, 0  # stage 7c
max_learned, learning_orgs = 0.0, 0  # stage 7d
while sim.tick < ticks and sim.life.count() > 0:
    sim.step(1000)
    pop = sim.life.count()
    if min_pop is None or pop < min_pop:  # the bottleneck, with its cells (super-individuals)
        min_pop, min_cells = pop, sim.life.cells()
    worst = max(worst, sim.budget_error().worst())
    alive = sim.life.state.to_numpy() == ALIVE
    pos = sim.life.pos.to_numpy()[alive].astype(int)
    photic = int((sim.world.light.to_numpy()[pos[:, 0], pos[:, 1]] > 0.01).sum())
    if first_photo is None and photic > 1000:
        first_photo = sim.tick  # first checkpoint with > 1000 cells living in light
    orgs = sim.life.orgs.table()
    if len(orgs["id"]):
        first_org = sim.tick if first_org is None else first_org
        max_orgs, max_seg = max(max_orgs, len(orgs["id"])), max(max_seg, int(orgs["n"].max()))
        max_swum = max(max_swum, float(orgs["travel"].max()))
        max_neurons = max(max_neurons, int(orgs["neurons"].max()))
        neural_orgs = max(neural_orgs, int((orgs["neurons"] > 0).sum()))
        learned = sim.life.orgs.learned()
        max_learned = max(max_learned, float(learned.max()))
        learning_orgs = max(learning_orgs, int((learned > 0.1).sum()))
snap = sim.life.snapshot()
if "save" in opts:
    alive = sim.life.state.to_numpy() == ALIVE
    np.savez_compressed(opts["save"], genome=sim.life.genome.to_numpy()[alive],
                        ncell=sim.life.ncell.to_numpy()[alive])
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
      f"min_cells {min_cells:.3g} worst_err {worst:.1e} {time.perf_counter() - t:.0f}s | photic {photic} first_photo "
      f"{first_photo} dark {sim.life.count() - photic} cells {sim.life.cells():.3g} body {body:.3f} {photo} O2 {o2:.3g} | "
      f"{genes} | orgs {sim.life.orgs.count()} first_org {first_org} max_orgs {max_orgs} "
      f"max_seg {max_seg} max_swum {max_swum:.1f} max_neurons {max_neurons} neural_orgs {neural_orgs} "
      f"org_refused {sim.life.orgs.failed_transitions[None]} neurons_refused "
      f"{sim.life.orgs.failed_neurons[None]} max_learned {max_learned:.3f} learning_orgs {learning_orgs}",
      flush=True)
