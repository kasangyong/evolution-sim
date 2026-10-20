"""Is behaviour selected or drifting? Compares the brain's hunt output between cells that can
digest prey (η_pred > 0.1) and cells that cannot (η_pred < 0.02). Under pure drift both groups
hunt equally often; hunting only pays for predators (and costs nothing to the others, so a gap
means the brain co-evolved with the gene). Usage: behaviour_check.py <seed> <ticks>."""
import sys

import numpy as np
import taichi as ti

seed, ticks = int(sys.argv[1]), int(sys.argv[2])
ti.init(arch=ti.cuda, random_seed=seed, log_level=ti.WARN)

from evo.config import Config
from evo.life.agents import ACT_HOLD, ACT_HUNT, ACT_ON, ALIVE
from evo.life.genome import BRAIN, COL_R, ETA_PRED, N_BRAIN, SPEED
from evo.sim import Sim

cfg = Config()
cfg.world.seed = seed
sim = Sim(cfg)
a = sim.life
sim.step(ticks)
al = a.state.to_numpy() == ALIVE
g, act = a.genome.to_numpy()[al], a.act.to_numpy()[al]
pred, prey = g[:, ETA_PRED] > 0.1, g[:, ETA_PRED] < 0.02
hunt = act[:, ACT_HUNT] > ACT_ON
w = np.abs(g[:, BRAIN:BRAIN + N_BRAIN])
print(f"seed {seed} t {sim.tick} slots {al.sum()} predators {pred.sum()} non-predators {prey.sum()} | "
      f"hunting: predators {hunt[pred].mean() if pred.any() else float('nan'):.2f} "
      f"non-predators {hunt[prey].mean() if prey.any() else float('nan'):.2f} | "
      f"holding {(act[:, ACT_HOLD] > ACT_ON).mean():.2f} speed {g[:, SPEED].mean():.3f} | "
      f"mean|w| {w.mean():.2f} (range 8) col_r sd {g[:, COL_R].std():.2f} (range 1)", flush=True)
