"""Induced photosynthesis boom (check only, never used by real runs): after the vent community
settles, plant photosynthetic cells at the surface and watch whether deep vent lineages
survive the full pool. Usage: boom_check.py <seed> [reports] [ticks_per_report] [n_plant]."""
import sys

import numpy as np
import taichi as ti

seed = int(sys.argv[1])
ti.init(arch=ti.cuda, random_seed=seed, log_level=ti.WARN)

from evo.config import Config
from evo.life.agents import ALIVE
from evo.life.genome import (
    ETA_AER,
    ETA_ANAER,
    ETA_GENES,
    ETA_PHOTO_AN,
    ETA_PHOTO_OX,
    O2_TOL,
    T_OPT,
    T_WIDTH,
)
from evo.sim import Sim
from evo.world.ledger import totals

reports = int(sys.argv[2]) if len(sys.argv) > 2 else 10
every = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
n_plant = int(sys.argv[4]) if len(sys.argv) > 4 else 10000
cfg = Config()
cfg.world.seed = seed
sim = Sim(cfg)
a, w = sim.life, sim.world
sim.step(3000)
w.update_light_temp()

st = a.state.to_numpy()
plant = np.where(st != ALIVE)[0][:n_plant]
pos, g = a.pos.to_numpy(), a.genome.to_numpy()
body, en, nc, uid = a.body.to_numpy(), a.energy.to_numpy(), a.ncell.to_numpy(), a.uid.to_numpy()
rng = np.random.default_rng(0)
temp, floor = w.temp.to_numpy(), w.floor_np
template = g[st == ALIVE][0].copy()
for k, s in enumerate(plant):
    x = rng.uniform(200, 1000)
    while floor[int(x)] <= 10:
        x = rng.uniform(200, 1000)
    pos[s] = (x, rng.uniform(0.5, 8.0))
    g[s] = template
    g[s, list(ETA_GENES)] = 0.0
    g[s, [ETA_PHOTO_OX, ETA_PHOTO_AN, ETA_AER, ETA_ANAER, O2_TOL]] = 0.9, 0.9, 0.5, 0.9, 1.0
    g[s, T_OPT], g[s, T_WIDTH] = temp[int(x), 3], 30.0
    body[s], en[s], nc[s], st[s], uid[s] = 0.01, 0.005, 1.0, ALIVE, 10_000_000 + k
dead = np.where(st != ALIVE)[0]
free = np.zeros(a.cap, np.int32)
free[: len(dead)] = dead[::-1]
a.state.from_numpy(st)
a.pos.from_numpy(pos)
a.genome.from_numpy(g)
a.body.from_numpy(body)
a.energy.from_numpy(en)
a.ncell.from_numpy(nc)
a.uid.from_numpy(uid)
a.free.from_numpy(free)
a.top[None] = len(dead)
# planted bodies are new matter: restart the ledger baseline at this state
sim.t0 = totals(w, a)
w.acc.fill(0.0)

print("tick   slots   cells    deep_slots deep_cells  photo_slots  free  err", flush=True)
for _ in range(reports):
    sim.step(every)
    al = a.state.to_numpy() == ALIVE
    p, n, gg = a.pos.to_numpy()[al], a.ncell.to_numpy()[al], a.genome.to_numpy()[al]
    deep = p[:, 1] > 200
    photo = (gg[:, ETA_PHOTO_OX] + gg[:, ETA_PHOTO_AN]) > 0.3
    print(f"{sim.tick:>6} {al.sum():>7} {n.sum():>9.3g} {deep.sum():>10} {n[deep].sum():>10.3g} "
          f"{photo.sum():>11} {a.top[None]:>6} {sim.budget_error().worst():.0e}", flush=True)
