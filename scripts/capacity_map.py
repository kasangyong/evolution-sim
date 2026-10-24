"""How many chemosynthetic cells can one grid cell feed? (organism-fate review.) In a lifeless world
with settled vent plumes, put a perfect H2 sink in one probe cell at a time (H2 set to 0 every
tick) and measure the inflow per tick once it is steady. Capacity = inflow / the most one cell
can use (2 H per unit of chemosynthesis × k_chemo × η × cell body). Compared with the
colony-to-organism threshold `org_min_cells`. Probe cells: each deep vent's own cell, cells above
it and along the floor.

    PYTHONPATH=. uv run python scripts/capacity_map.py [seed]
"""
import sys

import numpy as np
import taichi as ti

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
ti.init(arch=ti.cuda, random_seed=seed, log_level=ti.WARN)

from evo.config import Config, H
from evo.sim import Sim

cfg = Config()
cfg.world.seed = seed
sim = Sim(cfg)
sim.life.state.fill(0)  # no life: the plume as the vents and transport make it
sim.life.top[None] = sim.life.cap
w = sim.world
for _ in range(3000):
    w.emit_vents()
    w.transport()
    sim.chemistry.step()
base = w.chem.to_numpy()

eta, body = 0.93, 0.02  # a well-adapted chemosynthesizer at a typical division size
per_cell = 2.0 * cfg.life.k_chemo * eta * body
print(f"one cell uses at most {per_cell:.2e} H2 per tick; a vent emits {cfg.world.vent_h} per tick; "
      f"colony -> organism at {cfg.life.org_min_cells:.0f} cells")
print(f"{'vent':>10} {'probe':>14} {'H2 there':>10} {'inflow/tick':>12} {'cells fed':>10}")
for vx, vy in w.vents_np:
    if vy < w.d // 2:
        continue  # deep vents (the shallow one sits on the seamount top)
    floor = int(w.floor_np[vx]) - 1
    probes = [("vent cell", vx, vy)] + [(f"{k} up", vx, vy - k) for k in (1, 2, 3, 5, 8)] + [
        (f"{k} along", vx + k, int(w.floor_np[vx + k]) - 1) for k in (1, 2, 4)]
    for name, x, y in probes:
        if not (0 <= x < w.w and 0 <= y < floor + 1 and y < int(w.floor_np[x])):
            continue
        w.chem.from_numpy(base)
        removed = []
        for t in range(300):
            w.emit_vents()
            w.transport()
            sim.chemistry.step()
            h = float(w.chem[H, x, y])
            w.chem[H, x, y] = 0.0
            if t >= 100:
                removed.append(h)
        inflow = float(np.mean(removed))
        print(f"{f'({vx},{vy})':>10} {name:>14} {base[H, x, y]:>10.3g} {inflow:>12.3g} "
              f"{inflow / per_cell:>10.1f}")
