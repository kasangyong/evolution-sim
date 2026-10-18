import sys

import numpy as np
import taichi as ti

ti.init(arch=ti.cuda, log_level=ti.WARN)

from evo.config import Config, H
from evo.life.agents import ALIVE
from evo.life.genome import ETA_H2, T_OPT, T_WIDTH
from evo.sim import Sim

cfg = Config()
for kv in sys.argv[3:]:
    path, val = kv.split("=", 1)
    sect, key = path.split(".", 1)
    setattr(getattr(cfg, sect), key, type(getattr(getattr(cfg, sect), key))(val))
sim = Sim(cfg)
n, every = int(sys.argv[1]), int(sys.argv[2])
a, w = sim.life, sim.world
while sim.tick < n:
    sim.step(every)
    st = a.state.to_numpy() == ALIVE
    if not st.any():
        print(f"t={sim.tick} extinct")
        break
    pos = a.pos.to_numpy()[st].astype(int)
    g = a.genome.to_numpy()[st]
    hcell = w.chem.to_numpy()[H][pos[:, 0], pos[:, 1]]
    temp = w.temp.to_numpy()[pos[:, 0], pos[:, 1]]
    ft = np.exp(-((temp - g[:, T_OPT]) / g[:, T_WIDTH]) ** 2)
    body, en = a.body.to_numpy()[st], a.energy.to_numpy()[st]
    print(f"t={sim.tick:>6} pop {st.sum():>6} body {body.mean():.4f} E/b {np.mean(en / body):.3f} "
          f"H@cell {hcell.mean():.3f} (max {hcell.max():.2f}) T {temp.mean():.1f} fT {ft.mean():.2f} "
          f"etaH {g[:, ETA_H2].mean():.2f} err {sim.budget_error().worst():.1e}")
