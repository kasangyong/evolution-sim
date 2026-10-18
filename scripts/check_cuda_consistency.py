"""Exercise GPU atomics (resource takes, free-list push/pop) under real contention and assert
pool/ledger invariants every 1000 ticks. Exit code 1 on any violation."""
import sys

import numpy as np
import taichi as ti

ti.init(arch=ti.cuda, random_seed=7, log_level=ti.WARN)

from evo.config import Config
from evo.life.agents import ALIVE, DEAD
from evo.sim import Sim

cfg = Config()
cfg.world.vent_h = 1.2  # ~50k cells: heavy contention on vent cells and the free-list
sim = Sim(cfg)
a = sim.life
for _ in range(6):
    sim.step(1000)
    state = a.state.to_numpy()
    top = a.top[None]
    free = a.free.to_numpy()[:top]
    uids = a.uid.to_numpy()[state == ALIVE]
    checks = {
        "alive == cap - top": (state == ALIVE).sum() == a.cap - top,
        "free slots unique": len(np.unique(free)) == top,
        "free slots dead": (state[free] == DEAD).all(),
        "uids unique": len(np.unique(uids)) == len(uids),
        "dead bodies empty": (a.body.to_numpy()[state == DEAD] == 0).all(),
        "chem >= -1e-15": sim.world.chem.to_numpy().min() >= -1e-15,
        "budget < 1e-9": sim.budget_error().worst() < 1e-9,
    }
    print(f"tick {sim.tick} pop {a.count()} " + " ".join(f"{k}={v}" for k, v in checks.items()))
    if not all(checks.values()):
        sys.exit(1)
