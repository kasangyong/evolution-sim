"""S2 on a real state: restore a time-machine snapshot (e.g. a full-pool bloom), then time
1,000 ticks without recording. Stage 7a evaluation: the 7a world no longer fills 50k slots from a
fresh start, so S2 is measured on the 131k-slot bloom instead.

    PYTHONPATH=. uv run python scripts/bench_snapshot.py runs/history/<run>/b0/snaps/<tick>.npz
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import taichi as ti

ti.init(arch=ti.cuda, log_level=ti.WARN)

from evo.io.snapshot import config_from_json, restore
from evo.sim import Sim

snap = Path(sys.argv[1])
meta = json.loads((snap.parents[2] / "meta.json").read_text())
sim = Sim(config_from_json(json.dumps(meta["config"])))
with np.load(snap) as z:
    restore(sim, {k: z[k] for k in z.files})
sim.step(200)  # warm up: compile kernels
ti.sync()
pop0, cells0 = sim.life.count(), sim.life.cells()
for rep in range(3):
    t = time.perf_counter()
    sim.step(1000)
    ti.sync()
    dt = time.perf_counter() - t
    print(f"tick {sim.tick:,} slots {pop0:,} -> {sim.life.count():,} cells {cells0:.3g}: "
          f"1000 ticks {dt:.2f}s = {1000 / dt:,.0f} tick/s", flush=True)
