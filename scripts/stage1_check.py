"""Stage 1 gate: run the lifeless world on CUDA and report budget error / field health."""
import argparse
import time

import numpy as np
import taichi as ti

ti.init(arch=ti.cuda, log_level=ti.WARN)

from evo.config import NS, Config, D, M, O
from evo.sim import Sim
from evo.world.fields import BUR_CARB, BUR_ORG, CL_ICE, CL_TS

ap = argparse.ArgumentParser()
ap.add_argument("--ticks", type=int, default=100_000)
ap.add_argument("--seeded", action="store_true", help="seed O2/detritus/CH4 (same as tests)")
args = ap.parse_args()

cfg = Config()
if args.seeded:
    cfg.world.init.update({O: 0.3, D: 0.5, M: 0.1})
    cfg.climate.init_atm = (3.0e5, 1.0e4, 1.0e3, 0.0)
sim = Sim(cfg)
t = time.perf_counter()
worst = 0.0
step = args.ticks // 10
while sim.tick < args.ticks:
    sim.step(step)
    err = sim.budget_error()
    worst = max(worst, err.worst())
    chem = sim.world.chem.to_numpy()
    clim, acc = sim.world.clim.to_numpy(), sim.world.acc.to_numpy()
    print(f"tick {sim.tick:>8}  err C {err.carbon:.1e} N {err.nitrogen:.1e} e {err.electrons:.1e}"
          f"  min {chem.min():.1e} max {chem.reshape(NS, -1).max(axis=1).max():.3g}"
          f"  T {clim[CL_TS]:.1f} ice {clim[CL_ICE]:.2f}"
          f"  B_org {acc[BUR_ORG]:.3g} B_carb {acc[BUR_CARB]:.3g}  finite {np.isfinite(chem).all()}",
          flush=True)
dt = time.perf_counter() - t
print(f"{'seeded' if args.seeded else 'default'}: {args.ticks} ticks in {dt:.0f}s "
      f"({args.ticks / dt:,.0f} tick/s incl. checks); worst budget err {worst:.2e}")
