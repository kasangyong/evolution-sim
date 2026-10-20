"""World timeline: population split into lit / dark cells, mean body, climate and surface chemistry.
Usage: timeline.py <seed> <ticks> <every> [key=val ...]; appends to runs/timeline_seed<seed>.log."""
import sys
import time
from pathlib import Path

import taichi as ti

seed = int(sys.argv[1])
ti.init(arch=ti.cuda, random_seed=seed, log_level=ti.WARN)

from evo.config import ATM_C, ATM_M, ATM_O, Config, N, O, R
from evo.life.agents import ALIVE
from evo.sim import Sim
from evo.world.fields import BUR_ORG, CL_ICE, CL_TS

cfg = Config()
cfg.world.seed = seed
for kv in sys.argv[4:]:
    path, val = kv.split("=", 1)
    sect, key = path.split(".", 1)
    setattr(getattr(cfg, sect), key, type(getattr(getattr(cfg, sect), key))(val))
sim = Sim(cfg)
a, w = sim.life, sim.world
end, every = int(sys.argv[2]), int(sys.argv[3])
log = Path(__file__).resolve().parents[1] / "runs" / f"timeline_seed{seed}.log"
log.parent.mkdir(exist_ok=True)


def out(line: str):
    print(line, flush=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


out(f"# seed {seed} {' '.join(sys.argv[4:])}")
out("tick    slots  lit  dark  cells   body   Ts   ice  atmCO2  atmO2   atmCH4 | surf N  R     O      "
    "| buried  err    tick/s")
t = time.perf_counter()
while sim.tick < end:
    sim.step(every)
    ti.sync()
    tps = every / (time.perf_counter() - t)
    alive = a.state.to_numpy() == ALIVE
    p = a.pos.to_numpy()[alive].astype(int)
    lt = w.light.to_numpy()
    lit = int((lt[p[:, 0], p[:, 1]] > 0.01).sum()) if alive.any() else 0
    body = a.body.to_numpy()[alive].mean() if alive.any() else 0.0
    cells = a.ncell.to_numpy()[alive].sum()
    ch = w.chem.to_numpy()
    wet = w.water_np[:, :30]  # not light > 0: under full ice the surface is dark
    surf = {s: ch[s][:, :30][wet].mean() for s in (N, R, O)}
    out(f"{sim.tick:>6} {alive.sum():>7} {lit:>6} {alive.sum() - lit:>6} {cells:.3g} {body:.3f} "
        f"{w.clim[CL_TS]:5.1f} {w.clim[CL_ICE]:4.2f} {w.atm[ATM_C]:.3g} {w.atm[ATM_O]:.3g} "
        f"{w.atm[ATM_M]:.3g} | {surf[N]:.3f} {surf[R]:.3f} {surf[O]:.4f} | "
        f"{w.acc[BUR_ORG]:.3g} {sim.budget_error().worst():.0e} {tps:.0f}")
    if not alive.any():
        break
    t = time.perf_counter()
