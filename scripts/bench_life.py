"""S2 benchmark: grow the population (optionally with a richer vent H2 supply), then time steps."""
import sys
import time

import taichi as ti

ti.init(arch=ti.cuda, log_level=ti.WARN, kernel_profiler="--profile" in sys.argv)

from evo.config import Config
from evo.sim import Sim

cfg = Config()
for kv in [a for a in sys.argv[2:] if "=" in a]:
    path, val = kv.split("=", 1)
    sect, key = path.split(".", 1)
    setattr(getattr(cfg, sect), key, type(getattr(getattr(cfg, sect), key))(val))
sim = Sim(cfg)
sim.step(int(sys.argv[1]))
ti.sync()
pop0 = sim.life.count()
if "--profile" in sys.argv:
    ti.profiler.clear_kernel_profiler_info()
t = time.perf_counter()
sim.step(1000)
ti.sync()
dt = time.perf_counter() - t
print(f"pop {pop0:,} -> {sim.life.count():,}: 1000 ticks {dt:.2f}s = {1000 / dt:,.0f} tick/s")
if "--profile" in sys.argv:
    ti.profiler.print_kernel_profiler_info()
