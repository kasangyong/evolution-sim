import time

import taichi as ti

ti.init(arch=ti.cuda, log_level=ti.WARN)

from evo.sim import Sim

sim = Sim()
sim.step(100)
ti.sync()
for n in (2000, 2000):
    t = time.perf_counter()
    sim.step(n)
    ti.sync()
    dt = time.perf_counter() - t
    print(f"{n} ticks: {dt:.2f}s = {n / dt:,.0f} tick/s ({dt / n * 1e3:.2f} ms/tick)")
