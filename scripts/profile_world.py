import taichi as ti

ti.init(arch=ti.cuda, kernel_profiler=True, log_level=ti.WARN)

from evo.sim import Sim

sim = Sim()
sim.step(50)
ti.sync()
ti.profiler.clear_kernel_profiler_info()
sim.step(500)
ti.sync()
ti.profiler.print_kernel_profiler_info()
