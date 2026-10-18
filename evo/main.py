import argparse
import subprocess
import time

import numpy as np
import taichi as ti

from evo.config import Config


def gpu_name() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             check=False, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description="Evolution sim")
    ap.add_argument("--arch", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--headless", type=int, default=0, help="run N ticks without UI and report")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    if args.arch == "cuda":
        ti.init(arch=ti.cuda, log_level=ti.WARN)
        name = gpu_name()
        if "NVIDIA" not in name:
            print(f"warning: expected an NVIDIA GPU, got {name!r}")
    else:
        ti.init(arch=ti.cpu, cpu_max_num_threads=1, log_level=ti.WARN)
        name = "CPU (debug)"
    print(f"backend: {args.arch}  device: {name}")

    from evo.sim import Sim

    cfg = Config()
    cfg.world.seed = args.seed
    sim = Sim(cfg)

    if args.headless:
        headless(sim, args.headless)
    else:
        from evo.ui.app import run

        run(sim, name)


def headless(sim, n: int):
    from evo.world.fields import CL_ICE, CL_TS

    chunk = 1000
    sim.step(10)
    ti.sync()
    t = time.perf_counter()
    worst = 0.0
    while sim.tick < n:
        sim.step(min(chunk, n - sim.tick))
        err = sim.budget_error()
        worst = max(worst, err.worst())
        if sim.tick % 10000 < chunk:
            chem = sim.world.chem.to_numpy()
            clim = sim.world.clim.to_numpy()
            print(f"tick {sim.tick:>7}  err C {err.carbon:.1e} N {err.nitrogen:.1e} "
                  f"e {err.electrons:.1e}  min {chem.min():.2e}  max {chem.max():.3g}  "
                  f"T {clim[CL_TS]:.1f} ice {clim[CL_ICE]:.2f}  finite {np.isfinite(chem).all()}  "
                  f"pop {sim.life.count()}")
    ti.sync()
    dt = time.perf_counter() - t
    print(f"{n - 10} ticks in {dt:.1f}s = {(n - 10) / dt:,.0f} tick/s; worst budget err {worst:.2e}")


if __name__ == "__main__":
    main()
