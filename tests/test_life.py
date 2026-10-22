import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from evo.config import H
from evo.life.agents import ALIVE, DEAD
from evo.life.genome import GENE_RANGE
from evo.sim import Sim
from evo.world.fields import BUR_ORG


def test_life_persists_and_budget_closes(small_cfg):
    sim = Sim(small_cfg)
    sim.step(2000)
    assert sim.life.count() > small_cfg.life.n_founders
    err = sim.budget_error()
    assert err.worst() < 1e-9, err


def test_free_list_consistent(small_cfg):
    sim = Sim(small_cfg)
    sim.step(2000)
    a = sim.life
    state = a.state.to_numpy()
    top = a.top[None]
    free = a.free.to_numpy()[:top]
    assert (state == ALIVE).sum() == a.cap - top
    assert len(np.unique(free)) == top
    assert (state[free] == DEAD).all()
    uids = a.uid.to_numpy()[state == ALIVE]
    assert len(np.unique(uids)) == len(uids)


def test_genes_stay_in_range_and_vary(small_cfg):
    sim = Sim(small_cfg)
    sim.step(2000)
    g = sim.life.snapshot()["genome"]
    lo = np.array([r[0] for r in GENE_RANGE], np.float32)
    hi = np.array([r[1] for r in GENE_RANGE], np.float32)
    assert (g >= lo).all() and (g <= hi).all()
    assert (g.std(axis=0) > 0).all()  # mutation produced variation in every gene


def test_starvation_returns_bodies_as_detritus(small_cfg):
    small_cfg.world.vent_h = 0.0
    small_cfg.world.init[H] = 0.0
    sim = Sim(small_cfg)
    sim.step(3000)
    assert sim.life.count() == 0
    assert sim.world.acc[BUR_ORG] > 0  # dead cells became detritus and some was buried
    assert sim.budget_error().worst() < 1e-9


def test_single_cell_energy_follows_delta_g(small_cfg):
    from evo.config import CN_RATIO, C, N, X
    from evo.life.agents import G_CHEMO
    from evo.life.genome import ETA_ANAER, ETA_H2, ETA_METH, FOUNDER, T_OPT

    small_cfg.life.death_rate = 0.0
    sim = Sim(small_cfg)
    a, w, lc = sim.life, sim.world, small_cfg.life
    w.update_light_temp()
    x, y = 60, 10
    state = np.zeros(a.cap, np.int32)
    state[0] = ALIVE
    a.state.from_numpy(state)
    a.pos[0] = [x + 0.5, y + 0.5]
    for s, v in ((H, 5.0), (C, 5.0), (N, 5.0), (X, 0.0)):
        w.chem[s, x, y] = v
    genome = list(FOUNDER)
    genome[T_OPT] = w.temp[x, y]  # temperature factor = 1
    for g, v in enumerate(genome):
        a.genome[0, g] = v
    b0, e0 = 0.01, 0.005
    a.body[0], a.energy[0] = b0, e0

    a.metabolize()

    eta = genome[ETA_H2]
    r = lc.k_chemo * eta * b0 * 5.0 / (5.0 + lc.k_half)
    upkeep = lc.maintenance + lc.trait_upkeep * (eta + genome[ETA_METH] + genome[ETA_ANAER])
    e1 = e0 + G_CHEMO * eta * r - upkeep * (b0 + r)
    assert a.body[0] == pytest.approx(b0 + r, rel=1e-6)
    assert a.energy[0] == pytest.approx(e1, rel=1e-5)
    assert w.chem[H, x, y] == pytest.approx(5.0 - 2 * r, rel=1e-9)
    assert w.chem[C, x, y] == pytest.approx(5.0 - r, rel=1e-9)
    assert w.chem[N, x, y] == pytest.approx(5.0 - r / CN_RATIO, rel=1e-9)


@pytest.mark.skipif(shutil.which("nvidia-smi") is None, reason="needs an NVIDIA GPU")
def test_cuda_concurrency_consistency():
    # GPU atomics only race on CUDA; the in-process runtime here is single-thread CPU
    root = Path(__file__).resolve().parents[1]
    out = subprocess.run([sys.executable, str(root / "scripts" / "check_cuda_consistency.py")],
                         cwd=root, env={**os.environ, "PYTHONPATH": str(root)},
                         capture_output=True, text=True, timeout=600, check=False)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]


def test_full_pool_turns_slots_into_super_individuals(small_cfg):
    # with too few slots, division continues inside slots instead of failing (stage 3a, decision 7)
    from evo.life.genome import DIV_SIZE

    small_cfg.life.capacity = 512
    sim = Sim(small_cfg)
    sim.step(3000)
    a = sim.life
    snap = a.snapshot()
    assert a.count() >= 0.95 * a.cap  # slots freed by deaths refill next tick
    assert snap["ncell"].max() > 1 and a.cells() > a.cap
    # growth stops at 2 × division size: the cap is checked once per tick, so a fast grower can
    # overshoot by one tick of growth (up to ~0.3× for a shallow anoxygenic photosynthesizer,
    # stage 7c evaluation). The invariant: a slot at or over the cap fixes no more carbon
    alive = a.state.to_numpy() == ALIVE
    body, uid = a.body.to_numpy(), a.uid.to_numpy()
    over = alive & (body / np.maximum(a.ncell.to_numpy(), 1e-12)
                    >= 2.0 * a.genome.to_numpy()[:, DIV_SIZE])
    sim.step(1)
    same = over & (a.uid.to_numpy() == uid) & (a.state.to_numpy() == ALIVE)
    assert (a.body.to_numpy()[same] <= body[same] * (1 + 1e-12)).all()
    assert sim.budget_error().worst() < 1e-9


def test_bottom_row_cells_stay_attached(small_cfg):
    # benthic mats: a cell in the water row just above the floor does not drift (decision 7)
    sim = Sim(small_cfg)
    a, w = sim.life, sim.world
    x = 100
    y = int(w.floor_np[x]) - 1
    state = np.zeros(a.cap, np.int32)
    state[:2] = ALIVE
    a.state.from_numpy(state)
    a.pos[0] = [x + 0.5, y + 0.5]
    a.pos[1] = [x + 0.5, y - 10 + 0.5]
    a.move()
    assert tuple(a.pos[0]) == (x + 0.5, y + 0.5)
    assert tuple(a.pos[1]) != (x + 0.5, y - 10 + 0.5)
