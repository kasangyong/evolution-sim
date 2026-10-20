import numpy as np
import taichi as ti

from evo.config import ATM_C, ATM_O, D, O
from evo.sim import Sim
from evo.world.fields import BUR_CARB, BUR_ORG, CL_ICE, CL_TS


def test_budget_closes(small_cfg):
    sim = Sim(small_cfg)
    sim.step(3000)
    err = sim.budget_error()
    assert err.worst() < 1e-9, err  # f64 state; f32 drift would show up here


def test_budget_closes_with_all_reactions_active(seeded_cfg):
    # abiotic paths only: since stage 3a, photosynthesis can make O2 and break the O2-falls check
    seeded_cfg.life.n_founders = 0
    sim = Sim(seeded_cfg)
    w0 = sim.world.chem.to_numpy()
    sim.step(3000)
    w1 = sim.world.chem.to_numpy()
    # the paths under test really ran
    assert w1[O].sum() < w0[O].sum() and w1[D].sum() < w0[D].sum()
    assert sim.world.acc[BUR_ORG] > 0 and sim.world.acc[BUR_CARB] > 0
    assert sim.world.atm[ATM_O] < seeded_cfg.climate.init_atm[ATM_O]
    err = sim.budget_error()
    assert err.worst() < 1e-6, err


def test_ledger_detects_unbooked_loss(seeded_cfg):
    sim = Sim(seeded_cfg)
    sim.step(100)
    chem = sim.world.chem.to_numpy()
    chem[D] = 0.0  # remove all detritus without booking a sink
    sim.world.chem.from_numpy(chem)
    err = sim.budget_error()
    assert err.carbon > 1e-3 and err.nitrogen > 1e-3 and err.electrons > 1e-3


def test_ice_blocks_gas_exchange(seeded_cfg):
    sim = Sim(seeded_cfg)
    w = sim.world
    w.clim[CL_ICE] = 1.0
    surf0, atm0 = w.chem.to_numpy()[:, :, 0].copy(), w.atm.to_numpy().copy()
    sim.climate.gas_exchange()
    assert np.array_equal(w.chem.to_numpy()[:, :, 0], surf0)
    assert np.array_equal(w.atm.to_numpy(), atm0)


def test_fields_finite_nonnegative(small_cfg):
    sim = Sim(small_cfg)
    sim.step(3000)
    chem = sim.world.chem.to_numpy()
    assert np.isfinite(chem).all()
    assert chem.min() > -1e-5


def test_cpu_deterministic(small_cfg):
    # Taichi's RNG is global per runtime, so each run gets a freshly seeded runtime
    runs = []
    for _ in range(2):
        ti.init(arch=ti.cpu, cpu_max_num_threads=1, random_seed=0, log_level=ti.WARN)
        sim = Sim(small_cfg)
        sim.step(500)
        runs.append((sim.world.chem.to_numpy(), sim.life.pos.to_numpy()))
    assert np.array_equal(runs[0][0], runs[1][0])
    assert np.array_equal(runs[0][1], runs[1][1])


def test_ice_hysteresis(small_cfg):
    small_cfg.climate.weathering_on = False
    sim = Sim(small_cfg)
    w, cl = sim.world, sim.climate
    c_ref = small_cfg.climate.c_ref

    def settle(p):
        cl.force_co2(p * c_ref - w.atm[ATM_C])
        for _ in range(3000):
            cl.step()
        return w.clim[CL_ICE]

    sweep = [3.0, 2.0, 1.0, 0.5, 0.2, 0.05]
    down = {p: settle(p) for p in sweep}
    up = {p: settle(p) for p in reversed(sweep)}
    assert down[0.05] == 1.0
    assert down[1.0] == 0.0 and up[1.0] == 1.0  # same CO2, two climates
    assert up[3.0] == 0.0  # snowball is escapable


def test_volcanoes_free_a_snowball_without_forcing(small_cfg):
    # stage 4a: under full ice, gas exchange and weathering stop; only volcanic CO2 reaches the
    # atmosphere. Eruptions are 10× more frequent than default to keep the test short.
    small_cfg.climate.volcano_p = 0.1
    sim = Sim(small_cfg)
    w, cl = sim.world, sim.climate
    w.atm[ATM_C] = 0.05 * small_cfg.climate.c_ref
    w.clim[CL_TS], w.clim[CL_ICE] = -20.0, 1.0
    froze_until = None
    for t in range(80_000):
        cl.step()
        if w.clim[CL_ICE] < 1.0:
            froze_until = t
            break
    assert froze_until is not None and froze_until > 1000  # stayed frozen a while, then thawed
    for _ in range(5000):
        cl.step()
    assert w.clim[CL_ICE] == 0.0
