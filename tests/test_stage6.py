"""Stage 6: the Sun brightens; adhesion keeps daughters in a colony, which predators see whole."""
import numpy as np
import pytest

from evo.life.agents import ALIVE
from evo.life.genome import ADHESION, DIV_SIZE, ETA_PRED, FOUNDER, PREY_RATIO
from evo.sim import Sim
from evo.world.fields import CL_SOLAR


def test_sun_brightens_at_the_set_rate_and_stops_at_today(small_cfg):
    small_cfg.climate.solar_rate = 0.01
    sim = Sim(small_cfg)
    s0 = sim.world.clim[CL_SOLAR]
    sim.climate.step()
    assert sim.world.clim[CL_SOLAR] == pytest.approx(s0 + 0.01)
    for _ in range(100):
        sim.climate.step()
    assert sim.world.clim[CL_SOLAR] == small_cfg.climate.solar_max


def _one_dividing_cell(cfg, adhesion):
    sim = Sim(cfg)
    a = sim.life
    state = np.zeros(a.cap, np.int32)
    state[0] = ALIVE
    a.state.from_numpy(state)
    a.top[None] = a.cap - 1
    free = np.zeros(a.cap, np.int32)
    free[: a.cap - 1] = np.arange(a.cap - 1, 0, -1)
    a.free.from_numpy(free)
    g = list(FOUNDER)
    g[ADHESION] = adhesion
    for k, v in enumerate(g):
        a.genome[0, k] = v
    a.pos[0] = [60.5, 10.5]
    a.body[0], a.energy[0], a.ncell[0] = 0.04, 0.01, 1.0
    a.want_div[0] = 1
    a.weigh_divisions()
    a.reproduce()
    return a


def test_full_adhesion_keeps_the_daughter_in_a_colony(small_cfg):
    a = _one_dividing_cell(small_cfg, 1.0)
    assert a.count() == 1  # no new slot
    assert a.ncell[0] == 2.0 and a.colonies()[0]
    assert a.body[0] == pytest.approx(0.04)


def test_no_adhesion_divides_as_before(small_cfg):
    a = _one_dividing_cell(small_cfg, 0.0)
    assert a.count() == 2 and not a.colonies().any()
    assert a.ncell.to_numpy()[a.state.to_numpy() == ALIVE].tolist() == [1.0, 1.0]


@pytest.mark.parametrize("adhesion,eaten", [(0.0, True), (0.3, True), (0.9, False)])
def test_predators_see_a_colony_whole(small_cfg, adhesion, eaten):
    # four cells of 0.01 in one slot: a colony only if they stick (adhesion ≥ 0.5); a full-pool
    # group of non-sticky cells is still eaten cell by cell
    sim = Sim(small_cfg)
    a = sim.life
    state = np.zeros(a.cap, np.int32)
    state[:2] = ALIVE
    a.state.from_numpy(state)
    for i in range(2):
        g = list(FOUNDER)
        if i == 1:
            g[ETA_PRED], g[PREY_RATIO], g[DIV_SIZE] = 0.9, 2.0, 0.1
        for k, v in enumerate(g):
            a.genome[i, k] = v
        a.pos[i] = [60.5, 10.5]
    a.body[0], a.ncell[0] = 0.04, 4.0
    a.genome[0, ADHESION] = adhesion
    a.body[1], a.ncell[1] = 0.05, 1.0  # predator: > 2 × 0.01, but < 2 × 0.04
    a.census()
    a.encounter[60, 10] = 0
    a.act[1] = [0.0, 0.0, 1.0, 0.0]
    a.predate()
    assert (a.body[0] < 0.04) == eaten


def test_budget_closes_with_colonies(small_cfg):
    sim = Sim(small_cfg)
    g = sim.life.genome.to_numpy()
    g[:, ADHESION] = 0.8
    sim.life.genome.from_numpy(g)
    sim.step(800)
    assert sim.life.colonies().any()
    assert sim.budget_error().worst() < 1e-9
