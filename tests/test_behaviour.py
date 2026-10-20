"""Stage 3b: brain, active movement and predation."""
import numpy as np
import pytest

from evo.config import D, O, R
from evo.life.agents import ALIVE, DYING
from evo.life.genome import (
    BRAIN,
    DIV_SIZE,
    ETA_PRED,
    FOUNDER,
    MUT_RATE,
    N_BRAIN,
    N_HID,
    N_IN,
    N_OUT,
    PREY_RATIO,
    SPEED,
)
from evo.sim import Sim


def _place(sim, specs, x=60, y=10):
    """specs: list of (body, energy, {gene: value}) placed in one grid cell; all other slots dead."""
    a = sim.life
    state = np.zeros(a.cap, np.int32)
    state[: len(specs)] = ALIVE
    a.state.from_numpy(state)
    for i, (b, e, genes) in enumerate(specs):
        a.pos[i] = [x + 0.5, y + 0.5]
        g = list(FOUNDER)
        for k, v in genes.items():
            g[k] = v
        for k, v in enumerate(g):
            a.genome[i, k] = v
        a.body[i], a.energy[i], a.ncell[i] = b, e, 1.0


def test_brain_matches_hand_computation(small_cfg):
    sim = Sim(small_cfg)
    sim.world.update_light_temp()
    rng = np.random.default_rng(1)
    w = rng.uniform(-1, 1, N_BRAIN)
    _place(sim, [(0.01, 0.005, {BRAIN + k: w[k] for k in range(N_BRAIN)})])
    a = sim.life
    a.census()
    a.think()
    # rebuild the 12 inputs exactly as the kernel does; on its first tick a founder is adapted to
    # where it is, so the four temporal-comparison senses read 0
    x, y = 60, 10
    wd = sim.world
    k_half = small_cfg.life.k_half
    r, o = wd.chem[R, x, y], wd.chem[O, x, y]
    g = a.genome.to_numpy()[0]
    inp = np.array([
        wd.light[x, y], max(r, 0) / (max(r, 0) + k_half), max(o, 0) / (max(o, 0) + k_half),
        (wd.temp[x, y] - g[4]) / g[5], np.log(1 + 1.0), np.log(1.0),
        0.005 / (small_cfg.life.energy_cap * 0.01), (wd.floor_np[x] - y) / wd.d, 0, 0, 0, 0,
    ], np.float32)
    w1 = w[: N_HID * N_IN].reshape(N_HID, N_IN)
    b1 = w[N_HID * N_IN: N_HID * N_IN + N_HID]
    w2 = w[N_HID * N_IN + N_HID:].reshape(N_OUT, N_HID)
    out = np.tanh(w2 @ np.tanh(w1 @ inp + b1))
    assert a.act.to_numpy()[0] == pytest.approx(out, abs=1e-5)


def test_zero_brain_does_nothing(small_cfg):
    sim = Sim(small_cfg)
    sim.world.update_light_temp()
    _place(sim, [(0.01, 0.005, {})])
    sim.life.census()
    sim.life.think()
    assert (sim.life.act.to_numpy()[0] == 0).all()


def test_swimming_costs_speed_squared(small_cfg):
    small_cfg.life.drift_sigma = 0.0
    sim = Sim(small_cfg)
    sim.world.update_light_temp()
    _place(sim, [(0.02, 0.01, {SPEED: 0.5})])
    a = sim.life
    a.act[0] = [0.0, 1.0, 0.0, 0.0]  # no tumble, full speed along the heading (0 rad = +x)
    a.heading[0] = 0.0
    x0 = a.pos[0][0]
    a.move()
    swim = 0.5
    assert a.energy[0] == pytest.approx(0.01 - small_cfg.life.k_move * swim**2 * 0.02, rel=1e-5)
    assert a.pos[0][0] == pytest.approx(x0 + swim, abs=0.05)  # + the ocean current there


def _hunt(small_cfg, n_pred):
    sim = Sim(small_cfg)
    sim.world.update_light_temp()
    prey = (0.001, 0.0005, {})
    preds = [(0.05, 0.02, {ETA_PRED: 0.9, PREY_RATIO: 2.0, DIV_SIZE: 0.1})] * n_pred
    _place(sim, [prey] + preds)  # slot 0 = prey; census keeps the last writer, so force it
    a = sim.life
    a.census()
    sim.life.encounter[60, 10] = 0
    for i in range(1, n_pred + 1):
        a.act[i] = [0.0, 0.0, 1.0, 0.0]
    return sim


def test_bite_splits_between_predator_and_detritus(small_cfg):
    sim = _hunt(small_cfg, 1)
    a, w = sim.life, sim.world
    d0 = w.chem[D, 60, 10]
    t0 = a.body.to_numpy()[:2].sum() + d0
    a.predate()
    want = small_cfg.life.k_pred * 0.9 * 0.05
    got = 0.001 - a.body[0]
    assert got == pytest.approx(min(want, 0.001), rel=1e-6)
    assert a.body[1] == pytest.approx(0.05 + small_cfg.life.assimilation * got, rel=1e-9)
    assert w.chem[D, 60, 10] == pytest.approx(d0 + (1 - small_cfg.life.assimilation) * got,
                                              rel=1e-9)
    assert a.event[1] == 1 and a.event[0] == 2  # the viewer's "ate" / "was bitten" flags
    assert a.body.to_numpy()[:2].sum() + w.chem[D, 60, 10] == pytest.approx(t0, rel=1e-12)


def test_many_predators_never_overdraw_prey(small_cfg):
    sim = _hunt(small_cfg, 5)  # 5 × 0.00225 wanted from a prey body of 0.001
    a = sim.life
    a.predate()
    assert a.body[0] >= 0.0
    # eaten down to nothing: dies now (a zero-body live cell once produced 0/0 = NaN downstream)
    assert a.state[0] == DYING
    gained = a.body.to_numpy()[1:6].sum() - 5 * 0.05
    assert gained == pytest.approx(small_cfg.life.assimilation * 0.001, rel=1e-6)


def test_predator_needs_bigger_cells(small_cfg):
    sim = _hunt(small_cfg, 1)
    sim.life.genome[1, PREY_RATIO] = 100.0  # 0.05 is not 100× bigger than 0.001
    sim.life.predate()
    assert sim.life.body[0] == 0.001


@pytest.mark.parametrize("n0,n1", [(3.0, 1.0), (1.0, 3.0)])
def test_encounter_odds_follow_cells_not_slot_order(small_cfg, n0, n1):
    sim = Sim(small_cfg)
    _place(sim, [(0.01, 0.005, {}), (0.01, 0.005, {})])
    a = sim.life
    a.ncell[0], a.ncell[1] = n0, n1
    picks = []
    for _ in range(2000):
        a.census()
        picks.append(a.encounter[60, 10])
    share0 = np.mean(np.array(picks) == 0)
    assert share0 == pytest.approx(n0 / (n0 + n1), abs=0.04)


def test_bitten_super_individual_loses_cells_in_proportion(small_cfg):
    sim = _hunt(small_cfg, 1)
    a = sim.life
    a.body[0], a.ncell[0] = 0.004, 4.0  # 4 cells of 0.001; the predator (0.05) is > 2× each
    a.predate()
    got = 0.004 - a.body[0]
    assert got > 0
    assert a.ncell[0] == pytest.approx(4.0 * (1 - got / 0.004), rel=1e-5)


def test_budget_closes_while_predators_hunt(small_cfg):
    # half the founders are predators whose brain always outputs "hunt" (hidden biases 2, hunt
    # weights 1 -> tanh ≈ 1); the whole ledger must still close
    from evo.life.agents import ACT_HUNT, ACT_ON

    small_cfg.life.big_mutation_p = 0.0
    sim = Sim(small_cfg)
    a = sim.life
    g = a.genome.to_numpy()
    n = small_cfg.life.n_founders
    g[:n, MUT_RATE] = 0.001  # keep the planted brains intact
    preds = np.arange(0, n, 2)
    g[preds, ETA_PRED], g[preds, PREY_RATIO], g[preds, DIV_SIZE] = 0.9, 1.0, 0.1
    g[preds, BRAIN + N_HID * N_IN: BRAIN + N_HID * N_IN + N_HID] = 2.0
    g[preds, BRAIN + N_HID * N_IN + N_HID + N_HID * ACT_HUNT:
      BRAIN + N_HID * N_IN + N_HID + N_HID * (ACT_HUNT + 1)] = 1.0
    a.genome.from_numpy(g)
    d0 = sim.world.chem.to_numpy()[D].sum()
    sim.step(300)
    act, alive = a.act.to_numpy(), a.state.to_numpy() == ALIVE
    hunters = alive & (a.genome.to_numpy()[:, ETA_PRED] > 0.5)
    assert hunters.any() and (act[hunters, ACT_HUNT] > ACT_ON).mean() > 0.9
    assert sim.world.chem.to_numpy()[D].sum() != d0
    assert sim.budget_error().worst() < 1e-9
