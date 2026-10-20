"""Stage 7a: bacterial-style senses (temporal comparison with adaptation), run-and-tumble, and the
energy cost of the brain."""
import numpy as np
import pytest

from evo.config import H
from evo.life.agents import ACT_ON, ALIVE
from evo.life.genome import ADAPT_TAU, BRAIN, FOUNDER, N_HID, N_IN, SPEED
from evo.sim import Sim

W1 = BRAIN  # W1[h, k] at BRAIN + N_IN·h + k
B1 = BRAIN + N_IN * N_HID
W2 = B1 + N_HID  # W2[o, h] at W2 + N_HID·o + h
SENSE_H2 = 8  # input index of the H2 comparison


def _one(sim, genes=None, x=60, y=10, body=0.01, energy=0.005):
    a = sim.life
    state = np.zeros(a.cap, np.int32)
    state[0] = ALIVE
    a.state.from_numpy(state)
    g = list(FOUNDER)
    for k, v in (genes or {}).items():
        g[k] = v
    for k, v in enumerate(g):
        a.genome[0, k] = v
    a.pos[0] = [x + 0.5, y + 0.5]
    a.body[0], a.energy[0], a.ncell[0] = body, energy, 1.0
    return a


def _sense_probe(tau=4.0):
    # hidden 0 copies the H2 comparison (tanh), output 0 (tumble) reads hidden 0
    return {W1 + N_IN * 0 + SENSE_H2: 1.0, W2 + N_HID * 0 + 0: 1.0, ADAPT_TAU: tau}


def test_senses_adapt_and_report_rises_and_falls(small_cfg):
    sim = Sim(small_cfg)
    sim.world.update_light_temp()
    a = _one(sim, _sense_probe(tau=4.0))
    w = sim.world
    w.chem[H, 60, 10] = 1.0
    reads = []
    for level in (1.0, 1.0, 1.0, np.e, np.e, np.e, np.e, np.e, np.e, np.e, np.e, 1.0):
        w.chem[H, 60, 10] = level
        a.census()
        a.think()
        reads.append(a.act[0][0])  # = tanh(tanh(H2 comparison)): two tanh layers
    reads = np.arctanh(np.arctanh(np.clip(reads, -0.999999, 0.999999)))
    assert abs(reads[2]) < 1e-5  # constant: nothing to report
    assert reads[3] == pytest.approx(1.0, abs=1e-4)  # concentration rose by e: log ratio 1
    assert 0 < reads[10] < reads[3] * 0.3  # adapted: the response fades while it stays high
    assert reads[11] < 0  # a fall reads negative


def test_tumble_threshold_picks_a_new_heading(small_cfg):
    small_cfg.life.drift_sigma = 0.0
    sim = Sim(small_cfg)
    sim.world.update_light_temp()
    a = _one(sim, {SPEED: 0.5})
    a.heading[0] = 1.0
    a.act[0] = [ACT_ON - 0.1, 0.0, 0.0, 0.0]
    a.move()
    assert a.heading[0] == 1.0  # below the threshold: keep running
    a.act[0] = [ACT_ON + 0.1, 0.0, 0.0, 0.0]
    headings = set()
    for _ in range(5):
        a.move()
        headings.add(round(float(a.heading[0]), 4))
    assert len(headings) > 1 and all(0 <= h < 6.2832 for h in headings)


def test_brain_costs_energy_in_proportion_to_its_weights(small_cfg):
    small_cfg.life.death_rate = 0.0
    sim = Sim(small_cfg)
    sim.world.update_light_temp()
    costs = []
    for scale in (0.0, 1.0, 2.0):
        a = _one(sim, {W1 + k: scale for k in range(10)}, energy=0.008)
        a.census()
        a.think()
        e0 = a.energy[0]
        a.metabolize()
        costs.append(e0 - a.energy[0])
    extra1, extra2 = costs[1] - costs[0], costs[2] - costs[0]
    assert extra1 == pytest.approx(small_cfg.life.k_brain * 10 * 0.01, rel=0.05)
    assert extra2 == pytest.approx(2 * extra1, rel=1e-3)


def test_run_and_tumble_climbs_a_gradient(small_cfg):
    # hand-set "tumble when things get worse" brain in a static H2 gradient along x: the
    # population drifts up the gradient; the same cells with a zero brain do not
    def run(smart: bool) -> float:
        cfg = small_cfg
        cfg.life.drift_sigma = 0.0
        cfg.life.death_rate = 0.0
        sim = Sim(cfg)
        w = sim.world
        w.update_light_temp()
        a = sim.life
        n = 200
        genes = {SPEED: 0.5}
        if smart:  # tumble = tanh(-4 · H2 comparison) + run at a constant speed
            genes.update({W1 + SENSE_H2: -4.0, W2 + 0: 4.0, B1 + 1: 3.0, W2 + N_HID * 1 + 1: 3.0})
        else:
            genes.update({B1 + 1: 3.0, W2 + N_HID * 1 + 1: 3.0})  # same speed, never tumbles
        g = np.tile(np.array(FOUNDER, np.float32), (a.cap, 1))
        for k, v in genes.items():
            g[:, k] = v
        a.genome.from_numpy(g)
        state = np.zeros(a.cap, np.int32)
        state[:n] = ALIVE
        a.state.from_numpy(state)
        rng = np.random.default_rng(0)
        pos = a.pos.to_numpy()
        pos[:n, 0], pos[:n, 1] = rng.uniform(40, 80, n), rng.uniform(5, 15, n)
        a.pos.from_numpy(pos)
        hd = a.heading.to_numpy()
        hd[:n] = rng.uniform(0, 6.283, n)
        a.heading.from_numpy(hd)
        chem = w.chem.to_numpy()
        xs = np.arange(chem.shape[1])[:, None]
        for _ in range(150):
            chem[H] = np.exp(xs / 20.0) * (chem[H] >= 0)  # static gradient rising toward +x
            w.chem.from_numpy(chem)
            a.census()
            a.think()
            a.move()
        return float(a.pos.to_numpy()[:n, 0].mean())

    assert run(True) > run(False) + 5.0
