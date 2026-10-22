"""Stage 7d: synapses change with experience by a three-factor rule (pre × post × modulator),
habituate with use, and an organism can learn to answer a smell that came with food."""
import math

import numpy as np
import pytest
from test_neural import org_with, wiring

from evo.config import R
from evo.life.genome import CONN, N_TYPES, NEURO, NT_BIAS, PL_ETA, PL_HAB, PL_MOD, PL_REC, PLAST


def plast(k, p, v):
    return {PLAST + 4 * k + p: v}


def connect(sim, oid, post, pre, w):
    """One synapse from local neuron `pre` onto local neuron `post`, grown weight w."""
    o = sim.life.orgs
    g = int(o.nidx[oid, post])
    m = int(o.nsyn[g])
    o.pre[g, m], o.w[g, m], o.dw[g, m], o.res[g, m] = pre, w, 0.0, 1.0
    o.nsyn[g] = m + 1
    return g, m


def three_neurons(cfg, mod_bias, eta=0.05, modulatory=True):
    # local 0: pre (type 0, always on), local 1: post (type 1, always on, learns), local 2: the
    # modulatory neuron (type 2) whose output is broadcast
    over = wiring(t0={"count": 1.0, "bias": 2.0, "density": 0.0},
                  t1={"count": 1.0, "bias": 2.0, "density": 0.0},
                  t2={"count": 1.0, "bias": mod_bias, "density": 0.0})
    over |= plast(1, PL_ETA, eta) | plast(2, PL_MOD, 1.0 if modulatory else 0.0)
    sim, oid, _ = org_with(cfg, 2, over)
    g, m = connect(sim, oid, post=1, pre=0, w=0.5)
    for _ in range(60):
        sim.life.orgs.neural()
    return sim.life.orgs.dw[g, m]


def test_three_factor_rule(small_cfg):
    assert three_neurons(small_cfg, mod_bias=2.0) > 0.5  # pre and post on, reward: stronger
    assert three_neurons(small_cfg, mod_bias=-2.0) < -0.5  # punishment: weaker
    assert three_neurons(small_cfg, mod_bias=2.0, modulatory=False) == 0.0  # no modulator
    assert three_neurons(small_cfg, mod_bias=2.0, eta=0.0) == 0.0  # no plasticity


def test_learned_changes_stay_bounded(small_cfg):
    small_cfg.life.dw_max = 0.3
    assert three_neurons(small_cfg, mod_bias=2.0, eta=0.1) == pytest.approx(0.3)


def test_habituation_and_recovery(small_cfg):
    over = wiring(t0={"count": 1.0, "bias": 3.0, "density": 0.0},
                  t1={"count": 1.0, "density": 0.0})
    over |= plast(0, PL_HAB, 0.1) | plast(0, PL_REC, 40.0)
    sim, oid, _ = org_with(small_cfg, 2, over)
    o = sim.life.orgs
    g, m = connect(sim, oid, post=1, pre=0, w=1.0)
    for _ in range(100):
        o.neural()
    tired = o.res[g, m]
    assert tired < 0.3  # repeated use depletes the synapse
    sim.life.genome[o.slot[oid, 0], NEURO + NT_BIAS] = 0.0  # type 0 quiet (output → 0)
    for _ in range(200):
        o.neural()
    assert o.res[g, m] > 0.9 > tired  # rest restores it


def conditioned_response(cfg, schedule, adapt=False):
    """A smell (sulfide rise) and food (a full stomach) on a schedule, then the smell alone.
    local 0: smell neuron (sulfide channel), local 1: response neuron (learns), local 2:
    modulator reading hunger (on when fed, off at a neutral half-full stomach, negative hungry).
    schedule(t) -> (smell on, fed). With adapt, the slots' receptors adapt as in a real run
    (think() each tick), so a lasting smell fades from the input within the adaptation time."""
    over = wiring(t0={"count": 1.0, "channel": 1.5, "sgain": 1.0, "density": 0.0},
                  t1={"count": 1.0, "density": 0.0, "tau": 2.0},
                  t2={"count": 1.0, "channel": 4.5, "sgain": -4.0, "bias": 2.0, "density": 0.0})
    over |= plast(1, PL_ETA, 0.05) | plast(2, PL_MOD, 1.0)
    sim, oid, slots = org_with(cfg, 2, over)
    o, a, w = sim.life.orgs, sim.life, sim.world
    g, m = connect(sim, oid, post=1, pre=0, w=0.2)
    head = slots[0]
    x, y = (int(v) for v in a.pos[head])
    base = 0.01
    a.adapt[head] = [math.log(base)] * 4  # adapted to the background smell
    full = a.body[head] * cfg.life.energy_cap

    def tick(smell_on, energy):
        w.chem[R, x, y] = 1.0 if smell_on else base
        a.energy[head] = energy
        if adapt:
            a.census()
            a.think()
        o.neural()

    for t in range(600):
        smell_on, fed = schedule(t)
        tick(smell_on, full if fed else 0.5 * full)
    for _ in range(60):
        tick(False, 0.5 * full)  # settle without the smell
    r = int(o.nidx[oid, 1])
    before = o.rate[r]
    for _ in range(3):
        tick(True, 0.5 * full)
    return o.rate[r] - before, o.dw[g, m]


def paired(t):  # smell and food together, 5 ticks every 60
    on = t % 60 < 5
    return on, on


def unpaired(t):
    # the same amount of smell and of food, 25 ticks apart: beyond the neurons' memory (τ ≤ 5), so
    # no trace of the smell is left when food comes (a 5-tick gap still learns: trace conditioning)
    return t % 60 < 5, 30 <= t % 60 < 35


def test_a_smell_that_came_with_food_is_answered_more_strongly(small_cfg):
    resp_p, dw_p = conditioned_response(small_cfg, paired)
    resp_u, dw_u = conditioned_response(small_cfg, unpaired)
    assert dw_p > 0.3 and abs(dw_u) < 0.1 * dw_p  # unpaired: the rule finds nothing to learn
    assert resp_p > 1.5 * resp_u


def test_learning_works_with_adapting_receptors(small_cfg):
    # receptors adapt within ADAPT_TAU (5 ticks by default): a smell pulse is a brief signal,
    # which still meets the food signal when they come together
    resp_p, dw_p = conditioned_response(small_cfg, paired, adapt=True)
    resp_u, dw_u = conditioned_response(small_cfg, unpaired, adapt=True)
    assert dw_p > 0.1 and dw_p > 3 * abs(dw_u)
    assert resp_p > resp_u


def test_what_was_learned_survives_a_snapshot(small_cfg):
    from evo.io.snapshot import capture, restore
    from evo.sim import Sim

    over = wiring(t0={"count": 1.0, "bias": 2.0, "density": 0.0},
                  t1={"count": 1.0, "bias": 2.0, "density": 0.0},
                  t2={"count": 1.0, "bias": 2.0, "density": 0.0})
    over |= plast(1, PL_ETA, 0.05) | plast(2, PL_MOD, 1.0) | plast(0, PL_HAB, 0.05)
    sim, oid, _ = org_with(small_cfg, 2, over)
    g, m = connect(sim, oid, post=1, pre=0, w=0.5)
    for _ in range(30):
        sim.life.orgs.neural()
    sim2 = Sim(small_cfg)
    restore(sim2, capture(sim))
    o, o2 = sim.life.orgs, sim2.life.orgs
    assert o2.dw[g, m] == o.dw[g, m] != 0.0 and o2.res[g, m] == o.res[g, m] < 1.0
    assert np.isclose(o2.learned(), o.learned()).all()


def test_plastic_synapses_cost_energy(small_cfg):
    over = wiring(t0={"count": 1.0, "density": 0.0}, t1={"count": 1.0, "density": 0.0})
    over |= plast(1, PL_ETA, 0.05)
    sim, oid, slots = org_with(small_cfg, 2, over)
    o, a = sim.life.orgs, sim.life
    o.nplastic[oid] = 100
    o.wsum[oid] = 0.0
    a.energy[slots[0]] = 0.01
    e0 = a.energy[slots[0]]
    o.neural()
    cfg = small_cfg.life
    assert e0 - a.energy[slots[0]] == pytest.approx(2 * cfg.k_neuron + 100 * cfg.k_plastic, rel=1e-3)


def grown_org(cfg):
    from test_organisms import divide

    from evo.life.genome import N_SEG, SEG_CELLS

    over = wiring(t0={"count": 1.0, "bias": 2.0, "density": 4.0},
                  t1={"count": 1.0, "segmental": 1.0, "bias": 2.0, "density": 4.0})
    over |= {N_SEG: 6.0, SEG_CELLS: 4.0} | plast(1, PL_ETA, 0.05) | plast(0, PL_MOD, 1.0)
    over[CONN + N_TYPES * 1 + 0] = 4.0  # segmental neurons listen to the head neuron
    sim, oid, slots = org_with(cfg, 3, over)
    return sim, oid, slots, divide


def test_a_growing_body_keeps_what_it_learned(small_cfg):
    sim, oid, _, divide = grown_org(small_cfg)
    o, a = sim.life.orgs, sim.life
    for _ in range(40):
        o.neural()
    before = {int(g): (o.dw.to_numpy()[g].copy(), int(o.nsyn[g])) for g in o.neurons_of(oid)}
    assert any(np.abs(d).max() > 0.1 for d, _ in before.values())  # something was learned
    tail = int(o.slot[oid, o.n[oid] - 1])
    a.ncell[tail] = 4.0
    divide(sim, tail)  # a new segment
    o.integrity()
    o.wire()
    assert o.n[oid] == 4 and o.nn[oid] == len(before) + 1  # one segmental neuron added
    for g, (dw, ns) in before.items():  # the old neurons are the same, with what they learned
        assert g in o.neurons_of(oid).tolist() and int(o.nsyn[g]) == ns
        assert np.array_equal(o.dw.to_numpy()[g], dw)


def test_a_cut_body_keeps_the_front_and_drops_the_rest(small_cfg):
    sim, oid, slots, _ = grown_org(small_cfg)
    o, a = sim.life.orgs, sim.life
    for _ in range(40):
        o.neural()
    head_g = int(o.nidx[oid, 0])
    dw_head = o.dw.to_numpy()[head_g].copy()
    free0 = o.ntop[None]
    a.state[slots[2]] = 2  # the third segment dies: the body is cut there
    a.die()
    o.integrity()
    o.wire()
    assert o.n[oid] == 2 and o.nn[oid] == 3  # head neuron + segmental neurons of segments 0, 1
    assert o.ntop[None] == free0 + 1  # the third segment's neuron went back to the pool
    g = o.neurons_of(oid)
    assert (o.nseg.to_numpy()[g] < 2).all()
    pre, ns = o.pre.to_numpy()[g], o.nsyn.to_numpy()[g]
    assert all(pre[i, m] < len(g) for i in range(len(g)) for m in range(ns[i]))  # renumbered
    assert np.array_equal(o.dw.to_numpy()[head_g], dw_head)


def test_a_constant_modulator_does_not_drive_synapses_to_their_bound(small_cfg):
    # the modulator is phasic (output minus its recent mean): a steady "reward" is no news, so
    # after the first surprise the learned change fades instead of saturating
    over = wiring(t0={"count": 1.0, "bias": 2.0, "density": 0.0},
                  t1={"count": 1.0, "bias": 2.0, "density": 0.0},
                  t2={"count": 1.0, "bias": 2.0, "density": 0.0})
    over |= plast(1, PL_ETA, 0.1) | plast(2, PL_MOD, 1.0)
    sim, oid, _ = org_with(small_cfg, 2, over)
    g, m = connect(sim, oid, post=1, pre=0, w=0.5)
    o = sim.life.orgs
    peak = 0.0
    for _ in range(2500):
        o.neural()
        peak = max(peak, o.dw[g, m])
    assert 0.3 < peak < small_cfg.life.dw_max  # learned from the surprise, never to the bound
    assert o.dw[g, m] < 0.25 * peak  # then forgot, as nothing new happened
