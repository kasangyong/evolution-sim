"""Stage 7c: nervous systems grow from neuron-type genes and run as graded CTRNNs that bend the
body; a head oscillator relayed back by proprioception makes a swimming wave."""
import numpy as np
import pytest
from test_organisms import MUSCLE, empty_world, genes, make_org

from evo.life.genome import (
    CONN,
    CPG_AMP,
    N_TYPES,
    NEURO,
    NT_BIAS,
    NT_CHANNEL,
    NT_COUNT,
    NT_DENSITY,
    NT_MGAIN,
    NT_SEGMENTAL,
    NT_SGAIN,
    NT_TAU,
)


def nt(k, p):
    return NEURO + 8 * k + p


def wiring(**types):
    """Gene overrides: types=dict(k=dict(count=.., segmental=.., ...))."""
    names = {"count": NT_COUNT, "segmental": NT_SEGMENTAL, "tau": NT_TAU, "bias": NT_BIAS,
             "channel": NT_CHANNEL, "sgain": NT_SGAIN, "mgain": NT_MGAIN, "density": NT_DENSITY}
    over = {}
    for k, spec in types.items():
        for name, v in spec.items():
            over[nt(int(k[1:]), names[name])] = v
    return over


def org_with(cfg, n, over, still=True):
    sim = empty_world(cfg)
    if still:
        sim.world.psi.fill(0.0)
    oid, slots = make_org(sim, genes({MUSCLE: 4.0, CPG_AMP: 0.0, **over}), n, head=(70.5, 20.5),
                          energy=10.0)
    sim.life.orgs.dirty[oid] = 1
    sim.life.orgs.wire()
    return sim, oid, slots


def test_neurons_grow_in_the_head_or_in_every_segment(small_cfg):
    over = wiring(t0={"count": 3.0}, t1={"count": 2.0, "segmental": 1.0})
    sim, oid, _ = org_with(small_cfg, 4, over)
    o = sim.life.orgs
    n = o.nn[oid]
    types, segs = o.ntype.to_numpy()[oid, :n], o.nseg.to_numpy()[oid, :n]
    assert n == 3 + 2 * 4
    assert (segs[types == 0] == 0).all()  # head type
    assert sorted(segs[types == 1].tolist()) == [0, 0, 1, 1, 2, 2, 3, 3]


def test_segmental_neurons_wire_only_to_neighbours(small_cfg):
    over = wiring(t1={"count": 1.0, "segmental": 1.0, "density": 24.0})
    over[CONN + N_TYPES * 1 + 1] = 4.0  # always accepted
    sim, oid, _ = org_with(small_cfg, 8, over)
    o = sim.life.orgs
    n = o.nn[oid]
    seg, pre, ns = o.nseg.to_numpy()[oid], o.pre.to_numpy()[oid], o.nsyn.to_numpy()[oid]
    pairs = [(seg[i], seg[pre[i, m]]) for i in range(n) for m in range(ns[i])]
    assert len(pairs) > 50 and all(abs(a - b) <= 1 for a, b in pairs)


def test_a_neuron_relaxes_to_its_bias_with_its_time_constant(small_cfg):
    sim, oid, _ = org_with(small_cfg, 2, wiring(t0={"count": 1.0, "bias": 2.0, "tau": 5.0,
                                                    "density": 0.0}))
    o = sim.life.orgs
    for _ in range(5):
        o.neural()
    assert 0.55 * 2.0 < o.v[oid, 0] < 0.7 * 2.0  # ≈ 1 − e^−1 after one τ
    for _ in range(60):
        o.neural()
    assert o.v[oid, 0] == pytest.approx(2.0, rel=0.01)


def oscillator(sim, oid, k=1.5, c=2.0):
    """Two head neurons, self-excitation k and rotation c: an unstable spiral bounded by tanh."""
    o = sim.life.orgs
    for i, (wa, wb) in enumerate([(k, -c), (c, k)]):  # neuron i gets [from 0, from 1]
        o.pre[oid, i, 0], o.w[oid, i, 0] = 0, wa
        o.pre[oid, i, 1], o.w[oid, i, 1] = 1, wb
        o.nsyn[oid, i] = 2
    o.v[oid, 0] = 0.1


def test_two_neurons_make_a_rhythm(small_cfg):
    sim, oid, _ = org_with(small_cfg, 2, wiring(t0={"count": 1.0, "tau": 1.0, "density": 0.0},
                                                t1={"count": 1.0, "tau": 1.0, "density": 0.0}))
    oscillator(sim, oid)
    o = sim.life.orgs
    rates = []
    for _ in range(300):
        o.neural()
        rates.append(o.rate[oid, 0])
    r = np.array(rates[50:])
    assert (np.diff(np.sign(r)) != 0).sum() >= 6 and r.max() > 0.5 and r.min() < -0.5


def test_a_tonic_motor_neuron_curls_the_body(small_cfg):
    over = wiring(t0={"count": 1.0, "segmental": 1.0, "bias": 3.0, "mgain": 0.3, "density": 0.0})
    sim, oid, _ = org_with(small_cfg, 6, over)
    o = sim.life.orgs
    for _ in range(30):
        o.neural()
        o.swim()
    joints = o.joint.to_numpy()[oid, 1:5]  # 6 beads: joints 1 .. 4
    assert (joints > 0.25).all()  # every joint bent the same way by its motor neuron


def swim_with_nerves(cfg, relay_gain):
    # head oscillator (types 0, 1) bends joint 1; a segmental relay (type 2) bends each joint as
    # the joint ahead was bent: a wave runs head to tail with no myogenic rhythm (amplitude 0)
    over = wiring(t0={"count": 1.0, "tau": 6.0, "mgain": 0.7, "density": 0.0},
                  t1={"count": 1.0, "tau": 6.0, "density": 0.0},
                  t2={"count": 1.0, "segmental": 1.0, "channel": 6.5, "sgain": 1.0,
                      "mgain": relay_gain, "tau": 3.0, "density": 0.0})
    sim, oid, _ = org_with(cfg, 8, over)
    oscillator(sim, oid)
    o = sim.life.orgs
    start = o.head.to_numpy()[oid].copy()
    for _ in range(800):
        o.neural()
        o.swim()
    return np.linalg.norm(o.head.to_numpy()[oid] - start)


def test_a_neural_wave_swims_and_a_lone_hinge_does_not(small_cfg):
    wave = swim_with_nerves(small_cfg, 1.0)
    hinge = swim_with_nerves(small_cfg, 0.0)  # only the head joint flaps: reciprocal, no wave
    assert wave > 1.0 and wave > 5 * hinge


def test_nerves_cost_energy_per_neuron_and_synapse(small_cfg):
    sim, oid, slots = org_with(small_cfg, 2, wiring(t0={"count": 4.0, "density": 0.0}))
    o, a = sim.life.orgs, sim.life
    o.wsum[oid] = 2.0
    a.energy[slots[0]] = 0.01  # small, so f32 resolves the cost (~4e-6)
    e0 = a.energy[slots[0]]
    o.neural()
    cfg = small_cfg.life
    assert e0 - a.energy[slots[0]] == pytest.approx(4 * cfg.k_neuron + 2.0 * cfg.k_syn, rel=1e-3)


def test_budget_holds_with_nervous_organisms(small_cfg):
    from test_organisms import GERM

    from evo.life.genome import N_SEG, SEG_CELLS
    from evo.sim import Sim
    from evo.world.ledger import totals

    sim = Sim(small_cfg)
    over = wiring(t0={"count": 2.0, "segmental": 1.0, "mgain": 0.3, "density": 8.0},
                  t1={"count": 2.0, "channel": 6.5, "sgain": 1.0, "density": 8.0})
    over.update({CONN + 1: 2.0, CONN + N_TYPES: -2.0, N_SEG: 6.0, SEG_CELLS: 4.0, GERM: 1.0})
    for vx, vy in sim.world.vents_np[:2]:
        oid, _ = make_org(sim, genes(over), 2, head=(vx + 0.5, vy - 0.5), energy=0.04)
        sim.life.orgs.dirty[oid] = 1
    sim.t0 = totals(sim.world, sim.life)
    sim.step(20)
    assert sim.life.orgs.nn.to_numpy().max() > 0  # nerves grew and ran
    sim.step(380)
    assert sim.budget_error().worst() < 1e-9
