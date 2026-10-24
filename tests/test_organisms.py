"""Stage 7b: colonies become segmented organisms that grow, reproduce through single cells,
fragment, and swim by resistive force theory."""
import numpy as np
import pytest

from evo.life.agents import ALIVE, DYING
from evo.life.genome import (
    ADHESION,
    CPG_AMP,
    CPG_FREQ,
    CPG_LAG,
    DIFF,
    FOUNDER,
    MUT_RATE,
    N_SEG,
    SEG_CELLS,
    TISSUE,
)
from evo.life.organisms import O_ALIVE
from evo.sim import Sim

MUSCLE, CILIA, GERM = TISSUE, TISSUE + 2, TISSUE + 4  # the "a" gene of each tissue


def genes(over=None):
    g = list(FOUNDER)
    g[ADHESION], g[DIFF] = 0.9, 0.9
    g[MUSCLE] = g[CILIA] = g[GERM] = -4.0  # all metabolic tissue unless asked
    for k, v in (over or {}).items():
        g[k] = v
    return g


def empty_world(cfg):
    sim = Sim(cfg)
    a = sim.life
    a.state.fill(0)
    a.top[None] = a.cap
    a.free.from_numpy(np.arange(a.cap - 1, -1, -1, dtype=np.int32))
    return sim


def add_slot(a, g, pos, body=0.04, ncell=16.0, energy=None):
    """Put one living slot in the next free place; returns its index."""
    a.top[None] -= 1
    i = int(a.free[a.top[None]])
    a.state[i] = ALIVE
    for k, v in enumerate(g):
        a.genome[i, k] = v
    a.pos[i] = pos
    a.body[i], a.ncell[i] = body, ncell
    a.energy[i] = 0.5 * body if energy is None else energy
    a.uid[i] = a.next_uid[None]
    a.next_uid[None] += 1
    a.org[i], a.seg[i], a.metab_frac[i] = -1, 0, 1.0
    return i


def make_org(sim, g, n, head=(64.5, 20.5), psi=0.0, energy=None):
    """An organism of n segments laid out along −x from the head (psi = 0: facing +x)."""
    a, o = sim.life, sim.life.orgs
    slots = [add_slot(a, g, (head[0] - 0.5 * k, head[1]), energy=energy) for k in range(n)]
    o.top[None] -= 1
    oid = int(o.free[o.top[None]])
    o.state[oid], o.n[oid] = O_ALIVE, n
    for k, i in enumerate(slots):
        o.slot[oid, k], o.suid[oid, k] = i, a.uid[i]
        a.org[i], a.seg[i] = oid, k
    o.head[oid], o.psi[oid] = head, psi
    o.tissue()
    return oid, slots


def divide(sim, i):
    sim.life.want_div[i] = 1
    sim.life.weigh_divisions()
    sim.life.orgs.develop()
    sim.life.reproduce()


def test_only_a_big_differentiating_colony_becomes_an_organism(small_cfg):
    sim = empty_world(small_cfg)
    a = sim.life
    yes = add_slot(a, genes(), (30.5, 20.5), ncell=16.0)
    small = add_slot(a, genes(), (40.5, 20.5), ncell=8.0)
    plain = add_slot(a, genes({DIFF: 0.2}), (50.5, 20.5), ncell=64.0)
    loose = add_slot(a, genes({ADHESION: 0.1}), (60.5, 20.5), ncell=64.0)
    a.orgs.transition()
    assert a.orgs.count() == 1
    assert a.org[yes] >= 0 and a.org[small] < 0 and a.org[plain] < 0 and a.org[loose] < 0


def test_the_tail_buds_segments_up_to_n_seg_with_the_same_genome(small_cfg):
    sim = empty_world(small_cfg)
    a, o = sim.life, sim.life.orgs
    oid, (head,) = make_org(sim, genes({N_SEG: 3.0, SEG_CELLS: 4.0}), 1)
    a.ncell[head] = 4.0
    for _ in range(5):
        tail = int(o.slot[oid, o.n[oid] - 1])
        a.ncell[tail] = 4.0  # full: the next division buds a segment
        divide(sim, tail)
    assert o.n[oid] == 3  # stops at N_SEG
    slots = o.slot.to_numpy()[oid, :3]
    g = a.genome.to_numpy()[slots]
    assert (g == g[0]).all()  # somatic cells share one genome
    assert [a.seg[s] for s in slots] == [0, 1, 2] and all(a.org[s] == oid for s in slots)


def test_a_segment_grows_its_cells_before_budding(small_cfg):
    sim = empty_world(small_cfg)
    a, o = sim.life, sim.life.orgs
    oid, (head,) = make_org(sim, genes({SEG_CELLS: 32.0}), 1)
    a.ncell[head] = 8.0
    divide(sim, head)
    assert a.ncell[head] == 16.0 and o.n[oid] == 1


def test_germ_tissue_releases_one_mutated_cell(small_cfg):
    sim = empty_world(small_cfg)
    a, o = sim.life, sim.life.orgs
    oid, (head,) = make_org(sim, genes({GERM: 4.0, MUT_RATE: 0.05}), 1)
    a.ncell[head], a.body[head] = 8.0, 0.08
    before = a.count()
    divide(sim, head)
    assert a.count() == before + 1
    alive = np.nonzero(a.state.to_numpy() == ALIVE)[0]
    spore = int(next(i for i in alive if i != head))
    assert a.org[spore] == -1 and a.ncell[spore] == 1.0
    assert a.body[spore] == pytest.approx(0.01) and a.body[head] == pytest.approx(0.07)
    assert not np.array_equal(a.genome.to_numpy()[spore], a.genome.to_numpy()[head])
    assert o.n[oid] == 1 and a.ncell[head] == 7.0
    assert a.gen[spore] == a.gen[head] + 1  # the next generation


def test_a_dead_segment_cuts_the_body_and_a_dead_head_dissolves_it(small_cfg):
    sim = empty_world(small_cfg)
    a, o = sim.life, sim.life.orgs
    oid, slots = make_org(sim, genes(), 4)
    a.state[slots[2]] = DYING
    a.die()
    o.integrity()
    assert o.n[oid] == 2
    assert a.org[slots[3]] == -1 and a.metab_frac[slots[3]] == 1.0  # fell off as a colony
    a.state[slots[0]] = DYING
    a.die()
    o.integrity()
    assert o.state[oid] != O_ALIVE and o.count() == 0 and a.org[slots[1]] == -1


def test_a_reused_slot_is_not_taken_for_the_old_segment(small_cfg):
    sim = empty_world(small_cfg)
    a, o = sim.life, sim.life.orgs
    oid, slots = make_org(sim, genes(), 3)
    a.uid[slots[1]] = 999_999  # the slot died and a stranger was born into it
    o.integrity()
    assert o.n[oid] == 1


def test_energy_is_shared_by_body(small_cfg):
    sim = empty_world(small_cfg)
    a, o = sim.life, sim.life.orgs
    _, slots = make_org(sim, genes(), 3)
    a.body[slots[0]], a.body[slots[1]], a.body[slots[2]] = 0.02, 0.02, 0.04
    a.energy[slots[0]], a.energy[slots[1]], a.energy[slots[2]] = 0.08, 0.0, 0.0
    o.share_energy()
    assert [a.energy[s] for s in slots] == pytest.approx([0.02, 0.02, 0.04])


def test_metabolism_only_in_metabolic_tissue(small_cfg):
    sim = empty_world(small_cfg)
    a = sim.life
    _, (seg,) = make_org(sim, genes({MUSCLE: 4.0}), 1)
    assert a.metab_frac[seg] < 0.05


def swim_distance(cfg, n=8, ticks=400, over=None):
    """Head displacement, and its component along the body's time-averaged head-ward axis
    (head minus the mean of the segments): > 0 means head first."""
    sim = empty_world(cfg)
    sim.world.psi.fill(0.0)  # still water
    g = genes({MUSCLE: 4.0, CPG_AMP: 0.8, CPG_FREQ: 0.05, CPG_LAG: 0.6, **(over or {})})
    oid, slots = make_org(sim, g, n, head=(70.5, 20.5), energy=10.0)
    start = sim.life.orgs.head.to_numpy()[oid].copy()
    axis = np.zeros(2)
    for _ in range(ticks):
        sim.life.orgs.swim()
        pos = sim.life.pos.to_numpy()[slots]
        axis += pos[0] - pos.mean(axis=0)
    d = sim.life.orgs.head.to_numpy()[oid] - start
    return d, float(d @ axis / max(np.linalg.norm(axis), 1e-9))


def test_a_backward_bending_wave_swims_forward(small_cfg):
    _, along_f = swim_distance(small_cfg)
    _, along_b = swim_distance(small_cfg, over={CPG_LAG: -0.6})
    still, _ = swim_distance(small_cfg, over={CPG_AMP: 0.0})
    assert along_f > 1.0  # head first
    assert along_b < -1.0  # the wave the other way: tail first
    assert np.abs(still).max() < 0.01 * along_f  # only the trace of cilia left (share ~3e-4)


def test_a_standing_wave_goes_nowhere(small_cfg):
    # all joints in phase: a reciprocal stroke, which at low Reynolds number gives no net motion
    # (scallop theorem); the midpoint rule keeps the integration time-symmetric
    d, _ = swim_distance(small_cfg, over={CPG_LAG: 0.0, CILIA: -8.0})
    assert np.abs(d).max() < 0.01


def test_cilia_alone_glide_forward(small_cfg):
    d, along = swim_distance(small_cfg, n=4, over={MUSCLE: -4.0, CILIA: 4.0, CPG_AMP: 0.0})
    assert along > 0.5 and abs(d[1]) < 0.1 * d[0]


def test_budget_holds_with_organisms(small_cfg):
    from evo.world.ledger import totals

    sim = Sim(small_cfg)
    g = genes({N_SEG: 6.0, SEG_CELLS: 4.0, MUSCLE: 0.0, GERM: 1.0, CPG_AMP: 0.6})
    for vx, vy in sim.world.vents_np[:2]:
        make_org(sim, g, 2, head=(vx + 0.5, vy - 0.5), energy=0.04)
    sim.t0 = totals(sim.world, sim.life)  # the bodies above were added after the start
    sim.step(20)
    assert sim.life.orgs.count() >= 1  # swimming, sharing, budding and spores all ran
    sim.step(380)
    assert sim.budget_error().worst() < 1e-9


def test_same_seed_same_organisms(small_cfg):
    import taichi as ti

    def run():
        ti.init(arch=ti.cpu, cpu_max_num_threads=1, random_seed=0, log_level=ti.WARN)
        sim = Sim(small_cfg)
        make_org(sim, genes({N_SEG: 6.0, SEG_CELLS: 4.0, CPG_AMP: 0.6}), 2, head=(40.5, 30.5))
        sim.step(200)
        return sim.life.pos.to_numpy(), sim.life.orgs.head.to_numpy()

    (p1, h1), (p2, h2) = run(), run()
    assert np.array_equal(p1, p2) and np.array_equal(h1, h2)


def test_travel_counts_swimming_not_drifting(small_cfg):
    sim = empty_world(small_cfg)  # currents on
    oid, _ = make_org(sim, genes({MUSCLE: 4.0, CILIA: -8.0, CPG_AMP: 0.0}), 4, head=(64.5, 30.5),
                      energy=10.0)
    o = sim.life.orgs
    start = o.head.to_numpy()[oid].copy()
    for _ in range(200):
        o.swim()
    assert np.linalg.norm(o.head.to_numpy()[oid] - start) > 0.1  # carried by the current
    assert o.travel[oid] < 1e-3  # but it did not swim


def test_cells_keep_their_flagella_in_a_body(small_cfg):
    # a colony swims with its cells' flagella (speed gene × brain speed output); becoming an
    # organism must not take that away: a one-segment body glides at about that speed
    from evo.life.agents import ACT_SPEED
    from evo.life.genome import SPEED

    sim = empty_world(small_cfg)
    sim.world.psi.fill(0.0)
    oid, (head,) = make_org(sim, genes({SPEED: 0.2, CILIA: -8.0, CPG_AMP: 0.0}), 1,
                            head=(70.5, 20.5), energy=10.0)
    a, o = sim.life, sim.life.orgs
    act = a.act.to_numpy()
    act[head, ACT_SPEED] = 1.0
    a.act.from_numpy(act)
    for _ in range(50):
        o.swim()
    assert o.travel[oid] == pytest.approx(50 * 0.2, rel=0.05)
    act[head, ACT_SPEED] = 0.0  # the brain says rest: no flagellar push
    a.act.from_numpy(act)
    t0 = o.travel[oid]
    for _ in range(50):
        o.swim()
    assert o.travel[oid] - t0 < 1e-3


def test_flagellar_swimming_costs_the_same_in_a_body_as_in_a_colony(small_cfg):
    # straight body of n segments pushed at speed s: dissipated power n·c_t·s², charged
    # k_work × power × mean segment body = k_work·c_t·s²·B, the colony pays k_move·s²·B
    life = small_cfg.life
    assert life.k_work * life.drag_t == pytest.approx(life.k_move)
