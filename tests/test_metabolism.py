"""Stage 3a: every organism reaction moves matter by its reactions.py stoichiometry and yields
−ΔG·η (or is paid by light), plus the test-only O₂ accumulation scenario (§2.4)."""
import numpy as np
import pytest

from evo.config import ATM_H, ATM_M, ATM_O, CN_RATIO, C, Config, D, H, M, N, O, R, X
from evo.life.agents import ALIVE, BIOTIC, BODY, INPUTS, OUTPUTS, G
from evo.life.genome import (
    DIV_SIZE,
    ETA_AER,
    ETA_ANAER,
    ETA_GENES,
    ETA_H2,
    ETA_METH,
    ETA_MOX,
    ETA_PHOTO_AN,
    ETA_PHOTO_OX,
    ETA_SFIX,
    ETA_SOX,
    FOUNDER,
    O2_TOL,
    T_OPT,
    T_WIDTH,
)
from evo.sim import Sim
from evo.world.fields import BUR_ORG, SINK_E, SRC_E, make_terrain

SPECIES = (H, R, X, C, O, M, N)


def _cells(cfg, gene, eta, n=1, tol=1.0, x=60, y=10, b0=0.01, e0=0.0, div=None):
    """`n` identical organisms in one grid cell stocked with 5 units of every species; only
    `gene` is on. Returns the sim and each cell's upkeep rate per unit body."""
    cfg.life.death_rate = 0.0
    sim = Sim(cfg)
    a, w = sim.life, sim.world
    w.update_light_temp()
    state = np.zeros(a.cap, np.int32)
    state[:n] = ALIVE
    a.state.from_numpy(state)
    genome = list(FOUNDER)
    for g in ETA_GENES:
        genome[g] = 0.0
    genome[gene], genome[O2_TOL], genome[T_OPT], genome[T_WIDTH] = eta, tol, w.temp[x, y], 40.0
    if div is not None:
        genome[DIV_SIZE] = div
    for s in SPECIES:
        w.chem[s, x, y] = 5.0
    for i in range(n):
        a.pos[i] = [x + 0.5, y + 0.5]
        for g, v in enumerate(genome):
            a.genome[i, g] = v
        a.body[i], a.energy[i] = b0, e0
    upkeep = cfg.life.maintenance + cfg.life.trait_upkeep * (eta + tol)
    upkeep += cfg.life.k_tox * 5.0 * (1.0 - tol)
    return sim, upkeep


def _expect_cell(w, rx, ext, x=60, y=10):
    d = {s: 0.0 for s in SPECIES}
    for s, k in INPUTS[rx]:
        d[s] -= k * ext
    for s, k in OUTPUTS[rx]:
        d[s] += k * ext
    for s, dv in d.items():
        assert w.chem[s, x, y] == pytest.approx(5.0 + dv, rel=1e-9, abs=1e-15)


@pytest.mark.parametrize("gene,name", [(ETA_AER, "aerobic_respiration"),
                                       (ETA_ANAER, "anaerobic_respiration"),
                                       (ETA_METH, "methanogenesis")])
def test_respiration_follows_stoichiometry_and_delta_g(small_cfg, gene, name):
    rx = BIOTIC.index(name)
    eta, b0 = 0.5, 0.01
    sim, upkeep = _cells(small_cfg, gene, eta, b0=b0)  # zero energy: the cell must respire
    a, w = sim.life, sim.world
    a.metabolize()
    ext = (a.body[0] - b0) / BODY[rx]
    assert ext > 0
    _expect_cell(w, rx, ext)
    assert a.energy[0] == pytest.approx(G[rx] * eta * ext - upkeep * b0, rel=1e-5)


@pytest.mark.parametrize("gene,name,consumed", [(ETA_SOX, "sulfur_oxidation", R),
                                                (ETA_MOX, "methane_oxidation", M)])
def test_lithotrophy_yields_delta_g(small_cfg, gene, name, consumed):
    rx = BIOTIC.index(name)
    eta, b0 = 0.5, 0.01
    sim, upkeep = _cells(small_cfg, gene, eta, b0=b0)
    a, w = sim.life, sim.world
    a.metabolize()
    k = dict(INPUTS[rx])[consumed]
    ext = (5.0 - w.chem[consumed, 60, 10]) / k
    assert ext > 0 and a.body[0] == b0  # energy only, no carbon fixed
    _expect_cell(w, rx, ext)
    assert a.energy[0] == pytest.approx(G[rx] * eta * ext - upkeep * b0, rel=1e-5)


def test_sulfur_fixation_is_paid_from_stored_energy(small_cfg):
    rx = BIOTIC.index("sulfur_fixation")
    eta, b0, e0 = 0.5, 0.01, 0.008  # reserve is 0.2 × b0 = 0.002, so energy is available
    sim, upkeep = _cells(small_cfg, ETA_SFIX, eta, b0=b0, e0=e0)
    a, w = sim.life, sim.world
    a.metabolize()
    ext = (a.body[0] - b0) / BODY[rx]
    assert ext > 0
    _expect_cell(w, rx, ext)
    assert G[rx] < 0
    assert a.energy[0] == pytest.approx(e0 + G[rx] / eta * ext - upkeep * (b0 + ext), rel=1e-5)


@pytest.mark.parametrize("gene,name", [(ETA_PHOTO_AN, "anoxygenic_photosynthesis"),
                                       (ETA_PHOTO_OX, "photosynthesis")])
def test_photosynthesis_is_paid_by_light(small_cfg, gene, name):
    rx = BIOTIC.index(name)
    eta, b0, e0 = 0.5, 0.01, 0.005
    sim, upkeep = _cells(small_cfg, gene, eta, b0=b0, e0=e0)
    a, w = sim.life, sim.world
    light = w.light[60, 10]
    a.metabolize()
    ext = (a.body[0] - b0) / BODY[rx]
    # absorbed light Λ = L(1 − e^(−k·b)) pays ΔG/η: extent = η·Λ/ΔG (× R saturation, anoxygenic)
    sat = 5.0 / (5.0 + small_cfg.life.k_half) if name == "anoxygenic_photosynthesis" else 1.0
    lam = light * (1.0 - np.exp(-small_cfg.life.k_light * np.float32(b0)))
    assert light > 0.1
    assert ext == pytest.approx(eta * lam / -G[rx] * sat, rel=1e-5)
    _expect_cell(w, rx, ext)
    # light, not stored energy, paid for the fixation
    assert a.energy[0] == pytest.approx(e0 - upkeep * (b0 + ext), rel=1e-5)


def test_crowded_cell_never_absorbs_more_than_incident_light(small_cfg):
    rx = BIOTIC.index("photosynthesis")
    eta, n, b0 = 0.95, 50, 0.05  # 2.5 units of biomass in one grid cell
    sim, _ = _cells(small_cfg, ETA_PHOTO_OX, eta, n=n, b0=b0, e0=0.02, div=0.1)
    a, w = sim.life, sim.world
    light = w.light[60, 10]
    a.metabolize()
    ext = a.body.to_numpy()[:n].sum() - n * b0
    paid = ext * -G[rx] / eta  # light energy actually used
    expected = light * (1.0 - np.exp(-small_cfg.life.k_light * n * b0))
    assert paid == pytest.approx(expected, rel=1e-4)
    assert paid < light


def test_non_photosynthetic_cells_do_not_shade(small_cfg):
    # a lone photosynthetic mutant among 49 chemosynthetic cells gets the light it would alone
    rx = BIOTIC.index("photosynthesis")
    eta, b0 = 0.5, 0.01
    sim, _ = _cells(small_cfg, ETA_H2, 0.5, n=50, b0=b0, e0=0.005)
    a, w = sim.life, sim.world
    a.genome[0, ETA_H2], a.genome[0, ETA_PHOTO_OX] = 0.0, eta
    light = w.light[60, 10]
    a.metabolize()
    ext = (a.body[0] - b0) / BODY[rx]
    lam = light * (1.0 - np.exp(-small_cfg.life.k_light * np.float32(b0)))
    assert ext == pytest.approx(eta * lam / -G[rx], rel=1e-5)


def test_cell_at_twice_division_size_stops_fixing_carbon(small_cfg):
    # a full pool makes division fail; the cell must not grow without bound
    sim, _ = _cells(small_cfg, ETA_H2, 0.9, b0=0.05, e0=0.01, div=0.02)
    a, w = sim.life, sim.world
    a.metabolize()
    assert a.body[0] <= 0.05
    assert w.chem[H, 60, 10] == 5.0


def test_oxidant_costs_intolerant_cells(small_cfg):
    sim, upkeep = _cells(small_cfg, ETA_AER, 0.0, tol=0.0, e0=0.005)
    sim.life.metabolize()
    assert upkeep > small_cfg.life.maintenance
    assert sim.life.energy[0] == pytest.approx(0.005 - upkeep * 0.01, rel=1e-5)


def test_default_terrain_has_one_photic_vent():
    cfg = Config().world
    floor, vents = make_terrain(cfg)
    light = np.exp(-cfg.light_atten * vents[:, 1])
    assert (light > 0.05).sum() == cfg.n_shallow_vents == 1
    assert (floor[vents[:, 0]] - 1 == vents[:, 1]).all()


def _photosynthesis_world(cfg, burial: bool):
    """Test-only scenario (§2.4): oxygenic photosynthesizers planted on the shelf surface of a world
    with no reductant supply and little N (so the pool does not fill). Never used by real runs."""
    cfg.world.vent_h = cfg.world.vent_r = 0.0
    cfg.world.init.update({H: 0.0, R: 0.0, N: 0.02})
    cfg.life.capacity = 65536
    cfg.climate.volcano_p = 0.0  # volcanic CO2/H2 (stage 4a) would feed growth past the pool
    # escape to space is another electron sink (CH4 photolysis, H2 loss) and can dominate by
    # chance; switched off so burial is the only way O2 can be left behind
    cfg.climate.k_photo_m = cfg.climate.k_h_escape = 0.0
    if not burial:
        cfg.chem.k_bury_d = cfg.chem.k_bury_r = 0.0
    sim = Sim(cfg)
    a, w = sim.life, sim.world
    w.update_light_temp()
    n = cfg.life.n_founders
    rng = np.random.default_rng(0)
    pos = a.pos.to_numpy()
    pos[:n, 0] = rng.uniform(0, 30, n)  # flat shelf on the left
    pos[:n, 1] = rng.uniform(0.5, 5.0, n)
    a.pos.from_numpy(pos)
    genome = a.genome.to_numpy()
    genome[:n, list(ETA_GENES)] = 0.0
    genome[:n, ETA_PHOTO_OX], genome[:n, ETA_AER], genome[:n, O2_TOL] = 0.9, 0.5, 1.0
    genome[:n, T_OPT], genome[:n, T_WIDTH] = w.temp.to_numpy()[:30, :5].mean(), 40.0
    a.genome.from_numpy(genome)
    return sim


def _free_oxygen(sim) -> float:
    """O₂ not paired with reduced matter still in the system: O₂ − organic C (body + detritus)
    − (H₂ + R)/2 − 2 CH₄, ocean plus atmosphere. Only electron sinks can make it grow."""
    w, chem = sim.world, sim.world.chem.to_numpy()
    o2 = chem[O].sum() + w.atm[ATM_O]
    organic = chem[D].sum() + sim.life.body.to_numpy().sum()
    reduced = (chem[H].sum() + w.atm[ATM_H] + chem[R].sum()) / 2 + 2 * (chem[M].sum() + w.atm[ATM_M])
    return o2 - organic - reduced


def test_oxygen_accumulates_through_burial(small_cfg):
    import copy

    results = {}
    for burial in (True, False):
        sim = _photosynthesis_world(copy.deepcopy(small_cfg), burial)
        f0 = _free_oxygen(sim)
        sim.step(3000)
        w = sim.world
        assert sim.life.count() < sim.life.cap  # pool not saturated
        assert sim.budget_error().worst() < 1e-9
        gain = _free_oxygen(sim) - f0
        # electron bookkeeping: free O₂ rises by exactly the electrons that left, / 4
        assert gain == pytest.approx((w.acc[SINK_E] - w.acc[SRC_E]) / 4, rel=1e-6, abs=1e-6)
        results[burial] = (gain, w.acc[BUR_ORG])
    (gain_on, buried_on), (gain_off, buried_off) = results[True], results[False]
    assert buried_off == 0 and buried_on > 0
    # burial is what leaves O₂ behind
    # without escape to space, burial is the only electron sink: off must stay ~0
    assert gain_on > 1.0
    assert gain_off == pytest.approx(0.0, abs=1e-6)
    assert gain_on > 0.5 * buried_on


def test_body_n_ratio_is_exact():
    # growth/respiration carry N at exactly 1/CN_RATIO per unit body
    for rx, body in enumerate(BODY):
        if body:
            n = sum(k for s, k in INPUTS[rx] if s == N) - sum(k for s, k in OUTPUTS[rx] if s == N)
            assert n == body / CN_RATIO
