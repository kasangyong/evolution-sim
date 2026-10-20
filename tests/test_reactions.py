import pytest

from evo.world.reactions import REACTIONS, carbon_balance, delta_g, electron_balance


@pytest.mark.parametrize("name", REACTIONS)
def test_balanced(name):
    rx = REACTIONS[name]
    assert electron_balance(rx) == 0
    assert carbon_balance(rx) == 0


def test_delta_g_values():
    expected = {
        "h2_chemosynthesis": -1.2,
        "sulfur_fixation": 1.2,
        "sulfur_oxidation": -2.8,
        "photosynthesis": 4.0,
        "anoxygenic_photosynthesis": 1.2,
        "aerobic_respiration": -4.0,
        "anaerobic_respiration": -1.2,
        "methanogenesis": -0.8,
        "methane_oxidation": -7.2,
    }
    for name, dg in expected.items():
        assert delta_g(REACTIONS[name]) == pytest.approx(dg)


@pytest.mark.parametrize("eta", [0.3, 0.9, 0.999])
def test_no_free_energy_cycle(eta):
    # fixation (pay ΔG/η) followed by its reverse, anaerobic respiration (gain −ΔG·η)
    cost = delta_g(REACTIONS["sulfur_fixation"]) / eta
    gain = -delta_g(REACTIONS["anaerobic_respiration"]) * eta
    assert gain - cost < 0


def test_eta_genes_below_one():
    # η = 1 would let a reaction and its reverse break even (§2.3)
    from evo.life.genome import ETA_GENES, GENE_RANGE

    assert all(GENE_RANGE[g][1] < 1.0 for g in ETA_GENES)


@pytest.mark.parametrize("eta", [0.3, 0.9, 0.95])
def test_light_cycle_needs_light(eta):
    # anoxygenic photosynthesis + anaerobic respiration: the organism only recovers η² of the
    # light it spent, so the cycle is a light-driven battery, never a free-energy source
    light_spent = delta_g(REACTIONS["anoxygenic_photosynthesis"]) / eta
    gain = -delta_g(REACTIONS["anaerobic_respiration"]) * eta
    assert gain == pytest.approx(light_spent * eta * eta)
    assert gain < light_spent
