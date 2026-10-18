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
