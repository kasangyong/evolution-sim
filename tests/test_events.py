"""Event journal detectors: each change is logged once, with hysteresis."""
from dataclasses import replace

import pytest

from evo.analysis.events import EventDetector, Summary

BASE = Summary(tick=0, cells=5000, slots=5000, lit_slots=0, o2=0.0, ch4=1.0, co2=2e5, ts=8.0,
               ice=0.0, pred_frac=0.0, colony_frac=0.0, swim_frac=0.0, sense_frac=0.0,
               species={0: (5000.0, (600.0, 340.0), "수소 화학합성")})


def kinds(det, s):
    return [e.kind for e in det.update(s)]


def test_bloom_is_logged_once_and_rearms_after_it_ends():
    d = EventDetector()
    assert kinds(d, BASE) == []
    assert "bloom" in kinds(d, replace(BASE, tick=2000, lit_slots=5000))
    assert "bloom" not in kinds(d, replace(BASE, tick=4000, lit_slots=6000))  # still blooming
    kinds(d, replace(BASE, tick=6000, lit_slots=10))  # gone
    assert "bloom" in kinds(d, replace(BASE, tick=8000, lit_slots=3000))  # a second bloom


def test_snowball_then_thaw():
    d = EventDetector()
    assert "thaw" not in kinds(d, replace(BASE, tick=2000, ice=0.0))  # no thaw without a freeze
    assert "snowball" in kinds(d, replace(BASE, tick=4000, ice=1.0, ts=-15.0))
    assert kinds(d, replace(BASE, tick=6000, ice=1.0, ts=-12.0)) == []
    assert "thaw" in kinds(d, replace(BASE, tick=8000, ice=0.0, ts=8.0))
    assert "thaw" not in kinds(d, replace(BASE, tick=10000, ice=0.0))


def test_oxygen_methane_extinction_and_abilities():
    d = EventDetector()
    kinds(d, replace(BASE, ch4=500.0))
    got = kinds(d, replace(BASE, tick=2000, o2=5e3, ch4=20.0, cells=1000, pred_frac=0.2,
                           sense_frac=0.15))
    assert {"oxygen", "methane", "extinction", "predation", "chemotaxis"} <= set(got)


def test_major_species_appear_and_disappear_with_a_place():
    # ids flicker as clusters regroup, so only what persists two checks is announced
    d = EventDetector()
    kinds(d, BASE)  # species 0 is the founder lineage: known, not news
    both = {0: (4000.0, (600.0, 340.0), "수소 화학합성"), 7: (1000.0, (300.0, 10.0), "산소 광합성")}
    assert "species" not in kinds(d, replace(BASE, tick=2000, species=both))  # once: pending
    new = [e for e in d.update(replace(BASE, tick=4000, species=both)) if e.kind == "species"]
    assert len(new) == 1 and "#7" in new[0].title and new[0].where == (300.0, 10.0)
    only7 = {7: (5000.0, (300.0, 10.0), "산소 광합성")}
    assert "species_gone" not in kinds(d, replace(BASE, tick=6000, species=only7))  # once
    gone = d.update(replace(BASE, tick=8000, species=only7))
    assert [e.title for e in gone if e.kind == "species_gone"] == ["주요 종 #0 사라짐"]


def test_a_flickering_species_is_not_news():
    d = EventDetector()
    kinds(d, BASE)
    blip = {0: (4000.0, (600.0, 340.0), "수소 화학합성"), 9: (1000.0, (1.0, 1.0), "포식")}
    assert kinds(d, replace(BASE, tick=2000, species=blip)) == []
    assert kinds(d, replace(BASE, tick=4000)) == []  # gone again: never announced, no news


def test_chemotaxis_bias_ignores_drift_and_sweeps_and_sees_selection():
    import numpy as np

    from evo.life.genome import BRAIN, ETA_H2, FOUNDER, N_HID, N_IN, SPEED
    from evo.ui.web import chemotaxis_bias

    def cell(sign, other=0.0):  # chemosynthesizer: hidden unit 0 links senses to the tumble output
        g = np.array(FOUNDER, np.float32)
        g[SPEED], g[ETA_H2] = 0.5, 0.9
        g[BRAIN + N_IN * 0 + 8] = 2.0  # W1[0, H2 sense]
        g[BRAIN + N_IN * 0 + 9] = other  # W1[0, sulfide sense]: no food value for this cell
        g[BRAIN + N_IN * N_HID + N_HID + 0] = 2.0 * sign  # W2[tumble, 0]
        return g

    food, control, _ = chemotaxis_bias(np.stack([cell(+1)] * 50 + [cell(-1)] * 50))
    assert food == 0.0 and control == 0.0  # drift: both directions alike
    food, control, tuned = chemotaxis_bias(np.stack([cell(-1)] * 80 + [cell(+1)] * 20))
    assert food == pytest.approx(0.6) and control == 0.0  # tumble when H2 falls: selection
    assert tuned.sum() == 80
    # one lineage swept with the same wiring on a sense that does not feed it: no evidence
    food, control, _ = chemotaxis_bias(np.stack([cell(-1, other=2.0)] * 100))
    assert food == pytest.approx(1.0) and control == pytest.approx(1.0)


def test_chemotaxis_bias_uses_the_real_threshold():
    import numpy as np

    from evo.life.genome import BRAIN, ETA_H2, FOUNDER, N_HID, N_IN, SPEED
    from evo.ui.web import chemotaxis_bias

    g = np.array(FOUNDER, np.float32)
    g[SPEED], g[ETA_H2] = 0.5, 0.9
    g[BRAIN + N_IN * 0 + 8] = 2.0
    g[BRAIN + N_IN * N_HID + N_HID + 0] = -0.3  # wired the right way but too weak to tumble
    assert chemotaxis_bias(g[None])[0] == 0.0
