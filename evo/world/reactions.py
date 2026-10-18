"""Reaction table (§2.3). Single source of truth for stoichiometry, electron balance and ΔG.

Stoichiometry maps species name -> coefficient (negative = consumed). "B" is organism body
carbon (CH2O, carries the same e/μ as detritus). Energy yield for organisms is computed from
ΔG = Σμ(products) − Σμ(reactants), never hard-coded.
"""

E_UNITS = {"H": 2, "R": 2, "X": 0, "C": 0, "O": -4, "M": 8, "N": 0, "D": 4, "B": 4}
MU = {"H": 2.6, "R": 1.4, "X": 0.0, "C": 0.0, "O": 0.0, "M": 7.2, "N": 0.0, "D": 4.0, "B": 4.0}
CARBON = {"C": 1, "M": 1, "D": 1, "B": 1}

REACTIONS = {
    # biotic (used from stage 2)
    "h2_chemosynthesis": {"H": -2, "C": -1, "B": 1},
    "sulfur_fixation": {"R": -2, "C": -1, "B": 1, "X": 2},
    "sulfur_oxidation": {"R": -2, "O": -1, "X": 2},
    "photosynthesis": {"C": -1, "B": 1, "O": 1},
    "aerobic_respiration": {"B": -1, "O": -1, "C": 1},
    "anaerobic_respiration": {"B": -1, "X": -2, "C": 1, "R": 2},
    "methanogenesis": {"B": -2, "C": 1, "M": 1},
    "methane_oxidation": {"M": -1, "O": -2, "C": 1},
    # abiotic (stage 1)
    "abiotic_r_oxidation": {"R": -2, "O": -1, "X": 2},
    "abiotic_h_oxidation": {"H": -2, "O": -1},
    "abiotic_m_oxidation": {"M": -1, "O": -2, "C": 1},
    "decay_oxic": {"D": -1, "O": -1, "C": 1},
    "decay_sulfate": {"D": -1, "X": -2, "C": 1, "R": 2},
    "decay_methanogenic": {"D": -2, "C": 1, "M": 1},
}


def electron_balance(rx: dict) -> float:
    return sum(E_UNITS[s] * k for s, k in rx.items())


def carbon_balance(rx: dict) -> float:
    return sum(CARBON.get(s, 0) * k for s, k in rx.items())


def delta_g(rx: dict) -> float:
    return sum(MU[s] * k for s, k in rx.items())
