"""Element budget checks (S6): total(t) == total(0) + Σsources − Σsinks for C, N and electrons."""

from dataclasses import dataclass

import numpy as np

from evo.config import ATM_C, ATM_H, ATM_M, ATM_O, CN_RATIO, NS, C, D, H, M, N, O, R, X
from evo.world.fields import SINK_C, SINK_E, SINK_N, SRC_C, SRC_E, SRC_N, World
from evo.world.reactions import E_UNITS

SPECIES_KEY = {H: "H", R: "R", X: "X", C: "C", O: "O", M: "M", N: "N", D: "D"}
ATM_E = {ATM_C: 0.0, ATM_O: -4.0, ATM_M: 8.0, ATM_H: 2.0}


@dataclass
class Totals:
    carbon: float
    nitrogen: float
    electrons: float
    electron_scale: float  # Σ|contributions|, denominator for the relative e error


def totals(world: World, agents=None) -> Totals:
    s = world.chem.to_numpy().astype(np.float64).reshape(NS, -1).sum(axis=1)
    atm = world.atm.to_numpy()
    body = float(agents.body.to_numpy().sum()) if agents is not None else 0.0
    carbon = s[C] + s[D] + s[M] + atm[ATM_C] + atm[ATM_M] + body
    nitrogen = s[N] + (s[D] + body) / CN_RATIO
    parts = [E_UNITS[SPECIES_KEY[i]] * s[i] for i in range(NS)] + [E_UNITS["B"] * body]
    parts += [ATM_E[g] * atm[g] for g in range(4)]
    return Totals(carbon, nitrogen, float(sum(parts)), float(sum(abs(p) for p in parts)))


@dataclass
class BudgetError:
    carbon: float
    nitrogen: float
    electrons: float

    def worst(self) -> float:
        return max(self.carbon, self.nitrogen, self.electrons)


def budget_error(world: World, agents, t0: Totals) -> BudgetError:
    t = totals(world, agents)
    acc = world.acc.to_numpy()
    exp_c = t0.carbon + acc[SRC_C] - acc[SINK_C]
    exp_n = t0.nitrogen + acc[SRC_N] - acc[SINK_N]
    exp_e = t0.electrons + acc[SRC_E] - acc[SINK_E]
    return BudgetError(
        abs(t.carbon - exp_c) / max(t.carbon, 1e-12),
        abs(t.nitrogen - exp_n) / max(t.nitrogen, 1e-12),
        abs(t.electrons - exp_e) / max(t.electron_scale, 1e-12),
    )
