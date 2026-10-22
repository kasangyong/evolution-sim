"""Species and phylogeny (stage 4d, design decision 8; top-level design decision 6).

Every `every` ticks, sample up to `sample` genomes, cluster their trait genes (brain excluded,
each gene scaled by its range) leader-follower style, and match clusters to the previous species'
centres. A cluster near an existing species keeps its id; otherwise it is a new species whose
parent is the nearest existing one. Pure numpy: runs on the CPU, no Taichi calls."""
import json

import numpy as np

from evo.life.genome import GENE_RANGE, SPECIES_GENES

HISTORY_POINTS = 256  # per species; halved (every other point) when exceeded
_IDX = np.array(SPECIES_GENES)
_LO = np.array([GENE_RANGE[g][0] for g in SPECIES_GENES], np.float32)
_SPAN = np.array([GENE_RANGE[g][1] - GENE_RANGE[g][0] for g in SPECIES_GENES], np.float32)


def traits(genomes: np.ndarray) -> np.ndarray:
    """Trait genes scaled to [0, 1] by their ranges."""
    return (genomes[:, _IDX] - _LO) / _SPAN


def distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mean absolute scaled difference, broadcasting over leading axes."""
    return np.abs(a - b).mean(axis=-1)


class SpeciesTracker:
    # 0.2: above the 90th percentile (0.187) of distances within one population (seed 7, tick
    # 10,000; median 0.124), so ordinary variation stays inside a species
    KEEP = 1.25  # a cluster keeps an existing id up to KEEP × threshold (no flicker at the edge)

    MIN_FRAC = 0.005  # clusters under 0.5% of all cells that sit near a big one are merged into it

    def __init__(self, threshold: float = 0.2, sample: int = 10_000, seed: int = 0):
        self.threshold = threshold
        self.sample = sample
        self.rng = np.random.default_rng(seed)
        self.species: list[dict] = []  # id, parent, born, last, centre (list), cells, alive
        self.last_tick = None

    def _new(self, parent, tick: int, centre: np.ndarray, cells: float) -> int:
        sid = len(self.species)
        self.species.append({"id": sid, "parent": parent, "born": tick, "last": tick,
                             "centre": centre.tolist(), "cells": cells, "alive": True,
                             "history": [[tick, cells]]})
        return sid

    def update(self, tick: int, genomes: np.ndarray, ncell: np.ndarray) -> np.ndarray:
        """Classify the living slots (genomes: (n, N_GENES), ncell: (n,)); returns each slot's
        species id. Species with no members are marked extinct."""
        n = len(genomes)
        for sp in self.species:
            sp["alive"] = False
        if n == 0:
            return np.zeros(0, np.int32)
        t = traits(genomes)
        pick = self.rng.choice(n, size=min(n, self.sample), replace=False)
        # last time's species centres lead first, so the same population keeps the same id;
        # only genomes far from all of them start new clusters (random leaders alone redrew the
        # boundaries every check and made ids flicker)
        leaders: list[np.ndarray] = [np.array(sp["centre"], np.float32) for sp in self.species
                                     if sp["last"] == self.last_tick]
        for v in t[pick]:
            if not leaders or distance(np.array(leaders), v).min() > self.threshold:
                leaders.append(v)
        lead = np.array(leaders)
        # every slot joins its nearest leader (chunked: n × leaders × genes can be large);
        # centres are cell-weighted member means
        member = np.concatenate([np.argmin(distance(t[i:i + 8192, None, :], lead[None]), axis=1)
                                 for i in range(0, n, 8192)])
        w = ncell.astype(np.float64)
        size = np.bincount(member, weights=w, minlength=len(lead))
        big = size >= self.MIN_FRAC * w.sum()
        if not big.all() and big.any():
            # an outlier cluster joins a big one only if it is close (within KEEP × threshold);
            # a small but different community (e.g. deep vents under a bloom) stays its own species
            d_big = distance(lead[:, None, :], lead[big][None, :, :]).min(axis=1)
            lead = lead[big | (d_big > self.KEEP * self.threshold)]
            member = np.concatenate([np.argmin(distance(t[i:i + 8192, None, :], lead[None]), axis=1)
                                     for i in range(0, n, 8192)])
        # a leader nobody joined (a previous centre with no members now) is no cluster
        used = np.unique(member)
        if len(used) < len(lead):
            lead = lead[used]
            member = np.searchsorted(used, member)
        centres = np.zeros_like(lead)
        cells = np.zeros(len(lead))
        for k in range(len(lead)):
            m = member == k
            cells[k] = w[m].sum()
            centres[k] = (t[m] * w[m, None]).sum(axis=0) / max(cells[k], 1e-12)
        # match clusters to the species alive at the previous update, closest pairs first: each
        # species' id goes to its nearest cluster (within KEEP × threshold); every other cluster
        # becomes a daughter of its nearest previous species
        prev = [sp for sp in self.species if sp["last"] == self.last_tick]
        cluster_id = np.full(len(lead), -1, np.int32)
        parent_of = [None] * len(lead)
        if prev:
            d = distance(centres[:, None, :], np.array([sp["centre"] for sp in prev])[None, :, :])
            nearest = d.argmin(axis=1)
            parent_of = [prev[j]["id"] for j in nearest]
            taken = set()
            for flat in np.argsort(d, axis=None):
                k, j = np.unravel_index(flat, d.shape)
                if d[k, j] > self.KEEP * self.threshold:
                    break
                if cluster_id[k] < 0 and j not in taken:
                    cluster_id[k] = prev[j]["id"]
                    taken.add(j)
        for k in range(len(lead)):
            if cluster_id[k] < 0:
                cluster_id[k] = self._new(parent_of[k], tick, centres[k], float(cells[k]))
            else:
                sp = self.species[cluster_id[k]]
                sp["centre"], sp["cells"], sp["last"] = centres[k].tolist(), float(cells[k]), tick
                sp["history"].append([tick, float(cells[k])])
                if len(sp["history"]) > HISTORY_POINTS:  # bounded memory over very long runs
                    sp["history"] = sp["history"][:-1:2] + sp["history"][-1:]
            self.species[cluster_id[k]]["alive"] = True
        self.last_tick = tick
        return cluster_id[member]

    def tree(self, up_to: int | None = None) -> list[dict]:
        """Species born at or before `up_to` (all if None), with their abundance history."""
        # alive = present at the last classification at or before up_to (classifications happen
        # every few thousand ticks, so a tick in between must not read as "everything extinct")
        last = max((h[0] for sp in self.species for h in sp["history"]
                    if up_to is None or h[0] <= up_to), default=None)
        out = []
        for sp in self.species:
            if up_to is None or sp["born"] <= up_to:
                hist = [h for h in sp["history"] if up_to is None or h[0] <= up_to]
                end = hist[-1][0] if hist else sp["born"]
                out.append({**sp, "history": hist, "last": end, "alive": end == last})
        return out

    def to_json(self) -> str:
        return json.dumps({"threshold": self.threshold, "species": self.species,
                           "last_tick": self.last_tick})

    @classmethod
    def from_json(cls, text: str, sample: int = 10_000, seed: int = 0) -> "SpeciesTracker":
        d = json.loads(text)
        tr = cls(d["threshold"], sample, seed)
        tr.species, tr.last_tick = d["species"], d["last_tick"]
        return tr
