"""Stage 4d: species identity, splitting with parents, extinction, JSON round trip."""
import numpy as np

from evo.analysis.species import SpeciesTracker, distance, traits
from evo.life.genome import (
    COL_R,
    ETA_AER,
    ETA_H2,
    ETA_PHOTO_OX,
    ETA_PRED,
    FOUNDER,
    GENE_RANGE,
    O2_TOL,
    SPEED,
    T_OPT,
)


def _pop(n, **genes):
    g = np.tile(np.array(FOUNDER, np.float32), (n, 1))
    for k, v in genes.items():
        g[:, int(k)] = v
    return g


def test_one_genome_one_species_and_id_persists():
    tr = SpeciesTracker()
    g = _pop(500)
    ids = tr.update(0, g, np.ones(500))
    assert set(ids) == {0}
    ids = tr.update(2000, g, np.ones(500))
    assert set(ids) == {0} and len(tr.species) == 1
    assert tr.species[0]["history"] == [[0, 500.0], [2000, 500.0]]


def test_split_creates_daughter_with_parent_and_marks_extinction():
    tr = SpeciesTracker()
    tr.update(0, _pop(400), np.ones(400))
    # half the population moves far in trait space: an aerobic, O2-tolerant, cold-water
    # photosynthesizer with no H2 use and another lineage colour
    far = _pop(200, **{str(ETA_PHOTO_OX): GENE_RANGE[ETA_PHOTO_OX][1], str(ETA_H2): 0.0,
                       str(T_OPT): 5.0, str(ETA_AER): GENE_RANGE[ETA_AER][1], str(O2_TOL): 1.0,
                       str(COL_R): 0.0})
    ids = tr.update(2000, np.concatenate([_pop(200), far]), np.ones(400))
    assert set(ids[:200]) == {0}
    daughter = int(ids[200])
    assert daughter != 0 and set(ids[200:]) == {daughter}
    assert tr.species[daughter]["parent"] == 0 and tr.species[daughter]["born"] == 2000
    # the ancestor dies out
    tr.update(4000, far, np.ones(200))
    assert not tr.species[0]["alive"] and tr.species[daughter]["alive"]
    assert [s["id"] for s in tr.tree(up_to=0)] == [0]


def test_cells_weight_the_species_size():
    tr = SpeciesTracker()
    tr.update(0, _pop(10), np.full(10, 50.0))
    assert tr.species[0]["cells"] == 500.0


def test_json_round_trip_continues_the_same_tree():
    tr = SpeciesTracker()
    tr.update(0, _pop(100), np.ones(100))
    back = SpeciesTracker.from_json(tr.to_json())
    ids = back.update(2000, _pop(100), np.ones(100))
    assert set(ids) == {0} and len(back.species) == 1


def test_tiny_outlier_clusters_join_the_nearest_species():
    tr = SpeciesTracker()
    g = _pop(1000)
    g[0, ETA_PHOTO_OX], g[0, ETA_AER], g[0, O2_TOL], g[0, COL_R], g[0, ETA_H2] = 0.95, 0.95, 1, 0, 0
    ids = tr.update(0, g, np.ones(1000))  # one odd cell is 0.1% of all cells
    assert set(ids) == {0} and len(tr.species) == 1


def test_rewound_tree_keeps_living_species_alive_between_classifications():
    # 4d review: at a tick between two classifications every species read as extinct
    tr = SpeciesTracker()
    tr.update(0, _pop(100), np.ones(100))
    tr.update(2000, _pop(100), np.ones(100))
    assert all(s["alive"] for s in tr.tree(up_to=2100))
    assert all(s["alive"] for s in tr.tree(up_to=0))


def test_small_but_distant_community_stays_a_species():
    # 4d review: a deep-vent community under a bloom is tiny but metabolically different
    tr = SpeciesTracker()
    far = _pop(3, **{str(ETA_PHOTO_OX): GENE_RANGE[ETA_PHOTO_OX][1], str(ETA_H2): 0.0,
                      str(T_OPT): 5.0, str(ETA_AER): GENE_RANGE[ETA_AER][1], str(O2_TOL): 1.0,
                      str(COL_R): 0.0, str(ETA_PRED): GENE_RANGE[ETA_PRED][1], str(SPEED): 1.0})
    assert distance(traits(far[:1]), traits(_pop(1)))[0] > SpeciesTracker.KEEP * 0.2
    ids = tr.update(0, np.concatenate([_pop(1000), far]), np.ones(1003))  # 0.3% of cells
    assert len(set(ids)) == 2 and ids[-1] != ids[0]
