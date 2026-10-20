"""Stage 4b: the documentary view and the zoom-view cell records come from simulated values."""
import numpy as np

from evo.life.agents import ALIVE
from evo.life.genome import ETA_H2, ETA_PHOTO_OX, ETA_PRED
from evo.sim import Sim
from evo.ui.render import CELL_FIELDS, KIND_CHEMO, KIND_PHOTO_OX, KIND_PRED, Renderer
from evo.world.fields import CL_ICE


def _sim(cfg):
    sim = Sim(cfg)
    sim.world.update_light_temp()
    return sim, Renderer(sim.world, sim.life, max_cells=64)


def test_natural_view_is_finite_and_lit_from_the_light_field(small_cfg):
    sim, r = _sim(small_cfg)
    r.draw_natural()
    img = r.rgba.to_numpy()
    assert np.isfinite(img).all() and img.min() >= 0 and img.max() <= 1
    water = sim.world.water_np  # (w, d)
    x = int(np.argmax(water.sum(axis=1)))  # a deep column
    top, deep = img[1, x, :3].sum(), img[water[x].sum() - 2, x, :3].sum()
    assert top > deep  # sunlit surface brighter than the dark deep (unless a vent glows there)


def test_ice_whitens_the_surface(small_cfg):
    sim, r = _sim(small_cfg)
    r.draw_natural()
    open_sea = r.rgba.to_numpy()[0, :, :3].mean()
    sim.world.clim[CL_ICE] = 1.0
    r.draw_natural()
    assert r.rgba.to_numpy()[0, :, :3].mean() > open_sea + 0.2


def test_viewport_extracts_exactly_the_cells_inside(small_cfg):
    sim, r = _sim(small_cfg)
    a = sim.life
    rows, total = r.viewport_cells(0, 0, small_cfg.world.width, small_cfg.world.depth)
    assert total == a.count() and 0 < len(rows) <= min(total, 64)
    pos = a.pos.to_numpy()[a.state.to_numpy() == ALIVE]
    inside = ((pos[:, 0] >= 40) & (pos[:, 0] < 90) & (pos[:, 1] >= 30) & (pos[:, 1] < 60)).sum()
    _, total2 = r.viewport_cells(40, 30, 90, 60)
    assert total2 == inside
    assert rows.shape[1] == len(CELL_FIELDS)


def test_cell_kind_and_pigment_follow_metabolism(small_cfg):
    sim, r = _sim(small_cfg)
    a = sim.life
    state = np.zeros(a.cap, np.int32)
    state[:3] = ALIVE
    a.state.from_numpy(state)
    genome = a.genome.to_numpy()
    genome[:3, [ETA_H2, ETA_PHOTO_OX, ETA_PRED]] = [[0.9, 0, 0], [0.1, 0.9, 0], [0.1, 0, 0.9]]
    a.genome.from_numpy(genome)
    for i in range(3):
        a.pos[i] = [60.5, 10.5]
    rows, _ = r.viewport_cells(60, 10, 61, 11)
    by_kind = {int(k): row for row, k in zip(rows, rows[:, CELL_FIELDS.index("kind")])}
    assert set(by_kind) == {KIND_CHEMO, KIND_PHOTO_OX, KIND_PRED}
    g = CELL_FIELDS.index("g")
    assert by_kind[KIND_PHOTO_OX][g] > by_kind[KIND_PHOTO_OX][g - 1]  # chlorophyll: green > red


def test_crowded_view_keeps_the_same_sample_every_frame(small_cfg):
    # 4b review: a sample that changed between frames looked like deaths and births
    from evo.ui.render import in_sample_np

    sim, r = _sim(small_cfg)  # max_cells = 64
    a = sim.life
    n = 200
    state = np.zeros(a.cap, np.int32)
    state[:n] = ALIVE
    a.state.from_numpy(state)
    pos = a.pos.to_numpy()
    pos[:n] = [60.5, 10.5]
    a.pos.from_numpy(pos)
    uid = a.uid.to_numpy()
    uid[:n] = np.arange(1000, 1000 + n)
    a.uid.from_numpy(uid)
    first, total = r.viewport_cells(60, 10, 61, 11)
    second, _ = r.viewport_cells(60, 10, 61, 11)
    assert total == n and 0 < len(first) <= 64
    uid_col = CELL_FIELDS.index("uid")
    assert set(first[:, uid_col]) == set(second[:, uid_col])
    # the GPU sample is exactly the numpy rule used for recorded frames
    expect = {u for u in range(1000, 1000 + n) if in_sample_np(u, n, 64)}
    assert set(first[:, uid_col].astype(int)) == expect
