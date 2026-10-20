"""Stage 4c: snapshots restore the exact state; history keeps branches apart and stays in budget."""
import numpy as np

from evo.config import Config
from evo.io.history import History, pack_cells, unpack_cells
from evo.io.snapshot import capture, config_from_json, config_to_json, restore
from evo.sim import Sim


def test_snapshot_round_trip_is_bit_exact(small_cfg):
    sim = Sim(small_cfg)
    sim.step(300)
    saved = capture(sim)
    sim.step(200)  # move on, then go back
    restore(sim, saved)
    again = capture(sim)
    assert saved.keys() == again.keys()
    for k in saved:
        assert np.array_equal(saved[k], again[k]), k
    assert sim.tick == 300 and sim.budget_error().worst() < 1e-9
    sim.step(100)  # and it runs on from there
    assert sim.tick == 400 and sim.budget_error().worst() < 1e-9


def test_config_json_round_trip():
    cfg = Config()
    cfg.world.seed = 7
    back = config_from_json(config_to_json(cfg))
    assert back == cfg


def test_cell_packing_keeps_what_the_viewer_draws():
    rng = np.random.default_rng(0)
    rows = np.zeros((50, 15), np.float32)
    rows[:, 0], rows[:, 1] = rng.uniform(0, 1024, 50), rng.uniform(0, 384, 50)
    rows[:, 2], rows[:, 3] = rng.uniform(0.005, 0.2, 50), rng.integers(1, 1000, 50)
    rows[:, 4:7] = rng.uniform(0, 1, (50, 3))
    rows[:, 7], rows[:, 8] = rng.integers(0, 2**24, 50), rng.integers(0, 2**24, 50)
    rows[:, 9], rows[:, 10:12] = rng.integers(0, 4, 50), rng.uniform(-1, 1, (50, 2))
    rows[:, 12], rows[:, 13] = rng.integers(0, 500, 50), rng.integers(0, 4, 50)
    rows[:, 14] = rng.integers(0, 2, 50)
    back = unpack_cells(pack_cells(rows))
    assert np.allclose(back[:, :2], rows[:, :2], atol=1 / 64)
    assert np.allclose(back[:, 2], rows[:, 2], rtol=1e-3)
    assert np.array_equal(back[:, [3, 7, 8, 9, 12, 13, 14]], rows[:, [3, 7, 8, 9, 12, 13, 14]])
    assert np.allclose(back[:, 4:7], rows[:, 4:7], atol=1 / 255)


def _fake_frame(h: History, tick: int):
    h.record_frame(tick, np.zeros((4, 4, 3), np.uint8), np.zeros((3, 15), np.float32), {"tick": tick})


def test_branch_lineage_excludes_what_came_after_the_fork(tmp_path):
    h = History(tmp_path, Config())
    for t in range(0, 1000, 100):
        _fake_frame(h, t)
    for t in (0, 500):
        h.record_snapshot(t, {"x": np.array(t)})
    h.flush()
    fork = h.nearest("snap", 650)
    assert fork == (500, 0)
    b = h.new_branch(fork[1], fork[0])
    _fake_frame(h, 600)
    h.flush()
    assert h.frame_ticks() == [0, 100, 200, 300, 400, 500, 600]  # parent's 600..900 not included
    assert h.nearest("frame", 650, branch=b) == (600, b)
    assert h.nearest("frame", 650, branch=0) == (600, 0)  # the original history is untouched
    assert h.frame_ticks(branch=0)[-1] == 900


def test_thinning_keeps_budget_first_record_and_fork_points(tmp_path):
    h = History(tmp_path, Config(), budget_bytes=1, keep_recent=300)
    h.record_snapshot(200, {"x": np.zeros(1000)})
    h.flush()
    h.new_branch(0, 200)
    h.current = 0
    for t in range(0, 2000, 100):
        _fake_frame(h, t)
    h.flush()
    ticks = h.frame_ticks()
    assert ticks[0] == 0 and ticks[-1] == 1900  # first and newest kept
    assert len(ticks) < 20  # old records thinned
    assert 200 in h.snap_ticks()  # a branch forks there


def test_writer_survives_a_file_held_open_by_a_reader(tmp_path):
    # 4c review: thinning a file another thread has open (Windows refuses the delete) once
    # killed the writer, the queue filled and the simulation blocked
    import threading

    h = History(tmp_path, Config(), budget_bytes=10**12, keep_recent=300)
    for t in range(0, 1000, 100):
        _fake_frame(h, t)
    h.flush()
    def flush_within(seconds=10):  # a dead writer would make flush() wait forever
        t = threading.Thread(target=h.flush, daemon=True)
        t.start()
        t.join(seconds)
        assert not t.is_alive(), "writer thread stopped"

    with open(h.path("frame", 0, 100), "rb"):  # held open, like np.load in an HTTP thread
        h.budget = 1  # over budget: the next write thins tick 100
        _fake_frame(h, 1000)
        flush_within()
    done = threading.Event()

    def put_many():
        for t in range(1100, 1100 + 100 * 80, 100):  # more than the queue holds
            _fake_frame(h, t)
        done.set()

    threading.Thread(target=put_many, daemon=True).start()
    assert done.wait(10)  # producers never block
    flush_within()
    assert h.frame_ticks()[-1] >= 1000  # the writer kept going
