import pytest
import taichi as ti

from evo.config import Config, D, M, O


@pytest.fixture(scope="session", autouse=True)
def taichi_cpu():
    # single thread: deterministic atomics (S1)
    ti.init(arch=ti.cpu, cpu_max_num_threads=1, random_seed=0, log_level=ti.WARN)


@pytest.fixture
def small_cfg() -> Config:
    cfg = Config()
    cfg.world.width, cfg.world.depth, cfg.world.n_vents = 128, 64, 2
    cfg.life.capacity = 8192
    cfg.life.org_capacity = 256  # organism (and nervous system) table: ~450 MB at the default
    return cfg


def seed_reactive(cfg: Config) -> Config:
    """Seed O2, detritus and CH4 so every abiotic reaction, decay path, burial and the
    atmospheric CH4/O2 sinks actually run (the default anoxic, lifeless world leaves them idle)."""
    cfg.world.init.update({O: 0.3, D: 0.5, M: 0.1})
    cfg.climate.init_atm = (3.0e5, 1.0e4, 1.0e3, 0.0)
    return cfg


@pytest.fixture
def seeded_cfg(small_cfg) -> Config:
    return seed_reactive(small_cfg)
