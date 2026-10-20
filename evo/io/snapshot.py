"""Full simulation state <-> numpy arrays (stage 4c, design decisions 5 and 7). Main thread only:
reading or writing Taichi fields is a Taichi call. Derived fields (terrain, streamfunction,
light, temperature, per-tick scratch) are rebuilt from the config, not stored."""
import dataclasses
import json

import numpy as np

from evo.config import ChemConfig, ClimateConfig, Config, LifeConfig, WorldConfig
from evo.world.ledger import Totals

SCHEMA_VERSION = 3  # 2: stage 6 (adhesion gene); 3: stage 7a (12-input brain, senses)
WORLD_FIELDS = ("chem", "atm", "clim", "acc")
AGENT_FIELDS = ("state", "pos", "body", "energy", "age", "ncell", "uid", "parent", "genome", "free",
                "act", "species", "adapt", "heading")


def capture(sim) -> dict:
    """Every array needed to continue the run exactly where it is (up to GPU nondeterminism)."""
    w, a = sim.world, sim.life
    out = {f"world.{k}": getattr(w, k).to_numpy() for k in WORLD_FIELDS}
    out.update({f"life.{k}": getattr(a, k).to_numpy() for k in AGENT_FIELDS})
    out["life.top"] = np.array(a.top[None], np.int64)
    out["life.next_uid"] = np.array(a.next_uid[None], np.int64)
    out["sim.tick"] = np.array(sim.tick, np.int64)
    out["sim.t0"] = np.array(dataclasses.astuple(sim.t0), np.float64)
    out["schema_version"] = np.array(SCHEMA_VERSION, np.int64)
    return out


def restore(sim, arrays: dict) -> None:
    """Load `capture` output into a Sim built from the same Config."""
    if int(arrays["schema_version"]) != SCHEMA_VERSION:
        raise ValueError(f"snapshot schema {int(arrays['schema_version'])}, expected {SCHEMA_VERSION}")
    w, a = sim.world, sim.life
    for k in WORLD_FIELDS:
        getattr(w, k).from_numpy(arrays[f"world.{k}"])
    for k in AGENT_FIELDS:
        getattr(a, k).from_numpy(arrays[f"life.{k}"])
    a.top[None] = int(arrays["life.top"])
    a.next_uid[None] = int(arrays["life.next_uid"])
    a.want_div.fill(0)
    sim.tick = int(arrays["sim.tick"])
    sim.t0 = Totals(*(float(v) for v in arrays["sim.t0"]))
    w.update_light_temp()


def config_to_json(cfg: Config) -> str:
    d = dataclasses.asdict(cfg)
    d["world"]["init"] = [[int(k), v] for k, v in cfg.world.init.items()]
    return json.dumps(d)


def config_from_json(text: str) -> Config:
    d = json.loads(text)
    d["world"]["init"] = {int(k): v for k, v in d["world"]["init"]}
    for sect in ("chem", "climate"):
        for k, v in d[sect].items():
            if isinstance(v, list):
                d[sect][k] = tuple(v)
    return Config(world=WorldConfig(**d["world"]), chem=ChemConfig(**d["chem"]),
                  climate=ClimateConfig(**d["climate"]), life=LifeConfig(**d["life"]))
