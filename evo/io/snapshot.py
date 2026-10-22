"""Full simulation state <-> numpy arrays (stage 4c, design decisions 5 and 7). Main thread only:
reading or writing Taichi fields is a Taichi call. Derived fields (terrain, streamfunction,
light, temperature, per-tick scratch) are rebuilt from the config, not stored."""
import dataclasses
import json

import numpy as np

from evo.config import ChemConfig, ClimateConfig, Config, LifeConfig, WorldConfig
from evo.world.ledger import Totals

SCHEMA_VERSION = 5  # 2: stage 6 (adhesion gene); 3: 7a (12-input brain, senses); 4: 7b bodies;
# 5: 7c neural genes. Nervous systems are not stored: on restore they regrow from the genes
# (development draws new synapse noise, like a re-run on the GPU: same genes, not the same run)
WORLD_FIELDS = ("chem", "atm", "clim", "acc")
AGENT_FIELDS = ("state", "pos", "body", "energy", "age", "ncell", "uid", "parent", "genome", "free",
                "act", "species", "adapt", "heading", "org", "seg", "metab_frac", "gen")
ORG_FIELDS = ("state", "n", "slot", "suid", "head", "psi", "phase", "turn", "turn_sign", "uid",
              "travel", "free")


def capture(sim) -> dict:
    """Every array needed to continue the run exactly where it is (up to GPU nondeterminism)."""
    w, a = sim.world, sim.life
    out = {f"world.{k}": getattr(w, k).to_numpy() for k in WORLD_FIELDS}
    out.update({f"life.{k}": getattr(a, k).to_numpy() for k in AGENT_FIELDS})
    out.update({f"org.{k}": getattr(a.orgs, k).to_numpy() for k in ORG_FIELDS})
    out["org.top"] = np.array(a.orgs.top[None], np.int64)
    out["org.next_uid"] = np.array(a.orgs.next_uid[None], np.int64)
    out["life.top"] = np.array(a.top[None], np.int64)
    out["life.next_uid"] = np.array(a.next_uid[None], np.int64)
    out["sim.tick"] = np.array(sim.tick, np.int64)
    out["sim.t0"] = np.array(dataclasses.astuple(sim.t0), np.float64)
    out["schema_version"] = np.array(SCHEMA_VERSION, np.int64)
    return out


def upgrade(arrays: dict, sim) -> dict:
    """An older snapshot as the current schema: genes appended since then take their founder
    values (schema 3: no body plan, schema 4: no neurons); schema 3 also gets every slot as a free
    cell and an empty organism table. Lets a recorded world run on with the new abilities possible."""
    version = int(arrays["schema_version"])
    if version not in (3, 4):
        return arrays
    from evo.life.genome import FOUNDER, N_GENES

    out = dict(arrays)
    g = arrays["life.genome"]
    tail = np.broadcast_to(np.array(FOUNDER[g.shape[1]:], np.float32), (len(g), N_GENES - g.shape[1]))
    out["life.genome"] = np.ascontiguousarray(np.concatenate([g, tail], axis=1))
    if version == 3:
        cap = len(g)
        out["life.org"] = np.full(cap, -1, np.int32)
        out["life.seg"] = np.zeros(cap, np.int32)
        out["life.metab_frac"] = np.ones(cap, np.float32)
        out["life.gen"] = np.zeros(cap, np.int32)
        o = sim.life.orgs
        out.update({f"org.{k}": getattr(o, k).to_numpy() for k in ORG_FIELDS})  # fresh, empty
        out["org.top"] = np.array(o.top[None], np.int64)
        out["org.next_uid"] = np.array(o.next_uid[None], np.int64)
    out["schema_version"] = np.array(SCHEMA_VERSION, np.int64)
    return out


def restore(sim, arrays: dict) -> None:
    """Load `capture` output into a Sim built from the same Config (schema 3 is upgraded)."""
    arrays = upgrade(arrays, sim)
    if int(arrays["schema_version"]) != SCHEMA_VERSION:
        raise ValueError(f"snapshot schema {int(arrays['schema_version'])}, expected {SCHEMA_VERSION}")
    w, a = sim.world, sim.life
    for k in WORLD_FIELDS:
        getattr(w, k).from_numpy(arrays[f"world.{k}"])
    for k in AGENT_FIELDS:
        getattr(a, k).from_numpy(arrays[f"life.{k}"])
    for k in ORG_FIELDS:
        getattr(a.orgs, k).from_numpy(arrays[f"org.{k}"])
    a.orgs.top[None] = int(arrays["org.top"])
    a.orgs.dirty.from_numpy((arrays["org.state"] == 1).astype(np.int32))  # nerves regrow
    a.orgs.nn.fill(0)
    a.orgs.bend.fill(0.0)
    a.orgs.bend_prev.fill(0.0)
    a.orgs.next_uid[None] = int(arrays["org.next_uid"])
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
