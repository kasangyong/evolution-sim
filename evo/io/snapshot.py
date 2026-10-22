"""Full simulation state <-> numpy arrays (stage 4c, design decisions 5 and 7). Main thread only:
reading or writing Taichi fields is a Taichi call. Derived fields (terrain, streamfunction,
light, temperature, per-tick scratch) are rebuilt from the config, not stored."""
import dataclasses
import json

import numpy as np

from evo.config import ChemConfig, ClimateConfig, Config, LifeConfig, WorldConfig
from evo.world.ledger import Totals

SCHEMA_VERSION = 7  # 2: stage 6 (adhesion gene); 3: 7a (12-input brain, senses); 4: 7b bodies;
# 5: 7c neural genes; 6: organisms and their nervous systems stored, compacted (living organisms
# and the pool neurons they use only), so a rewind keeps each circuit; 7: 7d plasticity genes and
# what each synapse has learned (dw) and its habituation (res)
WORLD_FIELDS = ("chem", "atm", "clim", "acc")
AGENT_FIELDS = ("state", "pos", "body", "energy", "age", "ncell", "uid", "parent", "genome", "free",
                "act", "species", "adapt", "heading", "org", "seg", "metab_frac", "gen")
# per-organism rows (schema 6 keeps the rows of living organisms only); free-lists are rebuilt
ORG_FIELDS = ("state", "n", "slot", "suid", "head", "psi", "phase", "turn", "turn_sign", "uid",
              "travel", "dirty", "nn", "nidx", "wsum", "bend", "bend_prev", "joint", "wired_n",
              "born", "last_wire", "nplastic", "mod_avg")
OLD_ORG_FIELDS = ORG_FIELDS[:11]  # what schemas 4-5 stored (in full)
NEURON_FIELDS = ("ntype", "nseg", "v", "rate", "nsyn", "pre", "w", "dw", "res")
NEURON_FIELDS_6 = NEURON_FIELDS[:7]  # schema 6: no learning yet


def capture(sim) -> dict:
    """Every array needed to continue the run exactly where it is (up to GPU nondeterminism)."""
    w, a = sim.world, sim.life
    out = {f"world.{k}": getattr(w, k).to_numpy() for k in WORLD_FIELDS}
    out.update({f"life.{k}": getattr(a, k).to_numpy() for k in AGENT_FIELDS})
    o = a.orgs
    ids = np.nonzero(o.state.to_numpy() == 1)[0]
    out["org.ids"] = ids
    out.update({f"org.{k}": getattr(o, k).to_numpy()[ids] for k in ORG_FIELDS})
    nn, nidx = out["org.nn"], out["org.nidx"]
    used = np.concatenate([nidx[r, :nn[r]] for r in range(len(ids))]) if len(ids) else np.zeros(
        0, np.int32)
    out["neu.ids"] = used.astype(np.int32)
    out.update({f"neu.{k}": getattr(o, k).to_numpy()[used] for k in NEURON_FIELDS})
    out["org.next_uid"] = np.array(o.next_uid[None], np.int64)
    out["org.clock"] = np.array(o.clock[None], np.int64)
    out["life.top"] = np.array(a.top[None], np.int64)
    out["life.next_uid"] = np.array(a.next_uid[None], np.int64)
    out["sim.tick"] = np.array(sim.tick, np.int64)
    out["sim.t0"] = np.array(dataclasses.astuple(sim.t0), np.float64)
    out["schema_version"] = np.array(SCHEMA_VERSION, np.int64)
    return out


def upgrade(arrays: dict, sim) -> dict:
    """An older snapshot as the current schema: genes appended since then take their founder
    values (schema 3: no body plan, schema 4: no neurons); schema 3 also gets every slot as a free
    cell and no organisms; schemas 4-5 keep their organisms, whose nerves regrow from the genes.
    Lets a recorded world run on with the new abilities possible."""
    version = int(arrays["schema_version"])
    if version == 6:  # circuits kept; nothing learned yet, synapses fresh
        from evo.life.genome import FOUNDER, N_GENES

        out = dict(arrays)
        g = arrays["life.genome"]
        tail = np.broadcast_to(np.array(FOUNDER[g.shape[1]:], np.float32),
                               (len(g), N_GENES - g.shape[1]))
        out["life.genome"] = np.ascontiguousarray(np.concatenate([g, tail], axis=1))
        shape = arrays["neu.w"].shape
        out["neu.dw"] = np.zeros(shape, np.float32)
        out["neu.res"] = np.ones(shape, np.float32)
        out["org.wired_n"] = arrays["org.n"].astype(np.int32)  # circuits grown for these bodies
        for k in ("born", "last_wire", "nplastic"):  # no learning genes before schema 7
            out[f"org.{k}"] = np.zeros(len(arrays["org.ids"]), np.int32)
        out["org.mod_avg"] = np.zeros(len(arrays["org.ids"]), np.float32)
        out["schema_version"] = np.array(SCHEMA_VERSION, np.int64)
        return out
    if version not in (3, 4, 5):
        return arrays
    from evo.life.genome import FOUNDER, N_GENES

    out = dict(arrays)
    g = arrays["life.genome"]
    tail = np.broadcast_to(np.array(FOUNDER[g.shape[1]:], np.float32), (len(g), N_GENES - g.shape[1]))
    out["life.genome"] = np.ascontiguousarray(np.concatenate([g, tail], axis=1))
    o = sim.life.orgs
    if version == 3:
        cap = len(g)
        out["life.org"] = np.full(cap, -1, np.int32)
        out["life.seg"] = np.zeros(cap, np.int32)
        out["life.metab_frac"] = np.ones(cap, np.float32)
        out["life.gen"] = np.zeros(cap, np.int32)
        ids = np.zeros(0, np.int64)
        out["org.next_uid"] = np.array(o.next_uid[None], np.int64)
    else:
        ids = np.nonzero(arrays["org.state"] == 1)[0]
    rows = {k: arrays[f"org.{k}"][ids] for k in OLD_ORG_FIELDS} if version != 3 else {}
    out["org.ids"] = ids
    for k in ORG_FIELDS:
        full = getattr(o, k).to_numpy()
        out[f"org.{k}"] = rows.get(k, np.zeros((len(ids), *full.shape[1:]), full.dtype))
    out["org.dirty"] = np.ones(len(ids), np.int32)  # their nervous systems regrow
    out["neu.ids"] = np.zeros(0, np.int32)
    out.update({f"neu.{k}": getattr(o, k).to_numpy()[:0] for k in NEURON_FIELDS})
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
    o = a.orgs
    for prefix, names, cap in (("org", ORG_FIELDS, o.cap), ("neu", NEURON_FIELDS, o.ncap)):
        ids = arrays[f"{prefix}.ids"]
        for k in names:  # scatter the stored rows into empty tables
            full = np.zeros_like(getattr(o, k).to_numpy())
            full[ids] = arrays[f"{prefix}.{k}"]
            getattr(o, k).from_numpy(full)
        free = np.setdiff1d(np.arange(cap), ids)[::-1].astype(np.int32)  # the rest is free
        table = np.zeros(cap, np.int32)
        table[:len(free)] = free
        (o.free if prefix == "org" else o.nfree).from_numpy(table)
        (o.top if prefix == "org" else o.ntop)[None] = len(free)
    o.next_uid[None] = int(arrays["org.next_uid"])
    o.clock[None] = int(arrays["org.clock"]) if "org.clock" in arrays else 0
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
