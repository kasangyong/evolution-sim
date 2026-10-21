"""Live browser viewer. The sim (all Taichi calls) stays on the main thread; an HTTP thread only
serves the latest PNG frame and stats that the main loop publishes, and queues button commands."""
import json
import queue
import struct
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import taichi as ti

from evo.analysis.events import EventDetector, Summary
from evo.analysis.species import SpeciesTracker
from evo.config import ATM_C, ATM_M, ATM_O, SPECIES_NAMES
from evo.io.history import History, unpack_cells
from evo.io.snapshot import capture, restore
from evo.life.agents import ACT_ON, ALIVE
from evo.life.genome import (
    BRAIN,
    ETA_AER,
    ETA_H2,
    ETA_PHOTO_AN,
    ETA_PHOTO_OX,
    ETA_PRED,
    GENE_RANGE,
    N_HID,
    N_IN,
    SPEED,
    T_OPT,
    TRAIT_GENES,
)
from evo.sim import Sim
from evo.ui.app import layer_vmax
from evo.ui.render import (
    CELL_FIELDS,
    NATURAL_LAYER,
    TEMP_LAYER,
    Renderer,
    closeup,
    in_sample_np,
)
from evo.world.fields import BUR_ORG, CL_ICE, CL_TS

LAYERS = SPECIES_NAMES + ["Light", "Temperature", "Natural"]
FRAME_EVERY_S = 0.25
CELLS_EVERY_S = 0.1  # zoom view refresh
MAX_CELLS = 20000  # cells sent per zoom frame (Renderer max_cells)
MAX_VIEW_AREA = 400 * 160  # grid cells; wider views show the whole-ocean image only
STEP_BUDGET_S = 0.2
HISTORY_EVERY = 500
SPECIES_EVERY = 2000  # top-level design decision 6
PAGE = Path(__file__).with_name("web.html")


def png(rgb: np.ndarray) -> bytes:
    h, w, _ = rgb.shape
    raw = np.concatenate([np.zeros((h, 1), np.uint8), rgb.reshape(h, w * 3)], axis=1).tobytes()

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 1)) + chunk(b"IEND", b""))


class Shared:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = b""
        self.stats = b"{}"
        self.cells = b""
        self.tracker = SpeciesTracker()
        self.events: list[dict] = []  # event journal entries, newest last
        self.commands: queue.Queue = queue.Queue()


TREE_BIGGEST, TREE_ALIVE = 40, 20  # lanes the browser can show; long runs make thousands of species


def major_species(tree: list[dict]) -> list[dict]:
    """The species worth a lane: the TREE_BIGGEST largest ever (peak cells), the TREE_ALIVE largest
    living ones, and every ancestor of those so the branches connect."""
    by_id = {sp["id"]: sp for sp in tree}
    peak = {sp["id"]: max((h[1] for h in sp["history"]), default=0.0) for sp in tree}
    keep = set(sorted(peak, key=peak.get, reverse=True)[:TREE_BIGGEST])
    alive = sorted((sp for sp in tree if sp["alive"]), key=lambda sp: -sp["cells"])
    keep |= {sp["id"] for sp in alive[:TREE_ALIVE]}
    for sid in list(keep):
        p = by_id[sid]["parent"]
        while p is not None and p in by_id and p not in keep:
            keep.add(p)
            p = by_id[p]["parent"]
    return [sp for sp in tree if sp["id"] in keep]


def species_view(tracker: SpeciesTracker, up_to: int | None) -> dict:
    """Tree for the browser: the major species with centre traits in real units, a pigment colour
    and a thinned history, plus how many smaller species were left out."""
    lo = np.array([GENE_RANGE[g][0] for g in TRAIT_GENES])
    span = np.array([GENE_RANGE[g][1] - GENE_RANGE[g][0] for g in TRAIT_GENES])
    tree = tracker.tree(up_to)
    shown = major_species(tree)
    out = []
    for sp in shown:
        g = lo + np.array(sp["centre"]) * span
        w = {"ox": g[ETA_PHOTO_OX], "an": g[ETA_PHOTO_AN], "pr": g[ETA_PRED], "ch": 0.3 + g[ETA_H2]}
        col = (w["ox"] * np.array([0.22, 0.62, 0.26]) + w["an"] * np.array([0.55, 0.30, 0.42])
               + w["pr"] * np.array([0.78, 0.60, 0.40]) + w["ch"] * np.array([0.80, 0.82, 0.78]))
        col = col / sum(w.values())
        hist = sp["history"]
        step = max(1, len(hist) // 200)
        out.append({"id": sp["id"], "parent": sp["parent"], "born": sp["born"], "last": sp["last"],
                    "alive": sp["alive"],
                    "cells": hist[-1][1] if hist else 0.0, "history": hist[::step] + hist[-1:],
                    "color": [round(float(c), 3) for c in col],
                    "traits": {"eta_h2": float(g[ETA_H2]), "eta_photo_ox": float(g[ETA_PHOTO_OX]),
                               "eta_photo_an": float(g[ETA_PHOTO_AN]), "eta_aer": float(g[ETA_AER]),
                               "eta_pred": float(g[ETA_PRED]), "speed": float(g[SPEED]),
                               "t_opt": float(g[T_OPT]),
                               "adhesion": float(g[len(TRAIT_GENES) - 1])}})
    return {"species": out, "total": len(tree), "alive": sum(sp["alive"] for sp in tree)}


W2_TUMBLE = np.array([BRAIN + N_IN * N_HID + N_HID + h for h in range(N_HID)])
N_SENSES = N_IN - 8  # inputs 8.. are the adapted senses: 0 H2, 1 sulfide, 2 O2, 3 light


def sense_bias(g, s):
    """Per genome: +1 if its actual brain tumbles when sense s falls (input -1, the rest 0) and
    not when it rises (+1), -1 for the reverse wiring, else 0. Runs the real tanh net with the
    hidden biases and the tumble threshold, not a linearisation."""
    w1 = g[:, BRAIN:BRAIN + N_IN * N_HID].reshape(-1, N_HID, N_IN)[:, :, 8 + s]
    b1 = g[:, BRAIN + N_IN * N_HID:BRAIN + N_IN * N_HID + N_HID]
    worse, better = (np.tanh((g[:, W2_TUMBLE] * np.tanh(b1 + w1 * x)).sum(axis=1)) > ACT_ON
                     for x in (-1.0, 1.0))
    return (worse & ~better).astype(np.int8) - (better & ~worse).astype(np.int8)


def chemotaxis_bias(g, w=None):
    """(food, control, tuned): net shares of swimming cells wired to tumble when a sense falls: food is
    each cell's own food sense (chemosynthesis: H2, photosynthesis: light), control the largest
    |share| over its other three senses. All cells descend from one founder, so a lineage that
    spreads carries its wiring on every sense alike (measured, stage 7a evaluation); only a food
    share well above the controls hints at selection. Shares are weighted by w (cells per slot). tuned marks the swimming cells wired to
    tumble when their food falls."""
    photo = g[:, ETA_PHOTO_OX] + g[:, ETA_PHOTO_AN] > g[:, ETA_H2]
    food_s = np.where(photo, 3, 0)
    swims = g[:, SPEED] > 0.1
    bias = np.stack([sense_bias(g, s) for s in range(N_SENSES)], axis=1) * swims[:, None]
    rows = np.arange(len(g))
    w = np.ones(len(g)) if w is None else w
    food = float(np.average(bias[rows, food_s], weights=w)) if len(g) else 0.0
    others = np.array([[s for s in range(N_SENSES) if s != f] for f in range(N_SENSES)])[food_s]
    control = max((abs(float(np.average(bias[rows, others[:, k]], weights=w)))
                   for k in range(N_SENSES - 1)),
                  default=0.0) if len(g) else 0.0
    return food, control, bias[rows, food_s] > 0


def cells_blob(tick: int, rows: np.ndarray, total: int) -> bytes:
    return struct.pack("<qii", tick, total, len(rows)) + rows.astype("<f4").tobytes()


def make_handler(shared: Shared, history: History):
    cache: dict = {}  # last history frame read from disk, shared by the HTTP threads
    cache_lock = threading.Lock()

    def frame_at(t: int):
        hit = history.nearest("frame", t)
        if hit is None:
            return None
        with cache_lock:
            if cache.get("key") != hit:
                cache["key"], cache["data"] = hit, history.load("frame", hit[1], hit[0])
            return hit[0], cache["data"]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, body: bytes, ctype: str):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            path, q = url.path, parse_qs(url.query)
            if path.startswith("/history/"):
                self.history_get(path, q)
                return
            if path == "/events.json":
                with shared.lock:
                    body = json.dumps(shared.events[-300:]).encode()
                self.send(body, "application/json")
                return
            if path.startswith("/events/") and path.endswith(".png"):
                f = history.root / "events" / Path(path).name
                if f.is_file():
                    self.send(f.read_bytes(), "image/png")
                else:
                    self.send_error(404)
                return
            if path == "/species.json":
                try:
                    up_to = int(q["t"][0]) if "t" in q else None
                except ValueError:
                    up_to = None
                with shared.lock:
                    body = json.dumps(species_view(shared.tracker, up_to)).encode()
                self.send(body, "application/json")
                return
            if path == "/":
                self.send(PAGE.read_bytes(), "text/html; charset=utf-8")
            elif path == "/frame.png":
                with shared.lock:
                    body = shared.frame
                self.send(body, "image/png")
            elif path == "/stats.json":
                with shared.lock:
                    body = shared.stats
                self.send(body, "application/json")
            elif path == "/cells.bin":
                with shared.lock:
                    body = shared.cells
                self.send(body, "application/octet-stream")
            else:
                self.send_error(404)

        def history_get(self, path: str, q: dict):
            if path == "/history/index.json":
                body = {"current": history.current, "branches": history.branches,
                        "frames": history.frame_ticks(),
                        "snaps": history.snap_ticks(),
                        "used": history.used_bytes(), "budget": history.budget}
                self.send(json.dumps(body).encode(), "application/json")
                return
            try:
                got = frame_at(int(q.get("t", ["0"])[0]))
            except (ValueError, OSError):
                got = None
            if got is None:
                self.send_error(404)
                return
            tick, data = got
            if path == "/history/frame.png":
                self.send(png(data["rgb"]), "image/png")
            elif path == "/history/stats.json":
                self.send(data["stats"].tobytes(), "application/json")
            elif path == "/history/cells.bin":
                rows = unpack_cells(data["cells"])
                try:
                    x0, y0, x1, y1 = (float(v) for v in q["v"][0].split(","))
                    m = (rows[:, 0] >= x0) & (rows[:, 0] < x1) & (rows[:, 1] >= y0) & (rows[:, 1] < y1)
                    rows = rows[m]
                except (KeyError, ValueError):
                    pass
                total = len(rows)
                if total > MAX_CELLS:  # same stable uid-hash sample as the live view
                    rows = rows[in_sample_np(rows[:, 7], total, MAX_CELLS)][:MAX_CELLS]
                self.send(cells_blob(tick, rows, total), "application/octet-stream")
            else:
                self.send_error(404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            try:
                cmd = json.loads(self.rfile.read(min(n, 4096)) or b"{}")
            except ValueError:
                cmd = None
            if isinstance(cmd, dict):
                shared.commands.put(cmd)
            self.send(b"{}", "application/json")

    return Handler


def life_summary(sim: Sim) -> dict:
    a, w = sim.life, sim.world
    alive = a.state.to_numpy() == ALIVE
    if not alive.any():
        return {"pop": 0, "photic": 0}
    pos = a.pos.to_numpy()[alive].astype(int)
    g = a.genome.to_numpy()[alive]
    photic = int((w.light.to_numpy()[pos[:, 0], pos[:, 1]] > 0.01).sum())
    mean = g.mean(axis=0)
    return {"pop": int(alive.sum()), "photic": photic,
            "cells": float(a.ncell.to_numpy()[alive].sum()),
            "eta_h2": float(mean[ETA_H2]), "eta_photo_an": float(mean[ETA_PHOTO_AN]),
            "eta_photo_ox": float(mean[ETA_PHOTO_OX]), "eta_aer": float(mean[ETA_AER]),
            "t_opt": float(mean[T_OPT]), "eta_pred": float(mean[ETA_PRED]),
            "speed": float(mean[SPEED]), "biomass": float(a.body.to_numpy()[alive].sum())}


def serve(sim: Sim, gpu_name: str, host: str, port: int, history_root: Path, budget_gb: float = 20.0):
    w = sim.world
    renderer = Renderer(w, sim.life, max_cells=MAX_CELLS)
    recorder = Renderer(w, sim.life, max_cells=sim.life.cap)  # whole-world cell records
    history = History(history_root, sim.cfg, budget_bytes=int(budget_gb * 1024**3))
    print(f"history: {history_root}", flush=True)
    shared = Shared()
    server = ThreadingHTTPServer((host, port), make_handler(shared, history))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"live viewer: http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}",
          flush=True)

    state = {"paused": False, "speed": 0, "layer": NATURAL_LAYER, "show_life": True, "view": None}
    notice = ""  # last problem to show the viewer (e.g. a failed re-run)

    def vmax_for(layer: int) -> float:
        return 1.0 if layer == NATURAL_LAYER else layer_vmax(sim, layer)

    vmax = vmax_for(state["layer"])
    history_list: list[dict] = []
    life = life_summary(sim)
    err = sim.budget_error()
    tps, last_frame, last_cells, last_hist = 0.0, 0.0, 0.0, -HISTORY_EVERY
    hist_every = HISTORY_EVERY  # doubles whenever the chart history is halved: even spacing
    next_rec, next_snap, next_species = sim.tick, sim.tick, sim.tick

    detector = EventDetector()
    (history.root / "events").mkdir(exist_ok=True)

    def save_event(ev):
        """Capture the moment: the whole ocean, plus a close-up when the event has a place. A
        failed capture (disk full, ...) is logged and skipped: the world keeps running."""
        try:
            capture_event(ev)
        except Exception as e:  # noqa: BLE001 - the journal must never stop the simulation
            print(f"event capture failed at tick {ev.tick}: {e!r}", flush=True)

    def capture_event(ev):
        recorder.draw_natural()
        rgb = (np.clip(recorder.rgba.to_numpy()[:, :, :3], 0, 1) * 255).astype(np.uint8)
        stem = f"{ev.tick:010d}_{ev.kind}"
        (history.root / "events" / f"{stem}.png").write_bytes(png(rgb))
        zoom = None
        if ev.where is not None:
            cx, cy = int(ev.where[0]), int(ev.where[1])
            x0, y0 = max(min(cx - 48, w.w - 96), 0), max(min(cy - 18, w.d - 36), 0)
            rows, _ = recorder.viewport_cells(x0, y0, x0 + 96, y0 + 36)
            crop = closeup(np.ascontiguousarray(rgb[y0:y0 + 36, x0:x0 + 96]), rows, x0, y0, 8)
            zoom = f"{stem}_zoom.png"
            (history.root / "events" / zoom).write_bytes(png(crop))
        entry = {"tick": ev.tick, "kind": ev.kind, "title": ev.title, "detail": ev.detail,
                 "numbers": ev.numbers, "where": ev.where, "image": f"{stem}.png", "zoom": zoom,
                 "branch": history.current}
        with (history.root / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        with shared.lock:
            shared.events.append(entry)

    def classify():
        a = sim.life
        alive = a.state.to_numpy() == ALIVE
        g, n = a.genome.to_numpy()[alive], a.ncell.to_numpy()[alive]
        with shared.lock:
            ids = shared.tracker.update(sim.tick, g, n)
        full = np.zeros(a.cap, np.int32)
        full[alive] = ids
        a.species.from_numpy(full)
        if not alive.any():
            return
        # event journal: summarise the world and let the detectors look
        pos = a.pos.to_numpy()[alive]
        act = a.act.to_numpy()[alive]
        lit_mask = w.light.to_numpy()[pos[:, 0].astype(int), pos[:, 1].astype(int)] > 0.01
        lit = int(lit_mask.sum())

        def densest(mask):
            """Centre of the 32 × 12-cell block holding most of the masked slots."""
            if not mask.any():
                return None
            hist, xe, ye = np.histogram2d(pos[mask, 0], pos[mask, 1], bins=(w.w // 32, w.d // 12),
                                          range=((0, w.w), (0, w.d)))
            i, j = np.unravel_index(np.argmax(hist), hist.shape)
            return (float((xe[i] + xe[i + 1]) / 2), float((ye[j] + ye[j + 1]) / 2))
        names = ("수소 화학합성", "산소 광합성", "비산소 광합성", "포식")
        species = {}
        for sid in np.unique(ids):
            m = ids == sid
            mean = g[m][:, [ETA_H2, ETA_PHOTO_OX, ETA_PHOTO_AN, ETA_PRED]].mean(axis=0)
            # densest block, not the mean: a species spread over several vents averages to water
            species[int(sid)] = (float(n[m].sum()), densest(m), names[int(np.argmax(mean))])
        taxis_food, taxis_control, tuned = chemotaxis_bias(g, n)  # cell-weighted, as taxis_check
        summary = Summary(
            tick=sim.tick, cells=float(n.sum()), slots=int(alive.sum()), lit_slots=lit,
            o2=float(w.atm[ATM_O]), ch4=float(w.atm[ATM_M]), co2=float(w.atm[ATM_C]),
            ts=float(w.clim[CL_TS]), ice=float(w.clim[CL_ICE]),
            pred_frac=float(((g[:, ETA_PRED] > 0.2) & (act[:, 2] > 0.5)).mean()),  # hunting now
            colony_frac=float(a.colonies()[alive].mean()),
            swim_frac=float(((g[:, SPEED] > 0.1) & (act[:, 1] > 0.3)).mean()),
            sense_frac=max(taxis_food - taxis_control, 0.0),
            species=species,
            places={"bloom": densest(lit_mask),
                    "predation": densest((g[:, ETA_PRED] > 0.2) & (act[:, 2] > 0.5)),
                    "colony": densest(a.colonies()[alive]),
                    "swimming": densest((g[:, SPEED] > 0.1) & (act[:, 1] > 0.3)),
                    "chemotaxis": densest(tuned)})
        for ev in detector.update(summary):
            save_event(ev)

    def record():
        nonlocal next_rec, next_snap, next_species
        if sim.tick >= next_species:
            classify()
            next_species = (sim.tick // SPECIES_EVERY + 1) * SPECIES_EVERY
        if sim.tick >= next_snap:
            arrays = capture(sim)
            with shared.lock:
                arrays["species.json"] = np.frombuffer(shared.tracker.to_json().encode(), np.uint8)
            history.record_snapshot(sim.tick, arrays)
            next_snap = (sim.tick // history.snap_every + 1) * history.snap_every
        if sim.tick >= next_rec:
            recorder.draw_natural()
            rgb = (np.clip(recorder.rgba.to_numpy()[:, :, :3], 0, 1) * 255).astype(np.uint8)
            rows, _ = recorder.viewport_cells(0, 0, w.w, w.d)
            history.record_frame(sim.tick, rgb, rows, {
                "tick": sim.tick, "ts": float(w.clim[CL_TS]), "ice": float(w.clim[CL_ICE]),
                "o2": float(w.atm[ATM_O]), "ch4": float(w.atm[ATM_M]), "co2": float(w.atm[ATM_C]),
                "slots": int(sim.life.count())})
            next_rec = (sim.tick // history.frame_every + 1) * history.frame_every

    w.update_light_temp()  # the first frame (tick 0) would otherwise be drawn before any light
    record()

    while True:
        while not shared.commands.empty():
            cmd = shared.commands.get()
            try:
                if "paused" in cmd:
                    state["paused"] = bool(cmd["paused"])
                if "show_life" in cmd:
                    state["show_life"] = bool(cmd["show_life"])
                if "speed" in cmd:
                    state["speed"] = min(max(int(cmd["speed"]), 0), 1000)
                if "layer" in cmd:
                    state["layer"] = min(max(int(cmd["layer"]), 0), len(LAYERS) - 1)
                    vmax = vmax_for(state["layer"])
                if "branch_from" in cmd:
                    hit = history.nearest("snap", int(cmd["branch_from"]))
                    notice = "다시 실행할 저장점이 없습니다" if hit is None else ""
                    if hit is not None:
                        t_fork, owner = hit
                        try:
                            arrays = history.load("snap", owner, t_fork)
                        except OSError:
                            notice = f"tick {t_fork:,} 저장점을 읽지 못해 다시 실행하지 못했습니다"
                            hit = None
                    if hit is not None:
                        restore(sim, arrays)
                        if "species.json" in arrays:
                            with shared.lock:
                                shared.tracker = SpeciesTracker.from_json(
                                    arrays["species.json"].tobytes().decode())
                        history.new_branch(owner, t_fork)
                        next_species = sim.tick + SPECIES_EVERY
                        history_list = [h for h in history_list if h["tick"] <= t_fork]
                        next_rec = next_snap = sim.tick + 1
                        last_hist = sim.tick
                        err = sim.budget_error()
                        life = life_summary(sim)
                if "view" in cmd:
                    v = cmd["view"]
                    state["view"] = None
                    if isinstance(v, list) and len(v) == 4:
                        x0, y0, x1, y1 = (float(c) for c in v)
                        if 0 < (x1 - x0) * (y1 - y0) <= MAX_VIEW_AREA:
                            state["view"] = (x0, y0, x1, y1)
            except (TypeError, ValueError, KeyError, OSError):
                pass  # malformed command from the browser: ignore, keep the sim running

        t0 = time.perf_counter()
        done = 0
        if not state["paused"]:
            if state["speed"] > 0:
                sim.step(state["speed"])
                done = state["speed"]
                record()
                ti.sync()
                time.sleep(max(1 / 30 - (time.perf_counter() - t0), 0.0))  # speed = ticks per frame
            else:
                while time.perf_counter() - t0 < STEP_BUDGET_S:
                    sim.step(10)
                    done += 10
                    record()
                    ti.sync()
            ti.sync()
        else:
            time.sleep(0.05)
        if done:
            tps = 0.8 * tps + 0.2 * done / max(time.perf_counter() - t0, 1e-6)

        if sim.tick - last_hist >= hist_every:
            last_hist = sim.tick
            life = life_summary(sim)
            err = sim.budget_error()
            vmax = vmax_for(state["layer"])
            history_list.append({"tick": sim.tick, "pop": life["pop"], "photic": life["photic"],
                            "cells": life.get("cells", 0.0),
                            "o2": float(w.atm[ATM_O]), "ch4": float(w.atm[ATM_M]),
                            "co2": float(w.atm[ATM_C]), "ts": float(w.clim[CL_TS]),
                            "ice": float(w.clim[CL_ICE])})
            if len(history_list) > 1200:
                history_list = history_list[::2]  # bounded payload over very long runs
                hist_every *= 2

        if time.perf_counter() - last_frame >= FRAME_EVERY_S:
            last_frame = time.perf_counter()
            if state["layer"] == NATURAL_LAYER:
                renderer.draw_natural()  # life is drawn as pigment density
            else:
                renderer.draw(state["layer"], vmax)
                if state["show_life"]:
                    renderer.draw_life()
            rgb = (np.clip(renderer.rgba.to_numpy()[:, :, :3], 0, 1) * 255).astype(np.uint8)
            stats = {
                "tick": sim.tick, "myr": sim.tick * 1000 / 1e6, "tps": tps, "gpu": gpu_name,
                "natural_layer": NATURAL_LAYER, "world": [w.w, w.d], "cell_fields": CELL_FIELDS,
                "paused": state["paused"], "speed": state["speed"], "layer": state["layer"],
                "layers": LAYERS, "vmax": vmax, "ts": float(w.clim[CL_TS]),
                "ice": float(w.clim[CL_ICE]), "atm": {"co2": float(w.atm[ATM_C]),
                                                      "o2": float(w.atm[ATM_O]),
                                                      "ch4": float(w.atm[ATM_M])},
                "buried": float(w.acc[BUR_ORG]), "err": err.worst(), "life": life,
                "history": history_list, "branch": history.current, "temp_layer": TEMP_LAYER,
                "notice": notice, "rec_dropped": history.dropped, "rec_errors": history.errors,
            }
            frame = png(rgb)
            body = json.dumps(stats).encode()
            with shared.lock:
                shared.frame, shared.stats = frame, body

        if state["view"] is not None and time.perf_counter() - last_cells >= CELLS_EVERY_S:
            last_cells = time.perf_counter()
            rows, total = renderer.viewport_cells(*state["view"])
            blob = struct.pack("<qii", sim.tick, total, len(rows)) + rows.astype("<f4").tobytes()
            with shared.lock:
                shared.cells = blob
