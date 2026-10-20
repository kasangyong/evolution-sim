"""Time machine (stage 4c): frame records for rewinding and full snapshots for re-running from a
past moment, organised as a tree of branches under a disk budget.

Layout under <root>/:
  meta.json                     config, schema version
  branches.json                 [{id, parent, fork_tick}]
  b<id>/frames/<tick>.npz       natural-colour image (RGB u8), every cell (packed), stats JSON
  b<id>/snaps/<tick>.npz        full state from evo.io.snapshot.capture

A branch's timeline is its ancestors' records up to its fork tick, then its own. Writing and
compression run on a background thread; the main thread only hands over numpy copies."""
import json
import queue
import threading
from pathlib import Path

import numpy as np

from evo.io.snapshot import SCHEMA_VERSION, config_to_json

CELL_DTYPE = np.dtype([("x", "<u2"), ("y", "<u2"), ("size", "<f2"), ("ncell", "<f4"),
                       ("rgb", "u1", 3), ("uid", "<u4"), ("parent", "<u4"), ("kind", "u1"),
                       ("vx", "<f2"), ("vy", "<f2"), ("species", "<u4"), ("event", "u1"),
                       ("colony", "u1")])
POS_SCALE = 64.0  # positions stored as u16 in 1/64 grid cells


def pack_cells(rows: np.ndarray) -> np.ndarray:
    """Renderer rows (n, 15) f32 -> compact structured array (32 bytes per cell)."""
    out = np.zeros(len(rows), CELL_DTYPE)
    out["x"] = np.clip(rows[:, 0] * POS_SCALE, 0, 65535)
    out["y"] = np.clip(rows[:, 1] * POS_SCALE, 0, 65535)
    out["size"], out["ncell"] = rows[:, 2], rows[:, 3]
    out["rgb"] = np.clip(rows[:, 4:7] * 255, 0, 255)
    out["uid"], out["parent"], out["kind"] = rows[:, 7], rows[:, 8], rows[:, 9]
    out["vx"], out["vy"], out["species"] = rows[:, 10], rows[:, 11], rows[:, 12]
    out["event"], out["colony"] = rows[:, 13], rows[:, 14]
    return out


def unpack_cells(packed: np.ndarray) -> np.ndarray:
    """Inverse of pack_cells, back to renderer rows (n, 15) f32 for the browser."""
    rows = np.zeros((len(packed), 15), np.float32)
    rows[:, 0] = packed["x"] / POS_SCALE
    rows[:, 1] = packed["y"] / POS_SCALE
    rows[:, 2], rows[:, 3] = packed["size"], packed["ncell"]
    rows[:, 4:7] = packed["rgb"] / 255.0
    rows[:, 7], rows[:, 8], rows[:, 9] = packed["uid"], packed["parent"], packed["kind"]
    rows[:, 10], rows[:, 11], rows[:, 12] = packed["vx"], packed["vy"], packed["species"]
    rows[:, 13], rows[:, 14] = packed["event"], packed["colony"]
    return rows


class History:
    def __init__(self, root: Path, cfg, budget_bytes: int = 20 * 1024**3, frame_every: int = 100,
                 snap_every: int = 10_000, keep_recent: int = 200_000):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.budget = budget_bytes
        self.frame_every, self.snap_every, self.keep_recent = frame_every, snap_every, keep_recent
        self.lock = threading.Lock()
        (self.root / "meta.json").write_text(json.dumps(
            {"schema_version": SCHEMA_VERSION, "config": json.loads(config_to_json(cfg))}))
        self.branches: list[dict] = [{"id": 0, "parent": None, "fork_tick": 0}]
        self.frames: dict[int, list[int]] = {0: []}
        self.snaps: dict[int, list[int]] = {0: []}
        self.sizes: dict[Path, int] = {}
        self.dropped = 0  # records skipped because the writer fell behind (queue full)
        self.errors = 0  # records that failed to write (e.g. disk full)
        self.current = 0
        self._save_branches()
        self.q: queue.Queue = queue.Queue(maxsize=64)
        threading.Thread(target=self._writer, daemon=True).start()

    # ---- main thread: hand over copies ----
    def _put(self, item):
        # never block the simulation: if the writer falls behind, skip this record
        try:
            self.q.put_nowait(item)
        except queue.Full:
            self.dropped += 1

    def record_frame(self, tick: int, rgb: np.ndarray, cell_rows: np.ndarray, stats: dict):
        self._put(("frame", self.current, tick, {"rgb": rgb, "cells": pack_cells(cell_rows),
                                                  "stats": np.frombuffer(json.dumps(stats).encode(),
                                                                         np.uint8)}))

    def record_snapshot(self, tick: int, arrays: dict):
        self._put(("snap", self.current, tick, arrays))

    def new_branch(self, parent: int, fork_tick: int) -> int:
        """Start a branch from the snapshot (parent, fork_tick); `parent` must own that snapshot,
        so the new lineage holds nothing recorded after the fork."""
        with self.lock:
            bid = len(self.branches)
            self.branches.append({"id": bid, "parent": parent, "fork_tick": fork_tick})
            self.frames[bid], self.snaps[bid] = [], []
            self.current = bid
        self._save_branches()
        return bid

    # ---- lineage queries (any thread) ----
    def lineage(self, branch: int | None = None) -> list[tuple[int, int]]:
        """[(branch, last tick taken from it)] from the root to `branch`."""
        with self.lock:
            b = self.current if branch is None else branch
            chain, limit = [], None
            while b is not None:
                chain.append((b, limit))
                limit = self.branches[b]["fork_tick"]
                b = self.branches[b]["parent"]
            return chain[::-1]

    def _records(self, kind: str, branch: int | None) -> list[tuple[int, int]]:
        table = self.frames if kind == "frame" else self.snaps
        out = []
        for b, limit in self.lineage(branch):
            with self.lock:
                ticks = list(table[b])
            out += [(t, b) for t in ticks if limit is None or t <= limit]
        return sorted(out)

    def frame_ticks(self, branch: int | None = None) -> list[int]:
        return [t for t, _ in self._records("frame", branch)]

    def snap_ticks(self, branch: int | None = None) -> list[int]:
        return [t for t, _ in self._records("snap", branch)]

    def nearest(self, kind: str, tick: int, branch: int | None = None) -> tuple[int, int] | None:
        """Latest record at or before `tick` on the lineage, as (tick, branch)."""
        best = None
        for t, b in self._records(kind, branch):
            if t <= tick:
                best = (t, b)
        return best

    def path(self, kind: str, branch: int, tick: int) -> Path:
        return self.root / f"b{branch}" / ("frames" if kind == "frame" else "snaps") / f"{tick:010d}.npz"

    def load(self, kind: str, branch: int, tick: int) -> dict:
        with np.load(self.path(kind, branch, tick)) as z:
            return {k: z[k] for k in z.files}

    def used_bytes(self) -> int:
        with self.lock:
            return sum(self.sizes.values())

    # ---- writer thread ----
    def _writer(self):
        while True:
            kind, branch, tick, payload = self.q.get()
            p = self.path(kind, branch, tick)
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(p, **payload)
                with self.lock:
                    (self.frames if kind == "frame" else self.snaps)[branch].append(tick)
                    self.sizes[p] = p.stat().st_size
                if self.used_bytes() > self.budget:
                    self._thin()
            except Exception:  # noqa: BLE001 - the writer must outlive any single bad record
                # one failed write (disk full, file in use, out of memory) must not kill the writer;
                # otherwise recording stops for good
                self.errors += 1
                if p not in self.sizes:  # drop a half-written file, if we can
                    try:
                        p.unlink(missing_ok=True)
                    except OSError:
                        pass
            finally:
                self.q.task_done()

    def flush(self):
        """Block until every queued record is on disk."""
        self.q.join()

    def _protected(self) -> set[tuple[int, int]]:
        """Fork points stay: each branch's fork snapshot is where it restarts."""
        with self.lock:
            return {(b["parent"], b["fork_tick"]) for b in self.branches if b["parent"] is not None}

    def _thin(self):
        """Drop every other old record (older than keep_recent behind the newest) until under budget;
        the first record, fork points and the recent window are kept."""
        protected = self._protected()
        for kind in ("frame", "snap"):
            table = self.frames if kind == "frame" else self.snaps
            with self.lock:
                newest = max((t for ticks in table.values() for t in ticks), default=0)
                victims = []
                for b, ticks in table.items():
                    old = sorted(t for t in ticks if t < newest - self.keep_recent)
                    victims += [(b, t) for t in old[1::2] if (b, t) not in protected]
            for b, t in victims:
                p = self.path(kind, b, t)
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    continue  # a reader has it open (Windows); retry at the next thinning
                with self.lock:
                    table[b].remove(t)
                    self.sizes.pop(p, None)
                if self.used_bytes() <= self.budget:
                    return

    def _save_branches(self):
        with self.lock:
            text = json.dumps(self.branches)
        (self.root / "branches.json").write_text(text)
