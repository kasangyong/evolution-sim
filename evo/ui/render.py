import itertools

import numpy as np
import taichi as ti

from evo.life.agents import ALIVE, Agents
from evo.life.genome import (
    COL_B,
    COL_G,
    COL_R,
    ETA_H2,
    ETA_PHOTO_AN,
    ETA_PHOTO_OX,
    ETA_PRED,
    SPEED,
)
from evo.world.fields import CL_ICE, CL_TS, World

# layers selectable in the viewer: species 0..7, then light, temperature, natural colour
LIGHT_LAYER, TEMP_LAYER, NATURAL_LAYER = 8, 9, 10

# cell record for the zoom view, one row of f32 per cell (stage 4b, design decision 3)
CELL_FIELDS = ("x", "y", "size", "ncell", "r", "g", "b", "uid", "parent", "kind", "vx", "vy",
               "species", "event", "colony", "org", "seg")  # org: organism id + 1 (0: none)
N_CELL_FIELDS = len(CELL_FIELDS)
KIND_CHEMO, KIND_PHOTO_AN, KIND_PHOTO_OX, KIND_PRED = 0, 1, 2, 3
UID_MOD = 16777216  # uid kept in records modulo 2^24 (exact in f32)
SAMPLE_TARGET = 0.9  # sample aims at 90% of max_cells so random overshoot is rarely truncated


@ti.func
def uid_hash(uid: ti.i64) -> ti.i64:
    """Stable 32-bit hash per cell (Knuth multiplicative), same as uid_hash_np on recorded uids."""
    return ((uid % ti.i64(UID_MOD)) * ti.i64(2654435761)) & ti.i64(0xFFFFFFFF)


def uid_hash_np(uid24) -> np.ndarray:
    return (np.asarray(uid24).astype(np.int64) * 2654435761) & 0xFFFFFFFF


def in_sample_np(uid24, n_in: int, max_cells: int) -> np.ndarray:
    """Integer rule shared with `extract`: hash × n_in < 0.9·max_cells × 2^32 (exact)."""
    limit = int(SAMPLE_TARGET * max_cells) * 4294967296
    return uid_hash_np(uid24) * max(n_in, 1) < limit


@ti.data_oriented
class Renderer:
    """Maps world fields to an RGBA float image laid out row-major (y, x), and extracts the cells
    inside a viewport for the browser's zoom view. Every colour comes from simulated values."""

    def __init__(self, world: World, agents: Agents, max_cells: int = 20000):
        self.world = world
        self.agents = agents
        self.rgba = ti.field(ti.f32, shape=(world.d, world.w, 4))
        self.pig = ti.Vector.field(3, ti.f32, shape=(world.w, world.d))  # Σ pigment × biomass
        self.mass = ti.field(ti.f32, shape=(world.w, world.d))
        self.max_cells = max_cells
        self.cells = ti.field(ti.f32, shape=(max_cells, N_CELL_FIELDS))
        self.n_cells = ti.field(ti.i32, shape=())
        self.n_in = ti.field(ti.i32, shape=())  # cells in view before sampling

    @ti.func
    def ramp(self, t: ti.f32) -> ti.math.vec3:
        # dark blue -> teal -> yellow
        t = ti.min(ti.max(t, 0.0), 1.0)
        a = ti.math.vec3(0.03, 0.05, 0.20)
        b = ti.math.vec3(0.05, 0.55, 0.55)
        c = ti.math.vec3(0.98, 0.90, 0.25)
        col = a + (b - a) * (t / 0.5)
        if t > 0.5:
            col = b + (c - b) * ((t - 0.5) / 0.5)
        return col

    @ti.func
    def pigment(self, i: int) -> ti.math.vec3:
        """Cell colour from its metabolism, in the colours of real microbes: chlorophyll green
        (oxygenic photosynthesis), bacteriochlorophyll purple-brown (anoxygenic), near-colourless
        grey (chemosynthesis), pale tan (predators). Lineage colour genes shift it slightly."""
        w_ox = self.agents.genome[i, ETA_PHOTO_OX]
        w_an = self.agents.genome[i, ETA_PHOTO_AN]
        w_pr = self.agents.genome[i, ETA_PRED]
        w_ch = 0.3 + self.agents.genome[i, ETA_H2]
        col = (w_ox * ti.math.vec3(0.22, 0.62, 0.26) + w_an * ti.math.vec3(0.55, 0.30, 0.42)
               + w_pr * ti.math.vec3(0.78, 0.60, 0.40) + w_ch * ti.math.vec3(0.80, 0.82, 0.78))
        col /= w_ox + w_an + w_pr + w_ch
        lineage = ti.math.vec3(self.agents.genome[i, COL_R], self.agents.genome[i, COL_G],
                               self.agents.genome[i, COL_B]) - 0.5
        return ti.min(ti.max(col + 0.12 * lineage, 0.0), 1.0)

    @ti.func
    def kind(self, i: int) -> ti.i32:
        k = KIND_CHEMO
        best = 0.15  # below this every metabolism is weak: keep "chemosynthetic"
        if self.agents.genome[i, ETA_PHOTO_AN] > best:
            best = self.agents.genome[i, ETA_PHOTO_AN]
            k = KIND_PHOTO_AN
        if self.agents.genome[i, ETA_PHOTO_OX] > best:
            best = self.agents.genome[i, ETA_PHOTO_OX]
            k = KIND_PHOTO_OX
        if self.agents.genome[i, ETA_PRED] > best:
            k = KIND_PRED
        return k

    @ti.kernel
    def draw(self, layer: int, vmax: ti.f32):
        for y, x in ti.ndrange(self.world.d, self.world.w):
            col = ti.math.vec3(0.22, 0.16, 0.10)  # rock
            if self.world.is_water(x, y):
                v = 0.0
                if layer == LIGHT_LAYER:
                    v = self.world.light[x, y]
                elif layer == TEMP_LAYER:
                    v = self.world.temp[x, y]
                else:
                    v = ti.cast(self.world.chem[layer, x, y], ti.f32)
                col = self.ramp(v / vmax)
            for i in ti.static(range(3)):
                self.rgba[y, x, i] = col[i]
            self.rgba[y, x, 3] = 1.0

    @ti.kernel
    def draw_natural(self):
        """Documentary view: water lit by the simulated light field, vents glowing with their
        simulated heat, sea ice from the ice fraction, life as pigment density per pixel."""
        for x, y in self.mass:
            self.mass[x, y] = 0.0
            self.pig[x, y] = ti.math.vec3(0.0)
        for i in range(self.agents.cap):
            if self.agents.state[i] == ALIVE:
                c = ti.cast(self.agents.pos[i], ti.i32)
                m = ti.cast(self.agents.body[i], ti.f32)
                ti.atomic_add(self.mass[c.x, c.y], m)
                ti.atomic_add(self.pig[c.x, c.y], m * self.pigment(i))
        ts = ti.cast(self.world.clim[CL_TS], ti.f32)
        ice = ti.cast(self.world.clim[CL_ICE], ti.f32)
        for y, x in ti.ndrange(self.world.d, self.world.w):
            col = ti.math.vec3(0.0)
            if self.world.is_water(x, y):
                # light: log scale from the deep dark (1e-4) to full sun
                lt = ti.max(self.world.light[x, y], 1e-6)
                t = ti.min(ti.max((ti.log(lt) / ti.log(10.0) + 4.0) / 4.0, 0.0), 1.0)
                deep = ti.math.vec3(0.005, 0.015, 0.04)
                shallow = ti.math.vec3(0.10, 0.42, 0.55)
                col = deep + (shallow - deep) * t * t
                # vent heat: temperature above the undisturbed water column at this depth
                bg = (self.world.cfg.temp_deep
                      + (ts - self.world.cfg.temp_deep) * ti.exp(-y / self.world.cfg.thermocline))
                hot = ti.min(ti.max((self.world.temp[x, y] - bg) / 40.0, 0.0), 1.0)
                col += hot * ti.math.vec3(0.95, 0.42, 0.10)
                # life: pigment density, opacity grows with log biomass
                m = self.mass[x, y]
                if m > 0:
                    a = ti.min(ti.log(1.0 + m * 400.0) / ti.log(401.0), 1.0) * 0.9
                    col = col * (1.0 - a) + (self.pig[x, y] / m) * a
                # sea ice on the top rows, as thick as the ice fraction is large
                if y < 4 and ice > 0:
                    col = col * (1.0 - ice) + ice * ti.math.vec3(0.86, 0.92, 0.96)
            else:
                shade = 0.6 + 0.4 * ti.exp(-(y - self.world.floor[x]) / 30.0)
                col = shade * ti.math.vec3(0.20, 0.15, 0.11)
            for k in ti.static(range(3)):
                self.rgba[y, x, k] = ti.min(col[k], 1.0)
            self.rgba[y, x, 3] = 1.0

    @ti.kernel
    def draw_life(self):
        for i in range(self.agents.cap):
            if self.agents.state[i] == ALIVE:
                c = ti.cast(self.agents.pos[i], ti.i32)
                for dy, dx in ti.static(ti.ndrange(2, 2)):
                    py = ti.min(c.y + dy, self.world.d - 1)
                    px = ti.min(c.x + dx, self.world.w - 1)
                    self.rgba[py, px, 0] = self.agents.genome[i, COL_R]
                    self.rgba[py, px, 1] = self.agents.genome[i, COL_G]
                    self.rgba[py, px, 2] = self.agents.genome[i, COL_B]

    @ti.kernel
    def extract(self, x0: ti.f32, y0: ti.f32, x1: ti.f32, y1: ti.f32):
        """Copy the cells inside [x0, x1) × [y0, y1) into `cells`. When more than max_cells are in
        view, keep a fixed sample chosen by uid hash, so the same cells stay on screen from frame
        to frame (a moving sample would look like deaths and births that did not happen)."""
        self.n_cells[None] = 0
        self.n_in[None] = 0
        for i in range(self.agents.cap):
            if self.agents.state[i] == ALIVE:
                p = self.agents.pos[i]
                if x0 <= p.x < x1 and y0 <= p.y < y1:
                    ti.atomic_add(self.n_in[None], 1)
        n_in = ti.max(ti.cast(self.n_in[None], ti.i64), ti.i64(1))
        limit = ti.i64(int(SAMPLE_TARGET * self.max_cells)) * ti.i64(4294967296)
        for i in range(self.agents.cap):
            if self.agents.state[i] == ALIVE:
                p = self.agents.pos[i]
                # organism segments always: a sampled body would fall apart on screen
                if x0 <= p.x < x1 and y0 <= p.y < y1 and (
                        self.agents.org[i] >= 0 or uid_hash(self.agents.uid[i]) * n_in < limit):
                    k = ti.atomic_add(self.n_cells[None], 1)
                    if k < self.max_cells:
                        n = self.agents.ncell[i]
                        col = self.pigment(i)
                        act = self.agents.act[i]
                        sp = self.agents.genome[i, SPEED] * ti.max(act[1], 0.0)  # ACT_SPEED
                        hd = self.agents.heading[i]
                        row = ti.Vector([
                            p.x, p.y, ti.cast(self.agents.body[i], ti.f32) / n, n,
                            col[0], col[1], col[2],
                            ti.cast(self.agents.uid[i] % 16777216, ti.f32),
                            ti.cast(self.agents.parent[i] % 16777216, ti.f32),
                            ti.cast(self.kind(i), ti.f32), sp * ti.cos(hd), sp * ti.sin(hd),
                            ti.cast(self.agents.species[i], ti.f32),
                            ti.cast(self.agents.event[i], ti.f32),
                            ti.cast(self.agents.is_colony(i), ti.f32),
                            ti.cast(self.agents.org[i] + 1, ti.f32),
                            ti.cast(self.agents.seg[i], ti.f32)])
                        for f in ti.static(range(N_CELL_FIELDS)):
                            self.cells[k, f] = row[f]

    def viewport_cells(self, x0: float, y0: float, x1: float, y1: float):
        """Cells in the viewport as a float32 array (n, N_CELL_FIELDS); main thread only."""
        self.extract(x0, y0, x1, y1)
        n = min(self.n_cells[None], self.max_cells)
        return self.cells.to_numpy()[:n], self.n_in[None]


def closeup(background: np.ndarray, rows: np.ndarray, x0: float, y0: float, scale: int,
            organisms: bool = False, focus: tuple | None = None) -> np.ndarray:
    """Event close-up (numpy, any thread): the natural-colour crop scaled by `scale`, with every
    cell drawn as a disc in its pigment colour, sized by cell body like the browser zoom view,
    with a darker rim; colonies get a light ring. With `organisms`, the other cells are faded and
    each multicellular organism is drawn as its segments joined head to tail, outlined, so one
    body stands out among thousands of cells (stage 7b-7d events); the one whose head is nearest
    `focus` (the event's organism) in bright orange, the others thinly."""
    img = background.repeat(scale, axis=0).repeat(scale, axis=1).astype(np.float32)
    img *= 0.7  # dim the water so the cells read
    h, w, _ = img.shape
    f = {name: k for k, name in enumerate(CELL_FIELDS)}
    if organisms:
        bodies = {}
        for r in rows:
            if r[f["org"]] > 0:
                bodies.setdefault(int(r[f["org"]]), []).append(r)
        rows = [r for r in rows if r[f["org"]] <= 0]  # the rest first, faded, then the bodies
    for r in rows:
        cx, cy = (r[f["x"]] - x0) * scale, (r[f["y"]] - y0) * scale
        rad = max(scale * 0.42 * np.cbrt(max(r[f["size"]], 1e-6) / 0.02), 1.5)
        if organisms:
            rad *= 0.6  # background cells: smaller and faded
        xa, xb = int(max(cx - rad - 2, 0)), int(min(cx + rad + 3, w))
        ya, yb = int(max(cy - rad - 2, 0)), int(min(cy + rad + 3, h))
        if xa >= xb or ya >= yb:
            continue
        yy, xx = np.mgrid[ya:yb, xa:xb]
        d = np.hypot(xx + 0.5 - cx, yy + 0.5 - cy)
        col = np.array([r[f["r"]], r[f["g"]], r[f["b"]]]) * 255
        if organisms:
            col = 0.2 * col + 0.8 * img[int(min(max(cy, 0), h - 1)), int(min(max(cx, 0), w - 1))]
        img[ya:yb, xa:xb][d <= rad] = col
        img[ya:yb, xa:xb][(d > rad - 1.2) & (d <= rad)] = col * 0.55
        if r[f["colony"]] > 0 and not organisms:
            img[ya:yb, xa:xb][(d > rad + 0.8) & (d <= rad + 1.8)] = (235, 240, 245)
    if organisms:
        yy, xx = np.mgrid[0:h, 0:w]
        target = None
        if focus is not None and bodies:
            def head_dist(item):
                head = min(item[1], key=lambda r: r[f["seg"]])
                return np.hypot(head[f["x"]] - focus[0], head[f["y"]] - focus[1])
            target = min(bodies.items(), key=head_dist)[0]
        for oid in sorted(bodies, key=lambda o: o == target):  # the event's organism on top
            segs = sorted(bodies[oid], key=lambda r: r[f["seg"]])
            main = oid == target or target is None
            link = (255, 170, 60) if main else (200, 190, 160)
            pts = [((r[f["x"]] - x0) * scale, (r[f["y"]] - y0) * scale) for r in segs]
            body_w = max(scale * (0.3 if main else 0.15), 3.0 if main else 1.5)
            for (ax, ay), (bx, by) in itertools.pairwise(pts):  # the body: segments joined
                for t in np.linspace(0.0, 1.0, int(np.hypot(bx - ax, by - ay) / 2) + 2):
                    px, py = ax + (bx - ax) * t, ay + (by - ay) * t
                    near = np.hypot(xx + 0.5 - px, yy + 0.5 - py) <= body_w
                    img[near] = link
            for k, (r, (px, py)) in enumerate(zip(segs, pts)):
                rad = max(scale * (0.42 if main else 0.3), 4.0 if main else 2.5)
                d = np.hypot(xx + 0.5 - px, yy + 0.5 - py)
                img[d <= rad] = np.array([r[f["r"]], r[f["g"]], r[f["b"]]]) * 255
                ring = (255, 255, 255) if k == 0 else link  # the head: white
                img[(d > rad) & (d <= rad + (2.5 if main else 1.0))] = ring
    return np.clip(img, 0, 255).astype(np.uint8)
