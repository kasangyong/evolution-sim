import math

import numpy as np
import taichi as ti

from evo.config import NS, C, D, H, N, R, WorldConfig

# accumulator slots (f64) since t=0: ledger sources/sinks, then observational burial pools
SRC_C, SRC_N, SRC_E, SINK_C, SINK_N, SINK_E, BUR_ORG, BUR_CARB = range(8)
# f64: in steady state f32 rounding repeats identically every tick and drifts the budget
CHEM_DTYPE = ti.f64
# climate state slots
CL_TS, CL_ICE, CL_SOLAR = range(3)

MAX_CELL_OUTFLOW = 0.3


def make_terrain(cfg: WorldConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return (floor[x] = water cells in column x, vent (x, y) positions)."""
    rng = np.random.default_rng(cfg.seed)
    w, d = cfg.width, cfg.depth
    x = np.arange(w) / w
    shelf, slope_end = cfg.shelf_frac, cfg.shelf_frac + 0.15
    t = np.clip((x - shelf) / (slope_end - shelf), 0.0, 1.0)
    t = t * t * (3 - 2 * t)
    base = cfg.shelf_depth_frac + (cfg.abyss_depth_frac - cfg.shelf_depth_frac) * t
    noise = np.zeros(w)
    for k in range(1, 6):
        noise += rng.uniform(-1, 1) * np.sin(2 * math.pi * k * x + rng.uniform(0, 2 * math.pi)) / k
    noise *= 0.03 * t  # flat shelf, rough abyss
    floor = np.clip(np.round((base + noise) * d), 8, d - 2).astype(np.int32)

    lo, hi = int(slope_end * w) + 10, w - 10
    xs = np.sort(rng.choice(np.arange(lo, hi, 16), size=cfg.n_vents, replace=False))
    vents = np.stack([xs, floor[xs] - 1], axis=1).astype(np.int32)
    return floor, vents


def make_streamfunction(cfg: WorldConfig, floor: np.ndarray) -> np.ndarray:
    """Corner streamfunction psi[w+1, d+1]; zero on every corner touching rock or the surface,
    so face fluxes (psi differences) are divergence-free and vanish at walls."""
    w, d = cfg.width, cfg.depth
    psi = np.zeros((w + 1, d + 1))
    for i in range(1, w):
        fc = min(floor[i - 1], floor[i])
        j = np.arange(1, fc)
        psi[i, 1:fc] = fc / math.pi * math.sin(math.pi * i / w) * np.sin(math.pi * j / fc)

    u = psi[1:w, 1:] - psi[1:w, :-1]  # flux through vertical face at x=i (i=1..w-1), +x
    v = -(psi[1:, 1:d] - psi[:-1, 1:d])  # flux through horizontal face at y=j (j=1..d-1), +y
    out = np.zeros((w, d))
    out[:-1, :] += np.maximum(u, 0)
    out[1:, :] += np.maximum(-u, 0)
    out[:, :-1] += np.maximum(v, 0)
    out[:, 1:] += np.maximum(-v, 0)
    peak = out.max()
    return psi * (min(cfg.psi_amp, MAX_CELL_OUTFLOW) / peak if peak > 0 else 0.0)


@ti.data_oriented
class World:
    """Eulerian ocean grid: chemistry fields, transport, vents, light and temperature."""

    def __init__(self, cfg: WorldConfig):
        self.cfg = cfg
        w, d = cfg.width, cfg.depth
        self.w, self.d = w, d
        self.chem = ti.field(CHEM_DTYPE, shape=(NS, w, d))
        self.delta = ti.field(CHEM_DTYPE, shape=(NS, w, d))
        self.floor = ti.field(ti.i32, shape=w)
        self.psi = ti.field(ti.f32, shape=(w + 1, d + 1))
        self.light = ti.field(ti.f32, shape=(w, d))
        self.temp = ti.field(ti.f32, shape=(w, d))
        self.vents = ti.Vector.field(2, ti.i32, shape=cfg.n_vents)
        self.atm = ti.field(ti.f64, shape=4)
        self.atm_delta = ti.field(ti.f64, shape=4)
        self.clim = ti.field(ti.f64, shape=3)
        self.acc = ti.field(ti.f64, shape=8)

        floor, vents = make_terrain(cfg)
        self.floor_np, self.vents_np = floor, vents
        self.floor.from_numpy(floor)
        self.vents.from_numpy(vents)
        self.psi.from_numpy(make_streamfunction(cfg, floor).astype(np.float32))

        init = np.zeros((NS, w, d), dtype=np.float32)
        water = np.arange(d)[None, :] < floor[:, None]
        for s, val in cfg.init.items():
            init[s][water] = val
        self.chem.from_numpy(init)
        self.water_np = water

    @ti.func
    def is_water(self, x: int, y: int) -> bool:
        ok = False
        if x >= 0 and x < self.w and y >= 0 and y < self.d:
            ok = y < self.floor[x]
        return ok

    @ti.func
    def kv(self, y_face: int) -> ti.f32:
        k = self.cfg.kv_deep
        if y_face < self.cfg.mix_layer_frac * self.d:
            k = self.cfg.kv_mix
        return k

    @ti.kernel
    def transport(self):
        # gather form: each face flux is evaluated identically (antisymmetrically) from both
        # sides, so transport conserves every species to float rounding
        for s, x, y in self.delta:
            dc = ti.cast(0.0, CHEM_DTYPE)
            if self.is_water(x, y):
                c = self.chem[s, x, y]
                # right face at x+1: u>0 means flow +x
                if self.is_water(x + 1, y):
                    u = self.psi[x + 1, y + 1] - self.psi[x + 1, y]
                    cn = self.chem[s, x + 1, y]
                    dc += self.cfg.kh * (cn - c) - (u * c if u > 0 else u * cn)
                if self.is_water(x - 1, y):
                    u = self.psi[x, y + 1] - self.psi[x, y]
                    cn = self.chem[s, x - 1, y]
                    dc += self.cfg.kh * (cn - c) + (u * cn if u > 0 else u * c)
                # bottom face at y+1: v>0 means flow +y (down)
                if self.is_water(x, y + 1):
                    v = -(self.psi[x + 1, y + 1] - self.psi[x, y + 1])
                    cn = self.chem[s, x, y + 1]
                    dc += self.kv(y + 1) * (cn - c) - (v * c if v > 0 else v * cn)
                if self.is_water(x, y - 1):
                    v = -(self.psi[x + 1, y] - self.psi[x, y])
                    cn = self.chem[s, x, y - 1]
                    dc += self.kv(y) * (cn - c) + (v * cn if v > 0 else v * c)
                if s == D:
                    ws = self.cfg.sink_speed
                    if self.is_water(x, y + 1):
                        dc -= ws * c
                    if self.is_water(x, y - 1):
                        dc += ws * self.chem[s, x, y - 1]
            self.delta[s, x, y] = dc
        for s, x, y in self.chem:
            self.chem[s, x, y] += self.delta[s, x, y]

    @ti.kernel
    def emit_vents(self):
        for v in range(self.cfg.n_vents):
            p = self.vents[v]
            self.chem[H, p.x, p.y] += self.cfg.vent_h
            self.chem[R, p.x, p.y] += self.cfg.vent_r
            self.chem[C, p.x, p.y] += self.cfg.vent_c
            self.chem[N, p.x, p.y] += self.cfg.vent_n
        n = ti.cast(self.cfg.n_vents, ti.f64)
        self.acc[SRC_C] += n * self.cfg.vent_c
        self.acc[SRC_N] += n * self.cfg.vent_n
        self.acc[SRC_E] += n * 2.0 * (self.cfg.vent_h + self.cfg.vent_r)

    @ti.kernel
    def update_light_temp(self):
        ts = ti.cast(self.clim[CL_TS], ti.f32)
        ice = ti.cast(self.clim[CL_ICE], ti.f32)
        solar = ti.cast(self.clim[CL_SOLAR], ti.f32)
        for x, y in self.light:
            lt, tp = 0.0, 0.0
            if self.is_water(x, y):
                lt = solar * (1.0 - ice) * ti.exp(-self.cfg.light_atten * y)
                tp = self.cfg.temp_deep + (ts - self.cfg.temp_deep) * ti.exp(-y / self.cfg.thermocline)
                for v in range(self.cfg.n_vents):
                    p = self.vents[v]
                    r2 = ti.cast((x - p.x) ** 2 + (y - p.y) ** 2, ti.f32)
                    tp += self.cfg.vent_heat * ti.exp(-r2 / (self.cfg.vent_heat_radius**2))
            self.light[x, y] = lt
            self.temp[x, y] = tp
