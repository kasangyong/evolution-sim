import numpy as np
import taichi as ti

from evo.config import CN_RATIO, C, D, H, LifeConfig, M, N, O, R, X
from evo.life.genome import (
    ADHESION,
    BRAIN,
    DIV_SIZE,
    ETA_AER,
    ETA_ANAER,
    ETA_H2,
    ETA_METH,
    ETA_MOX,
    ETA_PHOTO_AN,
    ETA_PHOTO_OX,
    ETA_PRED,
    ETA_SFIX,
    ETA_SOX,
    FOUNDER,
    GENE_RANGE,
    MUT_RATE,
    N_GENES,
    N_HID,
    N_IN,
    N_OUT,
    O2_TOL,
    PREY_RATIO,
    SPEED,
    T_OPT,
    T_WIDTH,
)
from evo.world.fields import CHEM_DTYPE, World
from evo.world.reactions import REACTIONS, delta_g

DEAD, ALIVE, DYING = 0, 1, 2
COLONY_ADHESION = 0.5  # stage 6: adhesion at which a multi-cell slot counts as a colony
# brain outputs
ACT_X, ACT_Y, ACT_HUNT, ACT_HOLD = range(4)
# discrete actions need a clear signal: with a 0 threshold, tiny weights from the first mutation
# would switch them on for half of all offspring
ACT_ON = 0.5

# organism reactions; stoichiometry and yields come only from reactions.py (§2.3)
BIOTIC = ("h2_chemosynthesis", "anoxygenic_photosynthesis", "photosynthesis", "sulfur_oxidation",
          "methane_oxidation", "sulfur_fixation", "aerobic_respiration", "anaerobic_respiration",
          "methanogenesis")
(RX_CHEMO, RX_PHOTO_AN, RX_PHOTO_OX, RX_SOX, RX_MOX, RX_SFIX, RX_AER, RX_ANAER,
 RX_METH) = range(len(BIOTIC))
_SPECIES = {"H": H, "R": R, "X": X, "C": C, "O": O, "M": M, "N": N, "D": D}


def _sides(rx: dict) -> tuple[tuple, tuple, float]:
    """Cell inputs and outputs as ((species, amount), ...) per unit extent, plus the body
    term. Body carries N at C:N = CN_RATIO, so growth draws N and respiration releases it."""
    rx = dict(rx)
    body = float(rx.pop("B", 0))
    if body:
        rx["N"] = -body / CN_RATIO
    ins = tuple((_SPECIES[s], float(-k)) for s, k in rx.items() if k < 0)
    outs = tuple((_SPECIES[s], float(k)) for s, k in rx.items() if k > 0)
    return ins, outs, body


_TABLE = [_sides(REACTIONS[n]) for n in BIOTIC]
INPUTS = tuple(t[0] for t in _TABLE)
OUTPUTS = tuple(t[1] for t in _TABLE)
BODY = tuple(t[2] for t in _TABLE)
# energy released per unit extent (−ΔG); negative means light or stored energy must pay
G = tuple(-delta_g(REACTIONS[n]) for n in BIOTIC)
G_CHEMO = G[RX_CHEMO]


@ti.data_oriented
class Agents:
    """Fixed-capacity organism pool with an atomic free-list (§4). Slots never move, so a
    (slot, uid) pair stays a valid reference. Body carbon is f64 because it is part of the
    element ledger."""

    def __init__(self, world: World, cfg: LifeConfig, seed: int = 0):
        self.world = world
        self.cfg = cfg
        cap = cfg.capacity
        self.cap = cap
        self.state = ti.field(ti.i32, shape=cap)
        self.want_div = ti.field(ti.i32, shape=cap)
        self.pos = ti.Vector.field(2, ti.f32, shape=cap)
        self.body = ti.field(CHEM_DTYPE, shape=cap)
        self.energy = ti.field(ti.f32, shape=cap)
        self.age = ti.field(ti.i32, shape=cap)
        # super-individual: number of identical cells a slot stands for (1 until the pool fills)
        self.ncell = ti.field(ti.f32, shape=cap)
        self.species = ti.field(ti.i32, shape=cap)  # label from the species tracker (stage 4d)
        self.event = ti.field(ti.i32, shape=cap)  # this tick: 1 = ate, 2 = was bitten (viewer)
        self.uid = ti.field(ti.i64, shape=cap)
        self.parent = ti.field(ti.i64, shape=cap)
        self.genome = ti.field(ti.f32, shape=(cap, N_GENES))
        self.gene_lo = ti.field(ti.f32, shape=N_GENES)
        self.gene_hi = ti.field(ti.f32, shape=N_GENES)
        self.free = ti.field(ti.i32, shape=cap)
        # photosynthetic biomass per grid cell, for shading
        self.cell_body = ti.field(ti.f32, shape=(world.w, world.d))
        # census for sensing and encounters (stage 3b)
        self.cell_cells = ti.field(ti.f32, shape=(world.w, world.d))
        self.cell_mass = ti.field(ti.f32, shape=(world.w, world.d))
        self.encounter = ti.field(ti.i32, shape=(world.w, world.d))
        # encounter draw: key = (−ln u / ncell) quantized, then slot number in the low bits; the
        # smallest key wins, i.e. a slot is picked with probability ∝ its cells (Efraimidis–Spirakis)
        self.encounter_key = ti.field(ti.i64, shape=(world.w, world.d))
        self.slot_bits = max(1, (cap - 1).bit_length())
        self.act = ti.Vector.field(N_OUT, ti.f32, shape=cap)  # brain outputs
        self.top = ti.field(ti.i32, shape=())
        self.next_uid = ti.field(ti.i64, shape=())
        # slot lottery when divisions outnumber free slots: Σ cells of dividing slots, free slots
        self.div_cells = ti.field(ti.f64, shape=())
        self.free_at_start = ti.field(ti.f64, shape=())
        self.div_slots = ti.field(ti.f64, shape=())

        self.gene_lo.from_numpy(np.array([r[0] for r in GENE_RANGE], np.float32))
        self.gene_hi.from_numpy(np.array([r[1] for r in GENE_RANGE], np.float32))
        self._place_founders(np.random.default_rng(seed))

    def _place_founders(self, rng: np.random.Generator):
        cfg, w = self.cfg, self.world
        n = cfg.n_founders
        pos = np.zeros((self.cap, 2), np.float32)
        for k in range(n):
            vx, vy = w.vents_np[k % len(w.vents_np)]
            while True:
                x, y = vx + rng.uniform(-3, 4), vy + rng.uniform(-3, 1)
                if 0 <= x < w.w and 0 <= y < w.floor_np[int(x)]:
                    break
            pos[k] = (x, y)
        state = np.zeros(self.cap, np.int32)
        state[:n] = ALIVE
        body = np.zeros(self.cap)
        body[:n] = cfg.founder_body
        energy = np.zeros(self.cap, np.float32)
        energy[:n] = 0.5 * cfg.energy_cap * cfg.founder_body
        genome = np.zeros((self.cap, N_GENES), np.float32)
        genome[:n] = FOUNDER
        uid = np.zeros(self.cap, np.int64)
        uid[:n] = np.arange(1, n + 1)

        self.state.from_numpy(state)
        self.pos.from_numpy(pos)
        self.body.from_numpy(body)
        self.energy.from_numpy(energy)
        self.ncell.from_numpy((state == ALIVE).astype(np.float32))
        self.genome.from_numpy(genome)
        self.uid.from_numpy(uid)
        self.parent.from_numpy(np.zeros(self.cap, np.int64))
        self.free.from_numpy(np.arange(self.cap - 1, -1, -1, dtype=np.int32))
        self.top[None] = self.cap - n
        self.next_uid[None] = n + 1

    def count(self) -> int:
        """Occupied slots."""
        return self.cap - self.top[None]

    def cells(self) -> float:
        """Cells represented by all slots (Σ ncell)."""
        return float(self.ncell.to_numpy()[self.state.to_numpy() == ALIVE].sum())

    @ti.func
    def take(self, s: int, x: int, y: int, amount: CHEM_DTYPE) -> CHEM_DTYPE:
        """Atomically withdraw up to `amount` from a cell; never leaves it negative beyond f64
        rounding (~-1e-18) from the subtract-then-refund."""
        got = ti.cast(0.0, CHEM_DTYPE)
        if amount > 0:
            old = ti.atomic_sub(self.world.chem[s, x, y], amount)
            got = ti.min(amount, ti.max(old, 0.0))
            if got < amount:
                ti.atomic_add(self.world.chem[s, x, y], amount - got)
        return got

    @ti.func
    def give(self, s: int, x: int, y: int, amount: CHEM_DTYPE):
        ti.atomic_add(self.world.chem[s, x, y], amount)

    @ti.func
    def cell(self, i: int) -> ti.math.ivec2:
        return ti.cast(self.pos[i], ti.i32)

    @ti.kernel
    def move(self):
        for i in range(self.cap):
            if self.state[i] == ALIVE:
                p = self.pos[i]
                c = self.cell(i)
                # cell-centre velocity from the overturning streamfunction (cells per tick)
                u = 0.5 * ((self.world.psi[c.x, c.y + 1] - self.world.psi[c.x, c.y])
                           + (self.world.psi[c.x + 1, c.y + 1] - self.world.psi[c.x + 1, c.y]))
                v = -0.5 * ((self.world.psi[c.x + 1, c.y] - self.world.psi[c.x, c.y])
                            + (self.world.psi[c.x + 1, c.y + 1] - self.world.psi[c.x, c.y + 1]))
                # active swimming from the brain, paid as k_move × speed² × body
                swim = self.genome[i, SPEED] * ti.math.vec2(self.act[i][ACT_X], self.act[i][ACT_Y])
                self.energy[i] -= (self.cfg.k_move * swim.dot(swim)
                                   * ti.cast(self.body[i], ti.f32))
                # cells in the bottom water row stick to the sediment (benthic mats): no passive
                # drift there, but swimming can still carry them off
                drift = ti.math.vec2(u, v) + self.cfg.drift_sigma * ti.math.vec2(ti.randn(), ti.randn())
                if c.y == self.world.floor[c.x] - 1:
                    drift = ti.math.vec2(0.0, 0.0)
                q = p + drift + swim
                if q.x >= 0 and q.y >= 0 and self.world.is_water(ti.cast(q.x, ti.i32), ti.cast(q.y, ti.i32)):
                    self.pos[i] = q

    @ti.kernel
    def census(self):
        """Per-grid-cell cell count, biomass and one encounter slot drawn with probability
        proportional to its cells, independent of slot order (stage 3b, decision 1)."""
        for x, y in self.cell_cells:
            self.cell_cells[x, y] = 0.0
            self.cell_mass[x, y] = 0.0
            self.encounter_key[x, y] = ti.i64(9223372036854775807)
        for i in range(self.cap):
            self.event[i] = 0
            if self.state[i] == ALIVE:
                c = self.cell(i)
                ti.atomic_add(self.cell_cells[c.x, c.y], self.ncell[i])
                ti.atomic_add(self.cell_mass[c.x, c.y], ti.cast(self.body[i], ti.f32))
                u = ti.max(ti.random(ti.f64), 1e-300)
                q = ti.min(-ti.log(u) / ti.cast(self.ncell[i], ti.f64) * 1e9, 1e12)
                key = (ti.cast(q, ti.i64) << self.slot_bits) | ti.cast(i, ti.i64)
                ti.atomic_min(self.encounter_key[c.x, c.y], key)
        for x, y in self.encounter:
            k = self.encounter_key[x, y]
            j = -1
            if k != ti.i64(9223372036854775807):
                j = ti.cast(k & ((ti.i64(1) << self.slot_bits) - 1), ti.i32)
            self.encounter[x, y] = j

    @ti.kernel
    def think(self):
        """Brain 8 -> 8 -> 4 (tanh) from genome weights; writes act[i]."""
        for i in range(self.cap):
            if self.state[i] == ALIVE:
                c = self.cell(i)
                x, y = c.x, c.y
                body = ti.max(ti.cast(self.body[i], ti.f32), 1e-12)
                size = body / self.ncell[i]
                mean_size = self.cell_mass[x, y] / ti.max(self.cell_cells[x, y], 1.0)
                dt = (self.world.temp[x, y] - self.genome[i, T_OPT]) / self.genome[i, T_WIDTH]
                inp = ti.Vector([
                    self.world.light[x, y],
                    ti.cast(self.sat(R, x, y), ti.f32),
                    ti.cast(self.sat(O, x, y), ti.f32),
                    dt,
                    ti.log(1.0 + self.cell_cells[x, y]),
                    ti.log(size / ti.max(mean_size, 1e-9)),
                    self.energy[i] / (self.cfg.energy_cap * body),
                    ti.cast(self.world.floor[x] - y, ti.f32) / self.world.d,
                ])
                hid = ti.Vector.zero(ti.f32, N_HID)
                for h in ti.static(range(N_HID)):
                    z = self.genome[i, BRAIN + N_HID * N_IN + h]
                    for k in ti.static(range(N_IN)):
                        z += self.genome[i, BRAIN + N_IN * h + k] * inp[k]
                    hid[h] = ti.tanh(z)
                for o in ti.static(range(N_OUT)):
                    z = 0.0
                    for h in ti.static(range(N_HID)):
                        z += self.genome[i, BRAIN + N_HID * N_IN + N_HID + N_HID * o + h] * hid[h]
                    self.act[i][o] = ti.tanh(z)

    @ti.kernel
    def predate(self):
        """One bite per tick from this grid cell's encounter (stage 3b, decision 2). Body moves only
        between slots and detritus, so C, N and electrons balance by construction."""
        for i in range(self.cap):
            if (self.state[i] == ALIVE and self.genome[i, ETA_PRED] > 0
                    and self.act[i][ACT_HUNT] > ACT_ON):
                c = self.cell(i)
                j = self.encounter[c.x, c.y]
                if j >= 0 and j != i and self.state[j] == ALIVE:
                    size_i = self.body[i] / self.ncell[i]
                    size_j = self.body[j] / self.ncell[j]
                    if self.is_colony(j):
                        size_j = self.body[j]  # a colony is too big to swallow cell by cell
                    # a cell at its growth cap (twice division size) stops eating, like fixation
                    if (size_i > self.genome[i, PREY_RATIO] * size_j
                            and size_i < 2.0 * self.genome[i, DIV_SIZE]):
                        want = self.cfg.k_pred * self.genome[i, ETA_PRED] * self.body[i]
                        old = ti.atomic_sub(self.body[j], want)
                        got = ti.min(want, ti.max(old, 0.0))
                        if got < want:
                            ti.atomic_add(self.body[j], want - got)
                        if old - got <= self.cfg.min_body:
                            self.state[j] = DYING  # eaten: die() turns what is left into detritus
                        if got > 0:
                            # prey super-individual loses cells in proportion (count, not ledger)
                            self.ncell[j] = ti.max(self.ncell[j] * ti.cast(1.0 - got / old, ti.f32),
                                                   1.0)
                            ti.atomic_or(self.event[i], 1)
                            ti.atomic_or(self.event[j], 2)
                            keep = CHEM_DTYPE(self.cfg.assimilation) * got
                            ti.atomic_add(self.body[i], keep)
                            self.give(D, c.x, c.y, got - keep)

    @ti.func
    def is_colony(self, i: int) -> bool:
        """A colony is a slot of several cells whose genes make them stick (adhesion ≥ 0.5). A
        full-pool super-individual of non-sticky cells is a computational group, not a colony."""
        return self.ncell[i] > 1.0 and self.genome[i, ADHESION] >= COLONY_ADHESION

    def colonies(self) -> np.ndarray:
        """Per-slot colony flag (numpy), for stats and tests."""
        return ((self.state.to_numpy() == ALIVE) & (self.ncell.to_numpy() > 1.0)
                & (self.genome.to_numpy()[:, ADHESION] >= COLONY_ADHESION))

    @ti.func
    def sat(self, s: int, x: int, y: int) -> CHEM_DTYPE:
        # racy read can be ~-1e-18, so clamp before the saturation term
        v = ti.max(self.world.chem[s, x, y], 0.0)
        return v / (v + self.cfg.k_half)

    @ti.func
    def run(self, rx: ti.template(), x: int, y: int, ext: CHEM_DTYPE) -> CHEM_DTYPE:
        """Run up to `ext` units of reaction `rx` against cell (x, y): withdraw every cell
        reactant, keep only what the scarcest one allows, refund the rest, release products.
        Returns the achieved extent; the caller applies the body term BODY[rx] x extent."""
        f = ti.cast(0.0, CHEM_DTYPE)
        if ext > 0:
            f = 1.0
            got = ti.Vector.zero(CHEM_DTYPE, 3)
            # CHEM_DTYPE(k): a bare Python float in a kernel is f32 even next to an f64, and
            # 1/CN_RATIO rounded to f32 breaks the N budget
            for j in ti.static(range(len(INPUTS[rx]))):
                want = CHEM_DTYPE(INPUTS[rx][j][1]) * ext
                got[j] = self.take(ti.static(INPUTS[rx][j][0]), x, y, want)
                f = ti.min(f, got[j] / want)
            for j in ti.static(range(len(INPUTS[rx]))):
                self.give(ti.static(INPUTS[rx][j][0]), x, y,
                          got[j] - CHEM_DTYPE(INPUTS[rx][j][1]) * ext * f)
            for j in ti.static(range(len(OUTPUTS[rx]))):
                self.give(ti.static(OUTPUTS[rx][j][0]), x, y,
                          CHEM_DTYPE(OUTPUTS[rx][j][1]) * ext * f)
        return ext * f

    @ti.kernel
    def metabolize(self):
        for x, y in self.cell_body:
            self.cell_body[x, y] = 0.0
        for i in range(self.cap):
            # only pigmented (photosynthetic) biomass absorbs light; other cells are transparent
            if self.state[i] == ALIVE and self.genome[i, ETA_PHOTO_AN] + self.genome[i, ETA_PHOTO_OX] > 0:
                c = self.cell(i)
                ti.atomic_add(self.cell_body[c.x, c.y], ti.cast(self.body[i], ti.f32))
        for i in range(self.cap):
            if self.state[i] == ALIVE:
                c = self.cell(i)
                x, y = c.x, c.y
                b = self.body[i]
                nc = ti.cast(self.ncell[i], CHEM_DTYPE)
                # a cell stops fixing carbon at twice its division size (safety net; division into
                # the super-individual normally happens first)
                grow = ti.cast(b < 2.0 * self.genome[i, DIV_SIZE] * nc, CHEM_DTYPE)
                e = ti.cast(self.energy[i], CHEM_DTYPE)
                eta_h = self.genome[i, ETA_H2]
                eta_m = self.genome[i, ETA_METH]
                eta_a = self.genome[i, ETA_ANAER]
                eta_pa = self.genome[i, ETA_PHOTO_AN]
                eta_po = self.genome[i, ETA_PHOTO_OX]
                eta_ar = self.genome[i, ETA_AER]
                eta_sf = self.genome[i, ETA_SFIX]
                eta_so = self.genome[i, ETA_SOX]
                eta_mo = self.genome[i, ETA_MOX]
                tol = self.genome[i, O2_TOL]
                dt = (self.world.temp[x, y] - self.genome[i, T_OPT]) / self.genome[i, T_WIDTH]
                f_t = ti.exp(-dt * dt)

                # carbon fixation: H2 chemosynthesis (2H + C -> body)
                r = self.run(RX_CHEMO, x, y,
                             grow * self.cfg.k_chemo * eta_h * f_t * b * self.sat(H, x, y))
                b += BODY[RX_CHEMO] * r
                e += G[RX_CHEMO] * eta_h * r

                # photosynthesis: absorbed light pays ΔG/η, split between the two pathways. The cell's
                # biomass absorbs L(1 − exp(−k·B)) (Beer–Lambert, never more than the incident L),
                # shared in proportion to body
                if eta_pa + eta_po > 0:
                    bc = ti.max(self.cell_body[x, y], ti.cast(self.body[i], ti.f32), 1e-12)
                    absorbed = self.world.light[x, y] * (1.0 - ti.exp(-self.cfg.k_light * bc))
                    lam = grow * absorbed * self.body[i] / bc * f_t / (eta_pa + eta_po)
                    r = self.run(RX_PHOTO_AN, x, y,
                                 eta_pa * eta_pa * lam / -G[RX_PHOTO_AN] * self.sat(R, x, y))
                    r2 = self.run(RX_PHOTO_OX, x, y, eta_po * eta_po * lam / -G[RX_PHOTO_OX])
                    b += BODY[RX_PHOTO_AN] * r + BODY[RX_PHOTO_OX] * r2

                # energy from O2-dependent lithotrophy, up to the storage cap
                if eta_so > 0:
                    room = ti.max(self.cfg.energy_cap * b - e, 0.0)
                    r = self.run(RX_SOX, x, y, ti.min(
                        self.cfg.k_chemo * eta_so * f_t * b * self.sat(R, x, y) * self.sat(O, x, y),
                        room / (G[RX_SOX] * eta_so)))
                    e += G[RX_SOX] * eta_so * r
                if eta_mo > 0:
                    room = ti.max(self.cfg.energy_cap * b - e, 0.0)
                    r = self.run(RX_MOX, x, y, ti.min(
                        self.cfg.k_chemo * eta_mo * f_t * b * self.sat(M, x, y) * self.sat(O, x, y),
                        room / (G[RX_MOX] * eta_mo)))
                    e += G[RX_MOX] * eta_mo * r

                # sulfur carbon fixation, paid from stored energy above the reserve
                if eta_sf > 0:
                    avail = ti.max(e - self.cfg.reserve_frac * self.cfg.energy_cap * b, 0.0)
                    r = self.run(RX_SFIX, x, y, ti.min(
                        grow * self.cfg.k_chemo * eta_sf * f_t * b * self.sat(R, x, y),
                        avail * eta_sf / -G[RX_SFIX]))
                    b += BODY[RX_SFIX] * r
                    e += G[RX_SFIX] / eta_sf * r

                o2 = ti.max(self.world.chem[O, x, y], 0.0)
                e -= (self.cfg.maintenance
                      + self.cfg.trait_upkeep * (eta_h + eta_m + eta_a + eta_pa + eta_po + eta_ar
                                                 + eta_sf + eta_so + eta_mo + tol
                                                 + self.genome[i, ETA_PRED]
                                                 + self.genome[i, ADHESION])
                      + self.cfg.k_tox * o2 * (1.0 - tol)) * b

                # respire body when short of energy, highest yield first: O2 > X > methanogenic
                need = self.cfg.reserve_frac * self.cfg.energy_cap * b - e
                spare = ti.max(b - self.cfg.min_body, 0.0)
                if need > 0 and eta_ar > 0:
                    r = self.run(RX_AER, x, y, ti.min(
                        self.cfg.k_resp * eta_ar * b * self.sat(O, x, y),
                        ti.min(need / (G[RX_AER] * eta_ar), spare / -BODY[RX_AER])))
                    b += BODY[RX_AER] * r
                    spare += BODY[RX_AER] * r
                    e += G[RX_AER] * eta_ar * r
                    need -= G[RX_AER] * eta_ar * r
                if need > 0 and eta_a > 0:
                    r = self.run(RX_ANAER, x, y, ti.min(
                        self.cfg.k_resp * eta_a * b * self.sat(X, x, y),
                        ti.min(need / (G[RX_ANAER] * eta_a), spare / -BODY[RX_ANAER])))
                    b += BODY[RX_ANAER] * r
                    spare += BODY[RX_ANAER] * r
                    e += G[RX_ANAER] * eta_a * r
                    need -= G[RX_ANAER] * eta_a * r
                if need > 0 and eta_m > 0:
                    r = self.run(RX_METH, x, y, ti.min(
                        self.cfg.k_resp * eta_m * b,
                        ti.min(need / (G[RX_METH] * eta_m), spare / -BODY[RX_METH])))
                    b += BODY[RX_METH] * r
                    e += G[RX_METH] * eta_m * r

                e = ti.min(e, self.cfg.energy_cap * b)
                self.age[i] += 1
                self.body[i] = b
                self.energy[i] = ti.cast(e, ti.f32)
                dead = e < 0 or b <= self.cfg.min_body
                if nc > 1.0 and not dead:
                    # super-individual: background mortality removes that fraction of its cells
                    loss = self.cfg.death_rate * b
                    self.give(D, x, y, loss)
                    self.body[i] = b - loss
                    self.energy[i] = ti.cast(e * (1.0 - self.cfg.death_rate), ti.f32)
                    self.ncell[i] = ti.max(self.ncell[i] * (1.0 - self.cfg.death_rate), 1.0)
                elif ti.random() < self.cfg.death_rate:
                    dead = True
                if dead:
                    self.state[i] = DYING
                elif (b >= self.genome[i, DIV_SIZE] * nc and self.act[i][ACT_HOLD] <= ACT_ON
                      and e >= self.cfg.divide_energy_frac * self.cfg.energy_cap * b):
                    self.want_div[i] = 1

    @ti.kernel
    def die(self):
        """Dying bodies become detritus; slots return to the free-list (push-only kernel)."""
        for i in range(self.cap):
            if self.state[i] == DYING:
                c = self.cell(i)
                self.give(D, c.x, c.y, self.body[i])
                self.body[i] = 0.0
                self.energy[i] = 0.0
                self.ncell[i] = 0.0
                self.want_div[i] = 0
                self.state[i] = DEAD
                self.free[ti.atomic_add(self.top[None], 1)] = i

    @ti.kernel
    def weigh_divisions(self):
        self.div_cells[None] = 0.0
        self.div_slots[None] = 0.0
        self.free_at_start[None] = ti.cast(self.top[None], ti.f64)
        for i in range(self.cap):
            if self.state[i] == ALIVE and self.want_div[i] == 1:
                self.div_cells[None] += ti.cast(self.ncell[i], ti.f64)
                self.div_slots[None] += 1.0

    @ti.kernel
    def reproduce(self):
        """Division into two halves with mutation (pop-only kernel). When dividing slots outnumber
        free slots (otherwise every one gets a slot), each gets a new slot with probability free × ncell / Σ ncell, so lineages share
        free slots by how many cells they really have, not by how many slots they hold (stage 3a,
        decision 7). The others divide inside the super-individual."""
        for i in range(self.cap):
            if self.state[i] == ALIVE and self.want_div[i] == 1:
                self.want_div[i] = 0
                p_slot = (self.free_at_start[None] * ti.cast(self.ncell[i], ti.f64)
                          / ti.max(self.div_cells[None], 1e-12))
                if self.free_at_start[None] >= self.div_slots[None]:
                    p_slot = 1.0  # enough free slots for every dividing slot
                old = 0
                stay = ti.random() < self.genome[i, ADHESION]  # daughter stays bound: colony
                if not stay and (p_slot >= 1.0 or ti.random(ti.f64) < p_slot):
                    old = ti.atomic_sub(self.top[None], 1)
                    if old <= 0:
                        ti.atomic_add(self.top[None], 1)
                if stay or old <= 0:
                    # bound by adhesion, or no free slot: the cells divide inside the slot
                    self.ncell[i] *= 2.0
                else:
                    j = self.free[old - 1]
                    half = 0.5 * self.body[i]
                    self.body[j] = half
                    self.body[i] -= half
                    e_half = 0.5 * self.energy[i]
                    self.energy[j] = e_half
                    self.energy[i] -= e_half
                    self.age[j] = 0
                    self.act[j] = ti.Vector.zero(ti.f32, N_OUT)
                    n_half = ti.max(0.5 * self.ncell[i], 1.0)
                    self.ncell[i] = n_half
                    self.ncell[j] = n_half
                    self.species[j] = self.species[i]
                    self.want_div[j] = 0
                    self.uid[j] = ti.atomic_add(self.next_uid[None], 1)
                    self.parent[j] = self.uid[i]
                    q = self.pos[i] + ti.math.vec2(ti.random() - 0.5, ti.random() - 0.5)
                    if q.x >= 0 and q.y >= 0 and self.world.is_water(ti.cast(q.x, ti.i32),
                                                                     ti.cast(q.y, ti.i32)):
                        self.pos[j] = q
                    else:
                        self.pos[j] = self.pos[i]
                    sigma = self.genome[i, MUT_RATE]
                    for g in range(N_GENES):
                        lo, hi = self.gene_lo[g], self.gene_hi[g]
                        v = self.genome[i, g] + ti.randn() * sigma * (hi - lo)
                        self.genome[j, g] = ti.min(ti.max(v, lo), hi)
                    if ti.random() < self.cfg.big_mutation_p:
                        # trait genes only, so their odds don't shrink as the brain adds genes
                        g = ti.min(ti.cast(ti.random() * (BRAIN + 1), ti.i32), BRAIN)
                        if g == BRAIN:
                            g = ADHESION  # the one trait gene that sits after the brain
                        self.genome[j, g] = self.gene_lo[g] + ti.random() * (
                            self.gene_hi[g] - self.gene_lo[g])
                    self.state[j] = ALIVE

    def step(self):
        self.move()
        self.census()
        self.think()
        self.predate()
        self.metabolize()
        self.die()
        self.weigh_divisions()
        self.reproduce()

    def snapshot(self) -> dict:
        alive = self.state.to_numpy() == ALIVE
        return {"alive": alive, "body": self.body.to_numpy()[alive],
                "genome": self.genome.to_numpy()[alive], "ncell": self.ncell.to_numpy()[alive]}
