import numpy as np
import taichi as ti

from evo.config import CN_RATIO, C, D, H, LifeConfig, M, N, R, X
from evo.life.genome import (
    DIV_SIZE,
    ETA_ANAER,
    ETA_H2,
    ETA_METH,
    FOUNDER,
    GENE_RANGE,
    MUT_RATE,
    N_GENES,
    T_OPT,
    T_WIDTH,
)
from evo.world.fields import CHEM_DTYPE, World
from evo.world.reactions import REACTIONS, delta_g

DEAD, ALIVE, DYING = 0, 1, 2

# energy released per unit reaction extent (−ΔG from the μ table, §2.3)
G_CHEMO = -delta_g(REACTIONS["h2_chemosynthesis"])
G_METH = -delta_g(REACTIONS["methanogenesis"])
G_ANAER = -delta_g(REACTIONS["anaerobic_respiration"])


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
        self.uid = ti.field(ti.i64, shape=cap)
        self.parent = ti.field(ti.i64, shape=cap)
        self.genome = ti.field(ti.f32, shape=(cap, N_GENES))
        self.gene_lo = ti.field(ti.f32, shape=N_GENES)
        self.gene_hi = ti.field(ti.f32, shape=N_GENES)
        self.free = ti.field(ti.i32, shape=cap)
        self.top = ti.field(ti.i32, shape=())
        self.next_uid = ti.field(ti.i64, shape=())

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
        self.genome.from_numpy(genome)
        self.uid.from_numpy(uid)
        self.parent.from_numpy(np.zeros(self.cap, np.int64))
        self.free.from_numpy(np.arange(self.cap - 1, -1, -1, dtype=np.int32))
        self.top[None] = self.cap - n
        self.next_uid[None] = n + 1

    def count(self) -> int:
        return self.cap - self.top[None]

    @ti.func
    def take(self, s: int, x: int, y: int, amount: CHEM_DTYPE) -> CHEM_DTYPE:
        """Atomically withdraw up to `amount` from a cell; never leaves it negative."""
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
                q = p + ti.math.vec2(u, v) + self.cfg.drift_sigma * ti.math.vec2(ti.randn(), ti.randn())
                if q.x >= 0 and q.y >= 0 and self.world.is_water(ti.cast(q.x, ti.i32), ti.cast(q.y, ti.i32)):
                    self.pos[i] = q

    @ti.kernel
    def metabolize(self):
        for i in range(self.cap):
            if self.state[i] == ALIVE:
                c = self.cell(i)
                x, y = c.x, c.y
                b = self.body[i]
                e = ti.cast(self.energy[i], CHEM_DTYPE)
                eta_h = self.genome[i, ETA_H2]
                eta_m = self.genome[i, ETA_METH]
                eta_a = self.genome[i, ETA_ANAER]
                dt = (self.world.temp[x, y] - self.genome[i, T_OPT]) / self.genome[i, T_WIDTH]
                f_t = ti.exp(-dt * dt)

                # H2 chemosynthesis: 2H + C (+N for growth) -> body
                h = self.world.chem[H, x, y]
                want = self.cfg.k_chemo * eta_h * f_t * b * h / (h + self.cfg.k_half)
                r = 0.5 * self.take(H, x, y, 2.0 * want)
                rc = self.take(C, x, y, r)
                if rc < r:
                    self.give(H, x, y, 2.0 * (r - rc))
                    r = rc
                rn = self.take(N, x, y, r / CN_RATIO) * CN_RATIO
                if rn < r:
                    self.give(H, x, y, 2.0 * (r - rn))
                    self.give(C, x, y, r - rn)
                    r = rn
                b += r
                e += G_CHEMO * eta_h * r

                e -= (self.cfg.maintenance + self.cfg.trait_upkeep * (eta_h + eta_m + eta_a)) * b

                need = self.cfg.reserve_frac * self.cfg.energy_cap * b - e
                spare = ti.max(b - self.cfg.min_body, 0.0)
                if need > 0 and eta_a > 0:
                    # anaerobic respiration: body + 2X -> C + 2R
                    xx = self.world.chem[X, x, y]
                    ext = ti.min(self.cfg.k_resp * eta_a * b * xx / (xx + self.cfg.k_half),
                                 ti.min(need / (G_ANAER * eta_a), spare))
                    got = 0.5 * self.take(X, x, y, 2.0 * ext)
                    b -= got
                    spare -= got
                    self.give(C, x, y, got)
                    self.give(R, x, y, 2.0 * got)
                    self.give(N, x, y, got / CN_RATIO)
                    e += G_ANAER * eta_a * got
                    need -= G_ANAER * eta_a * got
                if need > 0 and eta_m > 0:
                    # methanogenesis: 2 body -> C + M
                    ext = ti.min(self.cfg.k_resp * eta_m * b,
                                 ti.min(need / (G_METH * eta_m), 0.5 * spare))
                    b -= 2.0 * ext
                    self.give(C, x, y, ext)
                    self.give(M, x, y, ext)
                    self.give(N, x, y, 2.0 * ext / CN_RATIO)
                    e += G_METH * eta_m * ext

                e = ti.min(e, self.cfg.energy_cap * b)
                self.age[i] += 1
                self.body[i] = b
                self.energy[i] = ti.cast(e, ti.f32)
                if e < 0 or b <= self.cfg.min_body or ti.random() < self.cfg.death_rate:
                    self.state[i] = DYING
                elif (b >= self.genome[i, DIV_SIZE]
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
                self.want_div[i] = 0
                self.state[i] = DEAD
                self.free[ti.atomic_add(self.top[None], 1)] = i

    @ti.kernel
    def reproduce(self):
        """Division into two halves with mutation (pop-only kernel)."""
        for i in range(self.cap):
            if self.state[i] == ALIVE and self.want_div[i] == 1:
                self.want_div[i] = 0
                old = ti.atomic_sub(self.top[None], 1)
                if old <= 0:
                    ti.atomic_add(self.top[None], 1)  # pool full: division fails
                else:
                    j = self.free[old - 1]
                    half = 0.5 * self.body[i]
                    self.body[j] = half
                    self.body[i] -= half
                    e_half = 0.5 * self.energy[i]
                    self.energy[j] = e_half
                    self.energy[i] -= e_half
                    self.age[j] = 0
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
                        g = ti.min(ti.cast(ti.random() * N_GENES, ti.i32), N_GENES - 1)
                        self.genome[j, g] = self.gene_lo[g] + ti.random() * (
                            self.gene_hi[g] - self.gene_lo[g])
                    self.state[j] = ALIVE

    def step(self):
        self.move()
        self.metabolize()
        self.die()
        self.reproduce()

    def snapshot(self) -> dict:
        alive = self.state.to_numpy() == ALIVE
        return {"alive": alive, "body": self.body.to_numpy()[alive],
                "genome": self.genome.to_numpy()[alive]}
