"""Multicellular organisms (stage 7b; stage7b-arch.md, "7b 구현 결정").

An organism is a chain of segments, head first; each segment is one colony slot of the cell pool
(Agents), so its body carbon stays in Agents.body and the element ledger is unchanged. This module
keeps the chain (slot and uid per segment) and does what only a body can do:

- tissue: segment k's tissue shares follow the body axis (Hox-like genes); only the metabolic share
  feeds (Agents.metab_frac)
- share_energy: circulation, stored energy shared in proportion to segment body
- swim: low-Reynolds locomotion by resistive force theory (no inertia: the drag forces and torques of
  all segments sum to zero), muscles bending each joint in a myogenic rhythm, cilia pushing along
  the body; the work done is paid from stored energy
- develop: the tail buds new segments up to N_SEG, germ tissue releases single mutated cells (the
  next generation starts as one cell), other segments grow their cells up to SEG_CELLS
- integrity: a dead segment cuts the body there (the rest falls off as colonies); a dead head
  dissolves the organism
- transition: a differentiating colony (DIFF ≥ 0.5) of org_min_cells cells becomes an organism

Free-lists: develop and transition pop, integrity pushes (never both in one kernel, §4.6)."""
import math

import numpy as np
import taichi as ti

from evo.life.agents import ACT_ON, ACT_TUMBLE, ALIVE, COLONY_ADHESION, N_OUT
from evo.life.genome import (
    ADHESION,
    CPG_AMP,
    CPG_FREQ,
    CPG_LAG,
    DIFF,
    N_GENES,
    N_SEG,
    SEG_CELLS,
    TISSUE,
)
from evo.world.fields import CHEM_DTYPE

MAX_SEG = 12
O_DEAD, O_ALIVE = 0, 1
T_METAB, T_MUSCLE, T_CILIA, T_GERM = range(4)
GERM_ON = 0.5  # a segment mostly of germ tissue releases single cells instead of growing
DIFF_ON = 0.5
TWO_PI = 2.0 * math.pi


@ti.data_oriented
class Organisms:
    def __init__(self, agents):
        self.a = agents
        self.cfg = agents.cfg
        cap = self.cfg.org_capacity
        self.cap = cap
        self.state = ti.field(ti.i32, shape=cap)
        self.n = ti.field(ti.i32, shape=cap)  # segments now
        self.slot = ti.field(ti.i32, shape=(cap, MAX_SEG))
        self.suid = ti.field(ti.i64, shape=(cap, MAX_SEG))  # slot uid, to notice a reused slot
        self.head = ti.Vector.field(2, ti.f32, shape=cap)  # head position (world)
        self.psi = ti.field(ti.f32, shape=cap)  # orientation of the body frame (radians)
        self.phase = ti.field(ti.f32, shape=cap)  # muscle rhythm phase (cycles)
        self.turn = ti.field(ti.i32, shape=cap)  # ticks left of a turn
        self.turn_sign = ti.field(ti.f32, shape=cap)
        self.uid = ti.field(ti.i64, shape=cap)
        self.travel = ti.field(ti.f32, shape=cap)  # distance the head has swum (cells)
        self.free = ti.field(ti.i32, shape=cap)
        self.top = ti.field(ti.i32, shape=())
        self.next_uid = ti.field(ti.i64, shape=())
        self.free.from_numpy(np.arange(cap - 1, -1, -1, dtype=np.int32))
        self.top[None] = cap
        self.next_uid[None] = 1

    def count(self) -> int:
        return self.cap - self.top[None]

    @ti.func
    def shares(self, i: int, k: int) -> ti.math.vec4:
        """Tissue shares (metabolism, muscle, cilia, germ) of segment k, from slot i's genome:
        softmax over position s = k / (N_SEG − 1) along the body (head 0, tail 1)."""
        nseg = ti.max(ti.round(self.a.genome[i, N_SEG]), 1.0)
        s = ti.cast(k, ti.f32) / ti.max(nseg - 1.0, 1.0)
        z1 = self.a.genome[i, TISSUE] + self.a.genome[i, TISSUE + 1] * s
        z2 = self.a.genome[i, TISSUE + 2] + self.a.genome[i, TISSUE + 3] * s
        z3 = self.a.genome[i, TISSUE + 4] + self.a.genome[i, TISSUE + 5] * s
        m = ti.max(0.0, z1, z2, z3)
        e = ti.math.vec4(ti.exp(-m), ti.exp(z1 - m), ti.exp(z2 - m), ti.exp(z3 - m))
        return e / e.sum()

    @ti.func
    def target(self, i: int) -> int:
        return ti.min(ti.max(ti.cast(ti.round(self.a.genome[i, N_SEG]), ti.i32), 1), MAX_SEG)

    @ti.kernel
    def tissue(self):
        for o in range(self.cap):
            if self.state[o] == O_ALIVE:
                for k in range(self.n[o]):
                    i = self.slot[o, k]
                    self.a.metab_frac[i] = self.shares(i, k)[T_METAB]

    @ti.kernel
    def share_energy(self):
        """Circulation: the body's stored energy is shared in proportion to segment body."""
        for o in range(self.cap):
            if self.state[o] == O_ALIVE:
                e = 0.0
                b = 0.0
                for k in range(self.n[o]):
                    i = self.slot[o, k]
                    e += self.a.energy[i]
                    b += ti.cast(self.a.body[i], ti.f32)
                for k in range(self.n[o]):
                    i = self.slot[o, k]
                    self.a.energy[i] = e * ti.cast(self.a.body[i], ti.f32) / ti.max(b, 1e-12)

    @ti.func
    def shape(self, o: int, phase: ti.f32, bias: ti.f32):
        """Body-frame segment positions and tangents (head at the origin facing +x): joint k bends
        by muscle share × (A·sin(2π·phase − lag·k) + bias)."""
        h = self.slot[o, 0]
        amp, lag = self.a.genome[h, CPG_AMP], self.a.genome[h, CPG_LAG]
        rx = ti.Vector.zero(ti.f32, MAX_SEG)
        ry = ti.Vector.zero(ti.f32, MAX_SEG)
        tx = ti.Vector.zero(ti.f32, MAX_SEG)
        ty = ti.Vector.zero(ti.f32, MAX_SEG)
        tx[0] = 1.0
        ang = 0.0
        for k in ti.static(range(1, MAX_SEG)):
            if k < self.n[o]:
                mus = self.shares(self.slot[o, k], k)[T_MUSCLE]
                ang += mus * (amp * ti.sin(TWO_PI * phase - lag * k) + bias)
                tx[k] = ti.cos(ang)
                ty[k] = ti.sin(ang)
                rx[k] = rx[k - 1] - self.cfg.seg_len * tx[k]
                ry[k] = ry[k - 1] - self.cfg.seg_len * ty[k]
        return rx, ry, tx, ty

    @ti.kernel
    def swim(self):
        """Resistive force theory: segment k moves at v_k = V + Ω·perp(r_k) + u_k (frame velocity,
        rotation, shape change); drag −D_k·v_k with D = c_t·t·tᵀ + c_n·n·nᵀ, cilia push along t.
        Σ forces = 0 and Σ torques = 0 give V and Ω (3 × 3)."""
        for o in range(self.cap):
            if self.state[o] == O_ALIVE:
                h = self.slot[o, 0]
                freq = self.a.genome[h, CPG_FREQ]
                # a turn: the head's (stage 7a) brain asks for one and the body bends to one side
                if self.turn[o] == 0 and self.a.act[h][ACT_TUMBLE] > ACT_ON:
                    self.turn[o] = self.cfg.turn_ticks
                    self.turn_sign[o] = 1.0 if ti.random() < 0.5 else -1.0
                bias = 0.0
                if self.turn[o] > 0:
                    bias = self.cfg.turn_bend * self.turn_sign[o]
                    self.turn[o] -= 1
                ph = self.phase[o] + freq
                rx, ry, tx, ty = self.shape(o, ph, bias)
                px, py, _, _ = self.shape(o, self.phase[o], bias)
                self.phase[o] = ph - ti.floor(ph)
                A = ti.Matrix.zero(ti.f32, 3, 3)
                rhs = ti.Vector.zero(ti.f32, 3)
                ct, cn, ll = self.cfg.drag_t, self.cfg.drag_n, self.cfg.seg_len
                for k in ti.static(range(MAX_SEG)):
                    if k < self.n[o]:
                        t = ti.math.vec2(tx[k], ty[k])
                        nn = ti.math.vec2(-t.y, t.x)
                        D = ct * t.outer_product(t) + cn * nn.outer_product(nn)
                        p = ti.math.vec2(-ry[k], rx[k])
                        u = ti.math.vec2(rx[k] - px[k], ry[k] - py[k])
                        f = self.cfg.k_cilia * self.shares(self.slot[o, k], k)[T_CILIA] * t
                        Dp = D @ p
                        g = f - D @ u
                        A[0, 0] += D[0, 0]
                        A[0, 1] += D[0, 1]
                        A[1, 0] += D[1, 0]
                        A[1, 1] += D[1, 1]
                        A[0, 2] += Dp[0]
                        A[1, 2] += Dp[1]
                        A[2, 0] += Dp[0]
                        A[2, 1] += Dp[1]
                        A[2, 2] += p.dot(Dp) + cn * ll * ll / 12.0  # own spin: moves across it
                        rhs[0] += g[0]
                        rhs[1] += g[1]
                        rhs[2] += p.dot(g)
                sol = A.inverse() @ rhs
                V = ti.math.vec2(sol[0], sol[1])
                W = sol[2]
                power = 0.0
                body = 0.0
                for k in ti.static(range(MAX_SEG)):
                    if k < self.n[o]:
                        t = ti.math.vec2(tx[k], ty[k])
                        nn = ti.math.vec2(-t.y, t.x)
                        v = V + W * ti.math.vec2(-ry[k], rx[k]) + ti.math.vec2(rx[k] - px[k], ry[k] - py[k])
                        power += ct * v.dot(t) ** 2 + cn * v.dot(nn) ** 2
                        body += ti.cast(self.a.body[self.slot[o, k]], ti.f32)
                self.a.energy[h] -= self.cfg.k_work * power * body / self.n[o]
                # to the world: rotate by ψ, move the head, drift with the current
                c, s = ti.cos(self.psi[o]), ti.sin(self.psi[o])
                head = self.head[o]
                new_head = head + ti.math.vec2(c * V.x - s * V.y, s * V.x + c * V.y) + self.a.flow(
                    ti.cast(head, ti.i32))
                new_psi = self.psi[o] + W
                c, s = ti.cos(new_psi), ti.sin(new_psi)
                ok = True
                for k in ti.static(range(MAX_SEG)):
                    if k < self.n[o]:
                        q = new_head + ti.math.vec2(c * rx[k] - s * ry[k], s * rx[k] + c * ry[k])
                        if q.x < 0 or q.y < 0 or not self.a.world.is_water(ti.cast(q.x, ti.i32),
                                                                          ti.cast(q.y, ti.i32)):
                            ok = False
                if not ok:  # blocked by rock or the edge: stay, and turn away
                    new_head, new_psi = head, self.psi[o] + 0.5 * self.turn_sign[o] + 0.1
                    c, s = ti.cos(new_psi), ti.sin(new_psi)
                self.travel[o] += (new_head - head).norm()
                self.head[o] = new_head
                self.psi[o] = new_psi
                for k in ti.static(range(MAX_SEG)):
                    if k < self.n[o]:
                        q = new_head + ti.math.vec2(c * rx[k] - s * ry[k], s * rx[k] + c * ry[k])
                        qc = ti.cast(q, ti.i32)
                        if q.x >= 0 and q.y >= 0 and self.a.world.is_water(qc.x, qc.y):
                            self.a.pos[self.slot[o, k]] = q

    @ti.func
    def pop_slot(self, i: int) -> int:
        """A free slot for a division of slot i, by the same cell-weighted lottery as reproduce."""
        p = (self.a.free_at_start[None] * ti.cast(self.a.ncell[i], ti.f64)
             / ti.max(self.a.div_cells[None], 1e-12))
        if self.a.free_at_start[None] >= self.a.div_slots[None]:
            p = 1.0
        j = -1
        if p >= 1.0 or ti.random(ti.f64) < p:
            old = ti.atomic_sub(self.a.top[None], 1)
            if old <= 0:
                ti.atomic_add(self.a.top[None], 1)
            else:
                j = self.a.free[old - 1]
        return j

    @ti.func
    def start_slot(self, i: int, j: int):
        """Fields a new slot j takes from its parent i (body and genome set by the caller)."""
        self.a.age[j] = 0
        self.a.act[j] = ti.Vector.zero(ti.f32, N_OUT)
        self.a.adapt[j] = self.a.adapt[i]
        self.a.heading[j] = ti.random() * 6.2831853
        self.a.brain_cost[j] = self.a.brain_cost[i]
        self.a.species[j] = self.a.species[i]
        self.a.want_div[j] = 0
        self.a.uid[j] = ti.atomic_add(self.a.next_uid[None], 1)
        self.a.parent[j] = self.a.uid[i]
        self.a.pos[j] = self.a.pos[i]
        self.a.gen[j] = self.a.gen[i]

    @ti.kernel
    def develop(self):
        """Divisions of segment slots, before reproduce (which skips them). Pops slots only."""
        for o in range(self.cap):
            if self.state[o] == O_ALIVE:
                n = self.n[o]
                for k in range(n):
                    i = self.slot[o, k]
                    if (self.a.state[i] == ALIVE and self.a.want_div[i] == 1
                            and self.a.uid[i] == self.suid[o, k]):
                        self.a.want_div[i] = 0
                        full = self.a.ncell[i] >= self.a.genome[i, SEG_CELLS]
                        if self.shares(i, k)[T_GERM] > GERM_ON and self.a.ncell[i] >= 2.0:
                            j = self.pop_slot(i)
                            if j >= 0:
                                # one cell leaves as the start of the next generation
                                cell = self.a.body[i] / ti.cast(self.a.ncell[i], CHEM_DTYPE)
                                e_cell = self.a.energy[i] / self.a.ncell[i]
                                self.a.body[j] = cell
                                self.a.body[i] -= cell
                                self.a.energy[j] = e_cell
                                self.a.energy[i] -= e_cell
                                self.a.ncell[j] = 1.0
                                self.a.ncell[i] = ti.max(self.a.ncell[i] - 1.0, 1.0)
                                self.start_slot(i, j)
                                self.a.org[j] = -1
                                self.a.seg[j] = 0
                                self.a.metab_frac[j] = 1.0
                                self.a.gen[j] = self.a.gen[i] + 1  # the next generation
                                self.a.mutate(i, j)
                                self.a.state[j] = ALIVE
                        elif not full:
                            self.a.ncell[i] *= 2.0  # the segment's cells divide in place
                        elif k == n - 1 and n < self.target(i):
                            j = self.pop_slot(i)
                            if j >= 0:
                                # the tail buds a new segment: half its cells, the same genome
                                half = 0.5 * self.a.body[i]
                                self.a.body[j] = half
                                self.a.body[i] -= half
                                e_half = 0.5 * self.a.energy[i]
                                self.a.energy[j] = e_half
                                self.a.energy[i] -= e_half
                                n_half = ti.max(0.5 * self.a.ncell[i], 1.0)
                                self.a.ncell[i] = n_half
                                self.a.ncell[j] = n_half
                                self.start_slot(i, j)
                                for g in range(N_GENES):
                                    self.a.genome[j, g] = self.a.genome[i, g]
                                self.a.org[j] = o
                                self.a.seg[j] = n
                                self.a.metab_frac[j] = self.shares(i, n)[T_METAB]
                                self.slot[o, n] = j
                                self.suid[o, n] = self.a.uid[j]
                                self.n[o] = n + 1
                                self.a.state[j] = ALIVE

    @ti.kernel
    def integrity(self):
        """After die/reproduce: cut each body at its first dead (or reused) segment. Pushes only."""
        for o in range(self.cap):
            if self.state[o] == O_ALIVE:
                n = self.n[o]
                cut = n
                for k in range(n):
                    i = self.slot[o, k]
                    if not (self.a.state[i] == ALIVE and self.a.uid[i] == self.suid[o, k]):
                        cut = ti.min(cut, k)
                if cut < n:
                    for k in range(cut, n):  # the rest falls off as plain colonies
                        i = self.slot[o, k]
                        if self.a.state[i] == ALIVE and self.a.uid[i] == self.suid[o, k]:
                            self.a.org[i] = -1
                            self.a.seg[i] = 0
                            self.a.metab_frac[i] = 1.0
                    self.n[o] = cut
                    if cut == 0:
                        self.state[o] = O_DEAD
                        self.free[ti.atomic_add(self.top[None], 1)] = o

    @ti.kernel
    def transition(self):
        """A differentiating colony big enough becomes an organism with itself as the head."""
        for i in range(self.a.cap):
            if (self.a.state[i] == ALIVE and self.a.org[i] < 0
                    and self.a.genome[i, DIFF] >= DIFF_ON
                    and self.a.genome[i, ADHESION] >= COLONY_ADHESION
                    and self.a.ncell[i] >= self.cfg.org_min_cells):
                old = ti.atomic_sub(self.top[None], 1)
                if old <= 0:
                    ti.atomic_add(self.top[None], 1)
                else:
                    o = self.free[old - 1]
                    self.state[o] = O_ALIVE
                    self.n[o] = 1
                    self.slot[o, 0] = i
                    self.suid[o, 0] = self.a.uid[i]
                    self.head[o] = self.a.pos[i]
                    self.psi[o] = ti.random() * 6.2831853
                    self.phase[o] = 0.0
                    self.turn[o] = 0
                    self.turn_sign[o] = 1.0
                    self.travel[o] = 0.0
                    self.uid[o] = ti.atomic_add(self.next_uid[None], 1)
                    self.a.org[i] = o
                    self.a.seg[i] = 0

    def table(self) -> dict:
        """Living organisms as numpy arrays (stats, viewer, tests)."""
        alive = self.state.to_numpy() == O_ALIVE
        heads = self.slot.to_numpy()[alive, 0]
        return {"id": np.nonzero(alive)[0], "n": self.n.to_numpy()[alive],
                "gen": self.a.gen.to_numpy()[heads],
                "slot": self.slot.to_numpy()[alive], "head": self.head.to_numpy()[alive],
                "uid": self.uid.to_numpy()[alive], "travel": self.travel.to_numpy()[alive]}
