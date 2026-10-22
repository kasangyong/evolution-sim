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
- wire / neural (stage 7c): the nervous system grows from neuron-type genes (developmental
  encoding) whenever the body changes, and runs as a continuous-time recurrent network of graded
  neurons; motor neurons bend their segment's joint, sensory ones read the segment's senses

Free-lists: develop and transition pop, integrity pushes (never both in one kernel, §4.6). The
neuron pool (stage 7c, shared by all organisms): wire pops, integrity pushes."""
import math

import numpy as np
import taichi as ti

from evo.config import H, O, R
from evo.life.agents import ACT_ON, ACT_TUMBLE, ADAPT_UNSET, ALIVE, COLONY_ADHESION, N_OUT
from evo.life.genome import (
    ADHESION,
    CONN,
    CPG_AMP,
    CPG_FREQ,
    CPG_LAG,
    DIFF,
    N_GENES,
    N_SEG,
    N_TYPES,
    NEURO,
    NT_BIAS,
    NT_CHANNEL,
    NT_COUNT,
    NT_DENSITY,
    NT_MGAIN,
    NT_SEGMENTAL,
    NT_SGAIN,
    NT_TAU,
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
MAX_NEU, MAX_SYN = 256, 24  # per organism: neurons, synapses received per neuron
N_CHANNELS = 7  # senses: 4 chemical/light changes, hunger, touch, bend ahead (proprioception)
NEURAL_SUBSTEPS = 2


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
        self.travel = ti.field(ti.f32, shape=cap)  # distance swum by itself (cells, no current)
        self.free = ti.field(ti.i32, shape=cap)
        self.top = ti.field(ti.i32, shape=())
        self.next_uid = ti.field(ti.i64, shape=())
        self.free.from_numpy(np.arange(cap - 1, -1, -1, dtype=np.int32))
        self.top[None] = cap
        self.next_uid[None] = 1
        # stage 7c: nervous system, rebuilt when `dirty`. Neurons live in one pool shared by all
        # organisms (an organism holds the pool ids of its neurons, local index -> id), so memory
        # follows the neurons that exist, not 256 per organism (stage 7c evaluation)
        self.dirty = ti.field(ti.i32, shape=cap)
        self.nn = ti.field(ti.i32, shape=cap)  # neurons
        self.nidx = ti.field(ti.i32, shape=(cap, MAX_NEU))  # local neuron -> pool id
        self.wsum = ti.field(ti.f32, shape=cap)  # sum of |w|, for the upkeep
        ncap = self.cfg.neuron_capacity
        self.ncap = ncap
        self.ntype = ti.field(ti.i32, shape=ncap)
        self.nseg = ti.field(ti.i32, shape=ncap)
        self.v = ti.field(ti.f32, shape=ncap)  # membrane potential
        self.rate = ti.field(ti.f32, shape=ncap)  # graded output tanh(v)
        self.nsyn = ti.field(ti.i32, shape=ncap)
        self.pre = ti.field(ti.i32, shape=(ncap, MAX_SYN))  # local index of the presynaptic neuron
        self.w = ti.field(ti.f32, shape=(ncap, MAX_SYN))
        self.nfree = ti.field(ti.i32, shape=ncap)
        self.ntop = ti.field(ti.i32, shape=())
        self.nfree.from_numpy(np.arange(ncap - 1, -1, -1, dtype=np.int32))
        self.ntop[None] = ncap
        # saturation, so a full table is visible rather than silent (stage 7c evaluation)
        self.failed_transitions = ti.field(ti.i64, shape=())  # colonies refused: table full
        self.failed_neurons = ti.field(ti.i64, shape=())  # neurons not grown: pool full
        self.bend = ti.field(ti.f32, shape=(cap, MAX_SEG))  # neural bend of each joint, now
        self.bend_prev = ti.field(ti.f32, shape=(cap, MAX_SEG))
        self.joint = ti.field(ti.f32, shape=(cap, MAX_SEG))  # joint angles (proprioception)

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
    def shape(self, o: int, phase: ti.f32, bias: ti.f32, bend: ti.template(), record: ti.template()):
        """Body frame of a bead-and-link chain: bead 0 (the head) at the origin, link 1 pointing
        back along −x, so the head faces +x. Joint j is the bend at bead j between links j and
        j + 1 (an n-bead body has n − 2 joints: two beads are rigid, three are a one-hinge
        scallop): muscle share of bead j × (A·sin(2π·phase − lag·j) + bias + neural bend).
        Returns bead positions and unit tangents pointing headward (the mean of the two links)."""
        h = self.slot[o, 0]
        amp, lag = self.a.genome[h, CPG_AMP], self.a.genome[h, CPG_LAG]
        n = self.n[o]
        rx = ti.Vector.zero(ti.f32, MAX_SEG)
        ry = ti.Vector.zero(ti.f32, MAX_SEG)
        dx = ti.Vector.zero(ti.f32, MAX_SEG)  # link k direction (bead k − 1 → bead k)
        dy = ti.Vector.zero(ti.f32, MAX_SEG)
        a = math.pi
        for k in ti.static(range(1, MAX_SEG)):
            if k < n:
                if k >= 2:
                    j = k - 1
                    mus = self.shares(self.slot[o, j], j)[T_MUSCLE]
                    th = mus * (amp * ti.sin(TWO_PI * phase - lag * j) + bias + bend[o, j])
                    if ti.static(record):
                        self.joint[o, j] = th
                    a += th
                dx[k] = ti.cos(a)
                dy[k] = ti.sin(a)
                rx[k] = rx[k - 1] + self.cfg.seg_len * dx[k]
                ry[k] = ry[k - 1] + self.cfg.seg_len * dy[k]
        tx = ti.Vector.zero(ti.f32, MAX_SEG)
        ty = ti.Vector.zero(ti.f32, MAX_SEG)
        tx[0] = 1.0
        for k in ti.static(range(1, MAX_SEG)):
            if k < n:
                ex, ey = -dx[k], -dy[k]
                if k + 1 < n:
                    ex -= dx[ti.min(k + 1, MAX_SEG - 1)]
                    ey -= dy[ti.min(k + 1, MAX_SEG - 1)]
                norm = ti.max(ti.sqrt(ex * ex + ey * ey), 1e-6)
                tx[k] = ex / norm
                ty[k] = ey / norm
        if n > 1:
            tx[0], ty[0] = -dx[1], -dy[1]
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
                # midpoint rule (time-symmetric): velocities from the shape halfway through the
                # tick, so a reciprocal stroke gives no net motion (scallop theorem)
                rx, ry, tx, ty = self.shape(o, ph, bias, self.bend, True)
                px, py, qx, qy = self.shape(o, self.phase[o], bias, self.bend_prev, False)
                for k in ti.static(range(MAX_SEG)):
                    self.bend_prev[o, k] = self.bend[o, k]
                self.phase[o] = ph - ti.floor(ph)
                A = ti.Matrix.zero(ti.f32, 3, 3)
                rhs = ti.Vector.zero(ti.f32, 3)
                ct, cn, ll = self.cfg.drag_t, self.cfg.drag_n, self.cfg.seg_len
                for k in ti.static(range(MAX_SEG)):
                    if k < self.n[o]:
                        t = ti.math.vec2(tx[k] + qx[k], ty[k] + qy[k]).normalized()
                        nn = ti.math.vec2(-t.y, t.x)
                        D = ct * t.outer_product(t) + cn * nn.outer_product(nn)
                        p = ti.math.vec2(-(ry[k] + py[k]), rx[k] + px[k]) * 0.5
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
                        t = ti.math.vec2(tx[k] + qx[k], ty[k] + qy[k]).normalized()
                        nn = ti.math.vec2(-t.y, t.x)
                        p = ti.math.vec2(-(ry[k] + py[k]), rx[k] + px[k]) * 0.5
                        v = V + W * p + ti.math.vec2(rx[k] - px[k], ry[k] - py[k])
                        power += ct * v.dot(t) ** 2 + cn * v.dot(nn) ** 2
                        body += ti.cast(self.a.body[self.slot[o, k]], ti.f32)
                self.a.energy[h] -= self.cfg.k_work * power * body / self.n[o]
                # to the world: the frame moves at V turned by the midpoint orientation
                c, s = ti.cos(self.psi[o] + 0.5 * W), ti.sin(self.psi[o] + 0.5 * W)
                head = self.head[o]
                own = ti.math.vec2(c * V.x - s * V.y, s * V.x + c * V.y)  # swum, not drifted
                new_head = head + own + self.a.flow(ti.cast(head, ti.i32))
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
                if ok:
                    self.travel[o] += own.norm()  # self-propelled distance only (no current)
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
                                self.dirty[o] = 1  # a new segment: the nervous system regrows
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
                if cut < n or self.dirty[o] == 1:  # a changed body regrows its nerves: release
                    for i in range(self.nn[o]):
                        self.nfree[ti.atomic_add(self.ntop[None], 1)] = self.nidx[o, i]
                    self.nn[o] = 0
                    self.dirty[o] = 1
                if cut < n:
                    for k in range(cut, n):  # the rest falls off as plain colonies
                        i = self.slot[o, k]
                        if self.a.state[i] == ALIVE and self.a.uid[i] == self.suid[o, k]:
                            self.a.org[i] = -1
                            self.a.seg[i] = 0
                            self.a.metab_frac[i] = 1.0
                    self.n[o] = cut
                    self.dirty[o] = 1
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
                    ti.atomic_add(self.failed_transitions[None], 1)  # the table is full
                else:
                    o = self.free[old - 1]
                    self.state[o] = O_ALIVE
                    self.nn[o] = 0
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
                    self.dirty[o] = 1
                    for k in ti.static(range(MAX_SEG)):
                        self.bend[o, k] = 0.0
                        self.bend_prev[o, k] = 0.0
                        self.joint[o, k] = 0.0
                    self.a.org[i] = o
                    self.a.seg[i] = 0

    @ti.func
    def gene_t(self, h: int, k: int, p: int) -> ti.f32:
        return self.a.genome[h, NEURO + 8 * k + p]

    @ti.kernel
    def wire(self):
        """Development of the nervous system (developmental encoding): neurons of each type in the
        head, or in every segment, taken from the shared pool (pops only), then each neuron draws
        `density` candidate inputs and keeps one from type l with probability |W[k, l]| / 4 at
        strength W[k, l] × (1 + 0.1·noise). Segmental types connect only within a segment or to
        the next. Same genome, slightly different circuit: individual variation."""
        for o in range(self.cap):
            if self.state[o] == O_ALIVE and self.dirty[o] == 1:
                self.dirty[o] = 0
                h = self.slot[o, 0]
                n = 0
                for k in range(N_TYPES):
                    cnt = ti.cast(ti.floor(self.gene_t(h, k, NT_COUNT)), ti.i32)
                    segs = 1
                    if self.gene_t(h, k, NT_SEGMENTAL) >= 0.5:
                        segs = self.n[o]
                    for s in range(segs):
                        for _ in range(cnt):
                            if n < MAX_NEU:
                                old = ti.atomic_sub(self.ntop[None], 1)
                                if old <= 0:
                                    ti.atomic_add(self.ntop[None], 1)
                                    ti.atomic_add(self.failed_neurons[None], 1)  # pool full
                                else:
                                    g = self.nfree[old - 1]
                                    self.nidx[o, n] = g
                                    self.ntype[g] = k
                                    self.nseg[g] = s
                                    self.v[g] = 0.0
                                    self.rate[g] = 0.0
                                    n += 1
                self.nn[o] = n
                wsum = 0.0
                for i in range(n):
                    g = self.nidx[o, i]
                    ki = self.ntype[g]
                    m = 0
                    dens = ti.min(ti.cast(ti.round(self.gene_t(h, ki, NT_DENSITY)), ti.i32), MAX_SYN)
                    for _ in range(dens):
                        j = ti.min(ti.cast(ti.random() * n, ti.i32), n - 1)
                        gj = self.nidx[o, j]
                        kj = self.ntype[gj]
                        wk = self.a.genome[h, CONN + N_TYPES * ki + kj]
                        near = True
                        if (self.gene_t(h, ki, NT_SEGMENTAL) >= 0.5
                                and self.gene_t(h, kj, NT_SEGMENTAL) >= 0.5):
                            near = ti.abs(self.nseg[g] - self.nseg[gj]) <= 1
                        if near and ti.random() < ti.abs(wk) / 4.0:
                            self.pre[g, m] = j
                            self.w[g, m] = wk * (1.0 + 0.1 * ti.randn())
                            wsum += ti.abs(self.w[g, m])
                            m += 1
                    self.nsyn[g] = m
                self.wsum[o] = wsum

    @ti.func
    def sense(self, o: int, s: int, c: int) -> ti.f32:
        """Sense channel c at segment s of organism o."""
        i = self.slot[o, ti.min(s, self.n[o] - 1)]
        p = ti.cast(self.a.pos[i], ti.i32)
        x, y = p.x, p.y
        val = 0.0
        if c < 4:  # change against the adapted level (stage 7a receptors), clipped
            now = 0.0
            if c == 0:
                now = ti.log(ti.max(ti.cast(self.a.world.chem[H, x, y], ti.f32), 1e-6))
            elif c == 1:
                now = ti.log(ti.max(ti.cast(self.a.world.chem[R, x, y], ti.f32), 1e-6))
            elif c == 2:
                now = ti.log(ti.max(ti.cast(self.a.world.chem[O, x, y], ti.f32), 1e-6))
            else:
                now = ti.log(ti.max(self.a.world.light[x, y], 1e-6))
            if self.a.adapt[i][0] != ADAPT_UNSET:
                val = ti.min(ti.max(now - self.a.adapt[i][c], -5.0), 5.0)
        elif c == 4:  # hunger
            b = ti.max(ti.cast(self.a.body[i], ti.f32), 1e-12)
            val = 1.0 - self.a.energy[i] / (self.cfg.energy_cap * b)
        elif c == 5:  # touch: other cells here
            val = ti.log(1.0 + ti.max(self.a.cell_cells[x, y] - self.a.ncell[i], 0.0))
        else:  # proprioception: the bend just in front (joint s − 1; the head reads joint 1), as
            # C. elegans motor neurons sense the curvature of the body ahead and pass the wave on
            val = self.joint[o, ti.max(s - 1, 1)]  # joints are 1 .. n − 2
        return val

    @ti.kernel
    def neural(self):
        """CTRNN, tau dv/dt = -v + sum w·tanh(v_pre) + bias + gain·sense, Euler substeps; motor
        neurons set the neural bend of their segment's joint. Upkeep: per neuron and per sum |w|."""
        for o in range(self.cap):
            if self.state[o] == O_ALIVE and self.nn[o] > 0:
                h = self.slot[o, 0]
                n = self.nn[o]
                dt = 1.0 / NEURAL_SUBSTEPS
                for _ in range(NEURAL_SUBSTEPS):
                    for i in range(n):
                        g = self.nidx[o, i]
                        k = self.ntype[g]
                        c = ti.min(ti.cast(ti.floor(self.gene_t(h, k, NT_CHANNEL)), ti.i32),
                                   N_CHANNELS - 1)
                        drive = self.gene_t(h, k, NT_BIAS) + self.gene_t(h, k, NT_SGAIN) * self.sense(
                            o, self.nseg[g], c)
                        for m in range(self.nsyn[g]):
                            drive += self.w[g, m] * self.rate[self.nidx[o, self.pre[g, m]]]
                        tau = self.gene_t(h, k, NT_TAU)
                        self.v[g] += dt / tau * (drive - self.v[g])
                    for i in range(n):
                        g = self.nidx[o, i]
                        self.rate[g] = ti.tanh(self.v[g])
                for k in ti.static(range(MAX_SEG)):
                    self.bend[o, k] = 0.0
                for i in range(n):
                    g = self.nidx[o, i]
                    # a segment's neurons bend the joint at that bead; head neurons the first one
                    s = ti.max(self.nseg[g], 1)
                    if s < self.n[o] - 1:
                        self.bend[o, s] += self.gene_t(h, self.ntype[g], NT_MGAIN) * self.rate[g]
                self.a.energy[h] -= self.cfg.k_neuron * n + self.cfg.k_syn * self.wsum[o]

    def table(self) -> dict:
        """Living organisms as numpy arrays (stats, viewer, tests)."""
        alive = self.state.to_numpy() == O_ALIVE
        heads = self.slot.to_numpy()[alive, 0]
        return {"id": np.nonzero(alive)[0], "n": self.n.to_numpy()[alive],
                "gen": self.a.gen.to_numpy()[heads],
                "slot": self.slot.to_numpy()[alive], "head": self.head.to_numpy()[alive],
                "uid": self.uid.to_numpy()[alive], "travel": self.travel.to_numpy()[alive],
                "neurons": self.nn.to_numpy()[alive]}

    def brain(self, o: int) -> dict:
        """One organism's nervous system for the viewer: neuron type, segment, rate; synapses."""
        n = int(self.nn[o])
        g = self.nidx.to_numpy()[o, :n]
        pre, w, ns = self.pre.to_numpy()[g], self.w.to_numpy()[g], self.nsyn.to_numpy()[g]
        syn = [[int(pre[i, m]), i, round(float(w[i, m]), 3)] for i in range(n) for m in range(ns[i])]
        return {"org": o, "segments": int(self.n[o]), "type": self.ntype.to_numpy()[g].tolist(),
                "seg": self.nseg.to_numpy()[g].tolist(),
                "rate": [round(float(r), 3) for r in self.rate.to_numpy()[g]], "syn": syn}

    def neurons_of(self, o: int) -> np.ndarray:
        """Pool ids of organism o's neurons (tests, viewer)."""
        return self.nidx.to_numpy()[o, :int(self.nn[o])]
