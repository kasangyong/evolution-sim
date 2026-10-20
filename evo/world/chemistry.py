import taichi as ti

from evo.config import CN_RATIO, C, ChemConfig, D, H, M, N, O, R, X
from evo.world.fields import BUR_ORG, CHEM_DTYPE, SINK_C, SINK_E, SINK_N, World


@ti.data_oriented
class Chemistry:
    """Abiotic reactions, decomposition and burial (§2.3, stage 1 subset)."""

    def __init__(self, world: World, cfg: ChemConfig):
        self.world = world
        self.cfg = cfg
        # per-column partial sums: atomics on a single address serialize badly on GPU
        self.col_sink = ti.field(CHEM_DTYPE, shape=(3, world.w))

    @ti.kernel
    def step(self):
        for x, y in ti.ndrange(self.world.w, self.world.d):
            if self.world.is_water(x, y):
                h, r, xx = self.world.chem[H, x, y], self.world.chem[R, x, y], self.world.chem[X, x, y]
                c, o, m = self.world.chem[C, x, y], self.world.chem[O, x, y], self.world.chem[M, x, y]
                n, d = self.world.chem[N, x, y], self.world.chem[D, x, y]

                # abiotic oxidation; each extent is limited by what is present
                e1 = ti.min(self.cfg.k_ox_r * r * o, ti.min(0.5 * r, o))
                r -= 2 * e1
                o -= e1
                xx += 2 * e1
                e2 = ti.min(self.cfg.k_ox_h * h * o, ti.min(0.5 * h, o))
                h -= 2 * e2
                o -= e2
                e3 = ti.min(self.cfg.k_ox_m * m * o, ti.min(m, 0.5 * o))
                m -= e3
                o -= 2 * e3
                c += e3

                # decomposition, oxidant preference O > X > methanogenic
                fo = o / (o + self.cfg.k_half)
                fx = (1.0 - fo) * xx / (xx + self.cfg.k_half)
                fm = 1.0 - fo - fx
                d1 = ti.min(self.cfg.k_dec_o * fo * d, ti.min(o, d))
                d -= d1
                o -= d1
                c += d1
                d2 = ti.min(self.cfg.k_dec_x * fx * d, ti.min(0.5 * xx, d))
                d -= d2
                xx -= 2 * d2
                c += d2
                r += 2 * d2
                d3 = self.cfg.k_dec_m * fm * d
                d -= d3
                c += 0.5 * d3
                m += 0.5 * d3
                n += (d1 + d2 + d3) / CHEM_DTYPE(CN_RATIO)  # bare literal would be f32

                nl = self.cfg.k_den * (self.cfg.k_half / (self.cfg.k_half + o)) * n
                n -= nl
                self.col_sink[1, x] += nl

                if y == self.world.floor[x] - 1:
                    bd = self.cfg.k_bury_d * d
                    d -= bd
                    br = self.cfg.k_bury_r * r
                    r -= br
                    self.col_sink[0, x] += bd
                    self.col_sink[1, x] += bd / CHEM_DTYPE(CN_RATIO)
                    self.col_sink[2, x] += 4.0 * bd + 2.0 * br

                self.world.chem[H, x, y], self.world.chem[R, x, y], self.world.chem[X, x, y] = h, r, xx
                self.world.chem[C, x, y], self.world.chem[O, x, y], self.world.chem[M, x, y] = c, o, m
                self.world.chem[N, x, y], self.world.chem[D, x, y] = n, d
        for x in range(self.world.w):
            self.world.acc[SINK_C] += ti.cast(self.col_sink[0, x], ti.f64)
            self.world.acc[BUR_ORG] += ti.cast(self.col_sink[0, x], ti.f64)
            self.world.acc[SINK_N] += ti.cast(self.col_sink[1, x], ti.f64)
            self.world.acc[SINK_E] += ti.cast(self.col_sink[2, x], ti.f64)
            self.col_sink[0, x], self.col_sink[1, x], self.col_sink[2, x] = 0.0, 0.0, 0.0
