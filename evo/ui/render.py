import taichi as ti

from evo.life.agents import ALIVE, Agents
from evo.life.genome import COL_B, COL_G, COL_R
from evo.world.fields import World

# layers selectable in the viewer: species 0..7, then light, temperature
LIGHT_LAYER, TEMP_LAYER = 8, 9


@ti.data_oriented
class Renderer:
    """Maps one world layer to an RGBA float image laid out row-major (y, x) for Dear PyGui."""

    def __init__(self, world: World, agents: Agents):
        self.world = world
        self.agents = agents
        self.rgba = ti.field(ti.f32, shape=(world.d, world.w, 4))

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
