from evo.config import Config
from evo.life.agents import Agents
from evo.world.chemistry import Chemistry
from evo.world.climate import Climate
from evo.world.fields import World
from evo.world.ledger import Totals, budget_error, totals


class Sim:
    """One tick = vents → transport → chemistry → gas exchange → climate → light/temp → life.
    Caller must ti.init() before constructing."""

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.world = World(self.cfg.world)
        self.chemistry = Chemistry(self.world, self.cfg.chem)
        self.climate = Climate(self.world, self.cfg.climate)
        self.life = Agents(self.world, self.cfg.life, seed=self.cfg.world.seed)
        self.tick = 0
        self.t0: Totals = totals(self.world, self.life)

    def step(self, n: int = 1):
        w = self.world
        for _ in range(n):
            w.emit_vents()
            w.transport()
            self.chemistry.step()
            self.climate.gas_exchange()
            self.climate.step()
            w.update_light_temp()
            self.life.step()
        self.tick += n

    def budget_error(self):
        return budget_error(self.world, self.life, self.t0)
