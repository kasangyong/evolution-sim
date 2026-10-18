import math

import taichi as ti

from evo.config import ATM_C, ATM_H, ATM_M, ATM_O, ATM_SPECIES, ClimateConfig
from evo.world.fields import (
    BUR_CARB,
    CHEM_DTYPE,
    CL_ICE,
    CL_SOLAR,
    CL_TS,
    SINK_C,
    SINK_E,
    SRC_C,
    World,
)


@ti.data_oriented
class Climate:
    """Global one-box atmosphere: gas exchange, greenhouse temperature, ice-albedo,
    weathering, methane photolysis and hydrogen escape (§2.5)."""

    def __init__(self, world: World, cfg: ClimateConfig):
        self.world = world
        self.cfg = cfg
        for g, amount in enumerate(cfg.init_atm):
            world.atm[g] = amount
        world.clim[CL_SOLAR] = cfg.solar
        world.clim[CL_ICE] = 0.0
        world.clim[CL_TS] = self.equilibrium_temp(0.0)

    def equilibrium_temp(self, ice: float) -> float:
        cf = self.cfg
        atm = self.world.atm
        pc = max(atm[ATM_C] / cf.c_ref, cf.p_min)
        return (
            cf.t0
            + cf.a_co2 * math.log(pc)
            + cf.a_ch4 * math.sqrt(max(atm[ATM_M], 0.0) / cf.m_ref)
            + cf.b_solar * (cf.solar - 1.0)
            - cf.c_ice * ice
        )

    @ti.kernel
    def gas_exchange(self):
        for g in range(4):
            self.world.atm_delta[g] = 0.0
        open_frac = ti.cast(1.0 - self.world.clim[CL_ICE], CHEM_DTYPE)
        for x in range(self.world.w):
            for gi in ti.static(range(4)):
                s = ATM_SPECIES[gi]
                c = self.world.chem[s, x, 0]
                eq = ti.cast(self.cfg.solubility[gi] * self.world.atm[gi], CHEM_DTYPE)
                f = self.cfg.k_gas * open_frac * (eq - c)
                self.world.chem[s, x, 0] = c + f
                self.world.atm_delta[gi] -= ti.cast(f, ti.f64)
        for g in range(4):
            self.world.atm[g] += self.world.atm_delta[g]

    @ti.kernel
    def step(self):
        ts, ice = self.world.clim[CL_TS], self.world.clim[CL_ICE]
        ac, am = self.world.atm[ATM_C], self.world.atm[ATM_M]
        pc = ti.max(ac / self.cfg.c_ref, self.cfg.p_min)
        target = (
            self.cfg.t0
            + self.cfg.a_co2 * ti.log(pc)
            + self.cfg.a_ch4 * ti.sqrt(ti.max(am, 0.0) / self.cfg.m_ref)
            + self.cfg.b_solar * (self.world.clim[CL_SOLAR] - 1.0)
            - self.cfg.c_ice * ice
        )
        ts += (target - ts) / self.cfg.tau
        ice = ti.min(ti.max((self.cfg.t_freeze - ts) / self.cfg.ice_width, 0.0), 1.0)

        if ti.static(self.cfg.weathering_on):
            wr = self.cfg.k_weather * ac * ti.exp(self.cfg.k_weather_t * (ts - self.cfg.t0)) * (1.0 - ice)
            self.world.atm[ATM_C] -= wr
            self.world.acc[SINK_C] += wr
            self.world.acc[BUR_CARB] += wr

        ph = self.cfg.k_photo_m * self.world.atm[ATM_M]
        self.world.atm[ATM_M] -= ph
        self.world.atm[ATM_C] += ph
        self.world.acc[SINK_E] += 8.0 * ph

        mo = ti.min(self.cfg.k_atm_mo * self.world.atm[ATM_M] * self.world.atm[ATM_O],
                    ti.min(self.world.atm[ATM_M], 0.5 * self.world.atm[ATM_O]))
        self.world.atm[ATM_M] -= mo
        self.world.atm[ATM_O] -= 2.0 * mo
        self.world.atm[ATM_C] += mo

        he = self.cfg.k_h_escape * self.world.atm[ATM_H]
        self.world.atm[ATM_H] -= he
        self.world.acc[SINK_E] += 2.0 * he

        self.world.clim[CL_TS] = ts
        self.world.clim[CL_ICE] = ice

    def force_co2(self, amount: float):
        """Test-only external CO2 forcing, booked as a source so the ledger stays exact."""
        self.world.atm[ATM_C] += amount
        self.world.acc[SRC_C] += amount
