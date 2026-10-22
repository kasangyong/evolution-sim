from dataclasses import dataclass, field

# Species indices into World.chem (see docs/design/evolution-sim-arch.md §2.1)
H, R, X, C, O, M, N, D = range(8)
NS = 8
SPECIES_NAMES = ["H2", "R", "X", "CO2", "O2", "CH4", "N", "Detritus"]
CN_RATIO = 6.6

# Atmosphere slots
ATM_C, ATM_O, ATM_M, ATM_H = range(4)
ATM_NAMES = ["CO2", "O2", "CH4", "H2"]
# ocean species exchanged with each atmosphere slot
ATM_SPECIES = (C, O, M, H)


@dataclass
class WorldConfig:
    width: int = 1024
    depth: int = 384
    seed: int = 1
    n_vents: int = 4
    # vents on seamount summits inside the photic zone (shallow Archean hydrothermal systems)
    n_shallow_vents: int = 1
    seamount_depth_frac: float = 0.15  # summit floor depth as a fraction of ocean depth
    seamount_width_frac: float = 0.012  # Gaussian half-width as a fraction of world width
    shelf_frac: float = 0.15  # fraction of width that is shallow shelf (left side)
    shelf_depth_frac: float = 0.12
    abyss_depth_frac: float = 0.9

    # transport (per tick, cell volume = 1)
    kh: float = 0.1  # horizontal diffusion
    kv_mix: float = 0.15  # vertical diffusion in mixed layer
    kv_deep: float = 0.02
    mix_layer_frac: float = 0.1
    psi_amp: float = 0.25  # peak advective outflow fraction of any cell per tick
    sink_speed: float = 0.05  # detritus settling fraction per tick

    # light / temperature
    light_atten: float = 0.03  # per cell depth
    temp_deep: float = 2.0
    thermocline: float = 40.0
    vent_heat: float = 60.0
    vent_heat_radius: float = 4.0

    # vent emissions per vent per tick
    vent_h: float = 0.5
    vent_r: float = 0.5
    vent_c: float = 0.2
    vent_n: float = 0.02

    # initial ocean (amount per water cell)
    init: dict = field(
        default_factory=lambda: {H: 0.0, R: 1.0, X: 0.05, C: 2.0, O: 0.0, M: 0.0, N: 0.5, D: 0.0}
    )


@dataclass
class ChemConfig:
    k_ox_r: float = 0.5  # 2R + O -> 2X
    k_ox_h: float = 0.5  # 2H + O -> ()
    k_ox_m: float = 0.5  # M + 2O -> C
    k_dec_o: float = 0.02  # D + O -> C
    k_dec_x: float = 0.005  # D + 2X -> C + 2R
    k_dec_m: float = 0.001  # 2D -> C + M
    k_half: float = 0.05  # half-saturation for oxidant switching
    k_bury_d: float = 0.002  # floor-cell D burial per tick
    k_bury_r: float = 0.002  # floor-cell R burial (sulfide/pyrite)
    k_den: float = 1.0e-6  # anoxic N loss; ocean N equilibrium ≈ vent N / k_den


@dataclass
class ClimateConfig:
    init_atm: tuple = (3.0e5, 0.0, 0.0, 0.0)  # CO2, O2, CH4, H2 (absolute amounts)
    c_ref: float = 1.0e5  # CO2 amount giving partial pressure 1
    m_ref: float = 1.0e4
    solubility: tuple = (2.0e-5, 1.0e-5, 1.0e-5, 1.0e-6)  # c_eq = sol * atm amount
    k_gas: float = 0.05
    t0: float = 15.0
    a_co2: float = 6.0
    p_min: float = 1.0e-3
    a_ch4: float = 4.0
    solar: float = 0.75  # relative to present, at the start
    # the Sun brightens ~7% per billion years; at 1 tick = 1,000 years that is 7e-8 per tick (stage 6)
    solar_rate: float = 7.0e-8
    solar_max: float = 1.0  # present-day Sun
    b_solar: float = 40.0
    c_ice: float = 15.0
    t_freeze: float = 0.0
    ice_width: float = 5.0
    tau: float = 50.0  # relaxation ticks
    k_weather: float = 2.0e-5  # doubled with volcanoes (stage 4a): balances vent + volcanic CO2
    k_weather_t: float = 0.07
    k_photo_m: float = 1.0e-4
    k_atm_mo: float = 1.0e-9
    k_h_escape: float = 1.0e-3
    weathering_on: bool = True
    # Poisson eruptions straight into the atmosphere (stage 4a, design decision 5): the escape route
    # from a snowball, since ice shuts both gas exchange and weathering
    volcano_p: float = 0.01  # eruption chance per tick
    volcano_mean_c: float = 80.0  # mean CO2 per eruption (exponential sizes); mean 0.8/tick = vents
    volcano_h_frac: float = 0.05  # H2 released per unit CO2 (reducing volcanic gas)


@dataclass
class LifeConfig:
    capacity: int = 131072
    n_founders: int = 200
    founder_body: float = 0.01
    k_chemo: float = 0.05  # max H2-fixation extent per unit body per tick at η = 1
    k_resp: float = 0.02  # max respiration extent per unit body per tick at η = 1
    k_half: float = 0.05  # half-saturation of substrate uptake
    maintenance: float = 0.003  # energy per unit body per tick
    trait_upkeep: float = 0.004  # extra energy per unit body per tick per unit η
    energy_cap: float = 1.0  # max stored energy per unit body
    reserve_frac: float = 0.2  # respire body when energy falls below this fraction of cap
    divide_energy_frac: float = 0.1  # below reserve_frac, so any non-starving cell can divide
    death_rate: float = 2.0e-4  # background mortality per tick
    min_body: float = 1.0e-4
    drift_sigma: float = 0.3  # random-walk step (cells)
    big_mutation_p: float = 0.01  # chance that one trait (non-brain) gene is redrawn uniformly
    k_light: float = 0.5  # light absorption per unit biomass in a cell (Beer–Lambert coefficient)
    k_tox: float = 0.05  # O2 damage: extra upkeep per unit body per unit O2 at zero tolerance
    k_pred: float = 0.05  # max bite per unit predator body per tick at η = 1
    assimilation: float = 0.6  # bitten body kept by the predator; the rest becomes detritus
    k_move: float = 0.01  # energy per unit body per (cell/tick)² of active movement
    # stage 7a: nerves and receptors cost energy in proportion to the connection strengths
    k_brain: float = 3.0e-5  # per unit body per unit Σ|weight| (≈1% of maintenance per weight)
    # stage 7b: multicellular organisms (stage7b-arch.md, 7b 구현 결정)
    org_capacity: int = 65536  # organism records; each segment is one slot of the pool above
    neuron_capacity: int = 262144  # neurons shared by all organisms (~55 MB with synapses)
    org_min_cells: float = 16.0  # a differentiating colony this big becomes an organism
    seg_len: float = 0.5  # segment length (grid cells)
    drag_t: float = 1.0  # resistive force per unit velocity along a segment
    drag_n: float = 2.0  # ... and across it (slender body: about twice the tangential)
    k_cilia: float = 0.02  # ciliary thrust per segment at full cilia share
    k_work: float = 0.01  # energy per unit body per unit dissipated swimming power
    turn_ticks: int = 10  # a turn bends the body to one side this long
    turn_bend: float = 0.4  # extra bend per muscle joint during a turn (radians)
    # stage 7c: nervous systems. 100 neurons ≈ 30% of the maintenance of a 0.1-body organism
    k_neuron: float = 1.0e-6  # energy per neuron per tick
    k_syn: float = 1.0e-7  # energy per unit Σ|synapse weight| per tick


@dataclass
class Config:
    world: WorldConfig = field(default_factory=WorldConfig)
    chem: ChemConfig = field(default_factory=ChemConfig)
    climate: ClimateConfig = field(default_factory=ClimateConfig)
    life: LifeConfig = field(default_factory=LifeConfig)
