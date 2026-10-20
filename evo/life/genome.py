"""Genome layout (§3, stage-7a; new genes are appended so indices stay stable). Genes
are stored directly in trait units and clamped to their range after mutation; mutation noise
is scaled by each trait's range."""

DIV_SIZE, ETA_H2, ETA_METH, ETA_ANAER, T_OPT, T_WIDTH, MUT_RATE, COL_R, COL_G, COL_B = range(10)
ETA_PHOTO_AN, ETA_PHOTO_OX, ETA_AER, ETA_SFIX, ETA_SOX, ETA_MOX, O2_TOL = range(10, 17)
ETA_PRED, SPEED, PREY_RATIO = range(17, 20)
# brain N_IN -> 8 -> 4 (tanh): W1[h, k] at BRAIN + N_IN·h + k, b1[h] at BRAIN + N_IN·N_HID + h,
# W2[o, h] after that. Stage 7a: 8 state inputs + 4 temporal-comparison senses (chemotaxis)
BRAIN, N_IN, N_HID, N_OUT = 20, 12, 8, 4
N_BRAIN = N_HID * N_IN + N_HID + N_OUT * N_HID
# stage 6: adhesion (colonies) appended after the brain, so earlier indices stay
ADHESION = BRAIN + N_BRAIN
# stage 7a: how many ticks the senses average over before comparing (adaptation time)
ADAPT_TAU = ADHESION + 1
N_GENES = ADAPT_TAU + 1
# genes that describe the organism (not the brain): species distance, big mutations
TRAIT_GENES = tuple(range(BRAIN)) + (ADHESION, ADAPT_TAU)
ETA_GENES = (ETA_H2, ETA_METH, ETA_ANAER, ETA_PHOTO_AN, ETA_PHOTO_OX, ETA_AER, ETA_SFIX, ETA_SOX,
             ETA_MOX)

# (low, high) per gene
GENE_RANGE = [
    (0.005, 0.1),  # body carbon at which the cell divides
    (0.0, 0.95),  # η, H2 chemosynthesis (η < 1: no reaction cycle can break even, §2.3)
    (0.0, 0.95),  # η, methanogenesis
    (0.0, 0.95),  # η, anaerobic respiration
    (0.0, 90.0),  # optimal temperature (°C)
    (2.0, 40.0),  # temperature tolerance width
    (0.001, 0.2),  # mutation σ as a fraction of each gene's range
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 0.95),  # η, anoxygenic photosynthesis (2R + C -> body + 2X, light pays)
    (0.0, 0.95),  # η, oxygenic photosynthesis (C -> body + O, light pays)
    (0.0, 0.95),  # η, aerobic respiration
    (0.0, 0.95),  # η, sulfur carbon fixation (2R + C -> body + 2X, energy pays)
    (0.0, 0.95),  # η, sulfur oxidation
    (0.0, 0.95),  # η, methane oxidation
    (0.0, 1.0),  # O2 tolerance
    (0.0, 0.95),  # predation efficiency (bite rate per unit body)
    (0.0, 1.0),  # top active speed (cells per tick)
    (1.0, 10.0),  # prey must be this many times smaller (per-cell size)
] + [(-4.0, 4.0)] * N_BRAIN + [
    (0.0, 1.0),  # adhesion: chance that a daughter stays bound in the colony
    (1.0, 20.0),  # adaptation time of the senses (ticks)
]
GENE_NAMES = ["div_size", "eta_h2", "eta_meth", "eta_anaer", "t_opt", "t_width", "mut_rate",
              "col_r", "col_g", "col_b", "eta_photo_an", "eta_photo_ox", "eta_aer", "eta_sfix",
              "eta_sox", "eta_mox", "o2_tol", "eta_pred", "speed", "prey_ratio"] + [
    f"brain{k}" for k in range(N_BRAIN)] + ["adhesion", "adapt_tau"]

# B-plan founder (§3): H2 chemosynthesis + methanogenesis only
# zero brain: no movement, no predation, no division inhibition (stage-3a behaviour)
FOUNDER = [0.02, 0.5, 0.3, 0.0, 30.0, 20.0, 0.02, 0.85, 0.35, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 2.0] + [0.0] * N_BRAIN + [0.0, 5.0]

assert len(GENE_RANGE) == len(GENE_NAMES) == len(FOUNDER) == N_GENES
