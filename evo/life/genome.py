"""Genome layout (§3, stage 7d; new genes are appended so indices stay stable). Genes
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
# stage 7b: multicellular bodies (stage7b-arch.md, 7b 구현 결정). A colony with DIFF ≥ 0.5 becomes an
# organism; it grows N_SEG segments of SEG_CELLS cells. Tissue shares along the body axis s (head 0,
# tail 1) are softmax(metabolism 0, muscle a+b·s, cilia a+b·s, germ a+b·s), like Hox patterning.
# Muscles bend each joint by A·sin(2π·f·t − lag·k) (myogenic rhythm, until nerves in 7c)
DIFF = ADAPT_TAU + 1
N_SEG, SEG_CELLS = DIFF + 1, DIFF + 2
TISSUE = DIFF + 3  # 6 genes: (a, b) for muscle, cilia, germ
CPG_AMP, CPG_FREQ, CPG_LAG = TISSUE + 6, TISSUE + 7, TISSUE + 8
# stage 7c: nervous system by developmental encoding (stage7b-arch.md, "7c 구현 결정"). 8 neuron
# types, each NEURO + 8·k + (count, segmental, tau, bias, sense channel, sense gain, motor gain,
# synapse density); connection strength between types at CONN + 8·k(post) + l(pre)
N_TYPES = 8
NEURO = CPG_LAG + 1
NT_COUNT, NT_SEGMENTAL, NT_TAU, NT_BIAS, NT_CHANNEL, NT_SGAIN, NT_MGAIN, NT_DENSITY = range(8)
CONN = NEURO + 8 * N_TYPES
# stage 7d: plasticity per neuron type at PLAST + 4·k + (modulatory, learning rate of synapses onto
# it, habituation of its output synapses, recovery time) — stage7b-arch.md, "7d 구현 결정"
PLAST = CONN + N_TYPES * N_TYPES
PL_MOD, PL_ETA, PL_HAB, PL_REC = range(4)
N_GENES = PLAST + 4 * N_TYPES
# genes that describe the organism (not the brains): species distance, big mutations
TRAIT_GENES = tuple(range(BRAIN)) + (ADHESION, ADAPT_TAU) + tuple(range(DIFF, NEURO))
# species distance: the stage-7a traits. Body-plan genes act only once a lineage differentiates; in
# single cells they drift freely and would dilute the distance (threshold calibrated on these 22)
SPECIES_GENES = tuple(range(BRAIN)) + (ADHESION, ADAPT_TAU)
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
    (0.0, 1.0),  # differentiation: a colony at ≥ 0.5 develops into a multicellular organism
    (1.0, 12.0),  # segments the body grows to
    (4.0, 256.0),  # cells per segment before the next segment buds off the tail
] + [(-4.0, 4.0)] * 6 + [
    (0.0, 1.2),  # muscle bend amplitude (radians per joint)
    (0.0, 0.25),  # muscle rhythm (cycles per tick)
    (-3.1416, 3.1416),  # phase lag per segment (> 0: bending wave runs head to tail)
] + [
    (0.0, 8.0),  # neurons of this type (per segment if segmental, else in the head)
    (0.0, 1.0),  # ≥ 0.5: repeated in every segment
    (1.0, 20.0),  # membrane time constant (ticks)
    (-4.0, 4.0),  # bias
    (0.0, 6.99),  # sense channel (floor): 0-3 H2/sulfide/O2/light change, 4 hunger, 5 touch, 6 bend
    (-4.0, 4.0),  # sense gain
    (-1.0, 1.0),  # motor gain: bend of its segment's joint per unit output (radians)
    (0.0, 24.0),  # synapses it tries to receive
] * N_TYPES + [(-4.0, 4.0)] * (N_TYPES * N_TYPES) + [
    (0.0, 1.0),  # ≥ 0.5: modulatory (its output is broadcast as the reward/punishment signal)
    (-0.1, 0.1),  # learning rate of the synapses onto it (three-factor rule)
    (0.0, 0.5),  # habituation: depression of its output synapses per unit activity
    (1.0, 200.0),  # recovery time of those synapses (ticks)
] * N_TYPES
GENE_NAMES = ["div_size", "eta_h2", "eta_meth", "eta_anaer", "t_opt", "t_width", "mut_rate",
              "col_r", "col_g", "col_b", "eta_photo_an", "eta_photo_ox", "eta_aer", "eta_sfix",
              "eta_sox", "eta_mox", "o2_tol", "eta_pred", "speed", "prey_ratio"] + [
    f"brain{k}" for k in range(N_BRAIN)] + ["adhesion", "adapt_tau", "diff", "n_seg", "seg_cells",
                                         "muscle_a", "muscle_b", "cilia_a", "cilia_b", "germ_a",
                                         "germ_b", "cpg_amp", "cpg_freq", "cpg_lag"] + [
    f"n{k}_{p}" for k in range(N_TYPES)
    for p in ("count", "segmental", "tau", "bias", "channel", "sgain", "mgain", "density")] + [
    f"w{k}_{l}" for k in range(N_TYPES) for l in range(N_TYPES)] + [
    f"n{k}_{p}" for k in range(N_TYPES) for p in ("modulatory", "eta", "habituation", "recovery")]

# B-plan founder (§3): H2 chemosynthesis + methanogenesis only
# zero brain: no movement, no predation, no division inhibition (stage-3a behaviour)
FOUNDER = [0.02, 0.5, 0.3, 0.0, 30.0, 20.0, 0.02, 0.85, 0.35, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 2.0] + [0.0] * N_BRAIN + [0.0, 5.0] + [
    0.0, 4.0, 32.0] + [0.0] * 6 + [0.3, 0.05, 0.5] + [  # no differentiation: a single cell
    0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 8.0] * N_TYPES + [0.0] * (N_TYPES * N_TYPES) + [  # no neurons
    0.0, 0.0, 0.0, 50.0] * N_TYPES  # no learning, no habituation

assert len(GENE_RANGE) == len(GENE_NAMES) == len(FOUNDER) == N_GENES
