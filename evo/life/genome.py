"""Genome layout (§3, stage-2 subset). Genes are stored directly in trait units and clamped to
their range after mutation; mutation noise is scaled by each trait's range."""

DIV_SIZE, ETA_H2, ETA_METH, ETA_ANAER, T_OPT, T_WIDTH, MUT_RATE, COL_R, COL_G, COL_B = range(10)
N_GENES = 10

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
]
GENE_NAMES = ["div_size", "eta_h2", "eta_meth", "eta_anaer", "t_opt", "t_width", "mut_rate",
              "col_r", "col_g", "col_b"]

# B-plan founder (§3): H2 chemosynthesis + methanogenesis only
FOUNDER = [0.02, 0.5, 0.3, 0.0, 30.0, 20.0, 0.02, 0.85, 0.35, 0.25]

assert len(GENE_RANGE) == len(GENE_NAMES) == len(FOUNDER) == N_GENES
