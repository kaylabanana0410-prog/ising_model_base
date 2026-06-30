from pathlib import Path
import numpy as np

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent

JIJ_DIR = PROJECT_ROOT / "Jij data_raw"
JIJ_PATTERN = "Jij_{}.csv"
JIJ_NEW_DIR = OUTPUT_DIR / "Jij_new_pearson"
AVG_JIJ_NEW_PATH = JIJ_NEW_DIR / "avg_Jij_new_pearson.csv"

FC1_PATH = PROJECT_ROOT / "FC data_processed" / "avg_TS_1"
FC2_PATH = PROJECT_ROOT / "FC data_processed" / "avg_TS_2"
FC3_PATH = PROJECT_ROOT / "FC data_processed" / "avg_TS_3"
RHO_SIMULATED_PATH = OUTPUT_DIR / "Rho_sim_pearson.csv"

# Data settings
SUBJECT_IDS = list(range(2, 26))
N_SUBJECTS = len(SUBJECT_IDS)
THRESHOLD = 0.03
PARTIAL = False

# True  = set empirical/simulated FC diagonals to 0 and compare off-diagonal FC only.
# False = set empirical/simulated FC diagonals to 1 and include diagonals in FC correlations.
ZERO_FC_DIAGONAL = True

# Simulation settings
SIM_STEPS = 10000
SIM_THERMALIZATION = 5000

# Random seed
SEED = 1

# Annealing settings
ANNEAL_STEPS = 10000
ANNEAL_MAXFUN = 500
ANNEAL_THERM = 5000
N_RESTARTS = 5

# Temperature sweep settings
T_MIN = 1.5
T_MAX = 18
T_STEPS = 150
TEMP_REPEATS = 10

# Null distribution settings
N_NULL = 100
NULL_RUNS = 5
NULL_STEPS = 2000
NULL_THERM = 1000

# Plot settings
BINS = 30

# Optional fixed alpha
USE_FIXED_ALPHA = False
FIXED_ALPHA = 2.07

# Colors
BLUE = "#2E86AB"
RED = "#E84855"
AMBER = "#F4A261"
PURPLE = "#6A0572"


def load_csv(path: Path) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",", dtype=float)


# Compatibility values used by older pearson scripts.
avg_Jij = load_csv(PROJECT_ROOT / "Jij data_processed" / "avg_Jij_no_outliers_norm")
regions = avg_Jij.shape[0]

FC_1 = load_csv(FC1_PATH)
FC_2 = load_csv(FC2_PATH)
FC_3 = load_csv(FC3_PATH)
avg_FC = (FC_1 + FC_2 + FC_3) / 3.0

FC_1p = load_csv(PROJECT_ROOT / "FC data_processed" / "avg_TS_1p")
FC_2p = load_csv(PROJECT_ROOT / "FC data_processed" / "avg_TS_2p")
FC_3p = load_csv(PROJECT_ROOT / "FC data_processed" / "avg_TS_3p")
avg_FCp = (FC_1p + FC_2p + FC_3p) / 3.0

ind_avg_Jij = np.mean(avg_Jij, axis=0)