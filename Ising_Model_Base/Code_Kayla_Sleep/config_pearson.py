from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent

# Random seed
SEED = 1

# Annealing settings
ANNEAL_STEPS = 10000
ANNEAL_MAXFUN = 500
ANNEAL_THERM = 5000
N_RESTARTS = 5

# Temperature sweep settings
T_MIN = 2
T_MAX = 25
T_STEPS = 400

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
