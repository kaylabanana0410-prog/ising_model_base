# ═══════════════════════════════════════════════════════════════════════════
# IIT/GIM Project
# simulate_fc.py  —  Script 2  (optional sanity check)
# ═══════════════════════════════════════════════════════════════════════════
#
# Loads avg_Jij_new_pearson.csv and avg Pearson FC (mean of avg_TS_1/2/3),
# runs a single Ising forward pass → Rho_sim_pearson.csv.
#
# partial=False throughout (Pearson FC).
#
# Run AFTER build_jij.py.  analysis.py runs its own internal simulation
# with optimised T*/alpha* — this script is just a quick forward-pass check.

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import ising as I
import utils
import config_pearson as cfg


# ══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════

def load_csv(path: str) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",", dtype=float)


def save_csv(matrix: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame(matrix).to_csv(path, index=False, header=False)
    print(f"  Saved → {path}")


def load_avg_FC() -> np.ndarray:
    """Load avg_TS_1, avg_TS_2, avg_TS_3 and return their mean (avg_FC)."""
    FC1 = load_csv(cfg.FC1_PATH).astype(float); np.fill_diagonal(FC1, 0)
    FC2 = load_csv(cfg.FC2_PATH).astype(float); np.fill_diagonal(FC2, 0)
    FC3 = load_csv(cfg.FC3_PATH).astype(float); np.fill_diagonal(FC3, 0)
    avg = (FC1 + FC2 + FC3) / 3.0;              np.fill_diagonal(avg,  0)
    return avg


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    np.random.seed(cfg.SEED)

    print("=" * 65)
    print("SIMULATE FC (Pearson)  —  avg_Jij_new_pearson  →  Rho_sim_pearson")
    print(f"  steps          = {cfg.SIM_STEPS}")
    print(f"  thermalization = {cfg.SIM_THERMALIZATION}")
    print(f"  partial        = {cfg.PARTIAL}")
    print("=" * 65)

    # ── load avg_Jij_new_pearson ──────────────────────────────────────────
    print(f"\n[1] Loading avg_Jij_new_pearson …")
    J_real = load_csv(cfg.AVG_JIJ_NEW_PATH).astype(float)
    N = J_real.shape[0]
    print(f"  J_real min={J_real.min():.4f}  max={J_real.max():.4f}  "
          f"has_neg={np.any(J_real < 0)}")

    # ── load avg_FC ───────────────────────────────────────────────────────
    print(f"\n[2] Loading avg_TS_1, avg_TS_2, avg_TS_3 → avg_FC …")
    rho_emp = load_avg_FC()
    idx = np.triu_indices(N, k=1)
    rho_emp_vec = rho_emp[idx]
    print(f"  emp FC neg fraction: {np.mean(rho_emp_vec < 0):.4f}")

    # ── temperature array ─────────────────────────────────────────────────
    if cfg.USE_FIXED_ALPHA:
        alpha_use = cfg.FIXED_ALPHA
        T_use     = 8.15
    else:
        alpha_use = 0.0    # uniform; optimised values come from analysis.py
        T_use     = 10.0

    print(f"\n[3] T_global={T_use:.2f}  alpha={alpha_use:.4f}")

    ind_avg    = np.mean(J_real, axis=0)
    multiplier = utils.normalize_array(ind_avg)

    temp_arr = (np.full(N, T_use, dtype=float) if alpha_use == 0.0
                else T_use * (multiplier ** alpha_use))

    print(f"  temp mean={temp_arr.mean():.4f}  "
          f"min={temp_arr.min():.4f}  max={temp_arr.max():.4f}")

    # ── run Ising ─────────────────────────────────────────────────────────
    print(f"\n[4] Running Ising simulation …")
    sim = I.Jij_sorted_ising(temp_arr, Jij=J_real,
                              spin_ar=np.random.choice([-1, 1], N))
    sim.simulate(cfg.SIM_STEPS, cfg.SIM_THERMALIZATION)
    print(f"  Finished in {sim.timer:.2f}s")

    # ── generate Pearson FC ───────────────────────────────────────────────
    print(f"\n[5] Generating Pearson FC (partial=False) …")
    Rho_sim = sim.generate_FC(partial=False)
    np.fill_diagonal(Rho_sim, 0)

    sim_vec = Rho_sim[idx]
    r, _    = pearsonr(sim_vec, rho_emp_vec)
    dist    = np.linalg.norm(sim_vec - rho_emp_vec)

    print(f"  Pearson r (sim vs avg_FC, upper tri) = {r:.4f}")
    print(f"  Euclidean distance                   = {dist:.4f}")
    print(f"  sim FC range : {sim_vec.min():.4f} → {sim_vec.max():.4f}")
    print(f"  emp FC range : {rho_emp_vec.min():.4f} → {rho_emp_vec.max():.4f}")
    print(f"  sim neg frac : {np.mean(sim_vec < 0):.4f}")

    # ── save ──────────────────────────────────────────────────────────────
    print(f"\n[6] Saving …")
    save_csv(Rho_sim, cfg.RHO_SIMULATED_PATH)

    print("\nDone.  Next step → analysis.py")


if __name__ == "__main__":
    main() 
