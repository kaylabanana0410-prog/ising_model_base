# ═══════════════════════════════════════════════════════════════════════════
# IIT/GIM Project
# build_jij.py  —  Script 1
# ═══════════════════════════════════════════════════════════════════════════
#
# For each of the 25 subjects:
#   1. Load Jij_initial_i
#   2. Load the matching session Pearson FC (avg_TS_1 / avg_TS_2 / avg_TS_3)
#   3. Apply threshold  →  Jij_new_i
#   4. Save to  Jij_new_pearson/Jij_new_pearson_{i:02d}.csv
#
# Then average all 25  →  avg_Jij_new_pearson.csv
#
# Threshold rule (off-diagonal entries only):
#   |Rho_ij| <  THRESHOLD  →  keep Jij_initial[i,j]               (structural sign)
#   |Rho_ij| >= THRESHOLD  →  abs(Jij_initial[i,j]) * sign(Rho_ij) (empirical sign)
#   diagonal               →  0
#
# Run FIRST, before simulate_fc.py and analysis.py.

import os
import numpy as np
import pandas as pd

import config_pearson as cfg


# ── subject → session mapping ─────────────────────────────────────────────
# Maps subject index (1-based) to session key.
# Adjust if your subject-to-session assignment differs.
def subject_session(s: int) -> str:
    if s <= 9:   return "FC1"
    if s <= 17:  return "FC2"
    return "FC3"


# ══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════

def load_csv(path: str) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",", dtype=float)


def save_csv(matrix: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame(matrix).to_csv(path, index=False, header=False)


def build_Jij_new(
    Jij_initial: np.ndarray,
    Rho_empirical: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sign-corrected Jij update using Pearson FC.
    Returns (Jij_new, Rho_thresh).
    """
    Jij       = Jij_initial.copy().astype(float)
    Rho       = Rho_empirical.copy().astype(float)
    N         = Jij.shape[0]

    Rho_thresh = Rho.copy()
    Rho_thresh[np.abs(Rho) < threshold] = 0.0

    Jij_new = Jij.copy()
    for i in range(N):
        for j in range(N):
            if i == j:
                Jij_new[i, j] = 0.0
            elif Rho_thresh[i, j] != 0.0:
                Jij_new[i, j] = np.sign(Rho_thresh[i, j]) * abs(Jij[i, j])
            # else: structural sign kept unchanged

    return Jij_new, Rho_thresh


def enforce_symmetry(mat: np.ndarray, label: str = "") -> np.ndarray:
    err = np.max(np.abs(mat - mat.T))
    if err > 1e-10:
        print(f"    [symmetry] {label}: max asymmetry={err:.2e} — enforcing (A+Aᵀ)/2")
        mat = (mat + mat.T) / 2.0
    return mat


def print_diagnostics(Jij_init, Jij_new, Rho_thresh, s):
    N         = Jij_init.shape[0]
    off       = ~np.eye(N, dtype=bool)
    n_above   = np.sum(Rho_thresh[off] != 0)
    n_total   = np.sum(off)
    n_flipped = np.sum(np.sign(Jij_new[off]) != np.sign(Jij_init[off]))
    print(
        f"  Subject {s:02d} [{subject_session(s)}]: "
        f"above-thresh={n_above}/{n_total} ({100*n_above/n_total:.1f}%)  "
        f"sign-flips={n_flipped} ({100*n_flipped/n_total:.1f}%)"
    )


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    np.random.seed(cfg.SEED)
    subject_ids = getattr(cfg, "SUBJECT_IDS", range(1, cfg.N_SUBJECTS + 1))

    print("=" * 65)
    print(f"BUILD JIJ_NEW (Pearson)  —  {len(subject_ids)} subjects  "
          f"|  threshold={cfg.THRESHOLD}")
    print("=" * 65)

    # pre-load the 3 session Pearson FC matrices
    session_fc = {
        "FC1": load_csv(cfg.FC1_PATH).astype(float),
        "FC2": load_csv(cfg.FC2_PATH).astype(float),
        "FC3": load_csv(cfg.FC3_PATH).astype(float),
    }
    for mat in session_fc.values():
        np.fill_diagonal(mat, 0)

    print("\nLoaded session FC matrices: avg_TS_1, avg_TS_2, avg_TS_3")

    os.makedirs(cfg.JIJ_NEW_DIR, exist_ok=True)
    all_Jij_new = []

    for s in subject_ids:

        jij_path = os.path.join(cfg.JIJ_DIR, cfg.JIJ_PATTERN.format(s))
        Jij_init  = load_csv(jij_path).astype(float)
        Rho_emp_s = session_fc[subject_session(s)]

        assert Jij_init.shape == Rho_emp_s.shape, (
            f"Subject {s:02d}: shape mismatch "
            f"Jij {Jij_init.shape} vs Rho_emp {Rho_emp_s.shape}"
        )

        Jij_new, Rho_thresh = build_Jij_new(Jij_init, Rho_emp_s, cfg.THRESHOLD)
        Jij_new = enforce_symmetry(Jij_new, f"Subject {s:02d}")
        print_diagnostics(Jij_init, Jij_new, Rho_thresh, s)

        out_path = os.path.join(cfg.JIJ_NEW_DIR, f"Jij_new_pearson_{s:02d}.csv")
        save_csv(Jij_new, out_path)
        all_Jij_new.append(Jij_new)

    # average across subjects
    print(f"\n[averaging]  avg_Jij_new across {len(subject_ids)} subjects …")
    avg_Jij_new = np.mean(all_Jij_new, axis=0)
    np.fill_diagonal(avg_Jij_new, 0)

    save_csv(avg_Jij_new, cfg.AVG_JIJ_NEW_PATH)

    off = avg_Jij_new[~np.eye(avg_Jij_new.shape[0], dtype=bool)]
    print(
        f"  avg_Jij_new_pearson : shape={avg_Jij_new.shape}  "
        f"min={off.min():.4f}  max={off.max():.4f}  "
        f"neg_fraction={np.mean(off < 0):.3f}"
    )
    print(f"  Saved → {cfg.AVG_JIJ_NEW_PATH}")
    print(f"  Per-subject files → {cfg.JIJ_NEW_DIR}/")
    print("\nDone.  Next step → analysis.py")


if __name__ == "__main__":
    main()