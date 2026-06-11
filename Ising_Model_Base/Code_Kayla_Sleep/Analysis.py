# ═══════════════════════════════════════════════════════════════════════════
# IIT/GIM Project
# analysis.py — Script 3
# ═══════════════════════════════════════════════════════════════════════════
#
# Run AFTER:
#   1) build_jij.py
#   2) simulate_fc.py  [optional sanity check]
#
# This script:
#   1. Loads avg_Jij_new_pearson.csv
#   2. Loads empirical Pearson FC: avg_TS_1, avg_TS_2, avg_TS_3
#   3. Runs parameter annealing to estimate T* and alpha*
#   4. Runs a temperature sweep
#   5. Finds:
#        - T_crit = peak specific heat
#        - T_best = best simulated-vs-empirical FC match
#   6. Saves:
#        - temperature_sweep.png
#        - correlation_vs_T.png
#        - matrix_comparison.png
#        - scatter_sim_vs_emp.png
#        - ising_null_distributions.png
#        - analysis_summary.txt
#
# Main rule:
#   Pearson is used throughout the main pipeline.
#   Partial FC is only computed at the end as an optional extra comparison.

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import pearsonr

# ─────────────────────────────────────────────────────────────────────────
# PATH SETUP
# ─────────────────────────────────────────────────────────────────────────

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent

# Allows importing root-level files: ising.py, utils.py, param_anneal.py, temp_sweep.py
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CODE_DIR))

import config_pearson as cfg
import config as base_cfg
import ising as I
import utils
import param_anneal as pa
import temp_sweep as ts


# ══════════════════════════════════════════════════════════════════════════
# CONFIG FALLBACKS
# ══════════════════════════════════════════════════════════════════════════

SEED = getattr(cfg, "SEED", 1)

ANNEAL_STEPS = getattr(cfg, "ANNEAL_STEPS", 10000)
ANNEAL_MAXFUN = getattr(cfg, "ANNEAL_MAXFUN", 500)
ANNEAL_THERM = getattr(cfg, "ANNEAL_THERM", 5000)
N_RESTARTS = getattr(cfg, "N_RESTARTS", 5)

T_MIN = getattr(cfg, "T_MIN", 2)
T_MAX = getattr(cfg, "T_MAX", 25)
T_STEPS = getattr(cfg, "T_STEPS", 400)

N_NULL = getattr(cfg, "N_NULL", 100)
NULL_RUNS = getattr(cfg, "NULL_RUNS", 5)
NULL_STEPS = getattr(cfg, "NULL_STEPS", 2000)
NULL_THERM = getattr(cfg, "NULL_THERM", 1000)

BINS = getattr(cfg, "BINS", 30)

USE_FIXED_ALPHA = getattr(cfg, "USE_FIXED_ALPHA", False)
FIXED_ALPHA = getattr(cfg, "FIXED_ALPHA", 2.07)

BLUE = getattr(cfg, "BLUE", "#2E86AB")
RED = getattr(cfg, "RED", "#E84855")
AMBER = getattr(cfg, "AMBER", "#F4A261")
PURPLE = getattr(cfg, "PURPLE", "#6A0572")

# Optional:
# Set to True if you want final partial-correlation comparison at the end.
DO_PARTIAL_AT_END = False

# Output paths
RESULTS_DIR = CODE_DIR / "analysis_results"
RESULTS_DIR.mkdir(exist_ok=True)

# Try multiple likely names/locations for avg_Jij_new_pearson.csv
AVG_JIJ_CANDIDATES = [
    getattr(cfg, "AVG_JIJ_NEW_PATH", None),
    CODE_DIR / "avg_Jij_new_pearson.csv",
    CODE_DIR / "Jij_new_pearson" / "avg_Jij_new_pearson.csv",
    PROJECT_ROOT / "avg_Jij_new_pearson.csv",
]

AVG_JIJ_CANDIDATES = [Path(p) for p in AVG_JIJ_CANDIDATES if p is not None]


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

def upper_tri_vec(mat: np.ndarray) -> np.ndarray:
    """Return off-diagonal upper triangle as vector."""
    idx = np.triu_indices(mat.shape[0], k=1)
    return mat[idx]


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r with NaN/constant guards."""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return pearsonr(x, y)[0]


def load_csv(path: Path) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",", dtype=float)


def save_csv(mat: np.ndarray, path: Path) -> None:
    np.savetxt(path, mat, delimiter=",")


def find_existing_path(candidates, label: str) -> Path:
    for p in candidates:
        if p.exists():
            return p

    print(f"\nCould not find {label}. Tried:")
    for p in candidates:
        print(f"  - {p}")

    raise FileNotFoundError(f"Missing required file: {label}")


def load_avg_jij_new() -> np.ndarray:
    path = find_existing_path(AVG_JIJ_CANDIDATES, "avg_Jij_new_pearson.csv")
    print(f"Loading Jij_new from: {path}")
    J = load_csv(path).astype(float)
    np.fill_diagonal(J, 0)
    return J


def load_empirical_pearson_fc() -> np.ndarray:
    """
    Use root-level config.py if possible:
        base_cfg.avg_FC = average of avg_TS_1, avg_TS_2, avg_TS_3

    If that fails, try CSV paths from config_pearson.py.
    """

    if hasattr(base_cfg, "avg_FC"):
        rho = base_cfg.avg_FC.copy().astype(float)
        np.fill_diagonal(rho, 0)
        print("Loaded empirical Pearson FC from base config: base_cfg.avg_FC")
        return rho

    fc_paths = [
        getattr(cfg, "FC1_PATH", None),
        getattr(cfg, "FC2_PATH", None),
        getattr(cfg, "FC3_PATH", None),
    ]

    if all(p is not None for p in fc_paths):
        mats = []
        for p in fc_paths:
            mat = load_csv(Path(p)).astype(float)
            np.fill_diagonal(mat, 0)
            mats.append(mat)

        rho = np.mean(mats, axis=0)
        np.fill_diagonal(rho, 0)
        print("Loaded empirical Pearson FC from cfg.FC1_PATH/FC2_PATH/FC3_PATH")
        return rho

    raise FileNotFoundError(
        "Could not load empirical Pearson FC. Expected base_cfg.avg_FC "
        "or cfg.FC1_PATH/cfg.FC2_PATH/cfg.FC3_PATH."
    )


def load_empirical_partial_fc_or_none() -> np.ndarray | None:
    """Optional final comparison only."""
    if hasattr(base_cfg, "avg_FCp"):
        rho = base_cfg.avg_FCp.copy().astype(float)
        np.fill_diagonal(rho, 0)
        print("Loaded empirical partial FC from base config: base_cfg.avg_FCp")
        return rho

    return None


def symmetric_norm_from_vec(vec, percentile=99, min_lim=0.05):
    lim = np.percentile(np.abs(vec[np.isfinite(vec)]), percentile)
    if not np.isfinite(lim) or lim < min_lim:
        lim = min_lim
    return TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim), lim


def make_temp_array(J: np.ndarray, T_global: float, alpha: float) -> np.ndarray:
    """
    Build node-wise temperature array from Jij node strength.

    If alpha = 0:
        uniform temperature.
    Else:
        T_i = T_global * multiplier_i^alpha
    """

    N = J.shape[0]
    multiplier = utils.normalize_array(np.mean(J, axis=0))

    if alpha == 0:
        temp_arr = np.full(N, T_global, dtype=float)
    else:
        temp_arr = T_global * (multiplier ** alpha)

    # Guard against exactly zero temperatures.
    temp_arr = np.maximum(temp_arr, 1e-8)
    return temp_arr


def run_single_ising_fc(
    J: np.ndarray,
    T_global: float,
    alpha: float,
    steps: int,
    thermalization: int,
    partial: bool = False,
) -> np.ndarray:
    """Run one Ising simulation and return FC matrix."""
    N = J.shape[0]
    temp_arr = make_temp_array(J, T_global, alpha)

    sim = I.Jij_sorted_ising(
        temp_arr,
        Jij=J,
        spin_ar=np.random.choice([-1, 1], N),
    )

    sim.simulate(steps, thermalization)
    fc = sim.generate_FC(partial=partial)

    # Different versions of your code store FC under different names.
    if fc is None:
        if hasattr(sim, "functional_connectivity"):
            fc = sim.functional_connectivity
        elif hasattr(sim, "FC"):
            fc = sim.FC
        else:
            raise AttributeError("Could not find generated FC in Ising object.")

    fc = np.asarray(fc, dtype=float)
    np.fill_diagonal(fc, 0)
    return fc


def shuffle_jij_preserve_symmetry(J: np.ndarray) -> np.ndarray:
    """Shuffle off-diagonal upper-triangle weights while preserving symmetry."""
    J_null = J.copy()
    idx = np.triu_indices(J.shape[0], k=1)
    vals = J_null[idx].copy()

    np.random.shuffle(vals)

    J_null[idx] = vals
    J_null[idx[1], idx[0]] = vals
    np.fill_diagonal(J_null, 0)

    return J_null


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    np.random.seed(SEED)

    print("=" * 70)
    print("ANALYSIS — Pearson Jij sign-corrected Ising model")
    print("=" * 70)

    # ─────────────────────────────────────────────────────────────────────
    # STEP 0: LOAD DATA
    # ─────────────────────────────────────────────────────────────────────

    print("\nSTEP 0 — Loading data")

    J_real = load_avg_jij_new()
    rho_emp = load_empirical_pearson_fc()

    N = J_real.shape[0]
    assert J_real.shape == rho_emp.shape, (
        f"Shape mismatch: J_real {J_real.shape} vs rho_emp {rho_emp.shape}"
    )

    rho_emp_vec = upper_tri_vec(rho_emp)
    J_vec = upper_tri_vec(J_real)

    print(f"J_real shape: {J_real.shape}")
    print(f"J_real range: {J_vec.min():.4f} → {J_vec.max():.4f}")
    print(f"J_real negative fraction: {np.mean(J_vec < 0):.4f}")
    print(f"Empirical Pearson FC range: {rho_emp_vec.min():.4f} → {rho_emp_vec.max():.4f}")
    print(f"Empirical Pearson FC negative fraction: {np.mean(rho_emp_vec < 0):.4f}")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 1: PARAMETER ANNEALING
    # ─────────────────────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("STEP 1 — Parameter annealing: searching T* and alpha*")
    print("=" * 70)

    best_result = None
    best_optim = None
    best_fun = np.inf

    multiplier = utils.normalize_array(np.mean(J_real, axis=0))

    for restart_idx in range(N_RESTARTS):
        np.random.seed(SEED + restart_idx)

        print(f"\nRestart {restart_idx + 1}/{N_RESTARTS}")

        optim = pa.optimize(
            ising=I.Jij_sorted_ising,
            Jij=J_real,
            partial=False,          # Pearson only
            multiplier=multiplier,
            save=(restart_idx == 0),
        )

        result = optim.anneal(
            steps=ANNEAL_STEPS,
            maxfun=ANNEAL_MAXFUN,
            emp_FC=rho_emp,
            therm=ANNEAL_THERM,
            no_local_search=False,
            show=False,
        )

        restart_best_r = np.nanmax(optim.correlate)
        print(f"restart best Pearson r = {restart_best_r:.4f}")
        print(f"restart objective fun = {result.fun:.6f}")

        if result.fun < best_fun:
            best_fun = result.fun
            best_result = result
            best_optim = optim

    result = best_result
    optim = best_optim

    T_star_annealed = float(result.x[0])
    alpha_star_annealed = float(result.x[1])

    print(f"\nAnnealed T* = {T_star_annealed:.4f}")
    print(f"Annealed alpha* = {alpha_star_annealed:.4f}")
    print(f"Annealing best Pearson r = {np.nanmax(optim.correlate):.4f}")

    try:
        optim.plot_error(show=False)
        plt.savefig(RESULTS_DIR / "param_anneal_error.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {RESULTS_DIR / 'param_anneal_error.png'}")
    except Exception as e:
        print(f"Could not save annealing error plot: {e}")

    if USE_FIXED_ALPHA:
        alpha_star = FIXED_ALPHA
        print(f"\nUsing fixed alpha = {alpha_star:.4f}")
    else:
        alpha_star = alpha_star_annealed
        print(f"\nUsing annealed alpha* = {alpha_star:.4f}")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 2: TEMPERATURE SWEEP
    # ─────────────────────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print(f"STEP 2 — Temperature sweep with alpha = {alpha_star:.4f}")
    print("=" * 70)

    sweep = ts.simulated_FC_vs_T_global(
        min_temp=T_MIN,
        max_temp=T_MAX,
        temp_step=T_STEPS,
        alpha=alpha_star,
        Jij=J_real,
        ising=I.Jij_sorted_ising,
        multiplier=multiplier,
        save=True,
    )

    sweep.simulate(
        steps=ANNEAL_STEPS,
        thermalization=ANNEAL_THERM,
        partial=False,       # Pearson only
        text=True,
    )

    corr_arr = np.array(sweep.corr_ar_total, dtype=float)
    spec_heat_arr = np.array(sweep.spec_heat_ar, dtype=float)

    n_nan = np.sum(~np.isfinite(corr_arr))
    print(f"NaN correlations in sweep: {n_nan}/{len(corr_arr)}")

    T_global = np.array(sweep.T_global, dtype=float)

    T_crit = T_global[np.nanargmax(spec_heat_arr)]
    T_best = T_global[np.nanargmax(corr_arr)]
    best_corr = np.nanmax(corr_arr)

    sweep.crit_temp = T_crit
    sweep.best_temp = T_best
    sweep.best_corr = best_corr
    sweep.best_ising = sweep.ising_ar[np.nanargmax(corr_arr)]
    sweep.crit_ising = sweep.ising_ar[np.nanargmax(spec_heat_arr)]

    print(f"\nCritical temperature T_crit: {T_crit:.4f}")
    print(f"Best-match temperature T_best: {T_best:.4f}")
    print(f"Best Pearson r: {best_corr:.4f}")

    # Observables
    avg_energy = np.array([np.mean(gd.ising.energy_series) for gd in sweep.ising_ar])
    avg_mag = np.array([np.mean(np.abs(gd.ising.mag_series)) for gd in sweep.ising_ar])
    suscept = np.array(sweep.suscept_ar, dtype=float)
    spec_heat = np.array(sweep.spec_heat_ar, dtype=float)

    # ── Figure 1: thermodynamic observables
    fig1, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    fig1.suptitle(
        f"Ising temperature sweep | Pearson FC | alpha = {alpha_star:.3f}",
        fontsize=14,
        fontweight="bold",
    )

    panels = [
        (axes[0, 0], avg_energy, r"average energy $\langle E \rangle$", "Energy vs T"),
        (axes[0, 1], avg_mag, r"average $|M|$", "|Magnetization| vs T"),
        (axes[1, 0], suscept, r"susceptibility $\chi$", "Susceptibility vs T"),
        (axes[1, 1], spec_heat, r"specific heat $C$", "Specific Heat vs T"),
    ]

    for ax, data, ylabel, title in panels:
        ax.plot(T_global, data, "o-", color=BLUE, lw=1.8, ms=3)
        ax.axvline(
            T_crit,
            color=RED,
            linestyle="--",
            lw=1.8,
            label=f"T_crit = {T_crit:.2f}",
        )
        ax.axvline(
            T_best,
            color=AMBER,
            linestyle=":",
            lw=1.8,
            label=f"T_best = {T_best:.2f}",
        )
        ax.set_xlabel("global temperature T")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9, framealpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    fig1_path = RESULTS_DIR / "temperature_sweep.png"
    plt.savefig(fig1_path, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved: {fig1_path}")

    # ── Figure 2: correlation vs T
    fig2, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)

    ax.plot(T_global, corr_arr, color=BLUE, lw=2, label="avg empirical Pearson FC")
    ax.axvline(T_crit, color=RED, linestyle="--", lw=1.5, label=f"T_crit = {T_crit:.2f}")
    ax.axvline(T_best, color=AMBER, linestyle=":", lw=1.5, label=f"T_best = {T_best:.2f}")

    ax.set_xlabel("global temperature T")
    ax.set_ylabel("Pearson r")
    ax.set_title("Correlation vs Temperature")
    ax.legend(fontsize=9, framealpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    fig2_path = RESULTS_DIR / "correlation_vs_T.png"
    plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {fig2_path}")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 3: MATRIX COMPARISON AT T_best
    # ─────────────────────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("STEP 3 — Matrix comparison at T_best")
    print("=" * 70)

    best_gd = sweep.best_ising

    sim_FC = best_gd.FC.copy()
    Jij_mat = best_gd.Jij.copy()

    np.fill_diagonal(sim_FC, 0)
    np.fill_diagonal(Jij_mat, 0)

    sim_FC_vec = upper_tri_vec(sim_FC)
    Jij_vec = upper_tri_vec(Jij_mat)

    r_best = safe_pearson(sim_FC_vec, rho_emp_vec)
    dist_best = np.linalg.norm(sim_FC_vec - rho_emp_vec)
    diss_best = 1.0 - r_best

    print(f"Sim FC negative fraction: {np.mean(sim_FC_vec < 0):.4f}")
    print(f"Emp FC negative fraction: {np.mean(rho_emp_vec < 0):.4f}")
    print(f"Sim FC range: {sim_FC_vec.min():.4f} → {sim_FC_vec.max():.4f}")
    print(f"Emp FC range: {rho_emp_vec.min():.4f} → {rho_emp_vec.max():.4f}")
    print(f"r = {r_best:.4f}")
    print(f"Euclidean distance = {dist_best:.4f}")
    print(f"Dissimilarity = {diss_best:.4f}")

    # Shared color norm for FCs
    combined_fc = np.concatenate([sim_FC_vec, rho_emp_vec])
    fc_norm, fc_lim = symmetric_norm_from_vec(combined_fc, percentile=99, min_lim=0.05)

    # Separate color norm for Jij
    j_norm, j_lim = symmetric_norm_from_vec(Jij_vec, percentile=99, min_lim=0.05)

    print(f"FC color limit: ±{fc_lim:.4f}")
    print(f"Jij color limit: ±{j_lim:.4f}")

    fig3, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    fig3.suptitle(
        f"Matrix comparison | T_best={T_best:.2f} | alpha={alpha_star:.2f} | r={r_best:.4f}",
        fontsize=13,
        fontweight="bold",
    )

    matrix_panels = [
        (sim_FC, f"Simulated Pearson FC\nT={T_best:.2f}, alpha={alpha_star:.2f}", fc_norm),
        (rho_emp, "Empirical Pearson FC", fc_norm),
        (Jij_mat, "Sign-corrected $J_{ij}$", j_norm),
    ]

    for ax, (mat, title, norm_to_use) in zip(axes, matrix_panels):
        im = ax.matshow(mat, cmap="RdBu_r", norm=norm_to_use)
        ax.set_title(title, fontsize=11, pad=12)
        ax.set_xlabel("region")
        ax.set_ylabel("region")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig3_path = RESULTS_DIR / "matrix_comparison.png"
    plt.savefig(fig3_path, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved: {fig3_path}")

    # Scatter plot
    fig4, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)

    ax.scatter(rho_emp_vec, sim_FC_vec, s=2, alpha=0.3, color=BLUE, rasterized=True)

    m, b = np.polyfit(rho_emp_vec, sim_FC_vec, 1)
    x_line = np.linspace(rho_emp_vec.min(), rho_emp_vec.max(), 200)
    ax.plot(x_line, m * x_line + b, color="black", lw=1.5, linestyle="--")

    ax.set_xlabel("empirical Pearson FC")
    ax.set_ylabel("simulated Pearson FC")
    ax.set_title(f"Sim vs Emp Pearson FC (r = {r_best:.4f})")
    ax.spines[["top", "right"]].set_visible(False)

    fig4_path = RESULTS_DIR / "scatter_sim_vs_emp.png"
    plt.savefig(fig4_path, dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print(f"Saved: {fig4_path}")

    save_csv(sim_FC, RESULTS_DIR / "sim_FC_best_pearson.csv")
    save_csv(rho_emp, RESULTS_DIR / "emp_FC_pearson.csv")
    save_csv(Jij_mat, RESULTS_DIR / "Jij_used.csv")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 4: NULL DISTRIBUTION
    # ─────────────────────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print(f"STEP 4 — Null distribution, N_NULL = {N_NULL}")
    print("=" * 70)

    null_corr = []
    null_diss = []

    for i in range(N_NULL):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Null run {i + 1}/{N_NULL}")

        J_null = shuffle_jij_preserve_symmetry(J_real)

        fc_sum = np.zeros((N, N), dtype=float)

        for _ in range(NULL_RUNS):
            fc_null = run_single_ising_fc(
                J=J_null,
                T_global=T_best,
                alpha=alpha_star,
                steps=NULL_STEPS,
                thermalization=NULL_THERM,
                partial=False,
            )
            fc_sum += fc_null

        fc_null_avg = fc_sum / NULL_RUNS
        np.fill_diagonal(fc_null_avg, 0)

        null_vec = upper_tri_vec(fc_null_avg)
        r_null = safe_pearson(null_vec, rho_emp_vec)

        null_corr.append(r_null)
        null_diss.append(1.0 - r_null)

    null_corr = np.array(null_corr, dtype=float)
    null_diss = np.array(null_diss, dtype=float)

    p_value = np.mean(null_corr >= r_best)

    print(f"\nReal r = {r_best:.4f}")
    print(f"Null mean r = {np.nanmean(null_corr):.4f}")
    print(f"Null std r = {np.nanstd(null_corr):.4f}")
    print(f"Empirical p-value = {p_value:.4f}")

    np.savetxt(RESULTS_DIR / "null_corr.csv", null_corr, delimiter=",")
    np.savetxt(RESULTS_DIR / "null_dissimilarity.csv", null_diss, delimiter=",")

    fig5, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)

    ax.hist(null_corr, bins=BINS, alpha=0.75, color=BLUE)
    ax.axvline(r_best, color=RED, linestyle="--", lw=2, label=f"real r = {r_best:.4f}")
    ax.set_xlabel("Pearson r under shuffled Jij null")
    ax.set_ylabel("count")
    ax.set_title(f"Null distribution | p = {p_value:.4f}")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    fig5_path = RESULTS_DIR / "ising_null_distributions.png"
    plt.savefig(fig5_path, dpi=150, bbox_inches="tight")
    plt.close(fig5)
    print(f"Saved: {fig5_path}")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 5: OPTIONAL PARTIAL FC AT THE END
    # ─────────────────────────────────────────────────────────────────────

    partial_r = None
    partial_dist = None

    if DO_PARTIAL_AT_END:
        print("\n" + "=" * 70)
        print("STEP 5 — Optional partial FC comparison at the end")
        print("=" * 70)

        rho_emp_partial = load_empirical_partial_fc_or_none()

        if rho_emp_partial is None:
            print("No empirical partial FC found, skipping partial comparison.")
        else:
            np.random.seed(SEED + 999)

            sim_partial = run_single_ising_fc(
                J=J_real,
                T_global=T_best,
                alpha=alpha_star,
                steps=ANNEAL_STEPS,
                thermalization=ANNEAL_THERM,
                partial=True,
            )

            rho_emp_partial_vec = upper_tri_vec(rho_emp_partial)
            sim_partial_vec = upper_tri_vec(sim_partial)

            partial_r = safe_pearson(sim_partial_vec, rho_emp_partial_vec)
            partial_dist = np.linalg.norm(sim_partial_vec - rho_emp_partial_vec)

            print(f"Partial FC final r = {partial_r:.4f}")
            print(f"Partial FC final distance = {partial_dist:.4f}")

            save_csv(sim_partial, RESULTS_DIR / "sim_FC_best_partial_final.csv")
            save_csv(rho_emp_partial, RESULTS_DIR / "emp_FC_partial.csv")

    # ─────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────

    summary_path = RESULTS_DIR / "analysis_summary.txt"

    with open(summary_path, "w") as f:
        f.write("ISING MODEL ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        f.write("Main analysis type: Pearson FC\n")
        f.write("Jij input: avg_Jij_new_pearson.csv\n\n")

        f.write(f"N regions: {N}\n")
        f.write(f"Annealed T*: {T_star_annealed:.6f}\n")
        f.write(f"Annealed alpha*: {alpha_star_annealed:.6f}\n")
        f.write(f"Used alpha: {alpha_star:.6f}\n\n")

        f.write(f"T_crit peak specific heat: {T_crit:.6f}\n")
        f.write(f"T_best peak Pearson r: {T_best:.6f}\n")
        f.write(f"Best Pearson r: {r_best:.6f}\n")
        f.write(f"Euclidean distance: {dist_best:.6f}\n")
        f.write(f"Dissimilarity 1-r: {diss_best:.6f}\n\n")

        f.write(f"Sim FC negative fraction: {np.mean(sim_FC_vec < 0):.6f}\n")
        f.write(f"Emp FC negative fraction: {np.mean(rho_emp_vec < 0):.6f}\n\n")

        f.write("Null distribution\n")
        f.write(f"N_NULL: {N_NULL}\n")
        f.write(f"Null mean r: {np.nanmean(null_corr):.6f}\n")
        f.write(f"Null std r: {np.nanstd(null_corr):.6f}\n")
        f.write(f"Empirical p-value: {p_value:.6f}\n\n")

        if partial_r is not None:
            f.write("Optional final partial FC comparison\n")
            f.write(f"Partial r: {partial_r:.6f}\n")
            f.write(f"Partial distance: {partial_dist:.6f}\n")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Saved all results to: {RESULTS_DIR}")
    print(f"Summary file: {summary_path}")


if __name__ == "__main__":
    main()
