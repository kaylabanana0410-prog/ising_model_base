import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "steven"
OUTPUT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = OUTPUT_DIR / "threshold_verification"
RESULTS_DIR.mkdir(exist_ok=True)


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import steven.ising3 as I
import steven.param_anneal as pa
import steven.temp_sweep as ts
import steven.utils as utils


JIJ_DIR = DATA_ROOT / "Jij data_raw"
JIJ_PATTERN = "Jij_{}.csv"
FC1_PATH = DATA_ROOT / "FC data_processed" / "avg_TS_1"
FC2_PATH = DATA_ROOT / "FC data_processed" / "avg_TS_2"
FC3_PATH = DATA_ROOT / "FC data_processed" / "avg_TS_3"
SUBJECT_IDS = list(range(2, 26))


# Match the current temperature-sweep settings in Project_1/pearson_2.py.
THRESHOLDS = np.arange(0.0, 1.001, 0.01)
SEED = 1
ANNEAL_STEPS = 1000
ANNEAL_MAXFUN = 500
ANNEAL_THERM = 600
SWEEP_STEPS = 1000
SWEEP_THERM = 2000
N_RESTARTS = 1
ANNEAL_BOUNDS = ((0.1, 10), (-3, 3))
REFINE_T_WINDOW = 1.0
REFINE_ALPHA_WINDOW = 0.5
REFINE_MAXFUN = 100
REFINE_MAX_ROUNDS = 2
REFINE_SHRINK = 0.5
REFINE_MIN_T_WINDOW = 0.001
REFINE_MIN_ALPHA_WINDOW = 0.0005
T_MIN = 0.5
T_MAX = 13
T_STEPS = 250
TEMP_REPEATS = 10
ZOOM_SWEEP_AROUND_T_STAR = False
SWEEP_T_WINDOW = 2.0


def subject_session(subject_id):
    if subject_id <= 9:
        return "FC1"
    if subject_id <= 17:
        return "FC2"
    return "FC3"


def upper_tri_vec(mat):
    idx = np.triu_indices(mat.shape[0], k=1)
    return mat[idx]


def safe_pearson(x, y):
    x = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(np.asarray(y, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0

    r = pearsonr(x, y)[0]
    return 0.0 if not np.isfinite(r) else float(r)


def threshold_jij(J, Rho, threshold):
    """
    Same threshold rule as Project_1/build_jij.py:
    above threshold: force Jij sign to match empirical Pearson FC sign.
    below threshold: keep original Jij sign/value.
    """
    J_thresh = J.copy().astype(float)
    Rho_thresh = Rho.copy().astype(float)
    Rho_thresh[np.abs(Rho_thresh) < threshold] = 0.0

    offdiag = ~np.eye(J_thresh.shape[0], dtype=bool)
    keep = offdiag & (Rho_thresh != 0.0)
    J_thresh[keep] = np.sign(Rho_thresh[keep]) * np.abs(J_thresh[keep])
    np.fill_diagonal(J_thresh, 0.0)

    return (J_thresh + J_thresh.T) / 2.0


def build_avg_thresholded_jij(subject_jij, session_fc, threshold):
    thresholded_jij = []

    for subject_id, Jij in subject_jij.items():
        Rho = session_fc[subject_session(subject_id)]
        Jij_thresh = threshold_jij(Jij, Rho, threshold)
        thresholded_jij.append(Jij_thresh)

    avg_jij = np.mean(thresholded_jij, axis=0)
    np.fill_diagonal(avg_jij, 0.0)
    return avg_jij


def refine_bounds(center, T_window, alpha_window, base_bounds=ANNEAL_BOUNDS):
    T_center, alpha_center = center
    (T_low, T_high), (alpha_low, alpha_high) = base_bounds
    return (
        max(T_low, T_center - T_window),
        min(T_high, T_center + T_window),
    ), (
        max(alpha_low, alpha_center - alpha_window),
        min(alpha_high, alpha_center + alpha_window),
    )


def anneal_params(Jij, multiplier, rho_emp, seed):
    best_result = None
    best_optim = None
    best_fun = np.inf

    for restart_idx in range(N_RESTARTS):
        np.random.seed(seed + restart_idx)
        print(f"    anneal restart {restart_idx + 1}/{N_RESTARTS}", flush=True)
        optim = pa.optimize(
            ising=I.Jij_sorted_ising,
            Jij=Jij,
            partial=False,
            multiplier=multiplier,
            save=False,
        )
        result = optim.anneal(
            steps=ANNEAL_STEPS,
            maxfun=ANNEAL_MAXFUN,
            emp_FC=rho_emp,
            therm=ANNEAL_THERM,
            no_local_search=False,
            show=False,
            bounds=ANNEAL_BOUNDS,
        )
        if result.fun < best_fun:
            best_fun = result.fun
            best_result = result
            best_optim = optim

    result = best_result
    optim = best_optim
    T_window = REFINE_T_WINDOW
    alpha_window = REFINE_ALPHA_WINDOW

    for refine_round in range(REFINE_MAX_ROUNDS):
        refined_bounds = refine_bounds(result.x, T_window, alpha_window)
        np.random.seed(seed + N_RESTARTS + refine_round)
        print(f"    refine round {refine_round + 1}/{REFINE_MAX_ROUNDS}", flush=True)
        refine_optim = pa.optimize(
            ising=I.Jij_sorted_ising,
            Jij=Jij,
            partial=False,
            multiplier=multiplier,
            save=False,
        )
        refine_result = refine_optim.anneal(
            steps=ANNEAL_STEPS,
            maxfun=REFINE_MAXFUN,
            emp_FC=rho_emp,
            therm=ANNEAL_THERM,
            no_local_search=False,
            show=False,
            bounds=refined_bounds,
        )
        if refine_result.fun < best_fun:
            best_fun = refine_result.fun
            result = refine_result
            optim = refine_optim

        T_window *= REFINE_SHRINK
        alpha_window *= REFINE_SHRINK
        if T_window <= REFINE_MIN_T_WINDOW and alpha_window <= REFINE_MIN_ALPHA_WINDOW:
            break

    return float(result.x[0]), float(result.x[1]), float(np.max(optim.correlate))


def run_pearson_pipeline_for_threshold(Jij, rho_emp, emp_FC1, emp_FC2, emp_FC3, seed):
    multiplier = utils.normalize_array(np.mean(np.abs(Jij), axis=0))
    T_star, alpha_star, anneal_best_r = anneal_params(Jij, multiplier, rho_emp, seed)

    if ZOOM_SWEEP_AROUND_T_STAR:
        T_sweep_min = max(ANNEAL_BOUNDS[0][0], T_star - SWEEP_T_WINDOW)
        T_sweep_max = min(ANNEAL_BOUNDS[0][1], T_star + SWEEP_T_WINDOW)
    else:
        T_sweep_min = T_MIN
        T_sweep_max = T_MAX

    sweep = ts.simulated_FC_vs_T_global(
        min_temp=T_sweep_min,
        max_temp=T_sweep_max,
        temp_step=T_STEPS,
        alpha=alpha_star,
        Jij=Jij,
        ising=I.Jij_sorted_ising,
        multiplier=multiplier,
        save=False,
    )
    sweep.simulate(
        steps=SWEEP_STEPS,
        thermalization=SWEEP_THERM,
        partial=False,
        diag=False,
        text=False,
        n_repeats=TEMP_REPEATS,
        emp_FC1=emp_FC1,
        emp_FC2=emp_FC2,
        emp_FC3=emp_FC3,
        avg_FC=rho_emp,
    )

    corr_arr = np.asarray(sweep.corr_ar_total, dtype=float)
    best_idx = int(np.nanargmax(corr_arr))
    return {
        "pearson_r": float(np.nanmax(corr_arr)),
        "T_best": float(sweep.T_global[best_idx]),
        "alpha": alpha_star,
        "T_star_annealed": T_star,
        "anneal_best_r": anneal_best_r,
    }


def main():
    np.random.seed(SEED)
    emp_FC1 = np.genfromtxt(FC1_PATH, delimiter=",").astype(float)
    emp_FC2 = np.genfromtxt(FC2_PATH, delimiter=",").astype(float)
    emp_FC3 = np.genfromtxt(FC3_PATH, delimiter=",").astype(float)

    session_fc = {
        "FC1": emp_FC1.copy(),
        "FC2": emp_FC2.copy(),
        "FC3": emp_FC3.copy(),
    }
    for mat in session_fc.values():
        np.fill_diagonal(mat, 0.0)

    subject_jij = {}
    for subject_id in SUBJECT_IDS:
        jij_path = JIJ_DIR / JIJ_PATTERN.format(subject_id)
        Jij = np.genfromtxt(jij_path, delimiter=",").astype(float)
        np.fill_diagonal(Jij, 0.0)
        subject_jij[subject_id] = Jij

    rho_emp = (emp_FC1 + emp_FC2 + emp_FC3) / 3.0
    np.fill_diagonal(rho_emp, 0.0)

    thresholds = THRESHOLDS
    pearson_rs = []

    for idx, threshold in enumerate(thresholds):
        print("\n" + "=" * 65, flush=True)
        print(f"Threshold {idx + 1}/{len(thresholds)}: {threshold:.2f}", flush=True)
        print("=" * 65, flush=True)
        Jij_thresh = build_avg_thresholded_jij(subject_jij, session_fc, threshold)
        result = run_pearson_pipeline_for_threshold(
            Jij_thresh,
            rho_emp,
            emp_FC1,
            emp_FC2,
            emp_FC3,
            SEED + idx * 1000,
        )
        pearson_rs.append(result["pearson_r"])
        print(
            f"threshold={threshold:.2f}  "
            f"Pearson r={result['pearson_r']:.4f}  "
            f"T_best={result['T_best']:.4f}  "
            f"alpha={result['alpha']:.4f}",
            flush=True,
        )

    results = pd.DataFrame({
        "threshold": thresholds,
        "pearson_r": pearson_rs,
    })
    csv_path = RESULTS_DIR / "threshold_vs_gim_pearson_r.csv"
    results.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    pearson_rs = np.asarray(pearson_rs, dtype=float)
    ax.plot(thresholds, pearson_rs, color="#2E86AB", lw=2)
    ax.scatter(thresholds, pearson_rs, color="#2E86AB", s=16)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Pearson r  (simulated FC vs empirical FC)")
    ax.set_title("Threshold vs GIM Pearson correlation")
    ax.spines[["top", "right"]].set_visible(False)

    plot_path = RESULTS_DIR / "threshold_vs_gim_pearson_r.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    best_idx = int(np.nanargmax(pearson_rs))
    print(f"Saved: {csv_path}")
    print(f"Saved: {plot_path}")
    print(
        "Best threshold: "
        f"{thresholds[best_idx]:.4f}  |  Pearson r: {pearson_rs[best_idx]:.4f}"
    )


if __name__ == "__main__":
    main()
