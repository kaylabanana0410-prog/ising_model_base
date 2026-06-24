# ═══════════════════════════════════════════════════════════════════════════
# IIT/GIM Project
# Cleaned version
# ═══════════════════════════════════════════════════════════════════════════

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import pearsonr

import ising as I
import utils
from Code_Kayla_Sleep import config_pearson as cfg
import param_anneal as pa
import temp_sweep as ts


# ── config ────────────────────────────────────────────────────────────────
SEED          = 1
N             = cfg.regions

ANNEAL_STEPS  = 100 #10000
ANNEAL_MAXFUN = 5 #500
ANNEAL_THERM  = 50 #5000

N_RESTARTS    = 3 ##  was 5 :5 annealing runs and picks the best one

T_MIN         = 2
T_MAX         = 18.5
T_STEPS       = 50 #400
TEMP_REPEATS  = 3 #10

# True  = set empirical/simulated FC diagonals to 0 and compare off-diagonal FC only.
# False = set empirical/simulated FC diagonals to 1 and include diagonals in FC correlations.
ZERO_FC_DIAGONAL = getattr(cfg, "ZERO_FC_DIAGONAL", True)

N_NULL        = 100
NULL_RUNS     = 5
NULL_STEPS    = 2000
NULL_THERM    = 1000

BINS          = 30
N_POST_CRIT_MATRICES = 5

# Optional:
# If you want to use the previous person's alpha = 2.07, set this to True.
# For your own optimized result, keep it False.
USE_FIXED_ALPHA = False
FIXED_ALPHA     = 2.07

BLUE   = "#2E86AB"
RED    = "#E84855"
AMBER  = "#F4A261"
PURPLE = "#6A0572"

np.random.seed(SEED)


# ── helper functions ──────────────────────────────────────────────────────
def upper_tri_vec(mat):
    idx = np.triu_indices(mat.shape[0], k=1)
    return mat[idx]


def set_fc_diagonal(mat):
    np.fill_diagonal(mat, 0 if ZERO_FC_DIAGONAL else 1)
    return mat


def fc_compare_vec(mat):
    if ZERO_FC_DIAGONAL:
        return upper_tri_vec(mat)
    return mat.ravel()


def clean_vec(vec):
    """Return a finite vector so null distributions/plots cannot become all-NaN."""
    return np.nan_to_num(np.asarray(vec, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def safe_pearson(x, y):
    """Pearson r that returns 0 instead of NaN for constant or non-finite vectors."""
    x = clean_vec(x)
    y = clean_vec(y)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0

    r = pearsonr(x, y)[0]
    return 0.0 if not np.isfinite(r) else float(r)


def finite_vals(vals, name):
    """Drop non-finite null values and stop with a clear error if none are usable."""
    vals = np.asarray(vals, dtype=float)
    good = vals[np.isfinite(vals)]

    dropped = len(vals) - len(good)
    if dropped > 0:
        print(f"WARNING: dropped {dropped}/{len(vals)} non-finite values from {name}")

    if len(good) == 0:
        raise ValueError(
            f"All values in {name} are NaN/inf. Check Ising simulation output and temperature array."
        )

    return good


def symmetric_norm_from_offdiag(mat, percentile=99, min_lim=0.05):
    """
    Creates a symmetric TwoSlopeNorm centered at 0.
    Uses off-diagonal entries only so the diagonal does not dominate color scale.
    """
    offdiag = mat[~np.eye(mat.shape[0], dtype=bool)]
    lim = np.percentile(np.abs(offdiag), percentile)

    if not np.isfinite(lim) or lim < min_lim:
        lim = min_lim

    return TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim), lim


def evenly_spaced_indices(indices, n_select):
    if len(indices) <= n_select:
        return indices

    positions = np.linspace(0, len(indices) - 1, n_select, dtype=int)
    return indices[positions]


# ── data ──────────────────────────────────────────────────────────────────
J_real  = np.genfromtxt(cfg.AVG_JIJ_NEW_PATH, delimiter=",").astype(float)
np.fill_diagonal(J_real, 0)
rho_emp = cfg.avg_FC.copy()          # Pearson empirical FC throughout
multiplier = utils.normalize_array(np.mean(J_real, axis=0))

# Set FC diagonal for plotting/saving and choose whether comparisons include it.
set_fc_diagonal(rho_emp)

rho_emp_vec = clean_vec(fc_compare_vec(rho_emp))

print("J_real min:          ", J_real.min())
print("J_real max:          ", J_real.max())
print("J_real has negatives:", np.any(J_real < 0))
print("emp FC neg fraction: ", np.mean(rho_emp_vec < 0))


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 : PARAMETER ANNEALING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 1 : PARAMETER ANNEALING — searching (T*, alpha*)")
print("=" * 65)

best_result = None
best_optim = None
best_fun = np.inf

for restart_idx in range(N_RESTARTS):
    np.random.seed(SEED + restart_idx)

    print(f"\nRestart {restart_idx + 1}/{N_RESTARTS}")

    optim = pa.optimize(
        ising      = I.Jij_sorted_ising,
        Jij        = J_real,
        partial    = False,
        multiplier = multiplier,
        save       = (restart_idx == 0)
    )
    
    result = optim.anneal(
        steps           = ANNEAL_STEPS,
        maxfun          = ANNEAL_MAXFUN,
        emp_FC          = rho_emp,
        therm           = ANNEAL_THERM,
        no_local_search = False,
        show            = False
    )

    print(f"restart best r = {max(optim.correlate):.4f}")
    print(f"restart fun    = {result.fun:.6f}")

    if result.fun < best_fun:
        best_fun = result.fun
        best_result = result
        best_optim = optim

result = best_result
optim = best_optim

T_star_annealed     = result.x[0]
alpha_star_annealed = result.x[1]

print(f"\nAnnealed T*     = {T_star_annealed:.4f}")
print(f"Annealed alpha* = {alpha_star_annealed:.4f}")
print(f"Annealing best r = {max(optim.correlate):.4f}")

optim.plot_error(show=False)
plt.savefig("param_anneal_error_3.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: param_anneal_error_3.png")


# ── choose alpha ──────────────────────────────────────────────────────────
if USE_FIXED_ALPHA:
    alpha_star = FIXED_ALPHA
    print(f"\nUsing fixed alpha = {alpha_star:.4f}")
else:
    alpha_star = alpha_star_annealed
    print(f"\nUsing annealed alpha* = {alpha_star:.4f}")

T_star = T_star_annealed


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 : TEMPERATURE SWEEP
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"STEP 2 : TEMPERATURE SWEEP  (alpha = {alpha_star:.4f})")
print("=" * 65)

sweep = ts.simulated_FC_vs_T_global(
    min_temp   = T_MIN,
    max_temp   = T_MAX,
    temp_step  = T_STEPS,
    alpha      = alpha_star,
    Jij        = J_real,
    ising      = I.Jij_sorted_ising,
    multiplier = multiplier,
    save       = True
)

sweep.simulate(
    steps          = ANNEAL_STEPS,
    thermalization = ANNEAL_THERM,
    partial        = False,
    diag           = not ZERO_FC_DIAGONAL,
    text           = True,
    n_repeats      = TEMP_REPEATS
)

# ── NaN guard ─────────────────────────────────────────────────────────────
corr_arr      = np.array(sweep.corr_ar_total)
spec_heat_arr = np.array(sweep.spec_heat_ar)

n_nan = np.sum(np.isnan(corr_arr))
print(f"NaN correlations in sweep: {n_nan}/{len(corr_arr)}")

# override T_crit and T_best using cleaned arrays
T_crit    = sweep.T_global[np.nanargmax(spec_heat_arr)]
T_best    = sweep.T_global[np.nanargmax(corr_arr)]
best_corr = np.nanmax(corr_arr)

# also patch the sweep object so downstream code is consistent
sweep.crit_temp  = T_crit
sweep.best_temp  = T_best
sweep.best_corr  = best_corr
sweep.best_ising = sweep.ising_ar[np.nanargmax(corr_arr)]
sweep.crit_ising = sweep.ising_ar[np.nanargmax(spec_heat_arr)]

T_global = sweep.T_global

print(f"\nCritical temperature (peak spec. heat) : {T_crit:.4f}")
print(f"Best-match temperature (peak r)        : {T_best:.4f}")
print(f"Best Pearson r                         : {best_corr:.4f}")


# ── observables ───────────────────────────────────────────────────────────
avg_energy = np.array(sweep.avg_energy_ar)
avg_energy_se = np.array(sweep.avg_energy_se_ar)
avg_mag = np.array(sweep.avg_mag_ar)
avg_mag_se = np.array(sweep.avg_mag_se_ar)
suscept = np.array(sweep.suscept_ar)
suscept_se = np.array(sweep.suscept_se_ar)
spec_heat = np.array(sweep.spec_heat_ar)
spec_heat_se = np.array(sweep.spec_heat_se_ar)


# ── Figure 1: E, |M|, susceptibility, specific heat vs T ──────────────────
fig1, axes1 = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
fig1.suptitle(
    f"Ising model — temperature sweep  |  alpha = {alpha_star:.3f}",
    fontsize=14,
    fontweight="bold"
)

panels = [
    (axes1[0, 0], avg_energy, avg_energy_se, r"average energy $\langle E \rangle$", "Energy vs T"),
    (axes1[0, 1], avg_mag, avg_mag_se, r"average $|M|$", "|Magnetization| vs T"),
    (axes1[1, 0], suscept, suscept_se, r"susceptibility $\chi$", "Susceptibility vs T"),
    (axes1[1, 1], spec_heat, spec_heat_se, r"specific heat $C$", "Specific Heat vs T"),
]

for ax, data, se, ylabel, title in panels:
    ax.plot(T_global, data, "o-", color=BLUE, lw=1.8, ms=3)
    ax.fill_between(T_global, data - se, data + se, color=BLUE, alpha=0.18, linewidth=0)

    ax.axvline(
        T_crit,
        color=RED,
        linestyle="--",
        lw=1.8,
        label=f"$T_{{crit}}$ = {T_crit:.2f}  (peak C)"
    )

    ax.axvline(
        T_best,
        color=AMBER,
        linestyle=":",
        lw=1.8,
        label=f"$T_{{best}}$ = {T_best:.2f}  (peak r)"
    )

    ax.set_xlabel("global temperature  T", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9, framealpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

plt.savefig("temperature_sweep_3.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
print("Saved: temperature_sweep_3.png")


# ── Figure 2: correlation vs T ────────────────────────────────────────────
fig_corr, ax_corr = plt.subplots(figsize=(7, 4), constrained_layout=True)

for data, se, color, label, lw, alpha in [
    (np.array(sweep.corr_ar_total), np.array(sweep.corr_se_ar_total), BLUE, "avg FC", 2, 1.0),
    (np.array(sweep.corr_ar_1), np.array(sweep.corr_se_ar_1), "green", "FC1", 1, 0.7),
    (np.array(sweep.corr_ar_2), np.array(sweep.corr_se_ar_2), "purple", "FC2", 1, 0.7),
    (np.array(sweep.corr_ar_3), np.array(sweep.corr_se_ar_3), "orange", "FC3", 1, 0.7),
]:
    ax_corr.plot(T_global, data, color=color, lw=lw, label=label, alpha=alpha)
    ax_corr.fill_between(T_global, data - se, data + se, color=color, alpha=0.12, linewidth=0)

ax_corr.axvline(T_crit, color=RED, linestyle="--", lw=1.5, label=f"T_crit = {T_crit:.2f}")
ax_corr.axvline(T_best, color=AMBER, linestyle=":", lw=1.5, label=f"T_best = {T_best:.2f}")

ax_corr.set_xlabel("global temperature  T", fontsize=11)
ax_corr.set_ylabel("Pearson r  (sim Pearson FC vs emp Pearson FC)", fontsize=11)
ax_corr.set_title("Correlation vs Temperature", fontsize=12)
ax_corr.legend(fontsize=9, framealpha=0.3)
ax_corr.spines[["top", "right"]].set_visible(False)

plt.savefig("correlation_vs_T_3.png", dpi=150, bbox_inches="tight")
plt.close(fig_corr)
print("Saved: correlation_vs_T_3.png")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 : MATRIX COMPARISON  (T_best) — Pearson FC only
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 3 : MATRIX COMPARISON  (T_best, Pearson FC)")
print("=" * 65)

best_gd = sweep.best_ising
sim_FC  = best_gd.FC.copy()
Jij_mat = best_gd.Jij.copy()

set_fc_diagonal(sim_FC)

sim_FC_vec = clean_vec(fc_compare_vec(sim_FC))

r_best    = safe_pearson(sim_FC_vec, rho_emp_vec)
dist_best = np.linalg.norm(sim_FC_vec - rho_emp_vec)
diss_best = 1.0 - r_best

print(f"sim FC neg fraction : {np.mean(sim_FC_vec  < 0):.4f}")
print(f"emp FC neg fraction : {np.mean(rho_emp_vec < 0):.4f}")
print(f"sim FC range        : {sim_FC_vec.min():.4f} → {sim_FC_vec.max():.4f}")
print(f"emp FC range        : {rho_emp_vec.min():.4f} → {rho_emp_vec.max():.4f}")
print(f"r              = {r_best:.4f}")
print(f"eucl. distance = {dist_best:.4f}")
print(f"dissimilarity  = {diss_best:.4f}")


# ── color normalization ──────────────────────────────────────────────────
# Use one shared norm for simulated and empirical FC.
combined_fc = np.concatenate([sim_FC_vec, rho_emp_vec])
fc_lim = np.percentile(np.abs(combined_fc), 99)

if not np.isfinite(fc_lim) or fc_lim < 0.05:
    fc_lim = 0.2

fc_norm = TwoSlopeNorm(vmin=-fc_lim, vcenter=0, vmax=fc_lim)

# Use separate norm for Jij because it may have a different scale.
j_offdiag = Jij_mat[~np.eye(Jij_mat.shape[0], dtype=bool)]
j_lim = np.percentile(np.abs(j_offdiag), 99)

if not np.isfinite(j_lim) or j_lim < 0.05:
    j_lim = 0.2

j_norm = TwoSlopeNorm(vmin=-j_lim, vcenter=0, vmax=j_lim)

print(f"FC color limit  : ±{fc_lim:.4f}")
print(f"Jij color limit : ±{j_lim:.4f}")


# ── matrix figure ────────────────────────────────────────────────────────
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

fig3.suptitle(
    f"Matrix comparison  |  T_best={T_best:.2f}  |  alpha={alpha_star:.2f}  |  r={r_best:.4f}",
    fontsize=13,
    fontweight="bold"
)

matrix_panels = [
    (sim_FC,  f"Simulated Pearson FC\n(T={T_best:.2f}, alpha={alpha_star:.2f})", fc_norm),
    (rho_emp, "Empirical Pearson FC", fc_norm),
    (Jij_mat, "Structural connectivity  $J_{ij}$", j_norm),
]

for ax, (mat, title, norm_to_use) in zip(axes3, matrix_panels):
    im = ax.matshow(mat, cmap="RdBu_r", norm=norm_to_use)
    ax.set_title(title, fontsize=11, pad=12)
    ax.set_xlabel("region", fontsize=9)
    ax.set_ylabel("region", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.savefig("matrix_comparison_3.png", dpi=150, bbox_inches="tight")
plt.close(fig3)


# ── scatter: sim vs emp ──────────────────────────────────────────────────
fig3s, ax3s = plt.subplots(figsize=(6, 5), constrained_layout=True)

ax3s.scatter(
    rho_emp_vec,
    sim_FC_vec,
    s=2,
    alpha=0.3,
    color=BLUE,
    rasterized=True
)

m, b = np.polyfit(rho_emp_vec, sim_FC_vec, 1)
x_line = np.linspace(rho_emp_vec.min(), rho_emp_vec.max(), 200)

ax3s.plot(x_line, m * x_line + b, color="black", lw=1.5, linestyle="--")

ax3s.set_xlabel("empirical Pearson FC", fontsize=11)
ax3s.set_ylabel("simulated Pearson FC", fontsize=11)
ax3s.set_title(f"Sim vs Emp Pearson FC  (r = {r_best:.4f})", fontsize=12)
ax3s.spines[["top", "right"]].set_visible(False)

plt.savefig("scatter_sim_vs_emp_3.png", dpi=150, bbox_inches="tight")
plt.close(fig3s)

print("Saved: matrix_comparison_3.png, scatter_sim_vs_emp_3.png")


# ── additional matrix comparisons after Tcrit ────────────────────────────
post_crit_indices = np.where(T_global > T_crit)[0]
best_idx = int(np.nanargmax(corr_arr))
post_crit_indices = post_crit_indices[post_crit_indices != best_idx]
post_crit_indices = evenly_spaced_indices(post_crit_indices, N_POST_CRIT_MATRICES)

if len(post_crit_indices) > 0:
    fig3_post, axes3_post = plt.subplots(
        len(post_crit_indices),
        3,
        figsize=(15, 3.8 * len(post_crit_indices)),
        constrained_layout=True,
        squeeze=False,
    )

    fig3_post.suptitle(
        f"Post-critical matrix comparisons  |  Tcrit={T_crit:.2f}  |  alpha={alpha_star:.2f}",
        fontsize=13,
        fontweight="bold"
    )

    print("\nPost-critical matrix comparisons:")

    for row, idx in enumerate(post_crit_indices):
        T_here = T_global[idx]
        gd_here = sweep.ising_ar[idx]
        sim_here = gd_here.FC.copy()
        set_fc_diagonal(sim_here)

        sim_here_vec = clean_vec(fc_compare_vec(sim_here))
        r_here = safe_pearson(sim_here_vec, rho_emp_vec)
        dist_here = np.linalg.norm(sim_here_vec - rho_emp_vec)

        print(f"  T={T_here:.4f}  r={r_here:.4f}  dist={dist_here:.4f}")

        row_panels = [
            (sim_here, f"Simulated Pearson FC\nT={T_here:.2f}, r={r_here:.4f}", fc_norm),
            (rho_emp, "Empirical Pearson FC", fc_norm),
            (Jij_mat, "Structural connectivity  $J_{ij}$", j_norm),
        ]

        for ax, (mat, title, norm_to_use) in zip(axes3_post[row], row_panels):
            im = ax.matshow(mat, cmap="RdBu_r", norm=norm_to_use)
            ax.set_title(title, fontsize=10, pad=10)
            ax.set_xlabel("region", fontsize=8)
            ax.set_ylabel("region", fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.savefig("matrix_comparisons_post_Tcrit_3.png", dpi=150, bbox_inches="tight")
    plt.close(fig3_post)
    print("Saved: matrix_comparisons_post_Tcrit_3.png")
else:
    print("No post-critical temperatures available for extra matrix comparisons.")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 : NULL DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"STEP 4 : NULL DISTRIBUTION  (N={N_NULL}, partial=False)")
print(f"         T_best = {T_best:.3f}  |  alpha = {alpha_star:.3f}")
print("=" * 65)


def shuffle_jij(J):
    J_null = J.copy()

    idx  = np.triu_indices(J.shape[0], k=1)
    vals = J_null[idx].copy()

    np.random.shuffle(vals)

    J_null[idx]            = vals
    J_null[idx[1], idx[0]] = vals

    return J_null


def pearson_threshold_jij(J, Rho, threshold):
    Rho_thresh = Rho.copy()
    Rho_thresh[np.abs(Rho_thresh) < threshold] = 0.0

    J_thresh = J.copy()
    offdiag = ~np.eye(J_thresh.shape[0], dtype=bool)
    keep = offdiag & (Rho_thresh != 0.0)
    drop = offdiag & (Rho_thresh == 0.0)

    J_thresh[drop] = 0.0
    J_thresh[keep] = np.sign(Rho_thresh[keep]) * np.abs(J_thresh[keep])
    np.fill_diagonal(J_thresh, 0)

    return (J_thresh + J_thresh.T) / 2.0


def ones_jij_like(J):
    J_ones = np.ones_like(J, dtype=float)
    np.fill_diagonal(J_ones, 0)
    return J_ones


def run_ising_avg(J, T_global_value, alpha, n_runs=NULL_RUNS):
    """
    Run null Ising model and return a finite Pearson FC matrix.

    Important fix: the null Jij matrices can have negative row/column means.
    With non-integer alpha, negative_mu ** alpha creates NaNs. Temperatures
    must be non-negative, so we build mu from absolute mean coupling strength.
    """
    J = np.asarray(J, dtype=float)

    # Signed/shuffled Jij can have negative means. Temperature multipliers must be >= 0.
    mu_loc = np.abs(np.mean(J, axis=0))
    mu_loc = np.nan_to_num(mu_loc, nan=0.0, posinf=0.0, neginf=0.0)

    max_mu = np.max(mu_loc)
    if not np.isfinite(max_mu) or max_mu <= 0:
        print("WARNING: null Jij has zero/non-finite mean coupling; using uniform temperature multipliers.")
        mu_loc = np.ones(J.shape[0], dtype=float)
    else:
        mu_loc = mu_loc / max_mu

    mu_loc_sorted = mu_loc[utils.cross_sort(mu_loc)]

    temp_arr = T_global_value * (mu_loc_sorted ** alpha)
    temp_arr = np.nan_to_num(temp_arr, nan=T_global_value, posinf=T_global_value, neginf=T_global_value)
    temp_arr[temp_arr <= 0] = 1e-12

    fc_sum = np.zeros((N, N), dtype=float)

    for _ in range(n_runs):
        sim = I.Jij_sorted_ising(temp_arr, Jij=J)
        sim.simulate(NULL_STEPS, NULL_THERM)
        sim.generate_FC(partial=False)

        fc = np.nan_to_num(sim.functional_connectivity, nan=0.0, posinf=0.0, neginf=0.0)
        fc_sum += fc

    rho = fc_sum / n_runs
    rho = np.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0)
    set_fc_diagonal(rho)

    return rho

null_dist = []
null_diss = []

for i in range(N_NULL):
    J_null = pearson_threshold_jij(
        shuffle_jij(cfg.avg_Jij),
        rho_emp,
        cfg.THRESHOLD,
    )

    # IMPORTANT:
    # Use T_best here because the real model was evaluated at T_best.
    rho_null = run_ising_avg(J_null, T_best, alpha_star)

    vec_null = clean_vec(fc_compare_vec(rho_null))

    r_null = safe_pearson(vec_null, rho_emp_vec)

    null_dist.append(np.linalg.norm(vec_null - rho_emp_vec))
    null_diss.append(1.0 - r_null)

    if (i + 1) % 10 == 0:
        print(
            f"  {i+1}/{N_NULL}  "
            f"dist={null_dist[-1]:.4f}  "
            f"diss={null_diss[-1]:.4f}  "
            f"r={r_null:.4f}"
        )

null_dist = finite_vals(null_dist, "null_dist")
null_diss = finite_vals(null_diss, "null_diss")

p_dist = np.mean(null_dist <= dist_best)
p_diss = np.mean(null_diss <= diss_best)

print(f"\nreal dist  = {dist_best:.4f} | null mean = {null_dist.mean():.4f} | p = {p_dist:.4f}")
print(f"real diss  = {diss_best:.4f} | null mean = {null_diss.mean():.4f} | p = {p_diss:.4f}")


ones_dist = []
ones_diss = []
J_ones = ones_jij_like(J_real)

print("\nRunning constant-ones Jij null distribution")

for i in range(N_NULL):
    rho_ones = run_ising_avg(J_ones, T_best, alpha_star)

    vec_ones = clean_vec(fc_compare_vec(rho_ones))

    r_ones = safe_pearson(vec_ones, rho_emp_vec)

    ones_dist.append(np.linalg.norm(vec_ones - rho_emp_vec))
    ones_diss.append(1.0 - r_ones)

    if (i + 1) % 10 == 0:
        print(
            f"  ones {i+1}/{N_NULL}  "
            f"dist={ones_dist[-1]:.4f}  "
            f"diss={ones_diss[-1]:.4f}  "
            f"r={r_ones:.4f}"
        )

ones_dist = finite_vals(ones_dist, "ones_dist")
ones_diss = finite_vals(ones_diss, "ones_diss")

p_ones_dist = np.mean(ones_dist <= dist_best)
p_ones_diss = np.mean(ones_diss <= diss_best)

print(f"\nreal dist  = {dist_best:.4f} | ones null mean = {ones_dist.mean():.4f} | p = {p_ones_dist:.4f}")
print(f"real diss  = {diss_best:.4f} | ones null mean = {ones_diss.mean():.4f} | p = {p_ones_diss:.4f}")


# ── effect sizes ──────────────────────────────────────────────────────────
def cohens_d(null_vals, real_val):
    null_vals = finite_vals(null_vals, "cohens_d input")
    sd = null_vals.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return 0.0
    return (real_val - null_vals.mean()) / sd


def cliffs_delta(null_vals, real_val):
    null_vals = finite_vals(null_vals, "cliffs_delta input")
    greater = np.sum(null_vals > real_val)
    less    = np.sum(null_vals < real_val)

    return (greater - less) / len(null_vals)


def cliffs_magnitude(delta):
    a = abs(delta)

    if a < 0.147:
        return "negligible"
    if a < 0.330:
        return "small"
    if a < 0.474:
        return "medium"

    return "large"


def cohens_magnitude(d):
    a = abs(d)

    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"

    return "large"


cd_dist  = cohens_d(null_dist, dist_best)
cd_diss  = cohens_d(null_diss, diss_best)
cd_ones_dist = cohens_d(ones_dist, dist_best)
cd_ones_diss = cohens_d(ones_diss, diss_best)

cld_dist = cliffs_delta(null_dist, dist_best)
cld_diss = cliffs_delta(null_diss, diss_best)
cld_ones_dist = cliffs_delta(ones_dist, dist_best)
cld_ones_diss = cliffs_delta(ones_diss, diss_best)

print(f"\nCohen's d  (dist) = {cd_dist:.4f}  [{cohens_magnitude(cd_dist)}]")
print(f"Cohen's d  (diss) = {cd_diss:.4f}  [{cohens_magnitude(cd_diss)}]")
print(f"Cliff's δ  (dist) = {cld_dist:.4f}  [{cliffs_magnitude(cld_dist)}]")
print(f"Cliff's δ  (diss) = {cld_diss:.4f}  [{cliffs_magnitude(cld_diss)}]")
print(f"Cohen's d  (ones dist) = {cd_ones_dist:.4f}  [{cohens_magnitude(cd_ones_dist)}]")
print(f"Cohen's d  (ones diss) = {cd_ones_diss:.4f}  [{cohens_magnitude(cd_ones_diss)}]")
print(f"Cliff's δ  (ones dist) = {cld_ones_dist:.4f}  [{cliffs_magnitude(cld_ones_dist)}]")
print(f"Cliff's δ  (ones diss) = {cld_ones_diss:.4f}  [{cliffs_magnitude(cld_ones_diss)}]")


# ── Figure 4 ──────────────────────────────────────────────────────────────
NULL_COLOR = "#5BA4CF"
REAL_COLOR = "#C0392B"


def plot_null(ax, null_vals, real_val, p_val, cd, cld, xlabel, title):
    null_vals = finite_vals(null_vals, title)
    real_val = float(np.nan_to_num(real_val, nan=0.0, posinf=0.0, neginf=0.0))

    counts, edges = np.histogram(null_vals, bins=BINS)
    widths = np.diff(edges)

    for c, left, w in zip(counts, edges[:-1], widths):
        ax.bar(
            left,
            c,
            width=w,
            align="edge",
            color=REAL_COLOR if (left + w) <= real_val else NULL_COLOR,
            alpha=0.40 if (left + w) <= real_val else 0.80,
            edgecolor="white",
            linewidth=0.5
        )

    ax.axvline(
        real_val,
        color=REAL_COLOR,
        linestyle="--",
        lw=2.2,
        label=f"real $J_{{ij}}$  ({real_val:.4f})"
    )

    ax.text(
        0.97,
        0.95,
        f"p = {p_val:.4f}\n"
        f"Cohen's d = {cd:.3f}  [{cohens_magnitude(cd)}]\n"
        f"Cliff's δ = {cld:.3f}  [{cliffs_magnitude(cld)}]",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color=REAL_COLOR,
        fontweight="medium",
        linespacing=1.6,
        bbox=dict(
            boxstyle="round,pad=0.3",
            fc="white",
            ec=REAL_COLOR,
            alpha=0.6
        )
    )

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("count", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9, framealpha=0.3)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.spines[["top", "right"]].set_visible(False)


fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

fig4.suptitle(
    f"Ising null distribution  |  T_best = {T_best:.2f}  |  alpha = {alpha_star:.2f}  |  r = {r_best:.4f}",
    fontsize=13,
    fontweight="bold"
)

plot_null(
    axes4[0],
    null_dist,
    dist_best,
    p_dist,
    cd_dist,
    cld_dist,
    xlabel=r"euclidean distance  $||\rho_{sim} - \rho_{emp}||$",
    title="null distribution — euclidean distance"
)

plot_null(
    axes4[1],
    null_diss,
    diss_best,
    p_diss,
    cd_diss,
    cld_diss,
    xlabel="dissimilarity  (1 − r)",
    title="null distribution — dissimilarity"
)

plt.savefig("ising_null_distributions_3.png", dpi=150, bbox_inches="tight")
plt.close(fig4)

print("Saved: ising_null_distributions_3.png")


fig4_ones, axes4_ones = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

fig4_ones.suptitle(
    f"Constant-ones Jij null distribution  |  T_best = {T_best:.2f}  |  alpha = {alpha_star:.2f}  |  r = {r_best:.4f}",
    fontsize=13,
    fontweight="bold"
)

plot_null(
    axes4_ones[0],
    ones_dist,
    dist_best,
    p_ones_dist,
    cd_ones_dist,
    cld_ones_dist,
    xlabel=r"euclidean distance  $||\rho_{sim} - \rho_{emp}||$",
    title="ones Jij null — euclidean distance"
)

plot_null(
    axes4_ones[1],
    ones_diss,
    diss_best,
    p_ones_diss,
    cd_ones_diss,
    cld_ones_diss,
    xlabel="dissimilarity  (1 − r)",
    title="ones Jij null — dissimilarity"
)

plt.savefig("ising_null_distributions_ones_3.png", dpi=150, bbox_inches="tight")
plt.close(fig4_ones)

print("Saved: ising_null_distributions_ones_3.png")


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)

print(f"Annealed T*       = {T_star_annealed:.4f}")
print(f"Annealed alpha*   = {alpha_star_annealed:.4f}")

if USE_FIXED_ALPHA:
    print(f"Used alpha         = {alpha_star:.4f}  [fixed previous value]")
else:
    print(f"Used alpha         = {alpha_star:.4f}  [annealed value]")

print(f"T_crit            = {T_crit:.4f}  (peak specific heat)")
print(f"T_best            = {T_best:.4f}  (peak Pearson r, Pearson FC)")
print(f"best r            = {r_best:.4f}  (Pearson FC)")
print(f"eucl. distance    = {dist_best:.4f}")
print(f"dissimilarity     = {diss_best:.4f}")
print(f"p (dist)          = {p_dist:.4f}")
print(f"p (diss)          = {p_diss:.4f}")
print(f"p ones (dist)     = {p_ones_dist:.4f}")
print(f"p ones (diss)     = {p_ones_diss:.4f}")
print(f"Cohen's d (dist)  = {cd_dist:.4f}  [{cohens_magnitude(cd_dist)}]")
print(f"Cohen's d (diss)  = {cd_diss:.4f}  [{cohens_magnitude(cd_diss)}]")
print(f"Cliff's δ (dist)  = {cld_dist:.4f}  [{cliffs_magnitude(cld_dist)}]")
print(f"Cliff's δ (diss)  = {cld_diss:.4f}  [{cliffs_magnitude(cld_diss)}]")
print(f"Cohen's d ones (dist) = {cd_ones_dist:.4f}  [{cohens_magnitude(cd_ones_dist)}]")
print(f"Cohen's d ones (diss) = {cd_ones_diss:.4f}  [{cohens_magnitude(cd_ones_diss)}]")
print(f"Cliff's δ ones (dist) = {cld_ones_dist:.4f}  [{cliffs_magnitude(cld_ones_dist)}]")
print(f"Cliff's δ ones (diss) = {cld_ones_diss:.4f}  [{cliffs_magnitude(cld_ones_diss)}]")

print("\nOutput files:")
for f in [
    "param_anneal_error_3.png",
    "temperature_sweep_3.png",
    "correlation_vs_T_3.png",
    "matrix_comparison_3.png",
    "scatter_sim_vs_emp_3.png",
    "matrix_comparisons_post_Tcrit_3.png",
    "ising_null_distributions_3.png",
    "ising_null_distributions_ones_3.png"
]:
    print(f"  {f}")