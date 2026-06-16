# ═══════════════════════════════════════════════════════════════════════════
# IIT/GIM Project  —  OPTIMIZED
# Improvements: adaptive thermalization near T_crit, equilibration diagnostics,
#               re-enabled local search, multi-restart annealing, consistent
#               null budget, raised NULL_RUNS / N_NULL.
# ═══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# Monday May 4 2026
# ---------------------------------------------------------------------------
# Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import pearsonr

import ising as I
import utils
import config as cf
import param_anneal as pa
import temp_sweep as ts


# ── config ──────────────────────────────────────────────────────────────────

SEED          = 1
N             = cf.regions      # (84x84)

ANNEAL_STEPS  = 5000            # ising steps/annealing
ANNEAL_MAXFUN = 500             # annealing group size before reset
ANNEAL_THERM  = 1000            # thermalization steps

T_MIN         = 1.0             # sweep range
T_MAX         = 20.0
T_STEPS       = 100             # number of temperature pts (coarse pass)

# Fine-pass window around T_crit
T_FINE_STEPS  = 40              # resolution inside the critical window
T_FINE_THERM  = 5_000           # 5× longer thermalization near phase transition
T_FINE_WINDOW = 3.0             # ± degrees around T_crit for fine pass

# Param-anneal restarts — run N_RESTARTS independent anneals, keep best
N_RESTARTS    = 5

# Null distribution
N_NULL        = 200             # raised from 100  → p-value resolution = 0.005
NULL_RUNS     = 10              # raised from 5    → more stable FC average per sample
NULL_STEPS    = ANNEAL_STEPS    # match real-run budget (was 2000, inconsistent)
NULL_THERM    = ANNEAL_THERM    # match real-run thermalization

BINS          = 30

np.random.seed(SEED)


# ── data ────────────────────────────────────────────────────────────────────

J_real  = cf.avg_Jij.copy()
rho_emp = cf.avg_FC.copy()
np.fill_diagonal(rho_emp, 0)

# per-neuron temp multiplier (same formula as param_anneal + temp_sweep)
mu        = np.mean(J_real, axis=0)
mu        = mu / mu.max()
mu_sorted = mu[utils.cross_sort(mu)]


# ── helpers ──────────────────────────────────────────────────────────────────

def upper_tri_vec(mat):
    """Pull upper triangle of a matrix into a flat 1-D array."""
    idx = np.triu_indices(mat.shape[0], k=1)
    return mat[idx]


def check_equilibrated(energy_series, window_frac=0.2, tol=0.01):
    """
    Compare mean energy of first vs last `window_frac` of the run.
    Returns (is_ok: bool, relative_drift: float).

    Near T_crit the autocorrelation time diverges; a large drift here
    means the system has not yet thermalised and results are unreliable.
    """
    n          = len(energy_series)
    w          = max(1, int(n * window_frac))
    mean_start = np.mean(energy_series[:w])
    mean_end   = np.mean(energy_series[-w:])
    rel_drift  = abs(mean_start - mean_end) / (abs(mean_end) + 1e-10)
    return rel_drift < tol, rel_drift


rho_emp_vec = upper_tri_vec(rho_emp)


# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Param_anneal — find T* and alpha*
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("STEP 1 : PARAMETER ANNEALING — searching (T*, alpha*)")
print("=" * 65)

# Run N_RESTARTS independent anneals and keep the best result.
# Simulated annealing can get trapped in local minima; multiple restarts
# with local-search re-enabled substantially improves convergence.
best_result = None
best_fun    = np.inf

for restart_idx in range(N_RESTARTS):
    np.random.seed(SEED + restart_idx)
    print(f"  restart {restart_idx + 1}/{N_RESTARTS} …", end="", flush=True)

    optim = pa.optimize(
        ising      = I.Jij_sorted_ising,
        Jij        = J_real,
        partial    = False,
        multiplier = utils.normalize_array(cf.ind_avg_Jij),
        save       = (restart_idx == 0)   # only save plotting data on first run
    )
    res = optim.anneal(
        steps           = ANNEAL_STEPS,
        maxfun          = ANNEAL_MAXFUN,
        emp_FC          = rho_emp,
        therm           = ANNEAL_THERM,
        no_local_search = False,          # re-enabled: refines solution after global search
        show            = False
    )
    print(f"  r = {max(optim.correlate):.4f}  fun = {res.fun:.6f}")

    if res.fun < best_fun:
        best_fun    = res.fun
        best_result = res
        best_optim  = optim

np.random.seed(SEED)   # restore seed for reproducibility

T_star     = best_result.x[0]
alpha_star = best_result.x[1]
print(f"\nT*     = {T_star:.4f}")
print(f"alpha* = {alpha_star:.4f}")
print(f"best r = {max(best_optim.correlate):.4f}")

best_optim.plot_error(show=False)
plt.savefig("param_anneal_error.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: param_anneal_error.png")

# param_anneal_error.png — correlation vs iteration during annealing.
# α = 0 → uniform temperature
# α > 0 → region-specific temperatures
# Clear valley of low error; temperature scales almost linearly with node
# strength (alpha* close to 1).


# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Temp_sweep — two-pass strategy for accurate critical-region results
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print(f"STEP 2 : TEMPERATURE SWEEP  (alpha* = {alpha_star:.4f})")
print("=" * 65)

# ── Pass 1: coarse sweep across full range ───────────────────────────────
print("\n  Pass 1 — coarse sweep (locating T_crit) …")
sweep_coarse = ts.simulated_FC_vs_T_global(
    min_temp   = T_MIN,
    max_temp   = T_MAX,
    temp_step  = T_STEPS,
    alpha      = alpha_star,
    Jij        = J_real,
    ising      = I.Jij_sorted_ising,
    multiplier = utils.normalize_array(cf.ind_avg_Jij),
    save       = False
)
sweep_coarse.simulate(
    steps          = ANNEAL_STEPS,
    thermalization = ANNEAL_THERM,
    partial        = False,
    text           = False
)
T_crit_approx = sweep_coarse.crit_temp
print(f"  Approximate T_crit = {T_crit_approx:.4f}")

# ── Pass 2: fine sweep near T_crit with longer thermalization ────────────
# Near the phase transition the autocorrelation time diverges (critical
# slowing down). T_FINE_THERM = 5000 gives the system time to equilibrate.
T_FINE_MIN = max(T_MIN, T_crit_approx - T_FINE_WINDOW)
T_FINE_MAX = min(T_MAX, T_crit_approx + T_FINE_WINDOW)

print(f"\n  Pass 2 — fine sweep  T ∈ [{T_FINE_MIN:.2f}, {T_FINE_MAX:.2f}]"
      f"  therm={T_FINE_THERM} …")
sweep_fine = ts.simulated_FC_vs_T_global(
    min_temp   = T_FINE_MIN,
    max_temp   = T_FINE_MAX,
    temp_step  = T_FINE_STEPS,
    alpha      = alpha_star,
    Jij        = J_real,
    ising      = I.Jij_sorted_ising,
    multiplier = utils.normalize_array(cf.ind_avg_Jij),
    save       = True
)
sweep_fine.simulate(
    steps          = ANNEAL_STEPS,
    thermalization = T_FINE_THERM,
    partial        = False,
    text           = True
)

# Use the fine sweep as the canonical result
sweep    = sweep_fine
T_crit   = sweep.crit_temp
T_best   = sweep.best_temp
best_corr = sweep.best_corr
T_global = sweep.T_global

print(f"\nCritical temperature (peak spec. heat) : {T_crit:.4f}")
print(f"Best-match temperature (peak r)        : {T_best:.4f}")
print(f"Best Pearson r                         : {best_corr:.4f}")

# ── Equilibration diagnostic ─────────────────────────────────────────────
print("\n  Equilibration check across fine sweep:")
bad_temps = []
for T_val, gd in zip(T_global, sweep.ising_ar):
    ok, drift = check_equilibrated(gd.ising.energy_series)
    if not ok:
        bad_temps.append((T_val, drift))

if bad_temps:
    print(f"  ⚠  {len(bad_temps)} temperature(s) NOT equilibrated:")
    for T_val, drift in bad_temps:
        print(f"       T={T_val:.3f}  relative drift={drift:.4f}")
    print("  → Consider raising T_FINE_THERM or ANNEAL_STEPS.")
else:
    print("  ✓  All temperatures appear equilibrated.")

# ── collect per-temperature observables ─────────────────────────────────
avg_energy = np.array([np.mean(gd.ising.energy_series)         for gd in sweep.ising_ar])
avg_mag    = np.array([np.mean(np.abs(gd.ising.mag_series))    for gd in sweep.ising_ar])

# ── Figure 1: E, |M|, susceptibility, specific heat vs T ─────────────────
fig1, axes1 = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
fig1.suptitle(
    f"Ising model — temperature sweep  |  α* = {alpha_star:.3f}"
    f"\n(fine pass: T ∈ [{T_FINE_MIN:.2f}, {T_FINE_MAX:.2f}], therm = {T_FINE_THERM})",
    fontsize=13, fontweight="bold"
)

BLUE  = "#2E86AB"
RED   = "#E84855"
AMBER = "#F4A261"

panels = [
    (axes1[0, 0], avg_energy,         r"average energy $\langle E \rangle$",  "Energy vs T"),
    (axes1[0, 1], avg_mag,            r"average $|M|$",                        "|Magnetization| vs T"),
    (axes1[1, 0], sweep.suscept_ar,   r"susceptibility $\chi$",                "Susceptibility vs T"),
    (axes1[1, 1], sweep.spec_heat_ar, r"specific heat $C$",                    "Specific Heat vs T"),
]

for ax, data, ylabel, title in panels:
    ax.plot(T_global, data, "o-", color=BLUE, lw=1.8, ms=3)
    ax.axvline(T_crit, color=RED,   linestyle="--", lw=1.8,
               label=f"$T_{{crit}}$ = {T_crit:.2f}  (peak C)")
    ax.axvline(T_best, color=AMBER, linestyle=":",  lw=1.8,
               label=f"$T_{{best}}$ = {T_best:.2f}  (peak r)")
    ax.set_xlabel("global temperature  T", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9, framealpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

plt.savefig("temperature_sweep.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: temperature_sweep.png")

# Temp Sweep Analysis
# T_crit : sharp peak in specific heat (C)
# Brain is often near-critical but slightly supercritical (T_best > T_crit)
# E vs T  : increases with T
# M vs T  : high at low T
# χ vs T  : peaks near phase transition (measures fluctuations)
# C vs T  : peak = T_crit

# ── Figure 2: correlation vs T ───────────────────────────────────────────
fig_corr, ax_corr = plt.subplots(figsize=(7, 4), constrained_layout=True)
ax_corr.plot(T_global, sweep.corr_ar_total, color=BLUE,    lw=2,   label="avg FC")
ax_corr.plot(T_global, sweep.corr_ar_1,     color="green", lw=1,   label="FC1",  alpha=0.7)
ax_corr.plot(T_global, sweep.corr_ar_2,     color="purple",lw=1,   label="FC2",  alpha=0.7)
ax_corr.plot(T_global, sweep.corr_ar_3,     color="orange",lw=1,   label="FC3",  alpha=0.7)
ax_corr.axvline(T_crit, color=RED,   linestyle="--", lw=1.5, label=f"T_crit = {T_crit:.2f}")
ax_corr.axvline(T_best, color=AMBER, linestyle=":",  lw=1.5, label=f"T_best = {T_best:.2f}")
ax_corr.set_xlabel("global temperature  T", fontsize=11)
ax_corr.set_ylabel("Pearson r  (sim FC vs emp FC)", fontsize=11)
ax_corr.set_title("Correlation vs Temperature  (fine pass)", fontsize=12)
ax_corr.legend(fontsize=9, framealpha=0.3)
ax_corr.spines[["top", "right"]].set_visible(False)
plt.savefig("correlation_vs_T.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: correlation_vs_T.png")


# ═══════════════════════════════════════════════════════════════════════════
# Step 3: Matrix analysis/comparison at best-match temperature
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("STEP 3 : MATRIX COMPARISON  (T_best)")
print("=" * 65)

best_gd = sweep.best_ising
sim_FC  = best_gd.FC.copy()
Jij_mat = best_gd.Jij.copy()
np.fill_diagonal(sim_FC, 0)

sim_FC_vec = upper_tri_vec(sim_FC)
r_best     = pearsonr(sim_FC_vec, rho_emp_vec)[0]
dist_best  = np.linalg.norm(sim_FC_vec - rho_emp_vec)
diss_best  = 1.0 - r_best

print(f"r              = {r_best:.4f}")
print(f"eucl. distance = {dist_best:.4f}")
print(f"dissimilarity  = {diss_best:.4f}")

# Equilibration check on the best-match simulation specifically
ok, drift = check_equilibrated(best_gd.ising.energy_series)
if ok:
    print(f"✓  best_ising equilibrated  (drift = {drift:.4f})")
else:
    print(f"⚠  best_ising NOT equilibrated  (drift = {drift:.4f})"
          " — r value may be unreliable.")

norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
fig3.suptitle(
    f"Matrix comparison  |  T_best = {T_best:.2f}  |  α* = {alpha_star:.2f}"
    f"  |  r = {r_best:.4f}",
    fontsize=13, fontweight="bold"
)

mat_panels = [
    (sim_FC,  f"Simulated FC\n(T={T_best:.2f}, α={alpha_star:.2f})"),
    (rho_emp, "Empirical FC  $\\rho_{emp}$"),
    (Jij_mat, "Structural connectivity  $J_{ij}$"),
]

for ax, (mat, title) in zip(axes3, mat_panels):
    im = ax.matshow(mat, cmap="coolwarm", norm=norm)
    ax.set_title(title, fontsize=11, pad=12)
    ax.set_xlabel("region", fontsize=9)
    ax.set_ylabel("region", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.savefig("matrix_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: matrix_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
# Step 4: Null distribution — shuffled-Jij baseline
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print(f"STEP 4 : NULL DISTRIBUTION  (N={N_NULL})")
print(f"         T* = {T_star:.3f}  |  alpha* = {alpha_star:.3f}")
print(f"         NULL_STEPS = {NULL_STEPS}  |  NULL_THERM = {NULL_THERM}"
      f"  |  NULL_RUNS = {NULL_RUNS}")
print("=" * 65)


def shuffle_jij(J):
    """Symmetric off-diagonal shuffle — preserves edge weights, destroys spatial structure."""
    J_null = J.copy()
    idx    = np.triu_indices(J.shape[0], k=1)
    vals   = J_null[idx].copy()
    np.random.shuffle(vals)
    J_null[idx]            = vals
    J_null[idx[1], idx[0]] = vals
    return J_null


def run_ising_avg(J, T_global, alpha, n_runs=NULL_RUNS):
    """
    Average FC over n_runs independent Ising simulations.
    Uses NULL_STEPS / NULL_THERM — now matched to the real-run budget
    so the null comparison is fair.
    """
    mu_loc        = np.mean(J, axis=0)
    mu_loc        = mu_loc / mu_loc.max()
    mu_loc_sorted = mu_loc[utils.cross_sort(mu_loc)]
    temp_arr      = T_global * (mu_loc_sorted ** alpha)

    fc_sum = np.zeros((N, N))
    for _ in range(n_runs):
        sim = I.Jij_sorted_ising(temp_arr, Jij=J)
        sim.simulate(NULL_STEPS, NULL_THERM)
        sim.generate_FC(partial=False)
        fc_sum += sim.functional_connectivity

    rho = fc_sum / n_runs
    np.fill_diagonal(rho, 0)
    return rho


null_dist = []
null_diss = []

for i in range(N_NULL):
    J_null   = shuffle_jij(J_real)
    rho_null = run_ising_avg(J_null, T_star, alpha_star)

    vec_null = upper_tri_vec(rho_null)
    r_null   = pearsonr(vec_null, rho_emp_vec)[0]

    null_dist.append(np.linalg.norm(vec_null - rho_emp_vec))
    null_diss.append(1.0 - r_null)

    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{N_NULL}  dist={null_dist[-1]:.4f}  "
              f"diss={null_diss[-1]:.4f}  r={r_null:.4f}")

null_dist = np.array(null_dist)
null_diss = np.array(null_diss)

# p-value: proportion of null samples with dist/diss <= real value
# (smaller dist/diss = better fit; p = how often null matches or beats real)
p_dist = np.mean(null_dist <= dist_best)
p_diss = np.mean(null_diss <= diss_best)

print(f"\nreal dist  = {dist_best:.4f} | null mean = {null_dist.mean():.4f} | p = {p_dist:.4f}")
print(f"real diss  = {diss_best:.4f} | null mean = {null_diss.mean():.4f} | p = {p_diss:.4f}")


# ── Figure 4: null distribution plots ────────────────────────────────────
NULL_COLOR = "#5BA4CF"
REAL_COLOR = "#C0392B"


def plot_null(ax, null_vals, real_val, p_val, xlabel, title):
    counts, edges = np.histogram(null_vals, bins=BINS)
    widths = np.diff(edges)
    for c, left, w in zip(counts, edges[:-1], widths):
        in_tail = (left + w) <= real_val
        ax.bar(left, c, width=w, align="edge",
               color=REAL_COLOR if in_tail else NULL_COLOR,
               alpha=0.40 if in_tail else 0.80,
               edgecolor="white", linewidth=0.5)
    ax.axvline(real_val, color=REAL_COLOR, linestyle="--", lw=2.2,
               label=f"real $J_{{ij}}$  ({real_val:.4f})")
    ax.text(0.97, 0.95, f"p = {p_val:.4f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=12, color=REAL_COLOR, fontweight="medium")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("count", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9, framealpha=0.3)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.spines[["top", "right"]].set_visible(False)


fig4, axes4 = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
fig4.suptitle(
    f"Ising null distribution  |  T* = {T_star:.2f}  |  "
    f"α* = {alpha_star:.2f}  |  r = {r_best:.4f}"
    f"\n(N_NULL={N_NULL}, NULL_RUNS={NULL_RUNS}, NULL_STEPS={NULL_STEPS})",
    fontsize=13, fontweight="bold"
)

plot_null(
    axes4[0], null_dist, dist_best, p_dist,
    xlabel=r"euclidean distance  $||\rho_{sim} - \rho_{emp}||$",
    title="null distribution — euclidean distance"
)
plot_null(
    axes4[1], null_diss, diss_best, p_diss,
    xlabel="dissimilarity  (1 − r)",
    title="null distribution — dissimilarity"
)

plt.savefig("ising_null_distributions.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: ising_null_distributions.png")

# Null distribution interpretation:
# Tests whether real Jij structure matters more than random wiring.
# Small p-value → brain connectivity is statistically significant.
# Red dashed line = real Jij; bars = shuffled (null) Jij.
# p < 0.05 → spatial structure of Jij drives the FC match.


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"T*              = {T_star:.4f}")
print(f"alpha*          = {alpha_star:.4f}")
print(f"T_crit          = {T_crit:.4f}  (peak specific heat, fine pass)")
print(f"T_best          = {T_best:.4f}  (peak Pearson r,    fine pass)")
print(f"best r          = {r_best:.4f}")
print(f"eucl. distance  = {dist_best:.4f}")
print(f"dissimilarity   = {diss_best:.4f}")
print(f"p (dist)        = {p_dist:.4f}")
print(f"p (diss)        = {p_diss:.4f}")
print(f"\nParam-anneal restarts : {N_RESTARTS}")
print(f"Fine-pass therm       : {T_FINE_THERM}  (coarse: {ANNEAL_THERM})")
print(f"Null budget           : steps={NULL_STEPS}, therm={NULL_THERM}, "
      f"runs={NULL_RUNS}, N={N_NULL}")
print("\nOutput files:")
for f in ["param_anneal_error.png", "temperature_sweep.png",
          "correlation_vs_T.png", "matrix_comparison.png",
          "ising_null_distributions.png"]:
    print(f"  {f}")