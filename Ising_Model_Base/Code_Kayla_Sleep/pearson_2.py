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
import config_pearson as cfg
import param_anneal as pa
import temp_sweep as ts


# ── config ────────────────────────────────────────────────────────────────
SEED          = 1
N             = cfg.regions

ANNEAL_STEPS  = 10000
ANNEAL_MAXFUN = 500
ANNEAL_THERM  = 5000

N_RESTARTS    = 5 ##  5 annealing runs and picks the best one

T_MIN         = 2
T_MAX         = 25
T_STEPS       = 400

N_NULL        = 100
NULL_RUNS     = 5
NULL_STEPS    = 2000
NULL_THERM    = 1000

BINS          = 30

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


# ── data ──────────────────────────────────────────────────────────────────
J_real  = cfg.avg_Jij.copy()
rho_emp = cfg.avg_FCp.copy()          # partial empirical FC throughout

# remove diagonal for comparison
np.fill_diagonal(rho_emp, 0)

rho_emp_vec = upper_tri_vec(rho_emp)

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
        multiplier = utils.normalize_array(cfg.ind_avg_Jij),
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
plt.savefig("param_anneal_error.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: param_anneal_error.png")


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
    multiplier = utils.normalize_array(cfg.ind_avg_Jij),
    save       = True
)

sweep.simulate(
    steps          = ANNEAL_STEPS,
    thermalization = ANNEAL_THERM,
    partial        = False,
    text           = True
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
avg_energy = np.array([np.mean(gd.ising.energy_series) for gd in sweep.ising_ar])
avg_mag    = np.array([np.mean(np.abs(gd.ising.mag_series)) for gd in sweep.ising_ar])


# ── Figure 1: E, |M|, susceptibility, specific heat vs T ──────────────────
fig1, axes1 = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
fig1.suptitle(
    f"Ising model — temperature sweep  |  alpha = {alpha_star:.3f}",
    fontsize=14,
    fontweight="bold"
)

panels = [
    (axes1[0, 0], avg_energy,         r"average energy $\langle E \rangle$", "Energy vs T"),
    (axes1[0, 1], avg_mag,            r"average $|M|$",                       "|Magnetization| vs T"),
    (axes1[1, 0], sweep.suscept_ar,   r"susceptibility $\chi$",               "Susceptibility vs T"),
    (axes1[1, 1], sweep.spec_heat_ar, r"specific heat $C$",                   "Specific Heat vs T"),
]

for ax, data, ylabel, title in panels:
    ax.plot(T_global, data, "o-", color=BLUE, lw=1.8, ms=3)

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

plt.savefig("temperature_sweep.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
print("Saved: temperature_sweep.png")


# ── Figure 2: correlation vs T ────────────────────────────────────────────
fig_corr, ax_corr = plt.subplots(figsize=(7, 4), constrained_layout=True)

ax_corr.plot(T_global, sweep.corr_ar_total, color=BLUE,    lw=2, label="avg FC")
ax_corr.plot(T_global, sweep.corr_ar_1,     color="green",  lw=1, label="FC1", alpha=0.7)
ax_corr.plot(T_global, sweep.corr_ar_2,     color="purple", lw=1, label="FC2", alpha=0.7)
ax_corr.plot(T_global, sweep.corr_ar_3,     color="orange", lw=1, label="FC3", alpha=0.7)

ax_corr.axvline(T_crit, color=RED, linestyle="--", lw=1.5, label=f"T_crit = {T_crit:.2f}")
ax_corr.axvline(T_best, color=AMBER, linestyle=":", lw=1.5, label=f"T_best = {T_best:.2f}")

ax_corr.set_xlabel("global temperature  T", fontsize=11)
ax_corr.set_ylabel("Pearson r  (sim partial FC vs emp partial FC)", fontsize=11)
ax_corr.set_title("Correlation vs Temperature", fontsize=12)
ax_corr.legend(fontsize=9, framealpha=0.3)
ax_corr.spines[["top", "right"]].set_visible(False)

plt.savefig("correlation_vs_T.png", dpi=150, bbox_inches="tight")
plt.close(fig_corr)
print("Saved: correlation_vs_T.png")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 : MATRIX COMPARISON  (T_best) — partial FC only
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 3 : MATRIX COMPARISON  (T_best, partial FC)")
print("=" * 65)

best_gd = sweep.best_ising
sim_FC  = best_gd.FC.copy()
Jij_mat = best_gd.Jij.copy()

np.fill_diagonal(sim_FC, 0)

sim_FC_vec = upper_tri_vec(sim_FC)

r_best    = pearsonr(sim_FC_vec, rho_emp_vec)[0]
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
    (sim_FC,  f"Simulated partial FC\n(T={T_best:.2f}, alpha={alpha_star:.2f})", fc_norm),
    (rho_emp, "Empirical partial FC", fc_norm),
    (Jij_mat, "Structural connectivity  $J_{ij}$", j_norm),
]

for ax, (mat, title, norm_to_use) in zip(axes3, matrix_panels):
    im = ax.matshow(mat, cmap="RdBu_r", norm=norm_to_use)
    ax.set_title(title, fontsize=11, pad=12)
    ax.set_xlabel("region", fontsize=9)
    ax.set_ylabel("region", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.savefig("matrix_comparison.png", dpi=150, bbox_inches="tight")
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

ax3s.set_xlabel("empirical partial FC", fontsize=11)
ax3s.set_ylabel("simulated partial FC", fontsize=11)
ax3s.set_title(f"Sim vs Emp partial FC  (r = {r_best:.4f})", fontsize=12)
ax3s.spines[["top", "right"]].set_visible(False)

plt.savefig("scatter_sim_vs_emp.png", dpi=150, bbox_inches="tight")
plt.close(fig3s)

print("Saved: matrix_comparison.png, scatter_sim_vs_emp.png")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 : NULL DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"STEP 4 : NULL DISTRIBUTION  (N={N_NULL}, partial=True)")
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


def run_ising_avg(J, T_global_value, alpha, n_runs=NULL_RUNS):
    mu_loc = np.mean(J, axis=0)

    if mu_loc.max() == 0:
        raise ValueError("mu_loc.max() is zero, cannot normalize.")

    mu_loc = mu_loc / mu_loc.max()

    mu_loc_sorted = mu_loc[utils.cross_sort(mu_loc)]

    temp_arr = T_global_value * (mu_loc_sorted ** alpha)

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
    J_null = shuffle_jij(J_real)

    # IMPORTANT:
    # Use T_best here because the real model was evaluated at T_best.
    rho_null = run_ising_avg(J_null, T_best, alpha_star)

    vec_null = upper_tri_vec(rho_null)

    r_null = pearsonr(vec_null, rho_emp_vec)[0]

    null_dist.append(np.linalg.norm(vec_null - rho_emp_vec))
    null_diss.append(1.0 - r_null)

    if (i + 1) % 10 == 0:
        print(
            f"  {i+1}/{N_NULL}  "
            f"dist={null_dist[-1]:.4f}  "
            f"diss={null_diss[-1]:.4f}  "
            f"r={r_null:.4f}"
        )

null_dist = np.array(null_dist)
null_diss = np.array(null_diss)

p_dist = np.mean(null_dist <= dist_best)
p_diss = np.mean(null_diss <= diss_best)

print(f"\nreal dist  = {dist_best:.4f} | null mean = {null_dist.mean():.4f} | p = {p_dist:.4f}")
print(f"real diss  = {diss_best:.4f} | null mean = {null_diss.mean():.4f} | p = {p_diss:.4f}")


# ── effect sizes ──────────────────────────────────────────────────────────
def cohens_d(null_vals, real_val):
    return (real_val - null_vals.mean()) / null_vals.std(ddof=1)


def cliffs_delta(null_vals, real_val):
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

cld_dist = cliffs_delta(null_dist, dist_best)
cld_diss = cliffs_delta(null_diss, diss_best)

print(f"\nCohen's d  (dist) = {cd_dist:.4f}  [{cohens_magnitude(cd_dist)}]")
print(f"Cohen's d  (diss) = {cd_diss:.4f}  [{cohens_magnitude(cd_diss)}]")
print(f"Cliff's δ  (dist) = {cld_dist:.4f}  [{cliffs_magnitude(cld_dist)}]")
print(f"Cliff's δ  (diss) = {cld_diss:.4f}  [{cliffs_magnitude(cld_diss)}]")


# ── Figure 4 ──────────────────────────────────────────────────────────────
NULL_COLOR = "#5BA4CF"
REAL_COLOR = "#C0392B"


def plot_null(ax, null_vals, real_val, p_val, cd, cld, xlabel, title):
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

plt.savefig("ising_null_distributions.png", dpi=150, bbox_inches="tight")
plt.close(fig4)

print("Saved: ising_null_distributions.png")


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
print(f"T_best            = {T_best:.4f}  (peak Pearson r, partial FC)")
print(f"best r            = {r_best:.4f}  (partial FC)")
print(f"eucl. distance    = {dist_best:.4f}")
print(f"dissimilarity     = {diss_best:.4f}")
print(f"p (dist)          = {p_dist:.4f}")
print(f"p (diss)          = {p_diss:.4f}")
print(f"Cohen's d (dist)  = {cd_dist:.4f}  [{cohens_magnitude(cd_dist)}]")
print(f"Cohen's d (diss)  = {cd_diss:.4f}  [{cohens_magnitude(cd_diss)}]")
print(f"Cliff's δ (dist)  = {cld_dist:.4f}  [{cliffs_magnitude(cld_dist)}]")
print(f"Cliff's δ (diss)  = {cld_diss:.4f}  [{cliffs_magnitude(cld_diss)}]")

print("\nOutput files:")
for f in [
    "param_anneal_error.png",
    "temperature_sweep.png",
    "correlation_vs_T.png",
    "matrix_comparison.png",
    "scatter_sim_vs_emp.png",
    "ising_null_distributions.png"
]:
    print(f"  {f}")