
# ═══════════════════════════════════════════════════════════════════════════
# IIT/GIM Project
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


# config

SEED          = 1
N             = cf.regions      # (84x84)

ANNEAL_STEPS  = 2000            # ising steps/annealing
ANNEAL_MAXFUN = 500             # annealing group size before reset
ANNEAL_THERM  = 1000            # thermalization steps

T_MIN         = 2           # sweep range
T_MAX         = 10
T_STEPS       = 50             # number of temperature pts

N_NULL        = 100            # null distribution samples
NULL_RUNS     = 5               # ising runs averaged/null sample
NULL_STEPS    = 2000
NULL_THERM    = 1000

BINS          = 30

np.random.seed(SEED)

# data
J_real  = cf.avg_Jij.copy()
rho_emp = cf.avg_FCp.copy()
np.fill_diagonal(rho_emp, 0)

# per-neuron temp multiplier (same formula as param_anneal + temp_sweep)
mu        = np.mean(J_real, axis=0)
mu        = mu / mu.max()               # == utils.normalize_array(ind_avg_Jij)
mu_sorted = mu[utils.cross_sort(mu)]    # sorted high→low, as Jij_sorted_ising expects

# Pulls out just the upper triangle of a matrix as a flat 1D array

def upper_tri_vec(mat):
    idx = np.triu_indices(mat.shape[0], k=1)
    return mat[idx]

rho_emp_vec = upper_tri_vec(rho_emp)

# Step 1: Param_anneal to get T* and alpha * (additionally plots an error graph)


print("=" * 65)
print("STEP 1 : PARAMETER ANNEALING — searching (T*, alpha*)")
print("=" * 65)

optim  = pa.optimize(
    ising      = I.Jij_sorted_ising,
    Jij        = J_real,
    partial    = True,
    multiplier = utils.normalize_array(cf.ind_avg_Jij),
    save       = True
)
result = optim.anneal(
    steps           = ANNEAL_STEPS,
    maxfun          = ANNEAL_MAXFUN,
    emp_FC          = rho_emp,
    therm           = ANNEAL_THERM,
    no_local_search = True,
    show            = False
)

T_star     = result.x[0]
alpha_star = result.x[1]
print(f"\nT*     = {T_star:.4f}")
print(f"alpha* = {alpha_star:.4f}")
print(f"best r = {max(optim.correlate):.4f}")

optim.plot_error(show=False)
plt.savefig("param_anneal_error.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: param_anneal_error.png")

# param_anneal_error.png is the error or correlation vs interation while annealing
# α = 0 → uniform temperature
# α > 0 → region-specific temperatures
# clear valley of low error
# Temperature scales almost linearly with node strength (a* close to 1)

# Step 2: Temp_sweep at fixed alpha* ( plots E, |M|, susceptibility, specific heat vs T AND correlation vs T)

print("\n" + "=" * 65)
print(f"STEP 2 : TEMPERATURE SWEEP  (alpha* = {alpha_star:.4f})")
print("=" * 65)

sweep = ts.simulated_FC_vs_T_global(
    min_temp   = T_MIN,
    max_temp   = T_MAX,
    temp_step  = T_STEPS,
    alpha      = alpha_star,
    Jij        = J_real,
    ising      = I.Jij_sorted_ising,
    multiplier = utils.normalize_array(cf.ind_avg_Jij),
    save       = True
)
sweep.simulate(
    steps          = ANNEAL_STEPS,
    thermalization = ANNEAL_THERM,
    partial        = True,
    text           = True
)

T_crit    = sweep.crit_temp
T_best    = sweep.best_temp
best_corr = sweep.best_corr
T_global  = sweep.T_global

print(f"\nCritical temperature (peak spec. heat) : {T_crit:.4f}")
print(f"Best-match temperature (peak r)        : {T_best:.4f}")
print(f"Best Pearson r                         : {best_corr:.4f}")

# ── collect per-temperature observables ─────────────────────────────────
# sweep.ising_ar  = list of get_data objects
# get_data.ising  = the underlying Ising simulation object
# Ising.energy_series / .mag_series confirmed in ising.py

avg_energy = []
avg_mag    = []

for gd in sweep.ising_ar:
    avg_energy.append(np.mean(gd.ising.energy_series))
    avg_mag.append(np.mean(np.abs(gd.ising.mag_series)))   # |M|

avg_energy = np.array(avg_energy)
avg_mag    = np.array(avg_mag)

# ── Figure 1: E, |M|, susceptibility, specific heat vs T ─────────────────
fig1, axes1 = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
fig1.suptitle(
    f"Ising model — temperature sweep  |  α* = {alpha_star:.3f}",
    fontsize=14, fontweight="bold"
)

BLUE  = "#2E86AB"
RED   = "#E84855"
AMBER = "#F4A261"

panels = [
    (axes1[0, 0], avg_energy,        r"average energy $\langle E \rangle$",  "Energy vs T"),
    (axes1[0, 1], avg_mag,           r"average $|M|$",                        "|Magnetization| vs T"),
    (axes1[1, 0], sweep.suscept_ar,  r"susceptibility $\chi$",                "Susceptibility vs T"),
    (axes1[1, 1], sweep.spec_heat_ar,r"specific heat $C$",                    "Specific Heat vs T"),
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
# Temp_crit : sharp peak in specific heat (c)
# The brain is often found in a “near-critical
# but slightly supercritical” regime (past phase transition T_best> T_crit
# E vs T : inc with T
# M vs T : high at low T
# Susceptibility vs T : peaks near phase transition (measures fluctuations)
# Specific heat vs T : peak = T_crit


# ── Figure 2: correlation vs T (bonus — already in sweep.graph_data) ─────
fig_corr, ax_corr = plt.subplots(figsize=(7, 4), constrained_layout=True)
ax_corr.plot(T_global, sweep.corr_ar_total, color=BLUE,  lw=2, label="avg FC")
ax_corr.plot(T_global, sweep.corr_ar_1,    color="green",lw=1, label="FC1",  alpha=0.7)
ax_corr.plot(T_global, sweep.corr_ar_2,    color="purple",lw=1,label="FC2",  alpha=0.7)
ax_corr.plot(T_global, sweep.corr_ar_3,    color="orange",lw=1,label="FC3",  alpha=0.7)
ax_corr.axvline(T_crit, color=RED,   linestyle="--", lw=1.5, label=f"T_crit = {T_crit:.2f}")
ax_corr.axvline(T_best, color=AMBER, linestyle=":",  lw=1.5, label=f"T_best = {T_best:.2f}")
ax_corr.set_xlabel("global temperature  T", fontsize=11)
ax_corr.set_ylabel("Pearson r  (sim FC vs emp FC)", fontsize=11)
ax_corr.set_title("Correlation vs Temperature", fontsize=12)
ax_corr.legend(fontsize=9, framealpha=0.3)
ax_corr.spines[["top", "right"]].set_visible(False)
plt.savefig("correlation_vs_T.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: correlation_vs_T.png")

# Correlation vs T analysis (Main result) (r)
# Highest peak (T_best) is where FC_sim best ressembles FC_emp
# FCs are consistent




# Step 3: Matrix analysis/comparison at best match temperature


print("\n" + "=" * 65)
print("STEP 3 : MATRIX COMPARISON  (T_best)")
print("=" * 65)

best_gd = sweep.best_ising          # get_data object
sim_FC  = best_gd.FC.copy()         # confirmed attribute name: .FC
Jij_mat = best_gd.Jij.copy()        # confirmed attribute name: .Jij
np.fill_diagonal(sim_FC, 0)

sim_FC_vec = upper_tri_vec(sim_FC)
r_best     = pearsonr(sim_FC_vec, rho_emp_vec)[0]
dist_best  = np.linalg.norm(sim_FC_vec - rho_emp_vec)
diss_best  = 1.0 - r_best

print(f"r              = {r_best:.4f}")
print(f"eucl. distance = {dist_best:.4f}")
print(f"dissimilarity  = {diss_best:.4f}")

norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
fig3.suptitle(
    f"Matrix comparison  |  T_best = {T_best:.2f}  |  α* = {alpha_star:.2f}  |  r = {r_best:.4f}",
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


# Step 4: Null Distribution (Plots distance and dissimilarity)


print("\n" + "=" * 65)
print(f"STEP 4 : NULL DISTRIBUTION  (N={N_NULL})")
print(f"         T* = {T_star:.3f}  |  alpha* = {alpha_star:.3f}")
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
    """Average FC over n_runs independent Ising simulations."""
    mu_loc        = np.mean(J, axis=0)
    mu_loc        = mu_loc / mu_loc.max()
    mu_loc_sorted = mu_loc[utils.cross_sort(mu_loc)]
    temp_arr      = T_global * (mu_loc_sorted ** alpha)

    fc_sum = np.zeros((N, N))
    for _ in range(n_runs):
        sim = I.Jij_sorted_ising(temp_arr, Jij=J)
        sim.simulate(NULL_STEPS, NULL_THERM)
        sim.generate_FC(partial=True)
        fc_sum += sim.functional_connectivity

    rho = fc_sum / n_runs
    np.fill_diagonal(rho, 0)
    return rho

null_dist = []
null_diss = []

for i in range(N_NULL):
    J_null   = shuffle_jij(J_real)
    rho_null = run_ising_avg(J_null, T_star, alpha_star)

    vec_null  = upper_tri_vec(rho_null)
    r_null    = pearsonr(vec_null, rho_emp_vec)[0]

    null_dist.append(np.linalg.norm(vec_null - rho_emp_vec))
    null_diss.append(1.0 - r_null)

    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{N_NULL}  dist={null_dist[-1]:.4f}  "
              f"diss={null_diss[-1]:.4f}  r={r_null:.4f}")

null_dist = np.array(null_dist)
null_diss = np.array(null_diss)

# p-value: proportion of null samples with dist/diss <= real value
# (a smaller dist/diss means a better fit, so p = how often null is as good or better)
p_dist = np.mean(null_dist <= dist_best)
p_diss = np.mean(null_diss <= diss_best)

print(f"\nreal dist  = {dist_best:.4f} | null mean = {null_dist.mean():.4f} | p = {p_dist:.4f}")
print(f"real diss  = {diss_best:.4f} | null mean = {null_diss.mean():.4f} | p = {p_diss:.4f}")


# ── Effect sizes ─────────────────────────────────────────────────────────

def cohens_d(null_vals, real_val):
    """
    Signed Cohen's d: how many SDs the real value sits below the null mean.
    Negative = real is better (smaller dist/diss) than the null centre.
    """
    return (real_val - null_vals.mean()) / null_vals.std(ddof=1)

def cliffs_delta(null_vals, real_val):
    """
    Cliff's delta: proportion of null samples that are GREATER than the
    real value minus proportion that are LESS (range −1 to +1).
    Positive = real value is smaller (better fit) than most of the null.
    """
    greater = np.sum(null_vals > real_val)
    less    = np.sum(null_vals < real_val)
    return (greater - less) / len(null_vals)

def cliffs_magnitude(delta):
    """Conventional magnitude labels for Cliff's delta."""
    a = abs(delta)
    if a < 0.147:  return "negligible"
    if a < 0.330:  return "small"
    if a < 0.474:  return "medium"
    return "large"

def cohens_magnitude(d):
    """Conventional magnitude labels for Cohen's d."""
    a = abs(d)
    if a < 0.2:  return "negligible"
    if a < 0.5:  return "small"
    if a < 0.8:  return "medium"
    return "large"

# ── Compute effect sizes ──────────────────────────────────────────────────
cd_dist   = cohens_d(null_dist, dist_best)
cd_diss   = cohens_d(null_diss, diss_best)
cld_dist  = cliffs_delta(null_dist, dist_best)
cld_diss  = cliffs_delta(null_diss, diss_best)

print(f"\nCohen's d  (dist) = {cd_dist:.4f}  [{cohens_magnitude(cd_dist)}]")
print(f"Cohen's d  (diss) = {cd_diss:.4f}  [{cohens_magnitude(cd_diss)}]")
print(f"Cliff's δ  (dist) = {cld_dist:.4f}  [{cliffs_magnitude(cld_dist)}]")
print(f"Cliff's δ  (diss) = {cld_diss:.4f}  [{cliffs_magnitude(cld_diss)}]")

# ── Figure 4: null distribution plots ────────────────────────────────────
NULL_COLOR = "#5BA4CF"
REAL_COLOR = "#C0392B"

def plot_null(ax, null_vals, real_val, p_val, cd, cld, xlabel, title):
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

    # stats annotation block
    stats_text = (
        f"p = {p_val:.4f}\n"
        f"Cohen's d = {cd:.3f}  [{cohens_magnitude(cd)}]\n"
        f"Cliff's δ = {cld:.3f}  [{cliffs_magnitude(cld)}]"
    )
    ax.text(0.97, 0.95, stats_text,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, color=REAL_COLOR, fontweight="medium",
            linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=REAL_COLOR,
                      alpha=0.6))

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("count", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9, framealpha=0.3)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.spines[["top", "right"]].set_visible(False)

fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
fig4.suptitle(
    f"Ising null distribution  |  T* = {T_star:.2f}  |  "
    f"α* = {alpha_star:.2f}  |  r = {r_best:.4f}",
    fontsize=13, fontweight="bold"
)

plot_null(
    axes4[0], null_dist, dist_best, p_dist, cd_dist, cld_dist,
    xlabel=r"euclidean distance  $||\rho_{sim} - \rho_{emp}||$",
    title="null distribution — euclidean distance"
)
plot_null(
    axes4[1], null_diss, diss_best, p_diss, cd_diss, cld_diss,
    xlabel="dissimilarity  (1 − r)",
    title="null distribution — dissimilarity"
)

plt.savefig("ising_null_distributions.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: ising_null_distributions.png")

# ── SUMMARY ───────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"T*              = {T_star:.4f}")
print(f"alpha*          = {alpha_star:.4f}")
print(f"T_crit          = {T_crit:.4f}  (peak specific heat)")
print(f"T_best          = {T_best:.4f}  (peak Pearson r)")
print(f"best r          = {r_best:.4f}")
print(f"eucl. distance  = {dist_best:.4f}")
print(f"dissimilarity   = {diss_best:.4f}")
print(f"p (dist)        = {p_dist:.4f}")
print(f"p (diss)        = {p_diss:.4f}")
print(f"Cohen's d (dist)= {cd_dist:.4f}  [{cohens_magnitude(cd_dist)}]")
print(f"Cohen's d (diss)= {cd_diss:.4f}  [{cohens_magnitude(cd_diss)}]")
print(f"Cliff's δ (dist)= {cld_dist:.4f}  [{cliffs_magnitude(cld_dist)}]")
print(f"Cliff's δ (diss)= {cld_diss:.4f}  [{cliffs_magnitude(cld_diss)}]")
print("\nOutput files:")
for f in ["param_anneal_error.png", "temperature_sweep.png",
          "correlation_vs_T.png", "matrix_comparison.png",
          "ising_null_distributions.png"]:
    print(f"  {f}")


# Null distributions (stats significance test
# tests whether or not a real Jij is more significant than a random Jij
# we want small p-value that suggests unlikely due to chance
# runs ising on random Jij, compares to FC_emp (correlation(r), dist, and diss are computed),
# repeated 100 times (to get distribution).
# red line is real Jij and bars are random (null)
# Jij (same wires as real Jij but just randomly reconnected) results

# Cohen's d — (real − null_mean) / null_std
#
# Measures how many standard deviations the real Jij sits below the null centre
# Negative value = real is better (smaller distance/dissimilarity) than average null
# Conventional cutoffs: |d| < 0.2 negligible, 0.2–0.5 small, 0.5–0.8 medium, >0.8 large
#
# Cliff's δ — (# null > real − # null < real) / N_null
#
# Non-parametric, so it doesn't assume null is Gaussian (it won't be for small N_NULL)
# Ranges −1 to +1; positive = real value is smaller than most nulls = better fit
# More robust than Cohen's d when the null distribution is skewed

# if p<0.5, structure matters and brain wiring is significant

    # ---------------------------------------------------------------------------
    # Tuesday May 8 2026
    # ---------------------------------------------------------------------------
    #

    # Code for simulation of circadian rhythm
#make a function that looks identical to the image
# above. (make it a simulation and a function
# i can input into desmos) -- i would like to get
# results of Magnetization, susceptibility, specific heat ,
# and energy vs time and then show which part is awake rem or nrem 1,2,or3

