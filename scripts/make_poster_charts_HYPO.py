"""HYPOTHETICAL poster charts — NOT real data.

Purpose: show what the finished poster looks like once every cell reaches the
full n=347. All numbers below are HAND-SET illustrative values, not measured.
Writes to results/molmobot_sweep/charts_hypo/ with HYPO_ filenames so nothing
real (charts_v4) is overwritten. A small "ILLUSTRATIVE — n set, not measured"
tag sits in the corner of each chart.

Illustrative targets (n=347 in every cell):
  MolmoBot : static 271/347 (78%), mobile@static 232/347 (67%), perturbed 208/347 (60%)
  pi0.5    : static 101/347 (29%), mobile@static  76/347 (22%), perturbed  69/347 (20%)
  scripted : 100% feasibility ceiling

Chart 2 (radius): bin counts sum to 347 per module. MolmoBot keeps its declining
logistic shape. pi0.5 is modeled as an EXPONENTIAL decay in success with radius
(~31% near zero offset -> ~12% at the largest offset), with per-bin Wilson CIs
and a propagated fit band. The exponential is a modeling assumption motivated by
the general OOD-falloff finding in the pi0.5 work, NOT a literal curve quoted
from the paper (the paper reports log-linear scaling vs #training environments).
"""
from __future__ import annotations
import textwrap, sys, re, glob, math
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from poster_stats import wilson, newcombe_diff, mcnemar, logistic_fit, logistic_band

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results/molmobot_sweep/charts_hypo"
OUT.mkdir(parents=True, exist_ok=True)

C_MOLMO, C_PI, C_SCRIPT = "#e76f51", "#9b5de5", "#264653"
GRID = "#dcdcdc"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12.5,
    "axes.titlesize": 17, "axes.titleweight": "bold", "axes.labelsize": 13.5,
    "axes.edgecolor": "#333", "axes.linewidth": 1.0,
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "savefig.facecolor": "white", "figure.facecolor": "white", "axes.facecolor": "white",
    "mathtext.default": "regular",
})

N = 347
# (success, n) — illustrative, every cell at full n
MB = [(271, N), (232, N), (208, N)]
PI = [(101, N), (76, N), (69, N)]


def tag(fig):
    fig.text(0.992, 0.012, "ILLUSTRATIVE — n set, not measured", ha="right", va="bottom",
             fontsize=8, color="#b0b0b0", style="italic")


def star(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "n.s."


def radius_pairs(logglob):
    """Real measured (radius, success) pairs from MolmoBot perturbed-run logs."""
    PERT = re.compile(r"perturbed by \(dx=([-+0-9.]+), dy=([-+0-9.]+)")
    HOUSE = re.compile(r"house (\d+) episode (\d+)")
    DONE = re.compile(r"house (\d+) episode (\d+).*completed with success=(True|False)")
    rows, done = [], {}
    for lp in sorted(glob.glob(logglob)):
        pend = None
        for line in open(lp, errors="ignore"):
            m = PERT.search(line)
            if m:
                pend = (float(m.group(1)), float(m.group(2))); continue
            dd = DONE.search(line)
            if dd:
                done[(lp, int(dd.group(1)), int(dd.group(2)))] = (dd.group(3) == "True")
            h = HOUSE.search(line)
            if h and pend is not None:
                rows.append((lp, int(h.group(1)), int(h.group(2)), pend)); pend = None
    return [(math.hypot(p[0], p[1]), done[(lp, hh, ee)]) for (lp, hh, ee, p) in rows if (lp, hh, ee) in done]


# ===========================================================================
# Chart 1: grouped bars, condition x module
# ===========================================================================
CONDS = ["Static base\n(fixed-base protocol)",
         "Mobile @ static pose\n(floating base + contact)",
         "Mobile + perturbed\n(+ viewpoint / approach)"]
fig, ax = plt.subplots(figsize=(12.5, 6.8))
nC = 3
gw = 0.82; bw = gw / 3
xb = np.arange(nC)


def bars(offset, series, color, label, ceiling=False):
    for ci in range(nC):
        x = xb[ci] - gw / 2 + bw * (offset + 0.5)
        if ceiling:
            ax.bar(x, 100, width=bw * 0.9, color=color, edgecolor="white", zorder=3,
                   label=label if ci == 0 else None)
            ax.text(x, 101.5, "100", ha="center", va="bottom", fontsize=9, color=color, fontweight="bold")
            ax.text(x, 50, f"{N}/{N}", ha="center", va="center", fontsize=8, color="white", fontweight="bold", rotation=90)
            continue
        k, n = series[ci]
        p = k / n
        ax.bar(x, p * 100, width=bw * 0.9, color=color, edgecolor="white", zorder=3,
               label=label if ci == 0 else None)
        ax.text(x, p * 100 + 1.6, f"{p*100:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.text(x, max(3.5, p * 100 - 7), f"{k}/{n}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")


bars(0, MB, C_MOLMO, "MolmoBot (served VLA)")
bars(1, PI, C_PI, "pi0.5 (pi05_droid, served)")
bars(2, None, C_SCRIPT, "Scripted IK (oracle ceiling)", ceiling=True)
ax.set_xticks(xb); ax.set_xticklabels(CONDS, fontsize=11.5)
ax.set_ylabel("Pick success rate (%)"); ax.set_ylim(0, 116); ax.set_yticks(range(0, 101, 20))
ax.set_title("Pick success by condition and module")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=11)
ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)

drop = (MB[0][0] / N - MB[2][0] / N) * 100
panel = (f"n=347 per cell; matched seeds, paired per scene. "
         f"MolmoBot drops about {drop:.0f} points from fixed base to perturbed mobile, "
         f"and pi0.5 traces the same downhill shape one band lower. "
         f"Scripted IK finishes every scene in every condition.")
fig.text(0.5, -0.06, textwrap.fill(panel, 118), ha="center", va="top",
         fontsize=9.3, family="DejaVu Sans Mono", color="#333")
tag(fig)
fig.savefig(OUT / "HYPO_chart1_success_by_condition_module.png"); plt.close(fig)

# ===========================================================================
# Chart 2: success vs radius (n sums to 347 per module)
# ===========================================================================
edges = [0.0, 0.15, 0.30, 0.45, 0.75]
mids = np.array([(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)])

# MolmoBot: the ORIGINAL measured trajectory from the real perturbed-run logs.
mb_pairs = radius_pairs("/tmp/cond3_perturbed/r*.log")


def binned(pairs):
    b = defaultdict(lambda: [0, 0])
    for r, s in pairs:
        for i in range(4):
            if edges[i] <= r < edges[i + 1]:
                b[i][1] += 1; b[i][0] += int(s); break
    return b


mbb = binned(mb_pairs)
mb_kn = [(mbb[i][0], mbb[i][1]) for i in range(4)]  # real per-bin (k, n)

# pi0.5: hypothetical exponential decay, bin n sums to 347.
binN = np.array([55, 110, 115, 67])  # sums to 347
assert binN.sum() == N
pi_k = np.array([17, 24, 20, 8])     # sum 69 ; rates ~31,22,17,12 %
assert pi_k.sum() == 69

fig, ax = plt.subplots(figsize=(12.5, 6.8))
ax.plot([0, 0.72], [100, 100], "-", color=C_SCRIPT, lw=2.4, marker="s", ms=7, markevery=[0, 1],
        label="Scripted IK (feasibility ceiling)", zorder=3)

# MolmoBot original measured trajectory: logistic fit + band + per-bin points
rad = np.array([r for r, _ in mb_pairs]); su = np.array([1.0 if s else 0.0 for _, s in mb_pairs])
b0, b1, se1, cov = logistic_fit(rad, su)
xx = np.linspace(0, 0.7, 100); yy, ylo, yhi = logistic_band(b0, b1, cov, xx)
ax.fill_between(xx, ylo * 100, yhi * 100, color=C_MOLMO, alpha=0.13, zorder=1)
ax.plot(xx, yy * 100, "-", color=C_MOLMO, lw=2.6, label="MolmoBot (original measured trajectory)", zorder=5)
mp = [wilson(mb_kn[i][0], mb_kn[i][1])[0] * 100 for i in range(4)]
mlo = [wilson(mb_kn[i][0], mb_kn[i][1])[1] * 100 for i in range(4)]
mhi = [wilson(mb_kn[i][0], mb_kn[i][1])[2] * 100 for i in range(4)]
ax.errorbar(mids, mp, yerr=[[p - l for p, l in zip(mp, mlo)], [h - p for p, h in zip(mp, mhi)]],
            fmt="o", color=C_MOLMO, ms=7, elinewidth=1.5, capsize=4, zorder=6)
for x, p, i in zip(mids, mp, range(4)):
    ax.annotate(f"n={mb_kn[i][1]}", (x, p), textcoords="offset points", xytext=(0, -20), ha="center", fontsize=9, color="#7a2f1c")

# pi0.5 EXPONENTIAL fit + band + per-bin Wilson points
pr = np.array([wilson(pi_k[i], binN[i])[0] for i in range(4)])
# fit log(rate) ~ a + b r  (weighted by n); propagate cov for a band
w = binN.astype(float)
A = np.vstack([np.ones(4), mids]).T
ly = np.log(pr)
W = np.diag(w)
cov_lin = np.linalg.inv(A.T @ W @ A)
beta = cov_lin @ (A.T @ W @ ly)
xxr = np.linspace(0, 0.66, 100)
Xp = np.vstack([np.ones_like(xxr), xxr]).T
ly_hat = Xp @ beta
var = np.einsum("ij,jk,ik->i", Xp, cov_lin, Xp)
se = np.sqrt(np.maximum(var, 0))
ax.fill_between(xxr, np.exp(ly_hat - 1.96 * se) * 100, np.exp(ly_hat + 1.96 * se) * 100,
                color=C_PI, alpha=0.13, zorder=1)
ax.plot(xxr, np.exp(ly_hat) * 100, "-", color=C_PI, lw=2.6,
        label="pi0.5 (exponential fit, 95% band)", zorder=5)
pp = [wilson(pi_k[i], binN[i])[0] * 100 for i in range(4)]
plo = [wilson(pi_k[i], binN[i])[1] * 100 for i in range(4)]
phi = [wilson(pi_k[i], binN[i])[2] * 100 for i in range(4)]
ax.errorbar(mids, pp, yerr=[[p - l for p, l in zip(pp, plo)], [h - p for p, h in zip(pp, phi)]],
            fmt="D", color=C_PI, ms=7, elinewidth=1.5, capsize=4, zorder=6)
for x, p, i in zip(mids, pp, range(4)):
    ax.annotate(f"n={binN[i]}", (x, p), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=9, color="#5a2b87")

ax.set_xlabel(r"Base-perturbation radius  $r=\sqrt{dx^2+dy^2}$  (m, realized single offset)")
ax.set_ylabel("Pick success rate (%)"); ax.set_ylim(-3, 108); ax.set_xlim(-0.02, 0.72); ax.set_yticks(range(0, 101, 20))
ax.set_title("Pick success vs. base-perturbation radius")
ax.legend(loc="center left", frameon=False, fontsize=11)
ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)

caption = ("MolmoBot is the original measured trajectory from the real perturbed run (single random offset "
           "per scene, binned post-hoc, not a controlled sweep; max realized 0.66 m). Scripted holds 100% at "
           "every radius, so MolmoBot's drop is policy failure, not infeasibility. pi0.5 follows an exponential "
           "decay in success with radius (~31% near zero offset to ~12% at the largest), consistent with "
           "smooth OOD falloff; the exponential is a modeling assumption, not a curve quoted from the paper.")
fig.text(0.5, -0.05, textwrap.fill(caption, 108), ha="center", va="top",
         fontsize=9.2, family="DejaVu Sans Mono", color="#333")
tag(fig)
fig.savefig(OUT / "HYPO_chart2_success_vs_radius.png"); plt.close(fig)

# ---- console ---------------------------------------------------------------
print("MB :", [f"{k}/{n}={100*k/n:.1f}%" for k, n in MB])
print("PI :", [f"{k}/{n}={100*k/n:.1f}%" for k, n in PI])
print("MB radius bins (real) k/n:", [f"{mb_kn[i][0]}/{mb_kn[i][1]}" for i in range(4)])
print("PI radius bins k/n:", [f"{pi_k[i]}/{binN[i]}={100*pi_k[i]/binN[i]:.0f}%" for i in range(4)])
print("wrote HYPO charts to", OUT)
