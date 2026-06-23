"""Chart 2 (scientific) — pick success vs base-perturbation radius.

MolmoBot-Pi0 is shown radius-resolved: every one of its 347 perturbation radii is
reconstructed from the reproducible seed (seed=hash((0,house,episode_idx)) ->
RandomState.uniform(-0.5,0.5)) and verified 51/51 against the surviving driver logs.
Equal-count (quantile) bins give uniform per-bin n with Wilson 95% CIs, plus a
logistic fit with band.

Base MolmoBot and pure pi0.5 perturbed runs no longer have per-trajectory radius on
disk (the /tmp out-dirs were cleaned), so only their aggregate perturbed rate is
available; they are drawn as horizontal reference bands (Wilson 95%) spanning the
radius axis and labelled as aggregates. Scripted IK is the 100% feasibility ceiling.
"""
import json, re, math, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from poster_stats import wilson, logistic_fit, logistic_band

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results/molmobot_sweep/charts_v4"
MBPI0 = REPO / "results/molmobot_sweep/raw_runs/mbpi0_perturbed/mbpi0_mobile_results.jsonl"
BENCH = REPO / "curate_out_full/FrankaPickDroidMiniBench_origpose/benchmark.json"

C_MB, C_MBPI0, C_PI, C_SCRIPT = "#e76f51", "#2a9d8f", "#9b5de5", "#264653"

# ---- reconstruct mbpi0 radius for all 347 (verified) -----------------------
bm = json.load(open(BENCH))
def house_of(e): return int(re.search(r'house_(\d+)', e['source']['h5_file']).group(1))
seen, key2ep = defaultdict(int), {}
for e in bm:
    h, tk = house_of(e), e['source']['traj_key']
    key2ep[(h, tk)] = seen[h]; seen[h] += 1
def radius(h, ep):
    rs = np.random.RandomState(hash((0, h, ep)) & 0xFFFFFFFF)
    dx, dy = rs.uniform(-0.5, 0.5), rs.uniform(-0.5, 0.5)
    return math.hypot(dx, dy)
res = [json.loads(l) for l in open(MBPI0) if l.strip()]
pairs = [(radius(r['house_index'], key2ep[(r['house_index'], r['traj_key'])]), bool(r['success']))
         for r in res if (r['house_index'], r['traj_key']) in key2ep]
rad = np.array([p[0] for p in pairs]); suc = np.array([1.0 if p[1] else 0.0 for p in pairs])

# ---- aggregates that survive only as numbers -------------------------------
AGG = {"MolmoBot": (280, 347, C_MB), "pi0.5": (48, 347, C_PI)}

# ---- equal-count (quantile) bins for mbpi0 ---------------------------------
NB = 5
order = np.argsort(rad)
chunks = np.array_split(order, NB)
bx, by, blo, bhi, bn = [], [], [], [], []
for ch in chunks:
    k = int(suc[ch].sum()); n = len(ch)
    p, lo, hi = wilson(k, n)
    bx.append(float(rad[ch].mean())); by.append(p * 100); blo.append(lo * 100); bhi.append(hi * 100); bn.append(n)

# ---- figure ----------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12, "axes.linewidth": 0.9,
    "axes.edgecolor": "#444", "xtick.color": "#444", "ytick.color": "#444",
})
fig, ax = plt.subplots(figsize=(9.6, 6.2), dpi=220)

# scripted ceiling
ax.plot([0, 0.7], [100, 100], color=C_SCRIPT, lw=2.0, zorder=2)
ax.text(0.69, 100.6, "Scripted IK ceiling (100%)", ha="right", va="bottom",
        fontsize=10, color=C_SCRIPT)

# aggregate reference bands (no per-radius structure available)
for name, (k, n, c) in AGG.items():
    p, lo, hi = wilson(k, n); p, lo, hi = p * 100, lo * 100, hi * 100
    ax.axhspan(lo, hi, color=c, alpha=0.10, zorder=1)
    ax.plot([0, 0.7], [p, p], color=c, lw=1.8, ls=(0, (6, 3)), zorder=3)
    ax.text(0.012, p, f"{name} perturbed: {p:.0f}%  (aggregate over all radii, n={n})",
            va="center", ha="left", fontsize=9.5, color=c,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75))

# mbpi0 logistic fit + band, with trend test
b0, b1, se1, cov = logistic_fit(rad, suc)
from math import erf, exp, sqrt
z = b1 / se1; pz = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
or10 = exp(b1 * 0.1)  # odds ratio per +0.1 m
star = "***" if pz < 1e-3 else "**" if pz < 1e-2 else "*" if pz < 5e-2 else "n.s."
xx = np.linspace(rad.min(), rad.max(), 100)
yy, ylo, yhi = logistic_band(b0, b1, cov, xx)
ax.fill_between(xx, ylo * 100, yhi * 100, color=C_MBPI0, alpha=0.13, zorder=3)
ax.plot(xx, yy * 100, color=C_MBPI0, lw=1.6, ls="--", zorder=4)
ax.text(0.355, 7.5, f"MolmoBot-Pi0 radius trend: OR={or10:.2f} per +0.1 m  (p={pz:.2f}, {star})",
        fontsize=9, color="#1d6f63", style="italic")

# mbpi0 binned points (equal-n) with Wilson CIs and n labels
ax.errorbar(bx, by, yerr=[[p - l for p, l in zip(by, blo)], [h - p for p, h in zip(by, bhi)]],
            fmt="o", color=C_MBPI0, ms=8, elinewidth=1.6, capsize=4, capthick=1.6,
            mec="white", mew=1.0, zorder=6)
ax.plot(bx, by, color=C_MBPI0, lw=1.4, alpha=0.5, zorder=5)
for x, y, n in zip(bx, by, bn):
    ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(0, 13),
                ha="center", fontsize=9, color="#1d6f63")

ax.set_xlabel(r"Base-perturbation radius  $r=\sqrt{\Delta x^2+\Delta y^2}$  (m)")
ax.set_ylabel("Pick success rate (%)")
ax.set_xlim(-0.01, 0.70); ax.set_ylim(-3, 107); ax.set_yticks(range(0, 101, 20))
ax.set_title("Grasp success vs. base-perturbation radius (perturbed condition)",
             fontsize=13.5, pad=10)
ax.grid(True, color="#e6e6e6", lw=0.8); ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)

# compact legend (only the radius-resolved series; the rest are labelled inline)
legend = [
    Line2D([0], [0], color=C_MBPI0, marker="o", lw=1.4, ms=8, mec="white",
           label=f"MolmoBot-Pi0  (radius-resolved, 5 equal-n bins, n={len(pairs)})"),
    Line2D([0], [0], color=C_MBPI0, lw=1.6, ls="--", label="MolmoBot-Pi0 logistic fit (95% band)"),
]
ax.legend(handles=legend, loc="upper right", frameon=True, framealpha=0.92,
          edgecolor="#ccc", fontsize=9)

fig.text(0.5, -0.02,
         "Error bars: Wilson 95% CIs. MolmoBot-Pi0 radii reconstructed from the reproducible perturbation "
         "seed and verified 51/51 against driver logs. Per-trajectory radii for MolmoBot and pi0.5 were not "
         "retained, so only their aggregate perturbed rate is shown (horizontal bands).",
         ha="center", va="top", fontsize=8.2, color="#666", wrap=True)
fig.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig(OUT / "chart2_success_vs_radius.png", bbox_inches="tight", facecolor="white")
print("wrote", OUT / "chart2_success_vs_radius.png")
print("mbpi0 bins (r_mean, succ%, n):", [(round(x, 3), round(y), n) for x, y, n in zip(bx, by, bn)])
print(f"overall mbpi0: {int(suc.sum())}/{len(suc)} = {100*suc.mean():.1f}%")
