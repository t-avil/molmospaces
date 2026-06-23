"""Chart 2 (simple) — success vs base-perturbation radius. Red/blue/purple."""
import json, re, math, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from poster_stats import wilson

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results/molmobot_sweep/charts_v4"
MBPI0 = REPO / "results/molmobot_sweep/raw_runs/mbpi0_perturbed/mbpi0_mobile_results.jsonl"
BENCH = REPO / "curate_out_full/FrankaPickDroidMiniBench_origpose/benchmark.json"

RED, BLUE, PURPLE = "#d62728", "#1f77b4", "#9467bd"  # MolmoBot, MolmoBot-Pi0, pi0.5

# reconstruct mbpi0 radius (verified) for all 347
bm = json.load(open(BENCH))
seen, key2ep = defaultdict(int), {}
for e in bm:
    h = int(re.search(r'house_(\d+)', e['source']['h5_file']).group(1)); tk = e['source']['traj_key']
    key2ep[(h, tk)] = seen[h]; seen[h] += 1
def radius(h, ep):
    rs = np.random.RandomState(hash((0, h, ep)) & 0xFFFFFFFF)
    return math.hypot(rs.uniform(-0.5, 0.5), rs.uniform(-0.5, 0.5))
res = [json.loads(l) for l in open(MBPI0) if l.strip()]
rad = np.array([radius(r['house_index'], key2ep[(r['house_index'], r['traj_key'])]) for r in res])
suc = np.array([1.0 if r['success'] else 0.0 for r in res])

# 5 equal-count bins
order = np.argsort(rad)
bx, by, ber = [], [], []
for ch in np.array_split(order, 5):
    k, n = int(suc[ch].sum()), len(ch)
    p, lo, hi = wilson(k, n)
    bx.append(rad[ch].mean() * 1); by.append(p * 100); ber.append((p - lo) * 100)

fig, ax = plt.subplots(figsize=(8, 5.5), dpi=200)
# MolmoBot (red) and pi0.5 (purple): aggregate horizontal lines
ax.axhline(280 / 347 * 100, color=RED, lw=2.2, label="MolmoBot (81%)")
ax.axhline(48 / 347 * 100, color=PURPLE, lw=2.2, label="pi0.5 (14%)")
# MolmoBot-Pi0 (blue): radius-resolved
ax.errorbar(bx, by, yerr=ber, fmt="o-", color=BLUE, lw=2.2, ms=8, capsize=4,
            label="MolmoBot-Pi0 (43%)")

ax.set_xlabel("Base-perturbation radius (m)")
ax.set_ylabel("Pick success rate (%)")
ax.set_xlim(0, 0.7); ax.set_ylim(0, 100)
ax.legend(frameon=False)
ax.grid(True, color="#eee")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "chart2_success_vs_radius.png", facecolor="white")
print("wrote", OUT / "chart2_success_vs_radius.png")
