"""Generate poster figures for the MolmoBot static-vs-mobile study.

Joins results/molmobot_sweep/comparison.jsonl (per-episode static/mobile success)
with the 347-episode curated benchmark (per-episode base perturbation + object).
Writes 4 PNGs to results/molmobot_sweep/charts/.
"""
from __future__ import annotations
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
COMP = REPO / "results/molmobot_sweep/comparison.jsonl"
BENCH = REPO / "curate_out_full/FrankaPickDroidMiniBench_mobile_v1/benchmark.json"
OUT = REPO / "results/molmobot_sweep/charts"
OUT.mkdir(parents=True, exist_ok=True)

STATIC_C, MOBILE_C = "#2a9d8f", "#e76f51"   # teal / coral
plt.rcParams.update({"font.size": 13, "axes.titlesize": 16, "axes.titleweight": "bold",
                     "figure.dpi": 160, "savefig.bbox": "tight"})

comp = [json.loads(l) for l in COMP.read_text().splitlines() if l.strip()]
N = len(comp)
s_ok = sum(1 for r in comp if r["static_success"])
m_ok = sum(1 for r in comp if r["mobile_success"])
both = sum(1 for r in comp if r["static_success"] and r["mobile_success"])
s_only = sum(1 for r in comp if r["static_success"] and not r["mobile_success"])
m_only = sum(1 for r in comp if not r["static_success"] and r["mobile_success"])
neither = sum(1 for r in comp if not r["static_success"] and not r["mobile_success"])

# join with benchmark for perturbation + object
bench = json.load(BENCH.open())
by_key = {(int(e["house_index"]), str(e["source"]["traj_key"])): e for e in bench}

# ---- Chart 1: headline success-rate bar -----------------------------------
fig, ax = plt.subplots(figsize=(6.5, 5))
rates = [100 * s_ok / N, 100 * m_ok / N]
bars = ax.bar(["Static\n(fixed DROID franka)", "Mobile\n(point→nav→MolmoBot)"], rates,
              color=[STATIC_C, MOBILE_C], width=0.6)
for b, r, n in zip(bars, rates, [s_ok, m_ok]):
    ax.text(b.get_x() + b.get_width() / 2, r + 1.2, f"{r:.1f}%\n({n}/{N})",
            ha="center", va="bottom", fontsize=13, fontweight="bold")
ax.annotate("", xy=(1, rates[1] + 0.5), xytext=(1, rates[0]),
            arrowprops=dict(arrowstyle="<->", color="#555", lw=1.5))
ax.text(1.08, (rates[0] + rates[1]) / 2, f"−{rates[0]-rates[1]:.1f} pts",
        color="#555", fontsize=12, va="center")
ax.set_ylabel("Grasp success rate (%)")
ax.set_ylim(0, 80)
ax.set_title("MolmoBot grasp success: static vs mobile franka\n(347 reachable cases)")
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(OUT / "1_success_rate.png"); plt.close(fig)

# ---- Chart 2: experiment funnel -------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 5))
stages = ["FrankaPickDroidMiniBench\n(all episodes)", "Mobile-reachable\n(scripted planner)",
          "Static MolmoBot\nsolved", "Mobile MolmoBot\nsolved"]
vals = [1000, 347, s_ok, m_ok]
colors = ["#264653", "#287271", STATIC_C, MOBILE_C]
y = list(range(len(vals)))[::-1]
for yi, v, c, lbl in zip(y, vals, colors, stages):
    ax.barh(yi, v, color=c, height=0.65)
    ax.text(v + 12, yi, f"{v}", va="center", fontsize=13, fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels(stages)
ax.set_xlabel("Episodes")
ax.set_xlim(0, 1100)
ax.set_title("From benchmark to grasps: attrition funnel")
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(OUT / "2_funnel.png"); plt.close(fig)

# ---- Chart 3: static x mobile agreement (2x2 heatmap) ---------------------
fig, ax = plt.subplots(figsize=(6, 5.2))
grid = [[both, s_only], [m_only, neither]]   # rows: mobile pass/fail ; cols: static pass/fail
im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=max(both, s_only, m_only, neither))
ax.set_xticks([0, 1], ["Static\nsuccess", "Static\nfail"])
ax.set_yticks([0, 1], ["Mobile\nsuccess", "Mobile\nfail"])
labels = [["both succeed", "mobile recovers\n(static fails)"],
          ["mobile-only\nsuccess", "both fail"]]
for i in range(2):
    for j in range(2):
        v = grid[i][j]
        ax.text(j, i - 0.12, f"{v}", ha="center", va="center", fontsize=22, fontweight="bold",
                color="white" if v > max(both, neither) * 0.6 else "#222")
        ax.text(j, i + 0.22, labels[i][j], ha="center", va="center", fontsize=10,
                color="white" if v > max(both, neither) * 0.6 else "#444")
ax.set_title("Where mobile & static agree\n(per-episode, n=347)")
fig.colorbar(im, fraction=0.046, pad=0.04, label="episodes")
fig.savefig(OUT / "3_agreement.png"); plt.close(fig)

# ---- Chart 4: mobile success vs base-perturbation magnitude ---------------
# translation offset the mobile base had to navigate back from
edges = [0.0, 0.10, 0.15, 0.20, 0.30, 1.0]
bins = defaultdict(lambda: [0, 0])  # bin_idx -> [mobile_ok, total]
for r in comp:
    e = by_key.get((int(r["house_index"]), str(r["traj_key"])))
    if not e:
        continue
    p = e["task"].get("mobile_franka_perturbation", {})
    dist = math.hypot(p.get("dx", 0.0), p.get("dy", 0.0))
    bi = next((k for k in range(len(edges) - 1) if edges[k] <= dist < edges[k + 1]), len(edges) - 2)
    bins[bi][1] += 1
    bins[bi][0] += 1 if r["mobile_success"] else 0
xs = list(range(len(edges) - 1))
rate = [100 * bins[k][0] / bins[k][1] if bins[k][1] else 0 for k in xs]
ns = [bins[k][1] for k in xs]
labels = [f"{edges[k]:.2f}–{edges[k+1]:.2f}" if edges[k+1] < 1 else f"≥{edges[k]:.2f}" for k in xs]
fig, ax = plt.subplots(figsize=(7.5, 5))
bars = ax.bar(xs, rate, color=MOBILE_C, width=0.7)
for b, r, n in zip(bars, rate, ns):
    ax.text(b.get_x() + b.get_width() / 2, r + 1, f"{r:.0f}%\nn={n}", ha="center", va="bottom", fontsize=11)
ax.axhline(100 * m_ok / N, ls="--", color="#888", lw=1)
ax.text(len(xs) - 0.5, 100 * m_ok / N + 1, f"overall {100*m_ok/N:.1f}%", color="#666", ha="right", fontsize=10)
ax.set_xticks(xs, labels)
ax.set_xlabel("Base perturbation the mobile robot navigated back from  (m)")
ax.set_ylabel("Mobile grasp success rate (%)")
ax.set_ylim(0, max(rate) + 14)
ax.set_title("Mobile grasp success vs. base-perturbation distance")
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(OUT / "4_perturbation.png"); plt.close(fig)

print("wrote 4 charts to", OUT)
print(f"summary: static {s_ok}/{N}={100*s_ok/N:.1f}%  mobile {m_ok}/{N}={100*m_ok/N:.1f}%")
print(f"2x2: both={both} static_only={s_only} mobile_only={m_only} neither={neither}")
print("perturbation bins (rate%, n):", list(zip([round(x,1) for x in rate], ns)))
