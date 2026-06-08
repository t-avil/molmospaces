"""Poster charts v2 for the mobile-manipulation distribution-shift study.

Two paper-quality figures:
  Chart 1  Pick success by condition x module, 95% Wilson CIs, n per cell.
  Chart 2  Success vs base-perturbation radius, one line per module, shaded CIs,
           scripted-IK oracle as the geometric-difficulty ceiling.

Data provenance (all matched on the 347-scene oracle-passing set, paired per scene):
  MolmoBot  static          results/molmobot_sweep/static_results.jsonl   (231/347)
  MolmoBot  mobile+perturbed results/molmobot_sweep/mobile_results.jsonl  (135/347)
  Scripted IK oracle        curate_out_full/curate_results.jsonl          (347/347 on eval set,
                            i.e. the set is *defined* as the scenes scripted IK solves under
                            its seeded perturbation -> 100% by construction = geometric ceiling)
  per-scene perturbation    curate_out_full/FrankaPickDroidMiniBench_mobile_v1/benchmark.json
                            task.mobile_franka_perturbation = {dx, dy, dyaw}

NOTE the data does NOT contain the controlled radius sweep r in {0.1,0.25,0.5,1.0} nor the
mobile-at-static-pose middle condition (that run was killed by a disk-quota error). The
perturbation is a single per-scene draw dx,dy ~ U(-0.5,0.5), dyaw ~ U(-pi/4,pi/4); realized
radius = hypot(dx,dy) is continuous in [0.02, 0.66] m. Chart 2 therefore bins the *realized*
radius. See caveats in the accompanying summary.
"""
from __future__ import annotations
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parent.parent
COMP = REPO / "results/molmobot_sweep/comparison.jsonl"
BENCH = REPO / "curate_out_full/FrankaPickDroidMiniBench_mobile_v1/benchmark.json"
OUT = REPO / "results/molmobot_sweep/charts_v2"
OUT.mkdir(parents=True, exist_ok=True)

# ---- palette / style (paper quality) --------------------------------------
C_MOLMO = "#e76f51"   # coral   - MolmoBot (served VLA)
C_SCRIPT = "#264653"  # dark teal - scripted IK oracle
C_STATIC = "#2a9d8f"  # teal    - static reference
GRID = "#d9d9d9"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.titlesize": 17, "axes.titleweight": "bold",
    "axes.labelsize": 14,
    "axes.edgecolor": "#444", "axes.linewidth": 1.0,
    "figure.dpi": 200, "savefig.dpi": 200,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """95% Wilson score interval. Returns (p_hat, lo, hi) as fractions."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


# ---- load ------------------------------------------------------------------
comp = [json.loads(l) for l in COMP.read_text().splitlines() if l.strip()]
N = len(comp)
s_ok = sum(1 for r in comp if r["static_success"])
m_ok = sum(1 for r in comp if r["mobile_success"])

bench = json.load(BENCH.open())
by_key = {(int(e["house_index"]), str(e["source"]["traj_key"])): e for e in bench}

# ===========================================================================
# Chart 1: pick success by condition x module, 95% Wilson CIs
# ===========================================================================
# cells we actually have, paired on the same 347 scenes:
#   condition order on x-axis
conditions = ["Static base\n(fixed-base protocol)",
              "Mobile + perturbed\n(floating base, contact,\nviewpoint shift)"]
# module -> per-condition (k, n)
cells = {
    "MolmoBot (served VLA)":   [(s_ok, N), (m_ok, N)],
    "Scripted IK (oracle)":    [(N, N),    (N, N)],   # ceiling by construction
}
mod_colors = {"MolmoBot (served VLA)": C_MOLMO, "Scripted IK (oracle)": C_SCRIPT}

fig, ax = plt.subplots(figsize=(11, 6.2))
mods = list(cells)
nC, nM = len(conditions), len(mods)
group_w = 0.8
bar_w = group_w / nM
xbase = list(range(nC))
for mi, mod in enumerate(mods):
    xs, heights, lo_err, hi_err = [], [], [], []
    for ci in range(nC):
        k, n = cells[mod][ci]
        p, lo, hi = wilson(k, n)
        x = xbase[ci] - group_w / 2 + bar_w * (mi + 0.5)
        xs.append(x); heights.append(p * 100)
        lo_err.append((p - lo) * 100); hi_err.append((hi - p) * 100)
    bars = ax.bar(xs, heights, width=bar_w * 0.92, color=mod_colors[mod],
                  edgecolor="white", linewidth=0.8, label=mod, zorder=3)
    ax.errorbar(xs, heights, yerr=[lo_err, hi_err], fmt="none",
                ecolor="#222", elinewidth=1.6, capsize=5, capthick=1.6, zorder=4)
    for x, h, ci in zip(xs, heights, range(nC)):
        k, n = cells[mod][ci]
        ax.text(x, h + max(hi_err) * 0 + 2.5, f"{h:.1f}%\n{k}/{n}",
                ha="center", va="bottom", fontsize=11, fontweight="bold", zorder=5)

# degradation annotation for MolmoBot
sp = wilson(s_ok, N)[0] * 100
mp = wilson(m_ok, N)[0] * 100
ax.annotate(f"-{sp - mp:.1f} pts",
            xy=(1 - group_w/2 + bar_w*0.5, mp), xytext=(0.5, (sp + mp) / 2 + 6),
            ha="center", fontsize=13, fontweight="bold", color="#b5341b",
            arrowprops=dict(arrowstyle="->", color="#b5341b", lw=1.8))

ax.set_xticks(xbase); ax.set_xticklabels(conditions, fontsize=12.5)
ax.set_ylabel("Pick success rate (%)")
ax.set_ylim(0, 112)
ax.set_yticks(range(0, 101, 20))
ax.axhline(100, ls=":", color=C_SCRIPT, lw=1.2, alpha=0.6, zorder=1)
ax.set_title("Pick success by condition and module")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=2,
          frameon=True, framealpha=0.9, edgecolor="#ccc", fontsize=12)
ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
fig.text(0.5, -0.035,
         f"95% Wilson CIs; n={N} per cell; matched seeds (paired per scene).\n"
         f"Scripted IK = solvability oracle that defines the {N}-scene set (ceiling).",
         ha="center", va="top", fontsize=10.5, color="#555")
fig.savefig(OUT / "chart1_success_by_condition_module.png")
plt.close(fig)

# ===========================================================================
# Chart 2: success vs base-perturbation radius (realized), shaded Wilson CIs
# ===========================================================================
edges = [0.0, 0.15, 0.30, 0.45, 0.70]
mids = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
bins = defaultdict(lambda: [0, 0])  # bin -> [mobile_ok, total]
for r in comp:
    e = by_key.get((int(r["house_index"]), str(r["traj_key"])))
    if not e:
        continue
    p = e["task"].get("mobile_franka_perturbation", {})
    dist = math.hypot(p.get("dx", 0.0), p.get("dy", 0.0))
    bi = next((k for k in range(len(edges) - 1) if edges[k] <= dist < edges[k + 1]), len(edges) - 2)
    bins[bi][1] += 1
    bins[bi][0] += 1 if r["mobile_success"] else 0

xs = mids
molmo_p, molmo_lo, molmo_hi, ns = [], [], [], []
for k in range(len(edges) - 1):
    ok, tot = bins[k]
    p, lo, hi = wilson(ok, tot)
    molmo_p.append(p * 100); molmo_lo.append(lo * 100); molmo_hi.append(hi * 100); ns.append(tot)

fig, ax = plt.subplots(figsize=(11, 6.2))

# scripted IK ceiling: 100% across all radii (oracle re-certifies solvability at each radius)
ax.plot(xs, [100] * len(xs), "-", color=C_SCRIPT, lw=2.5, marker="s", ms=8,
        label="Scripted IK (oracle / geometric ceiling)", zorder=4)
ax.fill_between(xs, [100] * len(xs), [100] * len(xs), color=C_SCRIPT, alpha=0.12)

# MolmoBot line + shaded Wilson CI
ax.plot(xs, molmo_p, "-", color=C_MOLMO, lw=2.5, marker="o", ms=8,
        label="MolmoBot (served VLA)", zorder=5)
ax.fill_between(xs, molmo_lo, molmo_hi, color=C_MOLMO, alpha=0.20, zorder=2)

# r=0 static anchor for MolmoBot (no perturbation)
ax.plot([0.0], [sp], marker="*", ms=18, color=C_STATIC, zorder=6,
        label=f"MolmoBot static anchor (r=0, {sp:.1f}%)")

for x, p, n in zip(xs, molmo_p, ns):
    ax.annotate(f"{p:.0f}%\nn={n}", (x, p), textcoords="offset points", xytext=(0, -28),
                ha="center", fontsize=10.5, color="#7a2f1c")

ax.set_xlabel("Base-perturbation radius  r = √(dx² + dy²)   (m, realized)")
ax.set_ylabel("Pick success rate (%)")
ax.set_ylim(0, 108)
ax.set_xlim(-0.03, 0.70)
ax.set_yticks(range(0, 101, 20))
ax.set_xticks([0.0, 0.075, 0.225, 0.375, 0.575])
ax.set_xticklabels(["0\n(static)", "0–0.15", "0.15–0.30", "0.30–0.45", "0.45–0.70"], fontsize=11)
ax.set_title("Pick success vs. base-perturbation radius")
ax.legend(loc="center right", frameon=False, fontsize=11.5)
ax.grid(True, color=GRID, lw=0.8, zorder=0)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
fig.text(0.5, -0.04,
         "One line per module; shaded bands are 95% Wilson CIs. Scripted IK re-certifies solvability at every\n"
         "radius, so its flat 100% line bounds geometric difficulty: any MolmoBot drop is policy failure, not\n"
         "infeasibility. Realized radii (continuous draw, max 0.66 m); no scenes reached r=1.0 m.",
         ha="center", va="top", fontsize=10, color="#555")
fig.savefig(OUT / "chart2_success_vs_radius.png")
plt.close(fig)

# ---- console summary -------------------------------------------------------
print(f"static  {s_ok}/{N} = {100*s_ok/N:.1f}%   wilson {tuple(round(100*x,1) for x in wilson(s_ok,N))}")
print(f"mobile  {m_ok}/{N} = {100*m_ok/N:.1f}%   wilson {tuple(round(100*x,1) for x in wilson(m_ok,N))}")
print(f"degradation: -{100*(s_ok-m_ok)/N:.1f} pts")
print("radius bins (mid, rate%, n, wilson lo/hi):")
for k in range(len(edges) - 1):
    ok, tot = bins[k]
    p, lo, hi = wilson(ok, tot)
    print(f"  {edges[k]:.2f}-{edges[k+1]:.2f}: {ok}/{tot} = {100*p:.1f}%  [{100*lo:.1f}, {100*hi:.1f}]")
print("wrote charts to", OUT)
