"""Chart 1 (simple) — pick success by condition and module. Red/blue/purple.

MolmoBot (red) and pi0.5 (purple) have all three cells (FINDINGS table, n=347).
MolmoBot-Pi0 (blue) retained only its perturbed cell (150/347); the static and
mobile@static cells were cleaned from /tmp before commit, so no bar is drawn there.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from poster_stats import wilson

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results/molmobot_sweep/charts_v4"

RED, BLUE, PURPLE = "#d62728", "#1f77b4", "#9467bd"  # MolmoBot, MolmoBot-Pi0, pi0.5
CONDS = ["Static\n(fixed base)", "Mobile @\nstatic pose", "Mobile +\nperturbed"]

# (k, n) per condition; None = not retained
DATA = {
    "MolmoBot":     (RED,    [(324, 347), (311, 347), (280, 347)]),
    "MolmoBot-Pi0": (BLUE,   [None,       None,       (150, 347)]),
    "pi0.5":        (PURPLE, [(72, 347),  (51, 347),  (48, 347)]),
}

fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=200)
x = np.arange(3)
w = 0.26
for i, (name, (color, cells)) in enumerate(DATA.items()):
    off = (i - 1) * w
    for j, cell in enumerate(cells):
        if cell is None:
            continue
        k, n = cell
        p, lo, hi = wilson(k, n)
        p, lo, hi = p * 100, lo * 100, hi * 100
        ax.bar(x[j] + off, p, w, color=color, label=name if j == (2 if name == "MolmoBot-Pi0" else 0) else None)
        ax.errorbar(x[j] + off, p, yerr=[[p - lo], [hi - p]], fmt="none",
                    ecolor="#333", elinewidth=1.2, capsize=3)
        ax.text(x[j] + off, p + 2.5, f"{p:.0f}", ha="center", va="bottom", fontsize=9, color=color)

ax.set_xticks(x); ax.set_xticklabels(CONDS)
ax.set_ylabel("Pick success rate (%)")
ax.set_ylim(0, 100)
ax.legend(frameon=False, loc="upper right")
ax.grid(True, axis="y", color="#eee")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "chart1_success_by_condition_module.png", facecolor="white")
print("wrote", OUT / "chart1_success_by_condition_module.png")
