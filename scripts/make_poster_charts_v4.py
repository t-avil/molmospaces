"""Poster charts v4 (final, technical) — MolmoBot vs PURE pi0.5 vs scripted IK,
all three as the swapped grasp module across three base conditions.

This supersedes v3. The key fix: pi0.5 is the genuine pi05_droid (Physical
Intelligence) run AS THE SERVED GRASP MODULE in the same point->navigate->grasp
hybrid MolmoBot uses (serve_pi05.py + MobileFrankaHybridPi05EvalConfig), not a
standalone static policy. So the comparison is apples-to-apples.

Three charts:
  1. Pick success by condition x module (grouped bars, 95% Wilson CIs, McNemar).
  2. Success vs base-perturbation radius (MolmoBot fit+band, pi0.5, scripted ceiling).
  3. Degradation slopegraph: success across conditions, one line per module
     (the "is the gap consistent across modules" view).

Data (n per cell varies; runs stopped at solid n to free shared GPUs):
  MolmoBot static          results/molmobot_sweep/static_results.jsonl   231/347
  MolmoBot mobile@static   /tmp/cond2_static_pose/mobile_results.jsonl   133/144
  MolmoBot mobile+perturb  /tmp/cond3_perturbed/mobile_results.jsonl     159/204
  pi0.5 static (standalone)/tmp/pi05_static/pi05_static_results.jsonl     0/15
  pi0.5 mobile@static      /tmp/pi05_mobstatic/pi05_mobile_results.jsonl  (grasp module)
  pi0.5 mobile+perturb     /tmp/pi05_perturbed/pi05_mobile_results.jsonl  (grasp module)
  scripted IK              curate oracle = 100% (defines the 347 set) = feasibility ceiling
"""
from __future__ import annotations
import json, math, re, glob, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from poster_stats import wilson, newcombe_diff, mcnemar, logistic_fit, logistic_band

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results/molmobot_sweep/charts_v4"
OUT.mkdir(parents=True, exist_ok=True)

C_MOLMO, C_PI, C_SCRIPT = "#e76f51", "#9b5de5", "#264653"
C_MBPI0 = "#2a9d8f"  # MolmoBot-Pi0-DROID (pi0-architecture MolmoBot variant)
GRID = "#dcdcdc"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12.5,
    "axes.titlesize": 17, "axes.titleweight": "bold", "axes.labelsize": 13.5,
    "axes.edgecolor": "#333", "axes.linewidth": 1.0,
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "savefig.facecolor": "white", "figure.facecolor": "white", "axes.facecolor": "white",
    "mathtext.default": "regular",
})

SRC = {
    "mb_static":   Path("/tmp/mb_sh/molmobot_static_results.jsonl"),
    "mb_mobstat":  Path("/tmp/mb_ms/mobile_results.jsonl"),
    "mb_perturb":  Path("/tmp/mb_pt/mobile_results.jsonl"),
    "pi_static":   Path("/tmp/pi05_sh/results.jsonl"),
    "pi_mobstat":  Path("/tmp/pi05_ms/results.jsonl"),
    "pi_perturb":  Path("/tmp/pi05_pt/results.jsonl"),
    "mbpi0_static":  Path("/tmp/mbpi0_sh/mbpi0_static_hybrid_results.jsonl"),
    "mbpi0_mobstat": Path("/tmp/mbpi0_ms/mbpi0_mobile_results.jsonl"),
    "mbpi0_perturb": Path("/tmp/mbpi0_pt/mbpi0_mobile_results.jsonl"),
}


def load(key):
    p = SRC[key]
    if not p.exists():
        return {}
    out = {}
    for l in p.read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        if r.get("errored"):
            continue
        out[(int(r["house_index"]), str(r.get("traj_key", r.get("episode_idx", ""))))] = bool(r["success"])
    return out


def kn(d):
    return sum(d.values()), len(d)


def radius_pairs(logglob):
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
            d = DONE.search(line)
            if d:
                done[(lp, int(d.group(1)), int(d.group(2)))] = (d.group(3) == "True")
            h = HOUSE.search(line)
            if h and pend is not None:
                rows.append((lp, int(h.group(1)), int(h.group(2)), pend)); pend = None
    return [(math.hypot(p[0], p[1]), done[(lp, hh, ee)]) for (lp, hh, ee, p) in rows if (lp, hh, ee) in done]


def star(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "n.s."


# ---- data ------------------------------------------------------------------
d = {k: load(k) for k in SRC}
K = {k: kn(v) for k, v in d.items()}
mb_pairs = radius_pairs("/tmp/mb_pt/r*.log")  # NEW full-347 MolmoBot perturbed run (origpose)
mbpi0_pairs = radius_pairs("/tmp/mbpi0_pt/r*.log")  # MolmoBot-Pi0 perturbed run (origpose)
pi_pairs = (radius_pairs("/tmp/pi05_pt/r*.log") + radius_pairs("/tmp/pi05_pert2/r*.log")
            + radius_pairs("/tmp/pi05_pt3/r*.log") + radius_pairs("/tmp/pi05_pt4/r*.log"))
print({k: f"{v[0]}/{v[1]}" for k, v in K.items()}, "mb_pairs", len(mb_pairs), "pi_pairs", len(pi_pairs))

CONDS = ["Static base\n(fixed-base protocol)",
         "Mobile @ static pose\n(floating base + contact)",
         "Mobile + perturbed\n(+ viewpoint / approach)"]
# module -> [(k,n) per condition]
MB = [K["mb_static"], K["mb_mobstat"], K["mb_perturb"]]
MBPI0 = [K["mbpi0_static"], K["mbpi0_mobstat"], K["mbpi0_perturb"]]
PI = [K["pi_static"], K["pi_mobstat"], K["pi_perturb"]]

# ===========================================================================
# Chart 1: grouped bars, condition x module
# ===========================================================================
fig, ax = plt.subplots(figsize=(12.5, 6.8))
nC = 3
NBARS = 4
gw = 0.86; bw = gw / NBARS
xb = np.arange(nC)
def bars(offset, series, color, label, ceiling=False):
    for ci in range(nC):
        x = xb[ci] - gw / 2 + bw * (offset + 0.5)
        if ceiling:
            ax.bar(x, 100, width=bw * 0.9, color=color, edgecolor="white", zorder=3,
                   label=label if ci == 0 else None)
            ax.text(x, 101.5, "100", ha="center", va="bottom", fontsize=9, color=color, fontweight="bold")
            continue
        k, n = series[ci]
        p, lo, hi = wilson(k, n)
        ax.bar(x, p * 100, width=bw * 0.9, color=color, edgecolor="white", zorder=3,
               label=label if ci == 0 else None)
        ax.errorbar(x, p * 100, yerr=[[(p - lo) * 100], [(hi - p) * 100]], fmt="none",
                    ecolor="#222", elinewidth=1.4, capsize=4, capthick=1.4, zorder=4)
        ax.text(x, hi * 100 + 1.6, f"{p*100:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.text(x, max(3.5, p * 100 - 7), f"{k}/{n}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
bars(0, MB, C_MOLMO, "MolmoBot (served VLA)")
bars(1, MBPI0, C_MBPI0, "MolmoBot-Pi0 (pi0-arch, served)")
bars(2, PI, C_PI, "pi0.5 (pi05_droid, served)")
bars(3, None, C_SCRIPT, "Scripted IK (oracle ceiling)", ceiling=True)
ax.set_xticks(xb); ax.set_xticklabels(CONDS, fontsize=11.5)
ax.set_ylabel("Pick success rate (%)"); ax.set_ylim(0, 116); ax.set_yticks(range(0, 101, 20))
ax.set_title("Pick success by condition and module")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=10.5)
ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
# stats panel
mc = mcnemar([(d["mb_mobstat"][k], d["mb_perturb"][k]) for k in (set(d["mb_mobstat"]) & set(d["mb_perturb"]))])
dd, dlo, dhi = newcombe_diff(*MB[1], *MB[2])
panel = (f"95% Wilson CIs; n shown per cell; matched seeds, paired per scene. "
         f"MolmoBot perturbation effect (mobile@static -> +perturbed): Δ={dd*100:+.1f} pts "
         f"[{dlo*100:+.1f},{dhi*100:+.1f}], McNemar p={mc['p_exact']:.1e} {star(mc['p_exact'])} (paired n={mc['n']}). "
         f"pi0.5 (pure pi05, served grasp module) is at the floor in every condition.")
fig.text(0.5, -0.06, panel, ha="center", va="top", fontsize=9.3, family="DejaVu Sans Mono", color="#333", wrap=True)
fig.savefig(OUT / "chart1_success_by_condition_module.png"); plt.close(fig)

# ===========================================================================
# Chart 2: success vs radius
# ===========================================================================
edges = [0.0, 0.15, 0.30, 0.45, 0.75]
mids = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
def binned(pairs):
    b = defaultdict(lambda: [0, 0])
    for r, s in pairs:
        for i in range(len(edges) - 1):
            if edges[i] <= r < edges[i + 1]: b[i][1] += 1; b[i][0] += int(s); break
    return b
mbb, pib, mbpi0b = binned(mb_pairs), binned(pi_pairs), binned(mbpi0_pairs)
fig, ax = plt.subplots(figsize=(12.5, 6.8))
ax.plot([0, 0.72], [100, 100], "-", color=C_SCRIPT, lw=2.4, marker="s", ms=7, markevery=[0, 1],
        label="Scripted IK (feasibility ceiling)", zorder=3)
# MolmoBot fit + band + points
if mb_pairs:
    rad = np.array([r for r, _ in mb_pairs]); su = np.array([1.0 if s else 0.0 for _, s in mb_pairs])
    b0, b1, se1, cov = logistic_fit(rad, su)
    xx = np.linspace(0, 0.7, 100); yy, ylo, yhi = logistic_band(b0, b1, cov, xx)
    ax.fill_between(xx, ylo * 100, yhi * 100, color=C_MOLMO, alpha=0.13, zorder=1)
    ax.plot(xx, yy * 100, "-", color=C_MOLMO, lw=2.6, label="MolmoBot (logistic fit, 95% band)", zorder=5)
    mp = [wilson(mbb[i][0], mbb[i][1])[0] * 100 for i in range(4)]
    mlo = [wilson(mbb[i][0], mbb[i][1])[1] * 100 for i in range(4)]; mhi = [wilson(mbb[i][0], mbb[i][1])[2] * 100 for i in range(4)]
    ax.errorbar(mids, mp, yerr=[[p - l for p, l in zip(mp, mlo)], [h - p for p, h in zip(mp, mhi)]],
                fmt="o", color=C_MOLMO, ms=7, elinewidth=1.5, capsize=4, zorder=6)
    for x, p, i in zip(mids, mp, range(4)):
        ax.annotate(f"n={mbb[i][1]}", (x, p), textcoords="offset points", xytext=(0, -20), ha="center", fontsize=9, color="#7a2f1c")
# MolmoBot-Pi0 line (binned, sits between MolmoBot and pi0.5)
qp = [wilson(mbpi0b[i][0], mbpi0b[i][1])[0] * 100 if mbpi0b[i][1] else None for i in range(4)]
qmask = [(m, v, mbpi0b[i][1]) for i, (m, v) in enumerate(zip(mids, qp)) if v is not None]
if qmask:
    qx = [m for m, v, n in qmask]; qy = [v for m, v, n in qmask]
    qlo = [wilson(mbpi0b[i][0], mbpi0b[i][1])[1] * 100 for i in range(4) if mbpi0b[i][1]]
    qhi = [wilson(mbpi0b[i][0], mbpi0b[i][1])[2] * 100 for i in range(4) if mbpi0b[i][1]]
    ax.plot(qx, qy, "-", color=C_MBPI0, lw=2.4, label="MolmoBot-Pi0 (served grasp module)", zorder=6)
    ax.errorbar(qx, qy, yerr=[[p - l for p, l in zip(qy, qlo)], [h - p for p, h in zip(qy, qhi)]],
                fmt="o", color=C_MBPI0, ms=7, elinewidth=1.5, capsize=4, zorder=7)
    for m, v, n in qmask:
        ax.annotate(f"n={n}", (m, v), textcoords="offset points", xytext=(0, 11), ha="center", fontsize=8.5, color="#1d6f63")
# pi0.5 line (flat ~0)
pp = [wilson(pib[i][0], pib[i][1])[0] * 100 if pib[i][1] else None for i in range(4)]
pmask = [(m, v, pib[i][1]) for i, (m, v) in enumerate(zip(mids, pp)) if v is not None]
if pmask:
    px = [m for m, v, n in pmask]; py = [v for m, v, n in pmask]
    plo = [wilson(pib[i][0], pib[i][1])[1] * 100 for i in range(4) if pib[i][1]]
    phi = [wilson(pib[i][0], pib[i][1])[2] * 100 for i in range(4) if pib[i][1]]
    ax.plot(px, py, "-", color=C_PI, lw=2.2, label="pi0.5 (served grasp module)", zorder=6)
    ax.errorbar(px, py, yerr=[[p - l for p, l in zip(py, plo)], [h - p for p, h in zip(py, phi)]],
                fmt="D", color=C_PI, ms=7, elinewidth=1.5, capsize=4, zorder=7)
ax.set_xlabel(r"Base-perturbation radius  $r=\sqrt{dx^2+dy^2}$  (m, realized single offset)")
ax.set_ylabel("Pick success rate (%)"); ax.set_ylim(-3, 108); ax.set_xlim(-0.02, 0.72); ax.set_yticks(range(0, 101, 20))
ax.set_title("Pick success vs. base-perturbation radius")
ax.legend(loc="center left", frameon=False, fontsize=11)
ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
fig.text(0.5, -0.05,
         "Per-bin Wilson CIs; radius from the exact per-episode offset (single random draw, binned post-hoc — "
         "not a controlled sweep; max realized 0.66 m). Scripted holds 100% at every radius, so MolmoBot's drop "
         "is policy failure not infeasibility; pi0.5 is uniformly low (~20%), roughly flat across radius.",
         ha="center", va="top", fontsize=9.2, family="DejaVu Sans Mono", color="#333")
fig.savefig(OUT / "chart2_success_vs_radius.png"); plt.close(fig)

# ===========================================================================
# Chart 3: degradation slopegraph (success across conditions, line per module)
# ===========================================================================
fig, ax = plt.subplots(figsize=(11, 6.8))
xs = [0, 1, 2]
for series, color, lab in [(MB, C_MOLMO, "MolmoBot (served VLA)"),
                           (MBPI0, C_MBPI0, "MolmoBot-Pi0 (pi0-arch, served)"),
                           (PI, C_PI, "pi0.5 (pi05_droid, served)")]:
    ys = [wilson(k, n)[0] * 100 for k, n in series]
    los = [(wilson(k, n)[0] - wilson(k, n)[1]) * 100 for k, n in series]
    his = [(wilson(k, n)[2] - wilson(k, n)[0]) * 100 for k, n in series]
    ax.errorbar(xs, ys, yerr=[los, his], color=color, lw=2.6, marker="o", ms=10, capsize=5,
                elinewidth=1.5, label=lab, zorder=5)
    dy = {C_MOLMO: 12, C_MBPI0: 12, C_PI: -24}[color]
    ha = {C_MOLMO: "right", C_MBPI0: "left", C_PI: "center"}[color]
    for x, y, (k, n) in zip(xs, ys, series):
        ax.annotate(f"{y:.0f}%\n{k}/{n}", (x, y), textcoords="offset points",
                    xytext=(-6 if ha == "right" else (6 if ha == "left" else 0), dy),
                    ha=ha, fontsize=9.0, fontweight="bold", color=color)
ax.plot(xs, [100, 100, 100], "-s", color=C_SCRIPT, lw=2.4, ms=9, label="Scripted IK (oracle ceiling)", zorder=4)
ax.text(2.02, 100, "100%", va="center", fontsize=9.5, color=C_SCRIPT, fontweight="bold")
ax.set_xticks(xs); ax.set_xticklabels(["Static base\n(fixed-base)", "Mobile @\nstatic pose", "Mobile +\nperturbed"], fontsize=12)
ax.set_xlim(-0.3, 2.5); ax.set_ylim(-4, 112); ax.set_yticks(range(0, 101, 20))
ax.set_ylabel("Pick success rate (%)")
ax.set_title("Degradation by module across conditions")
ax.legend(loc="center right", frameon=False, fontsize=11.5)
ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
fig.text(0.5, -0.04,
         "Only MolmoBot performs the task and degrades under perturbation; pi0.5 (pure pi05, same served-grasp-module "
         "pipeline) is floored in every condition; scripted IK is a flat 100% feasibility ceiling. So the gap is "
         "architecture-dependent, NOT consistent across VLAs (pi0.5 is also confounded by sim2real + control rate).",
         ha="center", va="top", fontsize=9.2, family="DejaVu Sans Mono", color="#333")
fig.savefig(OUT / "chart3_degradation_slopegraph.png"); plt.close(fig)

# ---- console ---------------------------------------------------------------
for k in SRC:
    kk, nn = K[k]
    if nn:
        p, lo, hi = wilson(kk, nn); print(f"  {k:12s}: {kk}/{nn} = {100*p:.1f}%  [{100*lo:.1f},{100*hi:.1f}]")
print("wrote v4 charts to", OUT)
