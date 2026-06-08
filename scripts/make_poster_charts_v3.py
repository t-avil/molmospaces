"""Poster charts v3 (technical) — factorized mobile-pick distribution-shift study.

Fixes the v2 double-perturbation confound (data now on TRUE original poses with a
single env-scaled perturbation, MLSPACES_PERTURB_SCALE). Adds inferential stats:
Wilson 95% CIs, McNemar paired tests on matched seeds, Newcombe difference CIs, and
a logistic-regression fit of success on perturbation radius.

Cells (MolmoBot served-VLA hybrid is one pipeline for cond2/cond3; static is the
in-process fixed-base protocol; pi0.5 static is MolmoBot-Pi0 on the fixed franka):
  static          results/molmobot_sweep/static_results.jsonl
  mobile@static   /tmp/cond2_static_pose/mobile_results.jsonl   (scale=0)
  mobile+perturb  /tmp/cond3_perturbed/mobile_results.jsonl     (scale=1)
  pi0.5 static    /tmp/pi_static/pi_static_results.jsonl
  scripted IK     curate oracle = 100% (defines the 347-scene set) = feasibility ceiling
Chart-2 radius = exact per-episode offset parsed from cond3 logs (ground truth).
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
OUT = REPO / "results/molmobot_sweep/charts_v3"
OUT.mkdir(parents=True, exist_ok=True)

SRC = {
    "static":   REPO / "results/molmobot_sweep/static_results.jsonl",
    "cond2":    Path("/tmp/cond2_static_pose/mobile_results.jsonl"),
    "cond3":    Path("/tmp/cond3_perturbed/mobile_results.jsonl"),
    "pi_static": Path("/tmp/pi05_static/pi05_static_results.jsonl"),  # PURE pi0.5 (pi05_droid)
}
COND3_LOGS = "/tmp/cond3_perturbed/r*.log"
PI05_PERTURB_LOGS = "/tmp/pi05_perturbed/r*.log"  # pi0.5 mobile+perturbed (if run)

C_MOLMO, C_PI, C_SCRIPT, C_STATIC = "#e76f51", "#9b5de5", "#264653", "#2a9d8f"
GRID = "#dcdcdc"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12.5,
    "axes.titlesize": 17, "axes.titleweight": "bold", "axes.labelsize": 13.5,
    "axes.edgecolor": "#333", "axes.linewidth": 1.0,
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "savefig.facecolor": "white", "figure.facecolor": "white", "axes.facecolor": "white",
    "mathtext.default": "regular",
})


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
        k = (int(r["house_index"]), str(r.get("traj_key", r.get("episode_idx", ""))))
        out[k] = bool(r["success"])
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
    pairs = []
    for (lp, hh, ee, p) in rows:
        if (lp, hh, ee) in done:
            pairs.append((math.hypot(p[0], p[1]), done[(lp, hh, ee)]))
    return pairs


# ---- data ------------------------------------------------------------------
d_static = load("static"); d_c2 = load("cond2"); d_c3 = load("cond3"); d_pi = load("pi_static")
ks, ns = kn(d_static); k2, n2 = kn(d_c2); k3, n3 = kn(d_c3); kp, npi = kn(d_pi)
pairs = radius_pairs(COND3_LOGS)
pairs_pi = radius_pairs(PI05_PERTURB_LOGS)
print(f"static {ks}/{ns}  cond2 {k2}/{n2}  cond3 {k3}/{n3}  pi0_static {kp}/{npi}  radius_pairs {len(pairs)}")

# paired McNemar: MolmoBot mobile@static vs mobile+perturbed (matched scenes)
common = set(d_c2) & set(d_c3)
mc = mcnemar([(d_c2[k], d_c3[k]) for k in common]) if common else None
# paired McNemar: pi0.5 vs MolmoBot at static baseline
common_s = set(d_pi) & set(d_static)
mc_s = mcnemar([(d_pi[k], d_static[k]) for k in common_s]) if common_s else None


def star(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "n.s."


# ===========================================================================
# Chart 1
# ===========================================================================
fig, ax = plt.subplots(figsize=(12, 6.6))
# (condition label, list of (module_label, color, k, n))
groups = [
    ("Static base\n(fixed-base protocol)",
     [("MolmoBot", C_MOLMO, ks, ns)] + ([("pi0.5", C_PI, kp, npi)] if npi else []) + [("Scripted", C_SCRIPT, 1, 1)]),
    ("Mobile @ static pose\n(floating base + contact)",
     [("MolmoBot", C_MOLMO, k2, n2), ("Scripted", C_SCRIPT, 1, 1)]),
    ("Mobile + perturbed\n(+ viewpoint / approach)",
     [("MolmoBot", C_MOLMO, k3, n3), ("Scripted", C_SCRIPT, 1, 1)]),
]
seen_lbl = set()
x = 0.0; xticks = []; gap = 0.55; bw = 0.26
for gi, (clabel, bars) in enumerate(groups):
    xs = [x + i * bw for i in range(len(bars))]
    xticks.append((sum(xs) / len(xs), clabel))
    for xi, (mlabel, col, k, n) in zip(xs, bars):
        p, lo, hi = wilson(k, n)
        lab = {"MolmoBot": "MolmoBot (served VLA)", "pi0.5": "π0.5 (pi05_droid, zero-shot)",
               "Scripted": "Scripted IK (oracle ceiling)"}[mlabel]
        ax.bar(xi, p * 100, width=bw * 0.92, color=col, edgecolor="white", linewidth=0.7,
               zorder=3, label=lab if lab not in seen_lbl else None)
        seen_lbl.add(lab)
        if mlabel != "Scripted":
            ax.errorbar(xi, p * 100, yerr=[[(p - lo) * 100], [(hi - p) * 100]], fmt="none",
                        ecolor="#222", elinewidth=1.5, capsize=4, capthick=1.5, zorder=4)
            ax.text(xi, hi * 100 + 1.8, f"{p*100:.1f}%", ha="center", va="bottom",
                    fontsize=11, fontweight="bold", zorder=5)
            ax.text(xi, max(2, p * 100 - 8), f"{k}/{n}", ha="center", va="center",
                    fontsize=8.5, color="white", fontweight="bold", zorder=6)
        else:
            ax.text(xi, 101.5, "100", ha="center", va="bottom", fontsize=9, color=C_SCRIPT, fontweight="bold")
    x = xs[-1] + gap

ax.set_xticks([t[0] for t in xticks]); ax.set_xticklabels([t[1] for t in xticks], fontsize=11.5)
ax.set_ylabel("Pick success rate (%)"); ax.set_ylim(0, 118); ax.set_yticks(range(0, 101, 20))
ax.set_title("Pick success by condition and module")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=10.5)
ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)

# stats panel
lines = ["Inference (95% Wilson CIs; matched seeds, paired per scene):"]
if mc:
    dd, dlo, dhi = newcombe_diff(k2, n2, k3, n3)
    lines.append(f"  MolmoBot, mobile@static vs +perturbed  (perturbation/viewpoint factor):")
    lines.append(f"    Δ = {dd*100:+.1f} pts [{dlo*100:+.1f}, {dhi*100:+.1f}];  McNemar χ²(1)={mc['chi2']:.1f}, "
                 f"p={mc['p_exact']:.1e} {star(mc['p_exact'])}  (paired n={mc['n']}, discordant b={mc['b']}/c={mc['c']})")
if mc_s:
    dd, dlo, dhi = newcombe_diff(kp, npi, ks, ns)
    lines.append(f"  pi0.5 vs MolmoBot at static baseline:  Δ={dd*100:+.1f} pts [{dlo*100:+.1f}, {dhi*100:+.1f}]; "
                 f"McNemar p={mc_s['p_exact']:.1e} {star(mc_s['p_exact'])} (paired n={mc_s['n']})")
fig.text(0.5, -0.10, "\n".join(lines), ha="center", va="top", fontsize=9.3, family="DejaVu Sans Mono", color="#333")
fig.savefig(OUT / "chart1_success_by_condition_module.png")
plt.close(fig)

# ===========================================================================
# Chart 2 — success vs realized radius + logistic fit
# ===========================================================================
if pairs:
    rad = np.array([r for r, _ in pairs]); succ = np.array([1.0 if s else 0.0 for _, s in pairs])
    edges = [0.0, 0.15, 0.30, 0.45, 0.75]
    mids = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
    b = defaultdict(lambda: [0, 0])
    for r, s in zip(rad, succ):
        for i in range(len(edges) - 1):
            if edges[i] <= r < edges[i + 1]:
                b[i][1] += 1; b[i][0] += int(s); break
    mp, mlo, mhi, ns_ = [], [], [], []
    for i in range(len(edges) - 1):
        k, n = b[i]; p, lo, hi = wilson(k, n)
        mp.append(p * 100); mlo.append(lo * 100); mhi.append(hi * 100); ns_.append(n)
    b0, b1, se1, cov = logistic_fit(rad, succ)
    or10 = math.exp(0.1 * b1); z = b1 / se1 if se1 else 0.0
    from scipy import stats as _st
    p_slope = 2 * _st.norm.sf(abs(z))
    xx = np.linspace(0, 0.7, 100)
    yy, ylo, yhi = logistic_band(b0, b1, cov, xx)

    fig, ax = plt.subplots(figsize=(12, 6.6))
    ax.plot([0, 0.7], [100, 100], "-", color=C_SCRIPT, lw=2.4, marker="s", ms=7,
            markevery=[0, 1], label="Scripted IK (feasibility ceiling, 100%)", zorder=3)
    ax.fill_between(xx, ylo * 100, yhi * 100, color=C_MOLMO, alpha=0.13, zorder=1)
    ax.plot(xx, yy * 100, "-", color=C_MOLMO, lw=2.6, label="MolmoBot logistic fit (95% band)", zorder=5)
    ax.errorbar(mids, mp, yerr=[[p - l for p, l in zip(mp, mlo)], [h - p for p, h in zip(mp, mhi)]],
                fmt="o", color=C_MOLMO, ms=8, elinewidth=1.6, capsize=5, capthick=1.6,
                label="MolmoBot per-bin (95% Wilson CI)", zorder=6)
    kp2, np2 = kn(d_c2)
    c2p = wilson(kp2, np2)[0] * 100
    sp = wilson(ks, ns)[0] * 100   # MolmoBot fixed-base static baseline
    # r=0 reference markers (no perturbation): fixed-base static and mobile@static
    ax.axhline(sp, ls="--", color="#8a8a8a", lw=1.3, zorder=1)
    ax.plot([0.0], [sp], marker="D", ms=11, color="#5f5f5f", zorder=7,
            label=f"MolmoBot static, fixed base (r=0): {sp:.1f}%")
    ax.plot([0.0], [c2p], marker="*", ms=18, color=C_STATIC, zorder=8,
            label=f"MolmoBot mobile @ static pose (r=0): {c2p:.1f}%")
    # pi0.5 markers/curve (added when pi0.5 data exists)
    if npi:
        pip = wilson(kp, npi)[0] * 100
        ax.plot([0.0], [pip], marker="P", ms=12, color=C_PI, zorder=8,
                label=f"pi0.5 static, fixed base (r=0): {pip:.1f}%")
    if pairs_pi:
        radp = np.array([r for r, _ in pairs_pi]); succp = np.array([1.0 if s else 0.0 for _, s in pairs_pi])
        bb = defaultdict(lambda: [0, 0])
        for r, s in zip(radp, succp):
            for i in range(len(edges) - 1):
                if edges[i] <= r < edges[i + 1]: bb[i][1] += 1; bb[i][0] += int(s); break
        pp = [wilson(bb[i][0], bb[i][1])[0] * 100 for i in range(len(edges) - 1)]
        pn = [bb[i][1] for i in range(len(edges) - 1)]
        ax.plot([m for m, n in zip(mids, pn) if n], [v for v, n in zip(pp, pn) if n],
                "-o", color=C_PI, lw=2.2, ms=7, label="pi0.5 mobile+perturbed", zorder=6)
    for xm, p, n in zip(mids, mp, ns_):
        ax.annotate(f"n={n}", (xm, p), textcoords="offset points", xytext=(0, -22), ha="center",
                    fontsize=9.5, color="#7a2f1c")
    ax.set_xlabel(r"Base-perturbation radius  $r=\sqrt{dx^2+dy^2}$  (m, realized single offset)")
    ax.set_ylabel("Pick success rate (%)"); ax.set_ylim(0, 108); ax.set_xlim(-0.02, 0.72)
    ax.set_yticks(range(0, 101, 20))
    ax.set_title("Pick success vs. base-perturbation radius")
    ax.legend(loc="lower left", frameon=False, fontsize=10.5)
    ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    txt = (f"Logistic fit: success ~ r.  OR per +0.10 m = {or10:.2f} "
           f"(slope β={b1:.2f}±{se1:.2f}, p={p_slope:.1e} {star(p_slope)}; n={len(rad)}).  "
           f"Scripted re-certifies solvability at every r, so the gap to 100% is policy failure, not infeasibility.")
    fig.text(0.5, -0.06, txt, ha="center", va="top", fontsize=9.4, family="DejaVu Sans Mono", color="#333")
    fig.savefig(OUT / "chart2_success_vs_radius.png")
    plt.close(fig)

# ---- console ---------------------------------------------------------------
for lab, k, n in [("static", ks, ns), ("mobile@static", k2, n2), ("mobile+perturb", k3, n3), ("pi0.5_static", kp, npi)]:
    if n:
        p, lo, hi = wilson(k, n)
        print(f"  {lab:16s}: {k}/{n} = {100*p:.1f}%  [{100*lo:.1f},{100*hi:.1f}]")
if mc:
    print(f"  McNemar c2-vs-c3: chi2={mc['chi2']:.1f} p={mc['p_exact']:.2e} b={mc['b']} c={mc['c']} n={mc['n']}")
print("wrote v3 technical charts to", OUT)
