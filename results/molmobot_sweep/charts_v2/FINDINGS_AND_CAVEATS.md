# Mobile-manipulation distribution-shift study — findings & caveats

Generated from `scripts/make_poster_charts_v2.py`. All numbers on the 347-scene
oracle-passing set, matched seeds, paired per scene.

## Headline numbers
| Condition (MolmoBot, served VLA) | success | 95% Wilson CI |
|---|---|---|
| Static base (fixed-base protocol) | 231/347 = **66.6%** | [61.4, 71.3] |
| Mobile + perturbed (floating base, contact, viewpoint) | 135/347 = **38.9%** | [33.9, 44.1] |
| **Degradation** | **−27.7 pts** | CIs disjoint → robust |
| Scripted IK (oracle) | 347/347 = 100% | by construction (ceiling) |

Success vs realized base-perturbation radius r = √(dx²+dy²):
| radius band (m) | success | n | 95% Wilson |
|---|---|---|---|
| 0.00–0.15 | 62.1% | 29 | [44.0, 77.3] |
| 0.15–0.30 | 68.5% | 73 | [57.1, 78.0] |
| 0.30–0.45 | 36.5% | 126 | [28.6, 45.2] |
| 0.45–0.66 | 17.6% | 119 | [11.8, 25.5] |

## Findings
1. Static-trained MolmoBot loses **27.7 pts** of pick success moving from the fixed-base
   protocol to a mobile base with full physics and a perturbed start pose (66.6%→38.9%).
   The 95% CIs are disjoint, so the gap is robust, not noise.
2. The loss is **radius-driven**: success holds ~65% out to ~0.30 m of base offset, then
   collapses to ~18% by 0.45–0.66 m.
3. Scripted IK solves **100% at every radius** — the geometry stays feasible — so the
   degradation is **policy failure under viewpoint/approach shift, not infeasibility**.

## Caveats (read before putting on a poster)
1. **Single module.** Only MolmoBot has complete data. pi0.5 has NO usable runs (one
   abandoned static run, ~34 episodes, 1 success; pi0-mobile path is not wired). RQ2
   ("is the gap consistent across pi0.5 and MolmoBot?") is **not answered** by these charts.
2. **Missing middle condition.** "Mobile-at-static-pose" (the cell that isolates floating
   base + contact from viewpoint shift) was killed by a disk-quota `OSError` (315/347 errored)
   and is unusable. So the −27.7 pts **conflates** base-mobility/contact with viewpoint/approach
   shift; the three factors are not yet separated. Chart 1 shows 2 conditions, not 3.
3. **Radius is realized, not controlled.** Perturbation is a single per-scene draw
   dx,dy ~ U(−0.5, 0.5), dyaw ~ U(−π/4, π/4) — NOT a sweep over r∈{0.1,0.25,0.5,1.0}.
   Realized radius maxes at **0.66 m** (theoretical max 0.707); **no scene reaches r=1.0 m**.
   Chart 2 bins the realized radius.
4. **Scripted = ceiling by construction.** The 347-scene set is *defined* as the scenes
   scripted IK solves under perturbation, so its 100% is tautological. It is a valid
   solvability floor (failures are policy-attributable) but not an independent result.
5. **Yaw not on the radius axis.** Chart 2's radius uses translation (dx,dy) only; the
   ±45° yaw perturbation is present in the episodes but not shown on the x-axis.
6. **Static pairing mixes factors.** Static (r=0) vs mobile pairs the same 347 scenes, but
   the static cell has no perturbation, so static-vs-mobile mixes base-mobility with perturbation.
7. **Low-radius non-monotonicity.** The 0–0.15 m bin (62%, n=29) sits below the 0.15–0.30 m
   bin (68%); small n, wide CI — do not over-read the leftmost point.
8. **Metric.** success = the pipeline's authoritative "completed with success=" flag, not the
   lenient h5 per-step proximity flag (which overcounted by 167/347). 0 errored in the clean cells.

## Reproducibility (verified)
- Perturbations are deterministic: `seed = hash((0, house_index, episode_idx)) & 0xFFFFFFFF`,
  then `np.random.RandomState(seed).uniform(-0.5,0.5)` for dx, dy and `uniform(-π/4,π/4)` for dyaw.
  Recomputed all **1059/1059** logged perturbations to within 1e-9. Tuple-of-ints hashing is not
  salted by PYTHONHASHSEED, so this is portable.
- Realized radius over the 347 eval scenes: min 0.020 m, max 0.657 m, mean 0.377 m.
</content>
