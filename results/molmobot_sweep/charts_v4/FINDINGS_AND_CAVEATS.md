# Mobile-pick distribution-shift study — findings & caveats (v4, final)

Three modules (MolmoBot, **pure pi0.5 = pi05_droid**, scripted IK) each run AS THE
SWAPPED GRASP MODULE in the same point→navigate→grasp hybrid, across three base
conditions. pi0.5 is the genuine Physical-Intelligence pi05_droid served over the
identical websocket protocol MolmoBot uses (serve_pi05.py + MobileFrankaHybridPi05EvalConfig),
NOT a standalone policy — so it is finally an apples-to-apples comparison.

## Results (95% Wilson CIs; n per cell)
| module | static base | mobile @ static pose | mobile + perturbed |
|---|---|---|---|
| **MolmoBot** (served VLA) | 231/347 = **66.6%** [61.4,71.3] | 133/144 = **92.4%** [86.8,95.7] | 159/204 = **77.9%** [71.8,83.1] |
| **pi0.5** (pi05_droid, served grasp module) | 0/15 = **0%** [0,20.4] | 0/20 = **0%** [0,16.1] | 0/21 = **0%** [0,15.5] |
| **scripted IK** (oracle) | 100% | 100% | 100% |

MolmoBot perturbation effect (mobile@static → +perturbed, same pipeline, matched seeds):
Δ = **−14.5 pts** [−6.9, −21.4]; McNemar **p = 4.9e-5 *** (paired n=144).

## Answers to the research questions
- **RQ1 (degradation of a static-trained VLA from a mobile base w/ full physics):** MolmoBot
  drops from 92.4% (mobile, parked at the trained pose) to 77.9% once the start pose is
  perturbed — a real, significant degradation driven by viewpoint/approach shift, not geometry
  (scripted stays 100%).
- **RQ2 (consistent across pi0.5 & MolmoBot? does the geometric oracle degrade?):** **No — the gap
  is architecture-dependent, not consistent.** MolmoBot performs the task and degrades gracefully;
  pure pi0.5 is at the **floor (0%) in every condition**, including the easiest (parked at the
  trained pose). The scripted IK oracle does **not** degrade (flat 100% feasibility ceiling), so
  every VLA failure is a policy failure, not infeasibility. See chart 3 (slopegraph) for the
  one-picture version.

## Caveats (read before the poster)
1. **pi0.5's floor is a FAIR measurement but heavily confounded.** It is now the served grasp
   module in the identical pipeline/conditions as MolmoBot, scored on the same PickTask (lift ~5 cm).
   It makes structured grasp attempts (drives the arm; gripper uses the DROID 0–255 convention,
   pipeline byte-identical to molmo) — it is not a no-op or wiring failure. BUT: (a) **sim2real** —
   pi05_droid is trained on *real* DROID-robot data and evaluated *zero-shot in simulation* (mujoco),
   a large visual/dynamics gap; (b) **control rate** 66 ms vs MolmoBot's 200 ms. So the 0% reflects
   "this real-trained model doesn't transfer into this sim zero-shot" as much as architecture. Do not
   over-claim pi0.5 is "worse"; claim the gap is architecture/training-dependent.
2. **n varies and some cells are partial.** MolmoBot: static n=347, mobile@static n=144, perturbed
   n=204. pi0.5: n=15/20/21 (runs still in progress). Rates were stable across rounds. The MolmoBot
   mobile cells could not be finished to 347 because the 27 G molmo serve-weights were deleted (disk
   cleanup) and re-pulling won't fit current free /tmp. All sweeps are resumable.
3. **Chart 2 radius is binned, not a controlled sweep.** The benchmark applies one random per-scene
   offset, so radius is binned post-hoc (max 0.66 m); there is no {0.1,0.25,0.5,1.0} sweep. A clean
   sweep needs fixed-radius re-runs (disk-blocked for MolmoBot; moot for pi0.5, which is floored).
4. **MolmoBot static (66.6%) is a different pipeline** (fixed-base standalone) than the mobile hybrid
   cells — the clean within-pipeline contrast is mobile@static vs perturbed.
5. **Scripted = ceiling by construction** (the 347 set is defined as the scenes it solves).
6. **Reproducible:** all 1059 perturbations reproduce from seed=hash((0,house,episode))&0xFFFFFFFF.
</content>
