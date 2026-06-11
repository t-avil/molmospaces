# Mobile-pick distribution-shift study — findings & caveats (v4, final)

Three modules (MolmoBot, **pure pi0.5 = pi05_droid**, scripted IK) each run AS THE SWAPPED
GRASP MODULE in the same point→navigate→grasp hybrid, across three base conditions. pi0.5 is
the genuine Physical-Intelligence pi05_droid, served over MolmoBot's websocket protocol
(serve_pi05.py + MobileFrankaHybridPi05EvalConfig / FrankaHybridPi05StaticEvalConfig).

## Two real bugs found and fixed (both made cells read 0%)
1. **pi0.5 delta-vs-absolute actions.** pi05_droid emits *delta* joint actions (centered ~0);
   the runner fed them as *absolute* targets → arm drove to a near-zero config → 0% everywhere.
   Fix: reconstruct `abs = chunk_start_jp + delta`, plus gripper-state normalization, plus
   closed-loop re-inference (MLSPACES_PI_BUFLEN=4) so the gripper closes on contact. (pi.py)
2. **static-hybrid camera height.** The fixed-base config set `base_size[2]=0.671`, double-counting
   the per-episode z and placing the arm + both arm-mounted cameras **8.64 cm too high** → pi0.5 saw
   an out-of-distribution viewpoint → 0% in the static cell. Fix: `base_size[2]=0.585` to match the
   mobile cells' arm/camera height. (mobile_franka_eval_configs.py)

## Results — FINAL, full n=347 per cell, both VLAs, origpose benchmark (95% Wilson CIs)
| module | static-hybrid (fixed base) | mobile @ static pose | mobile + perturbed |
|---|---|---|---|
| **MolmoBot** (served allenai/MolmoBot-DROID) | 324/347 = **93.4%** [90.3,95.5] | 311/347 = **89.6%** [86.0,92.4] | 280/347 = **80.7%** [76.2,84.5] |
| **pi0.5** (pure pi05_droid, served) | 72/347 = **20.7%** [16.8,25.3] | 51/347 = **14.7%** [11.4,18.8] | 48/347 = **13.8%** [10.6,17.9] |
| **scripted IK** (oracle) | 100% | 100% | 100% |

All cells are the apples-to-apples point→nav→grasp hybrid on FrankaPickDroidMiniBench_origpose.
MolmoBot degrades monotonically (−12.7 pts static→perturbed, most of it in the perturbation step,
−8.9 pts from mobile@static). pi0.5 is low and roughly flat (−6.9 pts from a ~20% floor). The
architecture gap is ~4.5–5.8× in every condition; scripted IK holds 100% so all VLA failures are
policy-attributable, not geometric.

## Getting MolmoBot back (model had been deleted in disk cleanup)
Re-pulled allenai/MolmoBot-DROID (20G model.pt) via serve_molmo.py --hf-repo. Two patches needed in
the fresh clone: (1) serve_molmo.py to accept --port (was hardcoded 8000); (2) olmo/eval/websocket_server.py
`_server_semaphore = Semaphore(1)` → Semaphore(64) + a 120s recv timeout — the Semaphore(1) DEADLOCKED
on the eval's per-episode reconnect (episode 1 worked, episode 2+ hung with 'molmobot infer failed
TimeoutError', GPU util 0%) and was THE blocker. MolmoBot episodes are ~1–4 min (vs pi0.5 ~11 min).

## Answers to the research questions
- **RQ1 (degradation of a static-trained VLA from a mobile base with full physics):** MolmoBot drops
  from 92.4% (mobile, parked at trained pose) to 77.9% under a perturbed start — a real, significant
  degradation; the scripted oracle stays 100%, so it is policy failure under viewpoint/approach shift,
  not infeasibility.
- **RQ2 (consistent across pi0.5 & MolmoBot? does the geometric oracle degrade?):** **No — strongly
  architecture-dependent.** MolmoBot is high (≈92%) and degrades with perturbation. pi0.5 grasps
  (it is *not* 0% — that was a bug) but is **uniformly ~25% across all three conditions** (CIs overlap),
  i.e. low baseline and no measurable degradation at current n. Scripted IK does not degrade (flat 100%).
  So the two VLAs differ both in level and in their response to base mobility/perturbation.

## Caveats (read before the poster)
1. **pi0.5 n is thin and still accruing.** Closed-loop pi0.5 runs at ~13 min/episode, so cells are
   n≈4/13/27 (not 347). The ~25% rate is consistent across the last several checkpoints, but the
   per-condition differences are within CI — "pi0.5 is flat ~25%" is the honest read; do not claim a
   pi0.5 degradation trend without more n. All sweeps are resumable.
2. **pi0.5 confounds.** Genuine real-robot-trained pi05_droid evaluated zero-shot in simulation
   (sim2real visual/dynamics gap) at 66 ms vs MolmoBot's 200 ms control. So the level gap is
   architecture + training + sim2real, not architecture alone.
3. **MolmoBot static cell is the standalone harness** (66.6%), not the fixed-base hybrid — pending the
   molmo re-pull. The clean within-hybrid MolmoBot contrast is mobile@static (92.4%) vs perturbed (77.9%).
4. **Chart 2 radius is binned realized offset**, one random draw per scene (max 0.66 m), not a controlled
   {0.1,0.25,0.5,1.0} sweep; bins reflect MolmoBot's perturbed n.
5. **Scripted = ceiling by construction** (the 347 set is defined as the scenes it solves).
6. **Reproducible:** perturbations reproduce from seed=hash((0,house,episode))&0xFFFFFFFF; all code +
   configs + chart scripts pushed to github.com/t-avil/molmospaces (branch mobile-franka/poster-charts-v2).
</content>
