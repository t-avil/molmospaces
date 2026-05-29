# MolmoBot integration — progress log

Goal: replace the scripted grasp in the hybrid perception->grasp policy with a
served MolmoBot VLA, behind the same BasePolicy interface. Autonomous work on
branch `mobile-franka/molmobot-integration`.

## Environment (2026-05-29)
- 8× Quadro RTX 6000, 24 GB each, idle → can serve the 7B MolmoBot-DROID on one card.
- Disk 4.6T free; github + HuggingFace reachable.

## Where we start
- Hybrid policy scaffold (committed here): `policy/solvers/navigation/hybrid_point_grasp_policy.py`
  + `HybridPointGraspPolicyConfig` + `MobileFrankaHybridPointGraspEvalConfig`.
  Pipeline: point -> deproject(depth) -> approach(base) -> grasp.
  Status: runs end-to-end with `pointing_mode="ground_truth"` + scripted
  PickPlanner grasp; 10/10 of completed episodes succeeded on a 12-ep smoke batch.
- MolmoBot facts (from research): Molmo2 VLM + flow-matching action head; inputs =
  exo + wrist RGB (preset `franka_one_random_then_wrist`), 8-dim state, language,
  current+prev frame; output = joint-position action chunks (`--action-type joint_pos`,
  `FrankaState8ClampAbsPosConfig`). Served via `serve_molmo.py`; molmospaces has a
  `WebsocketPolicy` (msgpack over ws) client.

## Plan
1. Clone MolmoBot, read serve_molmo.py + eval/camera/action presets (exact obs/action).
2. Download MolmoBot-DROID (Franka), serve it on a GPU, confirm an inference round-trip.
3. Add a molmobot grasp stage to the hybrid policy (after approach), via WebsocketPolicy.
4. Align observation format (cameras, 8-dim state, resolution); execute joint-pos chunks.
5. Smoke-test; keep the scripted grasp as fallback.

## Decisions / blockers
- (log here as work proceeds)
