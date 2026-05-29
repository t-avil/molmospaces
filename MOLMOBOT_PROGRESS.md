# MolmoBot integration — progress log

Goal: replace the scripted grasp in the hybrid perception->grasp policy with a
served MolmoBot VLA, behind the same BasePolicy interface. Autonomous work on
branch `mobile-franka/molmobot-integration`.

## Environment (2026-05-29)
- 8× Quadro RTX 6000, 24 GB each, idle → can serve the 7B MolmoBot-DROID on one card.
- Filesystem has TB free BUT a per-user HOME quota (~tens of GB) — heavy artifacts must live in /tmp.
- github + HuggingFace reachable.

## Where we start
- Hybrid policy scaffold (committed f66c9081): `policy/solvers/navigation/hybrid_point_grasp_policy.py`
  + `HybridPointGraspPolicyConfig` + `MobileFrankaHybridPointGraspEvalConfig`.
  point -> deproject(depth) -> approach(base) -> grasp. Runs end-to-end with ground-truth pointing +
  scripted grasp; 10/10 completed episodes on a 12-ep smoke batch.

## Interface contract (confirmed from MolmoBot source)
- Server: `launch_scripts/serve_molmo.py --hf-repo allenai/MolmoBot-DROID [--action-type joint_pos|joint_pos_rel]`
  -> loads `RealRobotVLAPolicy`, serves via `WebsocketPolicyServer` on **port 8000** (msgpack).
  Matches local `molmo_spaces/policy/learned_policy/websocket_policy.py`.
- In-process ("recommended external-repo pattern"): subclass `JsonBenchmarkEvalConfig` with
  `SynthVLAPolicyConfig` + `FrankaRobotConfig`
  (`olmo/eval/configure_molmo_spaces.py:SynthVLAFrankaBenchmarkOriginalEvalConfig`).
- `SynthVLAPolicyConfig`: camera_names=["exo_camera_1","wrist_camera"] (cameras come from the BENCHMARK json);
  action_spec {arm:7,gripper:1}; action_keys {arm: joint_pos_rel|joint_pos, gripper: joint_pos};
  action_horizon=16, execute_horizon=8 (server-side buffering); gripper_representation_count=1;
  relative_max_joint_delta=[0.2]x7; policy_dt_ms=200;
  obs = per-cam RGB + obs["robot_state"]["qpos"]["arm"|"gripper"] + task description.

## Plan (refined)
1. [done] env assess, clone, map interface.
2. [in progress] MolmoBot serving env (`uv sync --extra eval` in /tmp) + MolmoBot-DROID download (~14GB to /tmp).
3. VALIDATE molmobot standalone: run MolmoBot's own SynthVLAFrankaBenchmarkOriginalEvalConfig on a
   FrankaPickDroidMiniBench episode -> confirm it loads/grasps + observe the exact obs format. De-risk first.
4. Serve molmobot (port 8000); integrate as the GRASP stage of HybridPointGraspPolicy via the WebsocketPolicy
   client (point->approach in our repo, grasp delegated to served molmobot; base parked). Add
   grasp_mode="scripted"|"molmobot" + host/port to the config.
5. Align obs (exo+wrist cameras present; qpos; task); apply arm/gripper joint actions.

## Decisions
- Flagship MolmoBot-DROID (7B); 8x24GB GPUs available.
- Served (websocket) for the hybrid integration: decouples our sim env from MolmoBot's pinned-molmospaces
  serving env via the msgpack obs/action contract.
- Validate molmobot standalone (step 3) before hybrid wiring.

## Blockers / incidents
- 2026-05-29 HOME QUOTA EXCEEDED (os error 122): 14GB model + uv env blew the per-user home quota.
  Resolution: relocated ALL heavy molmobot artifacts to **/tmp** (48GB, no quota): repo /tmp/MolmoBot,
  env via UV_CACHE_DIR=/tmp/uv-cache, model + HF_HOME under /tmp. Freed home by deleting disposable
  eval_runs/ (trajectories; benchmark JSON + results kept) + the partial download.
  NOTE: /tmp is ephemeral (cleared on reboot) — re-fetch model+env if the box reboots.
- Risk: molmobot trained on DROID fixed-base Franka; our mobile_franka parks the base then grasps with the
  same 7-DOF arm+gripper (should transfer), but camera viewpoints/obs format must match the training preset.

## Status (2026-05-29)
- uv sync (/tmp) + MolmoBot-DROID download (/tmp) running in background. Next: validate standalone, then wire.
