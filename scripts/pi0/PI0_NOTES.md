# Pure pi0.5 (pi05_droid) as a served grasp module — notes

These scripts live in the MolmoBot-Pi0 repo (/tmp/MolmoBot/MolmoBot-Pi0) at runtime;
copied here for version control + reproducibility.

- serve_pi05.py: serves genuine pi05_droid over the molmospaces websocket protocol
  (same WebsocketPolicyServer molmo uses), so the mobile hybrid's grasp_mode="molmobot"
  client calls it unchanged. use_torch=False (JAX/orbax weights). buffer_length via
  MLSPACES_PI_BUFLEN (default 4) = closed-loop re-inference horizon for grasp timing.
- run_pi05_static.py: standalone fixed-franka pi0.5 eval (PiJointPosPolicy, model_name
  pi05_mlspaces_finetune). OPENPI_DATA_HOME=/tmp/openpi-cache, weights pre-staged via
  anonymous gcsfs from gs://openpi-assets/checkpoints/pi05_droid.

## Critical fix in molmobot_pi0/eval/policies/pi.py (PiJointPosPolicy)
Stock pi05_droid outputs DELTA joint actions (centered ~0); the runner treated them as
ABSOLUTE targets -> arm drove to a near-zero config -> 0% everywhere. Fix: reconstruct
absolute = chunk_start_joint_pos + delta*1.0 over the whole chunk, and feed gripper
proprio as clip(raw_grip[:1]/0.824033, 0, 1). This is the difference between 0% (bug)
and real grasp behaviour.
