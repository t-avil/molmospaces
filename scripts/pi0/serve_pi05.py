"""Serve PURE pi0.5 (pi05_droid) over the molmospaces websocket protocol so the mobile
hybrid's grasp client (grasp_mode="molmobot") can call it as a drop-in grasp module.

Standalone mirror of molmobot_pi0/eval/real/serve.py, hardcoded for genuine pi0.5:
PiJointPosPolicy(model_name="pi05_mlspaces_finetune", use_torch=False) -> JAX/orbax
pi05_droid weights from OPENPI_DATA_HOME. The served obs->action contract is identical
to molmo's ({exo_camera_1, wrist_camera, qpos, task} -> {arm, gripper}); the 32-dim /
action_horizon-16 internals are absorbed server-side.

Run in the MolmoBot-Pi0 venv, one per GPU:
  CUDA_VISIBLE_DEVICES=g POLICY_SERVER_PORT=$((8000+g)) OPENPI_DATA_HOME=/tmp/openpi-cache \
  HF_HOME=/tmp/hf-cache .venv/bin/python serve_pi05.py
"""
import os
import logging

os.environ["JAX_ENABLE_X64"] = "0"  # 64-bit jax breaks openpi inference

import torch
from molmo_spaces.evaluation.policy_server import WebsocketPolicyServer
from molmobot_pi0.eval.policies.pi import PiJointPosPolicy

logging.basicConfig(level=logging.INFO)


def main():
    torch._inductor.config.triton.cudagraphs = False
    # buffer_length = closed-loop re-inference horizon. pi05 predicts a 16-action
    # chunk; executing all 16 open-loop (~1s @66ms) mistimes the gripper close
    # (gripper still open at closest approach). Re-inferring every few steps lets
    # the policy see the live state and close on contact. Override via MLSPACES_PI_BUFLEN.
    buflen = int(os.environ.get("MLSPACES_PI_BUFLEN", "4"))
    policy = PiJointPosPolicy(
        model_name="pi05_mlspaces_finetune",
        checkpoint_dir=None,
        use_torch=False,          # JAX/orbax pi05_droid weights
        buffer_length=buflen,     # closed-loop grasping (default 4 of the 16-action chunk)
        cameras=None,             # default exo_camera_1 + wrist_camera mapping
        device_id=0,              # CUDA_VISIBLE_DEVICES pins the physical GPU
        compile_mode=None,
    )
    policy.prepare_model()
    port = int(os.getenv("POLICY_SERVER_PORT", "8000"))
    server = WebsocketPolicyServer(
        [policy], policy.model_name, port=port,
        metadata={"action_type": "jointpos", "model_name": policy.model_name},
    )
    logging.info(f"Serving pure pi0.5 ({policy.model_name}) on :{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
