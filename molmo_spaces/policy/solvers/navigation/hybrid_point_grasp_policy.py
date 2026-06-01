"""Hybrid perception -> grasp policy for mobile_franka (POC).

Pipeline:  point (external cam) -> deproject (depth + intrinsics/extrinsics)
           -> approach (drive base in) -> grasp (scripted PickPlannerPolicy).

This is the POC stand-in for molmobot: the GRASP stays scripted
(PickPlannerPolicy runs on the target object's geometry, since the scripted
grasp needs the mesh, not just a point), while PERCEPTION drives targeting +
approach. molmobot later replaces the grasp behind this same BasePolicy
interface.

Pointing is pluggable (see HybridPointGraspPolicyConfig.pointing_mode):
  * "ground_truth" : use the target object's world center as the perceived
    point. Runs the whole loop with NO served model -> validates the
    deproject/approach/grasp plumbing first.
  * "molmo"        : call a served Molmo (RUMClient.infer_point) on the
    exocentric RGB, then deproject. Requires a molmo server (host/port),
    same pattern as cap_policy.py.

STATUS: approach controller reuses NavToOriginalBasePolicy's planar driving
(drives to a perception-derived standoff pose, stops on contact or step cap).
The live-molmo pointing path is still a TODO (needs a served model + exo cam obs).
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

import numpy as np

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.policy.solvers.navigation.nav_to_original_policy import (
    NavToOriginalBasePolicy,
)
from molmo_spaces.policy.solvers.object_manipulation.pick_planner_policy import (
    PickPlannerPolicy,
)
from molmo_spaces.tasks.task import BaseMujocoTask
from molmo_spaces.utils.pose import pos_quat_to_pose_mat

log = logging.getLogger(__name__)


def deproject_pixel_to_world(
    px: float, py: float, depth: float,
    intrinsic_cv: np.ndarray, extrinsic_cv: np.ndarray,
) -> np.ndarray:
    """Pixel (px,py) + metric depth -> 3D world point.

    intrinsic_cv : 3x3 (CV convention, see env/sensors_cameras.py CameraParamsSensor).
    extrinsic_cv : 3x4 [R|t].
    TODO(verify): sensors_cameras.py builds extrinsic_cv as inv(world2cam) but
      comments it "world2cam" — confirm the convention in sim before trusting
      this. Below assumes world2cam (p_cam = R p_world + t). If it's actually
      cam2world, use p_world = R @ p_cam + t instead.
    """
    fx, fy = intrinsic_cv[0, 0], intrinsic_cv[1, 1]
    cx, cy = intrinsic_cv[0, 2], intrinsic_cv[1, 2]
    p_cam = np.array([(px - cx) * depth / fx, (py - cy) * depth / fy, depth], dtype=np.float64)
    R, t = extrinsic_cv[:3, :3], extrinsic_cv[:3, 3]
    return R.T @ (p_cam - t)  # world2cam assumption; see TODO above


class _NavToPointPolicy(NavToOriginalBasePolicy):
    """NavToOriginal that drives the planar base to an explicitly-provided
    (x, y, yaw) target (perception-derived) instead of env.original_robot_base_pose.
    Inherits the ramped driving, collision-free target search, and contact check."""

    def set_target(self, x: float, y: float, yaw: float, z: float = 0.0) -> None:
        self._goal_xyz_yaw = (float(x), float(y), float(yaw), float(z))

    def _resolve_target(self) -> bool:
        if self._target_xy_yaw is not None:
            return True
        goal = getattr(self, "_goal_xyz_yaw", None)
        if goal is None:
            return False
        x, y, yaw, z = goal
        fx, fy, fyaw = self._find_collision_free_pose(x, y, yaw)
        self._target_xy_yaw = np.array([fx, fy, fyaw], dtype=np.float64)
        qw, qz = math.cos(fyaw / 2.0), math.sin(fyaw / 2.0)
        self._target_pose_m = pos_quat_to_pose_mat([fx, fy, z], [qw, 0.0, 0.0, qz])
        return True


class HybridPointGraspPolicy(PickPlannerPolicy):
    """point -> approach -> scripted grasp, perception-driven targeting."""

    def __init__(self, config: MlSpacesExpConfig, task: BaseMujocoTask) -> None:
        super().__init__(config, task)
        self._phase = "point"          # point -> approach -> pick
        self._target_world: np.ndarray | None = None
        self._nav: _NavToPointPolicy | None = None
        self._approach_steps = 0
        self._mb = None  # molmobot WebsocketPolicy client (lazy, grasp_mode="molmobot")

    def reset(self, reset_retries: bool = True) -> None:
        self._phase = "point"
        self._target_world = None
        self._nav = None
        self._approach_steps = 0
        # Defer pick planning until after approach (mirror NavThenPick): plan
        # once the base is parked so IK feasibility reflects the real pose.
        self.action_primitives = []
        self.action_idx = 0
        if reset_retries:
            self.retry_count = 0
        self.sequential_ik_failures = 0
        self.target_poses = {"pregrasp": np.eye(4), "grasp": np.eye(4), "lift": np.eye(4)}
        if getattr(self.config.policy_config, "grasp_mode", "scripted") == "molmobot":
            self._ensure_mb_client()
            self._mb_reset_server()  # in-band reset on the persistent connection (no reconnect)

    def _ensure_mb_client(self) -> None:
        if self._mb is None:
            from molmo_spaces.policy.learned_policy.websocket_policy import WebsocketPolicy

            pc = self.config.policy_config
            self._mb = WebsocketPolicy(self.config, "synthvla", host=pc.molmobot_host, port=pc.molmobot_port)
            self._mb.task = self.task  # obs_to_model_input uses task.get_task_description()
            self._mb.prepare_model()  # connect ONCE; reused across episodes (reconnect deadlocks the server)

    def _mb_reset_server(self) -> None:
        """Reset the served policy's obs-history/action-buffer between episodes via
        an in-band sentinel (reconnecting deadlocks the single-connection server)."""
        import msgpack_numpy as _mnp

        try:
            self._mb._ws.send(_mnp.packb({"__reset__": True}))
            self._mb._ws.recv(timeout=30)  # ack
        except Exception as e:  # noqa: BLE001
            log.warning(f"[Hybrid] molmobot in-band reset failed ({e!r}); reconnecting")
            try:
                self._mb.reset()
            except Exception:  # noqa: BLE001
                pass

    def get_phase(self) -> str:
        if self._phase == "pick":
            return super().get_phase()
        if self._phase == "mb_grasp":
            return "mb-grasp"
        return f"hybrid-{self._phase}"

    def _check_for_failures(self) -> bool:
        return super()._check_for_failures() if self._phase == "pick" else False

    # ---- perception ---------------------------------------------------------
    def _perceive_target(self, observation: Any) -> np.ndarray:
        mode = getattr(self.config.policy_config, "pointing_mode", "ground_truth")
        if mode == "ground_truth":
            return self._ground_truth_point()
        if mode == "molmo":
            return self._molmo_point(observation)
        raise ValueError(f"unknown pointing_mode: {mode}")

    def _ground_truth_point(self) -> np.ndarray:
        """No-serving stub: target object's world center as the perceived point."""
        tc = self.config.task_config
        om = self.task.env.object_managers[self.task.env.current_batch_index]
        return np.asarray(om.get_object_by_name(tc.pickup_obj_name).position, dtype=np.float64)

    def _molmo_point(self, observation: Any) -> np.ndarray:
        # TODO(poc): (1) read exocentric RGB + depth + intrinsic_cv/extrinsic_cv
        #   from the sensor obs (env/sensors_cameras.py: CameraParamsSensor, DepthSensor;
        #   exo cam must have record_depth=True). (2) RUMClient(host, port).infer_point(
        #   rgb, object_name, task) -> normalized (y, x); scale to pixels.
        #   (3) deproject_pixel_to_world(px, py, depth[py, px], K, E). See cap_policy.py.
        raise NotImplementedError("molmo pointing path: wire RUMClient + exocentric cam obs")

    # ---- approach -----------------------------------------------------------
    def _approach_action(self, observation: Any) -> dict[str, Any] | None:
        """Drive the planar base toward a standoff pose facing the perceived
        point. Returns the base action while approaching, or None when the
        approach is complete (reached / contact / step cap)."""
        if self._nav is None:
            self._nav = _NavToPointPolicy(self.config, self.task)
            self._nav.reset()
            cur = self._nav._current_xy_yaw()  # (x, y, yaw) world
            pc = self.config.policy_config
            orig = getattr(self.task.env, "original_robot_base_pose", None)
            if getattr(pc, "approach_target", "perceived") == "original" and orig is not None:
                # Drive to the episode's training base pose (isolates grasp from approach error).
                ax, ay = float(orig[0]), float(orig[1])
                qw, qx, qy, qz = (float(v) for v in orig[3:7])
                yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
            else:
                p = self._target_world
                dx, dy = float(p[0] - cur[0]), float(p[1] - cur[1])
                dist = math.hypot(dx, dy)
                yaw = math.atan2(dy, dx)  # face the object
                standoff = float(getattr(pc, "approach_standoff_m", 0.45))
                if dist > standoff:
                    ux, uy = dx / dist, dy / dist
                    ax, ay = float(p[0] - standoff * ux), float(p[1] - standoff * uy)
                else:
                    ax, ay = float(cur[0]), float(cur[1])  # already close enough
            self._nav.set_target(ax, ay, yaw)
            self._approach_steps = 0
            log.info(
                f"[Hybrid] approach -> (x={ax:.3f}, y={ay:.3f}, yaw={yaw:+.3f}) "
                f"target={getattr(pc, 'approach_target', 'perceived')}"
            )

        self._approach_steps += 1
        max_steps = int(getattr(self.config.policy_config, "max_approach_steps", 500))
        # "move forward until collision, then stop" + a step cap so we always terminate.
        if self._nav._base_in_collision() or self._approach_steps > max_steps:
            return None
        action = self._nav.get_action(observation)
        if action is None or action.get("done"):
            return None
        return action

    # ---- main loop ----------------------------------------------------------
    def get_action(self, observation: Any) -> dict[str, Any]:
        if os.environ.get("MLSPACES_DEBUG_OBS") and not getattr(self, "_dumped_obs", False):
            self._dumped_obs = True
            o = observation[0] if isinstance(observation, list) else observation
            if isinstance(o, dict):
                for k, v in o.items():
                    try:
                        a = np.asarray(v)
                        log.info(f"[OBS] {k}: shape={a.shape} dtype={a.dtype}")
                    except Exception:
                        log.info(f"[OBS] {k}: type={type(v).__name__} val={str(v)[:80]}")
                    if isinstance(v, dict):
                        for kk, vv in v.items():
                            try:
                                log.info(f"[OBS]   {k}.{kk}: shape={np.asarray(vv).shape}")
                            except Exception:
                                log.info(f"[OBS]   {k}.{kk}: {type(vv).__name__}")
            else:
                log.info(f"[OBS] observation type={type(o).__name__}")

        if self._phase == "point":
            self._target_world = self._perceive_target(observation)
            log.info(f"[Hybrid] perceived target (world): {self._target_world}")
            self._phase = "approach"

        if self._phase == "approach":
            action = self._approach_action(observation)
            if action is not None:
                action.pop("done", None)
                for k, v in self.robot_view.get_noop_ctrl_dict().items():
                    action.setdefault(k, v)  # hold arm/gripper during approach
                return action
            if getattr(self.config.policy_config, "grasp_mode", "scripted") == "molmobot":
                self._phase = "mb_grasp"
                log.info("[Hybrid] approach complete; handing off to molmobot grasp")
            else:
                self._phase = "pick"
                log.info("[Hybrid] approach complete; planning scripted grasp")
                super().reset(reset_retries=False)  # plan grasp, base parked

        if self._phase == "mb_grasp":
            return self._molmobot_grasp_action(observation)

        return super().get_action(observation)

    def _molmobot_grasp_action(self, observation: Any) -> dict[str, Any]:
        """Query the served molmobot for the grasp; apply its arm/gripper joint
        targets while holding the (already-parked) base."""
        o = observation[0] if isinstance(observation, list) else observation
        if os.environ.get("MLSPACES_SAVE_MB_FRAMES") and not getattr(self, "_saved_frames", False):
            self._saved_frames = True
            try:
                from PIL import Image

                Image.fromarray(np.asarray(o["exo_camera_1"])).save("/tmp/mb_exo.png")
                Image.fromarray(np.asarray(o["wrist_camera"])).save("/tmp/mb_wrist.png")
                log.info("[Hybrid] saved molmobot input frames: /tmp/mb_exo.png /tmp/mb_wrist.png")
            except Exception as e:  # noqa: BLE001
                log.warning(f"[Hybrid] frame save failed: {e!r}")
        arm = np.asarray(o["qpos"]["arm"], dtype=np.float32)
        grip = np.asarray(o["qpos"]["gripper"], dtype=np.float32)
        task = (
            self.task.get_task_description()
            if hasattr(self.task, "get_task_description")
            else "pick up the object"
        )
        # Send a CLEAN obs (the full sim obs has non-serializable env_states/task_info objects).
        mb_obs = {
            "exo_camera_1": np.asarray(o["exo_camera_1"]),
            "wrist_camera": np.asarray(o["wrist_camera"]),
            "qpos": {"arm": arm, "gripper": grip},
            "robot_state": {"qpos": {"arm": arm, "gripper": grip}},
            "task": task,
        }
        # WebsocketPolicy.infer hardcodes recv timeout=10s, too tight for the
        # cold first inference after a per-episode reset. Send/recv directly with
        # a longer timeout, and hold (no crash) if the server is unresponsive.
        import msgpack_numpy as _mnp

        try:
            self._mb._ws.send(_mnp.packb(mb_obs))
            resp = self._mb._ws.recv(timeout=60)
            if isinstance(resp, str):
                raise RuntimeError(resp)
            out = _mnp.unpackb(resp)
        except Exception as e:  # noqa: BLE001
            log.warning(f"[Hybrid] molmobot infer failed ({e!r}); holding this step")
            return self.robot_view.get_noop_ctrl_dict()
        act = self.robot_view.get_noop_ctrl_dict()  # holds base + all groups
        if "arm" in act:
            act["arm"] = np.asarray(out["arm"], dtype=np.float64)
        if "gripper" in act:
            g = np.asarray(out["gripper"], dtype=np.float64).reshape(-1)
            tgt = np.asarray(act["gripper"], dtype=np.float64).reshape(-1)
            tgt[: len(g)] = g[: len(tgt)]
            act["gripper"] = tgt
        return act
