"""Drive the mobile_franka base back to the benchmark-authored pose stashed
on `env.original_robot_base_pose` by JsonEvalTaskSampler.

Acts as a workaround while the per-step IK on mobile_franka is still being
fixed: the manipulation phases expect the arm-root at the benchmark pose,
so we restore the base there first.

The mobile_franka base actuators are configured with very high kp (25000)
and tiny kd (1) — when commanded with position targets they overshoot
violently. Until those gains are retuned, this policy moves the base by
writing the planar-joint qpos directly each step (interpolated toward the
target), bypassing actuator dynamics. Same teleport trick used by the
scripted planner's snap-back primitive; Max called this "as close as
possible" to a real nav.
"""

from __future__ import annotations

import logging
import math

import mujoco
import numpy as np

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.policy.base_policy import PlannerPolicy
from molmo_spaces.tasks.task import BaseMujocoTask
from molmo_spaces.utils.pose import pos_quat_to_pose_mat

log = logging.getLogger(__name__)


def _yaw_from_quat(qw: float, qx: float, qy: float, qz: float) -> float:
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


class NavToOriginalBasePolicy(PlannerPolicy):
    """Teleport the planar base toward env.original_robot_base_pose.

    Each step, writes the base move group's qpos one increment closer to the
    saved original pose. After convergence, signals episode done.
    """

    POS_TOLERANCE = 0.02  # meters
    YAW_TOLERANCE = 0.05  # radians
    POS_STEP = 0.05  # meters per step (actuator ramp; physics respects contacts)
    YAW_STEP = 0.05  # radians per step

    def __init__(self, config: MlSpacesExpConfig, task: BaseMujocoTask) -> None:
        super().__init__(config, task)
        self.robot_view = task.env.current_robot.robot_view
        self._target_pose_m: np.ndarray | None = None
        self._target_xy_yaw: np.ndarray | None = None
        self._reached = False
        self._step = 0

    def planners(self):
        return {}

    def reset(self):
        self._target_pose_m = None
        self._target_xy_yaw = None
        self._reached = False
        self._step = 0

    def _resolve_target(self) -> bool:
        if self._target_xy_yaw is not None:
            return True
        original = getattr(self.task.env, "original_robot_base_pose", None)
        if original is None:
            log.warning(
                "[NavToOriginal] env.original_robot_base_pose not set; "
                "JsonEvalTaskSampler must run with MobileFrankaRobotConfig. "
                "Returning no-op."
            )
            return False
        x, y, z = float(original[0]), float(original[1]), float(original[2])
        qw, qx, qy, qz = (float(v) for v in original[3:7])
        yaw = _yaw_from_quat(qw, qx, qy, qz)
        log.info(
            f"[NavToOriginal] benchmark target (x, y, yaw) = "
            f"({x:.3f}, {y:.3f}, {yaw:+.3f} rad)"
        )
        # Search collision-free pose around benchmark target.
        fx, fy, fyaw = self._find_collision_free_pose(x, y, yaw)
        if (fx, fy, fyaw) != (x, y, yaw):
            log.info(
                f"[NavToOriginal] benchmark pose in collision; using nudged "
                f"pose (x, y, yaw) = ({fx:.3f}, {fy:.3f}, {fyaw:+.3f} rad)"
            )
        self._target_xy_yaw = np.array([fx, fy, fyaw], dtype=np.float64)
        self._target_pose_m = pos_quat_to_pose_mat([fx, fy, z], [qw, qx, qy, qz])
        return True

    def _base_in_collision(self) -> bool:
        model = self.task.env.current_model
        data = self.task.env.current_data
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot_0/base")
        for ci in range(data.ncon):
            c = data.contact[ci]
            g1, g2 = c.geom1, c.geom2
            b1 = model.geom_bodyid[g1]
            b2 = model.geom_bodyid[g2]
            # Walk up body tree: contact with anything rooted at robot_0/base
            # OR descendants is self-contact; we only want base-vs-world.
            if b1 == bid or b2 == bid:
                other = b2 if b1 == bid else b1
                # Skip if other body is also part of robot (descendant of base).
                cur = other
                is_self = False
                while cur > 0:
                    if cur == bid:
                        is_self = True
                        break
                    cur = model.body_parentid[cur]
                if not is_self:
                    return True
        return False

    def _find_collision_free_pose(
        self, wx: float, wy: float, wyaw: float
    ) -> tuple[float, float, float]:
        """Snap base to (wx, wy, wyaw) and test contacts. If colliding, spiral
        outward in xy (keeping yaw) and return first collision-free pose.
        Falls back to benchmark pose if search exhausts."""
        # Save qpos/qvel to restore after probing.
        data = self.task.env.current_data
        qpos_backup = data.qpos.copy()
        qvel_backup = data.qvel.copy()

        candidates: list[tuple[float, float, float]] = [(wx, wy, wyaw)]
        # Spiral: radii 0.1..1.0m, 8 angles each.
        n_angles = 8
        for r in np.linspace(0.1, 1.0, 10):
            for k in range(n_angles):
                ang = 2 * math.pi * k / n_angles
                candidates.append(
                    (wx + r * math.cos(ang), wy + r * math.sin(ang), wyaw)
                )

        result = (wx, wy, wyaw)
        for cx, cy, cyaw in candidates:
            jx, jy, jt = self._world_to_joint(cx, cy, cyaw)
            self._write_joint_qpos(jx, jy, jt)
            if not self._base_in_collision():
                result = (cx, cy, cyaw)
                break
        # Restore — actual nav drive happens in get_action loop.
        data.qpos[:] = qpos_backup
        data.qvel[:] = qvel_backup
        mujoco.mj_forward(self.task.env.current_model, data)
        return result

    # The mobile_franka base body is now added with pos=[0, 0] and identity
    # quat (Abhay's fix in task_sampler.py), so joint qpos values equal the
    # body's world pose directly. Conversions reduce to identity.
    _BODY_INITIAL_Y = 0.0
    _BODY_INITIAL_YAW = 0.0

    @classmethod
    def _world_to_joint(cls, wx: float, wy: float, wyaw: float) -> tuple[float, float, float]:
        return wx, wy, wyaw

    @classmethod
    def _joint_to_world(cls, jx: float, jy: float, jt: float) -> tuple[float, float, float]:
        return jx, jy, jt

    def _current_xy_yaw(self) -> np.ndarray:
        # Read directly from mj_data world body pose to dodge the misleading
        # robot_view.base.pose getter (returns joint qpos as if it were world).
        model = self.task.env.current_model
        data = self.task.env.current_data
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot_0/base")
        wx, wy = float(data.xpos[bid][0]), float(data.xpos[bid][1])
        # Yaw from body world rotation matrix row.
        xmat = data.xmat[bid].reshape(3, 3)
        cyaw = math.atan2(float(xmat[1, 0]), float(xmat[0, 0]))
        return np.array([wx, wy, cyaw], dtype=np.float64)

    def get_action(self, observation):
        if not self._resolve_target():
            return None

        current = self._current_xy_yaw()
        target = self._target_xy_yaw
        dx = target[0] - current[0]
        dy = target[1] - current[1]
        dyaw = ((target[2] - current[2] + math.pi) % (2 * math.pi)) - math.pi

        if (
            abs(dx) < self.POS_TOLERANCE
            and abs(dy) < self.POS_TOLERANCE
            and abs(dyaw) < self.YAW_TOLERANCE
        ):
            if not self._reached:
                log.info(
                    f"[NavToOriginal] reached in {self._step} steps "
                    f"(err xy=({dx:+.4f},{dy:+.4f}), yaw={dyaw:+.4f})"
                )
                self._reached = True
            # Hold at target via actuator ctrl (no direct qpos write so
            # contacts are still enforced).
            return {"base": [target[0], target[1], target[2]], "done": True}

        # Ramp the actuator target one step closer to the goal. No direct
        # joint qpos write: physics drives the base, so walls/objects can
        # block the path and contacts get resolved by the constraint solver.
        step_wx = current[0] + np.clip(dx, -self.POS_STEP, self.POS_STEP)
        step_wy = current[1] + np.clip(dy, -self.POS_STEP, self.POS_STEP)
        step_wyaw = current[2] + np.clip(dyaw, -self.YAW_STEP, self.YAW_STEP)
        self._step += 1
        return {"base": [step_wx, step_wy, step_wyaw], "done": False}

    def _write_joint_qpos(self, jx: float, jy: float, jt: float) -> None:
        data = self.task.env.current_data
        model = self.task.env.current_model
        for jname, val in zip(
            ("robot_0/base_x", "robot_0/base_y", "robot_0/base_theta"),
            (jx, jy, jt),
        ):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                continue
            data.qpos[model.jnt_qposadr[jid]] = val
            data.qvel[model.jnt_dofadr[jid]] = 0.0
        mujoco.mj_forward(model, data)
