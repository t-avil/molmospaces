"""Combined nav-then-pick policy for mobile_franka.

Drives the planar base to env.original_robot_base_pose via the
NavToOriginalBasePolicy demonstrator, then runs the scripted PickPlannerPolicy.

Trajectory planning still happens at the snap-back pose (so IK feasibility +
target_poses are correct), but we restore the perturbed base pose right after
planning so the episode actually has to navigate there.
"""

from __future__ import annotations

import logging
from typing import Any

import mujoco
import numpy as np

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.policy.solvers.navigation.nav_to_original_policy import (
    NavToOriginalBasePolicy,
)
from molmo_spaces.policy.solvers.object_manipulation.pick_planner_policy import (
    PickPlannerPolicy,
)
from molmo_spaces.tasks.task import BaseMujocoTask

log = logging.getLogger(__name__)


class NavThenPickPolicy(PickPlannerPolicy):
    def __init__(self, config: MlSpacesExpConfig, task: BaseMujocoTask) -> None:
        super().__init__(config, task)
        self._nav: NavToOriginalBasePolicy | None = None
        self._nav_done: bool = False

    def reset(self, reset_retries: bool = True) -> None:
        # Build nav helper now that env/task are set up.
        if self._nav is None:
            self._nav = NavToOriginalBasePolicy(self.config, self.task)
        self._nav.reset()
        self._nav_done = False

        # Do NOT call super().reset() yet — pick trajectory must be planned
        # *after* nav drives the base to the original pose, so start_ee_pose
        # and IK feasibility reflect the actual nav-end state.
        self.action_primitives = []
        self.action_idx = 0
        if reset_retries:
            self.retry_count = 0
        self.sequential_ik_failures = 0
        # Sensors read self.target_poses["grasp"]; populate placeholders that
        # get overwritten by super().reset() when nav completes.
        self.target_poses = {
            "pregrasp": np.eye(4),
            "grasp": np.eye(4),
            "lift": np.eye(4),
        }

    def get_phase(self) -> str:
        if not self._nav_done:
            return "nav-to-original"
        return super().get_phase()

    def _check_for_failures(self) -> bool:
        if not self._nav_done:
            return False
        return super()._check_for_failures()

    def get_action(self, observation: Any) -> dict[str, Any]:
        if not self._nav_done:
            action = self._nav.get_action(observation)
            if action is None:
                # No original pose stashed -> skip nav, run pick directly.
                self._nav_done = True
                super().reset(reset_retries=False)
                return super().get_action(observation)
            if action.get("done"):
                self._nav_done = True
                log.info("[NavThenPick] nav phase complete, planning pick now")
                # Plan trajectory now that base is at the original pose. This
                # populates self.action_primitives / target_poses and captures
                # start_ee_pose at the actual nav-end state.
                super().reset(reset_retries=False)
            # Don't pass nav's "done" sentinel up to the eval loop.
            action.pop("done", None)
            # Hold arm + gripper during nav so gravity doesn't drop the planned
            # start pose.
            noop = self.robot_view.get_noop_ctrl_dict()
            for k, v in noop.items():
                action.setdefault(k, v)
            return action

        return super().get_action(observation)
