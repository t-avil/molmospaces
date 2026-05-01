"""
Eval config for the mobile_franka robot against a JSON benchmark with the
scripted PickPlannerPolicy.

Usage:
    python molmo_spaces/evaluation/eval_main.py \\
        molmo_spaces.evaluation.configs.mobile_franka_eval_configs:MobileFrankaPlannerEvalConfig \\
        --benchmark_dir <path/to/json/benchmark> \\
        --idx 0 \\
        --task_horizon_sec 20 \\
        --no_wandb
"""

from __future__ import annotations

from molmo_spaces.configs.policy_configs import PickPlannerPolicyConfig
from molmo_spaces.configs.robot_configs import (
    ActionNoiseConfig,
    MobileFrankaRobotConfig,
)
from molmo_spaces.evaluation.configs.evaluation_configs import JsonBenchmarkEvalConfig


class MobileFrankaPlannerEvalConfig(JsonBenchmarkEvalConfig):
    """JSON benchmark eval with mobile_franka + scripted PickPlannerPolicy.

    The scripted policy snaps the mobile base back to the benchmark-authored
    `task.robot_base_pose` (which JsonEvalTaskSampler stashes on the env as
    `original_robot_base_pose` before applying its random perturbation), then
    runs the standard pick sequence with a fixed-base IK solver.
    """

    seed: int = 42
    # Match the rate the scripted planner was tuned at during datagen.
    policy_dt_ms: float = 66.0
    end_on_success: bool = True

    # Match the static franka's arm-origin world height (rbp.z = 0.091 in the
    # FrankaPickDroidMiniBench benchmark) so reachability is preserved.
    robot_config: MobileFrankaRobotConfig = MobileFrankaRobotConfig(
        base_size=[0.5, 0.5, 0.09141114843014408],
    )
    policy_config: PickPlannerPolicyConfig = PickPlannerPolicyConfig()

    use_filament: bool = False

    @property
    def tag(self) -> str:
        return "mobile_franka_planner_json_benchmark"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.robot_config.action_noise_config = ActionNoiseConfig(enabled=False)
