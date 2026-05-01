"""
Eval config that runs the scripted PickPlannerPolicy on a JSON benchmark.

PickPlannerPolicy is the same scripted IK planner used to generate the
FrankaPickDroidMiniBench data, so it should solve most of these episodes.
Useful as a "real action" demo to confirm the eval pipeline produces
non-zero success rates end-to-end.

Usage:
    python molmo_spaces/evaluation/eval_main.py \\
        molmo_spaces.evaluation.configs.pick_planner_eval_configs:PickPlannerBenchmarkEvalConfig \\
        --benchmark_dir assets/benchmarks/molmospaces-bench-v1/procthor-10k/FrankaPickDroidMiniBench/FrankaPickDroidMiniBench_json_benchmark_20251231 \\
        --idx 0 \\
        --task_horizon_sec 20 \\
        --no_wandb
"""

from __future__ import annotations

from molmo_spaces.configs.policy_configs import PickPlannerPolicyConfig
from molmo_spaces.configs.robot_configs import ActionNoiseConfig, FrankaRobotConfig
from molmo_spaces.evaluation.configs.evaluation_configs import JsonBenchmarkEvalConfig


class PickPlannerBenchmarkEvalConfig(JsonBenchmarkEvalConfig):
    """JSON benchmark eval with stationary Franka + scripted PickPlanner."""

    seed: int = 42
    # Match the rate the planner was tuned at during datagen.
    policy_dt_ms: float = 66.0
    end_on_success: bool = True

    robot_config: FrankaRobotConfig = FrankaRobotConfig()
    policy_config: PickPlannerPolicyConfig = PickPlannerPolicyConfig()

    use_filament: bool = False

    @property
    def tag(self) -> str:
        return "pick_planner_json_benchmark"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        # Deterministic for testing.
        self.robot_config.action_noise_config = ActionNoiseConfig(enabled=False)
