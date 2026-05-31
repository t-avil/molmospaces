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

import os

from molmo_spaces.configs.policy_configs import (
    HybridPointGraspPolicyConfig,
    NavThenPickPolicyConfig,
    NavToOriginalBasePolicyConfig,
    PickPlannerPolicyConfig,
)
from molmo_spaces.configs.policy_configs_baselines import TeleopPolicyConfig
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
        base_size=[0.5, 0.5, 0.09141114843014408 + .58],
    )
    policy_config: PickPlannerPolicyConfig = PickPlannerPolicyConfig()

    use_filament: bool = False

    @property
    def tag(self) -> str:
        return "mobile_franka_planner_json_benchmark"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.robot_config.action_noise_config = ActionNoiseConfig(enabled=False)


class MobileFrankaNavToOriginalEvalConfig(JsonBenchmarkEvalConfig):
    """JSON benchmark eval with mobile_franka + nav-to-original policy.

    JsonEvalTaskSampler perturbs the base around the benchmark pose and
    stashes the original on env. This eval runs a tiny policy that drives
    the planar base back to that original pose via base actuators (no IK
    involvement). Fallback while mobile-franka per-step IK is still broken.
    """

    seed: int = 42
    policy_dt_ms: float = 66.0
    end_on_success: bool = True
    task_horizon: int = 1500  # ~100s @ 66ms; base nav usually converges in <300
    use_passive_viewer: bool = False

    robot_config: MobileFrankaRobotConfig = MobileFrankaRobotConfig(
        base_size=[0.5, 0.5, 0.09141114843014408],
    )
    policy_config: NavToOriginalBasePolicyConfig = NavToOriginalBasePolicyConfig()

    use_filament: bool = False

    @property
    def tag(self) -> str:
        return "mobile_franka_nav_to_original_json_benchmark"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.robot_config.action_noise_config = ActionNoiseConfig(enabled=False)


class MobileFrankaNavThenPickEvalConfig(JsonBenchmarkEvalConfig):
    """Mobile_franka: drive base via NavToOriginalBasePolicy demonstrator,
    then run the scripted PickPlannerPolicy. Visible nav + pick in one episode.
    """

    seed: int = 42
    policy_dt_ms: float = 66.0
    end_on_success: bool = True
    task_horizon: int = 1500
    use_passive_viewer: bool = False

    robot_config: MobileFrankaRobotConfig = MobileFrankaRobotConfig(
        base_size=[0.5, 0.5, 0.6714111484301441],
    )
    policy_config: NavThenPickPolicyConfig = NavThenPickPolicyConfig()

    use_filament: bool = False

    @property
    def tag(self) -> str:
        return "mobile_franka_nav_then_pick_json_benchmark"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.robot_config.action_noise_config = ActionNoiseConfig(enabled=False)


class MobileFrankaTeleopEvalConfig(JsonBenchmarkEvalConfig):
    """JSON benchmark eval with mobile_franka + keyboard teleop.

    Lets you drive the arm by hand to confirm the IK frame mismatch is real:
    arrows / w / s for translation, a/d/e/r/z/c for rotation, space toggles
    gripper. Each keystroke triggers FrankaKinematics.ik() on a small TCP
    delta — if the mobile-base frame transform is wrong, the IK either
    silently fails or produces wrong joint targets.
    """

    seed: int = 42
    policy_dt_ms: float = 40.0
    use_passive_viewer: bool = True

    robot_config: MobileFrankaRobotConfig = MobileFrankaRobotConfig(
        base_size=[0.5, 0.5, 0.09141114843014408],
    )
    policy_config: TeleopPolicyConfig = TeleopPolicyConfig(device="keyboard")

    use_filament: bool = False

    @property
    def tag(self) -> str:
        return "mobile_franka_teleop_json_benchmark"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.robot_config.action_noise_config = ActionNoiseConfig(enabled=False)


class MobileFrankaHybridPointGraspEvalConfig(JsonBenchmarkEvalConfig):
    """POC perception->grasp eval for mobile_franka:
    point (external cam) -> deproject (depth+intrinsics/extrinsics) -> approach
    -> scripted grasp. Per-episode base height applies via the task_sampler fix.

    pointing_mode defaults to "ground_truth" so the loop runs with no served
    model; set policy_config.pointing_mode="molmo" (+ host/port) for real Molmo.
    Note: the "molmo" path needs an exocentric camera with record_depth=True.
    """

    seed: int = 42
    policy_dt_ms: float = 66.0
    end_on_success: bool = True
    task_horizon: int = 1500
    use_passive_viewer: bool = False

    robot_config: MobileFrankaRobotConfig = MobileFrankaRobotConfig(
        base_size=[0.5, 0.5, 0.6714111484301441],
    )
    policy_config: HybridPointGraspPolicyConfig = HybridPointGraspPolicyConfig()

    use_filament: bool = False

    @property
    def tag(self) -> str:
        return "mobile_franka_hybrid_point_grasp_json_benchmark"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.robot_config.action_noise_config = ActionNoiseConfig(enabled=False)


class MobileFrankaHybridMolmobotEvalConfig(MobileFrankaHybridPointGraspEvalConfig):
    """Hybrid POC with the GRASP delegated to a served molmobot VLA
    (point -> approach -> molmobot). Requires serve_molmo.py running on
    molmobot_host:molmobot_port. Pointing stays ground_truth to isolate the grasp.
    Run with MUJOCO_GL=egl so exo_camera_1 + wrist_camera render into the obs.
    """

    policy_config: HybridPointGraspPolicyConfig = HybridPointGraspPolicyConfig(
        grasp_mode="molmobot",
        approach_target="original",  # park at the training base pose to isolate the grasp
    )
    policy_dt_ms: float = 200.0  # match molmobot's trained control rate

    @property
    def tag(self) -> str:
        return "mobile_franka_hybrid_molmobot_json_benchmark"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        # Hide the mobile-base visual column so the exo camera matches the
        # fixed-DROID-franka distribution molmobot trained on (physics unchanged).
        os.environ["MLSPACES_HIDE_MOBILE_BASE_VIS"] = "1"
