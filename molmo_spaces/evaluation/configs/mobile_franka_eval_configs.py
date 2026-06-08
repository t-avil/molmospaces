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
    FrankaRobotConfig,
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
        # Allow per-shard server targeting for multi-GPU sweeps: each shard runs
        # its own serve_molmo.py on a distinct port (8000 + gpu) and sets MLSPACES_MB_PORT.
        _port = os.environ.get("MLSPACES_MB_PORT")
        if _port:
            self.policy_config.molmobot_port = int(_port)


class MobileFrankaHybridPi05EvalConfig(MobileFrankaHybridPointGraspEvalConfig):
    """Same hybrid (point -> approach -> grasp), but the GRASP module is a served
    PURE pi0.5 (Physical Intelligence pi05_droid) VLA instead of molmobot. The served
    pi0.5 policy speaks the identical websocket protocol (obs {exo_camera_1, wrist_camera,
    qpos, task} -> {arm, gripper}), so grasp_mode="molmobot" routes to it unchanged; only
    the served model and the control rate differ. Requires serve_pi05.py on
    molmobot_host:molmobot_port. Run with MUJOCO_GL=egl.
    """

    policy_config: HybridPointGraspPolicyConfig = HybridPointGraspPolicyConfig(
        grasp_mode="molmobot",  # protocol-identical; the served model is pi0.5
        approach_target="original",  # park at the training base pose to isolate the grasp
    )
    policy_dt_ms: float = 66.0  # pi0.5-DROID control rate (vs molmobot's 200 ms)

    @property
    def tag(self) -> str:
        return "mobile_franka_hybrid_pi05_json_benchmark"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        os.environ["MLSPACES_HIDE_MOBILE_BASE_VIS"] = "1"
        _port = os.environ.get("MLSPACES_MB_PORT")
        if _port:
            self.policy_config.molmobot_port = int(_port)


class FrankaHybridMolmobotStaticEvalConfig(MobileFrankaHybridMolmobotEvalConfig):
    """TRUE-STATIC counterpart of MobileFrankaHybridMolmobotEvalConfig.

    Same hybrid pipeline (point -> approach -> served-VLA grasp) and the SAME
    served molmobot grasp module (grasp_mode="molmobot"), but the base is the
    standard FIXED DROID franka (FrankaRobotConfig), not the floating mobile
    base. Two consequences make this the apples-to-apples "easiest" cell:

      * Fixed base = a welded mocap base (FrankaFR3BaseGroup, zero actuators).
        There is NO floating-base contact instability, and the robot sits
        exactly at the benchmark-authored pose.
      * JsonEvalTaskSampler only perturbs the base when robot_config is a
        MobileFrankaRobotConfig, so a fixed FrankaRobotConfig gets NO
        perturbation at all (independent of MLSPACES_PERTURB_SCALE).

    The hybrid approach phase is a no-op here: HybridPointGraspPolicy detects a
    non-mobile base (robot_view.base.is_mobile == False) and hands straight to
    the grasp from the authored pose. So this isolates exactly the served grasp
    module on the original fixed-base protocol:

        static (this)  >  mobile @ static pose (scale=0)  >  mobile + perturbed

    Run with MUJOCO_GL=egl so exo_camera_1 + wrist_camera render into the obs.
    """

    # Fixed DROID franka. base_size matches the mobile config's arm-origin world
    # height (0.09141114843014408 + 0.58) so reach/camera geometry is identical;
    # only the base *dynamics* (welded vs floating) differ.
    robot_config: FrankaRobotConfig = FrankaRobotConfig(
        base_size=[0.5, 0.5, 0.585],  # match mobile@static arm/camera height (was 0.671: double-counted rbp.z -> +8.64cm viewpoint shift -> pi0.5 0%)
    )

    @property
    def tag(self) -> str:
        return "franka_hybrid_molmobot_static_json_benchmark"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        # Fixed base draws no mobile-base column, so the hide-vis hack is moot;
        # leave it set (harmless) for parity with the mobile config's env.


class FrankaHybridPi05StaticEvalConfig(MobileFrankaHybridPi05EvalConfig):
    """TRUE-STATIC counterpart of MobileFrankaHybridPi05EvalConfig.

    Identical to FrankaHybridMolmobotStaticEvalConfig (fixed DROID franka, no
    perturbation, approach is a no-op) but the served grasp module is a PURE
    pi0.5 (pi05_droid) speaking the same websocket protocol, at the pi0.5
    control rate. Requires serve_pi05.py on molmobot_host:molmobot_port.
    Run with MUJOCO_GL=egl.
    """

    robot_config: FrankaRobotConfig = FrankaRobotConfig(
        base_size=[0.5, 0.5, 0.585],  # match mobile@static arm/camera height (was 0.671: double-counted rbp.z -> +8.64cm viewpoint shift -> pi0.5 0%)
    )

    @property
    def tag(self) -> str:
        return "franka_hybrid_pi05_static_json_benchmark"
