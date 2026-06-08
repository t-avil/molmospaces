"""PURE pi0.5 (Physical Intelligence pi05_droid) on the fixed franka.

Unlike run_pi_static.py (which loads the allenai/MolmoBot-Pi0-DROID checkpoint),
this loads the genuine pi0.5 DROID weights via the openpi `pi05_mlspaces_finetune`
config (weight_loader -> gs://openpi-assets/checkpoints/pi05_droid/params, pi05=True,
32-dim actions). Set OPENPI_DATA_HOME=/tmp/openpi-cache so weights resolve from the
pre-downloaded cache instead of re-hitting GCS.
"""
import argparse
from pathlib import Path

from molmo_spaces.evaluation import run_evaluation
from molmobot_pi0.eval.utils import PiPnPBenchmarkEvalConfig
from molmobot_pi0.eval.policies.pi import PiJointPosPolicy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_name", default="pi05_mlspaces_finetune")
    p.add_argument("--task_horizon_sec", type=int, default=150)
    a = p.parse_args()

    policy = PiJointPosPolicy(model_name=a.model_name, use_torch=False)  # pure pi0.5-DROID (JAX/orbax)
    policy.prepare_model()
    res = run_evaluation(
        PiPnPBenchmarkEvalConfig,
        Path(a.benchmark_dir),
        output_dir=a.output_dir,
        num_workers=1,
        preloaded_policy=policy,
        task_horizon_sec=a.task_horizon_sec,
    )
    print(f"PI05_STATIC success_rate={res.success_rate:.4f} total={res.total_count}")


if __name__ == "__main__":
    main()
