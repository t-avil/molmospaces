"""Resumable, multi-GPU MolmoBot sweep for the static-vs-mobile comparison.

End goal: how much worse MolmoBot grasps on the MOBILE franka (point->nav->molmobot
hybrid) vs the SAME cases on the STATIC DROID franka (MolmoBot driving directly).

Two modes:
  --mode mobile : MobileFrankaHybridMolmobotEvalConfig in the molmospaces venv.
                  Needs a serve_molmo.py server per shard; this script launches
                  one server per GPU (port 8000+gpu) and points each eval shard
                  at its own server via MLSPACES_MB_PORT. (One server per shard is
                  required: the served policy is stateful, so concurrent clients
                  would corrupt each other's obs-history.)
  --mode static : olmo.eval.configure_molmo_spaces:FrankaState8ClampAbsPosConfig in
                  the MolmoBot venv, model loaded in-process. No server.

Sharding: the remaining (not-yet-done) episodes are split across the chosen GPUs,
one eval_main process per GPU, each pinned with CUDA_VISIBLE_DEVICES.

Resume: per-episode results append to <out>/<mode>_results.jsonl, keyed by the
STABLE identity (house_index, traj_key) from the source benchmark. Re-running
skips done episodes; safe to kill and restart (cron-friendly).

Result collection: the pipeline saves each house's episodes as h5 groups named
traj_<N>, where N is the per-house ordinal WITHIN THAT eval run (not a global id).
So each round we map (house, run-local-ordinal) -> source traj_key to record the
stable key. Each round writes to its own eval dir to avoid cross-round aliasing.

Be nice to other users: pass --gpus explicitly and leave some free.

Usage:
  python scripts/molmobot_sweep.py --mode mobile \
      --benchmark_dir curate_out_full/FrankaPickDroidMiniBench_mobile_v1 \
      --out /tmp/sweep_mobile --gpus 0,1,2,3 --task_horizon_sec 150 --per_shard 12
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import time
from pathlib import Path

log = logging.getLogger("mbsweep")

REPO = Path(__file__).resolve().parent.parent
MOBILE_PY = REPO / ".venv" / "bin" / "python"
MOLMOBOT_PY = Path("/tmp/MolmoBot/MolmoBot/.venv/bin/python")
MOLMOBOT_DIR = Path("/tmp/MolmoBot/MolmoBot")
CKPT = MOLMOBOT_DIR / "ckpts" / "molmobot" / "MolmoBot-DROID"

# MolmoBot-Pi0 lives in its own env (openpi); checkpoint cached under HF_HOME.
PI_DIR = Path("/tmp/MolmoBot/MolmoBot-Pi0")
PI_PY = PI_DIR / ".venv" / "bin" / "python"
HF_HOME = "/tmp/hf-cache"


def pi_ckpt() -> str:
    import glob
    snaps = glob.glob(f"{HF_HOME}/hub/models--allenai--MolmoBot-Pi0-DROID/snapshots/*")
    if not snaps:
        raise FileNotFoundError("MolmoBot-Pi0-DROID checkpoint not found in HF cache")
    return sorted(snaps)[-1]


MODE_CFG = {
    "mobile": "molmo_spaces.evaluation.configs.mobile_franka_eval_configs:MobileFrankaHybridMolmobotEvalConfig",
    "pi05_mobile": "molmo_spaces.evaluation.configs.mobile_franka_eval_configs:MobileFrankaHybridPi05EvalConfig",
    # APPLES-TO-APPLES static cells: SAME molmospaces hybrid harness + SAME served
    # grasp module as the mobile modes, but on the FIXED DROID franka (no floating
    # base, no perturbation). These are the correct "easiest" baseline; they are
    # directly comparable to mobile/pi05_mobile (unlike the standalone-runner
    # "static"/"pi05_static" modes below, which use a different harness/venv).
    "molmobot_static": "molmo_spaces.evaluation.configs.mobile_franka_eval_configs:FrankaHybridMolmobotStaticEvalConfig",
    "pi05_static_hybrid": "molmo_spaces.evaluation.configs.mobile_franka_eval_configs:FrankaHybridPi05StaticEvalConfig",
    "static": "olmo.eval.configure_molmo_spaces:FrankaState8ClampAbsPosConfig",
}

# Static-hybrid modes that share the served-grasp + molmospaces-harness path with
# the mobile modes. Centralised so the launch logic stays in one place.
SERVED_HYBRID_MODES = {"mobile", "pi05_mobile", "molmobot_static", "pi05_static_hybrid"}
# Of those, which serve molmobot (vs pure pi0.5). Drives launch_server's branch.
MOLMOBOT_SERVE_MODES = {"mobile", "molmobot_static"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["mobile", "pi05_mobile", "molmobot_static", "pi05_static_hybrid", "static", "pi_static", "pi05_static", "pi_mobile"], required=True)
    p.add_argument("--benchmark_dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--gpus", default="0,1,2,3", help="comma-separated GPU ids to use")
    p.add_argument("--per_shard", type=int, default=12,
                   help="episodes per GPU per round (smaller = more frequent checkpointing)")
    p.add_argument("--task_horizon_sec", type=int, default=150)
    p.add_argument("--limit", type=int, default=None, help="cap total episodes (smoke tests)")
    p.add_argument("--keep_eval_dirs", action="store_true",
                   help="keep per-round eval output (default: delete after collecting to save disk)")
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def src_id(ep: dict) -> tuple[int, str]:
    """Stable per-episode identity: (house_index, source traj_key)."""
    return (int(ep["house_index"]), str(ep["source"]["traj_key"]))


def shard_local_map(src: list[dict], idxs: list[int]) -> dict[tuple[int, int], tuple[int, str]]:
    """Map (house, run-local per-house ordinal) -> stable src_id, in shard order.
    The eval names h5 groups traj_<ordinal> per house in benchmark order, so this
    reproduces the names it will save."""
    counter: dict[int, int] = {}
    out: dict[tuple[int, int], tuple[int, str]] = {}
    for i in idxs:
        h = int(src[i]["house_index"])
        e = counter.get(h, 0)
        out[(h, e)] = src_id(src[i])
        counter[h] = e + 1
    return out


def write_shard_benchmark(src: list[dict], idxs: list[int], bench_dir: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "benchmark.json").write_text(json.dumps([src[i] for i in idxs]))
    meta = bench_dir / "benchmark_metadata.json"
    if meta.exists():
        shutil.copy2(meta, dst / "benchmark_metadata.json")


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex(("127.0.0.1", port)) == 0


def launch_server(gpu: int, port: int, log_path: Path, mode: str = "mobile") -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["MUJOCO_GL"] = "egl"
    if mode not in MOLMOBOT_SERVE_MODES:
        # Serve PURE pi0.5 (pi05_droid) speaking the same websocket protocol as molmo,
        # so the hybrid grasp client calls it unchanged. Runs in the MolmoBot-Pi0 venv.
        env["POLICY_SERVER_PORT"] = str(port)
        env["OPENPI_DATA_HOME"] = "/tmp/openpi-cache"
        env["HF_HOME"] = HF_HOME
        cmd = [str(PI_PY), "serve_pi05.py"]
        return subprocess.Popen(cmd, cwd=str(PI_DIR), env=env,
                                stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
    cmd = [str(MOLMOBOT_PY), "launch_scripts/serve_molmo.py",
           "--local-path", str(CKPT), "--action-type", "joint_pos", "--port", str(port)]
    return subprocess.Popen(cmd, cwd=str(MOLMOBOT_DIR), env=env,
                            stdout=open(log_path, "w"), stderr=subprocess.STDOUT)


def launch_shard(mode: str, gpu: int, port: int, bench_dir: Path, out_dir: Path,
                 horizon: int, log_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["MUJOCO_GL"] = "egl"
    env["PYOPENGL_PLATFORM"] = "egl"
    env["PYTHONUNBUFFERED"] = "1"
    if mode in SERVED_HYBRID_MODES:
        # served VLA via the molmospaces hybrid; horizon in SECONDS. mobile -> molmobot,
        # pi05_mobile -> pure pi0.5 (same hybrid + protocol, different served model + dt).
        # *_static modes use the SAME harness/config family on the FIXED franka, so the
        # only difference vs the mobile cell is the base (no floating base, no perturbation).
        # MLSPACES_PERTURB_SCALE (if set in the parent env) flows through to the sampler;
        # for the fixed-base static modes the sampler ignores it (perturbation is mobile-only).
        env["MLSPACES_MB_PORT"] = str(port)
        # #1 parallelism: MLSPACES_EVAL_WORKERS concurrent episodes per GPU (one shared
        # server multiplexes them safely via per-client state save/restore). Default 1.
        nworkers = os.environ.get("MLSPACES_EVAL_WORKERS", "1")
        cmd = [str(MOBILE_PY), "molmo_spaces/evaluation/eval_main.py", MODE_CFG[mode],
               "--benchmark_dir", str(bench_dir), "--task_horizon_sec", str(horizon),
               "--no_wandb", "--num_workers", str(nworkers), "--output_dir", str(out_dir)]
        cwd = str(REPO)
    elif mode == "pi_static":
        # MolmoBot-Pi0 in-process on the fixed franka via the PI env runner.
        env["HF_HOME"] = HF_HOME
        cmd = [str(PI_PY), "run_pi_static.py", "--benchmark_dir", str(bench_dir),
               "--ckpt", pi_ckpt(), "--output_dir", str(out_dir),
               "--task_horizon_sec", str(horizon)]
        cwd = str(PI_DIR)
    elif mode == "pi05_static":
        # PURE pi0.5 (Physical Intelligence pi05_droid) in-process on the fixed franka.
        env["HF_HOME"] = HF_HOME
        env["OPENPI_DATA_HOME"] = "/tmp/openpi-cache"
        cmd = [str(PI_PY), "run_pi05_static.py", "--benchmark_dir", str(bench_dir),
               "--output_dir", str(out_dir), "--task_horizon_sec", str(horizon)]
        cwd = str(PI_DIR)
    else:
        # in-process VLA on the fixed DROID franka via MolmoBot's run_eval.py
        # (it injects checkpoint_path into the config). task_horizon is in STEPS;
        # convert from seconds at the 200ms control dt (5 steps/sec) to match the
        # mobile run's sim-time budget.
        steps = horizon * 5
        cmd = [str(MOLMOBOT_PY), "launch_scripts/run_eval.py",
               "--checkpoint_path", str(CKPT),
               "--benchmark_path", str(bench_dir),
               "--eval_config_cls", MODE_CFG["static"],
               "--task_horizon", str(steps),
               "--num_workers", "1", "--output_dir", str(out_dir)]
        cwd = str(MOLMOBOT_DIR)
    return subprocess.Popen(cmd, cwd=cwd, env=env,
                            stdout=open(log_path, "w"), stderr=subprocess.STDOUT)


_DONE_RE = re.compile(r"house (\d+) episode (\d+).*completed with success=(True|False)")


def collect(log_path: Path, local_to_id: dict[tuple[int, int], tuple[int, str]],
            results_log: Path) -> None:
    """Score by the AUTHORITATIVE pipeline metric (the 'completed with success=' log
    line), keyed by (house, run-local episode index), mapped to the stable src_id.
    NOTE: do NOT use the h5 `success` array's .max() — that is a lenient per-step
    proximity flag and overcounts (it diverged from the official metric by 167/347
    on the static franka). Episodes with no completion line are marked errored."""
    success_by_local: dict[tuple[int, int], bool] = {}
    try:
        with open(log_path, errors="ignore") as fh:
            for line in fh:
                m = _DONE_RE.search(line)
                if m:
                    success_by_local[(int(m.group(1)), int(m.group(2)))] = (m.group(3) == "True")
    except FileNotFoundError:
        pass
    already = {(int(r["house_index"]), str(r["traj_key"])) for r in load_jsonl(results_log)}
    with results_log.open("a") as f:
        for local, sid in local_to_id.items():
            if sid in already:
                continue
            h, traj = sid
            if local in success_by_local:
                f.write(json.dumps({"house_index": h, "traj_key": traj,
                                    "success": success_by_local[local], "errored": False}) + "\n")
            else:
                f.write(json.dumps({"house_index": h, "traj_key": traj,
                                    "success": False, "errored": True}) + "\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    gpus = [int(g) for g in args.gpus.split(",") if g.strip() != ""]
    args.out.mkdir(parents=True, exist_ok=True)
    results_log = args.out / f"{args.mode}_results.jsonl"

    src = json.load((args.benchmark_dir / "benchmark.json").open())
    if args.limit:
        src = src[: args.limit]
    ids = [src_id(ep) for ep in src]
    log.info(f"[{args.mode}] benchmark has {len(src)} episodes; GPUs={gpus}")

    round_id = 0
    while True:
        processed = {(int(r["house_index"]), str(r["traj_key"])) for r in load_jsonl(results_log)}
        todo = [i for i in range(len(src)) if ids[i] not in processed]
        log.info(f"[{args.mode}] round {round_id}: {len(processed)} done, {len(todo)} remaining")
        if not todo:
            break

        batch = todo[: len(gpus) * args.per_shard]
        shards: dict[int, list[int]] = {g: [] for g in gpus}
        for j, i in enumerate(batch):
            shards[gpus[j % len(gpus)]].append(i)
        shards = {g: idxs for g, idxs in shards.items() if idxs}

        if args.mode in SERVED_HYBRID_MODES:
            for g in shards:
                port = 8000 + g
                if not port_open(port):
                    log.info(f"  launching {args.mode} server gpu{g} :{port}")
                    launch_server(g, port, args.out / f"server_gpu{g}.log", mode=args.mode)
            deadline = time.time() + 420  # pi0.5 jax/orbax load is slower to come up
            for g in shards:
                while not port_open(8000 + g) and time.time() < deadline:
                    time.sleep(3)
                if not port_open(8000 + g):
                    log.error(f"  server :{8000+g} did not come up; skipping gpu{g} this round")

        procs: dict[int, subprocess.Popen] = {}
        maps: dict[int, dict] = {}
        out_dirs: dict[int, Path] = {}
        for g, idxs in shards.items():
            port = 8000 + g
            if args.mode in SERVED_HYBRID_MODES and not port_open(port):
                continue
            sb = args.out / f"r{round_id}_gpu{g}_bench"
            write_shard_benchmark(src, idxs, args.benchmark_dir, sb)
            out_dirs[g] = args.out / f"r{round_id}_gpu{g}_eval"
            maps[g] = shard_local_map(src, idxs)
            log.info(f"  shard gpu{g}: {len(idxs)} eps -> :{port}")
            procs[g] = launch_shard(args.mode, g, port, sb, out_dirs[g],
                                    args.task_horizon_sec, args.out / f"r{round_id}_gpu{g}.log")

        if not procs:
            log.error("no shards launched this round (servers down?); aborting")
            break

        for g, p in procs.items():
            rc = p.wait()
            log.info(f"  shard gpu{g} exit={rc}")
            collect(args.out / f"r{round_id}_gpu{g}.log", maps[g], results_log)
            shutil.rmtree(args.out / f"r{round_id}_gpu{g}_bench", ignore_errors=True)
            if not args.keep_eval_dirs:
                shutil.rmtree(out_dirs[g], ignore_errors=True)
        round_id += 1

    rs = load_jsonl(results_log)
    ok = sum(1 for r in rs if r["success"])
    err = sum(1 for r in rs if r.get("errored"))
    print(f"\n[{args.mode}] DONE: {ok}/{len(rs)} success "
          f"({100*ok/max(len(rs),1):.1f}%), {err} errored. -> {results_log}")


if __name__ == "__main__":
    main()
