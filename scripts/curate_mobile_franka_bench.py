"""Build a Level-2 mobile_franka pick benchmark from FrankaPickDroidMiniBench.

Run the NavThenPick eval over every episode, log per-episode (success,
perturbed_pose, original_pose), and emit a new benchmark JSON that contains
only the episodes mobile_franka can complete. The new JSON stores the
perturbed pose as `task.robot_base_pose` and the original as
`task.original_robot_base_pose`, so downstream policies (pi-0.5, molmobot)
can reproduce the eval without re-perturbing.

Checkpoint format: JSONL keyed by (house_index, episode_idx). Re-running the
script skips already-processed episodes. Two side channels:

* `MLSPACES_CURATE_LOG`: written by JsonEvalTaskSampler when the perturbation
  block fires. Captures perturbed + original pose per episode.
* `<output_dir>/curate_results.jsonl`: written here. Captures success/fail.

Usage:
    python scripts/curate_mobile_franka_bench.py \
        --benchmark_dir assets/benchmarks/.../FrankaPickDroidMiniBench_json_benchmark_20251231 \
        --output_dir curated_output \
        --num_workers 16 \
        [--limit 1000]

Resume: re-run with same --output_dir. Already-processed (house, ep) pairs
are skipped automatically (filtered out of the benchmark before pipeline
starts).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger("curate")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark_dir", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument(
        "--config",
        default="molmo_spaces.evaluation.configs.mobile_franka_eval_configs:MobileFrankaNavThenPickEvalConfig",
    )
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument(
        "--chunk_size",
        type=int,
        default=50,
        help="Episodes per pipeline invocation (smaller = more checkpointing overhead, "
        "more responsive resume; larger = fewer subprocess spawns)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after processing this many episodes total. Useful for smoke tests.",
    )
    p.add_argument(
        "--task_horizon_sec",
        type=int,
        default=60,
        help="Per-episode task horizon (seconds of sim).",
    )
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_filtered_benchmark(
    bench_dir: Path, todo_idx: list[int], tmp_dir: Path
) -> Path:
    """Build a temporary benchmark JSON containing only the episodes whose
    indices appear in todo_idx, in that order. The pipeline iterates this
    file end-to-end, which is how we skip already-processed episodes
    without touching pipeline internals."""
    src = json.load((bench_dir / "benchmark.json").open())
    filtered = [src[i] for i in todo_idx]
    tmp_dir.mkdir(parents=True, exist_ok=True)
    (tmp_dir / "benchmark.json").write_text(json.dumps(filtered))
    meta_src = bench_dir / "benchmark_metadata.json"
    if meta_src.exists():
        shutil.copy2(meta_src, tmp_dir / "benchmark_metadata.json")
    return tmp_dir


def run_chunk(
    args: argparse.Namespace, todo_idx: list[int], chunk_id: int
) -> None:
    """Build a temp benchmark of just the todo episodes, hand it to eval_main."""
    tmp_bench = args.output_dir / f"chunk_{chunk_id:05d}_bench"
    write_filtered_benchmark(args.benchmark_dir, todo_idx, tmp_bench)

    curate_log = args.output_dir / "curate_perturbations.jsonl"
    results_log = args.output_dir / "curate_results.jsonl"

    env = os.environ.copy()
    env["MLSPACES_CURATE_LOG"] = str(curate_log.resolve())
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        sys.executable,
        "molmo_spaces/evaluation/eval_main.py",
        args.config,
        "--benchmark_dir",
        str(tmp_bench),
        "--task_horizon_sec",
        str(args.task_horizon_sec),
        "--no_wandb",
        "--num_workers",
        str(args.num_workers),
        "--output_dir",
        str((args.output_dir / "eval_runs").resolve()),
    ]
    log.info(f"chunk {chunk_id}: running {len(todo_idx)} eps, cmd: {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, env=env)
    dt = time.time() - t0
    log.info(f"chunk {chunk_id}: exit={proc.returncode} in {dt:.1f}s")

    # Pull this chunk's per-episode success status out of the eval output.
    # eval_main writes EpisodeResult JSON under output_dir/<config>/<ts>/
    # but we also gate on what landed in curate_perturbations.jsonl.
    collect_chunk_results(args, todo_idx, results_log)
    # Cleanup temp benchmark.
    shutil.rmtree(tmp_bench, ignore_errors=True)


def collect_chunk_results(
    args: argparse.Namespace, todo_idx: list[int], results_log: Path
) -> None:
    """Walk the most-recent eval_runs subdirectory; for each episode that has
    a saved trajectory, mark it processed in curate_results.jsonl. Episodes
    that errored (no save) are recorded as success=False with status=errored
    so we don't re-process them on resume."""
    eval_root = args.output_dir / "eval_runs"
    if not eval_root.exists():
        return
    # Find most recent subdir.
    runs = sorted([p for p in eval_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not runs:
        return
    config_dir = runs[-1]
    ts_dirs = sorted([p for p in config_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not ts_dirs:
        return
    run_dir = ts_dirs[-1]

    # Parse per-episode results from saved h5 trajectory files. The pipeline
    # writes one h5 per (house_id, ep_idx). Each ep with success comes through
    # as data_file_path; failures are still in the result list. The easiest
    # signal is `running_log.log` which contains lines like:
    #   pipeline.py:1051 Worker 0 house 102 episode 0 ... completed with success=True
    log_path = run_dir / "running_log.log"
    succeeded: set[tuple[int, int]] = set()
    errored: set[tuple[int, int]] = set()
    seen: set[tuple[int, int]] = set()

    def _parse_house_ep(line: str) -> tuple[int, int] | None:
        try:
            parts = line.split()
            h = int(parts[parts.index("house") + 1])
            e = int(parts[parts.index("episode") + 1])
            return (h, e)
        except Exception:
            return None

    if log_path.exists():
        with log_path.open() as f:
            for line in f:
                if "completed with success=" in line:
                    key = _parse_house_ep(line)
                    if key is None:
                        continue
                    seen.add(key)
                    if "success=True" in line:
                        succeeded.add(key)
                elif (
                    "task sampling error" in line
                    or "rollout error" in line
                    or "HouseInvalidForTask" in line
                ):
                    key = _parse_house_ep(line)
                    if key is None:
                        continue
                    seen.add(key)
                    errored.add(key)

    # Append per-episode records to results log.
    already_logged = {
        (int(r["house_index"]), int(r["episode_idx"]))
        for r in load_jsonl(results_log)
    }
    with results_log.open("a") as f:
        for (h, e) in seen:
            if (h, e) in already_logged:
                continue
            f.write(
                json.dumps(
                    {
                        "house_index": h,
                        "episode_idx": e,
                        "success": (h, e) in succeeded,
                        "errored": (h, e) in errored,
                    }
                )
                + "\n"
            )


def emit_level2_benchmark(args: argparse.Namespace) -> Path:
    """Combine curate_results.jsonl + curate_perturbations.jsonl with the
    source benchmark to produce a new self-contained benchmark JSON listing
    only the successful (house_index, episode_idx) episodes, with
    `task.robot_base_pose` set to the perturbed pose and
    `task.original_robot_base_pose` carrying the original."""
    src = json.load((args.benchmark_dir / "benchmark.json").open())
    # Same per-house-ordinal derivation as in main() — the source JSON does
    # not carry episode_idx, so episodes are identified positionally within
    # their house.
    counter: dict[int, int] = {}
    src_by_key: dict[tuple[int, int], tuple[int, dict]] = {}
    for i, ep in enumerate(src):
        h = int(ep["house_index"])
        e = counter.get(h, 0)
        src_by_key[(h, e)] = (i, ep)
        counter[h] = e + 1

    results = load_jsonl(args.output_dir / "curate_results.jsonl")
    perturbs = load_jsonl(args.output_dir / "curate_perturbations.jsonl")
    perturb_by_key = {
        (int(p["house_index"]), int(p["episode_idx"])): p for p in perturbs
    }

    out_episodes = []
    for r in results:
        if not r["success"]:
            continue
        key = (int(r["house_index"]), int(r["episode_idx"]))
        if key not in src_by_key:
            continue
        _, ep = src_by_key[key]
        p = perturb_by_key.get(key)
        if p is None:
            continue
        ep_out = json.loads(json.dumps(ep))  # deep copy
        ep_out["task"]["robot_base_pose"] = p["perturbed_robot_base_pose"]
        ep_out["task"]["original_robot_base_pose"] = p["original_robot_base_pose"]
        ep_out["task"]["mobile_franka_perturbation"] = {
            "dx": p["dx"],
            "dy": p["dy"],
            "dyaw": p["dyaw"],
        }
        out_episodes.append(ep_out)

    out_dir = args.output_dir / "FrankaPickDroidMiniBench_mobile_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark.json").write_text(json.dumps(out_episodes, indent=2))
    src_meta = args.benchmark_dir / "benchmark_metadata.json"
    if src_meta.exists():
        meta = json.load(src_meta.open())
        meta["curated_for"] = "mobile_franka"
        meta["source_benchmark"] = str(args.benchmark_dir)
        meta["n_episodes"] = len(out_episodes)
        (out_dir / "benchmark_metadata.json").write_text(json.dumps(meta, indent=2))
    log.info(f"Wrote curated benchmark with {len(out_episodes)} episodes to {out_dir}")
    return out_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    src = json.load((args.benchmark_dir / "benchmark.json").open())
    n_total = len(src)
    if args.limit:
        n_total = min(n_total, args.limit)
    log.info(f"Source benchmark has {len(src)} episodes; processing first {n_total}")

    # Episodes don't carry a global index in the JSON; derive a per-house
    # ordinal so the (house_index, within_house_idx) key matches the
    # pipeline's "house X episode Y" log format.
    counter: dict[int, int] = {}
    src_keys: list[tuple[int, int]] = []
    for ep in src:
        h = int(ep["house_index"])
        e = counter.get(h, 0)
        src_keys.append((h, e))
        counter[h] = e + 1

    processed = set()
    for r in load_jsonl(args.output_dir / "curate_results.jsonl"):
        processed.add((int(r["house_index"]), int(r["episode_idx"])))
    log.info(f"Already processed: {len(processed)} episodes")

    todo_idx: list[int] = []
    for i in range(n_total):
        if src_keys[i] not in processed:
            todo_idx.append(i)
    log.info(f"To process: {len(todo_idx)} episodes")

    chunk_size = args.chunk_size
    for chunk_id, start in enumerate(range(0, len(todo_idx), chunk_size)):
        chunk = todo_idx[start : start + chunk_size]
        run_chunk(args, chunk, chunk_id)

    out = emit_level2_benchmark(args)
    print(f"\nDone. Curated benchmark: {out}")


if __name__ == "__main__":
    main()
