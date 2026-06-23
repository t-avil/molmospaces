# Handoff — full regather of the mobile-pick benchmark (9 cells, raw + reproducible)

Goal: re-run the three VLA grasp modules across the three base conditions on
`FrankaPickDroidMiniBench_origpose` (347 trajectories), and this time persist
**everything** to the `tavil` fork so each datapoint is reproducible to the
exact motion: same seeds, same perturbed base poses, same per-episode rollout.
Last time the per-episode rollouts and per-trajectory results were written to
`/tmp` and cleaned before commit; only aggregate numbers survived for 6 of 9
cells. This handoff exists so a fresh agent can refill them with no human help.

## The 9 cells (3 modules x 3 conditions)

| module        | static-hybrid (fixed base) | mobile @ static pose | mobile + perturbed |
|---------------|----------------------------|----------------------|--------------------|
| MolmoBot      | aggregated-only (324/347)  | aggregated-only (311)| aggregated-only (280) |
| MolmoBot-Pi0  | MISSING                    | MISSING              | **RAW (150/347)** ✓ |
| pi0.5         | aggregated-only (72/347)   | aggregated-only (51) | aggregated-only (48) |
| scripted IK   | 100% oracle (defines set)  | 100%                 | 100% |

Only `MolmoBot-Pi0 perturbed` is fully raw today
(`raw_runs/mbpi0_perturbed/`). Everything else marked "aggregated-only" must be
re-run to become raw. Scripted IK is the oracle ceiling and does not need re-running.

## How the conditions map to sweep modes

`scripts/molmobot_sweep.py --mode <MODE>` selects model + harness. Base condition
is set by `MLSPACES_PERTURB_SCALE` (env), read in
`molmo_spaces/tasks/json_eval_task_sampler.py`:

- **perturbed**      : `*_mobile` mode + `MLSPACES_PERTURB_SCALE=1.0` (default)
- **mobile@static**  : `*_mobile` mode + `MLSPACES_PERTURB_SCALE=0.0` (starts exactly at benchmark pose)
- **static-hybrid**  : the `*_static_hybrid` / `molmobot_static` mode (fixed DROID franka, no base, scale ignored)

Modes per module:
- MolmoBot      : `mobile` (mobile@static + perturbed), `molmobot_static` (static). Server: `serve_molmo.py`.
- MolmoBot-Pi0  : `mbpi0_mobile`, `mbpi0_static_hybrid`. Server: `serve_mbpi0.py` (openpi venv).
- pi0.5         : `pi05_mobile`, `pi05_static_hybrid`. Server: `serve_pi05.py` (openpi venv).

So the full set is 5 sweep invocations that still need raw data, each producing
its jsonl + per-episode h5 + curate-log:

```
1. MolmoBot      mobile  scale=1.0  -> perturbed   (280-equiv)
2. MolmoBot      mobile  scale=0.0  -> mobile@static(311-equiv)
3. MolmoBot      molmobot_static    -> static      (324-equiv)
4. pi0.5         pi05_mobile scale=1.0 -> perturbed
5. pi0.5         pi05_mobile scale=0.0 -> mobile@static
6. pi0.5         pi05_static_hybrid    -> static
7. MolmoBot-Pi0  mbpi0_mobile scale=0.0 -> mobile@static  (the two still-missing mbpi0 cells)
8. MolmoBot-Pi0  mbpi0_static_hybrid    -> static
   (mbpi0 perturbed already raw — skip, or re-run to get its h5 rollouts too)
```

## Reproducibility contract (this is the whole point)

The perturbation is fully deterministic. In `json_eval_task_sampler.py`:

```python
ep_seed = hash((episode_spec.seed or 0, house_index, episode_idx)) & 0xFFFFFFFF
rng = np.random.RandomState(ep_seed)
scale = float(os.environ.get("MLSPACES_PERTURB_SCALE", "1.0"))
dx   = rng.uniform(-0.5, 0.5)      * scale
dy   = rng.uniform(-0.5, 0.5)      * scale
dyaw = rng.uniform(-pi/4, pi/4)    * scale
```

`episode_idx` is the position of a `traj_key` within its house in
`benchmark.json` list order. Same (house, episode_idx, scale) -> identical
(dx, dy, dyaw) -> identical perturbed base pose -> identical rollout. This was
verified 51/51 against the surviving driver logs for the mbpi0 perturbed cell.

To capture the per-episode pose record, **set `MLSPACES_CURATE_LOG=<path>.jsonl`**
in the sweep env. The sampler then appends one line per episode with
`house_index, episode_idx, seed, original_robot_base_pose,
perturbed_robot_base_pose, dx, dy, dyaw`. That file + the code + the benchmark
is the complete recipe to replay any episode's exact motion.

## What MUST be committed per cell (so "raw" actually means raw)

1. `<mode>_results.jsonl` — per-trajectory `{house_index, traj_key, success, errored}` (resume ledger).
2. the curate-log jsonl — per-episode seed + perturbed pose (the replay recipe).
3. the per-round eval dirs containing `traj_<N>.h5` rollouts — **pass `--keep_eval_dirs`**, otherwise the sweep deletes them after collecting (this is exactly what was lost last time).
4. driver/server logs (gzipped) — they carry the `perturbed by (dx,dy,dyaw)` lines as a cross-check.

### Size warning (read before committing h5)
h5 rollouts are heavy. ~347 episodes x up to 8 cells can be GBs. GitHub rejects
files >100MB and bloats on large binaries. Pick one:
- **git-lfs** for `*.h5` / `*.mp4` (recommended; keeps "just commit" workflow), or
- tar each cell's rollouts and `gh release upload` to a tagged release on the
  fork (release assets allow up to 2GB/file, keeps git history small).
The small files (1, 2, 4) always go straight into git. Reproducibility does NOT
require the h5 (the seed+pose+code regenerates the motion) — the h5 is the
convenience replay artifact. Commit the small bundle first, then layer h5 in.

## Environment reconstruction (Phase 0 — the risky part)

The model servers live OUTSIDE this repo and were cleaned from `/tmp`:
- `MOLMOBOT_DIR = /tmp/MolmoBot/MolmoBot` (venv `.venv`, ckpt `ckpts/molmobot/MolmoBot-DROID`)
- `PI_DIR       = /tmp/MolmoBot/MolmoBot-Pi0` (openpi venv `.venv`)
- `HF_HOME      = /tmp/hf-cache` (MolmoBot-Pi0-DROID + pi05_droid snapshots)

Re-pull: `allenai/MolmoBot-DROID` (20G model.pt, via `serve_molmo.py --hf-repo`),
`allenai/MolmoBot-Pi0-DROID`, and `pi05_droid`. The molmospaces harness runs in
this repo's `.venv` (`MOBILE_PY`); the servers run in their own venvs.

Two server patches that were required last time (apply in the fresh clone):
1. `launch_scripts/serve_molmo.py` — accept `--port` (was hardcoded 8000).
2. `olmo/eval/websocket_server.py` — `_server_semaphore = Semaphore(1)` -> `Semaphore(64)`
   plus a 120s recv timeout. The `Semaphore(1)` DEADLOCKS on the eval's per-episode
   reconnect (episode 1 works, episode 2+ hangs, GPU util 0%). This was THE blocker.

The two science bugs are already fixed in committed code (no action needed):
pi0.5 delta-vs-absolute actions (`pi.py`) and the static-hybrid camera height
`base_size[2]=0.585` (`mobile_franka_eval_configs.py`).

Render needs `MUJOCO_GL=egl`.

## Run discipline (multiplex without bothering people)

- Auto-pick GPUs: query `nvidia-smi`, use only cards with low util AND free mem;
  leave a couple free. Do not hardcode all 8. One server per GPU (the served
  policy is stateful — concurrent clients corrupt each other's obs history), eval
  shard pinned via `CUDA_VISIBLE_DEVICES`, pointed at its server via `MLSPACES_MB_PORT`.
- Resumable + self-healing: results append per-episode keyed by (house, traj_key);
  re-running skips done episodes. Use `flock` (see `scripts/sweep_driver.sh`) so a
  watchdog can re-invoke safely. Small `--per_shard` (8-10) = more frequent checkpoints.
- Checkpoint-commit each round: after a shard round finishes, `git add` the new
  jsonl + curate-log + (lfs) h5, commit with the cell + count in the message,
  push to `tavil`. NEVER push to `origin` (allenai).

## Canonical invocation (one cell)

```bash
cd /homes/iws/timchick/mujoco-thor
export MUJOCO_GL=egl
export MLSPACES_PERTURB_SCALE=1.0          # 0.0 for mobile@static; omit/ignored for static modes
export MLSPACES_CURATE_LOG=results/molmobot_sweep/raw_runs/mb_perturbed/curate.jsonl
.venv/bin/python scripts/molmobot_sweep.py \
    --mode mobile \
    --benchmark_dir curate_out_full/FrankaPickDroidMiniBench_origpose \
    --out results/molmobot_sweep/raw_runs/mb_perturbed \
    --gpus <auto-picked> --per_shard 10 --task_horizon_sec 150 \
    --keep_eval_dirs
```

Repeat with the mode + scale + out-dir for each of the remaining cells above.

## Definition of done
All 8 still-missing/aggregate-only cells have, on the `tavil` fork:
`<mode>_results.jsonl` (n=347), curate-log (n=347), and h5 rollouts (git-lfs or
release asset), with per-cell counts matching (or footnoted against) the
FINDINGS table. Then regenerate the charts from raw and update
`ARTIFACT_INVENTORY.md` to flip every row to RAW.
```
```
