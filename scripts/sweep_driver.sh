#!/bin/bash
# Self-healing driver for the MolmoBot static-vs-mobile 347 sweep.
#
# Runs the mobile sweep to completion, then (once enabled) the static sweep.
# flock makes it safe to call repeatedly: a cron watchdog invokes this every
# few minutes; if a sweep is already running, the new call exits immediately;
# if the previous run died, the next call resumes (the sweep itself is
# resumable via its results.jsonl). Survives this Claude session.
#
# Static is gated behind scripts/.static_ok so it only runs after manual
# validation (avoids the watchdog hammering an unproven invocation).
set -u

REPO=/homes/iws/timchick/mujoco-thor
BENCH=$REPO/curate_out_full/FrankaPickDroidMiniBench_mobile_v1
PY=$REPO/.venv/bin/python
GPUS=0,1,2,3                 # leaves GPU4 free; 5-7 are other students'
HORIZON=150
PER_SHARD=10

exec 9>/tmp/molmobot_sweep.lock
flock -n 9 || { echo "$(date '+%F %T') driver: another instance holds the lock; exit"; exit 0; }

cd "$REPO" || exit 1
N=$($PY -c "import json;print(len(json.load(open('$BENCH/benchmark.json'))))")
count() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }

echo "$(date '+%F %T') driver: N=$N mobile=$(count /tmp/sweep_mobile/mobile_results.jsonl) static=$(count /tmp/sweep_static/static_results.jsonl)"

if [ "$(count /tmp/sweep_mobile/mobile_results.jsonl)" -lt "$N" ]; then
  mkdir -p /tmp/sweep_mobile
  echo "$(date '+%F %T') driver: running MOBILE sweep"
  $PY scripts/molmobot_sweep.py --mode mobile --benchmark_dir "$BENCH" \
      --out /tmp/sweep_mobile --gpus $GPUS --per_shard $PER_SHARD \
      --task_horizon_sec $HORIZON >> /tmp/sweep_mobile/driver.log 2>&1
fi

# Once mobile is fully done, free the per-GPU mobile servers (~10GB each) so the
# in-process static phase has GPU memory headroom on GPUs 0-3.
if [ "$(count /tmp/sweep_mobile/mobile_results.jsonl)" -ge "$N" ]; then
  pkill -9 -f 'serve_molmo.py' 2>/dev/null || true
  sleep 3
fi

if [ -f "$REPO/scripts/.static_ok" ] && [ "$(count /tmp/sweep_static/static_results.jsonl)" -lt "$N" ]; then
  mkdir -p /tmp/sweep_static
  echo "$(date '+%F %T') driver: running STATIC sweep"
  $PY scripts/molmobot_sweep.py --mode static --benchmark_dir "$BENCH" \
      --out /tmp/sweep_static --gpus $GPUS --per_shard $PER_SHARD \
      --task_horizon_sec $HORIZON >> /tmp/sweep_static/driver.log 2>&1
fi

echo "$(date '+%F %T') driver: done this invocation"
