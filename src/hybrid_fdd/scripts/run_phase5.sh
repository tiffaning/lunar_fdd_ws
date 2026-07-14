#!/usr/bin/env bash
# Phase 5 experiment runner: automates continuous-vs-cascade data collection.
#
# For each run it: cleans up any leftovers, launches the FDD stack (robot +
# monitor + fault injector + FDD node + evaluator), waits for startup, starts
# the repeated motion, lets the 300 s experiment run, then kills everything.
# The evaluator auto-logs one CSV per run to ~/lunar_fdd_ws/data/phase5.
#
# Usage:
#   bash run_phase5.sh                 # main: 5 reps x 4 scenarios x 2 strategies (~3.5 h)
#   bash run_phase5.sh 2               # 2 reps each (pilot, ~1.4 h)
#   bash run_phase5.sh 1 120           # 1 rep, only 120 s per run (quick smoke test)
#   MODE=grid bash run_phase5.sh 1     # grid-search recording: 1 cascade run/scenario
#
# Tip: run a `bash run_phase5.sh 1 120` first and check the CSVs look sane
# before committing to the full sweep.
set -u
WS=~/lunar_fdd_ws
# colcon's setup.bash references unbound vars (COLCON_TRACE); disable -u for it
set +u
source "$WS/install/setup.bash"
set -u

REPS="${1:-5}"
RUN_SECONDS="${2:-305}"     # time to let the 300 s experiment run after motion starts
STARTUP=16                  # wait for robot + controllers + nodes before motion
MODE="${MODE:-main}"        # main | grid
# Cascade thresholds (override with grid-chosen values, e.g. L1=0.9 L2=0.8 L2N=0.5)
CASCADE_ARGS="l1_threshold:=${L1:-0.9} l2_threshold:=${L2:-0.8} l2_none_threshold:=${L2N:-0.5}"
SCENARIOS=(baseline bearing_wear joint_stiffness sensor_noise)

cleanup() {
  pkill -f repeated_motion   2>/dev/null
  pkill -f fdd_evaluator     2>/dev/null
  pkill -f cascade_fdd_node  2>/dev/null
  pkill -f hybrid_fdd_node   2>/dev/null
  pkill -f fault_injector    2>/dev/null
  pkill -f monitor_node      2>/dev/null
  pkill -f 'ros2 launch'     2>/dev/null
  pkill -f robot_state_publisher 2>/dev/null
  pkill -f spawner           2>/dev/null
  pkill -f gzserver 2>/dev/null; pkill -f gzclient 2>/dev/null
  pkill -f gazebo            2>/dev/null
  sleep 5
}

run_one() {   # $1=launch file  $2=extra launch args  $3=experiment  $4=label
  echo ">>> [$(date +%H:%M:%S)] $4 : $3"
  cleanup
  # shellcheck disable=SC2086
  ros2 launch hybrid_fdd "$1" experiment:="$3" $2 >/tmp/phase5_launch.log 2>&1 &
  sleep "$STARTUP"
  ros2 run construction_robot repeated_motion >/tmp/phase5_motion.log 2>&1 &
  sleep "$RUN_SECONDS"
  cleanup
}

echo "=== Phase 5 runner | mode=$MODE reps=$REPS run_seconds=$RUN_SECONDS ==="
trap 'echo; echo "Interrupted - cleaning up"; cleanup; exit 1' INT TERM

if [ "$MODE" = "grid" ]; then
  for scen in "${SCENARIOS[@]}"; do
    run_one cascade_fdd.launch.py "record_all_layers:=true" "$scen" "grid-record"
  done
  echo "=== grid records in $WS/data/phase5_grid ==="
else
  total=$(( REPS * ${#SCENARIOS[@]} * 2 )); i=0
  for rep in $(seq 1 "$REPS"); do
    for scen in "${SCENARIOS[@]}"; do
      i=$((i+1)); echo "--- run $i/$total ---"
      run_one standard_fdd.launch.py "" "$scen" "continuous rep$rep"
      i=$((i+1)); echo "--- run $i/$total ---"
      run_one cascade_fdd.launch.py  "$CASCADE_ARGS" "$scen" "cascade rep$rep"
    done
  done
  echo "=== done. eval CSVs in $WS/data/phase5 ==="
fi
