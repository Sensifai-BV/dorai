#!/bin/bash
# Run the full DORAI graph on one simulated scene in the OrbStack ros2 VM and record every topic.
#   bash run_scene.sh <scene_dir> <out_dir>

SCENE=$1; OUT=$2
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"; rm -f "$OUT"/*.log "$OUT"/topics.jsonl
source /opt/ros/${ROS_DISTRO:-lyrical}/setup.bash
source ${DORAI_WS:-$HOME/dorai_fortis_ws}/install/setup.bash
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}
export PYTHONUNBUFFERED=1
pulseaudio --check 2>/dev/null || pulseaudio --start --exit-idle-time=-1 2>/dev/null
MODELS=${DORAI_MODELS_DIR:-$HOME/dorai/models}

pids=()
run() { local name=$1; shift; "$@" > "$OUT/$name.log" 2>&1 & pids+=($!); }

run recorder python3 "$HERE/recorder.py" "$OUT"
run stt     ros2 run stt_mod stt --ros-args -p models_dir:=$MODELS -p metrics:=true -p num_threads:=2
run intent  ros2 run intent_mod intent --ros-args -p models_dir:=$MODELS -p auto_download:=false -p debug:=true -p num_threads:=4
run doa     ros2 run doa_mod doa --ros-args -p source:=topic -p input_topic:=/dorai_sim/array
run voice   ros2 run voice_mod voice --ros-args -p source:=topic -p publish_interval:=1.0 -p delay_log_path:=$OUT/delay.csv

# wait until intent has loaded its model (or 120 s)
for i in $(seq 1 120); do
  grep -qiE "ready|engine=|loaded" "$OUT/intent.log" 2>/dev/null && break; sleep 1
done
sleep 5
echo "[run] nodes up after ${i}s, playing $SCENE"
ros2 run dorai_simulator virtual_mic --ros-args -p scene:=$SCENE > "$OUT/sim.log" 2>&1
echo "[run] scene finished, draining"
sleep 25
kill -INT "${pids[@]}" 2>/dev/null; sleep 4; kill "${pids[@]}" 2>/dev/null
wait 2>/dev/null
echo "[run] done -> $OUT"
