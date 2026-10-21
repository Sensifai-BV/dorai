#!/bin/bash
# Real e2e on the Pi: running stt_mod node <- ROS 2 /dorai_clean_audio <- real-mic takes.
set -eo pipefail
WS="${DORAI_WS:-$HOME/dorai}"
source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash"
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}           # keep the test off other robots' graph
tar xzf realmic.tgz
temp() { awk '{printf "%.1f", $1/1000}' /sys/class/thermal/thermal_zone0/temp; }
echo "start $(date -u +%T) temp $(temp)C"
ros2 run stt_mod stt --ros-args -p status_period_s:=0.5 -p num_threads:=2 > stt_node.log 2>&1 &
STT=$!
for i in $(seq 1 120); do grep -q "Nemotron ready" stt_node.log && break; sleep 1; done
grep "Nemotron ready" stt_node.log || { cat stt_node.log; kill $STT; exit 3; }
( while kill -0 $STT 2>/dev/null; do echo "$(date -u +%T) temp $(temp)C load $(cut -d' ' -f1-3 /proc/loadavg) freq $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null)"; sleep 15; done ) > sysmon.log &
MON=$!
python3 "$WS/src/stt_mod/test/e2e_realmic.py" --manifest realmic/manifest.jsonl --out e2e_out ${E2E_ARGS:-} 2>&1 | tee e2e.log
kill -TERM $STT; wait $STT || true; kill $MON 2>/dev/null || true
cp e2e_out/* . ; tail -5 sysmon.log; echo "end $(date -u +%T)"
ls -lh summary.md summary.json hyps.jsonl stt_node.log sysmon.log || true
