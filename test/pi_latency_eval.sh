#!/bin/bash
# All-latency evaluation of stt_mod (mc-ref) on the Pi, real-mic welding data.
set -o pipefail
WS=$HOME/dorai; HERE=$(pwd)
temp() { awk '{printf "%.1f", $1/1000}' /sys/class/thermal/thermal_zone0/temp; }
step() { echo; echo "##### $(date -u +%T) [$(temp)C] $*"; }
step "update stt_mod code (metrics + evaluate) and rebuild"
tar xf stt_mod_code.tar -C $WS/src            # models dir untouched (excluded from tar)
cd $WS && source /opt/ros/jazzy/setup.bash && colcon build --symlink-install --packages-select stt_mod 2>&1 | tail -1
source $WS/install/setup.bash; cd $HERE
(cd $WS/src/stt_mod && python3 -m pytest test -q -p no:cacheprovider 2>&1 | tail -1)
step "graph hash of baked encoders (metadata stripped) - compare with Mac"
python3 graph_hash.py $WS/src/stt_mod/models/*-dorai-mc-ref/encoder.int8.onnx | tee pi_graph_sha.txt | cut -c1-90
tar xzf realmic_weld.tgz
ros2 run stt_mod transcribe --list
cooldown() { for i in $(seq 1 60); do t=$(temp); [ "${t%.*}" -lt 60 ] && break; sleep 10; done; echo "temp before run $(temp)C"; }
for L in 1120 560 160 80; do
  M=realmic_weld/pi_manifest.jsonl; [ $L -le 160 ] && M=realmic_weld/pi_manifest_small.jsonl
  cooldown; step "evaluate ${L} ms mc-ref on $(wc -l < $M) utterances"
  ros2 run stt_mod evaluate --manifest $M --latency-ms $L --adapter mc-ref --threads 2 --gap-s 3 \
      --name artaban --out eval_pi_$L 2>&1 | grep -v "^|" | tail -8
done
step "done"
tar czf eval_pi_results.tgz eval_pi_* pi_graph_sha.txt && ls -la eval_pi_results.tgz
cat eval_pi_*/summary.md
