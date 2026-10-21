#!/bin/bash
# voice_mod -> stt_mod real e2e on the Pi through ALSA loopback (debug path).
#   A3: welding test set, 3 mics -> voice_mod (beamformer + enhancement) -> stt_mod
#   A1: same audio, mic 0 only -> voice_mod (enhancement only)          -> stt_mod
#   B1: real-mic mono takes   -> voice_mod (1 mic)                      -> stt_mod
#   A0: welding test set, ref mic 0 straight into stt_mod (no voice_mod) via e2e_realmic.py
set -o pipefail
WS=$HOME/dorai; HERE=$(pwd); OUT=$HERE/out; mkdir -p $OUT
temp() { awk '{printf "%.1f", $1/1000}' /sys/class/thermal/thermal_zone0/temp; }
step() { echo; echo "##### $(date -u +%T) [$(temp)C] $*"; }
export ROS_DOMAIN_ID=43

step "deps (scipy from pip: apt scipy is numpy-1 only, user site has numpy 2)"
python3 -c "import sounddevice, samplerate, scipy.signal" 2>/dev/null || \
  python3 -m pip install --user --break-system-packages -q scipy sounddevice samplerate 2>&1 | grep -v -i warn | tail -2
python3 -c "import sounddevice, samplerate, scipy, scipy.signal, onnxruntime, numpy; print('deps ok numpy', numpy.__version__, 'scipy', scipy.__version__)" || exit 5

step "workspace (provisioned by pi_provision.sh; stt_mod code updated by pi_latency_eval.sh)"
source /opt/ros/jazzy/setup.bash; source $WS/install/setup.bash
ros2 pkg executables voice_mod | head -2; ros2 pkg executables stt_mod | grep evaluate
step "loopback: 2 snd-aloop cards (hw:10, hw:11) = 4 capture devices"
sudo modprobe -r snd-aloop 2>/dev/null; sudo modprobe snd-aloop enable=1,1 index=10,11 pcm_substreams=1
sleep 1; sudo chmod a+rw /dev/snd/*; grep -i loopback /proc/asound/cards
python3 -c "import sounddevice as sd; [print(i, d['name'], d['max_input_channels']) for i, d in enumerate(sd.query_devices()) if 'Loopback' in d['name']]"

step "resample stimuli 16k -> 48k"
python3 - <<'PY'
import wave, numpy as np, glob
from scipy.signal import resample_poly
for p in sorted(glob.glob("*_ch[0-9].wav")):
    w = wave.open(p); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32); w.close()
    y = resample_poly(x, 3, 1)
    o = wave.open(p.replace(".wav", "_48k.wav"), "wb"); o.setnchannels(1); o.setsampwidth(2); o.setframerate(48000)
    o.writeframes(np.clip(y, -32768, 32767).astype(np.int16).tobytes()); o.close(); print(p, len(y) / 48000, "s")
PY
ls A_ch0_48k.wav A_ch1_48k.wav A_ch2_48k.wav B_ch0_48k.wav || { echo "stimulus resampling failed"; exit 6; }

# Start nodes by their own executables (not `ros2 run`, whose wrapper does not
# pass SIGTERM on to the node), so $! is the node and kill -TERM really stops it.
STT_BIN=$WS/install/stt_mod/lib/stt_mod/stt
VOICE_BIN=$WS/install/voice_mod/lib/voice_mod/voice
stop() {        # stop pid $1 with TERM, escalate to KILL after 10 s; report how it ended
  kill -TERM $1 2>/dev/null
  for i in $(seq 1 20); do kill -0 $1 2>/dev/null || { echo "stopped pid $1"; return 0; }; sleep 0.5; done
  echo "pid $1 ignored SIGTERM -> SIGKILL"; kill -KILL $1 2>/dev/null; wait $1 2>/dev/null
}
cooldown() { for i in $(seq 1 40); do t=$(temp); [ "${t%.*}" -lt 62 ] && break; sleep 10; done; echo "temp before run $(temp)C"; }

start_stt() {   # $1 = log
  $STT_BIN --ros-args -p status_period_s:=0.5 -p num_threads:=2 -p metrics:=true -p metrics_file:="${2:-}" > "$1" 2>&1 &
  STT=$!
  for i in $(seq 1 120); do grep -q "Nemotron ready" "$1" && break; sleep 1; done
  grep "Nemotron ready" "$1" | cut -c1-120
}

run_vm() {      # $1 = name, $2 = n mics, $3 = stimulus prefix (A|B)
  local N=$1 M=$2 S=$3 D=$OUT/$1; mkdir -p $D; rm -f $D/done
  cooldown; step "run $N: $M mic(s), stimulus $S"
  start_stt $D/stt.log $D/metrics.jsonl
  python3 $HERE/vm_e2e_harness.py --out $D --done-file $D/done --tail-s 12 > $D/harness.log 2>&1 &
  H=$!
  $VOICE_BIN --ros-args -p mic_name_filter:=Loopback -p max_mics:=$M -p capture_channels:=1 \
      -p publish_interval:=5.0 -p num_threads:=2 -p input_latency:=0.25 > $D/voice.log 2>&1 &
  V=$!
  for i in $(seq 1 60); do grep -q "FIFOs primed" $D/voice.log && break; sleep 1; done
  grep -q "FIFOs primed" $D/voice.log || { echo "voice_mod did not prime:"; tail -15 $D/voice.log; kill $STT $H $V 2>/dev/null; return 1; }
  grep -E "Auto-detected|ready \(|primed" $D/voice.log | cut -c1-160
  sleep 2
  ( while kill -0 $V 2>/dev/null; do echo "$(date -u +%T) temp $(temp)C load $(cut -d' ' -f1 /proc/loadavg) freq $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq)"; sleep 15; done ) > $D/sysmon.log &
  MON=$!
  DEV=(plughw:10,1,0 plughw:10,0,0 plughw:11,1,0)       # -> captured on hw:10,0 / hw:10,1 / hw:11,0 = mic0/1/2
  date -u +%s.%N > $D/play_start
  PIDS=(); for c in $(seq 0 $((M-1))); do aplay -q -D ${DEV[$c]} ${S}_ch${c}_48k.wav & PIDS+=($!); done
  for p in "${PIDS[@]}"; do wait $p; done
  date -u +%s.%N > $D/play_end; touch $D/done
  sleep 12; stop $V                                      # stop input: tail published, enhancer flushed
  echo "playback done, voice_mod stopped; waiting for stt to drain"
  wait $H; tail -1 $D/harness.log
  stop $STT; kill $MON 2>/dev/null
  (cd $OUT && tar czf $HERE/vm_e2e_$N.tgz $N) && echo "saved vm_e2e_$N.tgz"   # survive a later crash
  echo "voice_mod xruns: $(grep -c -i xrun $D/voice.log)  frames published: $(grep -c 'Published frame' $D/voice.log)"
  grep -c "Transcript:" $D/stt.log
}

run_vm A3 3 A
run_vm A1 1 A
run_vm B1 1 B

step "all done"
cd $OUT && tar czf $HERE/vm_e2e_results.tgz . && ls -la $HERE/vm_e2e_results.tgz
