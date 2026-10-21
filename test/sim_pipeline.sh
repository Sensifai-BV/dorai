#!/bin/bash
# End-to-end simulation of the dorai speech chain WITHOUT microphones:
#
#   lab recording (mic01/mic02, 16 kHz)  --aplay-->  ALSA loopback (snd-aloop)
#   --> voice_mod (real ROS node: capture, resample, HPF, beamformer, AGC,
#       fixed publish_interval frames on /dorai_clean_audio)
#   --> stt_mod (real ROS node, Nemotron + mc-ref) --> /dorai_transcript
#
# Each mic wav is played into its own loopback direction so voice_mod sees
# TWO independent capture devices (M=2 beamformer, realistic start skew):
#   aplay -> hw:L,1,0  is captured on hw:L,0   (voice_mod "mic0")
#   aplay -> hw:L,0,0  is captured on hw:L,1   (voice_mod "mic1")
# Requires: ROS 2 sourced workspace (install/setup.bash), snd-aloop module,
# scipy (resampling), the Nemotron model in ~/dorai/models.
#
# Usage (on the Pi):
#   bash test/sim_pipeline.sh [session_dir ...]      # default: two lab sessions
# Outputs in test/sim_out/<timestamp>/: voice.log, stt.log, transcript.log,
# partial.log, clean_frames.wav (what stt_mod actually received), report.txt
cd "$(dirname "$0")/.."
ROOT=$(pwd)
SESS=("$@")
[ ${#SESS[@]} -eq 0 ] && SESS=(lab/recordings/dorai_20260703_111359 lab/recordings/dorai_20260715_124024)
OUT=test/sim_out/$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"
PUB_INT=${PUB_INT:-5.0}
THREADS=${THREADS:-2}       # stt_mod onnxruntime threads
INPUT_LAT=${INPUT_LAT:-0.25} # voice_mod PortAudio input latency (s); 0 = device default
STT_NICE=${STT_NICE:-0}      # optional: run stt_mod at lower CPU priority
# Why: run 1 (input_latency 0, nice 0) -> capture callbacks overflowed ~2.4x/s
# per mic ("xrun (input overflow)"), ~45% of the audio was dropped, speech
# arrived time-compressed 1.8x and Nemotron heard "But I". Run 2
# (input_latency 0.25, STT at nice 10) -> 2 xruns in 60 s, but nice 10 starved
# stt_mod so badly under voice_mod's load that one 5 s frame took 77 s
# (RTF 15). Keep input_latency >= 0.25 on a Pi 4 when Nemotron runs on the
# same box; use nice only lightly (2-3) if xruns persist.

set +u   # ROS setup scripts reference unset variables
source /opt/ros/jazzy/setup.bash
source "$ROOT/install/setup.bash"
set -u

# --- loopback ---------------------------------------------------------------
sudo modprobe snd-aloop || { echo "snd-aloop unavailable"; exit 2; }
L=$(grep -m1 -i 'loopback' /proc/asound/cards | awk '{print $1}')
echo "loopback card $L" | tee "$OUT/report.txt"

# --- build 48 kHz mono stimulus wavs: [2 s sil] sess1 [3 s sil] sess2 ... [4 s sil]
python3 - "$OUT" "${SESS[@]}" <<'EOF'
import sys, wave, numpy as np
from scipy.signal import resample_poly
out, sess = sys.argv[1], sys.argv[2:]
def rd(p):
    w = wave.open(p); a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    return a, w.getframerate()
for mic in ("mic01", "mic02"):
    parts = [np.zeros(2 * 48000, np.float32)]
    for s in sess:
        a, sr = rd(f"{s}/{mic}.wav")
        if sr != 48000:
            a = resample_poly(a, 48000 // np.gcd(sr, 48000), sr // np.gcd(sr, 48000)).astype(np.float32)
        parts += [a, np.zeros(3 * 48000, np.float32)]
    parts.append(np.zeros(1 * 48000, np.float32))
    x = np.concatenate(parts)
    w = wave.open(f"{out}/{mic}_48k.wav", "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(48000)
    w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()); w.close()
    print(f"{mic}: {len(x)/48000:.1f}s")
EOF

# --- nodes ------------------------------------------------------------------
ros2 topic echo /dorai_transcript std_msgs/msg/String > "$OUT/transcript.log" 2>&1 &
ros2 topic echo /dorai_partial_transcript std_msgs/msg/String > "$OUT/partial.log" 2>&1 &
nice -n $STT_NICE ros2 run stt_mod stt --ros-args -p latency_ms:=${LATENCY_MS:-1120} -p adapter:=${ADAPTER:-mc-ref} -p num_threads:=$THREADS \
    -p nemotron_frame_is_utterance:=false -p debug:=true > "$OUT/stt.log" 2>&1 &
STT=$!
for i in $(seq 1 90); do grep -q "Nemotron ready" "$OUT/stt.log" && break; sleep 1; done
grep -q "Nemotron ready" "$OUT/stt.log" || { echo "stt_mod did not start"; tail -20 "$OUT/stt.log"; kill $STT; exit 3; }
echo "stt_mod ready after ~${i}s" | tee -a "$OUT/report.txt"

# record what stt_mod receives, for offline replay/inspection
ros2 run voice_mod record --ros-args -p source:=clean -p duration:=120 -p prefix:="$ROOT/$OUT/clean" > "$OUT/record.log" 2>&1 &
REC=$!

ros2 run voice_mod voice --ros-args -p mic_name_filter:=Loopback -p max_mics:=2 \
    -p capture_channels:=1 -p publish_interval:=$PUB_INT -p num_threads:=2 \
    -p input_latency:=$INPUT_LAT > "$OUT/voice.log" 2>&1 &
VOICE=$!
for i in $(seq 1 40); do grep -q "FIFOs primed" "$OUT/voice.log" && break; sleep 1; done
grep -q "FIFOs primed" "$OUT/voice.log" || { echo "voice_mod did not prime"; tail -30 "$OUT/voice.log"; kill $VOICE $STT $REC; exit 4; }
echo "voice_mod primed after ~${i}s" | tee -a "$OUT/report.txt"
sleep 2

# --- play the stimulus into the two loopback directions ---------------------
aplay -q -D plughw:$L,1,0 "$OUT/mic01_48k.wav" & P1=$!
aplay -q -D plughw:$L,0,0 "$OUT/mic02_48k.wav" & P2=$!
wait $P1 $P2
echo "playback done $(date +%T)" | tee -a "$OUT/report.txt"

# let voice_mod publish the tail frames and stt_mod catch up (RTF ~1.9 on Pi 4)
DUR=$(python3 -c "import wave;print(wave.open('$OUT/mic01_48k.wav').getnframes()/48000)")
sleep $(python3 -c "print(int($PUB_INT*2 + $DUR*1.2 + 10))")

kill -INT $VOICE $REC 2>/dev/null; sleep 3; kill -INT $STT 2>/dev/null; sleep 3
kill $VOICE $STT $REC 2>/dev/null; pkill -f "ros2 topic echo /dorai_" 2>/dev/null

# --- report -----------------------------------------------------------------
{
echo "== voice_mod"; grep -E "Auto-detected|primed|Published frame" "$OUT/voice.log" | head -20
echo "== voice_mod xruns: $(grep -c 'xrun' "$OUT/voice.log")  last diag: $(grep 'fifo=' "$OUT/voice.log" | tail -1 | cut -c40-)"
echo "== stt_mod"; grep -E "Nemotron ready|Transcript|RTF|endpoint|error|Error" "$OUT/stt.log" | head -40
echo "== /dorai_transcript"; grep "data:" "$OUT/transcript.log"
echo "== partials seen: $(grep -c 'data:' "$OUT/partial.log")"
echo "== reference (Nemotron directly on mic01, no voice_mod):"
for s in "${SESS[@]}"; do python3 stt_mod/test/nemotron_test.py --chunk-ms 1120 --threads $THREADS --quiet-partials "$s/mic01.wav" 2>&1 | grep -E "FINAL"; done
} | tee -a "$OUT/report.txt"
echo "DONE -> $OUT"
