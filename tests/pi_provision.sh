#!/bin/bash
# Provision a fresh Ubuntu 24.04 Raspberry Pi for dorai stt_mod (ROS 2 Jazzy +
# sherpa-onnx), install the stt_mod package (with its bundled Nemotron model)
# into a colcon workspace and smoke-test it. Idempotent: re-running skips what
# is already there.
#
#   bash pi_provision.sh <dir containing stt_mod_pkg.tar>
#
# stt_mod_pkg.tar = `tar cf stt_mod_pkg.tar stt_mod` from the repo root, or on a
# slow link stt_mod_code.tar (models excluded) + adapter_mc_ref.tar and
# EXTRA_LATENCIES="80 160 560 1120" to download + bake every model on the Pi.
set -eo pipefail
SRC="${1:-.}"
WS="${DORAI_WS:-$HOME/dorai}"
export DEBIAN_FRONTEND=noninteractive
step() { echo; echo "##### $(date -u +%T) $*"; }
apt_get() { sudo -E apt-get -o DPkg::Lock::Timeout=1800 "$@"; }

step "clock: $(date -u)  (no RTC: TLS fails if this is in the past)"
# A fresh image runs unattended-upgrades on first boot (~60 packages) and holds
# the dpkg lock for 10+ min; wait for it instead of failing.
# (match the upgrader itself, not the permanent unattended-upgrade-shutdown
#  daemon, which `pgrep -x unattended-upgr` also hits: comm is cut to 15 chars)
for i in $(seq 1 360); do pgrep -f "^/usr/bin/python3 /usr/bin/unattended-upgrade" >/dev/null || break
  [ $((i % 30)) -eq 1 ] && echo "waiting for unattended-upgrades to finish ($((i * 10 / 60)) min)"; sleep 10; done

if [ ! -f /opt/ros/jazzy/setup.bash ]; then
  step "apt: base + ROS 2 Jazzy repo"
  apt_get update -qq
  apt_get install -y -qq software-properties-common curl gnupg lsb-release bzip2 alsa-utils python3-pip >/dev/null
  sudo add-apt-repository -y universe >/dev/null 2>&1 || true
  sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu noble main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list >/dev/null
  # Stock Pi images list only noble + noble-security; -dev packages then cannot
  # match the -updates runtime libs ("held broken packages"). Add noble-updates.
  grep -q noble-updates /etc/apt/sources.list.d/ubuntu.sources || sudo tee -a /etc/apt/sources.list.d/ubuntu.sources >/dev/null <<'EOF2'

Types: deb
URIs: http://ports.ubuntu.com/ubuntu-ports/
Suites: noble-updates
Components: main restricted universe multiverse
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
EOF2
  apt_get update -qq
  step "apt: ros-jazzy-ros-base + colcon + launch + numpy/yaml"
  apt_get install -y -qq ros-jazzy-ros-base python3-colcon-common-extensions \
      ros-jazzy-launch-ros python3-numpy python3-yaml python3-pytest libportaudio2 >/dev/null
fi
ls /opt/ros/jazzy/setup.bash && echo ros-ok

step "pip: sherpa-onnx (+ onnx for bake_adapter tests) and voice_mod deps"
# sherpa-onnx pulls numpy 2.x into ~/.local; apt's python3-scipy is built for
# numpy 1.x and then fails to import (voice_mod imports scipy.signal), so scipy
# must come from pip as well.
python3 -c "import sherpa_onnx, scipy.signal, sounddevice, samplerate" 2>/dev/null || \
  python3 -m pip install --user --break-system-packages -q sherpa-onnx onnxruntime onnx scipy sounddevice samplerate 2>&1 | grep -v -i warn | tail -3 || true
python3 -c "import sherpa_onnx, onnxruntime, numpy, scipy.signal; print('py deps ok: sherpa-onnx', sherpa_onnx.__version__, 'onnxruntime', onnxruntime.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__)"

step "packages -> $WS/src (stt_mod, + voice_mod if voice_mod_pkg.tar is given)"
mkdir -p "$WS/src"
rm -rf "$WS/src/stt_mod"
if [ -f "$SRC/stt_mod_pkg.tar" ]; then
  tar xf "$SRC/stt_mod_pkg.tar" -C "$WS/src"            # code + bundled model
else
  tar xf "$SRC/stt_mod_code.tar" -C "$WS/src"           # code only: bake models below
  mkdir -p "$WS/src/stt_mod/models"
fi
PKGS="stt_mod"
if [ -f "$SRC/voice_mod_pkg.tar" ]; then
  rm -rf "$WS/src/voice_mod"; tar xf "$SRC/voice_mod_pkg.tar" -C "$WS/src"; PKGS="voice_mod stt_mod"
fi
# optional extra latency profiles / adapters (baked model dirs), installed with the package
[ -f "$SRC/models_extra.tar" ] && tar xf "$SRC/models_extra.tar" -C "$WS/src/stt_mod/models"
# Or bake extra profiles on the Pi: EXTRA_LATENCIES="80 160 560" + adapter_mc_ref.tar
# (the adapter dir, 6.5 MB) -> download each stock k2-fsa export and bake mc-ref in.
if [ -n "${EXTRA_LATENCIES:-}" ] && [ -f "$SRC/adapter_mc_ref.tar" ]; then
  mkdir -p "$WS/models_dl" && tar xf "$SRC/adapter_mc_ref.tar" -C "$WS/models_dl"
  for L in $EXTRA_LATENCIES; do
    N=sherpa-onnx-nemotron-speech-streaming-en-0.6b-${L}ms-int8-2026-04-25
    [ -d "$WS/src/stt_mod/models/$N-dorai-mc-ref" ] && continue
    ( cd "$WS/models_dl" && [ -d "$N" ] || { curl -sS -L --retry 3 -o "$N.tar.bz2" \
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$N.tar.bz2" && tar xjf "$N.tar.bz2" && rm -f "$N.tar.bz2"; } )
    PYTHONPATH="$WS/src/stt_mod" python3 -m stt_mod.bake_adapter --export "$WS/models_dl/$N" \
        --adapter "$WS/models_dl/mc-ref" --name mc-ref --out "$WS/src/stt_mod/models/$N-dorai-mc-ref" 2>&1 | grep -E "done|rror|warning"
    rm -rf "$WS/models_dl/$N"
  done
fi
ls "$WS/src/stt_mod/models"
sha256sum "$WS"/src/stt_mod/models/*/encoder.int8.onnx | cut -c1-16

step "colcon build ($PKGS)"
cd "$WS"
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select $PKGS 2>&1 | tail -3
source install/setup.bash
ros2 pkg executables stt_mod; ros2 pkg executables voice_mod 2>/dev/null || true

step "unit tests"
(cd src/stt_mod && python3 -m pytest test -q -p no:cacheprovider 2>&1 | tail -3)

step "smoke: transcribe bundled test wavs (1120 ms + mc-ref, 2 threads)"
ros2 run stt_mod transcribe --threads 2 src/stt_mod/models/*/test_wavs/0.wav src/stt_mod/models/*/test_wavs/1.wav
echo PROVISION-DONE
