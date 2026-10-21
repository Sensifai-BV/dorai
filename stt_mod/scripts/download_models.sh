#!/bin/bash
# Download the dorai Nemotron + mc-ref models from the Hugging Face Hub.
#   bash stt_mod/scripts/download_models.sh                      # 1120 ms -> ~/dorai/models
#   LATENCIES="1120 560" DEST=stt_mod/models bash stt_mod/scripts/download_models.sh
#   ADAPTER_RAW=1 ...   also fetch the raw adapter (adapter/mc-ref/adapter.safetensors)
set -euo pipefail
REPO_ID=${REPO_ID:-sensifai/dorai-nemotron-speech-0.6b-mc-ref-onnx}
LATENCIES=${LATENCIES:-1120}
DEST=${DEST:-$HOME/dorai/models}
command -v hf >/dev/null || python3 -m pip install --user --break-system-packages -q -U huggingface_hub
mkdir -p "$DEST"
INC=(--include SHA256SUMS --include README.md --include NOTICE.txt --include LICENSE.txt)
for L in $LATENCIES; do INC+=(--include "sherpa-onnx-nemotron-speech-streaming-en-0.6b-${L}ms-int8-2026-04-25-dorai-mc-ref/*"); done
[ "${ADAPTER_RAW:-0}" = 1 ] && INC+=(--include "adapter/*")
# hf takes ONE pattern per --include (several after one flag are read as file names)
hf download "$REPO_ID" "${INC[@]}" --local-dir "$DEST"
cd "$DEST"
PAT=$(for L in $LATENCIES; do printf '%s|' "en-0.6b-${L}ms-int8"; done); PAT=${PAT%|}
grep -E -e "$PAT" SHA256SUMS | shasum -a 256 -c --quiet && echo "checksums OK"
ls -d sherpa-onnx-nemotron-*-dorai-mc-ref
echo "stt_mod finds these automatically if DEST is ~/dorai/models or stt_mod/models (else -p models_dir:=$DEST)"
