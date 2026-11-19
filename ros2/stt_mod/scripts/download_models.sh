#!/bin/bash
# Download the dorai models from the Hugging Face Hub: https://huggingface.co/sensifai/dorai
#
#   bash stt_mod/scripts/download_models.sh                      # Whisper + Nemotron -> ~/dorai/models
#   MODELS=whisper bash stt_mod/scripts/download_models.sh       # only the default STT engine's model (1.0 GB)
#   MODELS="whisper nemotron intent" bash stt_mod/scripts/download_models.sh     # also intent_mod's GGUF (2.6 GB)
#   DEST=/data/models bash stt_mod/scripts/download_models.sh    # then run the nodes with -p models_dir:=/data/models
#
# The files keep the repo's layout under DEST; stt_mod and intent_mod look there on their own:
#   stt-model/whisper-finetuned/sherpa-onnx-whisper-large-v3-turbo-dorai-artaban/   Whisper large-v3-turbo, dorai fine-tune (int8)
#   stt-model/nemo-finetuned/                                                       Nemotron 0.6B, 1120 ms, mc-ref adapter (int8)
#   intent-model/intent-qwen35-4b-Q4_0.gguf                                         Qwen3.5-4B intent model
set -euo pipefail
REPO_ID=${REPO_ID:-sensifai/dorai}
MODELS=${MODELS:-${ENGINES:-whisper nemotron}}
DEST=${DEST:-$HOME/dorai/models}
WHISPER=stt-model/whisper-finetuned
WHISPER_DIR=sherpa-onnx-whisper-large-v3-turbo-dorai-artaban
command -v hf >/dev/null || python3 -m pip install --user --break-system-packages -q -U huggingface_hub
mkdir -p "$DEST"
INC=()            # hf takes ONE pattern per --include (several after one flag are read as file names)
for M in $MODELS; do
  case "$M" in
    whisper)  INC+=(--include "$WHISPER/$WHISPER_DIR/*" --include "$WHISPER/SHA256SUMS") ;;
    nemotron) INC+=(--include "stt-model/nemo-finetuned/*") ;;
    intent)   INC+=(--include "intent-model/*") ;;
    *) echo "unknown model '$M' (whisper | nemotron | intent)" >&2; exit 2 ;;
  esac
done
hf download "$REPO_ID" "${INC[@]}" --local-dir "$DEST"
case " $MODELS " in *" whisper "*)
  (cd "$DEST/$WHISPER" && grep -F -e "$WHISPER_DIR/" SHA256SUMS | shasum -a 256 -c --quiet && echo "whisper checksums OK") ;;
esac
(cd "$DEST" && ls -d "$WHISPER/$WHISPER_DIR" stt-model/nemo-finetuned intent-model/*.gguf 2>/dev/null) || true
echo "stt_mod and intent_mod find these on their own if DEST is ~/dorai/models (else -p models_dir:=$DEST)"
