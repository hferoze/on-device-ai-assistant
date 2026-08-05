#!/usr/bin/env bash

set -euo pipefail

WHISPER_DIR="../experiments/speech-to-text/whisper.cpp"
MODEL="$WHISPER_DIR/models/ggml-base.en.bin"
STREAM="$WHISPER_DIR/build/bin/whisper-stream"

if [[ ! -x "$STREAM" ]]; then
    echo "whisper-stream was not found: $STREAM"
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo "Whisper model was not found: $MODEL"
    exit 1
fi

"$STREAM" \
    -m "$MODEL" \
    --step 500 \
    --length 5000 \
    --keep 200 \
    --capture 0