#!/bin/bash

# Ensure required directories exist
WMA_DIR="$(dirname "$0")/wma"
OUT_DIR="$(dirname "$0")/processed"
mkdir -p "$OUT_DIR"

# Loop over each .wma file
for wma_file in $WMA_DIR/*.[wW][mM][aA]; do
    [ -e "$wma_file" ] || continue  # skip if no .wma files exist

    filename=$(basename "$wma_file" .wma)
    out_file="$OUT_DIR/$filename.wav"

    if [ -f "$out_file" ]; then
        echo "✅ Skipping: $filename.wav already exists"
    else
        echo "🎧 Converting: $filename.wma → $filename.wav"
        ffmpeg -y -i "$wma_file" -c:a pcm_s16le "$out_file"
    fi
done
