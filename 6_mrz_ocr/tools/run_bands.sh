#!/bin/bash
# Run the C recogniser over every exported product-path band and dump
# "stem|line1|line2" for the scorer.
#
# Usage: run_bands.sh <out-file> [ENV=VAL ...]
set -u
OUT="$1"; shift
DIR="${BANDS_DIR:-/tmp/holdout_export2}"
TOOL="${TOOL:-./build/mrz_ocr_cnn_tool}"
: > "$OUT"
for f in "$DIR"/*.ppm; do
  R=$(env MRZ_OCR_LINE_MODELS=1 MRZ_OCR_PAR=0 MRZ_OCR_CRNN=0 MRZ_OCR_FUSE=0 "$@" \
        "$TOOL" "$f" 2>/dev/null | tr -d '\r')
  L1=$(printf '%s\n' "$R" | sed -n 's/^result\.line1 *: *//p')
  L2=$(printf '%s\n' "$R" | sed -n 's/^result\.line2 *: *//p')
  printf '%s|%s|%s\n' "$(basename "$f" .ppm)" "$L1" "$L2" >> "$OUT"
done
