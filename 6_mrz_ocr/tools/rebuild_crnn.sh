#!/usr/bin/env bash
# Rebuild the CRNN sequence head end to end.
#
# WHY THIS SCRIPT EXISTS
# ----------------------
# The Native feature extractor depends on a 64 MB scratchpad whose bump
# allocation order changes whenever mrz_ocr_cnn.c is edited. Training on
# features exported from a DIFFERENT build than the one used at inference
# silently degrades accuracy (observed: 95% -> 34% char accuracy).
#
# The safe order is therefore:
#   1. build the native tools
#   2. export features for train / eval / real with the SAME binary
#   3. train and export the C weights
#   4. rebuild (the weights header changed) and verify
#
# Always run this script after touching 6_mrz_ocr/src/mrz_ocr_cnn.c.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$HOME/.workbuddy-ai/binaries/python/envs/crnn/bin/python}"
WORK="${WORK:-/tmp/crnn_build}"
REPEAT="${REPEAT:-20}"

cd "$ROOT"
mkdir -p "$WORK"

echo "== 1/4 build native tools =="
cmake -S . -B build >/dev/null
cmake --build build -j"$(sysctl -n hw.ncpu 2>/dev/null || echo 4)" >/dev/null

echo "== 2/4 export features (single consistent binary) =="
for c in corpus_train corpus_eval corpus_real; do
  [ -d "data/$c" ] || continue
  "$PY" tools/export_crnn_features.py --corpus "data/$c" \
      --out "$WORK/$c.npz" --crnn 1 --parallel 0
done

echo "== 3/4 train =="
EXTRA=()
[ -f "$WORK/corpus_real.npz" ] && EXTRA+=(--extra "$WORK/corpus_real.npz" --extra-repeat "$REPEAT")
"$PY" tools/train_crnn_seq.py --train "$WORK/corpus_train.npz" "${EXTRA[@]}" \
    --out src/crnn_weights.h --epochs "${EPOCHS:-60}"

echo "== 4/4 rebuild + verify =="
cmake --build build -j"$(sysctl -n hw.ncpu 2>/dev/null || echo 4)" >/dev/null
( cd build && ctest --output-on-failure | tail -3 )
"$PY" tools/bench_crnn.py --corpus data/corpus_eval --parallel 0
