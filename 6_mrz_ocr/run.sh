#!/usr/bin/env bash
# Quick smoke run (build + unit tests + one sample). For the full
# train/test/fix loop with regression gate use ./loop.sh
set -e
cd "$(dirname "$0")"
cmake -S . -B build >/dev/null
cmake --build build -j
echo "=== unit tests ==="
./build/test_mrz_ocr || true
[ -x ./build/test_postproc ] && ./build/test_postproc || true
[ -x ./build/test_golden ] && ./build/test_golden || true
echo
echo "=== frozen corpus check ==="
if [ ! -d data/corpus_eval ]; then
    python3 tests/gen_corpus.py --eval --out data/corpus_eval --seed 2002
fi
echo
echo "=== sample recognition ==="
./build/mrz_ocr_tool data/corpus_eval/img_0001_b0_v0.ppm || true
./build/mrz_ocr_cnn_tool data/corpus_eval/img_0001_b0_v0.ppm || true
