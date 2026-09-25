#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
cmake -S . -B build >/dev/null
cmake --build build -j
echo "=== unit tests ==="
./build/test_mrz_ocr || true
echo
echo "=== generate corpus (>= 100 samples) ==="
python3 tests/gen_corpus.py
echo
echo "=== corpus results ==="
python3 tests/test_corpus.py
echo
echo "=== sample recognition ==="
./build/mrz_ocr_tool data/corpus/img_0001_p0_v0.ppm || true
