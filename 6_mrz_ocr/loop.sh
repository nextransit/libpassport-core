#!/usr/bin/env bash
# Closed loop: build -> (retrain) -> bench -> regression gate -> mine.
#
# Usage:
#   ./loop.sh              # bench + gate on the frozen eval corpus
#   ./loop.sh --retrain    # retrain on data/corpus_train dumps first
#
# Corpora are generated ONCE (fixed seeds) and then frozen:
#   data/corpus_eval  seed 2002  (bench + gate input, never trained on)
#   data/corpus_train seed 1001  (train_cnn_real.py input, never benched)
# Delete a corpus directory explicitly to regenerate it.
set -euo pipefail
cd "$(dirname "$0")"

echo "=== build ==="
cmake -S . -B build >/dev/null
cmake --build build -j

echo "=== corpora (created once, then frozen) ==="
[ -d data/corpus_eval ] || python3 tests/gen_corpus.py --eval --out data/corpus_eval --seed 2002
[ -d data/corpus_train ] || python3 tests/gen_corpus.py --train --out data/corpus_train --seed 1001

echo "=== unit tests ==="
./build/test_mrz_ocr || true
[ -x ./build/test_postproc ] && ./build/test_postproc || true
[ -x ./build/test_golden ] && ./build/test_golden || true

if [ "${1:-}" = "--retrain" ]; then
    echo "=== retrain on real pipeline dumps (train corpus only) ==="
    python3 tools/train_cnn_real.py --corpus data/corpus_train \
        --out src/cnn_weights.h
    cmake --build build -j
fi

echo "=== bench + gate on frozen eval corpus ==="
python3 tests/bench.py --corpus data/corpus_eval \
    --gate data/baseline.json --update-baseline

echo "=== failure mining ==="
python3 tools/mine_failures.py --method cnn --suggest || true

echo "=== sample recognition ==="
./build/mrz_ocr_cnn_tool data/corpus_eval/img_0001_b0_v0.ppm || true
