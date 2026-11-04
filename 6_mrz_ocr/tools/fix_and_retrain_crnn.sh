#!/usr/bin/env bash
# Fix corpus_real GT labels using passport official GT, then retrain CRNN.
# Does NOT modify the original corpus_real/corpus.json.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$HOME/.workbuddy-ai/binaries/python/envs/crnn/bin/python}"
WORK="${WORK:-/tmp/crnn_build}"
cd "$ROOT"

echo "== 1/5 create fixed corpus_real copy =="
FIX_DIR="$WORK/corpus_real_fixed"
rm -rf "$FIX_DIR"
mkdir -p "$FIX_DIR"
cp data/corpus_real/*.ppm "$FIX_DIR/"

"$PY" -c "
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath('.')))
# Load passport GT (correct)
with open('tests/gt_photo_passport_official.json') as f:
    gt = json.load(f)['gt']
passport_gt = {}
for k, v in gt.items():
    key = k.replace('.jpg','')
    passport_gt[key] = (v['l1'], v['l2'])

# Load corpus_real
with open('data/corpus_real/corpus.json') as f:
    corpus = json.load(f)

updated = 0
for r in corpus['records']:
    key = r['id'].split('/')[-1]
    if key in passport_gt:
        p1, p2 = passport_gt[key]
        if r.get('line1','') != p1 or r.get('line2','') != p2:
            r['line1'] = p1
            r['line2'] = p2
            r['label_source'] = 'passport_gt_official'
            updated += 1

with open('$FIX_DIR/corpus.json', 'w') as f:
    json.dump(corpus, f, indent=2, ensure_ascii=False)
print(f'Fixed {updated} records in copy at $FIX_DIR/corpus.json')
"

echo "== 2/5 export features from fixed corpus =="
"$PY" tools/export_crnn_features.py --corpus "$FIX_DIR" \
    --out "$WORK/corpus_real_fixed.npz" --crnn 1 --parallel 0

echo "== 3/5 train with fixed real data =="
EPOCHS="${EPOCHS:-50}" EXTRA_REPEAT="${EXTRA_REPEAT:-50}" SYNTH_SUBSAMPLE="${SYNTH_SUBSAMPLE:-1500}"
"$PY" tools/train_crnn_seq.py \
    --train "$WORK/corpus_train.npz" \
    --extra "$WORK/corpus_real_fixed.npz" \
    --extra-repeat "$EXTRA_REPEAT" \
    --synth-subsample "$SYNTH_SUBSAMPLE" \
    --out src/crnn_weights.h \
    --epochs "$EPOCHS" \
    --weight-decay 3e-4 \
    --arch bi

echo "== 4/5 rebuild =="
cmake --build build --target mrz_ocr_cnn_tool --target mrz_ocr_cnn_tool_pic -j"$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"

echo "== 5/5 done =="
echo "Run eval to check:"
echo "  python3 tools/eval_all_engines.py --gt tests/gt_photo_passport_official.json --photo-dir tmp/PassPortPhoto --tool build/mrz_ocr_cnn_tool --engines crnn"
