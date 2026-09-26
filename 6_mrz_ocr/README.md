# 6_mrz_ocr -- MRZ image recognition

Recognises ICAO 9303 TD3 (passport) MRZ strings from raster images.
Pure C (float32 inference), no third-party runtime dependencies.

Two recognition backends share the band / line / segmentation stage.
The CNN backend is the product path; the template backend is kept as a
legacy A/B comparison.

## Pipeline (CNN backend)

1. Grayscale + global Otsu -> coarse bin map, locate the MRZ band by
   row-density projection
2. **Band-local refinement**: crop the band's grayscale, re-run Otsu
   inside the band only (robust to page-level illumination gradients),
   then estimate the band tilt from the ink-pixel rotation projection
   and **deskew** when |angle| >= 0.4 deg
3. Split into the two TD3 text lines by gap detection
4. **Pitch-grid segmentation**: a TD3 line is exactly 44 monospaced
   cells, so the segmenter fits pitch (median run-center difference,
   autocorrelation fallback) and phase ("run i is cell j" hypotheses)
   and cuts 44 fixed windows. One spurious or merged ink run can no
   longer shift every subsequent character; the old per-run windows
   remain as fallback.
5. Per cell: area-resample the grayscale window to 16x12 float,
   3x3 box smooth, centroid alignment
6. Whole-row batch conv+fc inference; optional +-1px test-time
   augmentation (env `MRZ_OCR_TTA=1`)
7. **Strict ICAO 9303 syntax mask** on the logits: doc-type line
   (col 1 filler, col 2-4 alpha issuer, name field alpha + '<'), and
   line 2 field sets (passport number alnum, check digits numeric,
   nationality alpha, dates numeric + '<', sex M/F/<, personal number
   alnum, tail check digits numeric + '<'). `MRZ_OCR_STRICT_ICAO=0`
   restores the historical loose mask.
8. Deterministic protections: **relative ink gate** ('<' is the
   lowest-ink glyph; X/R predictions on light patches are flipped,
   threshold = 0.55 x median ink of confident letter cells), the
   3-neighbour '<' rule, and the line-1 trailing-filler lock.
9. Line-2 checksum beam search: **top-3 candidates, lowest-margin
   positions first**, singles then pairs, until the five ICAO mod-7/3/1
   check digits validate.

```
input  12 x 16 grayscale float (ink-high, soft edges)
conv1  3x3, 8 filters, pad=1 -> ReLU
pool1  2x1 max (height only)          -> 6 x 16 x 8
conv2  3x3, 16 filters, pad=1 -> ReLU
pool2  2x2 max                        -> 3 x 8 x 16
conv3  1x1, 4 filters                 -> 3 x 8 x 4 = 96
fc1    96 -> 64 (ReLU)
fc2    64 -> 37 (softmax)
```

All large buffers (weights, glyph batch, probabilities) live on the
heap, not the stack (~110 KB of former stack frames removed — relevant
for embedded thread stacks).

## Training & evaluation protocol

The historical 100.0%/99.5% numbers were measured on the same 500
images the model trained on (85% of them, literally) and are invalid.
The protocol is now:

* **Two disjoint, deterministic corpora** (`tests/gen_corpus.py`):
  `data/corpus_eval` (seed 2002, frozen — bench + gate input, never
  trained on) and `data/corpus_train` (seed 1001 — training only).
  Every record renders with a fixed noise seed, so corpora are
  bit-reproducible and benchmark results comparable.
* **ICAO-valid ground truth everywhere**: line 1 name fields are
  A-Z/'<' only; line 2 carries real check digits. The strict syntax
  mask always contains the truth.
* **realistic profile**: rotation +-3 deg, blur, illumination gradient,
  invert, noise up to 0.05 — in both corpora (eval: 660/1380).
* **Distribution-matched training** (`tools/train_cnn_real.py`): runs
  the C pipeline with `MRZ_OCR_DUMP=1` over the TRAIN corpus, trains
  on the exact 16x12 glyphs the network sees, splits by image, cleans
  /tmp dumps before every run (stale-dump poisoning fixed). Optional
  `--mix-synthetic N` blends v2-renderer samples.
* **Regression gate** (`tests/bench.py --gate data/baseline.json`):
  fails the build on >0.3pp regression of any key metric or below
  absolute floors; refreshes the baseline on pass.

## Closed loop

```sh
./loop.sh              # build -> corpora -> unit tests -> bench -> gate -> mine
./loop.sh --retrain    # additionally retrain on fresh train-corpus dumps
python3 tools/mine_failures.py --method cnn --suggest   # failure mining
```

Each bench appends to `metrics_history.md` and writes
`failures_{method}.json`; `mine_failures.py` clusters failures by
rendering condition, column and confusion pair to drive the next
iteration.

## Performance (frozen eval corpus, 1380 images, honest split)

See `bench_results.md` / `metrics_history.md` for the current numbers
and `data/baseline.json` for the gated baseline.

## Test corpus

`tests/gen_corpus.py --eval|--train --out DIR --seed N` renders the
corpora (clean + realistic profiles, deterministic). `tests/bench.py`
regenerates `bench_results.{md,json,txt}` and `failures_*.json`.

## Build & test

```sh
./run.sh     # build + unit tests + golden test + sample
./loop.sh    # full train/test/fix loop with regression gate
```
