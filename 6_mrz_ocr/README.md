# 6_mrz_ocr -- MRZ image recognition

Recognises ICAO 9303 TD3 (passport) MRZ strings from raster images.
Pure C (float32 inference), no third-party runtime dependencies.

Three recognition backends share the band / line / segmentation stage.

| backend | how it classifies | when to use |
|---|---|---|
| **CNN** (`mrz_ocr_recognise_cnn`) | per-character CNN head only | **the shipping path** |
| fused (`mrz_ocr_recognise_fuse`) | the per-character CNN head AND the CRNN sequence head, combined in log space with a confidence gate | fallback only (not default), see the caveat below |
| template (`mrz_ocr_recognise`) | template-bank matching | legacy A/B only |

The fused engine exists because the two heads fail on DIFFERENT cells,
but on the current corpus fusion is a net loss.  Scored against the
human-corrected truth in `data/gt_frozen_passport_corrected.json` (41
photos, same band, same weights):

```
per-character CNN : 37/41 line2 exact
fused             : 31/41 line2 exact
```

When both heads decode, fusion can overwrite a checksum-beam repair made
by the CNN with the raw top-1 character.  On six photos (E/G/K/Q/T/AJ)
that changed a correct CNN line into a wrong one.  The earlier
22/41-vs-21/41 cross-validation number came from the board's noisy raw
labels; the corrected truth reverses the ranking.

Fusion is therefore not the default: ship the plain CNN and only enable
`MRZ_OCR_FUSE=1` for a deployment after a frozen-GT A/B.  On the frozen
assets set the gap is larger: 8/32 exact for fused against 21/32 for the
plain CNN.  See `../outputs/crnn_optimization_2026-10-23.md` for the
background and `tools/cv_crnn.py` for the cross-validation harness.

### End-to-end on the frozen corpus (shipped path: CNN first, fuse fallback)

`tools/eval_prod_path.py` drives the full photo-to-verdict pipeline over
the 41 frozen photos:

```
PASS     40/41
SUSPECT   1/41   (F — issuer-code suspect; its line2 is exact)
FAIL      0/41
chosen exact  37/41   (CNN first; fuse is chosen 0 times here)
```

The four residual non-exact lines are `M` (personal-number coin-flip at
pos33/37/41 — `...V22S2Z591` vs `...022S7Z593`, both readings
checksum-legal, CNN confidence p=1.0) and the two-character misses
`AC`/`AC_R` (pos28 `0`-vs-`O`, pos40 `1`-vs-`7`) and `AG` (pos28
`1`-vs-`T`, pos39 `0`-vs-`O`).  All four sit in the un-checksummable
personal-number zone (pos28-41).  There is no margin/ambiguity gate that
separates them from the correct low-margin beam repairs without also
rejecting five correct photos (A/W/I/AB/AI), so they ship as a documented
limitation.

Fusion knobs (environment, for field A/B): `MRZ_OCR_FUSE=1`,
`MRZ_OCR_FUSE_GATE` (absolute CRNN-confidence threshold, default 0),
`MRZ_OCR_FUSE_RATIO` (CRNN must be this many times as confident as the
CNN, default 1.0), `MRZ_OCR_FUSE_ALPHA` (weight of the CRNN log term,
default 0.5).

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
