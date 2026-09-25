# 6_mrz_ocr -- MRZ image recognition

Recognises ICAO 9303 TD3 (passport) MRZ strings from raster images.
Pure C (float32 inference), no third-party runtime dependencies.

Two recognition backends share the same band / line / segment stage:

## Pipeline

1. Grayscale + Otsu binarisation
2. Locate the MRZ band by row-wise darkness projection
3. Split into the two TD3 text lines by gap detection
4. Segment each line into characters by vertical projection
5. Recognise per character, then post-process:

### Backend A: Traditional template matching (`mrz_ocr_recognise`)
Normalised correlation against the embedded OCR-B template bank
(37 glyphs: A-Z, 0-9, `<`).

### Backend B: Tiny ConvNet (`mrz_ocr_recognise_cnn`)
A real 2-block convolutional network (not a flattened MLP):

```
input  12 x 16 binary glyph
conv1  3x3, 8 filters, pad=1 -> ReLU
pool1  2x2 max
conv2  3x3, 16 filters         -> ReLU
pool2  2x2 max                -> 2x3x16 = 96
fc1    96 -> 64 (ReLU)
fc2    64 -> 37 (softmax)
```

Row-level inference runs all 44 characters of a line in a single
vectorised batch (no per-char malloc), with:

* **centroid alignment** of the resampled glyph (kills segmenter
  offset drift),
* **ICAO 9303 syntax masking** (numeric-only / alpha-only positions
  zero out impossible logits),
* **Top-2 checksum beam-search** on Line 2 (flip 1-2 low-confidence
  positions until the four ICAO mod-7/3/1 check digits + composite
  check pass).

Training (`tools/train_cnn.py`, pure-numpy + torch ML backend, ~500
samples/class + hard-pair bootstrapping) emits float32 weights to
`src/cnn_weights.h`. Subpixel translation, dilate/erode, contrast and
a C-side-equivalent box downsample make the augmentation track the
real resampler, which is what keeps clean-template accuracy at
37/37 while generalising to the synthetic corpus.

## Performance (108-image corpus)

| method       | pipeline OK | ms p50 | line1 acc | line2 acc |
|--------------|-------------|--------|-----------|-----------|
| traditional  | 108/108     | 2.7    | 88.7%     | 93.1%     |
| cnn          | 108/108     | 3.2    | 83.6%     | 80.0%     |

## Test corpus

`tests/gen_corpus.py` renders >= 100 synthetic MRZ images covering
scales, gaps, skew and noise. `tests/bench.py` regenerates
`bench_results.{md,json,txt}`.

## Build & test

```sh
./run.sh   # cmake + ctest + corpus benchmark
```
