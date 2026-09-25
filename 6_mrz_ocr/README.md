# 6_mrz_ocr -- MRZ image recognition

Recognises ICAO 9303 TD3 (passport) MRZ strings from raster images.
Pure C, no third-party dependencies.

## Pipeline

1. Grayscale + Otsu binarisation
2. Locate the MRZ band by row-wise darkness projection
3. Split into the two TD3 text lines by gap detection
4. Segment each line into characters by vertical projection
5. Match each character against the embedded OCR-B template bank
   (37 glyphs: A-Z, 0-9, `<`) using normalised correlation

The recognised lines can be fed straight into `mrz_td3_decode` from
module 1 to verify the ICAO check digits.

## Test corpus

The Python helper `tests/gen_corpus.py` renders >= 100 synthetic
MRZ images covering various scales, gaps, skew and noise levels.
Run `python3 tests/test_corpus.py` to compute line-level accuracy.

## Build & test

```sh
./run.sh
```
