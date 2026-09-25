# 3_face_recognition -- passport-photo hashing and matching

Pure C99 image utilities for the passport-photo pipeline. No third-party
dependencies (OpenCV is intentionally avoided). Supported inputs:

* Binary PPM (P6) -- our test/sample format
* 24-bit uncompressed BMP (read/write)

Implemented operations:

* PPM/BMP I/O with row-alignment handling
* Grayscale conversion (BT.601 luma)
* Histogram equalisation
* Nearest-neighbour resize
* aHash (8x8 average hash, 64-bit)
* pHash (32x32 -> 4x4-mean -> 8x8 DCT, 64-bit median-thresholded)
* Hamming distance and image similarity score (0..100)

## Build & test

```sh
./run.sh
```

This builds, runs the unit tests, generates three synthetic sample
photos under `data/` and reports similarity scores:

```
face_a vs face_b  -- same seed, expected near-identical (score ~100)
face_a vs face_c  -- different seed, expected to differ
```

## Notes on real biometric data

The synthetic generator in `tools/gen_sample.c` writes deterministic
cartoon-style placeholder faces. To use the library on real passport
photos, drop a P6 PPM or 24-bit BMP into `data/` and call:

```sh
./build/face_tool data/captured.ppm data/reference.ppm 0   # aHash
./build/face_tool data/captured.ppm data/reference.ppm 1   # pHash
./build/face_tool data/captured.ppm data/reference.ppm 2   # both
```

For real biometric verification you would still need a face detector
(e.g. via dlib / OpenCV) and an embedding network; this module
provides only the image-loading and perceptual-hash matching layer.
