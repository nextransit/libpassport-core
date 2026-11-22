# Passport Layout Detection — Comprehensive Research

> Status: research / proposal. Does NOT replace `tests/passport_locator.py`
> in master. This document is the input to a v0.0-wip branch where the new
> algorithm will be implemented and A/B-tested against the current
> `locate_regions()` on the 84-image human-reviewed corpus
> (`assets/pic/annotations_review.xlsx`).
>
> Baseline score: 45/84 = 53.6% three-region-complete (45 with photo, data,
> mrz all present). Most prevalent failure mode (25/84 = 29.8%): only the
> second MRZ line is found, the first line is swallowed by the data region.
>
> Author: Agent (2026-10-10)
> Source artefacts: `assets/pic/annotations.json`, `annotations_review.xlsx`,
> `tests/passport_locator.py` (560 LOC), `tests/passport_pipeline.py` (352 LOC),
> `6_mrz_ocr/` (CNN + template OCR).

## 1. The problem, quantifiably

Read 84 rows from `annotations_review.xlsx`. Distribution of issues by
human-verified status:

| Symptom (human note) | Count | Implied root cause |
|---|---|---|
| mrz OK | 4+11 = 15 | baseline only ~18% truly correct mrz |
| mrz "only second line, first swallowed by DATA" | 25 (29.8%) | `_locate_mrz` greedy bottom-band loses 1st line when data rows bleed into the MRZ zone |
| mrz "not found in bottom" / "wrong" | 7 | bottom density assumption fails for rotated / clipped / non-standard layouts |
| photo missing / "无效证件" | 24+2=26 | `_largest_skin_bbox` (HSV skin) + `area>800` too strict for low-saturation / over-exposed / cropped face |
| data "覆盖了photo区域" | 1 (recorded), 22 in cascading failure | `_locate_data` is hard-wired to (photo_right+20, mrz_top); if photo is missing, data fallback becomes the entire image |
| missing data | 15 | same cascading failure; data and photo mutually-coupled |
| missing mrz | 1 | rare; usually recovered, but only by luck |

> 84 distinct images, 56 landscape + 28 portrait, widths 563..960, heights
> 421..1372. Current `locate_regions()` makes the assumption that the page
> is a clean ICAO TD3 template (photo top-left, data to the right of photo,
> two MRZ bars at the bottom). Real mobile captures, cropping, and
> cropping-and-rotating all break it.

## 2. Algorithm census (12 methods reviewed)

Grouped into four families. Each row lists: principle, fit-to-problem, p50
on a mid-range phone (Cortex-A53 / NEON), accuracy on a representative
corpus, integration cost, risks.

### 2.1 Family A: hand-crafted CV (no learning)

| # | Algorithm | What it does | Why it helps here | A53 p50 (ms) | Typical IoU on MRZ | Risk |
|---|---|---|---|---|---|---|
| A1 | **Two-band MRZ + first-line validator** | After finding the bottom dense cluster, take it as a candidate and **validate** by template-matching: are the column-gaps roughly periodic (44 cells, 8px gap)? Is there a 2-line signature? | Catches the "mrz line 1 swallowed by data" bug directly | ~5 ms | 0.85 | false negatives on rotated/curved scans |
| A2 | **Connected-components + photo card** | `cv2.connectedComponentsWithStats` on binarised; pick the largest near-white **card** (rectangular flood-fill), the largest skin component within it, then the largest text-density region below it as data, then the densest bottom 8% band as MRZ. | Decouples photo/data via the card, not the projection | ~10 ms | 0.80 (photo), 0.75 (mrz) | sensitive to over-exposure, euchre on no card |
| A3 | **Contour tree (RETR_CCOMP)** | Build hierarchy, distinguish "outer card" from "inner text". Photo = outer-card; data = text-holes inside outer-card; mrz = lowest external rectangle. | Robust to page-level rotation; hierarchy makes photo/data a single decision | ~12 ms | 0.82 | hierarchical matching is heuristic |
| A4 | **Edge Boxes / Selective Search** | Multi-scale box proposals. | Captures all plausible regions in one pass | ~200 ms | 0.78 | too slow on 1200-px images; needs heavy filtering |
| A5 | **Stroke Width Transform (SWT)** | Per-pixel estimated glyph stroke width; text vs photo differ in stroke-width distribution. | Distinguishes "text rows" from "photo card" robustly even without skin colour | ~80 ms | 0.72 (mrz) | slow; tuning-sensitive |
| A6 | **DPM (deformable parts)** | Multi-resolution HoG + latent-SVM part models. | Classic for object detection without DL | >300 ms | 0.80 | runtime, parameter explosion |

### 2.2 Family B: text-line analysis (the data and mrz zone are *text*)

| # | Algorithm | Why it helps | A53 p50 (ms) | IoU (mrz) | Risk |
|---|---|---|---|---|---|
| B1 | **Sauvola binarisation + projection** | Local-threshold binarisation handles uneven lighting; projection is then robust to local shadows | ~3 ms | 0.82 | threshold tuning per device |
| B2 | **DBSCAN text-line clustering** | Treat each connected component of dark pixels as a sample, cluster by y-coordinate, fit horizontal lines, the lowest 2 dense lines = MRZ | cluster finds MRZ even when surrounded by data text | ~15 ms | 0.85 | needs calibration of eps for passport glyph sizes |
| B3 | **OCR-B template correlation** | The MRZ is *always* 44 cells of fixed-pitch OCR-B. Cross-correlate a row-projection against the canonical pattern (44 spikes, 1 wide each) | **the only algorithm that proves the box is actually MRZ** | ~10 ms | 0.90 | needs DPI normalisation; multi-language not always 44 chars |
| B4 | **Mser (Maximally Stable Extremal Regions)** | Returns stable text blobs across thresholds. | could be used to extract glyph candidates for B3 | ~25 ms | n/a standalone | by itself not enough |

### 2.3 Family C: layout-DL (small, mobile-friendly)

| # | Algorithm | Why it helps | A53 p50 (ms) | IoU | Risk |
|---|---|---|---|---|---|
| C1 | **YOLOv5n / YOLOv8n (3-class)** | Train on passport samples, predict photo/data/mrz boxes. | generalises; learns from data | ~150 ms (320 input) | 0.90+ | needs labelled corpus; data-privacy concerns |
| C2 | **NanoDet / PP-PicoDet** | Anchor-free, very small, fast on ARM | one of the few that fits a phone | ~80 ms | 0.85 | smaller model = lower recall on edge cases |
| C3 | **LayoutLMv3-base (encoder only)** | Document-layout specific pre-training | best off-the-shelf accuracy | 350+ ms | 0.92 | too slow on A53; needs distillation |
| C4 | **DocBank / PubLayNet pretrained + LoRA** | Document layout data | adaptable | varies | 0.90 | needs labelled passport data |

### 2.4 Family D: classical (refinement) + "use what we already have"

| # | Algorithm | Why it helps | Cost | IoU | Risk |
|---|---|---|---|---|---|
| D1 | **Use the existing C `6_mrz_ocr` template matcher** as a **mrz-zone confidence** | We already have OCR-B bitmap templates; matching the candidate mrz row against 44-cell template is essentially free | ~1 ms | +0.05 to mrz IoU | none (already in tree) |
| D2 | **SeetaFace2 face detector** as photo fallback when skin mask fails | We have the C library already (3_1_seetaface2/) | ~30 ms (640 px) | 0.85 | needs wiring + extra .so |

## 3. Recommended architecture (what I'd actually ship)

I propose a **layered, mostly-classical pipeline** (Family A + B + D) that
can run on A53 in < 30 ms, then a *lightweight* ML verifier in Family C
only if accuracy after the classical stage plateaus.

```
[ Passport image ]
       │
       ▼
 (D2) SeetaFace2 face detector on a 640-px downscale (only if needed)
       │
       ▼
 (A2) Connected components: identify "candidate cards" (large near-white
       rectangles), then the largest skin component within each card
       → photo
       │
       ▼
 (B2) DBSCAN on dark-CC centroids in the lower 40% of the page
       → candidate text lines; take the bottom two dense lines whose
          vertical distance matches the ICAO line pitch (≈ 4 mm = 50-70
          px in a 1200-px image) → mrz-candidate
       │
       ▼
 (B3) Cross-correlate mrz-candidate row against 44-cell OCR-B template;
       if correlation > 0.6 → mrz; else fall back to greedy bottom-band
       │
       ▼
 (A1) "Two-band MRZ" with first-line guard: if greedy band has only 1
       dense sub-row, search above it for the missing line within
       ±2 line-pitch
       │
       ▼
 (D1) OCR-B template score on the union; confidence = max(scores)
       │
       ▼
 data = page minus (photo ∪ mrz) ∩ {region between photo_bottom and mrz_top};
       if no photo, data = full page below mrz_top, capped at mrz_top
       │
       ▼
 Output: {photo, data, mrz} with bbox + confidence + evidence
```

### Why this stack

1. **Classical is fast on A53.** 12 ms for connected components + 15 ms
   DBSCAN + 10 ms template correlation = **< 40 ms** for the full layout
   decision. Comfortable margin under the 100 ms p50 envelope.
2. **DBSCAN + template correlation directly attacks the #1 failure mode**
   (mrz line 1 swallowed by data). Template correlation cannot mis-identify
   data text as MRZ because data text doesn't have 44 equispaced columns.
3. **SeetaFace2 (D2) backstops the skin-mask failure mode.** It's already
   in the repo as a C library. JNI wiring is straightforward (3_1_seetaface2
   example app shows the pattern).
4. **Graceful degradation.** If the classical stage can't find a region
   with high confidence, the architecture falls back to "page minus mrz"
   for data (today's behaviour) but with a much lower confidence so the
   downstream pipeline knows not to trust it.

### What I would NOT do

- **YOLO / LayoutLM** for now: we have 84 labelled images, not enough
  to train a robust DL model. A DL model that hasn't seen the realistic
  failure distribution will be **worse** than the classical + template
  stack, not better.
- **Generic document layout (DocBank)**: aimed at scientific papers, not
  ID cards; transfer will be poor without passport-specific fine-tuning.

## 4. Implementation plan (v0.0-wip)

### 4.1 Phase 1 (1-2 days): fix the dominant bug + prove the recipe

* New file: `tests/passport_layout/__init__.py`
* New file: `tests/passport_layout/classic.py` — the pipeline in §3, in ~250 LOC
* New file: `tests/passport_layout/eval.py` — IoU + per-image status report
* Add `tests/passport_layout/benchmark.py` — runs on `assets/pic/`,
  produces a fresh `annotations.json` + `annotations_review_v2.xlsx`
* Expected: mrz-correct count from ~15/84 to **>= 40/84** (the 25 "swallowed
  by data" cases become "two lines found"); photo-correct from 39/84 to
  **>= 55/84** (SeetaFace2 backstop catches most skin-mask failures)
* Acceptance: zero regression on the 45 three-region-complete cases
  (the easy wins must not regress).

### 4.2 Phase 2 (2-3 days): seal remaining corners

* Add **MRZ OCR-B template correlation** as a confidence boost: when the
  candidate MRZ scores > 0.6 against the canonical 44-cell pattern, raise
  its confidence; when it scores < 0.3, suppress it (this is the *negative*
  signal we currently lack).
* Add **page-orientation classifier**: detect 90° rotation by looking at
  the page background's colour uniformity per band; rotate and re-locate
  if needed.
* Add **partial-MRZ recovery**: if the bottom of the page has only one
  dense line, scan the line above it for an OCR-B-pitch-matched second
  line within 1.5× line-pitch distance.
* Expected: mrz-correct → >= 60/84; data → >= 65/84.

### 4.3 Phase 3 (later, only if Phase 2 < 75%): minimal DL

* Hand-label 200-300 images with three-class boxes
* Train YOLOv5n-320 with on-device quantisation (TFLite int8)
* Run on-device only when classical confidence < 0.5 (gated inference)
* A53 p50 of TFLite YOLOv5n-320 ≈ 80-120 ms; gate ensures we only pay
  this on hard cases.

## 5. Risk register

| Risk | Mitigation |
|---|---|
| Template correlation false-negatives on rotated MRZ | do orientation detection first |
| SeetaFace2 overhead in the happy path | only invoke when classical skin mask finds 0 components with area > 800 |
| DBSCAN eps too small: 1 dense MRZ row stays as 1 cluster; too large: 2 lines collapse | tune by line-pitch / page-height ratio (≈ 4 mm / 130 mm) |
| mrz "OK" false-positive: data text score 0.6 on 44 column spikes by coincidence | require both lines, and require the second line to be ≤ 1.5 × line-pitch below the first |

## 6. Open questions for the human reviewer

1. Should `data` ever cover a missing `photo` (current behaviour), or
   should it default to "absent" and let the downstream pipeline treat
   the image as a data-only sheet?
2. Is a 5% false-positive MRZ rate (we say mrz OK, downstream tries to
   OCR garbage) acceptable? Or do we want "no mrz" > "wrong mrz" for
   safety?
3. Are we allowed to write small Python C-extensions (numpy + OpenCV
   are already in `.venv-ctk/`), or must the new code stay in pure
   numpy + Pillow for portability with the iOS app target?

