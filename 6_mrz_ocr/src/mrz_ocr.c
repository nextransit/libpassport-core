/* MRZ OCR pipeline.
 *
 * Pipeline:
 *   1. Load image (already loaded by caller)
 *   2. Convert to grayscale + binarise with a global Otsu-like threshold
 *   3. Locate the MRZ band: scan rows, find horizontal runs of dense
 *      dark pixels separated by gaps that match the ICAO layout (two
 *      rows of 44 characters, ~1.0 mm gap between lines, ~1.0 mm
 *      horizontal inter-character gap).
 *   4. Crop the band; rotate it (we accept only near-horizontal here)
 *   5. For each of the two text lines, segment characters by vertical
 *      projection and recognise each segment with a normalised cross-
 *      correlation against the embedded OCR-B template bank.
 *
 * This module is intentionally simple: it works on synthetic test
 * images produced by `tools/gen_mrz_image.c` and on printed MRZs
 * scanned at >= 300 dpi with a near-horizontal placement.
 */
#include "mrz_ocr.h"
#include "template.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifndef ABS
#define ABS(x) ((x) < 0 ? -(x) : (x))
#endif

static _Thread_local struct timespec g_mrz_deadline;
static _Thread_local int g_mrz_deadline_enabled;

static uint64_t mrz_now_ns(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (uint64_t)now.tv_sec * 1000000000ULL + (uint64_t)now.tv_nsec;
}

void mrz_ocr_deadline_set_ms(int timeout_ms) {
    if (timeout_ms <= 0) {
        g_mrz_deadline_enabled = 0;
        return;
    }
    uint64_t deadline = mrz_now_ns() + (uint64_t)timeout_ms * 1000000ULL;
    g_mrz_deadline.tv_sec = (time_t)(deadline / 1000000000ULL);
    g_mrz_deadline.tv_nsec = (long)(deadline % 1000000000ULL);
    g_mrz_deadline_enabled = 1;
}

int mrz_ocr_deadline_expired(void) {
    if (!g_mrz_deadline_enabled) return 0;
    uint64_t deadline = (uint64_t)g_mrz_deadline.tv_sec * 1000000000ULL +
                        (uint64_t)g_mrz_deadline.tv_nsec;
    return mrz_now_ns() >= deadline;
}

mrz_ocr_status_t mrz_ocr_init(void) { return MRZ_OCR_OK; }

void mrz_ocr_result_free(mrz_ocr_result_t *r) { (void)r; }

const char *mrz_ocr_strerror(mrz_ocr_status_t s) {
    switch (s) {
        case MRZ_OCR_OK:                 return "OK";
        case MRZ_OCR_ERR_LOAD:           return "image load error";
        case MRZ_OCR_ERR_NO_BAND:        return "could not locate MRZ band";
        case MRZ_OCR_ERR_BAD_GEOMETRY:   return "bad MRZ band geometry";
        case MRZ_OCR_ERR_BAD_LINES:      return "could not split into two lines";
        case MRZ_OCR_ERR_LOW_CONFIDENCE: return "average confidence too low";
        case MRZ_OCR_ERR_TEMPLATE:       return "template bank not initialised";
        case MRZ_OCR_ERR_TIMEOUT:         return "deadline exceeded";
    }
    return "unknown";
}

/* ---------- 1. Binarisation (Otsu threshold). ---------- */
/* Shared Otsu implementation. Both the traditional and the CNN backend
 * binarise through this entry point (global for band location, and
 * band-local as a refinement pass). */
int mrz_ocr_otsu(const uint8_t *gray, int n) {
    size_t hist[256] = {0};
    for (int i = 0; i < n; ++i) hist[gray[i]]++;
    double sum = 0;
    for (int i = 0; i < 256; ++i) sum += (double)i * hist[i];
    double sumB = 0;
    int wB = 0;
    double maxVar = -1;
    int threshold = 127;
    for (int t = 0; t < 256; ++t) {
        wB += (int)hist[t];
        if (wB == 0) continue;
        int wF = n - wB;
        if (wF == 0) break;
        sumB += (double)t * hist[t];
        double mB = sumB / wB;
        double mF = (sum - sumB) / wF;
        double varBetween = (double)wB * wF * (mB - mF) * (mB - mF);
        if (varBetween > maxVar) {
            maxVar = varBetween;
            threshold = t;
        }
    }
    /* Otsu on a two-tone (0/255) image would always pick 0; nudge it
     * up to the midpoint if the first bins are non-empty and dominant
     * (== the pure background case).  hist[0] alone is a single-pixel
     * fragility: a ±1 JPEG-decoder luma difference can move the one
     * pure-black pixel to value 1 and silently flip a 127 back into a
     * full Otsu pick (157/158), shifting every glyph window.  Aggregating
     * the lowest three bins makes the case robust to ±1 luma jitter. */
    if (hist[0] > 0 && hist[255] == 0) {
        return 127;
    }
    if (threshold == 0 && hist[255] > 0) {
        /* We must split the dominant white from foreground. */
        return 127;
    }
    return threshold;
}

/* ---------- 2. Locate MRZ band by row-wise density. ---------- */
typedef struct {
    int x, y, w, h;
} rect_t;

int mrz_ocr_locate_band(const uint8_t *bin, int W, int H, mrz_ocr_rect_t *out) {
    /* Project darkness: row_dark[y] = # dark pixels in row y. */
    int *row_dark = (int *)calloc((size_t)H, sizeof(int));
    if (!row_dark) return -1;
    /* Threshold: a row is "dense" if at least 15% of its pixels are ink. */
    int threshold = (int)(W * 0.10);
    /* Rows that are ~full-width ink are device/取景框 black bars, not MRZ
     * rows (real MRZ ink peaks far below W; e.g. 0.4-0.65 W on real
     * photos). Counting them as dense runs lets the black bar merge into
     * the band and inflates the row-dark peak, which then breaks
     * split_lines()' gap thresholds (gap_thr = peak/6). Exclude them
     * from run detection so the band and the peak stay text-only. */
    int max_text_dark = (int)(W * 0.90);
    for (int y = 0; y < H; ++y) {
        int c = 0;
        for (int x = 0; x < W; ++x) if (bin[y * W + x]) c++;
        row_dark[y] = c;
    }
    /* Find ALL dense runs; we then pick a band that is the union of two
     * dense runs separated by a relatively small gap (the inter-line
     * gap of a TD3 record). The simplest robust rule: scan for runs,
     * collect runs of height >= 2, and merge consecutive runs whose
     * vertical gap is <= H/8 (the inter-line gap).
     *
     * min_run_h MUST stay tiny: a '<'-heavy line has a natural density
     * valley at template row 1 ('<' has no ink there), which can split
     * the line's dense run. Discarding short runs used to chop the top
     * two template rows off every line-1 glyph (O -> U misreads). */
    int *run_top = (int *)malloc(sizeof(int) * H);
    int *run_bot = (int *)malloc(sizeof(int) * H);
    int n_runs = 0;
    int top = -1;
    int min_run_h = 2;
    int max_gap = H / 8;
    int max_run_h = 0;
    for (int y = 0; y < H; ++y) {
        if (row_dark[y] >= threshold && row_dark[y] < max_text_dark) {
            if (top < 0) top = y;
        } else {
            if (top >= 0 && y - top >= min_run_h) {
                run_top[n_runs] = top;
                run_bot[n_runs] = y;
                if (y - top > max_run_h) max_run_h = y - top;
                n_runs++;
            }
            top = -1;
        }
    }
    if (top >= 0 && H - top >= min_run_h) {
        run_top[n_runs] = top; run_bot[n_runs] = H;
        if (H - top > max_run_h) max_run_h = H - top;
        n_runs++;
    }
    /* The inter-line gap of a TD3 record scales with the character
     * height, not with the crop height. A TIGHT crop (caller has
     * already located the MRZ, so H is small) makes H/8 under-shoot
     * the real gap (~1.4x line height) and the two text lines stay
     * separate runs -> the band degenerates to one line and both
     * "lines" come out of the same physical line. Scale the merge
     * gap by the tallest run found as well; on full-page inputs H/8
     * dominates and behaviour is unchanged. */
    if (2 * max_run_h > max_gap) max_gap = 2 * max_run_h;
    if (n_runs == 0) {
        /* No dense horizontal run anywhere (blank / uniform image).
         * Must bail BEFORE touching run_top[0]: the arrays are malloc'd
         * and unwritten at this point. */
        free(row_dark); free(run_top); free(run_bot); return -1;
    }
    /* Find the merged run group whose total height is the tallest. */
    int best_top = run_top[0], best_bot = run_bot[0], best_h = run_bot[0] - run_top[0];
    for (int i = 0; i < n_runs; ++i) {
        int merged_top = run_top[i];
        int merged_bot = run_bot[i];
        for (int j = i + 1; j < n_runs; ++j) {
            if (run_top[j] - merged_bot > max_gap) break;
            merged_bot = run_bot[j];
        }
        int h = merged_bot - merged_top;
        if (h > best_h) {  /* accept any merged height */
            best_h = h;
            best_top = merged_top;
            best_bot = merged_bot;
        }
    }
    free(run_top); free(run_bot);
    /* Horizontal extent. */
    int left = W, right = -1;
    for (int y = best_top; y < best_bot; ++y) {
        for (int x = 0; x < W; ++x) {
            if (bin[y * W + x]) {
                if (x < left) left = x;
                if (x > right) right = x;
            }
        }
    }
    free(row_dark);
    if (right < left) return -1;
    /* Extend the horizontal extent by a small margin (a few px) so the
     * last glyph's full cell / pitch is included; the ink-only bbox can
     * otherwise clip the rightmost character by ~1/3 of a cell. */
    int margin = 8;
    if (margin > W / 4) margin = W / 4;
    left  -= margin;
    right += margin;
    if (left  < 0) left = 0;
    if (right >= W) right = W - 1;
    out->x = left; out->y = best_top;
    out->w = right - left + 1; out->h = best_bot - best_top;
    return 0;
}

/* ---------- 3. Split the band into the two TD3 lines. ---------- */
int mrz_ocr_split_lines(const uint8_t *bin, int W, int H, mrz_ocr_rect_t *out) {
    /* Project darkness on each row of the band. */
    int *row_dark = (int *)calloc((size_t)H, sizeof(int));
    if (!row_dark) return -1;
    for (int y = 0; y < H; ++y) {
        int c = 0;
        for (int x = 0; x < W; ++x) if (bin[y * W + x]) c++;
        row_dark[y] = c;
    }
    int peak = 0;
    /* Exclude near-full-width dark rows (black bars / page edges) from
     * the peak: they inflate gap_thr = peak/6 and prevent real text
     * gaps from being detected.  Same rationale as locate_band()'s
     * max_text_dark.  Threshold 0.75*W: real MRZ ink peaks at 0.4-0.65*W,
     * black bars / page edges reach 0.75-0.90*W. */
    int max_text_dark = (int)(W * 0.75);
    for (int y = 0; y < H; ++y)
        if (row_dark[y] > peak && row_dark[y] < max_text_dark) peak = row_dark[y];
    if (peak == 0)  /* all rows are black bars -- degenerate, use raw peak */
        for (int y = 0; y < H; ++y)
            if (row_dark[y] > peak) peak = row_dark[y];

    /* Step 1: find the split point with the proven deepest-gap scan. A
     * row is "gap" when its ink count is a small FRACTION OF THE TEXT
     * ROWS' PEAK -- a fixed fraction of W breaks under salt-and-pepper
     * noise (gap rows pick up specks: at 3% noise a 1518 px row collects
     * ~23 specks, above the old W/100 = 15 threshold). */
    int gap_thr = peak / 6;
    if (gap_thr < 4) gap_thr = 4;
    int best_gap_y = -1, best_gap_score = 0;
    int min_half = H / 5;
    for (int y = 1; y < H - 1; ++y) {
        int gap_h = 1;
        while (y + gap_h < H && row_dark[y + gap_h] < gap_thr) gap_h++;
        if (gap_h < H / 25) continue;
        if (y < min_half || (H - (y + gap_h)) < min_half) continue;
        if (gap_h > best_gap_score) {
            best_gap_score = gap_h;
            best_gap_y = y;
        }
    }
    if (best_gap_y >= 0) {
        int h1 = best_gap_y;
        int h2 = H - (best_gap_y + best_gap_score);
        /* TD3's two lines are the SAME font at the SAME size, so their
         * heights must be comparable. Without this check the deepest
         * gap is often the one between the top border / guilloche and
         * line 1: on document scans that produced a 18:54 split, i.e.
         * line 1 became the decorative border and every glyph in it was
         * misread (measured on assets/pic: line1 0/84 exact). Rejecting
         * the pair here lets step 3 look for two comparable runs
         * instead of blindly trusting the deepest valley. */
        if (h1 * 2 >= h2 && h2 * 2 >= h1) {
            /* DEFECT, FIXED HERE -- but only together with the retrain
             * that shipped with it: line 1's box was one row short on
             * every image.
             *
             * The split row is the first row whose ink drops below
             * gap_thr = peak/6, i.e. "few enough glyphs reach this
             * row".  For MRZ that is the row carrying the baseline
             * stroke -- and only a handful of glyphs have one (E, L,
             * G, the bottom of 0/O/Q), so it reads as a gap and line
             * 1's box loses it, while line 2's box cannot (it is
             * bounded by H).  Measured on synthetic renders of the
             * ICAO specimen:
             *
             *   scale 3  line0 h=35  line1 h=36   (glyph is 36)
             *   scale 4  line0 h=47  line1 h=48
             *   scale 5  line0 h=59  line1 h=60
             *   scale 6  line0 h=71  line1 h=72
             *
             * One row short, every time.  After the 16x12 resample
             * that is a whole cell row of the glyph, and it is the row
             * that separates E from F (test_golden fails at scale 3
             * with exactly that confusion) and L from 1.
             *
             * Reclaiming the row from the gap fixes test_golden
             * (32/32) and leaves line 2 bit-identical.  It MUST NOT be
             * turned on by itself: every weight set that existed before
             * this change was trained on glyphs resampled through the
             * 35-row box, so re-framing only at inference is a
             * train/inference mismatch (measured: line 1 changes in
             * 30/41 photos and "issuer != nationality" goes 1 -> 11,
             * CHN read as CAN once the extra row is included).
             *
             * It is therefore enabled together with a full retrain, in
             * this order:
             *   1. export training slices through THIS box
             *      (tools/export_omtest_slices.py,
             *       tools/export_photo_train.py);
             *   2. retrain cnn_weights_line1.h (line 2's slices are
             *      bit-identical, so its head is untouched);
             *   3. re-run tools/rebuild_crnn.sh -- the features change
             *      because line 1's framing changed;
             *   4. validate on the 41 photos + test_golden.
             *
             * The reclaim is bounded: at most the height deficit
             * against line 2 (which bounds a same-size second line),
             * at most 2 rows, and only while the row still carries a
             * real fraction of the line's peak ink.  "Carries any ink"
             * is not enough -- on a real photo the inter-line gap holds
             * guilloche and speckle, and a plain >0 test pulls a noise
             * row into line 1 (measured: 31/41 photos changed, mostly
             * worse).  A baseline stroke is peak/20 or more; a speckle
             * row is a handful of pixels. */
            {
                /* MRZ_OCR_FRAME_FIX=1 enables the reclaim; the default is
                 * OFF.  The shipped weights were trained on the historical
                 * 35-row box, so enabling the reclaim without the matching
                 * full retrain (export slices -> retrain cnn_weights_line1.h
                 * -> tools/rebuild_crnn.sh) is a train/inference mismatch:
                 * measured 30/41 photos change and "issuer != nationality"
                 * goes 1 -> 11.  Measured on the 114-photo bench with the
                 * shipped weights: frame fix on  -> fuse decode 92/106 but
                 * GT-exact-both 8/114; frame fix off -> fuse decode 92/106
                 * and GT-exact-both 14/114 (same as the 0ab2131 baseline).
                 * Keep it opt-in until a clean full retrain exists. */
                const char *ff = getenv("MRZ_OCR_FRAME_FIX");
                int frame_fix = (ff && ff[0] == '1');
                int take = 0;
                if (frame_fix) {
                    int deficit = h2 - h1;
                    int spare = best_gap_score - 1;
                    int ink_min = peak / 20;
                    if (deficit > 2) deficit = 2;   /* only an off-by-a-row */
                    if (deficit > spare) deficit = spare;
                    if (ink_min < 2) ink_min = 2;
                    while (take < deficit &&
                           row_dark[best_gap_y + take] >= ink_min)
                        take++;
                }
                out[0].h = best_gap_y + take;
            }
            out[0].x = 0; out[0].y = 0;
            out[0].w = W;
            out[1].x = 0; out[1].y = best_gap_y + best_gap_score;
            out[1].w = W; out[1].h = H - out[1].y;
            free(row_dark);
            return 0;
        }
    }

    /* Step 3 (old path failed: no clean gap -- typically guilloche ink
     * filling the inter-line rows on real photos): line-CORE detection.
     * Group core-dense rows into runs (merging 1-2 row intra-line
     * valleys from the '<' template's empty rows), then pick the
     * adjacent run pair whose gap is comparable to the line height
     * (gap <= 2 x shorter run, min 3) with the most combined ink. The
     * gap rule rejects unrelated content merged near the band (e.g. a
     * dark page edge dozens of rows below line 2); ties go to the
     * bottom-most pair (MRZ sits at the bottom of the crop). */
     int *r_top = (int *)malloc(sizeof(int) * (size_t)(H / 2 + 2));
     int *r_bot = (int *)malloc(sizeof(int) * (size_t)(H / 2 + 2));
     int *r_ink = (int *)malloc(sizeof(int) * (size_t)(H / 2 + 2));
     /* Try progressively higher core thresholds: peak/4 merges inter-line
      * gaps filled with guilloche ink on some photo scans, so the two MRZ
      * rows become one giant run and no pair is found.  peak/3 separates
      * them on those bands while still rejecting noise rows. */
     int core_thrs[] = { peak / 4, peak / 3, peak * 2 / 5 };
     for (int ct_idx = 0; ct_idx < 3; ++ct_idx) {
     int core_thr = core_thrs[ct_idx];
     if (core_thr < 4) core_thr = 4;
     int n_runs = 0;
     int core_ok = (r_top && r_bot && r_ink);
     if (core_ok) {
        int y = 0;
        while (y < H) {
            if (row_dark[y] < core_thr) { y++; continue; }
            int top = y, last = y;
            long ink = 0;
            for (;;) {
                while (y < H && row_dark[y] >= core_thr) {
                    ink += row_dark[y]; last = y; y++;
                }
                if (y >= H) break;
                int z = y;
                while (z < H && row_dark[z] < core_thr) z++;
                if (z < H && z - last <= 2) {
                    for (int q = y; q < z; ++q) ink += row_dark[q];
                    y = z;
                } else {
                    break;
                }
            }
            if (last - top + 1 >= 2 && n_runs < H / 2 + 2) {
                r_top[n_runs] = top; r_bot[n_runs] = last;
                r_ink[n_runs] = (int)ink; n_runs++;
            }
        }
        int best_i = -1;
        long best_score = -1;
        for (int i = n_runs - 2; i >= 0; --i) {
            int gap = r_top[i + 1] - r_bot[i] - 1;
            int h1 = r_bot[i] - r_top[i] + 1;
            int h2 = r_bot[i + 1] - r_top[i + 1] + 1;
            int hmin = h1 < h2 ? h1 : h2;
            int hmax = h1 > h2 ? h1 : h2;
            /* Same rule as step 1: the two MRZ rows are one font at one
             * size, so a 1:3 height pair is a decoration row, not line 1. */
            if (hmax > hmin * 2) continue;
            int max_gap = hmin * 2 > 3 ? hmin * 2 : 3;
            if (gap > max_gap) continue;
            if (h1 + h2 < H / 4) continue;
            long score = (long)r_ink[i] + r_ink[i + 1];
            if (score > best_score) { best_score = score; best_i = i; }
        }
        if (best_i >= 0) {
            out[0].x = 0; out[0].y = r_top[best_i];
            out[0].w = W; out[0].h = r_bot[best_i] - r_top[best_i] + 1;
            out[1].x = 0; out[1].y = r_top[best_i + 1];
            out[1].w = W; out[1].h = r_bot[best_i + 1] - r_top[best_i + 1] + 1;
            free(r_top); free(r_bot); free(r_ink);
            free(row_dark);
            return 0;
        }
    }
    }
    free(r_top); free(r_bot); free(r_ink);
    /* Step 4 (last resort): single merged blob -- split at the deepest
     * valley in the middle half of the band. */
    int valley = -1, valley_ink = 0;
    for (int y = H / 4; y < H - H / 4; ++y) {
        if (valley < 0 || row_dark[y] < valley_ink) {
            valley = y; valley_ink = row_dark[y];
        }
    }
    if (valley >= 0 && valley_ink <= peak / 2) {
    int h1v = valley + 1, h2v = H - (valley + 1);
    if (h1v * 2 >= h2v && h2v * 2 >= h1v) {
    free(row_dark);
    out[0].x = 0; out[0].y = 0;
    out[0].w = W; out[0].h = valley + 1;
    out[1].x = 0; out[1].y = valley + 1;
    out[1].w = W; out[1].h = H - (valley + 1);
    return 0;
    }
    }
    /* Step 5 (relaxed fallback): retry step-3 run-pair matching with
     * looser thresholds (hmax:hmin 3:1, gap 3x hmin).  Only reached
     * when all strict steps failed, so it cannot break existing splits. */
    {
    int *r2_top = (int *)malloc(sizeof(int) * (size_t)(H / 2 + 2));
    int *r2_bot = (int *)malloc(sizeof(int) * (size_t)(H / 2 + 2));
    int *r2_ink = (int *)malloc(sizeof(int) * (size_t)(H / 2 + 2));
    int core_thr = peak / 3;
    if (core_thr < 4) core_thr = 4;
    int n2 = 0;
    if (r2_top && r2_bot && r2_ink) {
        int y = 0;
        while (y < H) {
            if (row_dark[y] < core_thr) { y++; continue; }
            int top = y, last = y;
            long ink = 0;
            for (;;) {
                while (y < H && row_dark[y] >= core_thr) {
                    ink += row_dark[y]; last = y; y++;
                }
                if (y >= H) break;
                int z = y;
                while (z < H && row_dark[z] < core_thr) z++;
                if (z < H && z - last <= 2) {
                    for (int q = y; q < z; ++q) ink += row_dark[q];
                    y = z;
                } else { break; }
            }
            if (last - top + 1 >= 2 && n2 < H / 2 + 2) {
                r2_top[n2] = top; r2_bot[n2] = last;
                r2_ink[n2] = (int)ink; n2++;
            }
        }
        int best_i2 = -1;
        long best_score2 = -1;
        for (int i = n2 - 2; i >= 0; --i) {
            int gap = r2_top[i + 1] - r2_bot[i] - 1;
            int h1 = r2_bot[i] - r2_top[i] + 1;
            int h2 = r2_bot[i + 1] - r2_top[i + 1] + 1;
            int hmin = h1 < h2 ? h1 : h2;
            int hmax = h1 > h2 ? h1 : h2;
            if (hmax > hmin * 3) continue;
            int max_gap = hmin * 3 > 3 ? hmin * 3 : 3;
            if (gap > max_gap) continue;
            if (h1 + h2 < H / 4) continue;
            long score = (long)r2_ink[i] + r2_ink[i + 1];
            if (score > best_score2) { best_score2 = score; best_i2 = i; }
        }
        if (best_i2 >= 0) {
            out[0].x = 0; out[0].y = r2_top[best_i2];
            out[0].w = W; out[0].h = r2_bot[best_i2] - r2_top[best_i2] + 1;
            out[1].x = 0; out[1].y = r2_top[best_i2 + 1];
            out[1].w = W; out[1].h = r2_bot[best_i2 + 1] - r2_top[best_i2 + 1] + 1;
            free(r2_top); free(r2_bot); free(r2_ink);
            free(row_dark);
            return 0;
        }
    }
    free(r2_top); free(r2_bot); free(r2_ink);
    }
    free(row_dark);
    return -1;
}

/* ---------- 3b. Single-row fallback: the band holds ONE MRZ row. ---------- */
/* A CS/CD travel permit (往来港澳/台湾通行证) prints a single 30-character
 * MRZ row.  The two-line splitter above cannot describe that: it either
 * rejects the band outright (which is what used to happen -- the whole
 * pipeline then returned "could not split into two lines" and the permit
 * was unreadable) or slices one physical row in half.  This returns the
 * dominant dense row-group as one box.
 *
 * It is only reached when mrz_ocr_split_lines() failed, and that function
 * fails only when there is no usable valley in the middle half of the
 * band -- i.e. when the band really is one solid row of text.  A TD3 band
 * whose two rows are blurred into one blob cannot reach here, because the
 * valley path would have split it.
 *
 * Deciding WHICH document this is (30 vs 44 characters) is not this
 * function's job: the layout classifier owns that, because it is the only
 * component that looks at the character grid. */
int mrz_ocr_split_lines_single(const uint8_t *bin, int W, int H,
                               mrz_ocr_rect_t *out) {
    if (!bin || !out || W <= 0 || H <= 0) return -1;
    int *row_dark = (int *)calloc((size_t)H, sizeof(int));
    if (!row_dark) return -1;
    for (int y = 0; y < H; ++y) {
        int c = 0;
        for (int x = 0; x < W; ++x) if (bin[y * W + x]) c++;
        row_dark[y] = c;
    }
    int peak = 0;
    for (int y = 0; y < H; ++y) if (row_dark[y] > peak) peak = row_dark[y];
    if (peak < 4) { free(row_dark); return -1; }

    int thr = peak / 6;
    if (thr < 4) thr = 4;

    /* Group ink rows into runs, merging gaps of <= 2 rows: the '<' glyph
     * template has empty rows inside a line, so a strict contiguous test
     * would cut one row into pieces. */
    int best_top = -1, best_bot = -1;
    long best_ink = -1;
    int y = 0;
    while (y < H) {
        if (row_dark[y] < thr) { y++; continue; }
        int top = y, last = y;
        long ink = 0;
        for (;;) {
            while (y < H && row_dark[y] >= thr) { ink += row_dark[y]; last = y; y++; }
            if (y >= H) break;
            int z = y;
            while (z < H && row_dark[z] < thr) z++;
            if (z < H && z - last <= 2) {
                for (int q = y; q < z; ++q) ink += row_dark[q];
                y = z;
            } else break;
        }
        if (last - top + 1 >= 2 && ink > best_ink) {
            best_ink = ink; best_top = top; best_bot = last;
        }
    }
    free(row_dark);
    if (best_top < 0) return -1;

    int h = best_bot - best_top + 1;
    /* The row must dominate the band.  A short fragment is not a document
     * we can read -- let the caller drop the frame instead of guessing.
     * Allow a 3px margin.  Relaxed from h*2 to h*3 for CS/CD single-line
     * bands where padding + eval PAD can make the band up to 3x the text
     * row height.  Further relaxed to h*4 for very thin CS/CD permits
     * whose text is only 7-8 px after binarisation but the band carries
     * padding to 30 px minimum (measured on 2026-9-30-J). */
    if (h * 4 < H - 3) return -1;

    int left = W, right = -1;
    for (int yy = best_top; yy <= best_bot; ++yy) {
        for (int x = 0; x < W; ++x) {
            if (bin[yy * W + x]) {
                if (x < left) left = x;
                if (x > right) right = x;
            }
        }
    }
    if (right < left) return -1;
    out->x = left;
    out->y = best_top;
    out->w = right - left + 1;
    out->h = h;
    return 0;
}

/* Try two rows, then one.  n_lines reports which shape was found; the
 * caller must consult it before touching out[1]. */
int mrz_ocr_split_lines_n(const uint8_t *bin, int W, int H,
                          mrz_ocr_rect_t *out, int *n_lines) {
    if (!out || !n_lines) return -1;
    if (mrz_ocr_split_lines(bin, W, H, out) == 0) {
        *n_lines = 2;
        return 0;
    }
    if (mrz_ocr_split_lines_single(bin, W, H, &out[0]) == 0) {
        *n_lines = 1;
        return 0;
    }
    return -1;
}

/* ---------- 4. Segment a line into characters by vertical projection. ---------- */
int mrz_ocr_segment_line(const uint8_t *bin, int W, int H, mrz_ocr_rect_t *chars, int max_chars) {
    /* column_dark[x] = # dark pixels in column x. */
    int *col_dark = (int *)calloc((size_t)W, sizeof(int));
    if (!col_dark) return -1;
    for (int x = 0; x < W; ++x) {
        int c = 0;
        for (int y = 0; y < H; ++y) if (bin[y * W + x]) c++;
        col_dark[x] = c;
    }
    /* A column is "ink" if more than 8% of pixels are dark. */
    int threshold = H / 12;
    if (threshold < 1) threshold = 1;
    int n = 0;
    int in_run = 0;
    int run_start = 0;
    for (int x = 0; x < W; ++x) {
        if (col_dark[x] >= threshold) {
            if (!in_run) { in_run = 1; run_start = x; }
        } else {
            if (in_run) {
                if (n < max_chars) {
                    chars[n].x = run_start;
                    chars[n].y = 0;
                    chars[n].w = x - run_start;
                    chars[n].h = H;
                    n++;
                }
                in_run = 0;
            }
        }
    }
    if (in_run && n < max_chars) {
        chars[n].x = run_start; chars[n].y = 0;
        chars[n].w = W - run_start; chars[n].h = H; n++;
    }
    free(col_dark);
    return n;
}

/* ============================================================================
 * LEGACY BASELINE - DO NOT ADD NEW FEATURES
 * ----------------------------------------------------------------------------
 * This template-matching backend is the CONTROL GROUP for A/B reporting
 * only. Product/embedded deployment uses the CNN backend exclusively
 * (mrz_ocr_recognise_cnn); the regression gate (FLOORS) binds only CNN.
 * The 8x12 bitwise-XOR matcher, run-length segmentation and the absence
 * of syntax-mask / checksum-beam post-processing are deliberately FROZEN.
 * It may receive shared-preprocessing fixes (deskew) so the benchmark
 * measures the classifier gap, but NO new recognition features.
 * ========================================================================== */
/* ---------- 5. Match one binarised char to the best glyph. ----------
 *
 * We compute a normalised correlation score between the character
 * (resized to 8x12) and each glyph. The resize uses simple box
 * averaging. Score in [0..100]. */
/* Resample a rectangle of `bin` (size W x H) at offsets (rx, ry)
 * (relative to the bin top-left) with size (rw, rh), into an 8x12
 * bitmap using box averaging. */
static int resample_char(const uint8_t *bin, int W, int H,
                         int rx, int ry, int rw, int rh,
                         uint8_t out[MRZ_OCR_GLYPH_H][MRZ_OCR_GLYPH_W]) {
    if (rw <= 0 || rh <= 0) return -1;
    for (int oy = 0; oy < MRZ_OCR_GLYPH_H; ++oy) {
        int sy0 = oy * rh / MRZ_OCR_GLYPH_H;
        int sy1 = (oy + 1) * rh / MRZ_OCR_GLYPH_H;
        if (sy1 <= sy0) sy1 = sy0 + 1;
        for (int ox = 0; ox < MRZ_OCR_GLYPH_W; ++ox) {
            int sx0 = ox * rw / MRZ_OCR_GLYPH_W;
            int sx1 = (ox + 1) * rw / MRZ_OCR_GLYPH_W;
            if (sx1 <= sx0) sx1 = sx0 + 1;
            int ink = 0, total = 0;
            for (int sy = sy0; sy < sy1; ++sy) {
                int yy = ry + sy;
                if (yy < 0 || yy >= H) continue;
                for (int sx = sx0; sx < sx1; ++sx) {
                    int xx = rx + sx;
                    if (xx < 0 || xx >= W) continue;
                    ink += bin[yy * W + xx];
                    total++;
                }
            }
            out[oy][ox] = (total > 0 && ink * 2 >= total) ? 1 : 0;
        }
    }
    return 0;
}

static int match_glyph(const uint8_t ch[MRZ_OCR_GLYPH_H][MRZ_OCR_GLYPH_W]) {
    int best = -1;
    double bestScore = -1;
    for (int g = 0; g < MRZ_OCR_GLYPH_COUNT; ++g) {
        int same = 0, diff = 0;
        for (int y = 0; y < MRZ_OCR_GLYPH_H; ++y) {
            uint8_t row = MRZ_OCR_GLYPHS[g].rows[y];
            for (int x = 0; x < MRZ_OCR_GLYPH_W; ++x) {
                int ink = (row >> (7 - x)) & 1;
                if (ch[y][x] == ink) same++; else diff++;
            }
        }
        double score = (double)same / (double)(same + diff);
        if (score > bestScore) { bestScore = score; best = g; }
    }
    return best;
}

/* ---------- Top-level pipeline. ---------- */
mrz_ocr_status_t mrz_ocr_recognise(const face_image_t *img,
                                  mrz_ocr_result_t *out) {
    if (!img || !out) return MRZ_OCR_ERR_LOAD;
    if (mrz_ocr_deadline_expired()) return MRZ_OCR_ERR_TIMEOUT;
    memset(out, 0, sizeof(*out));
    int W = img->width, H = img->height;
    if (W <= 0 || H <= 0) return MRZ_OCR_ERR_LOAD;

    /* 1. Grayscale. */
    size_t n = (size_t)W * H;
    uint8_t *gray = (uint8_t *)malloc(n);
    if (!gray) return MRZ_OCR_ERR_LOAD;
    face_to_grayscale(img, gray);
    if (mrz_ocr_deadline_expired()) {
        free(gray);
        return MRZ_OCR_ERR_TIMEOUT;
    }

    /* 2. Binarise (ink=black => 1). Polarity-normalise first: white
     * ink on black background (invert scans) must not silently mark
     * the background as ink. */
    int t = mrz_ocr_otsu(gray, (int)n);
    {
        size_t dark = 0;
        for (size_t i = 0; i < n; ++i) if (gray[i] < t) dark++;
        if (dark * 2 > n) {
            for (size_t i = 0; i < n; ++i) gray[i] = (uint8_t)(255 - gray[i]);
            t = 255 - t;
        }
    }
    uint8_t *bin = (uint8_t *)malloc(n);
    if (!bin) { free(gray); return MRZ_OCR_ERR_LOAD; }
    for (size_t i = 0; i < n; ++i)
        bin[i] = (gray[i] < t) ? 1 : 0;
    free(gray);

    /* 3. Locate MRZ band. */
    mrz_ocr_rect_t band;
    int lrc = mrz_ocr_locate_band(bin, W, H, &band);
    if (lrc != 0) {
        free(bin);
        return MRZ_OCR_ERR_NO_BAND;
    }
    if (mrz_ocr_deadline_expired()) {
        free(bin);
        return MRZ_OCR_ERR_TIMEOUT;
    }
    out->band_x = band.x; out->band_y = band.y;
    out->band_w = band.w; out->band_h = band.h;
    if (band.w < MRZ_OCR_GLYPH_W * 5 || band.h < MRZ_OCR_GLYPH_H) {
        free(bin);
        return MRZ_OCR_ERR_BAD_GEOMETRY;
    }

    /* 4. Extract the band and split into two lines.  The band is
     * deskewed FIRST via the shared geometry layer, so rotated scans
     * (realistic corpus: +-3 deg) reach the classifier instead of
     * dying in row projection.  `band_pixels` stays the raw crop for
     * no-skew fast path; `band_work` is the (possibly rotated) copy
     * that line splitting and per-line segmenting operate on. */
    uint8_t *band_pixels = (uint8_t *)malloc((size_t)band.w * band.h);
    if (!band_pixels) { free(bin); return MRZ_OCR_ERR_LOAD; }
    for (int y = 0; y < band.h; ++y)
        memcpy(band_pixels + y * band.w,
               bin + (band.y + y) * W + band.x, band.w);

    uint8_t *band_work = band_pixels;
    uint8_t *rot = NULL;
    double deg = mrz_estimate_band_skew(band_pixels, band.w, band.h);
    if (deg != 0.0) {
        rot = (uint8_t *)malloc((size_t)band.w * band.h);
        if (!rot) { free(bin); free(band_pixels); return MRZ_OCR_ERR_LOAD; }
        mrz_rotate_band(band_pixels, band.w, band.h, deg, 0, 0, rot);   /* legacy: NN keeps 0/1 */
        band_work = rot;
    }

    mrz_ocr_rect_t lines[2];
    if (mrz_ocr_split_lines(band_work, band.w, band.h, lines) != 0) {
        /* H/2 fallback: when all split strategies fail (guilloche-filled
         * gaps, extreme aspect ratios, extra content in band), split at
         * the midpoint.  Crude, but produces output instead of failing
         * silently -- the CNN may still recognise most characters. */
        int mid = band.h / 2;
        lines[0].x = 0; lines[0].y = 0;
        lines[0].w = band.w; lines[0].h = mid;
        lines[1].x = 0; lines[1].y = mid;
        lines[1].w = band.w; lines[1].h = band.h - mid;
    }

    /* 5. For each line, segment and recognise. */
    int total_chars = 0, total_conf = 0;
    for (int li = 0; li < 2; ++li) {
        if (mrz_ocr_deadline_expired()) {
            free(band_pixels);
            free(bin);
            free(rot);
            return MRZ_OCR_ERR_TIMEOUT;
        }
        mrz_ocr_rect_t line_rect = lines[li];
        uint8_t *line_pixels = (uint8_t *)malloc((size_t)line_rect.w * line_rect.h);
        if (!line_pixels) { free(band_pixels); free(bin); return MRZ_OCR_ERR_LOAD; }
        for (int y = 0; y < line_rect.h; ++y)
            memcpy(line_pixels + y * line_rect.w,
                   band_work + (line_rect.y + y) * band.w + line_rect.x,
                   line_rect.w);
        mrz_ocr_rect_t chars[64];
        int nchars = mrz_ocr_segment_line(line_pixels, line_rect.w, line_rect.h, chars, 64);
        if (nchars < 20) {
            free(line_pixels); free(band_pixels); free(bin);
            return MRZ_OCR_ERR_BAD_LINES;
        }
        char *dest = (li == 0) ? out->line1 : out->line2;
        int *dest_len = (li == 0) ? &out->line1_len : &out->line2_len;
        int *dest_conf = (li == 0) ? &out->line1_avg_conf : &out->line2_avg_conf;
        int conf_sum = 0;
        for (int c = 0; c < nchars && c < 44; ++c) {
            uint8_t ch[MRZ_OCR_GLYPH_H][MRZ_OCR_GLYPH_W];
            resample_char(line_pixels, line_rect.w, line_rect.h,
                          chars[c].x, chars[c].y, chars[c].w, chars[c].h, ch);
            int gi = match_glyph(ch);
            if (gi < 0) {
                dest[c] = '?';
                conf_sum += 0;
            } else {
                dest[c] = MRZ_OCR_GLYPHS[gi].ch;
                /* Compute confidence for this match. */
                int same = 0, total = MRZ_OCR_GLYPH_H * MRZ_OCR_GLYPH_W;
                for (int y = 0; y < MRZ_OCR_GLYPH_H; ++y) {
                    uint8_t row = MRZ_OCR_GLYPHS[gi].rows[y];
                    for (int x = 0; x < MRZ_OCR_GLYPH_W; ++x) {
                        int ink = (row >> (7 - x)) & 1;
                        if (ch[y][x] == ink) same++;
                    }
                }
                conf_sum += same * 100 / total;
            }
        }
        *dest_len = nchars < 44 ? nchars : 44;
        dest[*dest_len] = '\0';
        *dest_conf = (nchars > 0) ? (conf_sum / nchars) : 0;
        total_chars += nchars;
        total_conf += conf_sum;
        free(line_pixels);
    }
    free(rot);
    free(band_pixels);
    free(bin);
    (void)total_chars; (void)total_conf;
    return MRZ_OCR_OK;
}
