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

#ifndef ABS
#define ABS(x) ((x) < 0 ? -(x) : (x))
#endif

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
    }
    return "unknown";
}

/* ---------- 1. Binarisation (Otsu threshold). ---------- */
static int otsu_threshold(const uint8_t *gray, int n) {
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
     * up to the midpoint if the first bin is non-empty and dominant
     * (== the pure background case). */
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
    int threshold = (int)(W * 0.15);
    for (int y = 0; y < H; ++y) {
        int c = 0;
        for (int x = 0; x < W; ++x) if (bin[y * W + x]) c++;
        row_dark[y] = c;
    }
    /* Find ALL dense runs; we then pick a band that is the union of two
     * dense runs separated by a relatively small gap (the inter-line
     * gap of a TD3 record). The simplest robust rule: scan for runs,
     * collect runs of height >= H/12, and merge consecutive runs whose
     * vertical gap is <= H/8 (the inter-line gap). */
    int *run_top = (int *)malloc(sizeof(int) * H);
    int *run_bot = (int *)malloc(sizeof(int) * H);
    int n_runs = 0;
    int top = -1;
    int min_run_h = H / 12;
    int max_gap = H / 8;
    for (int y = 0; y < H; ++y) {
        if (row_dark[y] >= threshold) {
            if (top < 0) top = y;
        } else {
            if (top >= 0 && y - top >= min_run_h) {
                run_top[n_runs] = top;
                run_bot[n_runs] = y;
                n_runs++;
            }
            top = -1;
        }
    }
    if (top >= 0 && H - top >= min_run_h) {
        run_top[n_runs] = top; run_bot[n_runs] = H; n_runs++;
    }
    for (int i = 0; i < n_runs; ++i)
    if (n_runs == 0) { free(row_dark); free(run_top); free(run_bot); return -1; }
    /* Find the pair (i, i+1) whose union is the tallest. */
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
    out->x = left; out->y = best_top;
    out->w = right - left + 1; out->h = best_bot - best_top;
    return 0;
}

/* ---------- 3. Split the band into the two TD3 lines. ---------- */
int mrz_ocr_split_lines(const uint8_t *bin, int W, int H, mrz_ocr_rect_t *out) {
    /* Project darkness on each row of the band to find the gap. */
    int *row_dark = (int *)calloc((size_t)H, sizeof(int));
    if (!row_dark) return -1;
    for (int y = 0; y < H; ++y) {
        int c = 0;
        for (int x = 0; x < W; ++x) if (bin[y * W + x]) c++;
        row_dark[y] = c;
    }
    /* Find the deepest gap. A row is "gap" if its dark count is
     * below max(W/100, 4). The gap must span at least 4% of H and
     * both halves must span at least 20% of H. */
    int gap_thr = W / 100;
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
    free(row_dark);
    if (best_gap_y < 0) return -1;
    /* The two text lines are [0, best_gap_y) and [best_gap_y + best_gap_score, H). */
    out[0].x = 0; out[0].y = 0;
    out[0].w = W; out[0].h = best_gap_y;
    out[1].x = 0; out[1].y = best_gap_y + best_gap_score;
    out[1].w = W; out[1].h = H - out[1].y;
    return 0;
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
    memset(out, 0, sizeof(*out));
    int W = img->width, H = img->height;
    if (W <= 0 || H <= 0) return MRZ_OCR_ERR_LOAD;

    /* 1. Grayscale. */
    size_t n = (size_t)W * H;
    uint8_t *gray = (uint8_t *)malloc(n);
    if (!gray) return MRZ_OCR_ERR_LOAD;
    face_to_grayscale(img, gray);

    /* 2. Binarise (ink=black => 1). */
    int t = otsu_threshold(gray, (int)n);
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
    out->band_x = band.x; out->band_y = band.y;
    out->band_w = band.w; out->band_h = band.h;
    if (band.w < MRZ_OCR_GLYPH_W * 5 || band.h < MRZ_OCR_GLYPH_H) {
        free(bin);
        return MRZ_OCR_ERR_BAD_GEOMETRY;
    }

    /* 4. Extract the band and split into two lines. */
    uint8_t *band_pixels = (uint8_t *)malloc((size_t)band.w * band.h);
    if (!band_pixels) { free(bin); return MRZ_OCR_ERR_LOAD; }
    for (int y = 0; y < band.h; ++y)
        memcpy(band_pixels + y * band.w,
               bin + (band.y + y) * W + band.x, band.w);

    mrz_ocr_rect_t lines[2];
    if (mrz_ocr_split_lines(band_pixels, band.w, band.h, lines) != 0) {
        free(band_pixels); free(bin);
        return MRZ_OCR_ERR_BAD_LINES;
    }

    /* 5. For each line, segment and recognise. */
    int total_chars = 0, total_conf = 0;
    for (int li = 0; li < 2; ++li) {
        mrz_ocr_rect_t line_rect = lines[li];
        uint8_t *line_pixels = (uint8_t *)malloc((size_t)line_rect.w * line_rect.h);
        if (!line_pixels) { free(band_pixels); free(bin); return MRZ_OCR_ERR_LOAD; }
        for (int y = 0; y < line_rect.h; ++y)
            memcpy(line_pixels + y * line_rect.w,
                   band_pixels + (line_rect.y + y) * band.w + line_rect.x,
                   line_rect.w);
        mrz_ocr_rect_t chars[64];
        int nchars = mrz_ocr_segment_line(line_pixels, line_rect.w, line_rect.h, chars, 64);
        if (nchars < 30) {
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
    free(band_pixels);
    free(bin);
    (void)total_chars; (void)total_conf;
    return MRZ_OCR_OK;
}
