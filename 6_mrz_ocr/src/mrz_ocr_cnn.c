/* CNN ConvNet-based MRZ OCR pipeline.
 *
 * Pipeline is identical to mrz_ocr.c up to band/lines/segment, then:
 *   1. resample each char to 12x16 (16 wide to preserve aspect ratio)
 *   2. centroid alignment (drop the off-centre jitter)
 *   3. whole-row batch inference (all 44 chars in one fc pass)
 *   4. ICAO 9303 syntax mask on the logits
 *   5. Line-2 checksum beam-search (Top-2 backtracking)
 *
 * The actual conv net lives in cnn.c; this file only glues the row
 * decoder + business-layer constraints on top of it.
 */
#include "mrz_ocr.h"
#include "cnn.h"
#include "template.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- ICAO 9303 mod-7/3/1 weighted check digit ---- */
static int check_digit_of(const char *seg, int n) {
    int sum = 0;
    static const int wtab[3] = {7, 3, 1};
    for (int i = 0; i < n; ++i) {
        int c = (unsigned char)seg[i];
        int v;
        if (c >= '0' && c <= '9')      v = c - '0';
        else if (c >= 'A' && c <= 'Z') v = c - 'A' + 10;
        else                           v = 0;   /* '<' => 0 */
        sum += v * wtab[i % 3];
    }
    return sum % 10;
}

/* ---- top-2 candidates for a column ---- */
typedef struct { int cand[2]; float p0, p1; } top2_t;

static void top2_of(const float *probs, int allow[37], int n_mask,
                    int *c0, int *c1, float *p0, float *p1) {
    /* allow[] == 0 means forbidden at this position. */
    int b0 = -1, b1 = -1;
    float v0 = -1.0f, v1 = -1.0f;
    for (int o = 0; o < 37; ++o) {
        if (allow[o] == 0) continue;
        float p = probs[o];
        if (p > v0) { v1 = v0; b1 = b0; v0 = p; b0 = o; }
        else if (p > v1) { v1 = p; b1 = o; }
    }
    if (b0 < 0) b0 = mrz_ocr_glyph_index('<');
    if (b1 < 0) b1 = b0;
    *c0 = b0; *c1 = b1; *p0 = v0; *p1 = v1;
}

/* ---- build the per-position character allow-set (syntax mask) ---- */
static void fill_allow(int allow[37], int line_idx, int col) {
    for (int i = 0; i < 37; ++i) allow[i] = 1;
    int d0 = mrz_ocr_glyph_index('0');   /* 0 */
    int a0 = mrz_ocr_glyph_index('A');   /* 10 */
    int lt = mrz_ocr_glyph_index('<');   /* 36 */
    if (line_idx == 0) {
        if (col == 1) { /* fixed '<' separator */
            for (int i = 0; i < 37; ++i) allow[i] = (i == lt);
        } else if (col >= 2 && col <= 4) { /* country code: A-Z only */
            for (int i = 0; i < 37; ++i)
                allow[i] = (i >= a0 && i <= lt - 1) ? 1 : 0;
        } else if (col >= 5) { /* name / filler: A-Z or '<' */
            for (int i = 0; i < 37; ++i)
                allow[i] = ((i >= a0 && i <= lt - 1) || i == lt) ? 1 : 0;
        }
        /* col 0: document type, keep all 37. */
    } else {
        if (col == 9 || col == 19 || col == 27 ||
            col == 42 || col == 43) {          /* check digits */
            for (int i = 0; i < 37; ++i)
                allow[i] = (i >= d0 && i <= d0 + 9) ? 1 : 0;
        } else if (col >= 13 && col <= 18) {   /* birth date */
            for (int i = 0; i < 37; ++i)
                allow[i] = (i >= d0 && i <= d0 + 9) ? 1 : 0;
        } else if (col >= 21 && col <= 26) {   /* expiry date */
            for (int i = 0; i < 37; ++i)
                allow[i] = (i >= d0 && i <= d0 + 9) ? 1 : 0;
        } else if (col >= 10 && col <= 12) {   /* nationality */
            for (int i = 0; i < 37; ++i)
                allow[i] = (i >= a0 && i <= lt - 1) ? 1 : 0;
        } else if (col == 20) {                /* sex M/F/< */
            for (int i = 0; i < 37; ++i)
                allow[i] = 0;
            allow[mrz_ocr_glyph_index('M')] = 1;
            allow[mrz_ocr_glyph_index('F')] = 1;
            allow[lt] = 1;
        }
        /* others (passport no 0-8, personal no 28-41): all 37 */
    }
}

/* ---- checksum validation over a decoded Line 2 ---- */
static int line2_checksum_ok(const char *l2) {
    char buf[15];
    /* passport no 0..8 + ck @9 */
    memcpy(buf, l2, 9); buf[9] = 0;
    if (check_digit_of(buf, 9) != (l2[9] - '0')) return 0;
    /* birth 13..18 + ck @19 */
    memcpy(buf, l2 + 13, 6); buf[6] = 0;
    if (check_digit_of(buf, 6) != (l2[19] - '0')) return 0;
    /* expiry 21..26 + ck @27 */
    memcpy(buf, l2 + 21, 6); buf[6] = 0;
    if (check_digit_of(buf, 6) != (l2[27] - '0')) return 0;
    /* personal 28..41 + ck @42 */
    memcpy(buf, l2 + 28, 14); buf[14] = 0;
    if (check_digit_of(buf, 14) != (l2[42] - '0')) return 0;
    /* composite: 43 chars of line 2 (0..42) + ck @43 */
    char comp[45];
    memcpy(comp, l2, 43); comp[43] = 0;
    if (check_digit_of(comp, 43) != (l2[43] - '0')) return 0;
    return 1;
}

/* ---- resample a segmented char box to 12x16 binary glyph ---- */
static int resample_char_cnn(const uint8_t *bin, int W, int H,
                             int rx, int ry, int rw, int rh,
                             uint8_t out[CNN_IN_H][CNN_IN_W]) {
    if (rw <= 0 || rh <= 0) return -1;
    /* Step 1: normalise the segmented box to the canonical 8x12 glyph
     * aspect ratio (as the template bank), using the same 1/2 ink rule
     * as before. This keeps the glyph shape faithful to OCR-B even
     * when the box includes per-char pitch whitespace. */
    uint8_t g8[12][8];
    for (int oy = 0; oy < 12; ++oy) {
        int sy0 = oy * rh / 12;
        int sy1 = (oy + 1) * rh / 12;
        if (sy1 <= sy0) sy1 = sy0 + 1;
        for (int ox = 0; ox < 8; ++ox) {
            int sx0 = ox * rw / 8;
            int sx1 = (ox + 1) * rw / 8;
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
            g8[oy][ox] = (total > 0 && ink * 2 >= total) ? 1 : 0;
        }
    }
    /* Step 2: upscale 8x12 -> 16x12 with 2x nearest-neighbour, giving
     * the conv net a 2x-wide glyph while preserving aspect ratio. This
     * must match how the trainer builds its clean glyphs (to_16). */
    for (int oy = 0; oy < CNN_IN_H; ++oy)
        for (int ox = 0; ox < CNN_IN_W; ++ox)
            out[oy][ox] = g8[oy][ox / 2];
    return 0;
}

/* ---- whole-row decode with syntax mask + checksum beam search ---- */
static void decode_row(const cnn_t *net,
                       const uint8_t (*glyphs)[CNN_IN_H][CNN_IN_W],
                       int n, int line_idx, char *dest, int *conf_avg) {
    /* batch feature + fc in one pass (stack workspace, no alloc) */
    float feats_stk[CNN_BATCH_MAX * CNN_FLAT];
    float probs_stk[CNN_BATCH_MAX * CNN_OUT];
    float *feats = feats_stk;
    float *probs = probs_stk;
    cnn_row_features_batch(net, glyphs, n, feats);
    cnn_fc_batch(net, feats, n, probs);

    /* top-2 per column after allow-mask */
    top2_t t2[44];
    int allow[37];
    for (int c = 0; c < n && c < 44; ++c) {
        fill_allow(allow, line_idx, c);
        top2_of(probs + c * CNN_OUT, allow, 0,
                &t2[c].cand[0], &t2[c].cand[1],
                &t2[c].p0, &t2[c].p1);
    }

    /* default decode = top-1 */
    for (int c = 0; c < n && c < 44; ++c)
        dest[c] = MRZ_OCR_GLYPHS[t2[c].cand[0]].ch;
    if (n < 44) dest[n] = '\0'; else dest[44] = '\0';

    /* Line-2 checksum beam search: flip at most 2 low-confidence
     * positions to their 2nd candidate until checksums validate. */
    if (line_idx == 1 && n == 44) {
        char best[45];
        memcpy(best, dest, 44); best[44] = 0;
        int ok = line2_checksum_ok(best);
        if (!ok) {
            /* candidate flips: pick the 3 lowest p0 positions among
             * numeric fields, try each 1- and 2-flip combo. */
            int idxs[44], picks = 0;
            for (int c = 0; c < 44; ++c) {
                if (t2[c].cand[0] != t2[c].cand[1]) {
                    /* only flip positions that are part of checksummed
                     * segments (numeric fields) to keep search tight */
                    if (c == 9 || c == 19 || c == 27 || c == 42 || c == 43 ||
                        (c >= 13 && c <= 18) || (c >= 21 && c <= 26))
                        idxs[picks++] = c;
                }
            }
            if (picks > 8) picks = 8;
            /* try single flips */
            for (int a = 0; a < picks && !ok; ++a) {
                char t[45];
                memcpy(t, dest, 44); t[44] = 0;
                t[idxs[a]] = MRZ_OCR_GLYPHS[t2[idxs[a]].cand[1]].ch;
                if (line2_checksum_ok(t)) { memcpy(best, t, 44); ok = 1; break; }
            }
            /* try double flips */
            for (int a = 0; a < picks && !ok; ++a)
                for (int b = a + 1; b < picks && !ok; ++b) {
                    char t[45];
                    memcpy(t, dest, 44); t[44] = 0;
                    t[idxs[a]] = MRZ_OCR_GLYPHS[t2[idxs[a]].cand[1]].ch;
                    t[idxs[b]] = MRZ_OCR_GLYPHS[t2[idxs[b]].cand[1]].ch;
                    if (line2_checksum_ok(t)) { memcpy(best, t, 44); ok = 1; break; }
                }
            memcpy(dest, best, 44);
            if (n < 44) dest[n] = '\0'; else dest[44] = '\0';
        }
    }

    /* average confidence of top-1 (before flipping) */
    int sum = 0;
    for (int c = 0; c < n && c < 44; ++c)
        sum += (int)((t2[c].p0 > 0 ? t2[c].p0 : 0.0f) * 100);
    *conf_avg = n > 0 ? sum / n : 0;
}

mrz_ocr_status_t mrz_ocr_recognise_cnn(const face_image_t *img,
                                       mrz_ocr_result_t *out) {
    if (!img || !out) return MRZ_OCR_ERR_LOAD;
    memset(out, 0, sizeof(*out));
    int W = img->width, H = img->height;
    if (W <= 0 || H <= 0) return MRZ_OCR_ERR_LOAD;

    size_t n = (size_t)W * H;
    uint8_t *gray = (uint8_t *)malloc(n);
    if (!gray) return MRZ_OCR_ERR_LOAD;
    face_to_grayscale(img, gray);
    /* Otsu threshold */
    int t = 127;
    {
        size_t hist[256] = {0};
        for (size_t i = 0; i < n; ++i) hist[gray[i]]++;
        double sum = 0;
        for (int i = 0; i < 256; ++i) sum += (double)i * hist[i];
        double sumB = 0;
        int wB = 0;
        double maxVar = -1;
        for (int i = 0; i < 256; ++i) {
            wB += (int)hist[i];
            if (wB == 0) continue;
            int wF = (int)n - wB;
            if (wF == 0) break;
            sumB += (double)i * hist[i];
            double mB = sumB / wB, mF = (sum - sumB) / wF;
            double v = (double)wB * wF * (mB - mF) * (mB - mF);
            if (v > maxVar) { maxVar = v; t = i; }
        }
        if (hist[0] > 0 && hist[255] == 0) t = 127;
        if (t == 0 && hist[255] > 0) t = 127;
    }
    uint8_t *bin = (uint8_t *)malloc(n);
    if (!bin) { free(gray); return MRZ_OCR_ERR_LOAD; }
    for (size_t i = 0; i < n; ++i) bin[i] = (gray[i] < t) ? 1 : 0;
    free(gray);

    mrz_ocr_rect_t band;
    if (mrz_ocr_locate_band(bin, W, H, &band) != 0) { free(bin); return MRZ_OCR_ERR_NO_BAND; }
    int bx = band.x, by = band.y, bw = band.w, bh = band.h;
    out->band_x = bx; out->band_y = by; out->band_w = bw; out->band_h = bh;
    if (bw < MRZ_OCR_GLYPH_W * 5 || bh < MRZ_OCR_GLYPH_H) { free(bin); return MRZ_OCR_ERR_BAD_GEOMETRY; }
    uint8_t *band_pixels = (uint8_t *)malloc((size_t)bw * bh);
    if (!band_pixels) { free(bin); return MRZ_OCR_ERR_LOAD; }
    for (int y = 0; y < bh; ++y)
        memcpy(band_pixels + y * bw, bin + (by + y) * W + bx, bw);

    mrz_ocr_rect_t lines[2];
    if (mrz_ocr_split_lines(band_pixels, bw, bh, lines) != 0) {
        free(band_pixels); free(bin); return MRZ_OCR_ERR_BAD_LINES;
    }
    cnn_t net;
    cnn_default_init(&net);

    for (int li = 0; li < 2; ++li) {
        int base = (li == 0) ? lines[0].y : lines[1].y;
        int line_h = (li == 0) ? lines[0].h : lines[1].h;
        if (line_h <= 0) continue;
        uint8_t *line_pixels = (uint8_t *)malloc((size_t)bw * line_h);
        if (!line_pixels) { free(band_pixels); free(bin); return MRZ_OCR_ERR_LOAD; }
        for (int y = 0; y < line_h; ++y)
            memcpy(line_pixels + y * bw, band_pixels + (base + y) * bw, bw);
        mrz_ocr_rect_t chars[64];
        int nchars = mrz_ocr_segment_line(line_pixels, bw, line_h, chars, 64);
        if (nchars < 30) { free(line_pixels); free(band_pixels); free(bin); return MRZ_OCR_ERR_BAD_LINES; }
        if (nchars > 44) nchars = 44;
        uint8_t glyphs[44][CNN_IN_H][CNN_IN_W];
        for (int c = 0; c < nchars; ++c)
            resample_char_cnn(line_pixels, bw, line_h,
                              chars[c].x, chars[c].y,
                              chars[c].w, chars[c].h, glyphs[c]);
        char *dest = (li == 0) ? out->line1 : out->line2;
        int *conf = (li == 0) ? &out->line1_avg_conf : &out->line2_avg_conf;
        decode_row(&net, glyphs, nchars, li, dest, conf);
        if (li == 0) out->line1_len = nchars; else out->line2_len = nchars;
        free(line_pixels);
    }
    free(band_pixels);
    free(bin);
    return MRZ_OCR_OK;
}
