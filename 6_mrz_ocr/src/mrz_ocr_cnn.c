/* CNN ConvNet-based MRZ OCR pipeline (grayscale backend).
 *
 * Pipeline:
 *   1. locate band / split lines / segment chars on the *binary* map
 *   2. for each char, BILINEAR-resample the ORIGINAL GRAYSCALE box to
 *      16x12 float (NO hard 1/2 threshold), keep sub-pixel stroke edges
 *   3. X-axis centroid alignment (kill the segmenter offset drift)
 *   4. whole-row batch conv+fc inference
 *   5. ICAO 9303 syntax mask on the logits
 *   6. Line-2 checksum beam-search (Top-2 backtracking)
 *   7. trailing "<" filler whitelist (suppress false positives at the
 *      end of the line where the grid naturally drifts)
 */
#include "mrz_ocr.h"
#include "cnn.h"
#include "template.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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

typedef struct { int cand[2]; float p0, p1; } top2_t;

static void top2_of(const float *probs, int allow[37],
                    int *c0, int *c1, float *p0, float *p1) {
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

/* ---- per-position character whitelist (syntax mask) ---- */
static void fill_allow(int allow[37], int line_idx, int col) {
    for (int i = 0; i < 37; ++i) allow[i] = 1;
    int d0 = mrz_ocr_glyph_index('0');
    int a0 = mrz_ocr_glyph_index('A');
    int lt = mrz_ocr_glyph_index('<');
    if (line_idx == 0) {
        if (col == 1) {
            for (int i = 0; i < 37; ++i) allow[i] = (i == lt);
        } else if (col >= 2 && col <= 4) {
            for (int i = 0; i < 37; ++i)
                allow[i] = (i >= a0 && i <= lt - 1) ? 1 : 0;
        } else if (col >= 5) {
            for (int i = 0; i < 37; ++i)
                allow[i] = ((i >= a0 && i <= lt - 1) || i == lt) ? 1 : 0;
        }
    } else {
        if (col == 9 || col == 19 || col == 27 ||
            col == 42 || col == 43) {
            for (int i = 0; i < 37; ++i)
                allow[i] = (i >= d0 && i <= d0 + 9) ? 1 : 0;
        } else if (col >= 13 && col <= 18) {
            for (int i = 0; i < 37; ++i)
                allow[i] = (i >= d0 && i <= d0 + 9) ? 1 : 0;
        } else if (col >= 21 && col <= 26) {
            for (int i = 0; i < 37; ++i)
                allow[i] = (i >= d0 && i <= d0 + 9) ? 1 : 0;
        } else if (col >= 10 && col <= 12) {
            for (int i = 0; i < 37; ++i)
                allow[i] = (i >= a0 && i <= lt - 1) ? 1 : 0;
        } else if (col == 20) {
            for (int i = 0; i < 37; ++i) allow[i] = 0;
            allow[mrz_ocr_glyph_index('M')] = 1;
            allow[mrz_ocr_glyph_index('F')] = 1;
            allow[lt] = 1;
        }
    }
}

static int line2_checksum_ok(const char *l2) {
    char buf[15];
    memcpy(buf, l2, 9); buf[9] = 0;
    if (check_digit_of(buf, 9) != (l2[9] - '0')) return 0;
    memcpy(buf, l2 + 13, 6); buf[6] = 0;
    if (check_digit_of(buf, 6) != (l2[19] - '0')) return 0;
    memcpy(buf, l2 + 21, 6); buf[6] = 0;
    if (check_digit_of(buf, 6) != (l2[27] - '0')) return 0;
    memcpy(buf, l2 + 28, 14); buf[14] = 0;
    if (check_digit_of(buf, 14) != (l2[42] - '0')) return 0;
    char comp[45];
    memcpy(comp, l2, 43); comp[43] = 0;
    if (check_digit_of(comp, 43) != (l2[43] - '0')) return 0;
    return 1;
}

/* ---- grayscale area-resample of a char box to 16x12 float ---- */
static void resample_char_gray(const uint8_t *gray, int W, int H,
                               int rx, int ry, int rw, int rh,
                               float out[CNN_IN_H][CNN_IN_W]) {
    if (rw <= 0 || rh <= 0) return;
    for (int oy = 0; oy < CNN_IN_H; ++oy) {
        int sy0 = (oy * rh) / CNN_IN_H;
        int sy1 = ((oy + 1) * rh) / CNN_IN_H;
        if (sy1 <= sy0) sy1 = sy0 + 1;
        for (int ox = 0; ox < CNN_IN_W; ++ox) {
            int sx0 = (ox * rw) / CNN_IN_W;
            int sx1 = ((ox + 1) * rw) / CNN_IN_W;
            if (sx1 <= sx0) sx1 = sx0 + 1;
            int sum = 0, cnt = 0;
            for (int sy = sy0; sy < sy1; ++sy) {
                int yy = ry + sy;
                if (yy < 0 || yy >= H) continue;
                for (int sx = sx0; sx < sx1; ++sx) {
                    int xx = rx + sx;
                    if (xx < 0 || xx >= W) continue;
                    sum += gray[yy * W + xx];
                    cnt++;
                }
            }
            /* ink = dark => value near 255 on the rendered band.
             * map to [0,1] ink coverage with soft edges. */
            float ink = cnt > 0 ? (float)sum / cnt : 0.0f;
            /* band canvas is white(255)/ink(0); invert so that ink is
             * HIGH (matches trainer: 1 = ink). */
            out[oy][ox] = 1.0f - ink / 255.0f;
        }
    }
    /* X-axis centroid alignment: shift ink mass to the glyph centre so
     * a +-1px segmenter drift does not move the glyph inside the cell.
     * Only use strong-ink pixels to compute the mass centre (clamped to
     * +-2 px so continuous "<" regions do not skew the window). */
    double sx = 0.0, sw = 0.0;
    for (int y = 0; y < CNN_IN_H; ++y)
        for (int x = 0; x < CNN_IN_W; ++x) {
            float v = out[y][x];
            if (v > 0.35f) { sx += (double)x * (double)v; sw += v; }
        }
    if (sw > 1e-3) {
        double cx = sx / sw;               /* centroid x */
        double target = (CNN_IN_W - 1) * 0.5;
        int delta = (int)llround(cx - target);
        if (delta < -2) delta = -2;
        if (delta > 2) delta = 2;
        if (delta != 0) {
            float tmp[CNN_IN_H][CNN_IN_W];
            memcpy(tmp, out, sizeof(tmp));
            memset(out, 0, sizeof(out));
            for (int y = 0; y < CNN_IN_H; ++y)
                for (int x = 0; x < CNN_IN_W; ++x) {
                    int nx = x - delta;
                    if (nx >= 0 && nx < CNN_IN_W)
                        out[y][x] = tmp[y][nx];
                }
        }
    }
}

/* ---- whole-row decode with syntax mask + checksum beam + "<" tail ---- */
static void decode_row(const cnn_t *net,
                       const float (*glyphs)[CNN_IN_H][CNN_IN_W],
                       int n, int line_idx, char *dest, int *conf_avg) {
    float feats_stk[CNN_BATCH_MAX * CNN_FLAT];
    float probs_stk[CNN_BATCH_MAX * CNN_OUT];
    cnn_row_features_batch(net, glyphs, n, feats_stk);
    cnn_fc_batch(net, feats_stk, n, probs_stk);

    top2_t t2[44];
    int allow[37];
    for (int c = 0; c < n && c < 44; ++c) {
        fill_allow(allow, line_idx, c);
        top2_of(probs_stk + c * CNN_OUT, allow,
                &t2[c].cand[0], &t2[c].cand[1],
                &t2[c].p0, &t2[c].p1);
    }

    for (int c = 0; c < n && c < 44; ++c)
        dest[c] = MRZ_OCR_GLYPHS[t2[c].cand[0]].ch;
    if (n < 44) dest[n] = '\0'; else dest[44] = '\0';

    /* Trailing "<" filler whitelist: once we've seen consecutive "<",
     * low-confidence mis-reads at the very tail (grid drift zone) are
     * forced back to "<". */
    int lt = mrz_ocr_glyph_index('<');
    int run = 0;
    int total = n < 44 ? n : 44;
    for (int c = 0; c < total; ++c) {
        if (dest[c] == '<') { run++; continue; }
        if (run >= 2 && c >= total - 12) {
            /* in the tail filler region */
            if (t2[c].p0 < 0.70f && t2[c].cand[1] == lt) {
                dest[c] = '<';
                run++;
                continue;
            }
        }
        run = 0;
    }

    /* Line-2 checksum beam search: flip at most 2 low-confidence
     * positions to their 2nd candidate until checksums validate. */
    if (line_idx == 1 && total == 44) {
        char best[45];
        memcpy(best, dest, 44); best[44] = 0;
        int ok = line2_checksum_ok(best);
        if (!ok) {
            int idxs[44], picks = 0;
            for (int c = 0; c < 44; ++c) {
                if (t2[c].cand[0] != t2[c].cand[1]) {
                    if (c == 9 || c == 19 || c == 27 || c == 42 || c == 43 ||
                        (c >= 13 && c <= 18) || (c >= 21 && c <= 26))
                        idxs[picks++] = c;
                }
            }
            if (picks > 8) picks = 8;
            for (int a = 0; a < picks && !ok; ++a) {
                char t[45];
                memcpy(t, dest, 44); t[44] = 0;
                t[idxs[a]] = MRZ_OCR_GLYPHS[t2[idxs[a]].cand[1]].ch;
                if (line2_checksum_ok(t)) { memcpy(best, t, 44); ok = 1; break; }
            }
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

    int sum = 0;
    for (int c = 0; c < total; ++c)
        sum += (int)((t2[c].p0 > 0 ? t2[c].p0 : 0.0f) * 100);
    *conf_avg = total > 0 ? sum / total : 0;
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
    /* Otsu threshold for band/lines/segment only. */
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

    mrz_ocr_rect_t band;
    if (mrz_ocr_locate_band(bin, W, H, &band) != 0) {
        free(bin); free(gray); return MRZ_OCR_ERR_NO_BAND;
    }
    int bx = band.x, by = band.y, bw = band.w, bh = band.h;
    out->band_x = bx; out->band_y = by; out->band_w = bw; out->band_h = bh;
    if (bw < MRZ_OCR_GLYPH_W * 5 || bh < MRZ_OCR_GLYPH_H) {
        free(bin); free(gray); return MRZ_OCR_ERR_BAD_GEOMETRY;
    }
    uint8_t *band_pixels = (uint8_t *)malloc((size_t)bw * bh);
    if (!band_pixels) { free(bin); free(gray); return MRZ_OCR_ERR_LOAD; }
    for (int y = 0; y < bh; ++y)
        memcpy(band_pixels + y * bw, bin + (by + y) * W + bx, bw);

    mrz_ocr_rect_t lines[2];
    if (mrz_ocr_split_lines(band_pixels, bw, bh, lines) != 0) {
        free(band_pixels); free(bin); free(gray); return MRZ_OCR_ERR_BAD_LINES;
    }
    cnn_t net;
    cnn_default_init(&net);

    for (int li = 0; li < 2; ++li) {
        int base = (li == 0) ? lines[0].y : lines[1].y;
        int line_h = (li == 0) ? lines[0].h : lines[1].h;
        if (line_h <= 0) continue;
        uint8_t *line_pixels = (uint8_t *)malloc((size_t)bw * line_h);
        if (!line_pixels) { free(band_pixels); free(bin); free(gray); return MRZ_OCR_ERR_LOAD; }
        for (int y = 0; y < line_h; ++y)
            memcpy(line_pixels + y * bw, band_pixels + (base + y) * bw, bw);
        mrz_ocr_rect_t chars[64];
        int nchars = mrz_ocr_segment_line(line_pixels, bw, line_h, chars, 64);
        if (nchars < 30) { free(line_pixels); free(band_pixels); free(bin); free(gray); return MRZ_OCR_ERR_BAD_LINES; }
        if (nchars > 44) nchars = 44;
        /* pre-extract each glyph from the ORIGINAL grayscale band */
        float glyphs[44][CNN_IN_H][CNN_IN_W];
        for (int c = 0; c < nchars; ++c) {
            resample_char_gray(gray, W, H,
                               bx + chars[c].x, by + base + chars[c].y,
                               chars[c].w, chars[c].h, glyphs[c]);
        }
        char *dest = (li == 0) ? out->line1 : out->line2;
        int *conf = (li == 0) ? &out->line1_avg_conf : &out->line2_avg_conf;
        decode_row(&net, glyphs, nchars, li, dest, conf);
        if (li == 0) out->line1_len = nchars; else out->line2_len = nchars;
        free(line_pixels);
    }
    free(band_pixels);
    free(bin);
    free(gray);
    return MRZ_OCR_OK;
}
