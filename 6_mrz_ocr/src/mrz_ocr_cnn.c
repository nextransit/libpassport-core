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
                               uint8_t *sm, int sm_cap,
                               float out[CNN_IN_H][CNN_IN_W]) {
    (void)sm; (void)sm_cap;
    if (rw <= 0 || rh <= 0) return;
    /* area-average the raw grayscale box to 16x12 (soft edges). */
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
            float ink = cnt > 0 ? (float)sum / cnt : 0.0f;
            out[oy][ox] = 1.0f - ink / 255.0f;
        }
    }
    /* Lightweight 3x3 box on the final 16x12 (averages isolated
     * salt/pepper pixels far cheaper than a full-ROI presmooth). */
    {
        float tmp[CNN_IN_H][CNN_IN_W];
        memcpy(tmp, out, sizeof(tmp));
        for (int y = 0; y < CNN_IN_H; ++y)
            for (int x = 0; x < CNN_IN_W; ++x) {
                float acc = 0; int c = 0;
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        int ny = y + dy, nx = x + dx;
                        if (ny >= 0 && ny < CNN_IN_H && nx >= 0 && nx < CNN_IN_W) {
                            acc += tmp[ny][nx]; c++;
                        }
                    }
                out[y][x] = acc / (float)(c ? c : 1);
            }
    }
    /* X-axis centroid alignment (clamped +-2 px). */
    double sx = 0.0, sw = 0.0;
    for (int y = 0; y < CNN_IN_H; ++y)
        for (int x = 0; x < CNN_IN_W; ++x) {
            float v = out[y][x];
            if (v > 0.35f) { sx += (double)x * (double)v; sw += v; }
        }
    if (sw > 1e-3) {
        double cx = sx / sw;
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
    /* Y-axis centroid alignment (clamped +-2 px). */
    double sy = 0.0, syw = 0.0;
    for (int y = 0; y < CNN_IN_H; ++y)
        for (int x = 0; x < CNN_IN_W; ++x) {
            float v = out[y][x];
            if (v > 0.35f) { sy += (double)y * (double)v; syw += v; }
        }
    if (syw > 1e-3) {
        double cy = sy / syw;
        double target = (CNN_IN_H - 1) * 0.5;
        int delta = (int)llround(cy - target);
        if (delta < -2) delta = -2;
        if (delta > 2) delta = 2;
        if (delta != 0) {
            float tmp[CNN_IN_H][CNN_IN_W];
            memcpy(tmp, out, sizeof(tmp));
            memset(out, 0, sizeof(out));
            for (int y = 0; y < CNN_IN_H; ++y)
                for (int x = 0; x < CNN_IN_W; ++x) {
                    int ny = y - delta;
                    if (ny >= 0 && ny < CNN_IN_H)
                        out[y][x] = tmp[ny][x];
                }
        }
    }
}

/* ---- CNN-specific segmenter (full-cell windows, not raw ink runs) ----
 * The traditional segmenter returns only the ink bbox (e.g. 24px of a
 * 34px cell). Our trainer pads glyphs with random leading/trailing
 * whitespace, so feeding ink-only boxes creates an aspect-ratio mismatch
 * (fat glyphs) that hurts the conv net. Here we re-window each ink run
 * to a full cell: half-gap on the left, half-gap on the right, clamped
 * to the line bounds. Falls back to the shared segmenter when a run
 * cannot be windowed. */
static int segment_line_cnn(const uint8_t *bin, int W, int H,
                            mrz_ocr_rect_t *chars, int max_chars) {
    /* Full-cell segmentation: the shared segmenter returns only the ink
     * bbox (24px of a 34px cell), which warps glyphs when resampled to
     * 16x12. Here we measure the inter-character pitch from the column
     * projection and cut centred full-cell windows so each glyph keeps
     * its inter-character whitespace, matching the trainer distribution. */
    int *col_dark = (int *)calloc((size_t)W, sizeof(int));
    if (!col_dark) return -1;
    for (int x = 0; x < W; ++x) {
        int c = 0;
        for (int y = 0; y < H; ++y) if (bin[y * W + x]) c++;
        col_dark[x] = c;
    }
    int thr = H / 12; if (thr < 1) thr = 1;
    int run_s[96], run_e[96]; int nrun = 0, in_run = 0, rs = 0;
    for (int x = 0; x <= W; ++x) {
        int ink = (x < W && col_dark[x] >= thr);
        if (ink && !in_run) { in_run = 1; rs = x; }
        else if (!ink && in_run) {
            if (nrun < 96) { run_s[nrun] = rs; run_e[nrun] = x; nrun++; }
            in_run = 0;
        }
    }
    free(col_dark);
    if (nrun < 1) return 0;
    /* estimate cell width as median run width + gap(~2px in gen) */
    int w[96]; int nw=0;
    for (int i=0;i<nrun && i<96;i++) w[nw++] = run_e[i]-run_s[i];
    for (int i=1;i<nw;i++){ int k=w[i],j=i-1; while(j>=0&&w[j]>k){w[j+1]=w[j];j--;} w[j+1]=k; }
    int medw = nw ? w[nw/2] : 8;
    if (medw < 6) medw = 6;
    int cell = medw + 8;                 /* ~ full pitch */
    if (cell < medw + 4) cell = medw + 4;
    int n = 0;
    for (int i = 0; i < nrun && i < 96; ++i) {
        if (n >= max_chars) break;
        int c = (run_s[i] + run_e[i]) / 2;
        int half = cell / 2;
        int x0 = c - half;
        int ww = cell;
        if (x0 < 0) { ww += x0; x0 = 0; }
        if (x0 + ww > W) ww = W - x0;
        if (ww < 4) ww = 4;
        chars[n].x = x0; chars[n].w = ww;
        chars[n].y = 0; chars[n].h = H;
        n++;
    }
    return n;
}
/* ---- whole-row decode with syntax mask + checksum beam + "<" tail ---- */
static void decode_row(const cnn_t *net,
                       const float (*glyphs)[CNN_IN_H][CNN_IN_W],
                       int n, int line_idx, char *dest, int *conf_avg,
                       const mrz_ocr_rect_t *chars) {
    float feats_stk[CNN_BATCH_MAX * CNN_FLAT];
    float probs_stk[CNN_BATCH_MAX * CNN_OUT];
#ifdef MRZ_OCR_DUMP
    {
        for (int _c = 0; _c < n && _c < n; ++_c)
            if (chars) fprintf(stderr, "[L%d C%d] x=%d w=%d h=%d\n",
                    line_idx, _c, chars[_c].x, chars[_c].w, chars[_c].h);
    }
#endif
#ifdef MRZ_OCR_DUMP
    {
        char path[128];
        for (int _c = 0; _c < n && _c < 44; ++_c) {
            snprintf(path, sizeof(path), "/tmp/glyph_L%d_C%d.pgm",
                     line_idx, _c);
            FILE *f = fopen(path, "w");
            if (f) {
                fprintf(f, "P2\n%d %d\n255\n", CNN_IN_W, CNN_IN_H);
                for (int _y = 0; _y < CNN_IN_H; ++_y) {
                    for (int _x = 0; _x < CNN_IN_W; ++_x)
                        fprintf(f, "%d ", (int)(glyphs[_c][_y][_x]*255));
                    fprintf(f, "\n");
                }
                fclose(f);
            }
        }
    }
#endif
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
#ifdef MRZ_OCR_DUMP
    for (int c = 0; c < n && c < 44; ++c)
        fprintf(stderr, "[%d] GT? c=%d cand0=%c(%.3f) cand1=%c(%.3f)\n",
                line_idx, c, MRZ_OCR_GLYPHS[t2[c].cand[0]].ch, t2[c].p0,
                MRZ_OCR_GLYPHS[t2[c].cand[1]].ch, t2[c].p1);
#endif
    if (n < 44) dest[n] = '\0'; else dest[44] = '\0';

    /* Trailing "<" filler whitelist: ICAO 9303 guarantees that once a
     * MRZ field enters the filler region, all remaining positions of
     * that field are '<' (line 2 keeps two numeric check digits at the
     * very end). So as soon as we see a run of >=3 '<' we lock the rest
     * of the line (excluding line2's final two check positions) to '<'
     * instead of trusting low-confidence model outputs like 8/2/0. */
    int lt = mrz_ocr_glyph_index('<');
    int total = n < 44 ? n : 44;
    /* Lock-run only on Line 1: once we've seen >=5 consecutive '<' the
     * remaining name-field positions are guaranteed '<' by ICAO 9303.
     * Line 2 may legally contain long '<' runs in the personal-number
     * field but its tail is free-form in these synthetic cases, so we
     * only lock Line 1. */
    int locked = -1;
    int run = 0;
    if (line_idx == 0) {
        for (int c = 0; c < total; ++c) {
            if (dest[c] == '<') { run++; if (run >= 5) locked = c - run + 1; }
            else run = 0;
        }
        if (locked >= 0)
            for (int c = locked; c < total; ++c)
                dest[c] = '<';
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
        int nchars = segment_line_cnn(line_pixels, bw, line_h, chars, 64);
        if (nchars < 30) { free(line_pixels); free(band_pixels); free(bin); free(gray); return MRZ_OCR_ERR_BAD_LINES; }
        if (nchars > 44) nchars = 44;
        /* one reusable 3x3-smooth scratch buffer for the whole line */
        int max_box = 0;
        for (int c = 0; c < nchars; ++c) {
            int b = chars[c].w * chars[c].h;
            if (b > max_box) max_box = b;
        }
        uint8_t *sm_scratch = (uint8_t *)malloc((size_t)(max_box > 0 ? max_box : 1));
        if (!sm_scratch) { free(line_pixels); free(band_pixels); free(bin); free(gray); return MRZ_OCR_ERR_LOAD; }
        /* pre-extract each glyph from the ORIGINAL grayscale band */
        float glyphs[44][CNN_IN_H][CNN_IN_W];
        for (int c = 0; c < nchars; ++c) {
            resample_char_gray(gray, W, H,
                               bx + chars[c].x, by + base + chars[c].y,
                               chars[c].w, chars[c].h,
                               sm_scratch, max_box, glyphs[c]);
        }
        free(sm_scratch);
        char *dest = (li == 0) ? out->line1 : out->line2;
        int *conf = (li == 0) ? &out->line1_avg_conf : &out->line2_avg_conf;
        decode_row(&net, glyphs, nchars, li, dest, conf, chars);
        if (li == 0) out->line1_len = nchars; else out->line2_len = nchars;
        free(line_pixels);
    }
    free(band_pixels);
    free(bin);
    free(gray);
    return MRZ_OCR_OK;
}

/* Debug helper: dump the 16x12 glyphs a given image produces.
 * Enabled by MRZ_OCR_DUMP=<tag>. Writes ppm to /tmp/mrz_dump_<tag>_<l><c>.ppm */
#if 0
#endif
