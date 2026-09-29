/* CNN ConvNet-based MRZ OCR pipeline (grayscale backend).
 *
 * Pipeline:
 *   1. global Otsu on the grayscale page -> coarse bin map
 *   2. locate the MRZ band by row density (shared with backend A)
 *   3. extract the band's GRAYSCALE crop; re-Otsu LOCALLY inside the
 *      band (two-pass threshold: robust to page-level illumination
 *      gradients a global threshold cannot handle)
 *   4. estimate the band tilt from the ink-pixel rotation projection
 *      and deskew (rotate back) when |angle| >= 0.4 deg
 *   5. split the two TD3 lines; segment each line into 44 fixed
 *      pitch-grid windows (run centers only fit pitch/phase; run-based
 *      windows kept as fallback) so one spurious/broken ink run can no
 *      longer shift every subsequent character
 *   6. per cell: area-resample the grayscale window to 16x12 float
 *      (3x3 box smooth + centroid alignment), whole-row batch
 *      conv+fc inference; optional ±1px test-time augmentation
 *      (MRZ_OCR_TTA=1) averages logits over shifts
 *   7. ICAO 9303 syntax mask on the logits (strict TD3 field whitelist
 *      by default; MRZ_OCR_STRICT_ICAO=0 restores the loose mask)
 *   8. deterministic protections: relative ink gate + 3-neighbour '<'
 *      rule + trailing-filler lock (line 1)
 *   9. line-2 checksum beam search (top-3 candidates, lowest-margin
 *      positions first)
 */
/* _GNU_SOURCE must precede ANY system header include:
 * Bionic gates cpu_set_t/CPU_SET in <sched.h> behind __USE_GNU,
 * which is only defined when _GNU_SOURCE is seen first. */
#define _GNU_SOURCE
#include "mrz_ocr.h"
#include "cnn.h"
#include "template.h"

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#ifdef __linux__
/* Keep <sched.h> ahead of <pthread.h>: bionic's pthread.h includes
 * sched.h, so a later include would be skipped by its #pragma once
 * and cpu_set_t would never be visible. */
#include <sched.h>
#endif
#include <pthread.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

/* Strict ICAO 9303 TD3 field whitelist by default. MRZ_OCR_STRICT_ICAO=0
 * restores the historical loose mask (kept for legacy free-form
 * corpora / debugging). */
static int strict_icao_mode(void) {
    const char *e = getenv("MRZ_OCR_STRICT_ICAO");
    return !(e && e[0] && strcmp(e, "0") == 0);
}

/* ---- top-3 candidates among the allowed classes ---- */
typedef struct { int cand[3]; float p[3]; } top3_t;

static void top3_of(const float *probs, int allow[37], top3_t *t) {
    for (int k = 0; k < 3; ++k) { t->cand[k] = -1; t->p[k] = -1.0f; }
    for (int o = 0; o < 37; ++o) {
        if (allow[o] == 0) continue;
        float p = probs[o];
        if (p > t->p[0]) {
            t->p[2] = t->p[1]; t->cand[2] = t->cand[1];
            t->p[1] = t->p[0]; t->cand[1] = t->cand[0];
            t->p[0] = p; t->cand[0] = o;
        } else if (p > t->p[1]) {
            t->p[2] = t->p[1]; t->cand[2] = t->cand[1];
            t->p[1] = p; t->cand[1] = o;
        } else if (p > t->p[2]) {
            t->p[2] = p; t->cand[2] = o;
        }
    }
    int lt = mrz_ocr_glyph_index('<');
    if (t->cand[0] < 0) {
        t->cand[0] = t->cand[1] = t->cand[2] = lt;
        t->p[0] = t->p[1] = t->p[2] = 0.0f;
    } else {
        for (int k = 1; k < 3; ++k) {
            if (t->cand[k] < 0) { t->cand[k] = t->cand[0]; t->p[k] = 0.0f; }
        }
    }
}

/* ---- per-position character whitelist (syntax mask) ---- */
static void fill_allow(int allow[37], int line_idx, int col) {
    int d0 = mrz_ocr_glyph_index('0');
    int a0 = mrz_ocr_glyph_index('A');
    int lt = mrz_ocr_glyph_index('<');
    for (int i = 0; i < 37; ++i) allow[i] = 1;

    if (!strict_icao_mode()) {
        /* Loose legacy mask: numeric check digits, M/F/< sex, and the
         * line-1 col-1 filler only. Kept for legacy free-form corpora
         * that intentionally mix digits into alpha fields. */
        if (line_idx == 0) {
            if (col == 1) {
                for (int i = 0; i < 37; ++i) allow[i] = (i == lt);
            }
        } else {
            if (col == 9 || col == 19 || col == 27) {
                for (int i = 0; i < 37; ++i)
                    allow[i] = (i >= d0 && i <= d0 + 9) ? 1 : 0;
            } else if (col == 42 || col == 43) {
                for (int i = 0; i < 37; ++i)
                    allow[i] = (i >= d0 && i <= d0 + 9) ? 1 : 0;
                allow[lt] = 1;
            } else if (col == 20) {
                for (int i = 0; i < 37; ++i) allow[i] = 0;
                allow[mrz_ocr_glyph_index('M')] = 1;
                allow[mrz_ocr_glyph_index('F')] = 1;
                allow[lt] = 1;
            }
        }
        return;
    }

    /* Strict ICAO 9303 TD3 field mask. Every slot of a real passport
     * MRZ falls into exactly one of these sets, so the mask always
     * contains the ground truth and only removes impossible classes. */
    if (line_idx == 0) {
        if (col == 0) {
            /* Document type letter: ICAO 9303 limits this to the
             * standard one-letter types P/I/A/V/C + '<' filler. P->F,
             * P->B confusions at this column were a top full-match
             * error source; the mask only removes impossible letters. */
            for (int i = 0; i < 37; ++i) allow[i] = 0;
            for (const char *types = "PIAVC<"; *types; ++types)
                allow[mrz_ocr_glyph_index(*types)] = 1;
        } else if (col == 1) {
            /* Type-line filler: "P<..." */
            for (int i = 0; i < 37; ++i) allow[i] = (i == lt);
        } else if (col >= 2 && col <= 4) {
            /* Issuing state: 3 alpha */
            for (int i = 0; i < 37; ++i) allow[i] = (i >= a0 && i < a0 + 26);
        } else if (col >= 5) {
            /* Name field: A-Z + '<' (ICAO 9303 has no digits here) */
            for (int i = 0; i < 37; ++i)
                allow[i] = (i >= a0 && i < a0 + 26) || i == lt;
        }
        /* col 0 handled above: standard ICAO doc-type letters only. */
    } else {
        if (col <= 8) {
            /* Passport number: alnum + '<' filler */
            for (int i = 0; i < 37; ++i)
                allow[i] = (i >= a0 && i < a0 + 26) || (i >= d0 && i <= d0 + 9) || i == lt;
        } else if (col == 9 || col == 19 || col == 27) {
            for (int i = 0; i < 37; ++i)
                allow[i] = (i >= d0 && i <= d0 + 9) ? 1 : 0;
        } else if (col <= 12) {
            /* Nationality: alpha + '<' (unspecified nationality) */
            for (int i = 0; i < 37; ++i)
                allow[i] = (i >= a0 && i < a0 + 26) || i == lt;
        } else if ((col <= 18) || (col >= 21 && col <= 26)) {
            /* Birth / expiry dates: numeric ('<' = unknown date) */
            for (int i = 0; i < 37; ++i)
                allow[i] = ((i >= d0 && i <= d0 + 9) || i == lt) ? 1 : 0;
        } else if (col == 20) {
            for (int i = 0; i < 37; ++i) allow[i] = 0;
            allow[mrz_ocr_glyph_index('M')] = 1;
            allow[mrz_ocr_glyph_index('F')] = 1;
            allow[lt] = 1;
        } else if (col <= 41) {
            /* Personal number: alnum + '<' */
            for (int i = 0; i < 37; ++i)
                allow[i] = (i >= a0 && i < a0 + 26) || (i >= d0 && i <= d0 + 9) || i == lt;
        } else {
            /* col 42/43: check digits, '<' tolerated for the tail-drift
             * case; the checksum beam prefers digits when valid. */
            for (int i = 0; i < 37; ++i)
                allow[i] = ((i >= d0 && i <= d0 + 9) || i == lt) ? 1 : 0;
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
    char comp[41];
    memcpy(comp, l2, 10);            /* doc no + ck           (0..9)  */
    memcpy(comp + 10, l2 + 13, 7);   /* birth + ck            (13..19) */
    memcpy(comp + 17, l2 + 21, 22);  /* expiry + ck + personal (21..42) */
    comp[39] = 0;
    if (check_digit_of(comp, 39) != (l2[43] - '0')) return 0;
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
            memset(out, 0, sizeof(float) * CNN_IN_H * CNN_IN_W);
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
            memset(out, 0, sizeof(float) * CNN_IN_H * CNN_IN_W);
            for (int y = 0; y < CNN_IN_H; ++y)
                for (int x = 0; x < CNN_IN_W; ++x) {
                    int ny = y - delta;
                    if (ny >= 0 && ny < CNN_IN_H)
                        out[y][x] = tmp[ny][x];
                }
        }
    }
}

/* ---- shared ink-run finder over a line ----
 * Returns the number of ink runs (<=96) and fills centers/widths.
 * Used by both the run-based fallback segmenter and the pitch-grid
 * segmenter (which only consumes centers + median width). */
typedef struct {
    int s[96], e[96];
    int center[96];
    int width[96];
    int n;
    int medw;   /* median run width */
} runs_t;

static int find_runs(const uint8_t *bin, int W, int H, runs_t *r) {
    int *col_dark = (int *)calloc((size_t)W, sizeof(int));
    if (!col_dark) return -1;
    for (int x = 0; x < W; ++x) {
        int c = 0;
        for (int y = 0; y < H; ++y) if (bin[y * W + x]) c++;
        col_dark[x] = c;
    }
    int thr = H / 12; if (thr < 1) thr = 1;
    int in_run = 0, rs = 0;
    r->n = 0;
    for (int x = 0; x <= W; ++x) {
        int ink = (x < W && col_dark[x] >= thr);
        if (ink && !in_run) { in_run = 1; rs = x; }
        else if (!ink && in_run) {
            if (r->n < 96) {
                r->s[r->n] = rs; r->e[r->n] = x;
                r->center[r->n] = (rs + x) / 2;
                r->width[r->n] = x - rs;
                r->n++;
            }
            in_run = 0;
        }
    }
    free(col_dark);
    if (r->n == 0) { r->medw = 8; return 0; }
    int w[96]; int nw = 0;
    for (int i = 0; i < r->n; ++i) w[nw++] = r->width[i];
    for (int i = 1; i < nw; ++i) {
        int k = w[i], j = i - 1;
        while (j >= 0 && w[j] > k) { w[j+1] = w[j]; j--; }
        w[j+1] = k;
    }
    r->medw = w[nw / 2];
    if (r->medw < 6) r->medw = 6;
    return r->n;
}

/* Debug helper: dump the 16x12 glyphs a given image produces.
 * Enabled by MRZ_OCR_DUMP=<tag>. Writes ppm to /tmp/mrz_dump_<tag>_<l><c>.ppm */
static void mrz_ocr_dump_glyphs(const char *tag, int line_idx,
                                const float (*glyphs)[CNN_IN_H][CNN_IN_W],
                                int nchars, const mrz_ocr_rect_t *chars) {
    const char *env = getenv("MRZ_OCR_DUMP");
    if (!env || env[0] == 0 || strcmp(env, "0") == 0) return;
    char name[256];
    for (int c = 0; c < nchars; ++c) {
        snprintf(name, sizeof(name), "/tmp/mrz_dump_%s_l%d_c%02d_r%d_w%d.ppm",
                 tag, line_idx, c, chars[c].y, chars[c].w);
        FILE *f = fopen(name, "wb");
        if (!f) continue;
        fprintf(f, "P6\n16 12\n255\n");
        for (int y = 0; y < CNN_IN_H; ++y)
            for (int x = 0; x < CNN_IN_W; ++x) {
                uint8_t v = (uint8_t)(glyphs[c][y][x] * 255.0f);
                fputc(v, f); fputc(v, f); fputc(v, f);
            }
        fclose(f);
    }
}

/* ---- run-based fallback segmenter (original behaviour) ----
 * One window per ink run, width medw+8 centred on the run. Kept as
 * the fallback when the pitch grid cannot be fitted. */
static int segment_line_cnn(const uint8_t *bin, int W, int H,
                            mrz_ocr_rect_t *chars, int max_chars) {
    runs_t runs;
    if (find_runs(bin, W, H, &runs) < 0) return -1;
    if (runs.n < 1) return 0;
    int medw = runs.medw;
    int cell = medw + 8;                 /* ~ full pitch */
    if (cell < medw + 4) cell = medw + 4;
    int n = 0;
    for (int i = 0; i < runs.n; ++i) {
        if (n >= max_chars) break;
        int c = runs.center[i];
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

/* ---- pitch-grid segmenter (primary) ----
 * TD3 lines are exactly 44 monospaced cells. Instead of trusting each
 * ink run to be one character (fragile: one noise run shifts every
 * later window; one merged run loses characters), we fit the global
 * grid parameters:
 *
 *   pitch L  — median of consecutive run-center differences (plausible
 *              range only), autocorrelation fallback;
 *   phase c0 — exhaustive scan over [0, L), scored by weighted ink at
 *              the window edges (edges must sit in the inter-glyph
 *              gaps).
 *
 * Window POSITIONS come from the fitted grid; window WIDTH stays
 * medw+8 (the distribution the CNN was trained on). When no good grid
 * exists the caller falls back to the run-based segmenter. */
static int segment_line_grid(const uint8_t *bin, int W, int H,
                             mrz_ocr_rect_t *chars, int max_chars) {
    runs_t runs;
    if (find_runs(bin, W, H, &runs) < 0) return -1;
    int nrun = runs.n;
    if (nrun < 2) return -1;              /* not enough structure to fit */
    int medw = runs.medw;
    int cell = medw + 8;
    if (cell < medw + 4) cell = medw + 4;
    if (max_chars < 44) return -1;        /* grid mode needs all 44 slots */

    /* column projection, computed once and shared by pitch search and
     * phase scoring */
    int *col = (int *)calloc((size_t)W, sizeof(int));
    if (!col) return -1;
    for (int x = 0; x < W; ++x)
        for (int y = 0; y < H; ++y) col[x] += bin[y * W + x] ? 1 : 0;

    /* --- pitch estimate --- */
    int Lmin = W / 52; if (Lmin < 10) Lmin = 10;
    int Lmax = W / 32; if (Lmax < Lmin + 2) Lmax = Lmin + 2;
    int diffs[95]; int nd = 0;
    for (int i = 1; i < nrun; ++i) {
        int d = runs.center[i] - runs.center[i-1];
        if (d >= Lmin && d <= Lmax) diffs[nd++] = d;
    }
    int L0 = 0;
    if (nd > 0) {
        for (int i = 1; i < nd; ++i) {
            int k = diffs[i], j = i - 1;
            while (j >= 0 && diffs[j] > k) { diffs[j+1] = diffs[j]; j--; }
            diffs[j+1] = k;
        }
        L0 = diffs[nd / 2];
    }
    if (L0 < Lmin || L0 > Lmax) {
        /* Autocorrelation of the column projection. */
        double best = -1; int bestL = (Lmin + Lmax) / 2;
        for (int L = Lmin; L <= Lmax; ++L) {
            double acc = 0;
            for (int x = 0; x + L < W; ++x) acc += (double)col[x] * col[x + L];
            if (acc > best) { best = acc; bestL = L; }
        }
        L0 = bestL;
    }
    int Lcand[4]; int nL = 0;
    Lcand[nL++] = L0;
    if (L0 - 1 >= Lmin) Lcand[nL++] = L0 - 1;
    if (L0 + 1 <= Lmax) Lcand[nL++] = L0 + 1;
    /* Fractional candidate only when the median is clearly
     * non-integral (real photos: 547px/44 ~ 12.43px). Synthetic
     * corpora render at integral pitch, so the integer anchors above
     * reproduce the previous behaviour exactly. */

    /* --- phase search ---
     * The TRUE grid places every window edge inside an inter-glyph
     * gap. Cost = weighted ink near the 88 window edges (both edges,
     * +-2 px, distance-weighted); the true phase minimises it and the
     * weights break the zero-plateau towards the gap centre. An
     * earlier "ink in the middle half" metric was actively WRONG: a
     * 3-4 px shifted grid pulls strokes into the middle half and
     * scores HIGHER than the aligned one. */
    static const int wt[5] = {1, 2, 3, 2, 1};
    /* Sub-pixel phase scan: the true grid can sit on a fractional
     * offset, and a 0.25px slip at the first cell accumulates into a
     * full-cell mis-slice at the last one. Scanning c0 at quarter-px
     * steps keeps the whole 44-cell grid aligned; integer pitch keeps
     * synthetic corpora (integral pitch) bit-identical. */
    const double PSTEP = 0.25;
    double best_c0 = -1; int best_L = 0; double best_cost = 0;
    for (int li = 0; li < nL; ++li) {
        int L = Lcand[li];
        if (L <= 0) continue;
        int nsteps = (int)((double)L / PSTEP);
        for (int si = 0; si < nsteps; ++si) {
            double c0d = si * PSTEP;
            if (c0d + 43.0 * (double)L > (double)(W - 1 + cell)) continue;
            int cost = 0;
            for (int k = 0; k < 44; ++k) {
                double cx = c0d + (double)k * (double)L;
                int edges[2] = {
                    (int)(cx - (double)(cell / 2) + 0.5),
                    (int)(cx + (double)(cell / 2) - 1.0 + 0.5) };
                for (int e = 0; e < 2; ++e)
                    for (int dx = -2; dx <= 2; ++dx) {
                        int x = edges[e] + dx;
                        if (x < 0 || x >= W) continue;
                        cost += wt[dx + 2] * col[x];
                    }
            }
            if (best_c0 < 0 || cost < best_cost) {
                best_cost = cost; best_c0 = c0d; best_L = L;
            }
        }
    }
    free(col);
    if (best_c0 < 0) return -1;

    int n = 0;
    for (int k = 0; k < 44; ++k) {
        double cx = best_c0 + (double)k * (double)best_L;
        int x0 = (int)(cx - (double)(cell / 2) + 0.5);
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

/* ---- whole-line decode: syntax mask + protections + checksum beam ----
 * `probs` holds n rows of softmaxed class probabilities (possibly
 * TTA-averaged by the caller). Glyphs are needed for the ink gate. */
static void decode_row(const float (*glyphs)[CNN_IN_H][CNN_IN_W],
                       int n, int line_idx, char *dest, int *conf_avg,
                       const float *probs) {
    int total = n < 44 ? n : 44;
    top3_t t3[44];
    int allow[37];
    static int dbg = -1;
    if (dbg < 0) { const char *e = getenv("MRZ_OCR_TOPK"); dbg = (e && e[0] && strcmp(e,"0")!=0) ? 1 : 0; }
    for (int c = 0; c < n && c < 44; ++c) {
        fill_allow(allow, line_idx, c);
        top3_of(probs + c * CNN_OUT, allow, &t3[c]);
        if (dbg) {
            fprintf(stderr, "TOPK L%d c%02d: %c(%.3f) %c(%.3f) %c(%.3f)\n", line_idx,
                    c, MRZ_OCR_GLYPHS[t3[c].cand[0]].ch, t3[c].p[0],
                    MRZ_OCR_GLYPHS[t3[c].cand[1]].ch, t3[c].p[1],
                    MRZ_OCR_GLYPHS[t3[c].cand[2]].ch, t3[c].p[2]);
        }
    }

    /* patch ink per cell (relative '<' gate statistic) */
    float patch_ink[44];
    for (int c = 0; c < n && c < 44; ++c) {
        float ink = 0.0f;
        for (int y = 0; y < CNN_IN_H; ++y)
            for (int x = 0; x < CNN_IN_W; ++x)
                ink += glyphs[c][y][x];
        patch_ink[c] = ink;
        dest[c] = MRZ_OCR_GLYPHS[t3[c].cand[0]].ch;
    }
    if (n < 44) dest[n] = '\0'; else dest[44] = '\0';

    /* (a) Ink gate: '<' is the lowest-ink OCR-B glyph. If the model
     * says X/R but the patch carries far less ink than the line's
     * confident letters, it cannot be X/R. The threshold is RELATIVE
     * (0.55 x median ink of confident letter cells, p0 >= 0.85) so it
     * follows contrast/resolution instead of a hand-tuned constant;
     * falls back to the historical absolute 60 when the line has too
     * few confident letters. T/I/L/M are NOT gated (ink too close to
     * '<' to be safe). */
    {
        float letter_ink[44]; int nl = 0;
        for (int c = 0; c < total; ++c) {
            int g = t3[c].cand[0];
            if (g >= 10 && g <= 35 && t3[c].p[0] >= 0.85f)
                letter_ink[nl++] = patch_ink[c];
        }
        float gate = -1.0f;
        if (nl >= 5) {
            for (int i = 1; i < nl; ++i) {
                float k = letter_ink[i]; int j = i - 1;
                while (j >= 0 && letter_ink[j] > k) { letter_ink[j+1] = letter_ink[j]; j--; }
                letter_ink[j+1] = k;
            }
            gate = 0.55f * letter_ink[nl / 2];
        }
        for (int c = 0; c < total; ++c) {
            char ch0 = dest[c];
            if (ch0 == 'X' || ch0 == 'R') {
                float thr = (gate > 0.0f) ? gate : 60.0f;
                if (patch_ink[c] < thr) dest[c] = '<';
            }
        }
    }

    /* (b) 3-neighbour smoothing: an isolated X/R/T sandwiched between
     * '<' on both sides is syntactically impossible in ICAO 9303. */
    for (int c = 1; c + 1 < total; ++c) {
        if (dest[c-1] == '<' && dest[c+1] == '<' &&
            (dest[c] == 'X' || dest[c] == 'R' || dest[c] == 'T'))
            dest[c] = '<';
    }

    /* (c) Trailing "<" filler lock on Line 1: once >=5 consecutive '<'
     * the rest of the name field is guaranteed filler by ICAO 9303.
     * Line 2's tail keeps two numeric check digits, so only Line 1
     * locks. */
    if (line_idx == 0) {
        int locked = -1, run = 0;
        for (int c = 0; c < total; ++c) {
            if (dest[c] == '<') { run++; if (run >= 5) locked = c - run + 1; }
            else run = 0;
        }
        if (locked >= 0)
            for (int c = locked; c < total; ++c)
                dest[c] = '<';
    }

    /* (d) Line-2 checksum beam search: candidates sorted by margin
     * (p0-p1 ascending = least confident first); flips try top-3
     * candidates per position, then pairs over the 6 least confident
     * positions. Every column participates via the composite check
     * digit, so no column is excluded a priori. */
    if (line_idx == 1 && total == 44) {
        char best[45];
        memcpy(best, dest, 44); best[44] = 0;
        int ok = line2_checksum_ok(best);
        if (!ok) {
            int idxs[44]; float marg[44]; int picks = 0;
            for (int c = 0; c < 44; ++c) {
                if (t3[c].cand[0] != t3[c].cand[1]) {
                    idxs[picks] = c;
                    marg[picks] = t3[c].p[0] - t3[c].p[1];
                    picks++;
                }
            }
            for (int a = 1; a < picks; ++a) {
                int ki = idxs[a]; float km = marg[a];
                int j = a - 1;
                while (j >= 0 && marg[j] > km) { idxs[j+1] = idxs[j]; marg[j+1] = marg[j]; j--; }
                idxs[j+1] = ki; marg[j+1] = km;
            }
            if (picks > 10) picks = 10;
            int npair = picks < 6 ? picks : 6;
            for (int a = 0; a < picks && !ok; ++a) {
                for (int k = 1; k < 3 && !ok; ++k) {
                    int g = t3[idxs[a]].cand[k];
                    if (k > 1 && g == t3[idxs[a]].cand[1]) continue;
                    if (g == t3[idxs[a]].cand[0]) continue;
                    char t[45];
                    memcpy(t, dest, 44); t[44] = 0;
                    t[idxs[a]] = MRZ_OCR_GLYPHS[g].ch;
                    if (line2_checksum_ok(t)) { memcpy(best, t, 44); ok = 1; }
                }
            }
            for (int a = 0; a < npair && !ok; ++a)
                for (int b = a + 1; b < npair && !ok; ++b)
                    for (int ka = 1; ka < 3 && !ok; ++ka)
                        for (int kb = 1; kb < 3 && !ok; ++kb) {
                            int ga = t3[idxs[a]].cand[ka];
                            int gb = t3[idxs[b]].cand[kb];
                            if (ga == t3[idxs[a]].cand[0] || gb == t3[idxs[b]].cand[0]) continue;
                            if (ka > 1 && ga == t3[idxs[a]].cand[1]) continue;
                            if (kb > 1 && gb == t3[idxs[b]].cand[1]) continue;
                            char t[45];
                            memcpy(t, dest, 44); t[44] = 0;
                            t[idxs[a]] = MRZ_OCR_GLYPHS[ga].ch;
                            t[idxs[b]] = MRZ_OCR_GLYPHS[gb].ch;
                            if (line2_checksum_ok(t)) { memcpy(best, t, 44); ok = 1; }
                        }
            memcpy(dest, best, 44);
            if (n < 44) dest[n] = '\0'; else dest[44] = '\0';
        }
    }

    int sum = 0;
    for (int c = 0; c < total; ++c)
        sum += (int)((t3[c].p[0] > 0 ? t3[c].p[0] : 0.0f) * 100);
    *conf_avg = total > 0 ? sum / total : 0;
}

/* Band tilt estimation + deskew live in the shared geometry layer
 * (src/mrz_geom.c): mrz_estimate_band_skew() / mrz_rotate_band().
 * They are used by both the CNN and legacy backends. */

/* ---- Process-static workspace scratchpad ----
 *
 * The CNN backend needs ~1.3 MB of working buffers per image
 * (gray + bin + band_gray + band_bin + rot + line_pixels + glyphs
 * + feats + probs + cnn_t + overhead).  Allocating that on every
 * call (12 malloc/free pairs in the original) was the largest
 * source of jank on the NDK ARMv7 build.
 *
 * The safe alternative: a single process-static scratchpad
 * (4 MB, 64-byte aligned) handed out by a bump allocator.
 * The scratchpad is reset on every recogniser entry; the JNI
 * dispatcher is single-threaded so two concurrent calls are not
 * supported.  No free() in the hot path: the bump allocator
 * only moves a pointer.
 *
 * pthread_once + a static guard initialise the scratchpad
 * exactly once on the first call.  All pointers below are
 * 64-byte aligned so they sit on cache-line boundaries
 * (helpful for the cnn_t weight matrix read pattern).
 */
/* 8 MB scratchpad split into two 4 MB halves (Step 2).  Line 0
 * uses half 0; line 1 uses half 1.  Even on the current
 * sequential path this gives the bump allocator disjoint
 * regions so adding pthread-based dual-line parallelism
 * later is just a one-line toggle (MRZ_OCR_PAR=1). */
#define MRZ_SCRATCH_BYTES  (8u * 1024u * 1024u)
static uint8_t  g_mrz_scratch[MRZ_SCRATCH_BYTES]
                __attribute__((aligned(64)));
/* g_mrz_scratch_off removed (Step 2): per-half offsets
 * g_mrz_scratch_off0/1 are tracked below. */
static int      g_mrz_scratch_active = 0;
static size_t   g_mrz_scratch_off0 = 0;
static size_t   g_mrz_scratch_off1 = 0;
static pthread_once_t g_mrz_scratch_once = PTHREAD_ONCE_INIT;
static int      g_mrz_scratch_cnn_ready = 0;
static cnn_t    g_mrz_scratch_cnn_template;  /* one-shot weights cache */

static void mrz_scratch_init_once(void) {
    g_mrz_scratch_off0 = 0;
    g_mrz_scratch_off1 = 0;
    if (!g_mrz_scratch_cnn_ready) {
        cnn_default_init(&g_mrz_scratch_cnn_template);
        g_mrz_scratch_cnn_ready = 1;
    }
}

/* 64-byte aligned bump allocator. */
#define MRZ_SCRATCH_HALF_BYTES (MRZ_SCRATCH_BYTES / 2u)


void mrz_scratch_select_half(int half) { g_mrz_scratch_active = half; }
static void *mrz_scratch_active_base(void) {
    return g_mrz_scratch + (g_mrz_scratch_active == 0 ? 0 : MRZ_SCRATCH_HALF_BYTES);
}
static size_t mrz_scratch_active_off(void) {
    return g_mrz_scratch_active == 0 ? g_mrz_scratch_off0 : g_mrz_scratch_off1;
}
static void mrz_scratch_set_active_off(size_t off) {
    if (g_mrz_scratch_active == 0) g_mrz_scratch_off0 = off;
    else                          g_mrz_scratch_off1 = off;
}

static void *mrz_scratch_alloc(size_t need) {
    pthread_once(&g_mrz_scratch_once, mrz_scratch_init_once);
    size_t aligned = (need + 63u) & ~(size_t)63u;
    size_t off = mrz_scratch_active_off();
    if (off + aligned > MRZ_SCRATCH_HALF_BYTES) return NULL;
    void *p = mrz_scratch_active_base() + off;
    mrz_scratch_set_active_off(off + aligned);
    return p;
}

/* Reset for a fresh recogniser call.  O(1); no system calls. */
static void mrz_scratch_reset(void) {
    g_mrz_scratch_off0 = 0;
    g_mrz_scratch_off1 = 0;
    g_mrz_scratch_active = 0;
}

/* Per-line worker context (Step 2: reserved for the future
 * dual-line pthread parallelism).  When MRZ_OCR_PAR=1, the
 * main thread spawns one worker that runs the CNN forward
 * for line 1 in parallel with the main thread doing line 0.
 * Per-line state lives in disjoint scratchpad halves so the
 * bump allocator never aliases across threads. */
typedef struct {
    const cnn_t   *net;
    /* glyphs is the canonical input; tta_glyphs holds the +-1 shift
     * snapshots so the worker can run TTA in-thread without the main
     * thread having to share band_gray. */
    float        (*glyphs)[CNN_IN_H][CNN_IN_W];
    float        (*tta_glyphs)[CNN_IN_H][CNN_IN_W];
    float         *feats;
    float         *probs;
    float         *tta_probs;
    int            nchars;
    /* decode_row inputs */
    const uint8_t *band_gray;
    int            bw, bh;
    const mrz_ocr_rect_t *chars;
    int            base_y;
    int            tta;
    /* outputs */
    char          *line_out;
    int           *conf_avg;
} mrz_line1_job_t;

/* CPU affinity: the 4x Cortex-A53 are homogeneous, but the Linux
 * scheduler (schedplus) still bounces a worker thread across cores,
 * which thrashes the L1 (32KB)/L2 (256KB) caches and inflates the
 * p50 on every image.  Pin the worker to a specific core at spawn so
 * it stays put for the whole batch.  Failure is non-fatal (falls back
 * to the scheduler's default placement). */
#ifdef __linux__
static void mrz_pin_to_cpu(unsigned int cpu) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET((int)cpu, &mask);
    sched_setaffinity(0, sizeof(mask), &mask);
}
#else
static void mrz_pin_to_cpu(unsigned int cpu) { (void)cpu; }
#endif

static void *mrz_line1_worker(void *arg) {
    mrz_line1_job_t *j = (mrz_line1_job_t *)arg;
    mrz_pin_to_cpu(3);  /* keep the worker away from the main thread's core */
    cnn_row_features_batch(j->net, j->glyphs, j->nchars, j->feats);
    cnn_fc_batch(j->net, j->feats, j->nchars, j->probs);
    if (j->tta) {
        /* +-1px horizontal shift TTA averaged into j->probs. */
        for (int c = 0; c < j->nchars; ++c)
            for (int o = 0; o < CNN_OUT; ++o)
                j->tta_probs[c * CNN_OUT + o] = j->probs[c * CNN_OUT + o];
        for (int sh = -1; sh <= 1; sh += 2) {
            for (int c = 0; c < j->nchars; ++c)
                resample_char_gray(j->band_gray, j->bw, j->bh,
                                   j->chars[c].x + sh,
                                   j->base_y + j->chars[c].y,
                                   j->chars[c].w, j->chars[c].h,
                                   NULL, 0, j->tta_glyphs[c]);
            cnn_row_features_batch(j->net, j->tta_glyphs, j->nchars, j->feats);
            cnn_fc_batch(j->net, j->feats, j->nchars, j->probs);
            for (int c = 0; c < j->nchars; ++c)
                for (int o = 0; o < CNN_OUT; ++o)
                    j->tta_probs[c * CNN_OUT + o] += j->probs[c * CNN_OUT + o];
        }
        for (int c = 0; c < j->nchars; ++c)
            for (int o = 0; o < CNN_OUT; ++o)
                j->probs[c * CNN_OUT + o] = j->tta_probs[c * CNN_OUT + o] / 3.0f;
    }
    decode_row((const float (*)[CNN_IN_H][CNN_IN_W])j->glyphs,
               j->nchars, 1, j->line_out, j->conf_avg, j->probs);
    return NULL;
}

/* Convenience: typed allocations for the ten safe fields. */
static uint8_t *mrz_scratch_gray    (size_t n)              { return (uint8_t *)mrz_scratch_alloc(n); }
static uint8_t *mrz_scratch_bin     (size_t n)              { return (uint8_t *)mrz_scratch_alloc(n); }
static uint8_t *mrz_scratch_band_gray(size_t n)             { return (uint8_t *)mrz_scratch_alloc(n); }
static uint8_t *mrz_scratch_band_bin (size_t n)             { return (uint8_t *)mrz_scratch_alloc(n); }
static uint8_t *mrz_scratch_rot     (size_t n)              { return (uint8_t *)mrz_scratch_alloc(n); }
static uint8_t *mrz_scratch_norm    (size_t n)              { return (uint8_t *)mrz_scratch_alloc(n); }
static uint8_t *mrz_scratch_line    (size_t n)              { return (uint8_t *)mrz_scratch_alloc(n); }
static float   *mrz_scratch_glyphs  (size_t n_floats)       { return (float *)mrz_scratch_alloc(n_floats * sizeof(float)); }
static float   *mrz_scratch_feats   (size_t n_floats)       { return (float *)mrz_scratch_alloc(n_floats * sizeof(float)); }
static float   *mrz_scratch_probs   (size_t n_floats)       { return (float *)mrz_scratch_alloc(n_floats * sizeof(float)); }
static cnn_t   *mrz_scratch_cnn     (void)                  { return (cnn_t *)mrz_scratch_alloc(sizeof(cnn_t)); }

/* ---- top-level pipeline ---- */
mrz_ocr_status_t mrz_ocr_recognise_cnn(const face_image_t *img,
                                       mrz_ocr_result_t *out) {
    mrz_scratch_reset();   /* O(1) reset before any alloc */
    if (!img || !out) return MRZ_OCR_ERR_LOAD;
    memset(out, 0, sizeof(*out));
    int W = img->width, H = img->height;
    if (W <= 0 || H <= 0) return MRZ_OCR_ERR_LOAD;

    const char *tm = getenv("MRZ_OCR_TIMING");
    struct timespec _t0, _t1;
    double _ms_gray = 0, _ms_band = 0, _ms_resample = 0, _ms_decode = 0;
    if (tm && tm[0] && strcmp(tm, "0")) clock_gettime(CLOCK_MONOTONIC, &_t0);

    /* Heap workspaces (the CNN weights alone are ~41 KB — keeping the
     * whole pipeline off the stack matters for embedded thread stacks). */
    size_t n = (size_t)W * H;
    uint8_t *gray = mrz_scratch_gray(n);
    uint8_t *bin = NULL, *band_gray = NULL, *band_bin = NULL, *rot = NULL;
    uint8_t *line_pixels = NULL;
    cnn_t *net = NULL;
    float (*glyphs)[CNN_IN_H][CNN_IN_W] = NULL;
    float (*glyphs_tta)[CNN_IN_H][CNN_IN_W] = NULL;
    float *feats = NULL, *probs = NULL, *probs_tta = NULL;
    if (!gray) return MRZ_OCR_ERR_LOAD;

    mrz_ocr_status_t status = MRZ_OCR_OK;
    face_to_grayscale(img, gray);

    /* 1. Global Otsu -> coarse bin map for band location. Polarity is
     * normalised first (white-on-black scans would otherwise mark the
     * background as ink everywhere downstream). */
    int t = mrz_ocr_otsu(gray, (int)n);
    {
        size_t dark = 0;
        for (size_t i = 0; i < n; ++i) if (gray[i] < t) dark++;
        if (dark * 2 > n) {
            for (size_t i = 0; i < n; ++i) gray[i] = (uint8_t)(255 - gray[i]);
            t = 255 - t;
        }
    }
    bin = mrz_scratch_bin(n);
    if (!bin) { status = MRZ_OCR_ERR_LOAD; goto cleanup; }
    for (size_t i = 0; i < n; ++i) bin[i] = (gray[i] < t) ? 1 : 0;
    if (tm && tm[0] && strcmp(tm, "0")) { clock_gettime(CLOCK_MONOTONIC, &_t1); _ms_gray = (_t1.tv_sec-_t0.tv_sec)*1e3+(_t1.tv_nsec-_t0.tv_nsec)/1e6; _t0=_t1; }

    /* 2. Locate the MRZ band. */
    mrz_ocr_rect_t band;
    if (mrz_ocr_locate_band(bin, W, H, &band) != 0) {
        status = MRZ_OCR_ERR_NO_BAND; goto cleanup;
    }
    int bx = band.x, by = band.y, bw = band.w, bh = band.h;
    out->band_x = bx; out->band_y = by; out->band_w = bw; out->band_h = bh;
    if (bw < MRZ_OCR_GLYPH_W * 5 || bh < MRZ_OCR_GLYPH_H) {
        status = MRZ_OCR_ERR_BAD_GEOMETRY; goto cleanup;
    }

    /* 3. Band-local refine: crop the band's grayscale, Otsu inside the
     * band only (a global threshold over a full passport page can be
     * dominated by the photo / background), estimate + remove tilt. */
    band_gray = mrz_scratch_band_gray((size_t)bw * bh);
    band_bin  = mrz_scratch_band_bin ((size_t)bw * bh);
    if (!band_gray || !band_bin) { status = MRZ_OCR_ERR_LOAD; goto cleanup; }
    for (int y = 0; y < bh; ++y)
        memcpy(band_gray + y * bw, gray + (by + y) * W + bx, bw);
    {
        int tb = mrz_ocr_otsu(band_gray, bw * bh);
        for (int i = 0; i < bw * bh; ++i) band_bin[i] = (band_gray[i] < tb) ? 1 : 0;
        double deg = mrz_estimate_band_skew(band_bin, bw, bh);
        if (deg != 0.0) {
            rot = mrz_scratch_rot((size_t)bw * bh);
            if (!rot) { status = MRZ_OCR_ERR_LOAD; goto cleanup; }
            /* Bilinear deskew is OPT-IN (MRZ_OCR_BILINEAR=1). On the
             * current (nearest-neighbour-trained) weights it raises
             * char/cksum but LOWERS full-match, i.e. it changes the
             * input distribution and only pays off after a
             * distribution-matched retrain. Default = NN stays. */
            const char *bl_env = getenv("MRZ_OCR_BILINEAR");
            int bl = (bl_env && bl_env[0] && strcmp(bl_env, "0") != 0);
            mrz_rotate_band(band_gray, bw, bh, deg, 255, bl, rot);
            memcpy(band_gray, rot, (size_t)bw * bh);
            for (int i = 0; i < bw * bh; ++i) band_bin[i] = (band_gray[i] < tb) ? 1 : 0;
        }
    }
    if (tm && tm[0] && strcmp(tm, "0")) { clock_gettime(CLOCK_MONOTONIC, &_t1); _ms_band=(_t1.tv_sec-_t0.tv_sec)*1e3+(_t1.tv_nsec-_t0.tv_nsec)/1e6; _t0=_t1; }

    /* 4b. Local-contrast normalisation against illumination gradients.
     * OPT-IN (MRZ_OCR_LOCAL_NORM=1): off by default.  A/B on the 1380
     * eval corpus shows it raises char-level accuracy a hair (95.6->
     * 97.6 / 98.9->99.1 / cksum 94.6->97.1) but LOWERS realistic
     * full-match (45.0%->37.9%) and adds ~0.5 ms -- it is NOT the
     * path to the realistic 45->85% full-match goal. Kept for field
     * A/B; do not flip the default without a full-match win. */
    {
        const char *ln_env = getenv("MRZ_OCR_LOCAL_NORM");
        int ln = (ln_env && ln_env[0] && strcmp(ln_env, "0") != 0);
        if (ln) {
            uint8_t *norm = mrz_scratch_norm((size_t)bw * bh);
            if (!norm) { status = MRZ_OCR_ERR_LOAD; goto cleanup; }
            mrz_normalize_local_contrast(band_gray, bw, bh, norm);
            memcpy(band_gray, norm, (size_t)bw * bh);
            /* scratchpad: no free */
        }
    }

    /* 4. Split the two TD3 lines on the deskewed band bin map. */
    mrz_ocr_rect_t lines[2];
    if (mrz_ocr_split_lines(band_bin, bw, bh, lines) != 0) {
        status = MRZ_OCR_ERR_BAD_LINES; goto cleanup;
    }

    net = mrz_scratch_cnn();
    glyphs     = (float (*)[CNN_IN_H][CNN_IN_W])mrz_scratch_glyphs(44 * (CNN_IN_H * CNN_IN_W));
    glyphs_tta = (float (*)[CNN_IN_H][CNN_IN_W])mrz_scratch_glyphs(44 * (CNN_IN_H * CNN_IN_W));
    feats      = mrz_scratch_feats(44 * CNN_FLAT);
    probs      = mrz_scratch_probs(44 * CNN_OUT);
    probs_tta  = mrz_scratch_probs(44 * CNN_OUT);
    if (!net || !glyphs || !glyphs_tta || !feats || !probs || !probs_tta) {
        status = MRZ_OCR_ERR_LOAD; goto cleanup;
    }
    /* Populate weights from the static template (init once at module
     * load via pthread_once).  The 140 KB memcpy is one per image,
     * far cheaper than the 12 malloc/free pairs the bump allocator
     * replaces. */
    pthread_once(&g_mrz_scratch_once, mrz_scratch_init_once);
    *net = g_mrz_scratch_cnn_template;
    {
        const char *tta_env = getenv("MRZ_OCR_TTA");
        int tta = (tta_env && tta_env[0] && strcmp(tta_env, "0") != 0);

        /* Step 2 fork:
         *   default (sequential)  -> single loop, cache-hot
         *   MRZ_OCR_PAR=1         -> serialise resample, then
         *                            fork a worker for line 1's
         *                            CNN forward while the main
         *                            thread runs line 0.
         * The sequential path keeps the original cache-hot
         * pattern (resample -> cnn -> decode back-to-back) so
         * the L1 cache stays warm on the just-written glyph
         * buffer; splitting resample and cnn into separate
         * passes was measurably slower on A53 (~+5 ms p50). */
        const char *par_env = getenv("MRZ_OCR_PAR");
        int do_par = par_env && par_env[0] && strcmp(par_env, "0") != 0;
        if (do_par) {
            /* Workspace for line 0/1 (chars from segmenter, line
             * pixels, base offset, line height). */
            static mrz_ocr_rect_t chars0[44], chars1[44];
            mrz_ocr_rect_t *chars_l[2]    = { chars0, chars1 };
            int nchars_l[2]                = { 0, 0 };
            int line_h_l[2]                = { 0, 0 };
            int base_l[2]                  = { 0, 0 };
            uint8_t *line_pixels_l[2]     = { NULL, NULL };
            /* Per-line glyph buffers live in disjoint scratchpad
             * halves.  Save the addresses in static globals so
             * the worker thread + main thread can both see them. */
            static float (*glyphs0_keep)[CNN_IN_H][CNN_IN_W];
            static float (*glyphs1_keep)[CNN_IN_H][CNN_IN_W];

            /* Pass 1: serialise the resample step into the
             * per-half glyphs_keep buffers so the second pass
             * below doesn't redo the work. */
            for (int li = 0; li < 2; ++li) {
                mrz_scratch_select_half(li);
                base_l[li]   = (li == 0) ? lines[0].y : lines[1].y;
                line_h_l[li] = (li == 0) ? lines[0].h : lines[1].h;
                if (line_h_l[li] <= 0) continue;
                line_pixels_l[li] = mrz_scratch_line((size_t)bw * line_h_l[li]);
                if (!line_pixels_l[li]) { status = MRZ_OCR_ERR_LOAD; goto cleanup; }
                for (int y = 0; y < line_h_l[li]; ++y)
                    memcpy(line_pixels_l[li] + y * bw,
                           band_bin + (base_l[li] + y) * bw, bw);

                int nchars = segment_line_grid(line_pixels_l[li], bw,
                                              line_h_l[li], chars_l[li], 44);
                if (nchars <= 0)
                    nchars = segment_line_cnn(line_pixels_l[li], bw,
                                              line_h_l[li], chars_l[li], 44);
                if (nchars < 30) {
                    status = MRZ_OCR_ERR_BAD_LINES; goto cleanup;
                }
                if (nchars > 44) nchars = 44;
                nchars_l[li] = nchars;
                float (*glyphs_l)[CNN_IN_H][CNN_IN_W] =
                    (float (*)[CNN_IN_H][CNN_IN_W])mrz_scratch_glyphs(44 * (CNN_IN_H * CNN_IN_W));
                if (li == 0) glyphs0_keep = glyphs_l;
                else          glyphs1_keep = glyphs_l;
                for (int c = 0; c < nchars; ++c)
                    resample_char_gray(band_gray, bw, bh,
                                       chars_l[li][c].x,
                                       base_l[li] + chars_l[li][c].y,
                                       chars_l[li][c].w, chars_l[li][c].h,
                                       NULL, 0, glyphs_l[c]);
            }
            if (tm && tm[0] && strcmp(tm, "0")) {
                clock_gettime(CLOCK_MONOTONIC, &_t1);
                _ms_resample += (_t1.tv_sec-_t0.tv_sec)*1e3 + (_t1.tv_nsec-_t0.tv_nsec)/1e6;
                _t0 = _t1;
            }

            /* Allocate the per-line feat/prob buffers.  Glyphs
             * are already populated above; we still need
             * glyphs_tta for TTA on each line. */
            mrz_scratch_select_half(0);
            float (*glyphs0_tta)[CNN_IN_H][CNN_IN_W] =
                (float (*)[CNN_IN_H][CNN_IN_W])mrz_scratch_glyphs(44 * (CNN_IN_H * CNN_IN_W));
            float *feats0     = mrz_scratch_feats(44 * CNN_FLAT);
            float *probs0     = mrz_scratch_probs(44 * CNN_OUT);
            float *probs0_tta = mrz_scratch_probs(44 * CNN_OUT);
            mrz_scratch_select_half(1);
            float (*glyphs1_tta)[CNN_IN_H][CNN_IN_W] =
                (float (*)[CNN_IN_H][CNN_IN_W])mrz_scratch_glyphs(44 * (CNN_IN_H * CNN_IN_W));
            float *feats1     = mrz_scratch_feats(44 * CNN_FLAT);
            float *probs1     = mrz_scratch_probs(44 * CNN_OUT);
            float *probs1_tta = mrz_scratch_probs(44 * CNN_OUT);
            if (!glyphs0_tta || !feats0 || !probs0 || !probs0_tta ||
                !glyphs1_tta || !feats1 || !probs1 || !probs1_tta) {
                status = MRZ_OCR_ERR_LOAD; goto cleanup;
            }

            /* Pass 1 above already populated glyphs0_keep and
             * glyphs1_keep, so we skip the original wasteful
             * re-resample pass.  The line-1 worker now also runs
             * TTA + decode_row in-thread so the join is
             * essentially free on the main side. */
            pthread_t worker;
            mrz_line1_job_t job;
            int worker_spawned = 0;
            int line1_did_inline = 0;
            if (nchars_l[1] >= 30) {
                job.net       = net;
                job.glyphs    = glyphs1_keep;
                job.tta_glyphs= glyphs1_tta;
                job.feats     = feats1;
                job.probs     = probs1;
                job.tta_probs = probs1_tta;
                job.nchars    = nchars_l[1];
                job.band_gray = band_gray;
                job.bw        = bw;
                job.bh        = bh;
                job.chars     = chars_l[1];
                job.base_y    = base_l[1];
                job.tta       = tta;
                job.line_out  = out->line2;
                job.conf_avg  = &out->line2_avg_conf;
                if (pthread_create(&worker, NULL, mrz_line1_worker, &job) == 0) {
                    worker_spawned = 1;
                } else {
                    /* Fallback: in-line (also does TTA+decode
                     * so the pipeline stays byte-exact). */
                    line1_did_inline = 1;
                    mrz_line1_worker(&job);
                }
            }

            /* Line 0 forward (and TTA) on the main thread. */
            if (nchars_l[0] >= 30) {
                cnn_row_features_batch(net, glyphs0_keep, nchars_l[0], feats0);
                cnn_fc_batch(net, feats0, nchars_l[0], probs0);
                if (tta) {
                    for (int c = 0; c < nchars_l[0]; ++c)
                        for (int o = 0; o < CNN_OUT; ++o)
                            probs0_tta[c * CNN_OUT + o] = probs0[c * CNN_OUT + o];
                    for (int sh = -1; sh <= 1; sh += 2) {
                        for (int c = 0; c < nchars_l[0]; ++c)
                            resample_char_gray(band_gray, bw, bh,
                                               chars_l[0][c].x + sh,
                                               base_l[0] + chars_l[0][c].y,
                                               chars_l[0][c].w, chars_l[0][c].h,
                                               NULL, 0, glyphs0_tta[c]);
                        cnn_row_features_batch(net, glyphs0_tta, nchars_l[0], feats0);
                        cnn_fc_batch(net, feats0, nchars_l[0], probs0);
                        for (int c = 0; c < nchars_l[0]; ++c)
                            for (int o = 0; o < CNN_OUT; ++o)
                                probs0_tta[c * CNN_OUT + o] += probs0[c * CNN_OUT + o];
                    }
                    for (int c = 0; c < nchars_l[0]; ++c)
                        for (int o = 0; o < CNN_OUT; ++o)
                            probs0[c * CNN_OUT + o] = probs0_tta[c * CNN_OUT + o] / 3.0f;
                }
                decode_row((const float (*)[CNN_IN_H][CNN_IN_W])glyphs0_keep,
                           nchars_l[0], 0, out->line1, &out->line1_avg_conf, probs0);
                out->line1_len = nchars_l[0];
                mrz_ocr_dump_glyphs("img", 0, glyphs0_keep, nchars_l[0], chars_l[0]);
            }
            if (tm && tm[0] && strcmp(tm, "0")) {
                clock_gettime(CLOCK_MONOTONIC, &_t1);
                _ms_decode += (_t1.tv_sec-_t0.tv_sec)*1e3 + (_t1.tv_nsec-_t0.tv_nsec)/1e6;
                _t0 = _t1;
            }

            /* Join the line-1 worker.  If pthread_create failed
             * we already ran the job in-line; in that case the
             * decode_row inside the job has already written
             * out->line2. */
            if (worker_spawned) {
                pthread_join(worker, NULL);
            }
            if (nchars_l[1] >= 30) {
                out->line2_len = nchars_l[1];
                mrz_ocr_dump_glyphs("img", 1, glyphs1_keep, nchars_l[1], chars_l[1]);
            }
            if (tm && tm[0] && strcmp(tm, "0")) {
                clock_gettime(CLOCK_MONOTONIC, &_t1);
                _ms_decode += (_t1.tv_sec-_t0.tv_sec)*1e3 + (_t1.tv_nsec-_t0.tv_nsec)/1e6;
                _t0 = _t1;
            }
            (void)line1_did_inline; /* silence unused-var */
        } else {
            /* Sequential: cache-hot single loop.  Resample,
             * cnn forward, TTA, decode back-to-back per line so
             * the just-written glyph buffer is still in L1 when
             * cnn_row_features_batch reads it. */
            for (int li = 0; li < 2; ++li) {
                int base = (li == 0) ? lines[0].y : lines[1].y;
                int line_h = (li == 0) ? lines[0].h : lines[1].h;
                if (line_h <= 0) continue;
                mrz_scratch_select_half(li);
                line_pixels = mrz_scratch_line((size_t)bw * line_h);
                if (!line_pixels) { status = MRZ_OCR_ERR_LOAD; goto cleanup; }
                for (int y = 0; y < line_h; ++y)
                    memcpy(line_pixels + y * bw, band_bin + (base + y) * bw, bw);

                mrz_ocr_rect_t chars[44];
                int nchars = segment_line_grid(line_pixels, bw, line_h, chars, 44);
                if (nchars <= 0)
                    nchars = segment_line_cnn(line_pixels, bw, line_h, chars, 44);
                if (nchars < 30) {
                    line_pixels = NULL;
                    status = MRZ_OCR_ERR_BAD_LINES; goto cleanup;
                }
                if (nchars > 44) nchars = 44;
                for (int c = 0; c < nchars; ++c)
                    resample_char_gray(band_gray, bw, bh,
                                       chars[c].x, base + chars[c].y,
                                       chars[c].w, chars[c].h,
                                       NULL, 0, glyphs[c]);
                if (tm && tm[0] && strcmp(tm, "0")) {
                    clock_gettime(CLOCK_MONOTONIC, &_t1);
                    _ms_resample += (_t1.tv_sec-_t0.tv_sec)*1e3 + (_t1.tv_nsec-_t0.tv_nsec)/1e6;
                    _t0 = _t1;
                }

                cnn_row_features_batch(net, glyphs, nchars, feats);
                cnn_fc_batch(net, feats, nchars, probs);
                if (tta) {
                    for (int c = 0; c < nchars; ++c)
                        for (int o = 0; o < CNN_OUT; ++o) probs_tta[c * CNN_OUT + o] = probs[c * CNN_OUT + o];
                    for (int sh = -1; sh <= 1; sh += 2) {
                        for (int c = 0; c < nchars; ++c)
                            resample_char_gray(band_gray, bw, bh,
                                               chars[c].x + sh, base + chars[c].y,
                                               chars[c].w, chars[c].h,
                                               NULL, 0, glyphs_tta[c]);
                        cnn_row_features_batch(net, glyphs_tta, nchars, feats);
                        cnn_fc_batch(net, feats, nchars, probs);
                        for (int c = 0; c < nchars; ++c)
                            for (int o = 0; o < CNN_OUT; ++o)
                                probs_tta[c * CNN_OUT + o] += probs[c * CNN_OUT + o];
                    }
                    for (int c = 0; c < nchars; ++c)
                        for (int o = 0; o < CNN_OUT; ++o)
                            probs[c * CNN_OUT + o] = probs_tta[c * CNN_OUT + o] / 3.0f;
                }

                char *dest = (li == 0) ? out->line1 : out->line2;
                int *conf = (li == 0) ? &out->line1_avg_conf : &out->line2_avg_conf;
                decode_row((const float (*)[CNN_IN_H][CNN_IN_W])glyphs, nchars, li, dest, conf, probs);
                if (li == 0) out->line1_len = nchars; else out->line2_len = nchars;
                mrz_ocr_dump_glyphs("img", li, glyphs, nchars, chars);
                line_pixels = NULL;
                if (tm && tm[0] && strcmp(tm, "0")) {
                    clock_gettime(CLOCK_MONOTONIC, &_t1);
                    _ms_decode += (_t1.tv_sec-_t0.tv_sec)*1e3 + (_t1.tv_nsec-_t0.tv_nsec)/1e6;
                    _t0 = _t1;
                }
            }
        }
    }

cleanup:
    /* scratchpad: no per-call frees.  The whole buffer is reset
     * on the next recogniser entry (mrz_scratch_reset()). */
    (void)probs_tta; (void)probs; (void)feats;
    (void)glyphs_tta; (void)glyphs; (void)net;
    (void)line_pixels;
    (void)rot; (void)band_bin; (void)band_gray;
    (void)bin; (void)gray;
    if (tm && tm[0] && strcmp(tm, "0")) {
        fprintf(stderr, "TIMING gray=%.2fms band+deskew=%.2fms resample=%.2fms decode=%.2fms\n",
                _ms_gray, _ms_band, _ms_resample, _ms_decode);
#ifdef __ANDROID__
        /* Mirror to logcat under the same tag the JNI shim uses
         * ("MrzNative") so `adb logcat -s MrzNative:TIMING` shows
         * the per-stage breakdown while the bench is running. */
        extern int __android_log_print(int, const char *, const char *, ...);
        __android_log_print(6 /*ANDROID_LOG_ERROR*/, "MrzNative",
            "TIMING gray=%.2fms band+deskew=%.2fms resample=%.2fms decode=%.2fms",
            _ms_gray, _ms_band, _ms_resample, _ms_decode);
#endif
    }
    return status;
}
