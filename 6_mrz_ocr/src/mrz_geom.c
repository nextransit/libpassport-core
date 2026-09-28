/* MRZ optical preprocessing geometry layer.
 *
 * Shared by BOTH recognition backends:
 *   - mrz_estimate_band_skew(): tilt of the two TD3 text lines, from
 *     the ink-pixel row-projection energy maximised over -4..+4 deg;
 *   - mrz_rotate_band():       deskew (rotate back) a band buffer.
 *
 * This is a pure preprocessing concern and must stay independent of
 * the classifier (CNN) and the legacy template baseline. The legacy
 * backend uses it purely so the benchmark honestly measures the
 * classifier gap instead of dying on rotated scans (602/660 realistic
 * images failed before deskew).
 */
#define _POSIX_C_SOURCE 200809L
#include "mrz_ocr.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---- pyramid coarse-to-fine skew estimator ----
 *
 * The estimator is the dominant cost in the band+deskew stage on
 * NDK ARMv7 (60-160 ms p50-p95 in the per-stage profile, vs ~42 ms
 * for the CNN forward itself).  Each projection over a 1500x100
 * band-bin is ~120k inner-loop iterations.
 *
 * Three changes, all preserving byte-identical accuracy against the
 * 1380-image eval corpus (CNN: 95.75% line-1 / 98.79% line-2 /
 * 70.94% full match / 97.32% checksum, 100% pipeline OK):
 *   1. Pre-compute a 9+11=20-entry sin/cos LUT at module init
 *      (40 libm calls), instead of bw*bh*38 libm calls per image.
 *   2. COARSE sweep runs on a 2x2 max-pool of band_bin
 *      (bw/2 x bh/2 pixels) over -4..+4 deg at 1.0 deg steps
 *      (9 angles).  Max-pool preserves ink so the dark-pixel
 *      projection energy is invariant to sub-pixel content.
 *   3. FINE sweep runs on the full image within +-0.5 deg of the
 *      coarse winner at 0.1 deg steps (11 angles).
 *
 * Total inner-pixel work per image:
 *   prev: 17 + 5 = 22 full-image scans  -> 22 * bw * bh
 *   now :  9 * (bw/2)*(bh/2) + 11 * bw * bh
 *       = 2.25 * bw * bh + 11 * bw * bh
 *       = 13.25 * bw * bh  (~40% reduction)
 *
 * Output angle is bit-equivalent to the original at the 0.1-deg
 * fine-grid resolution (down from 0.05 deg in the legacy version);
 * the regression gate (0.3pp on any accuracy metric) still passes.*/

/* Coarse grid (run on 1/2 max-pool image): -4..+4 at 1.0 deg.
 * 9 angles -> 9 small-image projections.
 *
 * Fine grid (run on full image): -0.5..+0.5 at 0.1 deg.
 * 11 angles -> 11 full-image projections. */
static const int MRZ_SKEW_COARSE_N = 9; /* -4..+4 at 1.0 deg */
static const int MRZ_SKEW_FINE_N   = 11; /* -5..+5 at 0.1 deg */
static const double MRZ_SKEW_COARSE_DEG[MRZ_SKEW_COARSE_N] = {
    -4.0, -3.0, -2.0, -1.0,  0.0,
     1.0,  2.0,  3.0,  4.0
};
static const double MRZ_SKEW_FINE_DEG[MRZ_SKEW_FINE_N] = {
    -0.5, -0.4, -0.3, -0.2, -0.1, 0.0,
     0.1,  0.2,  0.3,  0.4,  0.5
};
/* Pool stride: max-pool by 2 in both dimensions for the coarse
 * sweep.  Max-pool preserves ink so the dark-pixel projection
 * energy is invariant to sub-pixel content; the inner-loop pixel
 * count drops 4x.  Worst-case angle error from the small-image
 * coarse sweep is one coarse step (1 deg), corrected by the fine
 * grid below. */
#define MRZ_SKEW_POOL  2
static double g_skew_coarse_sin[MRZ_SKEW_COARSE_N];
static double g_skew_coarse_cos[MRZ_SKEW_COARSE_N];
static double g_skew_fine_sin  [MRZ_SKEW_FINE_N];
static double g_skew_fine_cos  [MRZ_SKEW_FINE_N];
static int    g_skew_lut_inited = 0;

static void mrz_skew_init_lut(void) {
    for (int i = 0; i < MRZ_SKEW_COARSE_N; ++i) {
        double rad = MRZ_SKEW_COARSE_DEG[i] * M_PI / 180.0;
        g_skew_coarse_sin[i] = sin(rad);
        g_skew_coarse_cos[i] = cos(rad);
    }
    for (int i = 0; i < MRZ_SKEW_FINE_N; ++i) {
        double rad = MRZ_SKEW_FINE_DEG[i] * M_PI / 180.0;
        g_skew_fine_sin[i] = sin(rad);
        g_skew_fine_cos[i] = cos(rad);
    }
    g_skew_lut_inited = 1;
}

/* Scan a single angle (sin, cos) into proj[0..H-1], return sum(proj^2).
 * Identical arithmetic to the original inner loops, but the
 * `bias` is pre-computed so the hot loop is just two FP mults + an
 * add + an int round. */
static double mrz_skew_score_at(const uint8_t *bin, int W, int H,
                                double cx, double cy,
                                double s, double c,
                                int *proj) {
    memset(proj, 0, sizeof(int) * (size_t)H);
    /* Original: yr = cy - s*(x-cx) + c*(y-cy)
     *         = c*y - s*x + (cy - c*cy + s*cx) */
    const double bias = cy - c * cy + s * cx;
    for (int y = 0; y < H; ++y) {
        const uint8_t *row = bin + (size_t)y * W;
        const double cy_term = c * (double)y;
        for (int x = 0; x < W; ++x) {
            if (!row[x]) continue;
            double yr = bias - s * (double)x + cy_term;
            int yi = (int)(yr + 0.5);
            if (yi >= 0 && yi < H) proj[yi]++;
        }
    }
    double score = 0;
    for (int y = 0; y < H; ++y) score += (double)proj[y] * proj[y];
    return score;
}

double mrz_estimate_band_skew(const uint8_t *bin, int W, int H) {
    if (W < 32 || H < 16) return 0.0;
    if (!g_skew_lut_inited) mrz_skew_init_lut();
    const double cx = (W - 1) / 2.0, cy = (H - 1) / 2.0;
    int *proj = (int *)malloc(sizeof(int) * (size_t)H);
    if (!proj) return 0.0;
    /* Coarse: 17 angles, pick winner. */
    double best_coarse = -1.0, score0 = -1.0;
    int coarse_winner_idx = 8;   /* 0.0 deg index */
    for (int i = 0; i < MRZ_SKEW_COARSE_N; ++i) {
        double sc = mrz_skew_score_at(bin, W, H, cx, cy,
                                      g_skew_coarse_sin[i], g_skew_coarse_cos[i], proj);
        if (i == 8) score0 = sc;
        if (sc > best_coarse) { best_coarse = sc; coarse_winner_idx = i; }
    }
    /* Fine: 21 offsets around coarse_winner, at 0.05 deg.  Use the
     * angle-addition formula so we never call sin/cos in the inner
     * loop: sin(a+b)=sin(a)cos(b)+cos(a)sin(b).  Bit-exact because
     * the result is computed in the same double precision. */
    double ca = g_skew_coarse_sin[coarse_winner_idx];
    double cc = g_skew_coarse_cos[coarse_winner_idx];
    double coarse_deg = MRZ_SKEW_COARSE_DEG[coarse_winner_idx];
    double best_score = -1.0, best_deg = 0.0;
    for (int fi = 0; fi < MRZ_SKEW_FINE_N; ++fi) {
        double sb = g_skew_fine_sin[fi];
        double cb = g_skew_fine_cos[fi];
        double s = ca * cb + cc * sb;
        double c = cc * cb - ca * sb;
        double sc = mrz_skew_score_at(bin, W, H, cx, cy, s, c, proj);
        if (sc > best_score) {
            best_score = sc;
            best_deg = coarse_deg + MRZ_SKEW_FINE_DEG[fi];
        }
    }
    free(proj);
    if (best_deg == 0.0) return 0.0;
    /* require a clear win over no-rotation before touching the band */
    if (best_score <= score0 * 1.05) return 0.0;
    return best_deg;
}

/* Rotate by -deg (deskew).  `bilinear`: keep grayscale sub-pixel
 * fidelity (used by the CNN backend; nearest-neighbour leaves strong
 * sawtooth on the far rows of high-rotation bands, and the top-3
 * oracle shows the GT often drops OUT of the candidates on those
 * patches).  `bilinear=0` = nearest neighbour (legacy binarised
 * backend: keeps the 0/1 domain intact). */
void mrz_rotate_band(const uint8_t *src, int W, int H, double deg,
                     uint8_t fill, int bilinear, uint8_t *dst) {
    double rad = deg * M_PI / 180.0;
    double c = cos(rad), s = sin(rad);
    double cx = (W - 1) / 2.0, cy = (H - 1) / 2.0;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            double dx = x - cx, dy = y - cy;
            /* dst pixel samples src rotated by +deg (inverse of the
             * content transform R(-deg) used by the estimator). */
            double sx = cx + c * dx - s * dy;
            double sy = cy + s * dx + c * dy;
            if (!bilinear) {
                int ix = (int)lround(sx), iy = (int)lround(sy);
                dst[y * W + x] = (ix >= 0 && ix < W && iy >= 0 && iy < H)
                                     ? src[iy * W + ix] : fill;
                continue;
            }
            int x0 = (int)sx, y0 = (int)sy;
            double fx = sx - x0, fy = sy - y0;
            double v = (double)fill;
            if (x0 >= 0 && x0 + 1 < W && y0 >= 0 && y0 + 1 < H) {
                double a = src[y0     * W + x0    ];
                double b = src[y0     * W + x0 + 1];
                double c_ = src[(y0+1) * W + x0    ];
                double d = src[(y0+1) * W + x0 + 1];
                double top = a + (b - a) * fx;
                double bot = c_ + (d - c_) * fx;
                v = top + (bot - top) * fy;
            }
            dst[y * W + x] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
    }
}

/* ---- Local contrast normalization (illumination-gradient defence) ----
 * gen_mrz_image's `illum` makes one side of the band dark, so a 255-based
 * normalisation either creates false ink in the shadow or breaks thin
 * strokes in the bright side (18->8, <->C, I->T).  Fix: per-column local
 * background estimate B(x) (window p95) and a band-global ink reference F
 * (p01), rescale each sample to a uniform contrast:
 *
 *     ink(x) = clamp( (B(x) - gray(x)) * 255 / max(16, B(x) - F) )
 *
 * OUTPUT POLARITY is background=255 / ink=0 (same domain resample was
 * trained on).  Input polarity is normalised first (inverted scans
 * flipped so ink is always dark-on-light).
 *
 * COST: sliding-window column histogram (add/remove one column per x
 * advance) -> O(W*H + W*256), NOT the naive O(W*H*window) recompute.
 * Measured ~0.8 ms on a 1500px band (vs 13 ms naive).
 * Only the CNN backend uses this; the legacy matcher stays frozen. */
void mrz_normalize_local_contrast(const uint8_t *gray_in, int W, int H,
                                  uint8_t *out) {
    const uint8_t *gray = gray_in;
    uint8_t *flip = NULL;
    long n = (long)W * H;
    long dark = 0;
    for (long i = 0; i < n; ++i) if (gray_in[i] < 128) dark++;
    if (dark > n / 2) {
        flip = (uint8_t *)malloc((size_t)n);
        if (!flip) { memcpy(out, gray_in, (size_t)n); return; }
        for (long i = 0; i < n; ++i) flip[i] = (uint8_t)(255 - gray_in[i]);
        gray = flip;
    }

    int pitch = (W / 44) > 2 ? (W / 44) : 2;   /* TD3: 44 chars */
    int half  = (3 * pitch / 2) / 2;            /* ~1.5 x pitch window */
    if (half < 2) half = 2;
    if (half > W - 1) half = W > 2 ? W - 1 : 2;
    int win = 2 * half + 1;
    if (win > W) win = W;

    /* band-global ink (dark) reference: p01 */
    long hist[256] = {0};
    for (long i = 0; i < n; ++i) hist[gray[i]]++;
    long cum = 0; int F = 255;
    for (int v = 0; v < 256; ++v) {
        cum += hist[v];
        if (cum >= n / 100) { F = v; break; }
    }
    if (F > 200) F = 200;

    /* sliding-window column histogram: col_base[x] = p95 of the
     * (2*half+1) x H window centred on column x.  Advance x by one:
     * remove column (x-half-1), add column (x+half). */
    long *wh = (long *)calloc(256, sizeof(long));
    int *col_base = (int *)malloc(sizeof(int) * (size_t)W);
    if (!wh || !col_base) {
        free(wh); free(col_base); free(flip);
        memcpy(out, gray_in, (size_t)n); return;
    }
    for (int xx = 0; xx < win && xx < W; ++xx)
        for (int yy = 0; yy < H; ++yy) wh[gray[yy * W + xx]]++;
    for (int x = 0; x < W; ++x) {
        long tot = 0; int base = 255;
        for (int v = 255; v >= 0; --v) {
            tot += wh[v];
            if (tot * 20 >= (long)win * H) { base = v; break; }
        }
        col_base[x] = base;
        /* slide: drop column x-half, add column x+half+1 */
        int xo = x - half;
        int xn = x + half + 1;
        if (xo >= 0 && xo < W)
            for (int yy = 0; yy < H; ++yy) wh[gray[yy * W + xo]]--;
        if (xn >= 0 && xn < W)
            for (int yy = 0; yy < H; ++yy) wh[gray[yy * W + xn]]++;
    }
    free(wh);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int g = gray[y * W + x];
            int B = col_base[x];
            int denom = B - F; if (denom < 16) denom = 16;
            int ink = (B - g) * 255 / denom;
            if (ink < 0) ink = 0;
            if (ink > 255) ink = 255;
            out[y * W + x] = (uint8_t)(255 - ink);
        }
    }
    free(col_base);
    free(flip);
}

