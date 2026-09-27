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

double mrz_estimate_band_skew(const uint8_t *bin, int W, int H) {
    if (W < 32 || H < 16) return 0.0;
    double cx = (W - 1) / 2.0, cy = (H - 1) / 2.0;
    double best_score = -1.0, score0 = -1.0, best_deg = 0.0;
    int *proj = (int *)malloc(sizeof(int) * (size_t)(H + 4));
    if (!proj) return 0.0;
    /* Coarse scan -4..+4 deg at 0.5 deg steps, then a fine scan
     * +-0.5 deg around the coarse winner at 0.05 deg. The coarse
     * grid alone leaves up to 0.25 deg of residual tilt, which over
     * a 1500px line is ~6.5px of vertical wander and breaks the
     * row-projection uniformity the pitch-grid segmenter relies on. */
    double deg_coarse = 0.0, best_coarse = -1.0;
    for (int deg4 = -8; deg4 <= 8; ++deg4) {
        double deg = deg4 * 0.5;
        double rad = deg * M_PI / 180.0;
        double s = sin(rad), c = cos(rad);
        memset(proj, 0, sizeof(int) * (size_t)(H + 4));
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                if (!bin[y * W + x]) continue;
                double yr = cy - s * (x - cx) + c * (y - cy);
                int yi = (int)(yr + 0.5);
                if (yi >= 0 && yi < H) proj[yi]++;
            }
        }
        double score = 0;
        for (int y = 0; y < H; ++y) score += (double)proj[y] * proj[y];
        if (deg == 0.0) score0 = score;
        if (score > best_coarse) { best_coarse = score; deg_coarse = deg; }
    }
    for (int fi = -10; fi <= 10; ++fi) {
        double deg = deg_coarse + fi * 0.05;
        double rad = deg * M_PI / 180.0;
        double s = sin(rad), c = cos(rad);
        memset(proj, 0, sizeof(int) * (size_t)(H + 4));
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                if (!bin[y * W + x]) continue;
                double yr = cy - s * (x - cx) + c * (y - cy);
                int yi = (int)(yr + 0.5);
                if (yi >= 0 && yi < H) proj[yi]++;
            }
        }
        double score = 0;
        for (int y = 0; y < H; ++y) score += (double)proj[y] * proj[y];
        if (score > best_score) { best_score = score; best_deg = deg; }
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

