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
        if (score > best_score) { best_score = score; best_deg = deg; }
    }
    free(proj);
    if (best_deg == 0.0) return 0.0;
    /* require a clear win over no-rotation before touching the band */
    if (best_score <= score0 * 1.05) return 0.0;
    return best_deg;
}

void mrz_rotate_band(const uint8_t *src, int W, int H, double deg,
                     uint8_t fill, uint8_t *dst) {
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
            int ix = (int)lround(sx), iy = (int)lround(sy);
            dst[y * W + x] = (ix >= 0 && ix < W && iy >= 0 && iy < H)
                                 ? src[iy * W + ix] : fill;
        }
    }
}
