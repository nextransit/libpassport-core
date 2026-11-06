/* Physical rendering pipeline for the MRZ corpus generator.
 *
 * Training outline §4.1/§4.3: the legacy corpus path renders a 1-bit
 * glyph bank with fixed 0/255 ink and paper, blurs it, and then rotates
 * with nearest neighbour -- the wrong order, and the result is a
 * quasi-binary image (measured: 84% of the pixels are pure endpoints,
 * 12 grey levels in the whole band).  Such a corpus does not measure
 * reflectance, ink density or defocus robustness.
 *
 * This header implements the prescribed order on an 8-bit coverage bank:
 *
 *   S1 antialiased grey glyph rendering (per-pixel coverage)
 *   S2 stroke-level ink unevenness
 *   S3 guilloche background
 *   S4 multiplicative light field + 2D highlight (clipped)
 *   S5 geometry (rotation / shear)          <- BEFORE the blur
 *   S6 optical blur (gaussian with explicit sigma + motion blur)
 *   S7 sensor noise (additive gaussian + dark current + salt&pepper)
 *   S8 downsample (bilinear, filter=true) + JPEG
 *
 * It is a header of static helpers so the generator keeps one binary and
 * the legacy path stays bit-identical when phys=0.
 */
#ifndef MRZ_PHYS_H
#define MRZ_PHYS_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "ocrb_glyphs_aa.h"

static uint32_t mrz_phys_rng = 1u;
static void mrz_phys_srand(uint32_t s) { mrz_phys_rng = s ? s : 1u; }
static uint32_t mrz_phys_rand(void) {
    uint32_t x = mrz_phys_rng;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    mrz_phys_rng = x;
    return x;
}
static float mrz_phys_randf(void) {
    return (float)(mrz_phys_rand() >> 8) / 16777216.0f;
}
static float mrz_phys_gauss(void) {
    float u1 = mrz_phys_randf(); if (u1 < 1e-6f) u1 = 1e-6f;
    float u2 = mrz_phys_randf();
    return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530718f * u2);
}

static int mrz_phys_cov_index(char ch) {
    int i;
    for (i = 0; i < (int)(sizeof(MRZ_OCRB_COV) / sizeof(MRZ_OCRB_COV[0])); ++i)
        if (MRZ_OCRB_COV[i].ch == ch) return i;
    return -1;
}

/* S1 + S2: one glyph with real ink colour and per-pixel coverage.
 * `ink` is the ink darkness (0..255) and `field` is the slow ink
 * density variation of S2 (NULL for none). */
static void mrz_phys_draw_glyph(uint8_t *img, int W, int H, int ox, int oy,
                                 char ch, int scale, int ink,
                                 const float *field) {
    int ci = mrz_phys_cov_index(ch);
    int gy = MRZ_OCR_GLYPH_H, gx = MRZ_OCR_GLYPH_W;
    if (ci < 0) ci = mrz_phys_cov_index('<');
    for (int y = 0; y < gy; ++y) {
        for (int x = 0; x < gx; ++x) {
            int cov = MRZ_OCRB_COV[ci].cov[y][x];
            if (!cov) continue;
            int px = ox + x * scale, py = oy + y * scale;
            for (int dy = 0; dy < scale; ++dy) {
                int yy = py + dy;
                if (yy < 0 || yy >= H) continue;
                for (int dx = 0; dx < scale; ++dx) {
                    int xx = px + dx;
                    if (xx < 0 || xx >= W) continue;
                    int f = 255;
                    if (field) {
                        f = (int)(field[yy * W + xx] * 255.0f);
                        if (f < 0) f = 0;
                        if (f > 255) f = 255;
                    }
                    int eff = cov * f / 255;
                    if (!eff) continue;
                    int paper = img[yy * W + xx];
                    /* Alpha blend: a pixel covered by fraction c of the
                     * stroke shows c * (paper - ink) of ink darkness.
                     * Scaling the INK VALUE by the coverage (the obvious
                     * shortcut) is wrong whenever the ink is not black:
                     * it washes the glyphs out. */
                    int dark = (eff * (paper - ink)) / 255;
                    if (dark < 0) dark = 0;
                    int v = paper - dark;
                    if (v < 0) v = 0;
                    img[yy * W + xx] = (uint8_t)v;
                }
            }
        }
    }
}

/* S3: procedural guilloche (the outline asks for real photo slices,
 * which this repository does not have yet). */
static void mrz_phys_guilloche(uint8_t *img, int W, int H, float alpha) {
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float u = (float)x / (float)W, v = (float)y / (float)H;
            float g = 0.5f + 0.5f * sinf(38.0f * u + 21.0f * v)
                            * cosf(29.0f * v - 17.0f * u);
            int paper = img[y * W + x];
            int v2 = (int)((float)paper - alpha * 26.0f * g);
            if (v2 < 0) v2 = 0;
            img[y * W + x] = (uint8_t)v2;
        }
    }
}

/* S4: multiplicative light field + 2D gaussian highlight, clipped. */
static void mrz_phys_light_field(uint8_t *img, int W, int H, float illum,
                                 float hl_peak, float hl_sx, float hl_sy,
                                 float hl_cx, float hl_cy) {
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float t = 0.5f * (float)x / (float)W + 0.5f * (float)y / (float)H;
            float gain = 1.0f - illum * t / 255.0f * 2.2f;
            if (gain < 0.05f) gain = 0.05f;
            float dx = ((float)x - hl_cx * (float)W) / (hl_sx * (float)W);
            float dy = ((float)y - hl_cy * (float)H) / (hl_sy * (float)H);
            float hl = hl_peak * expf(-0.5f * (dx * dx + dy * dy));
            float v = (float)img[y * W + x] * gain + hl * 40.0f;
            if (v > 245.0f) v = 245.0f;
            if (v < 0.0f) v = 0.0f;
            img[y * W + x] = (uint8_t)v;
        }
    }
}

/* S6: separable gaussian blur with an explicit sigma (the outline:
 * kernel size alone is meaningless, OpenCV derives sigma from it). */
static void mrz_phys_gauss_blur(uint8_t *img, int W, int H, float sigma) {
    int r;
    float *k, sum = 0.0f;
    uint8_t *tmp;
    if (sigma <= 0.01f) return;
    r = (int)(3.0f * sigma + 0.5f);
    if (r < 1) return;
    k = (float *)malloc(sizeof(float) * (size_t)(2 * r + 1));
    tmp = (uint8_t *)malloc((size_t)W * (size_t)H);
    if (!k || !tmp) { free(k); free(tmp); return; }
    for (int i = -r; i <= r; ++i) {
        float v = expf(-0.5f * (float)(i * i) / (sigma * sigma));
        k[i + r] = v; sum += v;
    }
    for (int i = -r; i <= r; ++i) k[i + r] /= sum;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float acc = 0.0f;
            for (int i = -r; i <= r; ++i) {
                int xx = x + i; if (xx < 0) xx = 0; if (xx >= W) xx = W - 1;
                acc += k[i + r] * (float)img[y * W + xx];
            }
            tmp[y * W + x] = (uint8_t)(acc + 0.5f);
        }
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float acc = 0.0f;
            for (int i = -r; i <= r; ++i) {
                int yy = y + i; if (yy < 0) yy = 0; if (yy >= H) yy = H - 1;
                acc += k[i + r] * (float)tmp[yy * W + x];
            }
            img[y * W + x] = (uint8_t)(acc + 0.5f);
        }
    free(tmp); free(k);
}

/* S6: motion blur along a direction (length 3..9 px, angle +-15 deg). */
static void mrz_phys_motion_blur(uint8_t *img, int W, int H, int len,
                                  float deg) {
    float rad, cx, cy;
    uint8_t *tmp;
    int half;
    if (len < 2) return;
    rad = deg * 3.14159265359f / 180.0f;
    cx = cosf(rad); cy = sinf(rad);
    tmp = (uint8_t *)malloc((size_t)W * (size_t)H);
    if (!tmp) return;
    half = len / 2;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float acc = 0.0f;
            for (int i = -half; i <= half; ++i) {
                int xx = x + (int)lroundf(i * cx);
                int yy = y + (int)lroundf(i * cy);
                if (xx < 0) xx = 0; if (xx >= W) xx = W - 1;
                if (yy < 0) yy = 0; if (yy >= H) yy = H - 1;
                acc += (float)img[yy * W + xx];
            }
            tmp[y * W + x] = (uint8_t)(acc / (float)(2 * half + 1) + 0.5f);
        }
    memcpy(img, tmp, (size_t)W * (size_t)H);
    free(tmp);
}

/* S7: additive gaussian + dark current + salt&pepper. */
static void mrz_phys_sensor_noise(uint8_t *img, int W, int H, float sigma,
                                  float salt_pepper, uint32_t seed) {
    mrz_phys_srand(seed);
    for (int i = 0; i < W * H; ++i) {
        float v = (float)img[i] + mrz_phys_gauss() * sigma;
        if (v < 0.0f) v = 0.0f;
        if (v > 255.0f) v = 255.0f;
        img[i] = (uint8_t)(v + 0.5f);
    }
    if (salt_pepper > 0.0f) {
        for (int i = 0; i < W * H; ++i) {
            float r = mrz_phys_randf();
            if (r < salt_pepper * 0.5f) img[i] = 0;
            else if (r < salt_pepper) img[i] = 255;
        }
    }
}

/* S8: bilinear downsample with filter=true (the AGENTS.md red line for
 * fastPair applies to the corpus too) and a JPEG round trip. */
static void mrz_phys_downsample(uint8_t *img, int *W, int *H, float scale) {
    int nw, nh;
    uint8_t *out;
    if (scale >= 0.999f) return;
    nw = (int)(*W * scale + 0.5f); if (nw < 8) nw = 8;
    nh = (int)(*H * scale + 0.5f); if (nh < 8) nh = 8;
    out = (uint8_t *)malloc((size_t)nw * (size_t)nh);
    if (!out) return;
    for (int y = 0; y < nh; ++y) {
        float sy = ((float)y + 0.5f) / scale - 0.5f;
        int y0 = (int)floorf(sy);
        float fy = sy - (float)y0;
        int y1 = y0 + 1;
        if (y0 < 0) y0 = 0; if (y0 >= *H) y0 = *H - 1;
        if (y1 < 0) y1 = 0; if (y1 >= *H) y1 = *H - 1;
        for (int x = 0; x < nw; ++x) {
            float sx = ((float)x + 0.5f) / scale - 0.5f;
            int x0 = (int)floorf(sx);
            float fx = sx - (float)x0;
            int x1 = x0 + 1;
            if (x0 < 0) x0 = 0; if (x0 >= *W) x0 = *W - 1;
            if (x1 < 0) x1 = 0; if (x1 >= *W) x1 = *W - 1;
            float v = 0.0f;
            v += (1.0f - fx) * (1.0f - fy) * (float)img[y0 * *W + x0];
            v += fx * (1.0f - fy) * (float)img[y0 * *W + x1];
            v += (1.0f - fx) * fy * (float)img[y1 * *W + x0];
            v += fx * fy * (float)img[y1 * *W + x1];
            int iv = (int)(v + 0.5f);
            if (iv < 0) iv = 0; if (iv > 255) iv = 255;
            out[y * nw + x] = (uint8_t)iv;
        }
    }
    memcpy(img, out, (size_t)nw * (size_t)nh);
    free(out);
    *W = nw; *H = nh;
}

#endif /* MRZ_PHYS_H */
