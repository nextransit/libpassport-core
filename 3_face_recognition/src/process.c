/* Grayscale conversion, resize, histogram, aHash, pHash. */
#include "face.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int face_to_grayscale(const face_image_t *in, uint8_t *out) {
    if (!in || !out) return -1;
    int n = in->width * in->height;
    for (int i = 0; i < n; ++i) {
        /* BT.601 luma */
        int y = (299 * in->pixels[i*3 + 0] +
                 587 * in->pixels[i*3 + 1] +
                 114 * in->pixels[i*3 + 2]) / 1000;
        if (y < 0) y = 0; else if (y > 255) y = 255;
        out[i] = (uint8_t)y;
    }
    return 0;
}

int face_histogram_equalise(uint8_t *gray, int w, int h) {
    if (!gray || w <= 0 || h <= 0) return -1;
    size_t n = (size_t)w * h;
    /* Build histogram. */
    size_t hist[256] = {0};
    for (size_t i = 0; i < n; ++i) hist[gray[i]]++;
    /* Cumulative. */
    size_t cdf[256];
    cdf[0] = hist[0];
    for (int i = 1; i < 256; ++i) cdf[i] = cdf[i-1] + hist[i];
    /* Skip empty leading bins. */
    size_t cdf_min = 0;
    for (int i = 0; i < 256; ++i) if (cdf[i] != 0) { cdf_min = cdf[i]; break; }
    if (cdf_min == n) return 0; /* all same -> identity */
    /* Map. */
    for (size_t i = 0; i < n; ++i) {
        size_t v = cdf[gray[i]] - cdf_min;
        v = (v * 255) / (n - cdf_min);
        gray[i] = (uint8_t)(v & 0xff);
    }
    return 0;
}

int face_resize_nn(const uint8_t *in, int iw, int ih, int ow, int oh, uint8_t *out) {
    if (!in || !out || iw <= 0 || ih <= 0 || ow <= 0 || oh <= 0) return -1;
    for (int y = 0; y < oh; ++y) {
        int sy = y * ih / oh;
        for (int x = 0; x < ow; ++x) {
            int sx = x * iw / ow;
            out[y * ow + x] = in[sy * iw + sx];
        }
    }
    return 0;
}

/* aHash: resize to 8x8 grayscale, threshold against mean. */
int face_ahash(const uint8_t *gray, int w, int h, uint64_t *out_hash) {
    if (!gray || !out_hash) return -1;
    uint8_t small[64];
    if (face_resize_nn(gray, w, h, 8, 8, small) != 0) return -1;
    int sum = 0;
    for (int i = 0; i < 64; ++i) sum += small[i];
    int mean = sum / 64;
    uint64_t h64 = 0;
    for (int i = 0; i < 64; ++i) {
        if (small[i] >= mean) h64 |= ((uint64_t)1) << i;
    }
    *out_hash = h64;
    return 0;
}

/* 8x8 DCT-II (AAN-style would be overkill for perceptual hashing). */
static void dct8x8(const double in[64], double out[64]) {
    const double PI = 3.14159265358979323846;
    for (int v = 0; v < 8; ++v) {
        for (int u = 0; u < 8; ++u) {
            double s = 0;
            for (int y = 0; y < 8; ++y)
                for (int x = 0; x < 8; ++x)
                    s += in[y*8 + x] *
                         cos((2*x + 1) * u * PI / 16.0) *
                         cos((2*y + 1) * v * PI / 16.0);
            double cu = (u == 0) ? 1.0 / sqrt(2.0) : 1.0;
            double cv = (v == 0) ? 1.0 / sqrt(2.0) : 1.0;
            out[v*8 + u] = 0.25 * cu * cv * s;
        }
    }
}

int face_phash(const uint8_t *gray, int w, int h, uint64_t *out_hash) {
    if (!gray || !out_hash) return -1;
    uint8_t small[1024];  // 32*32 block for resize
    if (face_resize_nn(gray, w, h, 32, 32, small) != 0) return -1;
    /* Reduce to 8x8 by averaging 4x4 blocks. */
    double reduced[64] = {0};
    for (int by = 0; by < 8; ++by)
        for (int bx = 0; bx < 8; ++bx) {
            double s = 0;
            for (int y = 0; y < 4; ++y)
                for (int x = 0; x < 4; ++x)
                    s += small[(by*4 + y) * 32 + (bx*4 + x)];
            reduced[by*8 + bx] = s / 16.0;
        }
    double dct[64];
    dct8x8(reduced, dct);
    /* Skip DC component (lowest frequency) and threshold against the
     * median of the remaining 63 coefficients. */
    double coeffs[63];
    int k = 0;
    for (int i = 1; i < 64; ++i) coeffs[k++] = dct[i];
    /* Selection-sort median. */
    for (int i = 0; i < k/2; ++i) {
        int min = i;
        for (int j = i + 1; j < k; ++j)
            if (coeffs[j] < coeffs[min]) min = j;
        double t = coeffs[i]; coeffs[i] = coeffs[min]; coeffs[min] = t;
    }
    double median = coeffs[k/2];
    uint64_t h64 = 0;
    for (int i = 1; i < 64; ++i) {
        if (dct[i] > median) h64 |= ((uint64_t)1) << (i - 1);
    }
    *out_hash = h64;
    return 0;
}

int face_hamming(uint64_t a, uint64_t b) {
    uint64_t x = a ^ b;
    /* Brian Kernighan's popcount. */
    int c = 0;
    while (x) { x &= x - 1; ++c; }
    return c;
}

int face_compare(const face_image_t *a, const face_image_t *b, int method) {
    if (!a || !b || a->width <= 0 || b->width <= 0) return -1;
    size_t sa = (size_t)a->width * a->height;
    size_t sb = (size_t)b->width * b->height;
    uint8_t *ga = (uint8_t *)malloc(sa);
    uint8_t *gb = (uint8_t *)malloc(sb);
    if (!ga || !gb) { free(ga); free(gb); return -1; }
    if (face_to_grayscale(a, ga) != 0) { free(ga); free(gb); return -1; }
    if (face_to_grayscale(b, gb) != 0) { free(ga); free(gb); return -1; }
    face_histogram_equalise(ga, a->width, a->height);
    face_histogram_equalise(gb, b->width, b->height);
    int score = -1;
    if (method == 0) {
        uint64_t ha, hb;
        face_ahash(ga, a->width, a->height, &ha);
        face_ahash(gb, b->width, b->height, &hb);
        int d = face_hamming(ha, hb);
        score = 100 - (d * 100 / 64);
    } else if (method == 1) {
        uint64_t ha, hb;
        face_phash(ga, a->width, a->height, &ha);
        face_phash(gb, b->width, b->height, &hb);
        int d = face_hamming(ha, hb);
        score = 100 - (d * 100 / 64);
    } else {
        uint64_t a1, b1, a2, b2;
        face_ahash(ga, a->width, a->height, &a1);
        face_ahash(gb, b->width, b->height, &b1);
        face_phash(ga, a->width, a->height, &a2);
        face_phash(gb, b->width, b->height, &b2);
        int d = face_hamming(a1, b1) + face_hamming(a2, b2);
        score = 100 - (d * 100 / 128);
    }
    free(ga); free(gb);
    return score;
}
