/* gen_mrz_image -- render a synthetic MRZ band into a PPM image.
 *
 * Usage:
 *   gen_mrz_image <out.ppm> <line1_44chars> <line2_44chars> [opts]
 *
 * Options (space-separated, key=value):
 *   scale=<int>           glyph scale factor (default 4)
 *   hgap=<int>            horizontal gap between glyphs (default 2)
 *   vgap=<int>            vertical gap between two lines (default 6)
 *   margin=<int>          outer margin (default 12)
 *   noise=<float>         pixel noise probability 0..1 (default 0)
 *   noise-seed=<int>      deterministic noise seed. When given, the
 *                         noise is reproducible for a given corpus
 *                         record (required for regression baselines).
 *                         When omitted, falls back to time^pid.
 *   skew=<int>            tilt the band by -N..N pixels across the row
 *                         (default 0)
 *   rot=<float>           rotate the whole band by degrees (real
 *                         scanner tilt, default 0; range +-8)
 *   illum=<int>           illumination-gradient amplitude 0..200:
 *                         background dims linearly across the image
 *                         (scan shading, default 0)
 *   blur=<0|1|2>          3x3 binomial blur passes (simulates defocus
 *                         / anti-aliased strokes, default 0)
 *   invert=<0|1>          render ink=white-on-black instead of
 *                         black-on-white (default 0)
 *
 * The output image is grayscale: white background (255) and black ink
 * (0), unless `invert=1` is passed.
 *
 * The output dimension is computed from the parameters; we don't try
 * to embed the band into a full passport page -- the OCR pipeline
 * expects the band to occupy a significant fraction of the height.
 */
#include "face.h"
#define _DEFAULT_SOURCE
#include "template.h"

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

static int get_int(const char *name, const char *kv) {
    if (strncmp(kv, name, strlen(name)) != 0) return -1;
    return atoi(kv + strlen(name) + 1);
}
static float get_float(const char *name, const char *kv) {
    if (strncmp(kv, name, strlen(name)) != 0) return -1;
    return (float)atof(kv + strlen(name) + 1);
}

/* 3x3 binomial blur on a single-channel buffer (weights 1-2-1). */
static void blur3(uint8_t *g, int W, int H) {
    uint8_t *out = (uint8_t *)malloc((size_t)W * H);
    if (!out) return;
    static const int w[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            int acc = 0, wsum = 0;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    int yy = y + dy, xx = x + dx;
                    if (yy < 0 || yy >= H || xx < 0 || xx >= W) continue;
                    acc += g[yy * W + xx] * w[dy + 1][dx + 1];
                    wsum += w[dy + 1][dx + 1];
                }
            out[y * W + x] = (uint8_t)(acc / wsum);
        }
    memcpy(g, out, (size_t)W * H);
    free(out);
}

/* Rotate a single-channel buffer by deg about its center
 * (inverse-map nearest neighbour, background fill). */
static void rotate_gray(uint8_t *g, int W, int H, double deg, uint8_t bg) {
    double rad = deg * M_PI / 180.0;
    double c = cos(rad), s = sin(rad);
    double cx = (W - 1) / 2.0, cy = (H - 1) / 2.0;
    uint8_t *out = (uint8_t *)malloc((size_t)W * H);
    if (!out) return;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            double dx = x - cx, dy = y - cy;
            double sx = cx + c * dx + s * dy;
            double sy = cy - s * dx + c * dy;
            int ix = (int)lround(sx), iy = (int)lround(sy);
            out[y * W + x] = (ix >= 0 && ix < W && iy >= 0 && iy < H)
                                 ? g[iy * W + ix] : bg;
        }
    memcpy(g, out, (size_t)W * H);
    free(out);
}

static void put_ink(uint8_t *img, int W, int H, int x, int y,
                    int invert, uint8_t v) {
    if (x < 0 || y < 0 || x >= W || y >= H) return;
    img[y * W + x] = invert ? (uint8_t)(255 - v) : v;
}

static void draw_glyph(uint8_t *img, int W, int H, int ox, int oy,
                       char c, int scale, int invert) {
    int gi = mrz_ocr_glyph_index(c);
    if (gi < 0) gi = mrz_ocr_glyph_index('<');
    const mrz_ocr_glyph_t *g = &MRZ_OCR_GLYPHS[gi];
    for (int y = 0; y < MRZ_OCR_GLYPH_H; ++y)
        for (int x = 0; x < MRZ_OCR_GLYPH_W; ++x) {
            int ink = (g->rows[y] >> (7 - x)) & 1;
            if (!ink) continue;
            for (int dy = 0; dy < scale; ++dy)
                for (int dx = 0; dx < scale; ++dx)
                    put_ink(img, W, H,
                            ox + x * scale + dx,
                            oy + y * scale + dy,
                            invert, 0);
        }
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <out.ppm> <line1> <line2> [opts...]\n",
                argv[0]);
        return 2;
    }
    const char *out = argv[1];
    const char *l1 = argv[2];
    const char *l2 = argv[3];
    int scale = 4, hgap = 2, vgap = 6, margin = 12, skew = 0, invert = 0;
    int noise_seed = -1, illum = 0, blur = 0;
    float noise = 0, rot = 0;
    for (int i = 4; i < argc; ++i) {
        int v;
        float f;
        /* NOTE: every branch must first match its KEY. A bare
         * "get_int(name, kv) >= lo && <= hi" range test also matches
         * the -1 sentinel of ANY other key and swallows the rest of
         * the chain (the historical bug that silently disabled
         * noise/invert/... in every corpus rendered before 2026-09). */
        if (strncmp(argv[i], "scale", 5) == 0 && (v = get_int("scale", argv[i])) >= 0) scale = v;
        else if (strncmp(argv[i], "hgap", 4) == 0 && (v = get_int("hgap", argv[i])) >= 0) hgap = v;
        else if (strncmp(argv[i], "vgap", 4) == 0 && (v = get_int("vgap", argv[i])) >= 0) vgap = v;
        else if (strncmp(argv[i], "margin", 6) == 0 && (v = get_int("margin", argv[i])) >= 0) margin = v;
        else if (strncmp(argv[i], "skew", 4) == 0 && (v = get_int("skew", argv[i])) >= -1000 && v <= 1000) skew = v;
        else if (strncmp(argv[i], "invert", 6) == 0 && (v = get_int("invert", argv[i])) >= 0) invert = v;
        else if (strncmp(argv[i], "noise-seed", 10) == 0 && (v = get_int("noise-seed", argv[i])) >= 0) noise_seed = v;
        else if (strncmp(argv[i], "illum", 5) == 0 && (v = get_int("illum", argv[i])) >= 0) illum = v;
        else if (strncmp(argv[i], "blur", 4) == 0 && (v = get_int("blur", argv[i])) >= 0) blur = v;
        else if (strncmp(argv[i], "noise", 5) == 0 && (f = get_float("noise", argv[i])) >= 0) noise = f;
        else if (strncmp(argv[i], "rot", 3) == 0 && (f = get_float("rot", argv[i])) >= -8.0f && f <= 8.0f) rot = f;
    }
    int n1 = (int)strlen(l1);
    int n2 = (int)strlen(l2);
    int n = n1 > n2 ? n1 : n2;
    int glyph_w = MRZ_OCR_GLYPH_W * scale;
    int glyph_h = MRZ_OCR_GLYPH_H * scale;
    int line_w = n * glyph_w + (n - 1) * hgap;
    int band_w = line_w + margin * 2;
    int line_h = glyph_h;
    int band_h = line_h * 2 + vgap + margin * 2;
    if (skew < 0) skew = -skew;

    face_image_t *img = face_image_new(band_w, band_h);
    if (!img) return 2;
    /* Build a 1-channel scratch buffer of the band so we can add noise
     * to the ink properly. Background carries an illumination gradient
     * (scan shading): brightest at top-left, dims by `illum` toward the
     * bottom-right corner. */
    uint8_t *gray = (uint8_t *)malloc((size_t)band_w * band_h);
    if (!gray) { face_image_free(img); return 2; }
    uint8_t bg = invert ? 0 : 255;
    for (int y = 0; y < band_h; ++y)
        for (int x = 0; x < band_w; ++x) {
            float t = 0.5f * (float)x / (float)band_w
                    + 0.5f * (float)y / (float)band_h;
            int v = (int)((float)bg - (float)illum * t);
            if (v < 0) v = 0;
            if (v > 255) v = 255;
            gray[y * band_w + x] = (uint8_t)v;
        }

    for (int li = 0; li < 2; ++li) {
        const char *line = (li == 0) ? l1 : l2;
        int len = (int)strlen(line);
        int row_y = margin + li * (line_h + vgap);
        int row_x_start = margin;
        for (int i = 0; i < len; ++i) {
            int skew_dx = skew * i / len;
            int x = row_x_start + i * (glyph_w + hgap) + skew_dx;
            draw_glyph(gray, band_w, band_h, x, row_y, line[i], scale, invert);
        }
    }
    /* Defocus / anti-aliased strokes. */
    for (int p = 0; p < blur; ++p) blur3(gray, band_w, band_h);
    /* Real scanner tilt. */
    if (rot != 0.0f) rotate_gray(gray, band_w, band_h, (double)rot, bg);
    /* Add noise (deterministic when noise-seed is given — required for
     * reproducible corpora and regression baselines). */
    if (noise > 0) {
        if (noise_seed >= 0)
            srand((unsigned)noise_seed);
        else
            srand((unsigned)(time(NULL) ^ getpid()));
        for (int i = 0; i < band_w * band_h; ++i) {
            float r = (float)rand() / (float)RAND_MAX;
            if (r < noise) {
                /* Flip with 50% probability. */
                int flip = (rand() & 1);
                gray[i] = flip ? 0 : 255;
            }
        }
    }
    /* Convert grayscale back to RGB. */
    for (int i = 0; i < band_w * band_h; ++i) {
        img->pixels[i*3 + 0] = gray[i];
        img->pixels[i*3 + 1] = gray[i];
        img->pixels[i*3 + 2] = gray[i];
    }
    free(gray);
    if (face_image_save_ppm(img, out) != 0) {
        face_image_free(img);
        return 2;
    }
    face_image_free(img);
    fprintf(stderr, "wrote %s (%dx%d, scale=%d)\n", out, band_w, band_h, scale);
    return 0;
}
