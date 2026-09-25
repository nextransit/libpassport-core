/* Generate a synthetic passport-style face image (no real biometrics).
 * The output is a deterministic PPM containing a circular face shape on
 * a soft background plus simple facial-feature markers. Two flavours:
 *   gen_sample <out.ppm> <seed> [w] [h]
 * Different seeds produce visibly different layouts so the matcher can
 * be exercised with same/different pairs.
 */
#include "face.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static uint32_t lcg(uint32_t *s) {
    *s = (*s) * 1664525u + 1013904223u;
    return *s;
}

static void fill_bg(face_image_t *img, uint8_t r, uint8_t g, uint8_t b) {
    size_t n = (size_t)img->width * img->height;
    for (size_t i = 0; i < n; ++i) {
        img->pixels[i*3 + 0] = r;
        img->pixels[i*3 + 1] = g;
        img->pixels[i*3 + 2] = b;
    }
}

static void putpixel(face_image_t *img, int x, int y, uint8_t r, uint8_t g, uint8_t b) {
    if (x < 0 || y < 0 || x >= img->width || y >= img->height) return;
    uint8_t *p = img->pixels + ((size_t)y * img->width + x) * 3;
    p[0] = r; p[1] = g; p[2] = b;
}

static void disc(face_image_t *img, int cx, int cy, int radius,
                 uint8_t r, uint8_t g, uint8_t b) {
    int r2 = radius * radius;
    for (int y = cy - radius; y <= cy + radius; ++y) {
        for (int x = cx - radius; x <= cx + radius; ++x) {
            int dx = x - cx, dy = y - cy;
            if (dx*dx + dy*dy <= r2) putpixel(img, x, y, r, g, b);
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 5) {
        fprintf(stderr, "usage: %s <out.ppm> <seed> [w] [h]\n", argv[0]);
        return 2;
    }
    int w = argc >= 4 ? atoi(argv[3]) : 128;
    int h = argc >= 5 ? atoi(argv[4]) : 128;
    uint32_t seed = (uint32_t)atoi(argv[2]);
    face_image_t *img = face_image_new(w, h);
    if (!img) return 2;
    /* gradient background: top light blue, bottom light gray */
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            uint8_t r = (uint8_t)(200 + y * 30 / h);
            uint8_t g = (uint8_t)(220 + y * 20 / h);
            uint8_t b = (uint8_t)(230);
            putpixel(img, x, y, r, g, b);
        }
    }
    /* face: skin-coloured disc, size depends on seed */
    int cx = w / 2 + (int)(lcg(&seed) % (w / 10)) - w/20;
    int cy = h / 2 + (int)(lcg(&seed) % (h / 10)) - h/20;
    int face_r = (w < h ? w : h) / 3;
    disc(img, cx, cy, face_r, 230, 200, 170);
    /* eyes */
    int ey = cy - face_r / 6;
    int eo = face_r / 3;
    disc(img, cx - eo, ey, face_r/10, 40, 40, 40);
    disc(img, cx + eo, ey, face_r/10, 40, 40, 40);
    /* mouth: small line */
    int my = cy + face_r / 3;
    for (int x = cx - face_r/3; x <= cx + face_r/3; ++x) {
        if (x % 2 == 0) putpixel(img, x, my, 120, 30, 30);
    }
    /* hairline arc */
    int hy = cy - face_r * 3 / 4;
    for (int x = cx - face_r; x <= cx + face_r; ++x) {
        int dx = x - cx;
        int dy2 = (face_r * face_r) - dx*dx;
        if (dy2 <= 0) continue;
        int dy = (int)sqrt((double)dy2);
        putpixel(img, x, cy - dy, 60, 40, 30);
        if (x % 2 == 0) putpixel(img, x, cy - dy + 1, 60, 40, 30);
        (void)hy;
    }
    if (face_image_save_ppm(img, argv[1]) != 0) {
        fprintf(stderr, "save failed\n");
        face_image_free(img);
        return 2;
    }
    face_image_free(img);
    fprintf(stderr, "wrote %s (%dx%d, seed=%u)\n", argv[1], w, h, (unsigned)atoi(argv[2]));
    return 0;
}
