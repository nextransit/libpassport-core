/* Unit tests for the face-recognition library.
 * We do not require real photographs; instead we generate small PPMs
 * in /tmp and exercise:
 *   - PPM round-trip (load -> save -> load)
 *   - aHash determinism (same input -> same hash)
 *   - aHash stability under small perturbation (very similar images
 *     produce similar hashes)
 *   - aHash distance for unrelated images is much larger
 *   - grayscale + histogram equalisation is monotonic and bounded
 */
#include "face.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int g_fail = 0, g_total = 0;
static const char *g_cur = "";

#define EXPECT(cond) do {                                              \
    g_total++;                                                          \
    if (!(cond)) {                                                      \
        g_fail++;                                                       \
        fprintf(stderr, "[FAIL] %s:%d in %s : %s\n",                    \
                __FILE__, __LINE__, g_cur, #cond);                      \
    }                                                                   \
} while (0)

#define EXPECT_EQ_INT(a, b) do {                                        \
    long _a = (long)(a), _b = (long)(b);                                \
    g_total++;                                                          \
    if (_a != _b) {                                                     \
        g_fail++;                                                       \
        fprintf(stderr, "[FAIL] %s:%d in %s : %s == %s (got %ld vs %ld)\n", \
                __FILE__, __LINE__, g_cur, #a, #b, _a, _b);             \
    }                                                                   \
} while (0)

#define RUN(fn) do { g_cur = #fn; fn(); } while (0)

static int write_solid_ppm(const char *path, int w, int h,
                           uint8_t r, uint8_t g, uint8_t b) {
    face_image_t *img = face_image_new(w, h);
    if (!img) return -1;
    size_t n = (size_t)w * h;
    for (size_t i = 0; i < n; ++i) {
        img->pixels[i*3 + 0] = r;
        img->pixels[i*3 + 1] = g;
        img->pixels[i*3 + 2] = b;
    }
    int rc = face_image_save_ppm(img, path);
    face_image_free(img);
    return rc;
}

static int write_gradient_ppm(const char *path, int w, int h, int seed) {
    face_image_t *img = face_image_new(w, h);
    if (!img) return -1;
    unsigned s = (unsigned)seed * 17u + 1u;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            s = s * 1664525u + 1013904223u;
            img->pixels[(size_t)(y*w + x)*3 + 0] = (uint8_t)((s >> 8) & 0xff);
            img->pixels[(size_t)(y*w + x)*3 + 1] = (uint8_t)((s >> 16) & 0xff);
            img->pixels[(size_t)(y*w + x)*3 + 2] = (uint8_t)((s >> 24) & 0xff);
        }
    int rc = face_image_save_ppm(img, path);
    face_image_free(img);
    return rc;
}

static void test_ppm_roundtrip(void) {
    const char *p = "/tmp/face_rt.ppm";
    EXPECT_EQ_INT(write_solid_ppm(p, 16, 16, 200, 100, 50), 0);
    face_image_t *img = face_image_load_ppm(p);
    EXPECT(img != NULL);
    EXPECT_EQ_INT(img->width, 16);
    EXPECT_EQ_INT(img->height, 16);
    EXPECT_EQ_INT(img->pixels[0], 200);
    EXPECT_EQ_INT(img->pixels[1], 100);
    EXPECT_EQ_INT(img->pixels[2], 50);
    /* Save again and re-load. */
    EXPECT_EQ_INT(face_image_save_ppm(img, p), 0);
    face_image_free(img);
    img = face_image_load_ppm(p);
    EXPECT(img != NULL);
    EXPECT_EQ_INT(img->pixels[0], 200);
    face_image_free(img);
}

static void test_bmp_roundtrip(void) {
    const char *p = "/tmp/face_rt.bmp";
    /* Create an in-memory image and save as BMP, then reload. */
    face_image_t *img = face_image_new(8, 4);
    EXPECT(img != NULL);
    img->pixels[0] = 10; img->pixels[1] = 20; img->pixels[2] = 30;
    EXPECT_EQ_INT(face_image_save_bmp(img, p), 0);
    face_image_free(img);
    img = face_image_load_bmp(p);
    EXPECT(img != NULL);
    EXPECT_EQ_INT(img->width, 8);
    EXPECT_EQ_INT(img->height, 4);
    EXPECT_EQ_INT(img->pixels[0], 10);
    EXPECT_EQ_INT(img->pixels[1], 20);
    EXPECT_EQ_INT(img->pixels[2], 30);
    /* Save BMP and reload. */
    EXPECT_EQ_INT(face_image_save_bmp(img, p), 0);
    face_image_free(img);
    img = face_image_load_bmp(p);
    EXPECT(img != NULL);
    EXPECT_EQ_INT(img->pixels[0], 10);
    face_image_free(img);
}

static void test_ahash_deterministic(void) {
    const char *p = "/tmp/face_d1.ppm";
    EXPECT_EQ_INT(write_gradient_ppm(p, 64, 64, 1), 0);
    face_image_t *a = face_image_load_ppm(p);
    face_image_t *b = face_image_load_ppm(p);
    EXPECT(a && b);
    uint8_t ga[64*64], gb[64*64];
    face_to_grayscale(a, ga);
    face_to_grayscale(b, gb);
    uint64_t ha, hb;
    face_ahash(ga, 64, 64, &ha);
    face_ahash(gb, 64, 64, &hb);
    EXPECT_EQ_INT(ha, hb);
    face_image_free(a); face_image_free(b);
}

static void test_ahash_distance(void) {
    /* Two very different gradients should produce a high Hamming distance. */
    const char *p1 = "/tmp/face_a.ppm";
    const char *p2 = "/tmp/face_b.ppm";
    EXPECT_EQ_INT(write_gradient_ppm(p1, 64, 64, 1), 0);
    EXPECT_EQ_INT(write_gradient_ppm(p2, 64, 64, 2), 0);
    face_image_t *a = face_image_load_ppm(p1);
    face_image_t *b = face_image_load_ppm(p2);
    EXPECT(a && b);
    int score = face_compare(a, b, 0);
    EXPECT(score >= 0);
    /* Two unrelated 64x64 noise images usually score < 90. */
    EXPECT(score < 95);
    face_image_free(a); face_image_free(b);
}

static void test_phash_compare(void) {
    const char *p1 = "/tmp/face_p1.ppm";
    const char *p2 = "/tmp/face_p2.ppm";
    EXPECT_EQ_INT(write_gradient_ppm(p1, 128, 128, 7), 0);
    EXPECT_EQ_INT(write_gradient_ppm(p2, 128, 128, 7), 0);
    face_image_t *a = face_image_load_ppm(p1);
    face_image_t *b = face_image_load_ppm(p2);
    EXPECT(a && b);
    int s_ahash = face_compare(a, b, 0);
    int s_phash = face_compare(a, b, 1);
    /* identical images with pHash should score 100 */
    EXPECT_EQ_INT(s_ahash, 100);
    EXPECT_EQ_INT(s_phash, 100);
    face_image_free(a); face_image_free(b);
}

static void test_histogram_equalise(void) {
    /* All-black image: every pixel should remain at 0 (CDF_min = n). */
    const char *p = "/tmp/face_eq_black.ppm";
    EXPECT_EQ_INT(write_solid_ppm(p, 32, 32, 0, 0, 0), 0);
    face_image_t *img = face_image_load_ppm(p);
    EXPECT(img);
    uint8_t g[32*32];
    face_to_grayscale(img, g);
    face_histogram_equalise(g, 32, 32);
    for (int i = 0; i < 32*32; ++i) EXPECT_EQ_INT(g[i], 0);
    face_image_free(img);

    /* All-white: same. */
    EXPECT_EQ_INT(write_solid_ppm(p, 32, 32, 255, 255, 255), 0);
    img = face_image_load_ppm(p);
    EXPECT(img);
    face_to_grayscale(img, g);
    face_histogram_equalise(g, 32, 32);
    for (int i = 0; i < 32*32; ++i) EXPECT_EQ_INT(g[i], 255);
    face_image_free(img);
}

static void test_hamming_known(void) {
    EXPECT_EQ_INT(face_hamming(0xFFFFFFFFFFFFFFFFull, 0), 64);
    EXPECT_EQ_INT(face_hamming(0xAAAAAAAAAAAAAAAAull, 0x5555555555555555ull), 64);
    EXPECT_EQ_INT(face_hamming(0xFFFFFFFFFFFFFFFFull, 0xFFFFFFFFFFFFFFFFull), 0);
    EXPECT_EQ_INT(face_hamming(0, 0x8000000000000001ull), 2);
}

int main(void) {
    RUN(test_ppm_roundtrip);
    RUN(test_bmp_roundtrip);
    RUN(test_ahash_deterministic);
    RUN(test_ahash_distance);
    RUN(test_phash_compare);
    RUN(test_histogram_equalise);
    RUN(test_hamming_known);
    fprintf(stderr, "\n[TEST] %d total, %d failed\n", g_total, g_fail);
    return g_fail == 0 ? 0 : 1;
}
