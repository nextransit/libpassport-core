/* Micro-benchmark for the glyph CNN kernel.
 *
 * Isolates conv+fc from process start, PPM load and the MRZ band
 * pipeline, so a kernel change can be measured on its own.  Reports
 * nanoseconds per glyph for the feature extractor and the FC head
 * separately, plus a checksum of the outputs so a "faster" build that
 * also changed the numbers is caught immediately.
 *
 * Build:  cc -O3 -march=native -Iinclude -Isrc bench_cnn_kernel.c cnn.c -lm
 * Run:    ./bench_cnn_kernel [iterations]
 */
#include "cnn.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ROWS 44

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

/* Deterministic pseudo-random glyphs: identical across builds so the
 * output checksum is comparable. */
static void fill_glyphs(float (*g)[CNN_IN_H][CNN_IN_W], int rows) {
    unsigned s = 12345u;
    for (int r = 0; r < rows; ++r)
        for (int y = 0; y < CNN_IN_H; ++y)
            for (int x = 0; x < CNN_IN_W; ++x) {
                s = s * 1103515245u + 12345u;
                g[r][y][x] = ((float)((s >> 16) & 0xFF) / 255.0f) * 2.0f - 1.0f;
            }
}

int main(int argc, char **argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 2000;
    if (iters < 1) iters = 1;

    static float glyphs[ROWS][CNN_IN_H][CNN_IN_W];
    static float feat[ROWS * CNN_FLAT];
    static float probs[ROWS * CNN_OUT];
    fill_glyphs(glyphs, ROWS);

    cnn_t net;
    cnn_default_init(&net);

    /* warm up (page faults, cache, branch predictors) */
    for (int i = 0; i < 20; ++i) {
        cnn_row_features_batch(&net, glyphs, ROWS, feat);
        cnn_fc_batch(&net, feat, ROWS, probs);
    }

    double t0 = now_ns();
    for (int i = 0; i < iters; ++i)
        cnn_row_features_batch(&net, glyphs, ROWS, feat);
    double t1 = now_ns();

    for (int i = 0; i < iters; ++i)
        cnn_fc_batch(&net, feat, ROWS, probs);
    double t2 = now_ns();

    double conv_ns = (t1 - t0) / (double)(iters * ROWS);
    double fc_ns = (t2 - t1) / (double)(iters * ROWS);

    /* FNV-1a over the raw float bits: any change in the arithmetic
     * shows up here. */
    unsigned long long h = 1469598103934665603ULL;
    const unsigned char *p = (const unsigned char *)probs;
    for (size_t i = 0; i < sizeof(probs); ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }

    printf("rows=%d iters=%d\n", ROWS, iters);
    printf("conv_ns_per_glyph  = %.1f\n", conv_ns);
    printf("fc_ns_per_glyph    = %.1f\n", fc_ns);
    printf("total_ns_per_glyph = %.1f\n", conv_ns + fc_ns);
    printf("line_ns_44glyph    = %.0f\n", (conv_ns + fc_ns) * 44.0);
    printf("probs_fnv          = %016llx\n", h);
    return 0;
}
