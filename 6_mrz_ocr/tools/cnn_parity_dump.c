/* Dump CNN inputs and the C reference probabilities for parity checking.
 *
 * The ONNX export (tools/export_cnn_onnx.py) must reproduce the C
 * implementation bit-for-bit-close, otherwise an NPU backend would silently
 * change the recognition result.  This program feeds deterministic pseudo
 * random glyphs to the C path and writes
 *
 *   magic "MRZC" | uint32 count | uint32 in_dim | uint32 out_dim
 *   per glyph: float32 in[in_dim] | float32 probs[out_dim]
 *
 * so that tools/check_cnn_onnx_parity.py can replay the same inputs through
 * onnxruntime and compare.
 *
 * Usage: cnn_parity_dump <default|line1|line2> <count> <out.bin>
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cnn.h"

static uint32_t rng_state = 0x13579bdfu;

static float nextf(float lo, float hi) {
    rng_state = rng_state * 1664525u + 1013904223u;
    uint32_t x = rng_state >> 8;              /* 24 bits */
    float u = (float)x / 16777216.0f;          /* [0,1) */
    return lo + u * (hi - lo);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr,
                "usage: %s <default|line1|line2> <count> <out.bin>\n", argv[0]);
        return 2;
    }
    const char *head = argv[1];
    long count = strtol(argv[2], NULL, 10);
    const char *out_path = argv[3];
    if (count <= 0 || count > 1000000) {
        fprintf(stderr, "bad count\n");
        return 2;
    }

    cnn_t net;
    if (strcmp(head, "default") == 0) {
        cnn_default_init(&net);
    } else if (strcmp(head, "line1") == 0) {
        cnn_line_init(&net, 0);
    } else if (strcmp(head, "line2") == 0) {
        cnn_line_init(&net, 1);
    } else {
        fprintf(stderr, "unknown head %s\n", head);
        return 2;
    }

    FILE *fh = fopen(out_path, "wb");
    if (!fh) { perror("fopen"); return 1; }

    uint32_t hdr[4] = {0x435a524dU /* "MRZC" little-endian */,
                        (uint32_t)count, CNN_IN_H * CNN_IN_W, CNN_OUT};
    fwrite(hdr, sizeof(uint32_t), 4, fh);

    float glyph[CNN_IN_H][CNN_IN_W];
    float probs[CNN_OUT];
    for (long i = 0; i < count; ++i) {
        /* Mix smooth background, blocky strokes and noise so the samples
         * cover the range the real glyph resampler produces. */
        for (int y = 0; y < CNN_IN_H; ++y) {
            for (int x = 0; x < CNN_IN_W; ++x) {
                float v = nextf(-0.2f, 0.2f);            /* background */
                if (nextf(0.0f, 1.0f) > 0.55f)           /* strokes */
                    v += nextf(0.3f, 1.0f);
                if (nextf(0.0f, 1.0f) > 0.93f)           /* noise */
                    v += nextf(-0.3f, 0.3f);
                glyph[y][x] = v;
            }
        }
        cnn_forward(&net, glyph, probs);
        fwrite(glyph, sizeof(float), CNN_IN_H * CNN_IN_W, fh);
        fwrite(probs, sizeof(float), CNN_OUT, fh);
    }
    fclose(fh);
    return 0;
}
