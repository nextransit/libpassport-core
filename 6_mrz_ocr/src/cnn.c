/* Tiny OCR-B MLP with hand-crafted pooling features.
 *
 * Inference layout (matches trainer):
 *   fc1 120 -> HIDDEN, ReLU
 *   fc2 HIDDEN -> 37
 */
#include "cnn.h"
#include "template.h"
#include "cnn_weights.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void cnn_default_init(cnn_t *net) {
    memset(net, 0, sizeof(*net));
    memcpy(net->conv_w, DEFAULT_CONV_W, sizeof(net->conv_w));
    memcpy(net->conv_b, DEFAULT_CONV_B, sizeof(net->conv_b));
    memcpy(net->fc1_w,  DEFAULT_FC1_W, sizeof(net->fc1_w));
    memcpy(net->fc1_b,  DEFAULT_FC1_B, sizeof(net->fc1_b));
    memcpy(net->fc2_w,  DEFAULT_FC2_W, sizeof(net->fc2_w));
    memcpy(net->fc2_b,  DEFAULT_FC2_B, sizeof(net->fc2_b));
    net->fc1_scale = DEFAULT_FC1_SCALE;
    net->fc2_scale = DEFAULT_FC2_SCALE;
}

static const int FILTERS[CNN_NF][CNN_K][CNN_K] = {
    {{-1,-1,-1}, {0,0,0}, {1,1,1}},
    {{-1,0,1}, {-1,0,1}, {-1,0,1}},
    {{-1,-1,0}, {-1,0,1}, {0,1,1}},
    {{0,1,1}, {-1,0,1}, {-1,-1,0}},
    {{0,0,0}, {0,1,0}, {0,0,0}},
    {{0,0,0}, {0,0,0}, {0,0,0}},
    {{0,1,0}, {1,1,1}, {0,1,0}},
    {{1,1,1}, {0,0,0}, {0,0,0}},
};

static inline int conv_at(const uint8_t in[CNN_IN_H][CNN_IN_W],
                          int f_idx, int y, int x) {
    int s = 0;
    for (int dy = 0; dy < CNN_K; ++dy)
        for (int dx = 0; dx < CNN_K; ++dx)
            s += (int)in[y+dy][x+dx] * FILTERS[f_idx][dy][dx];
    return s;
}

static void filter_pool3(const uint8_t in[CNN_IN_H][CNN_IN_W],
                         int f_idx, int out[3]) {
    /* Match tools/train_cnn.py conv_pool3 exactly:
     *   12x8 input -> 3x3 conv (valid) -> 10x6 -> 2x2 max-pool
     *   stride 2 -> 5x3. We only collect the first row (oy=0) of the
     *   5x3 pool, columns ox=0,1,2.
     *
     * For each pool cell (oy=0, ox), the 2x2 max-pool window covers
     * conv outputs at input positions (py+dy, ox*2+px+dx) where
     * py in 0..1, px in 0..1, dy in 0..1, dx in 0..1, then each
     * conv output itself is a 3x3 window of the input -> filter.
     *
     * Combined input span:  sy in [py+dy..py+dy+2],
     *                       sx in [ox*2+px+dx..ox*2+px+dx+2].
     * We therefore iterate directly over the 4x4 input slice per
     * (py,px,dy,dx) window and accumulate, then take the max over
     * the 2x2 pool, then the max over py for each ox.
     */
    int best[3] = {-1, -1, -1};
    for (int ox = 0; ox < 3; ++ox) {
        for (int py = 0; py < 2; ++py) {
            for (int px = 0; px < 2; ++px) {
                /* 2x2 max-pool window over conv outputs.
                 * conv output at (oy_c, ox_c) is dot3x3 at input
                 * position (oy_c, ox_c). */
                int cell = -1 << 30;
                for (int dy = 0; dy < 2; ++dy) {
                    for (int dx = 0; dx < 2; ++dx) {
                        int s = 0;
                        /* conv input position (conv_row, conv_col). */
                        int cr = py + dy;       /* 2x2 over rows 0..1 only */
                        int cc = ox * 2 + px + dx;  /* 2x2 over cols ox*2..ox*2+1 */
                        for (int fy = 0; fy < 3; ++fy) {
                            for (int fx = 0; fx < 3; ++fx) {
                                int sy = cr + fy;
                                int sx = cc + fx;
                                if (sy >= 0 && sy < CNN_IN_H &&
                                    sx >= 0 && sx < CNN_IN_W)
                                    s += (int)in[sy][sx] *
                                         FILTERS[f_idx][fy][fx];
                            }
                        }
                        if (s < 0) s = 0;
                        if (s > cell) cell = s;
                    }
                }
                if (cell > best[ox]) best[ox] = cell;
            }
        }
    }
    out[0] = best[0]; out[1] = best[1]; out[2] = best[2];
}

void cnn_forward(const cnn_t *net,
                 const uint8_t glyph[CNN_IN_H][CNN_IN_W],
                 int *pred, float *probs) {
    /* Raw 96-bit bitmap -> FC. */
    float fc_in[CNN_FEATURE];
    for (int i = 0; i < CNN_FEATURE; ++i) fc_in[i] = 0.0f;
    int idx = 0;
    for (int y = 0; y < CNN_IN_H; ++y)
        for (int x = 0; x < CNN_IN_W; ++x)
            fc_in[idx++] = (float)glyph[y][x];
    /* Layer 1: 96 -> HIDDEN, ReLU. */
    float h[CNN_HIDDEN];
    for (int j = 0; j < CNN_HIDDEN; ++j) {
        float s = net->fc1_b[j];
        for (int i = 0; i < CNN_FEATURE; ++i)
            s += fc_in[i] * net->fc1_w[j * CNN_FEATURE + i];
        s *= net->fc1_scale;
        h[j] = s > 0 ? s : 0;   /* ReLU */
    }
    /* Layer 2: HIDDEN -> 37, with per-tensor scale fc2_scale. */
    float logits[CNN_OUT];
    for (int o = 0; o < CNN_OUT; ++o) {
        float s = net->fc2_b[o];
        for (int j = 0; j < CNN_HIDDEN; ++j)
            s += h[j] * net->fc2_w[o * CNN_HIDDEN + j];
        s *= net->fc2_scale;
        logits[o] = s;
    }
    /* Numerically-stable softmax. */
    float mx = logits[0];
    for (int o = 1; o < CNN_OUT; ++o)
        if (logits[o] > mx) mx = logits[o];
    float sum = 0;
    for (int o = 0; o < CNN_OUT; ++o) {
        probs[o] = expf(logits[o] - mx);
        sum += probs[o];
    }
    for (int o = 0; o < CNN_OUT; ++o) probs[o] /= sum > 0 ? sum : 1.0f;
    int best = 0;
    float best_p = probs[0];
    for (int o = 1; o < CNN_OUT; ++o)
        if (probs[o] > best_p) { best_p = probs[o]; best = o; }
    *pred = best;
}
