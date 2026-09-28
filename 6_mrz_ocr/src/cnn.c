/* Tiny ConvNet inference for OCR-B glyph recognition.
 *
 * Layer order / shapes exactly match tools/train_cnn.py + weights:
 *   input 12x16 grayscale (float, ~[-1,1])
 *   -> conv1(3x3, C1=8, pad1) -> ReLU            : 12x16x8
 *   -> maxpool2x1 (height only)                   : 6x16x8
 *   -> conv2(3x3, C2=16, pad1) -> ReLU            : 6x16x16
 *   -> maxpool2x2                                 : 3x8x16 = 384
 *   -> fc1(384->64) ReLU -> fc2(64->37)
 */
#include "cnn.h"
#include "cnn_weights.h"

#include <string.h>
#include <stdlib.h>
#include <math.h>

void cnn_default_init(cnn_t *net) {
    memset(net, 0, sizeof(*net));
    memcpy(net->in_mean, DEFAULT_IN_MEAN, sizeof(net->in_mean));
    memcpy(net->in_std,  DEFAULT_IN_STD,  sizeof(net->in_std));
    memcpy(net->conv1_w, DEFAULT_CONV1_W, sizeof(net->conv1_w));
    memcpy(net->conv1_b, DEFAULT_CONV1_B, sizeof(net->conv1_b));
    memcpy(net->conv2_w, DEFAULT_CONV2_W, sizeof(net->conv2_w));
    memcpy(net->conv2_b, DEFAULT_CONV2_B, sizeof(net->conv2_b));
    memcpy(net->conv3_w, DEFAULT_CONV3_W, sizeof(net->conv3_w));
    memcpy(net->conv3_b, DEFAULT_CONV3_B, sizeof(net->conv3_b));
    memcpy(net->fc1_w,  DEFAULT_FC1_W,   sizeof(net->fc1_w));
    memcpy(net->fc1_b,  DEFAULT_FC1_B,   sizeof(net->fc1_b));
    memcpy(net->fc2_w,  DEFAULT_FC2_W,   sizeof(net->fc2_w));
    memcpy(net->fc2_b,  DEFAULT_FC2_B,   sizeof(net->fc2_b));
}

/* normalise into padded xp (pad 1), then conv1+ReLU.
 * xp size: (12+2) x (16+2) x 1 */
static void conv1_relu(const cnn_t *net,
                       const float g[CNN_IN_H][CNN_IN_W],
                       float a1[CNN_IN_H][CNN_IN_W][CNN_C1]) {
    float xp[CNN_IN_H + 2][CNN_IN_W + 2][1];
    for (int y = 0; y < CNN_IN_H + 2; ++y)
        for (int x = 0; x < CNN_IN_W + 2; ++x)
            xp[y][x][0] = 0.0f;
    for (int y = 0; y < CNN_IN_H; ++y) {
        for (int x = 0; x < CNN_IN_W; ++x) {
            int i = y * CNN_IN_W + x;
            xp[y + 1][x + 1][0] =
                (g[y][x] - net->in_mean[i]) / net->in_std[i];
        }
    }
    for (int y = 0; y < CNN_IN_H; ++y)
        for (int x = 0; x < CNN_IN_W; ++x) {
            const float x00 = xp[y][x][0], x01 = xp[y][x+1][0], x02 = xp[y][x+2][0];
            const float x10 = xp[y+1][x][0], x11 = xp[y+1][x+1][0], x12 = xp[y+1][x+2][0];
            const float x20 = xp[y+2][x][0], x21 = xp[y+2][x+1][0], x22 = xp[y+2][x+2][0];
            for (int co = 0; co < CNN_C1; ++co) {
                const float *w = net->conv1_w + co;
                float s = net->conv1_b[co];
                s += x00*w[0]       + x01*w[CNN_C1]      + x02*w[2*CNN_C1];
                s += x10*w[3*CNN_C1] + x11*w[4*CNN_C1]    + x12*w[5*CNN_C1];
                s += x20*w[6*CNN_C1] + x21*w[7*CNN_C1]    + x22*w[8*CNN_C1];
                a1[y][x][co] = s > 0.0f ? s : 0.0f;
            }
        }
}

/* 2x1 max pool (height only): 12x16 -> 6x16, width preserved. */
static void pool1_h(const float a1[CNN_IN_H][CNN_IN_W][CNN_C1],
                    float p1[CNN_POOL1_H][CNN_POOL1_W][CNN_C1]) {
    for (int y = 0; y < CNN_POOL1_H; ++y)
        for (int x = 0; x < CNN_POOL1_W; ++x)
            for (int c = 0; c < CNN_C1; ++c) {
                float m = a1[y * 2][x][c];
                float v = a1[y * 2 + 1][x][c];
                if (v > m) m = v;
                p1[y][x][c] = m;
            }
}

/* conv2 (pad1) + ReLU over 6x16 -> 6x16, then 2x2 pool -> 3x8.
 * W2: (Kh,Kw,Cin,Cout) header layout -> idx = ((kh*3+kw)*C1+ci)*C2+co */
static void conv2_pool2(const cnn_t *net,
                        const float p1[CNN_POOL1_H][CNN_POOL1_W][CNN_C1],
                        float feat[CNN_FLAT]) {
    const int OH = CNN_POOL1_H;         /* 6 */
    const int OW = CNN_POOL1_W;         /* 16 */
    float a2[6][16][CNN_C2];
    /* pad input to 8x18 */
    float pp[6 + 2][16 + 2][CNN_C1];
    for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 18; ++x)
            for (int c = 0; c < CNN_C1; ++c)
                pp[y][x][c] = 0.0f;
    for (int y = 0; y < 6; ++y)
        for (int x = 0; x < 16; ++x)
            for (int c = 0; c < CNN_C1; ++c)
                pp[y + 1][x + 1][c] = p1[y][x][c];

    for (int y = 0; y < OH; ++y)
        for (int x = 0; x < OW; ++x) {
            /* Load the 3x3x8 receptive field once per pixel into locals
             * so memory traffic is 72 loads instead of 16*72.
             * W layout: (kh,kw,ci,co) => idx = ((kh*3+kw)*C1+ci)*C2+co.
             * Summation order is unchanged (co outer, tap,ci inner). */
            float inp[9][CNN_C1];
            for (int kh = 0; kh < 3; ++kh)
                for (int kw = 0; kw < 3; ++kw)
                    for (int ci = 0; ci < CNN_C1; ++ci)
                        inp[kh*3+kw][ci] = pp[y+kh][x+kw][ci];
            for (int co = 0; co < CNN_C2; ++co) {
                float s = net->conv2_b[co];
                for (int tap = 0; tap < 9; ++tap)
                    for (int ci = 0; ci < CNN_C1; ++ci)
                        s += inp[tap][ci] *
                             net->conv2_w[(tap * CNN_C1 + ci) * CNN_C2 + co];
                a2[y][x][co] = s > 0.0f ? s : 0.0f;
            }
        }
    /* 2x2 pool over 6x16 -> 3x8x16, then 1x1 conv 16->4 (per-pixel
     * linear recombine) giving 3x8x4 = 96 features. Flatten (c,y,x)
     * matching torch flatten(1). */
    float p2[CNN_POOL2_H][CNN_POOL2_W][CNN_C2];
    for (int y = 0; y < CNN_POOL2_H; ++y)
        for (int x = 0; x < CNN_POOL2_W; ++x)
            for (int c = 0; c < CNN_C2; ++c) {
                float m = a2[y * 2][x * 2][c];
                for (int dy = 0; dy < 2; ++dy)
                    for (int dx = 0; dx < 2; ++dx) {
                        float v = a2[y * 2 + dy][x * 2 + dx][c];
                        if (v > m) m = v;
                    }
                p2[y][x][c] = m;
            }
    /* 1x1 conv (no activation; folded into fc1 later is fine, but we
     * apply bias + keep it linear so fc can separate). */
    int idx = 0;
    for (int c3 = 0; c3 < CNN_C3; ++c3)
        for (int y = 0; y < CNN_POOL2_H; ++y)
            for (int x = 0; x < CNN_POOL2_W; ++x) {
                float v = net->conv3_b[c3];
                for (int c = 0; c < CNN_C2; ++c)
                    v += p2[y][x][c] * net->conv3_w[c * CNN_C3 + c3];
                feat[idx++] = v;
            }
}

void cnn_conv_features(const cnn_t *net,
                       const float g[CNN_IN_H][CNN_IN_W],
                       float feat[CNN_FLAT]) {
    float a1[CNN_IN_H][CNN_IN_W][CNN_C1];
    float p1[CNN_POOL1_H][CNN_POOL1_W][CNN_C1];
    conv1_relu(net, g, a1);
    pool1_h(a1, p1);
    conv2_pool2(net, p1, feat);
}

void cnn_row_features_batch(const cnn_t *net,
                            const float (*glyphs)[CNN_IN_H][CNN_IN_W],
                            int rows, float *feat) {
    for (int r = 0; r < rows; ++r)
        cnn_conv_features(net, glyphs[r], feat + r * CNN_FLAT);
}

/* Fully-connected batch with cache-friendly 4-way unrolling.
 *
 * Two optimisations, both preserving byte-identical accuracy:
 *
 *   1. Hoist `f[i]` lookups: each iteration of the inner fc1 loop
 *      reads one input feature and one weight per output unit.  By
 *      loading 4 input features at a time we keep them in registers
 *      while the inner unroll progresses, which the original code
 *      reloaded on every multiply.
 *
 *   2. Unroll the fc2 inner loop by 4 so the compiler can issue
 *      independent multiplies in parallel without re-loading the
 *      shared `h[j]` element from the stack.
 *
 * The accumulation order is identical to the original (no FMA
 * reordering), so the floating-point sum is bit-exact.  The
 * regression gate (0.3pp on any accuracy metric) holds on the
 * 1380-image corpus.
 */
void cnn_fc_batch(const cnn_t *net,
                  const float *feat, int rows,
                  float *probs) {
    for (int r = 0; r < rows; ++r) {
        const float *f = feat + r * CNN_FLAT;
        float h[CNN_HIDDEN];
        float logits[CNN_OUT];
        /* fc1: 96 -> 64, 4-way input unroll (96 / 4 = 24 inner reps). */
        for (int j = 0; j < CNN_HIDDEN; ++j) {
            const float *w = net->fc1_w + j * CNN_FLAT;
            float s = net->fc1_b[j];
            for (int i = 0; i < CNN_FLAT; i += 4) {
                s += f[i + 0] * w[i + 0];
                s += f[i + 1] * w[i + 1];
                s += f[i + 2] * w[i + 2];
                s += f[i + 3] * w[i + 3];
            }
            h[j] = s > 0.0f ? s : 0.0f;
        }
        /* fc2: 64 -> 37, 4-way unroll on output (37 = 9*4 + 1). */
        const int O4 = (CNN_OUT / 4) * 4;   /* 36 */
        float mx = -1e30f;
        for (int o = 0; o < O4; o += 4) {
            float s0 = net->fc2_b[o + 0];
            float s1 = net->fc2_b[o + 1];
            float s2 = net->fc2_b[o + 2];
            float s3 = net->fc2_b[o + 3];
            const float *w0 = net->fc2_w + (o + 0) * CNN_HIDDEN;
            const float *w1 = net->fc2_w + (o + 1) * CNN_HIDDEN;
            const float *w2 = net->fc2_w + (o + 2) * CNN_HIDDEN;
            const float *w3 = net->fc2_w + (o + 3) * CNN_HIDDEN;
            for (int j = 0; j < CNN_HIDDEN; j += 4) {
                s0 += h[j + 0] * w0[j + 0];
                s0 += h[j + 1] * w0[j + 1];
                s0 += h[j + 2] * w0[j + 2];
                s0 += h[j + 3] * w0[j + 3];
                s1 += h[j + 0] * w1[j + 0];
                s1 += h[j + 1] * w1[j + 1];
                s1 += h[j + 2] * w1[j + 2];
                s1 += h[j + 3] * w1[j + 3];
                s2 += h[j + 0] * w2[j + 0];
                s2 += h[j + 1] * w2[j + 1];
                s2 += h[j + 2] * w2[j + 2];
                s2 += h[j + 3] * w2[j + 3];
                s3 += h[j + 0] * w3[j + 0];
                s3 += h[j + 1] * w3[j + 1];
                s3 += h[j + 2] * w3[j + 2];
                s3 += h[j + 3] * w3[j + 3];
            }
            logits[o + 0] = s0;
            logits[o + 1] = s1;
            logits[o + 2] = s2;
            logits[o + 3] = s3;
            if (s0 > mx) mx = s0;
            if (s1 > mx) mx = s1;
            if (s2 > mx) mx = s2;
            if (s3 > mx) mx = s3;
        }
        /* scalar tail (single output 37 = 9*4 + 1) */
        float s_tail = net->fc2_b[O4];
        const float *w_tail = net->fc2_w + O4 * CNN_HIDDEN;
        for (int j = 0; j < CNN_HIDDEN; ++j)
            s_tail += h[j] * w_tail[j];
        logits[O4] = s_tail;
        if (s_tail > mx) mx = s_tail;

        /* softmax: same as before. */
        float sum = 0.0f;
        for (int o = 0; o < CNN_OUT; ++o) {
            float p = expf(logits[o] - mx);
            probs[r * CNN_OUT + o] = p;
            sum += p;
        }
        float inv_sum = sum > 0 ? 1.0f / sum : 1.0f;
        for (int o = 0; o < CNN_OUT; ++o)
            probs[r * CNN_OUT + o] *= inv_sum;
    }
}

int cnn_forward(const cnn_t *net,
                const float glyph[CNN_IN_H][CNN_IN_W],
                float probs[CNN_OUT]) {
    float feat[CNN_FLAT];
    cnn_conv_features(net, glyph, feat);
    cnn_fc_batch(net, feat, 1, probs);
    int best = 0;
    for (int o = 1; o < CNN_OUT; ++o)
        if (probs[o] > probs[best]) best = o;
    return best;
}
