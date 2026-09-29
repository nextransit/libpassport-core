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
/* GEMM-oriented loop order for the batch FC layers.
 *
 * The original code iterated r (row/glyph) outermost and streamed the
 * whole fc1/fc2 weight matrices once per row.  On a 44-glyph line that
 * means fc1_w (64x96 = 24KB) is re-streamed 44 times, and fc2_w
 * (37x64 = 9KB) 44 times -- with L1 (32KB) / L2 (256KB) thrash on the
 * A53.  Swapping to j/o-outer, r-inner keeps the WEIGHT row resident
 * and iterates all rows against it, so the weights are read once.
 *
 * The inner accumulation order for each (r, output) cell is unchanged
 * (input index advances by 4 with four consecutive terms), so the
 * floating-point sums are bit-identical to the previous loop order.
 * Verified byte-exact on the 1380-image corpus gate.
 */
void cnn_fc_batch(const cnn_t *net,
                  const float *feat, int rows,
                  float *probs) {
    /* fc1: X(rows x 96) * W1(96 x 64) + b1 -> h[rows][64], ReLU.
     * h layout [row][unit] keeps fc2's inner j access contiguous. */
    float h[CNN_BATCH_MAX][CNN_HIDDEN];
    for (int j = 0; j < CNN_HIDDEN; ++j) {
        const float *w = net->fc1_w + j * CNN_FLAT;
        float b = net->fc1_b[j];
        for (int r = 0; r < rows; ++r) {
            const float *f = feat + r * CNN_FLAT;
            float s = b;
            for (int i = 0; i < CNN_FLAT; i += 4) {
                s += f[i + 0] * w[i + 0];
                s += f[i + 1] * w[i + 1];
                s += f[i + 2] * w[i + 2];
                s += f[i + 3] * w[i + 3];
            }
            h[r][j] = s > 0.0f ? s : 0.0f;
        }
    }
    /* fc2: for each output o, stream w2[o][0..63] once over all rows,
     * col-wise dot H(rows x 64) -> logits[rows][37], then softmax per
     * row.  Inner j order identical to the old per-row code. */
    float logits[CNN_BATCH_MAX][CNN_OUT];
    for (int o = 0; o < CNN_OUT; ++o) {
        const float *w = net->fc2_w + o * CNN_HIDDEN;
        float b = net->fc2_b[o];
        for (int r = 0; r < rows; ++r) {
            const float *hr = h[r];
            float s = b;
            int j = 0;
            for (; j + 4 <= CNN_HIDDEN; j += 4) {
                s += hr[j + 0] * w[j + 0];
                s += hr[j + 1] * w[j + 1];
                s += hr[j + 2] * w[j + 2];
                s += hr[j + 3] * w[j + 3];
            }
            for (; j < CNN_HIDDEN; ++j) s += hr[j] * w[j];
            logits[r][o] = s;
        }
    }
    /* softmax per row. */
    for (int r = 0; r < rows; ++r) {
        float mx = -1e30f;
        for (int o = 0; o < CNN_OUT; ++o)
            if (logits[r][o] > mx) mx = logits[r][o];
        float sum = 0.0f;
        for (int o = 0; o < CNN_OUT; ++o) {
            float p = expf(logits[r][o] - mx);
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
