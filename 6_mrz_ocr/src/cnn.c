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
        for (int x = 0; x < CNN_IN_W; ++x)
            for (int co = 0; co < CNN_C1; ++co) {
                float s = net->conv1_b[co];
                for (int kh = 0; kh < 3; ++kh)
                    for (int kw = 0; kw < 3; ++kw)
                        s += xp[y + kh][x + kw][0] *
                             net->conv1_w[(kh * 3 + kw) * CNN_C1 + co];
                a1[y][x][co] = s > 0.0f ? s : 0.0f;
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
    for (int y = 0; y < OH + 2; ++y)
        for (int x = 0; x < OW + 2; ++x) {}
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
        for (int x = 0; x < OW; ++x)
            for (int co = 0; co < CNN_C2; ++co) {
                float s = net->conv2_b[co];
                for (int kh = 0; kh < 3; ++kh)
                    for (int kw = 0; kw < 3; ++kw)
                        for (int ci = 0; ci < CNN_C1; ++ci)
                            s += pp[y + kh][x + kw][ci] *
                                 net->conv2_w[((kh * 3 + kw) * CNN_C1 + ci)
                                                 * CNN_C2 + co];
                a2[y][x][co] = s > 0.0f ? s : 0.0f;
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

void cnn_fc_batch(const cnn_t *net,
                  const float *feat, int rows,
                  float *probs) {
    for (int r = 0; r < rows; ++r) {
        const float *f = feat + r * CNN_FLAT;
        float h[CNN_HIDDEN];
        float logits[CNN_OUT];
        for (int j = 0; j < CNN_HIDDEN; ++j) {
            float s = net->fc1_b[j];
            for (int i = 0; i < CNN_FLAT; ++i)
                s += f[i] * net->fc1_w[j * CNN_FLAT + i];
            h[j] = s > 0.0f ? s : 0.0f;
        }
        float mx = -1e30f;
        for (int o = 0; o < CNN_OUT; ++o) {
            float s = net->fc2_b[o];
            for (int j = 0; j < CNN_HIDDEN; ++j)
                s += h[j] * net->fc2_w[o * CNN_HIDDEN + j];
            logits[o] = s;
            if (s > mx) mx = s;
        }
        float sum = 0.0f;
        for (int o = 0; o < CNN_OUT; ++o) {
            float p = expf(logits[o] - mx);
            probs[r * CNN_OUT + o] = p;
            sum += p;
        }
        for (int o = 0; o < CNN_OUT; ++o)
            probs[r * CNN_OUT + o] /= sum > 0 ? sum : 1.0f;
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
