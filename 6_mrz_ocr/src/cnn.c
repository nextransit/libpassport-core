/* Tiny ConvNet inference for OCR-B glyph recognition (12x16 -> 37).
 *
 * Layer order and tensor shapes exactly match tools/train_cnn.py and
 * src/cnn_weights.h:
 *   input 12x16x1 -> conv1(3x3,CNN_C1=8,pad1) -> ReLU
 *   -> maxpool 2x2 -> 6x8x8
 *   -> conv2(3x3,CNN_C2=16,pad0) -> ReLU
 *   -> maxpool 2x2 -> 2x3x16 = 96
 *   -> fc1(96->64) ReLU -> fc2(64->37)
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
    memcpy(net->fc1_w,  DEFAULT_FC1_W,   sizeof(net->fc1_w));
    memcpy(net->fc1_b,  DEFAULT_FC1_B,   sizeof(net->fc1_b));
    memcpy(net->fc2_w,  DEFAULT_FC2_W,   sizeof(net->fc2_w));
    memcpy(net->fc2_b,  DEFAULT_FC2_B,   sizeof(net->fc2_b));
}

/* Normalise + conv1 (pad 1) + ReLU.
 * out: 12x16x8 (float), x stays the padded normalised field. */
static void conv1_relu(const cnn_t *net,
                       const uint8_t g[CNN_IN_H][CNN_IN_W],
                       float a1[CNN_IN_H][CNN_IN_W][CNN_C1],
                       float xp[CNN_IN_H + 2][CNN_IN_W + 2][1]) {
    /* normalise into padded xp with zero padding */
    for (int y = 0; y < CNN_IN_H + 2; ++y)
        for (int x = 0; x < CNN_IN_W + 2; ++x)
            xp[y][x][0] = 0.0f;
    for (int y = 0; y < CNN_IN_H; ++y) {
        for (int x = 0; x < CNN_IN_W; ++x) {
            int i = y * CNN_IN_W + x;
            float v = (float)g[y][x];
            xp[y + 1][x + 1][0] =
                (v - net->in_mean[i]) / net->in_std[i];
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
                a1[y][x][co] = s > 0.0f ? s : 0.0f;  /* ReLU */
            }
}

/* 2x2 max pool: p1[6][8][CNN_C1]. */
static void pool1(const float a1[CNN_IN_H][CNN_IN_W][CNN_C1],
                  float p1[CNN_POOL1_H][CNN_POOL1_W][CNN_C1]) {
    for (int y = 0; y < CNN_POOL1_H; ++y)
        for (int x = 0; x < CNN_POOL1_W; ++x)
            for (int c = 0; c < CNN_C1; ++c) {
                float m = a1[y * 2][x * 2][c];
                for (int dy = 0; dy < 2; ++dy)
                    for (int dx = 0; dx < 2; ++dx) {
                        float v = a1[y * 2 + dy][x * 2 + dx][c];
                        if (v > m) m = v;
                    }
                p1[y][x][c] = m;
            }
}

/* conv2 (pad 0) + ReLU over 6x8 -> 4x6 out; then 2x2 pool -> 2x3.
 * W2 layout: (kh*3+kw)*CNN_C1*CNN_C2 + ci*CNN_C2 + co. Header stores as
 * Kh,Kw,Cin,Cout (numpy transpose). Emit used
 * W2.ravel() from (Kh,Kw,Cin,Cout) -> index = ((kh*3+kw)*CNN_C1+ci)*CNN_C2+co. */
static void conv2_pool2(const cnn_t *net,
                        const float p1[CNN_POOL1_H][CNN_POOL1_W][CNN_C1],
                        float feat[CNN_FLAT]) {
    /* conv out dims: (6-3+1)=4 rows, (8-3+1)=6 cols */
    const int OH = CNN_POOL1_H - 2;  /* 4 */
    const int OW = CNN_POOL1_W - 2;  /* 6 */
    float a2[4][6][CNN_C2];
    for (int y = 0; y < OH; ++y)
        for (int x = 0; x < OW; ++x)
            for (int co = 0; co < CNN_C2; ++co) {
                float s = net->conv2_b[co];
                for (int kh = 0; kh < 3; ++kh)
                    for (int kw = 0; kw < 3; ++kw)
                        for (int ci = 0; ci < CNN_C1; ++ci)
                            s += p1[y + kh][x + kw][ci] *
                                 net->conv2_w[((kh * 3 + kw) * CNN_C1 + ci)
                                                 * CNN_C2 + co];
                a2[y][x][co] = s > 0.0f ? s : 0.0f;
            }
    /* 2x2 pool -> 2x3x16, flattened (c,y,x) to match torch
     * flatten(1) layout (channel-outer, then row, then col). */
    int idx = 0;
    for (int c = 0; c < CNN_C2; ++c)
        for (int y = 0; y < CNN_POOL2_H; ++y)
            for (int x = 0; x < CNN_POOL2_W; ++x) {
                float m = a2[y * 2][x * 2][c];
                for (int dy = 0; dy < 2; ++dy)
                    for (int dx = 0; dx < 2; ++dx) {
                        float v = a2[y * 2 + dy][x * 2 + dx][c];
                        if (v > m) m = v;
                    }
                feat[idx++] = m;
            }
}

void cnn_conv_features(const cnn_t *net,
                       const uint8_t g[CNN_IN_H][CNN_IN_W],
                       float feat[CNN_FLAT]) {
    float xp[CNN_IN_H + 2][CNN_IN_W + 2][1];
    float a1[CNN_IN_H][CNN_IN_W][CNN_C1];
    float p1[CNN_POOL1_H][CNN_POOL1_W][CNN_C1];
    conv1_relu(net, g, a1, xp);
    pool1(a1, p1);
    conv2_pool2(net, p1, feat);
}

/* fc1 ReLU + fc2 -> softmax over a batch of `rows` feature vectors.
 * feat/rows: rows x CNN_FLAT; probs/rows x CNN_OUT. */
void cnn_row_features_batch(const cnn_t *net,
                            const uint8_t (*glyphs)[CNN_IN_H][CNN_IN_W],
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
                const uint8_t glyph[CNN_IN_H][CNN_IN_W],
                float probs[CNN_OUT]) {
    float feat[CNN_FLAT];
    cnn_conv_features(net, glyph, feat);
    cnn_fc_batch(net, feat, 1, probs);
    int best = 0;
    for (int o = 1; o < CNN_OUT; ++o)
        if (probs[o] > probs[best]) best = o;
    return best;
}

/* Batch-friendly entry used by the OCR row decoder: given a buffer of
 * `rows` glyphs laid out contiguously, fill `probs` for the whole row
 * with one fc pass (cache-friendly sequential features). */
void cnn_row_forward(const cnn_t *net,
                     const uint8_t (*glyphs)[CNN_IN_H][CNN_IN_W],
                     int rows, float *probs) {
    float *feats = (float *)malloc((size_t)rows * CNN_FLAT * sizeof(float));
    if (!feats) return;
    for (int r = 0; r < rows; ++r)
        cnn_conv_features(net, glyphs[r], feats + r * CNN_FLAT);
    cnn_fc_batch(net, feats, rows, probs);
    free(feats);
}

