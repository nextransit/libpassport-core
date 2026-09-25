#ifndef MRZ_OCR_CNN_H
#define MRZ_OCR_CNN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Tiny ConvNet for single-character OCR-B recognition.
 *
 * Architecture (matches tools/train_cnn.py):
 *   input : 12 rows x 16 cols binary glyph  (binary 0/1)
 *   conv1 : 3x3, 8 filters, stride 1, pad 1 -> ReLU
 *   pool1 : 2x2 max                          -> 6 x 8 x 8
 *   conv2 : 3x3, 16 filters, stride 1, pad 0 -> ReLU
 *   pool2 : 2x2 max                          -> 2 x 3 x 16 = 96
 *   fc1   : 96 -> 64, ReLU
 *   fc2   : 64 -> 37 (softmax)
 *
 * Weights and input normalisation (mean/std) are float32 and live in
 * cnn_weights.h. */
#define CNN_IN_H       12
#define CNN_IN_W       16
#define CNN_K           3
#define CNN_C1           8
#define CNN_C2          16
#define CNN_POOL1_H      6     /* 12/2 */
#define CNN_POOL1_W      8     /* 16/2 */
#define CNN_POOL2_H      2     /* (6-3+1)/2 */
#define CNN_POOL2_W      3     /* (8-3+1)/2 */
#define CNN_FLAT        (CNN_C2 * CNN_POOL2_H * CNN_POOL2_W)  /* 96 */
#define CNN_HIDDEN      64
#define CNN_OUT         37

typedef struct {
    float in_mean[CNN_IN_H * CNN_IN_W];
    float in_std[CNN_IN_H * CNN_IN_W];
    float conv1_w[CNN_K * CNN_K * 1 * CNN_C1];
    float conv1_b[CNN_C1];
    float conv2_w[CNN_K * CNN_K * CNN_C1 * CNN_C2];
    float conv2_b[CNN_C2];
    float fc1_w[CNN_FLAT * CNN_HIDDEN];
    float fc1_b[CNN_HIDDEN];
    float fc2_w[CNN_HIDDEN * CNN_OUT];
    float fc2_b[CNN_OUT];
} cnn_t;

void cnn_default_init(cnn_t *net);

/* Run the conv net on one glyph. Fills `probs` (CNN_OUT floats) with
 * softmax probabilities and returns the argmax class index (0..36). */
int cnn_forward(const cnn_t *net,
                const uint8_t glyph[CNN_IN_H][CNN_IN_W],
                float probs[CNN_OUT]);

/* Row-level API used by the OCR pipeline:
 *   cnn_build_row_features()  builds the 96-dim flattened feature for
 *     each char of the row and stores them in `feat` (n x 96) after
 *     convolving the whole row. This keeps the GEMM vector friendly.
 *   cnn_row_logits()          applies fc1+fc2 to a batch of features.
 */
void cnn_conv_features(const cnn_t *net,
                       const uint8_t glyph[CNN_IN_H][CNN_IN_W],
                       float feat[CNN_FLAT]);

/* Run fc over a batch of pre-computed feature vectors.
 * feat: rows x CNN_FLAT (row-major), probs: rows x CNN_OUT. */
void cnn_fc_batch(const cnn_t *net, const float *feat, int rows,
                  float *probs);

/* Reusable row-batch workspace to avoid per-call malloc/free.
 * A row holds at most MRZ band width <= 64 glyphs. */
#define CNN_BATCH_MAX 64
typedef struct {
    float feat[CNN_BATCH_MAX * CNN_FLAT];
} cnn_batch_t;

/* One-shot convenience: run conv features for `rows` glyphs into a
 * caller-provided buffer (rows x CNN_FLAT). */
void cnn_row_features_batch(const cnn_t *net,
                            const uint8_t (*glyphs)[CNN_IN_H][CNN_IN_W],
                            int rows, float *feat);

#ifdef __cplusplus
}
#endif
#endif /* MRZ_OCR_CNN_H */
