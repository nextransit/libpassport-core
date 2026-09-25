#ifndef MRZ_OCR_CNN_H
#define MRZ_OCR_CNN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Tiny ConvNet for single-character OCR-B recognition.
 *
 * KEY DESIGN (fixes the 8x12 binarisation + 2px-wide feature-map trap):
 *   - INPUT IS SM00TH GRAYSCALE, not binary. The ocr pipeline bilinear-
 *     resamples the original char box to 16x12 float (NOT a hard 1/2
 *     threshold), preserving sub-pixel stroke edges for conv1.
 *   - Horizontal spatial resolution is protected: pool1 pools ONLY in
 *     the height direction (2x1); conv2 uses pad=1; after pool2 the
 *     feature map is 3 rows x 8 cols (>= 4 cols), so a +-1px shift
 *     does not collapse onto a 2-wide feature map.
 *
 * Architecture (matches tools/train_cnn.py):
 *   input : 12 rows x 16 cols grayscale float (normalised to [-1,1])
 *   conv1 : 3x3, 8 filters, stride 1, pad 1 -> ReLU
 *   pool1 : 2x1 max (height only)           -> 6 x 16 x 8
 *   conv2 : 3x3, 16 filters, stride 1, pad 1 -> ReLU
 *   pool2 : 2x2 max                          -> 3 x 8 x 16 = 384
 *   fc1   : 384 -> 64, ReLU
 *   fc2   : 64  -> 37 (softmax)
 */
#define CNN_IN_H       12
#define CNN_IN_W       16
#define CNN_K           3
#define CNN_C1           8
#define CNN_C2          16
#define CNN_POOL1_H      6     /* 12 / 2 (height-only pooling) */
#define CNN_POOL1_W     16     /* width kept unchanged            */
#define CNN_POOL2_H      3     /* (6-3+1)/2                       */
#define CNN_POOL2_W      8     /* (16-3+1)/2                      */
#define CNN_FLAT        (CNN_C2 * CNN_POOL2_H * CNN_POOL2_W)  /* 384 */
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

/* Run the conv net on one grayscale glyph. Fills `probs` (CNN_OUT
 * floats) with softmax probabilities and returns argmax class. */
int cnn_forward(const cnn_t *net,
                const float glyph[CNN_IN_H][CNN_IN_W],
                float probs[CNN_OUT]);

/* Conv + pool feature extraction for one glyph. */
void cnn_conv_features(const cnn_t *net,
                       const float glyph[CNN_IN_H][CNN_IN_W],
                       float feat[CNN_FLAT]);

/* Run fc over a batch of feature vectors (rows x CNN_FLAT in,
 * rows x CNN_OUT out). */
void cnn_fc_batch(const cnn_t *net, const float *feat, int rows,
                  float *probs);

/* Reusable row-batch workspace to avoid per-call malloc/free. */
#define CNN_BATCH_MAX 64
typedef struct {
    float feat[CNN_BATCH_MAX * CNN_FLAT];
} cnn_batch_t;

void cnn_row_features_batch(const cnn_t *net,
                            const float (*glyphs)[CNN_IN_H][CNN_IN_W],
                            int rows, float *feat);

#ifdef __cplusplus
}
#endif
#endif /* MRZ_OCR_CNN_H */
