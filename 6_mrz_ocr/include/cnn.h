#ifndef MRZ_OCR_CNN_H
#define MRZ_OCR_CNN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Tiny OCR-B classifier.
 *
 * Architecture:
 *   fc1 96 -> 96, ReLU
 *   fc2 96 -> 37
 *
 * Weights are float32 (no quantisation).
 */
#define CNN_IN_H       12
#define CNN_IN_W        8
#define CNN_K           3
#define CNN_NF          8
#define CNN_FEATURE    96
#define CNN_HIDDEN     96
#define CNN_OUT        37

typedef struct {
    int8_t  conv_w[CNN_NF * CNN_K * CNN_K];
    int8_t  conv_b[CNN_NF];
    float   fc1_w[CNN_FEATURE * CNN_HIDDEN];   /* 96 -> 96 */
    float   fc1_b[CNN_HIDDEN];
    float   fc2_w[CNN_HIDDEN * CNN_OUT];       /* 96 -> 37 */
    float   fc2_b[CNN_OUT];
    float   fc1_scale;
    float   fc2_scale;
} cnn_t;

void cnn_default_init(cnn_t *net);

/* Run inference on an 8x12 binary glyph, return class index and
 * softmax probabilities (size 37). */
void cnn_forward(const cnn_t *net,
                 const uint8_t glyph[CNN_IN_H][CNN_IN_W],
                 int *pred, float *probs);

#ifdef __cplusplus
}
#endif
#endif /* MRZ_OCR_CNN_H */
