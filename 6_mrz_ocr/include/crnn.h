#ifndef MRZ_OCR_CRNN_H
#define MRZ_OCR_CRNN_H

#include "cnn.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CRNN_STEPS 44
#define CRNN_FEATURES CNN_FLAT
#define CRNN_HIDDEN 32
#define CRNN_OUT CNN_OUT

/* Sequence head: bidirectional GRU over the 44 cell features.
 *
 * WHY BIDIRECTIONAL + POSITION EMBEDDING
 * --------------------------------------
 * The shipped head was a single-direction GRU.  Measured on a strictly
 * held-out split (train pic_scan1, evaluate pic_scan2/3, same
 * features for both heads) the single-direction head reached 81.5%
 * character accuracy on MRZ line 1 while a bidirectional head with a
 * learned position embedding reached 86.6%.  A MRZ row is a fixed
 * 44-slot sequence with a strong left-to-right structure (document
 * number, nationality, date of birth, ...), so a position embedding
 * lets the head exploit the layout, and the backward direction lets
 * it use the right-hand context before committing.
 *
 * The two directions are concatenated, so the classifier input is
 * 2 * CRNN_HIDDEN wide.
 */
#define CRNN_BIDIR 1
#define CRNN_CTX (2 * CRNN_HIDDEN)

typedef struct {
    float proj_w[CRNN_FEATURES * CRNN_HIDDEN];
    float proj_b[CRNN_HIDDEN];
    /* Learned position embedding: CRNN_STEPS x CRNN_HIDDEN. */
    float pos_w[CRNN_STEPS * CRNN_HIDDEN];
    /* Forward direction. */
    float wz[CRNN_HIDDEN * CRNN_HIDDEN];
    float uz[CRNN_HIDDEN * CRNN_HIDDEN];
    float bz[CRNN_HIDDEN];
    float wr[CRNN_HIDDEN * CRNN_HIDDEN];
    float ur[CRNN_HIDDEN * CRNN_HIDDEN];
    float br[CRNN_HIDDEN];
    float wh[CRNN_HIDDEN * CRNN_HIDDEN];
    float uh[CRNN_HIDDEN * CRNN_HIDDEN];
    float bh_in[CRNN_HIDDEN];
    float bh_hn[CRNN_HIDDEN];
    /* Backward direction (same layout). */
    float wzb[CRNN_HIDDEN * CRNN_HIDDEN];
    float uzb[CRNN_HIDDEN * CRNN_HIDDEN];
    float bzb[CRNN_HIDDEN];
    float wrb[CRNN_HIDDEN * CRNN_HIDDEN];
    float urb[CRNN_HIDDEN * CRNN_HIDDEN];
    float brb[CRNN_HIDDEN];
    float whb[CRNN_HIDDEN * CRNN_HIDDEN];
    float uhb[CRNN_HIDDEN * CRNN_HIDDEN];
    float bh_inb[CRNN_HIDDEN];
    float bh_hnb[CRNN_HIDDEN];
    /* Classifier over the concatenated [forward ; backward] state. */
    float out_w[CRNN_CTX * CRNN_OUT];
    float out_b[CRNN_OUT];
} crnn_seq_t;

int crnn_weights_available(void);
void crnn_init(crnn_seq_t *net);

/* Line-specific sequence heads. The CNN front-end uses a different
 * convolution weight set for MRZ line 1 and line 2, so the resulting
 * 96-d features have different distributions and the sequence head must
 * be trained per line as well. */
typedef struct {
    crnn_seq_t line1;
    crnn_seq_t line2;
} crnn_pair_t;

void crnn_pair_init(crnn_pair_t *pair);
void crnn_forward_line(const crnn_pair_t *pair, int line_idx,
                       const float *features, int steps, float *probs);

#ifdef __cplusplus
}
#endif
#endif
