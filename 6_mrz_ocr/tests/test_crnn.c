#include "crnn.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    crnn_pair_t pair;
    crnn_pair_init(&pair);
    assert(crnn_weights_available() == 1);

    for (int line = 0; line < 2; ++line) {
        float features[CRNN_STEPS * CRNN_FEATURES] = {0};
        float probs[CRNN_STEPS * CRNN_OUT] = {0};
        crnn_forward_line(&pair, line, features, CRNN_STEPS, probs);
        for (int t = 0; t < CRNN_STEPS; ++t) {
            float sum = 0.0f;
            for (int c = 0; c < CRNN_OUT; ++c) {
                assert(isfinite(probs[t * CRNN_OUT + c]));
                sum += probs[t * CRNN_OUT + c];
            }
            assert(fabsf(sum - 1.0f) < 1e-5f);
        }
    }

    /* The two heads must be independent: if they collapsed back to a
     * shared weight set, the per-line training silently stopped working. */
    assert(memcmp(pair.line1.out_b, pair.line2.out_b,
                  sizeof(pair.line1.out_b)) != 0);

    puts("crnn contract ok (dual line heads, softmax sums to 1)");
    return 0;
}