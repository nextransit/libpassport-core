/* Numerical parity between the exported C weights and a reference
 * implementation of the same bidirectional head.
 *
 * The reference is deliberately simple (plain C, same arithmetic
 * order as src/crnn.c) so a mismatch points at the weight EXPORT, not
 * at the maths.  The Python side (tools/check_crnn_parity.py) prints
 * the same steps; the two must agree step for step.
 *
 * Build: cmake --build build --target test_crnn_parity
 * Run:   build/test_crnn_parity features.bin
 */
#include "crnn.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s features.bin\n", argv[0]);
        return 2;
    }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("open"); return 2; }
    int n = 0;
    if (fread(&n, sizeof(int), 1, f) != 1) { fprintf(stderr, "short read\n"); return 2; }
    if (n != CRNN_STEPS) { fprintf(stderr, "expected %d steps, got %d\n", CRNN_STEPS, n); return 2; }
    float *feat = malloc(sizeof(float) * (size_t)n * CRNN_FEATURES);
    if (!feat) { fprintf(stderr, "oom\n"); return 2; }
    if (fread(feat, sizeof(float), (size_t)n * CRNN_FEATURES, f) !=
        (size_t)n * CRNN_FEATURES) { fprintf(stderr, "short read\n"); return 2; }
    fclose(f);

    crnn_pair_t pair;
    crnn_pair_init(&pair);
    for (int line = 0; line < 2; ++line) {
        float probs[CRNN_STEPS * CRNN_OUT];
        crnn_forward_line(&pair, line, feat, CRNN_STEPS, probs);
        printf("line%d:", line + 1);
        for (int t = 0; t < CRNN_STEPS; ++t) {
            int best = 0;
            for (int o = 1; o < CRNN_OUT; ++o)
                if (probs[t * CRNN_OUT + o] > probs[t * CRNN_OUT + best]) best = o;
            printf(" %d", best);
        }
        printf("\n");
        for (int t = 0; t < 3; ++t) {
            printf("p%d:", t);
            for (int o = 0; o < 5; ++o)
                printf(" %.6f", (double)probs[t * CRNN_OUT + o]);
            printf("\n");
        }
    }
    free(feat);
    return 0;
}
