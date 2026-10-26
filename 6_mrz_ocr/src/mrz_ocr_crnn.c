/* Sequence-aware CRNN entry point.
 *
 * The CRNN reuses the proven CNN geometry front-end (band locate, deskew,
 * line split, 44-cell pitch grid, 16x12 resample) and replaces only the
 * per-character classifier with a GRU over the row's CNN features. The
 * output contract (44 x 37 probabilities) is identical, so the ICAO
 * syntax mask, ink gate and line-2 checksum beam are reused unchanged.
 *
 * The sequence head is enabled through MRZ_OCR_CRNN=1. When the generated
 * weights are absent the recogniser deterministically falls back to the
 * per-character CNN so production output can never degrade because of an
 * untrained model.
 *
 * mrz_ocr_recognise_fuse() is the shipped engine: it runs BOTH heads
 * and combines them with the confidence gate documented at
 * mrz_fuse_probs() in src/mrz_ocr_cnn.c.  Measured on a 3-fold
 * grouped cross-validation over the 41 real photos (every photo held
 * out of its own training fold):
 *
 *     per-character CNN : 21/41 exact, 96.78% char, 19 checksum-wrong
 *     CRNN alone        :  0/41 exact, 67.35% char, 18 checksum-wrong
 *     fused (g=0 r=1 a=0.5): 22/41 exact, 96.90% char, 18 checksum-wrong
 *
 * so the fusion is the only variant that is both at least as good as
 * the CNN and better than it.
 */
#include "mrz_ocr.h"
#include "crnn.h"
#include <stdlib.h>
#include <string.h>

/* Save an environment variable so it can be restored byte-for-byte.
 * Returns NULL when the variable is unset (or the copy cannot be made);
 * callers must not distinguish those two cases: restoring an unset
 * variable is the safe fallback either way. */
static char *mrz_env_save(const char *name) {
    const char *prev = getenv(name);
    if (!prev) return NULL;
    size_t n = strlen(prev) + 1;
    char *copy = malloc(n);
    if (copy) memcpy(copy, prev, n);
    return copy;
}

static void mrz_env_restore(const char *name, const char *saved) {
    if (saved) setenv(name, saved, 1);
    else unsetenv(name);
}

/* Run the recogniser with one environment override in place, then
 * restore the caller's environment exactly (including "not set"). */
static mrz_ocr_status_t mrz_ocr_with_env(const face_image_t *img,
                                          mrz_ocr_result_t *out,
                                          const char *name, const char *value) {
    char *saved = mrz_env_save(name);
    setenv(name, value, 1);
    mrz_ocr_status_t st = mrz_ocr_recognise_cnn(img, out);
    mrz_env_restore(name, saved);
    free(saved);
    return st;
}

mrz_ocr_status_t mrz_ocr_recognise_crnn(const face_image_t *img,
                                        mrz_ocr_result_t *out) {
    if (!crnn_weights_available()) return mrz_ocr_recognise_cnn(img, out);
    /* MRZ_OCR_CRNN is read inside the shared CNN recogniser, so the host
     * tool and the Android JNI shim select the same code path. The
     * variable is restored (not blindly unset) because the caller may
     * have had it set already. */
    return mrz_ocr_with_env(img, out, "MRZ_OCR_CRNN", "1");
}

mrz_ocr_status_t mrz_ocr_recognise_fuse(const face_image_t *img,
                                        mrz_ocr_result_t *out) {
    if (!crnn_weights_available()) return mrz_ocr_recognise_cnn(img, out);
    /* The fusion needs the per-character head AND the sequence head, so
     * MRZ_OCR_CRNN must stay off (it would replace the CNN head) and
     * MRZ_OCR_FUSE must be on. */
    char *saved_c = mrz_env_save("MRZ_OCR_CRNN");
    setenv("MRZ_OCR_CRNN", "0", 1);
    mrz_ocr_status_t st = mrz_ocr_with_env(img, out, "MRZ_OCR_FUSE", "1");
    mrz_env_restore("MRZ_OCR_CRNN", saved_c);
    free(saved_c);
    return st;
}
