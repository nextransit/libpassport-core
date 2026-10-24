/* bench_loop -- in-process latency of mrz_ocr_recognise_cnn on one PPM.
 *
 * Why a separate harness: `mrz_ocr_cnn_tool` reports ~7.4 ms wall per image,
 * but ~4.7 ms of that is process spawn + load + mrz_ocr_init.  Timing the
 * library in a loop removes the startup cost and the scheduler noise, which
 * is what "how fast is the recogniser" actually means.  Android measures the
 * same thing (`ms` in the JNI JSON is a steady_clock delta around the call).
 *
 * Build (not part of CMake on purpose -- it is a measurement tool, not a
 * shipped artifact):
 *   clang -O2 -o build/bench_loop tools/bench_loop.c \
 *         -Iinclude -I../3_face_recognition/include \
 *         build/libmrz_ocr.a build/face_build/libface.a -lm -lpthread
 *
 * Usage:
 *   bench_loop <image.ppm> [repeats]
 * Prints every sample in microseconds so the caller can take percentiles.
 */
#include "mrz_ocr.h"
#include "face.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <image.ppm> [repeats]\n", argv[0]);
        return 2;
    }
    int reps = (argc > 2) ? atoi(argv[2]) : 30;
    if (reps < 1) reps = 1;

    face_image_t *img = face_image_load_ppm(argv[1]);
    if (!img) img = face_image_load_bmp(argv[1]);
    if (!img) { fprintf(stderr, "cannot load %s\n", argv[1]); return 2; }

    /* Init once, outside the timed region: Android does the same (it is
     * idempotent and the JNI shim calls it per request, but the first call
     * is the only one that allocates). */
    mrz_ocr_init();

    mrz_ocr_result_t r;
    /* one warm-up pass so the first sample is not the one that faults in
     * the weight pages */
    mrz_ocr_recognise_cnn(img, &r);

    for (int i = 0; i < reps; ++i) {
        double t0 = now_ms();
        mrz_ocr_status_t s = mrz_ocr_recognise_cnn(img, &r);
        double t1 = now_ms();
        if (i == 0) {
            /* report the reading once, on stderr, so stdout stays numeric */
            fprintf(stderr, "status=%d line2=%s\n", (int)s, r.line2);
        }
        printf("%.3f\n", t1 - t0);
    }
    face_image_free(img);
    return 0;
}
