/* mrz_ocr_tool -- recognise MRZ text in a passport photo.
 *
 * Usage:
 *   mrz_ocr_tool <image.ppm|img.bmp>
 *
 * Output:
 *   result.ok
 *   result.line1
 *   result.line2
 *   result.conf1
 *   result.conf2
 *   band.x band.y band.w band.h
 */
#include "mrz_ocr.h"
#include "face.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Print one recognition result in the shared KV contract. */
static int run_one(const char *path) {
    face_image_t *img = face_image_load_ppm(path);
    if (!img) img = face_image_load_bmp(path);
    if (!img) { fprintf(stderr, "cannot load %s\n", path); return 2; }
    mrz_ocr_result_t r;
    mrz_ocr_status_t s = mrz_ocr_recognise(img, &r);
    printf("result.ok       : %s\n", mrz_ocr_strerror(s));
    printf("result.line1    : %s\n", r.line1);
    printf("result.line2    : %s\n", r.line2);
    printf("result.conf1    : %d\n", r.line1_avg_conf);
    printf("result.conf2    : %d\n", r.line2_avg_conf);
    printf("band.x band.y band.w band.h : %d %d %d %d\n",
           r.band_x, r.band_y, r.band_w, r.band_h);
    printf("result.done     : 1\n");
    face_image_free(img);
    return s == MRZ_OCR_OK ? 0 : 1;
}

/* Resident mode: one image path per stdin line, one result block out. */
static int serve(void) {
    char line[4096];
    mrz_ocr_init();
    while (fgets(line, sizeof(line), stdin)) {
        size_t n = strlen(line);
        while (n && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = 0;
        if (!n) continue;
        run_one(line);
        fflush(stdout);
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--serve") == 0) return serve();
    if (argc != 2) {
        fprintf(stderr, "usage: %s <image> | --serve\n", argv[0]);
        return 2;
    }
    mrz_ocr_init();
    return run_one(argv[1]);
}
