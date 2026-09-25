/* mrz_ocr_cnn_tool -- like mrz_ocr_tool but uses the CNN recogniser. */
#include "mrz_ocr.h"
#include "face.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <image>\n", argv[0]);
        return 2;
    }
    face_image_t *img = face_image_load_ppm(argv[1]);
    if (!img) img = face_image_load_bmp(argv[1]);
    if (!img) { fprintf(stderr, "cannot load %s\n", argv[1]); return 2; }
    mrz_ocr_init();
    mrz_ocr_result_t r;
    mrz_ocr_status_t s = mrz_ocr_recognise_cnn(img, &r);
    printf("method            : cnn\n");
    printf("result.ok         : %s\n", mrz_ocr_strerror(s));
    printf("result.line1      : %s\n", r.line1);
    printf("result.line2      : %s\n", r.line2);
    printf("result.conf1      : %d\n", r.line1_avg_conf);
    printf("result.conf2      : %d\n", r.line2_avg_conf);
    printf("band.x band.y band.w band.h : %d %d %d %d\n",
           r.band_x, r.band_y, r.band_w, r.band_h);
    face_image_free(img);
    return s == MRZ_OCR_OK ? 0 : 1;
}
