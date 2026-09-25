/* face_tool -- compare two face images and report similarity.
 * Usage:
 *   face_tool <image-a> <image-b> [method]
 *   method: 0 = aHash (default), 1 = pHash, 2 = both
 * Returns exit code 0 if score >= threshold (default 60), else 1.
 */
#include "face.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "usage: %s <image-a> <image-b> [0|1|2]\n", argv[0]);
        return 2;
    }
    int method = argc == 4 ? atoi(argv[3]) : 0;
    face_image_t *a = face_image_load_ppm(argv[1]);
    if (!a) a = face_image_load_bmp(argv[1]);
    if (!a) { fprintf(stderr, "cannot load %s\n", argv[1]); return 2; }
    face_image_t *b = face_image_load_ppm(argv[2]);
    if (!b) b = face_image_load_bmp(argv[2]);
    if (!b) { fprintf(stderr, "cannot load %s\n", argv[2]); face_image_free(a); return 2; }

    int score = face_compare(a, b, method);
    if (score < 0) {
        fprintf(stderr, "compare failed\n");
        face_image_free(a); face_image_free(b);
        return 2;
    }
    const char *mname[] = {"aHash", "pHash", "aHash+pHash"};
    printf("method: %s\n", mname[method]);
    printf("score : %d/100\n", score);
    int rc = score >= 60 ? 0 : 1;
    face_image_free(a); face_image_free(b);
    return rc;
}
