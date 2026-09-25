#ifndef FACE_H
#define FACE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* PPM (P6) reader/writer -- the simplest portable pixel format. */
typedef struct {
    int width;
    int height;
    uint8_t *pixels;  /* RGB triplets, row-major, length = width*height*3 */
} face_image_t;

face_image_t *face_image_new(int w, int h);
void          face_image_free(face_image_t *img);

/* Read a binary P6 PPM from `path`. Returns NULL on error. */
face_image_t *face_image_load_ppm(const char *path);

/* Write a binary P6 PPM to `path`. Returns 0 on success. */
int face_image_save_ppm(const face_image_t *img, const char *path);

/* Read a 24-bit uncompressed BMP. Returns NULL on error. */
face_image_t *face_image_load_bmp(const char *path);
int           face_image_save_bmp(const face_image_t *img, const char *path);

/* Grayscale conversion (BT.601 luma). Allocates `out` of size w*h. */
int face_to_grayscale(const face_image_t *in, uint8_t *out);

/* Histogram equalisation on a single-channel grayscale buffer in place. */
int face_histogram_equalise(uint8_t *gray, int w, int h);

/* Resize to w x h using nearest-neighbour sampling. Allocates `out`. */
int face_resize_nn(const uint8_t *in, int iw, int ih, int ow, int oh, uint8_t *out);

/* Average-hash (aHash) over a grayscale image. The output is an 8-byte
 * (64-bit) hash stored in `out_hash`. The image is internally resized
 * to 8x8. */
int face_ahash(const uint8_t *gray, int w, int h, uint64_t *out_hash);

/* pHash based on 8x8 DCT (a tiny self-contained DCT implementation).
 * Same 64-bit output layout as aHash. */
int face_phash(const uint8_t *gray, int w, int h, uint64_t *out_hash);

/* Hamming distance between two 64-bit hashes. */
int face_hamming(uint64_t a, uint64_t b);

/* Convenience: compare two images and return the similarity score
 * (0..100). `method` is 0 for aHash, 1 for pHash, 2 for both averaged. */
int face_compare(const face_image_t *a, const face_image_t *b, int method);

#ifdef __cplusplus
}
#endif
#endif /* FACE_H */
