#ifndef MRZ_OCR_H
#define MRZ_OCR_H

#include <stddef.h>
#include <stdint.h>

#include "face.h"   /* for face_image_t (PPM/BMP I/O already in 3_face_recognition) */

#ifdef __cplusplus
extern "C" {
#endif

/* MRZ-OCR pipeline steps. */
typedef enum {
    MRZ_OCR_OK                  = 0,
    MRZ_OCR_ERR_LOAD            = -1,
    MRZ_OCR_ERR_NO_BAND         = -2,
    MRZ_OCR_ERR_BAD_GEOMETRY    = -3,
    MRZ_OCR_ERR_BAD_LINES       = -4,
    MRZ_OCR_ERR_LOW_CONFIDENCE  = -5,
    MRZ_OCR_ERR_TEMPLATE        = -6,
} mrz_ocr_status_t;

/* One recognised character: the symbol, the per-character confidence
 * (0..100), and the bounding box (in original image coords). */
typedef struct {
    char      ch;
    int       confidence;     /* 0..100 */
    int       x, y, w, h;     /* bounding box in cropped MRZ image */
} mrz_ocr_char_t;

/* Result of an OCR run. */
typedef struct {
    char            line1[64]; /* up to 44 chars + NUL */
    char            line2[64];
    int             line1_len;
    int             line2_len;
    int             line1_avg_conf;
    int             line2_avg_conf;
    int             band_x, band_y, band_w, band_h; /* located MRZ band */
} mrz_ocr_result_t;

/* Initialise the embedded OCR-B template bank. Always returns
 * MRZ_OCR_OK. The bank is a static array of glyph bitmaps covering
 * A-Z, 0-9, and '<' (37 glyphs). */
mrz_ocr_status_t mrz_ocr_init(void);

/* Full pipeline: load the image, locate the MRZ band, segment the
 * characters, recognise them, return the two reconstructed lines.
 *
 *  - `img` is the source image in any PPM/BMP-supported format.
 *  - `out` is filled with the recognised text and per-line confidence.
 *
 * The function does NOT verify the result against ICAO 9303 checks;
 * the caller can pipe the result through `mrz_td3_decode` for that.
 */
mrz_ocr_status_t mrz_ocr_recognise(const face_image_t *img,
                                  mrz_ocr_result_t *out);

/* Free any per-result buffers (none today, but reserved). */
void mrz_ocr_result_free(mrz_ocr_result_t *r);

const char *mrz_ocr_strerror(mrz_ocr_status_t s);

#ifdef __cplusplus
}
#endif

/* Internal helpers exposed for the CNN recogniser. Treat as
 * implementation detail; not part of the public OCR API. */
typedef struct { int x, y, w, h; } mrz_ocr_rect_t;

int mrz_ocr_locate_band(const uint8_t *bin, int W, int H,
                        mrz_ocr_rect_t *out);
int mrz_ocr_split_lines(const uint8_t *bin, int W, int H,
                        mrz_ocr_rect_t *out);
int mrz_ocr_segment_line(const uint8_t *bin, int W, int H,
                         mrz_ocr_rect_t *chars, int max_chars);

/* CNN-based recognition (alternative to the template-bank pipeline). */
mrz_ocr_status_t mrz_ocr_recognise_cnn(const face_image_t *img,
                                       mrz_ocr_result_t *out);

#endif /* MRZ_OCR_H */
