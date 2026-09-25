/* OCR-B glyph bank for MRZ.
 *
 * Each glyph is a 12-row x 8-column bitmap, MSB-first on the leftmost
 * pixel of each row. Rows are stored in 8-bit words.
 *
 * The shapes are simplified renderings of ICAO 9303 OCR-B character
 * forms. They are good enough for synthetic test data and for printed
 * machine-readable zones scanned at >= 200 dpi. They are NOT a
 * general-purpose OCR-B font.
 *
 * 0xFF = foreground (ink), 0x00 = background (paper).
 */
#ifndef MRZ_OCR_TEMPLATE_H
#define MRZ_OCR_TEMPLATE_H

#include <stdint.h>

#define MRZ_OCR_GLYPH_W  8
#define MRZ_OCR_GLYPH_H  12
#define MRZ_OCR_GLYPH_COUNT 37   /* A-Z + 0-9 + '<' */

typedef struct {
    char     ch;
    uint8_t  rows[MRZ_OCR_GLYPH_H]; /* bit 7 = leftmost pixel */
} mrz_ocr_glyph_t;

extern const mrz_ocr_glyph_t MRZ_OCR_GLYPHS[MRZ_OCR_GLYPH_COUNT];

/* Returns the glyph index (0..36) for ASCII `c` or -1 if unsupported. */
int mrz_ocr_glyph_index(char c);

#endif
