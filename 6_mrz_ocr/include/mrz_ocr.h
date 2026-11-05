#ifndef MRZ_OCR_H
#define MRZ_OCR_H

#include <stddef.h>
#include <stdint.h>
#include <time.h>   /* struct timespec (deadline snapshot API) */

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
    MRZ_OCR_ERR_TIMEOUT         = -7,
} mrz_ocr_status_t;

/* Document layout classification (TD3 vs CS/CD vs TD1). */
typedef enum {
    MRZ_LAYOUT_UNKNOWN       = 0,
    MRZ_LAYOUT_SINGLE_LINE_30,   /* 港澳通行证(CS) / 台湾通行证(CD) 正面单行 30 字符 */
    MRZ_LAYOUT_TD3_DOUBLE_44,    /* 护照 (TD3) 双行 44 字符 */
    MRZ_LAYOUT_TD1_TRIPLE_30,    /* 回乡证(CR) / 台胞证背面 三行 30 字符 */
} mrz_doc_layout_t;

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
    /* Ambiguity of the line-2 reading.
     *
     * A checksum-valid line is NOT automatically the right one: a 'B'
     * read as '8' can keep every check digit satisfied.  When the
     * search found a second checksum-valid reading whose probability
     * is close to the winner's, the reading is genuinely ambiguous and
     * the caller may want to ask the user to retake the photo.
     *
     * `line2_ambiguous` is 1 when such a runner-up exists.
     * `line2_alt_margin` is the log-probability gap between the best
     * and the runner-up (0 when there is no runner-up); a small value
     * means "nearly a coin flip".
     * `line2_alt[64]` holds the runner-up reading. */
    int             line2_ambiguous;
    int             line2_alt_margin;
    char            line2_alt[64];
    /* Structural sanity of the line-2 reading, independent of the five
     * check digits.
     *
     * The check digits only prove that 44 characters are internally
     * consistent, not that they are text.  On a corpus of scraped ID
     * photos the decoder's beam search closed the checksum on 49 of 84
     * images while 35 of those 49 carried an impossible nationality
     * code and 9 carried an impossible birth or expiry date -- i.e. the
     * beam can satisfy arithmetic over noise.  A caller that shows the
     * result to a user, or stores it, needs to know the difference.
     *
     * `line2_struct_ok` is 1 when every field passes the checks that a
     * real TD3 record cannot fail:
     *   - both dates parse as a real calendar date (month 1..12, day
     *     within the month, leap years included)
     *   - the birth date is not in the future
     *   - the expiry date is not before the birth date
     *   - neither date is more than a lifetime away from today
     * It is 0 otherwise, and 0 when no 44-character line was produced.
     * It is a veto, never an endorsement: 1 does not mean "correct". */
    int             line2_struct_ok;
    /* Advisory: 1 when the nationality field (cols 10..12) is three
     * letters with no '<' and is not an ISO-3166-1 alpha-3 code nor one
     * of the ICAO special codes (XXA/XXB/XXC/XXX/UNO).  Fields that
     * contain '<' are never flagged, because Germany prints "D<<".
     *
     * Not a veto: Kosovo prints "RKS", which is not ISO.  But 19 of the
     * 28 readings on assets/pic that satisfied every check digit
     * carried a code no state issues ("HOL", "UKH", "HUS", "UST"), so
     * this flag separates "arithmetically consistent" from "could be a
     * real document" far better than the check digits alone. */
    int             line2_nat_suspect;
    /* Cross-line consistency: the issuing state in line 1 (cols 2..4)
     * versus the nationality in line 2 (cols 10..12).  1 means they
     * agree (or the comparison does not apply), 0 means they disagree.
     *
     * This is the only signal that caught a single-character error in
     * line 1 on the 41-photo phone corpus: it flagged 4 images and all
     * 4 were wrong.  Neither the confidence nor the ambiguity flag did
     * -- one image with a wrong line 1 still scored 98.
     *
     * ADVISORY, not a veto: a refugee travel document legitimately
     * carries different codes in the two lines, and "D<<" / "RKS" style
     * fields are already excluded by the '<' test inside. */
    int             line1_country_mismatch;
    /* Advisory: 1 when the ISSUING STATE in line 1 (cols 2..4) is three
     * letters with no '<' that is neither an ISO-3166-1 alpha-3 code nor
     * one of the ICAO special codes (XXA/XXB/XXC/XXX/UNO).
     *
     * This closes an asymmetry that had been sitting in the code since
     * the start: `line2_nat_suspect` has always checked the nationality
     * in line 2, but nothing checked the issuing state in line 1 -- and
     * line 1 is the row that carries NO check digits, so it is exactly
     * the row that needs a content test.
     *
     * On the 41-photo phone corpus all four images with a wrong line 1
     * carried a fabricated issuing-state code ("HUG", "CHY", "CNN",
     * "ATA") that no state issues, and every hand-read ground truth in
     * both corpora carried a real alpha-3 code.
     *
     * ADVISORY, not a veto: Kosovo prints "RKS", which is not ISO, and
     * a state may print a partial code.  A field containing '<' is
     * never flagged. */
    int             line1_issuer_suspect;
    /* Document layout classification result. */
    mrz_doc_layout_t doc_layout;
} mrz_ocr_result_t;

/* Initialise the embedded OCR-B template bank. Always returns
 * MRZ_OCR_OK. The bank is a static array of glyph bitmaps covering
 * A-Z, 0-9, and '<' (37 glyphs). */
mrz_ocr_status_t mrz_ocr_init(void);

/* Structural validation of a 44-character TD3 line 2.
 *
 * `mrz_ocr_line2_struct_ok` returns 1 when every field is at least
 * structurally possible: both dates parse as real calendar dates (leap
 * years included), the birth date is not in the future, the expiry date
 * is not before the birth date, and neither is more than a lifetime
 * away.  It returns 0 for any other length.
 *
 * This is independent of the five ICAO check digits, and that is the
 * point: the check digits only prove 44 characters are internally
 * consistent.  On a corpus of scraped ID photos the decoder closed the
 * checksum on 28 of 84 images while 19 of those 28 carried a
 * nationality code that no state issues and 1 carried a date that does
 * not exist.  Use this as a VETO, never as an endorsement: 1 does not
 * mean the reading is correct, 0 means it cannot be.
 *
 * `mrz_ocr_line2_nat_suspect` returns 1 when columns 10..12 are three
 * letters containing no '<' and are not an ISO-3166-1 alpha-3 code nor
 * one of the ICAO special codes (XXA/XXB/XXC/XXX/UNO).  Fields that
 * contain '<' are never flagged, because Germany prints "D<<".
 *
 * ADVISORY, not a veto: Kosovo prints "RKS", which is not ISO, so a
 * caller that rejects on this trades a small number of false rejects
 * for a large cut in silently wrong output.
 *
 * Both functions are pure and thread-safe; the recogniser fills the
 * matching fields of mrz_ocr_result_t for you, but they are exported so
 * that other backends (and callers holding a line from elsewhere) can
 * apply the same rules.
 *
 * `mrz_ocr_line1_country_mismatch` compares the issuing state in line 1
 * (cols 2..4) with the nationality in line 2 (cols 10..12).  It returns
 * 1 when they agree or when the comparison does not apply (either field
 * contains '<', or a line is not 44 characters), and 0 when they are
 * both three plain letters and differ.
 *
 * Like the nationality check this is ADVISORY: a refugee travel
 * document may legitimately differ.  But on the 41-photo phone corpus
 * it was the only signal that caught a single-character error in line 1
 * -- it flagged 4 images and all 4 were wrong, while the confidence and
 * the ambiguity flag together caught none of them.
 *
 * `mrz_ocr_line1_issuer_suspect` asks the same question about the
 * issuing state that `mrz_ocr_line2_nat_suspect` asks about the
 * nationality: is it a code that some state actually issues?  It
 * returns 1 when the code is not a plausible ISO-3166-1 alpha-3 code
 * (and contains no '<'), 0 when it is plausible or the comparison does
 * not apply.  Same advisory status, same reason. */
#ifdef __cplusplus
extern "C" {
#endif
int mrz_ocr_line2_struct_ok(const char *line2);
int mrz_ocr_line2_nat_suspect(const char *line2);
int mrz_ocr_line1_country_mismatch(const char *line1, const char *line2);
int mrz_ocr_line1_issuer_suspect(const char *line1);
#ifdef __cplusplus
}
#endif

/* Set a per-thread monotonic deadline for the next recogniser call.
 * A non-positive value disables the deadline. */
void mrz_ocr_deadline_set_ms(int timeout_ms);
int  mrz_ocr_deadline_expired(void);

/* Snapshot / restore the calling thread's deadline.  The deadline lives
 * in thread-local storage, so the parallel line-1 worker would otherwise
 * run with an empty one: its TTA and beam could extend the frame past
 * the caller's budget while the main thread waits without a timeout.
 * Snapshot the deadline before submitting the job and apply it at the
 * top of the worker routine. */
void mrz_ocr_deadline_get(struct timespec *ts, int *enabled);
void mrz_ocr_deadline_apply(const struct timespec *ts, int enabled);

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
#ifdef __cplusplus
extern "C" {
#endif

typedef struct { int x, y, w, h; } mrz_ocr_rect_t;

/* Shared Otsu threshold over a grayscale buffer (single
 * implementation for both backends and for the band-local
 * refinement pass). */
int mrz_ocr_otsu(const uint8_t *gray, int n);

/* Shared geometry layer (see src/mrz_geom.c). */
double mrz_estimate_band_skew(const uint8_t *bin, int W, int H);
void mrz_rotate_band(const uint8_t *src, int W, int H, double deg,
                     uint8_t fill, int bilinear, uint8_t *dst);
void mrz_normalize_local_contrast(const uint8_t *gray, int W, int H,
                                  uint8_t *out);
void mrz_dewarp_parabolic(const uint8_t *src, uint8_t *dst,
                          int width, int height, int src_stride, int dst_stride);

int mrz_ocr_locate_band(const uint8_t *bin, int W, int H,
                        mrz_ocr_rect_t *out);
int mrz_ocr_split_lines(const uint8_t *bin, int W, int H,
                        mrz_ocr_rect_t *out);

/* Single-row fallback for documents that carry ONE MRZ row (CS/CD
 * travel permits, 1 x 30 characters).  Returns 0 and fills out[0] when
 * the band is dominated by a single text row. */
int mrz_ocr_split_lines_single(const uint8_t *bin, int W, int H,
                               mrz_ocr_rect_t *out);

/* Two rows if the band has them, one row otherwise.
 * `n_lines` is set to 1 or 2 and the caller must not read out[1] when it
 * is 1.  Returns -1 when neither shape fits (drop the frame). */
int mrz_ocr_split_lines_n(const uint8_t *bin, int W, int H,
                          mrz_ocr_rect_t *out, int *n_lines);
int mrz_ocr_segment_line(const uint8_t *bin, int W, int H,
                         mrz_ocr_rect_t *chars, int max_chars);

/* Classify the MRZ document layout (TD3 vs CS/CD vs TD1) using
 * aspect-ratio prior + dual-hypothesis grid valley energy.
 * binaryBottomLine: the bottom line's binary image (foreground=1).
 * lineW/lineH: dimensions of the bottom line.
 * stride: row stride of binaryBottomLine.
 * allCandidateLines: all candidate line rects found by split_lines.
 * numLines: number of candidate lines. */
mrz_doc_layout_t mrz_classify_layout(const uint8_t *binaryBottomLine,
                                     int lineW, int lineH, int stride,
                                     const mrz_ocr_rect_t *allCandidateLines,
                                     int numLines);

/* CNN-based recognition (alternative to the template-bank pipeline). */
mrz_ocr_status_t mrz_ocr_recognise_cnn(const face_image_t *img,
                                       mrz_ocr_result_t *out);

int mrz_ocr_last_seg_score(void);

/* Sequence-aware CRNN entry point. The sequence head replaces
 * only the per-character classifier and reuses the same geometry
 * front-end, so the ICAO/checksum post-processing is unchanged.
 * Falls back to the per-character CNN when no weights are compiled in. */
mrz_ocr_status_t mrz_ocr_recognise_crnn(const face_image_t *img,
                                       mrz_ocr_result_t *out);

/* Fused entry point (the shipped engine).
 *
 * Runs the per-character CNN head AND the CRNN sequence head over the
 * same features and combines them in log space with a confidence gate.
 * On a 3-fold grouped cross-validation over the 41 real photos the
 * fused engine was the only variant that beat the per-character CNN
 * (22/41 exact vs 21/41, 96.90% vs 96.78% character accuracy, and one
 * fewer checksum-valid-but-wrong output).  Falls back to the plain CNN
 * when no sequence weights are compiled in. */
mrz_ocr_status_t mrz_ocr_recognise_fuse(const face_image_t *img,
                                       mrz_ocr_result_t *out);

#ifdef __cplusplus
}
#endif

#endif /* MRZ_OCR_H */
