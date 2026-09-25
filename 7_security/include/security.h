#ifndef PASSPORT_SECURITY_H
#define PASSPORT_SECURITY_H

#include <stddef.h>
#include <stdint.h>

#include "mrz.h"   /* re-use mrz_td3_t for cross-modality check */
#include "face.h"  /* face_image_t for image-based checks */

#ifdef __cplusplus
extern "C" {
#endif

/* ----- Cryptographic primitives (pure C, no external deps). ----- */

/* SHA-1. Output is 20 bytes. */
void sha1(const uint8_t *data, size_t n, uint8_t out[20]);

/* Single DES (8-byte key, 8-byte block, ECB). */
void des_ecb_encrypt(const uint8_t key[8], const uint8_t in[8],
                      uint8_t out[8]);
void des_ecb_decrypt(const uint8_t key[8], const uint8_t in[8],
                      uint8_t out[8]);

/* 3DES-EDE2 CBC (16-byte key = K1||K2 with K1=K3 in EDE2 mode).
 * Encrypt/decrypt `len` bytes (must be a multiple of 8) starting from
 * `iv` (updated in place). */
void des3_cbc_encrypt(const uint8_t key[16], uint8_t iv[8],
                      const uint8_t *in, uint8_t *out, size_t len);
void des3_cbc_decrypt(const uint8_t key[16], uint8_t iv[8],
                      const uint8_t *in, uint8_t *out, size_t len);

/* AES-128 (ECB + CBC). */
void aes128_ecb_encrypt(const uint8_t key[16], const uint8_t in[16],
                        uint8_t out[16]);
void aes128_ecb_decrypt(const uint8_t key[16], const uint8_t in[16],
                        uint8_t out[16]);
void aes128_cbc_encrypt(const uint8_t key[16], uint8_t iv[16],
                        const uint8_t *in, uint8_t *out, size_t len);
void aes128_cbc_decrypt(const uint8_t key[16], uint8_t iv[16],
                        const uint8_t *in, uint8_t *out, size_t len);

/* MAC Algorithm 3 (ISO/IEC 9797-1) on top of DES (retail MAC).
 * Input may be any length; the output is 8 bytes.
 * Implementation uses the standard zero-pad-then-DES-CBC-then-DES-E-D
 * recipe. */
void mac3_des(const uint8_t key[16], const uint8_t *data, size_t n,
              uint8_t out[8]);

/* ----- ICAO BAC key derivation ----- */
/* Derive K_enc and K_mac from the MRZ (passport_no, birth, expiry).
 * The MRZ must already have its check digits -- the calling code
 * (the verifier pipeline) is expected to have read them.
 * Algorithm (ICAO Doc 9303 Part 11): SHA-1 over the concatenation of
 *   passport_no (9) + passport_no_check (1) +
 *   birth_yymmdd (6)  + birth_check (1) +
 *   expiry_yymmdd (6) + expiry_check (1)
 * Split the 20-byte digest into two 8-byte halves; XOR each half
 * with the constant 0x3736353433323130 0x4645444342414039 (high
 * parity byte adjusted version of "01234567" / "89ABCDEF") to
 * produce K_enc and K_mac. */
typedef struct {
    uint8_t k_enc[16];
    uint8_t k_mac[16];
} bac_keys_t;

bac_keys_t bac_derive_keys(const char *passport_no,
                           const char *birth_yymmdd,
                           const char *expiry_yymmdd);

/* ----- Cross-modality consistency check ----- */
typedef enum {
    AC_FIELD_OK         = 0,
    AC_FIELD_MISMATCH   = 1,
    AC_FIELD_MISSING    = 2,
} ac_match_t;

typedef struct {
    ac_match_t passport_no;
    ac_match_t birth_date;
    ac_match_t expiry_date;
    ac_match_t nationality;
    ac_match_t name;
    int        ok_count;
    int        fail_count;
} cross_modal_report_t;

/* Compare the OCR'd MRZ (or DG1 data) against the visual-inspection
 * zone text. Both `mrz` and `viz` are NUL-terminated strings. Either
 * can be NULL to indicate "missing"; in that case fields default to
 * MISSING. */
cross_modal_report_t security_cross_check(const char *mrz, const char *viz);

/* ----- Anti-counterfeit feature simulation ----- */

/* Simulate the B900 IR absorption check: the ratio of (dark pixel
 * density) under visible vs IR light. Genuine OCR-B ink is strongly
 * IR-absorbent, so the ratio should be close to 1.0. Forged ink
 * (laser/inkjet) usually scores < 0.6. We compute the ratio on
 * already-captured images. */
typedef struct {
    int    visible_dark_count;
    int    ir_dark_count;
    double ratio;          /* IR / VISIBLE */
    int    passes;         /* 1 if ratio >= 0.85, 0 otherwise */
} ir_absorption_t;

ir_absorption_t security_ir_absorption(const face_image_t *visible,
                                       const face_image_t *ir);

/* SSIM-style template matching for the IR watermark. We compare two
 * pre-aligned images with a 3x3 box filter and return [0..1]. */
typedef struct {
    double ssim;           /* 0..1 */
    int    passes;         /* ssim >= 0.70 */
} ssim_match_t;

ssim_match_t security_ssim(const face_image_t *a, const face_image_t *b);

/* UV fluorescence: paper brightness under UV must stay below a
 * threshold (security paper does not fluoresce). We measure mean
 * grayscale luminance. */
typedef struct {
    double mean_luminance;
    int    passes;         /* mean < 100 */
} uv_check_t;

uv_check_t security_uv_luminance(const face_image_t *uv);

#ifdef __cplusplus
}
#endif
#endif /* PASSPORT_SECURITY_H */
