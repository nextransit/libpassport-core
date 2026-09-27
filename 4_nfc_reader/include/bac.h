#ifndef BAC_H
#define BAC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* MRZ_information = doc number(9)+ck + DOB(6)+ck + expiry(6)+ck = 24 chars. */
#define BAC_MRZ_INFO_LEN 24

/* Per-run BAC context (keys, random values, session keys). */
typedef struct {
    char     mrz_info[BAC_MRZ_INFO_LEN + 1];
    uint8_t  kseed[16];
    uint8_t  kenc[16];
    uint8_t  kmac[16];
    /* Session keys after mutual auth. */
    uint8_t  ksenc[16];
    uint8_t  ksmac[16];
    uint8_t  rnd_icc[8];
    uint8_t  rnd_ifd[8];
    uint8_t  kifd[16];
    uint8_t  kicc[16];
    int      authed;
} bac_ctx_t;

/* Extract the 24-char MRZ_information from MRZ line2 (>=28 chars).
 * Returns 0 on success, -1 on malformed input. */
int bac_mrz_info(const char *line2, char out[BAC_MRZ_INFO_LEN + 1]);

/* Derive Kseed, Kenc, Kmac from MRZ_information (ICAO 9303-11 9.8.1). */
void bac_derive_keys(const char *mrz_info, uint8_t kseed[16],
                     uint8_t kenc[16], uint8_t kmac[16]);

/* Step A: build the 40-byte MUTUAL AUTHENTICATE payload (E||MAC).
 * ctx->rnd_icc must already hold the GET CHALLENGE value. */
void bac_build_auth1(bac_ctx_t *ctx, uint8_t out[40]);

/* Step B: verify the 40-byte response, extract KICC and derive the
 * session keys. Returns 0 on success, -1 on mismatch/MAC failure. */
int bac_verify_auth2(bac_ctx_t *ctx, const uint8_t resp[40]);

/* Secure-messaging MAC over `len` bytes using the session MAC key. */
void bac_sm_mac(const uint8_t ksmac[16], const uint8_t *data, size_t len,
                uint8_t out[8]);

#ifdef __cplusplus
}
#endif

#endif /* BAC_H */