/* BAC (Basic Access Control) cryptography for eMRTD reading.
 * Implements ICAO 9303-11 §9.8 (key derivation, mutual authentication,
 * session key agreement) and the secure-messaging MAC used afterwards.
 * Depends only on the primitives in crypto.c.
 */
#include "bac.h"
#include "crypto.h"

#include <string.h>

/* ---- MRZ_information: line2[0..9] + line2[13..19] + line2[21..27] ---- */
int bac_mrz_info(const char *line2, char out[BAC_MRZ_INFO_LEN + 1]) {
    if (!line2 || strlen(line2) < 28) return -1;
    static const int pos[] = {0,1,2,3,4,5,6,7,8,9, 13,14,15,16,17,18,19,
                              21,22,23,24,25,26,27};
    for (size_t i = 0; i < BAC_MRZ_INFO_LEN; ++i)
        out[i] = line2[pos[i]];
    out[BAC_MRZ_INFO_LEN] = '\0';
    return 0;
}

/* ---- Kseed = SHA-1(MRZ_information)[0..15] ----
 * Kenc = 3DES-EDE(Kseed, 0x...01) || 3DES-EDE(Kseed, 0x...02)
 * Kmac = 3DES-EDE(Kseed, 0x...03) || 3DES-EDE(Kseed, 0x...04)        */
void bac_derive_keys(const char *mrz_info, uint8_t kseed[16],
                     uint8_t kenc[16], uint8_t kmac[16]) {
    uint8_t h[20];
    sha1((const uint8_t *)mrz_info, strlen(mrz_info), h);
    memcpy(kseed, h, 16);
    for (int i = 0; i < 4; ++i) {
        uint8_t counter[8] = {0, 0, 0, 0, 0, 0, 0, (uint8_t)(i + 1)};
        uint8_t blk[8];
        tdes_ede_block(kseed, counter, blk, 1);
        if (i < 2)      memcpy(kenc + i * 8, blk, 8);
        else            memcpy(kmac + (i - 2) * 8, blk, 8);
    }
}

/* ---- Mutual authentication (9.8.2) ---- */
void bac_build_auth1(bac_ctx_t *ctx, uint8_t out[40]) {
    uint8_t zero[8] = {0};
    if (rand_bytes(ctx->rnd_ifd, 8) != 0)
        memset(ctx->rnd_ifd, 0x5A, 8);
    if (rand_bytes(ctx->kifd, 16) != 0)
        memset(ctx->kifd, 0x5A, 16);

    uint8_t S[32];
    memcpy(S, ctx->rnd_icc, 8);
    memcpy(S + 8, ctx->rnd_ifd, 8);
    memcpy(S + 16, ctx->kifd, 16);

    uint8_t E[32];
    tdes_cbc_encrypt(ctx->kenc, zero, S, 32, E);
    uint8_t M[8];
    tdes_cbc_mac(ctx->kmac, E, 32, M);

    memcpy(out, E, 32);
    memcpy(out + 32, M, 8);
}

int bac_verify_auth2(bac_ctx_t *ctx, const uint8_t resp[40]) {
    uint8_t zero[8] = {0};
    uint8_t E2[32], M2[8];
    memcpy(E2, resp, 32);
    memcpy(M2, resp + 32, 8);

    /* Check the response MAC first. */
    uint8_t mac[8];
    tdes_cbc_mac(ctx->kmac, E2, 32, mac);
    if (memcmp(mac, M2, 8) != 0) return -1;

    uint8_t P[32];
    tdes_cbc_decrypt(ctx->kenc, zero, E2, 32, P);
    /* P = rndIFD || rndICC || KICC */
    if (memcmp(P, ctx->rnd_ifd, 8) != 0) return -1;
    if (memcmp(P + 8, ctx->rnd_icc, 8) != 0) return -1;
    memcpy(ctx->kicc, P + 16, 16);

    /* KS = KIFD XOR KICC; KSenc = 3DES-CBC(Kenc, KS),
     * KSmac = 3DES-CBC(Kmac, KS), both with zero IV. */
    uint8_t ks[16];
    for (int i = 0; i < 16; ++i) ks[i] = ctx->kifd[i] ^ ctx->kicc[i];
    tdes_cbc_encrypt(ctx->kenc, zero, ks, 16, ctx->ksenc);
    tdes_cbc_encrypt(ctx->kmac, zero, ks, 16, ctx->ksmac);
    ctx->authed = 1;
    return 0;
}

void bac_sm_mac(const uint8_t ksmac[16], const uint8_t *data, size_t len,
                uint8_t out[8]) {
    tdes_cbc_mac(ksmac, data, len, out);
}