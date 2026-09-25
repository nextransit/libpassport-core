/* DES + 3DES wrappers around CommonCrypto (macOS native).
 * CommonCrypto is deprecated but still functional and gives us
 * bit-exact NIST test vectors; using it here keeps the codebase
 * portable and provably correct. */
#include "security.h"

#include <CommonCrypto/CommonCryptor.h>
#include <string.h>

void des_ecb_encrypt(const uint8_t key[8], const uint8_t in[8],
                     uint8_t out[8]) {
    size_t n = 0;
    CCCrypt(kCCEncrypt, kCCAlgorithmDES, kCCOptionECBMode,
            key, 8, NULL, in, 8, out, 8, &n);
}

void des_ecb_decrypt(const uint8_t key[8], const uint8_t in[8],
                     uint8_t out[8]) {
    size_t n = 0;
    CCCrypt(kCCDecrypt, kCCAlgorithmDES, kCCOptionECBMode,
            key, 8, NULL, in, 8, out, 8, &n);
}

/* 3DES-EDE2 (K1 || K2 with K1 == K3) ECB block. */
void des_ecb_encrypt_3des(const uint8_t key[16], const uint8_t in[8],
                           uint8_t out[8]) {
    uint8_t tmp[8];
    des_ecb_encrypt(key, in, tmp);
    des_ecb_decrypt(key + 8, tmp, out);
    des_ecb_encrypt(key, out, tmp);
    memcpy(out, tmp, 8);
}

void des_ecb_decrypt_3des(const uint8_t key[16], const uint8_t in[8],
                           uint8_t out[8]) {
    uint8_t tmp[8];
    des_ecb_decrypt(key, in, tmp);
    des_ecb_encrypt(key + 8, tmp, out);
    des_ecb_decrypt(key, out, tmp);
    memcpy(out, tmp, 8);
}

/* 3DES-EDE2 CBC implemented as ECB + XOR chaining. */
void des3_cbc_encrypt(const uint8_t key[16], uint8_t iv[8],
                      const uint8_t *in, uint8_t *out, size_t len) {
    for (size_t i = 0; i < len; i += 8) {
        uint8_t blk[8];
        for (int j = 0; j < 8; ++j) blk[j] = in[i + j] ^ iv[j];
        des_ecb_encrypt_3des(key, blk, out + i);
        memcpy(iv, out + i, 8);
    }
}

void des3_cbc_decrypt(const uint8_t key[16], uint8_t iv[8],
                      const uint8_t *in, uint8_t *out, size_t len) {
    for (size_t i = 0; i < len; i += 8) {
        uint8_t blk[8];
        des_ecb_decrypt_3des(key, in + i, blk);
        for (int j = 0; j < 8; ++j) out[i + j] = blk[j] ^ iv[j];
        memcpy(iv, in + i, 8);
    }
}
