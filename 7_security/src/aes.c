/* AES-128 wrappers around CommonCrypto (ECB only; CBC is built on
 * top of ECB with explicit IV chaining). */
#include "security.h"

#include <CommonCrypto/CommonCryptor.h>
#include <string.h>

void aes128_ecb_encrypt(const uint8_t key[16], const uint8_t in[16],
                        uint8_t out[16]) {
    size_t n = 0;
    CCCrypt(kCCEncrypt, kCCAlgorithmAES, kCCOptionECBMode,
            key, 16, NULL, in, 16, out, 16, &n);
}

void aes128_ecb_decrypt(const uint8_t key[16], const uint8_t in[16],
                        uint8_t out[16]) {
    size_t n = 0;
    CCCrypt(kCCDecrypt, kCCAlgorithmAES, kCCOptionECBMode,
            key, 16, NULL, in, 16, out, 16, &n);
}

void aes128_cbc_encrypt(const uint8_t key[16], uint8_t iv[16],
                        const uint8_t *in, uint8_t *out, size_t len) {
    for (size_t i = 0; i < len; i += 16) {
        uint8_t blk[16];
        for (int j = 0; j < 16; ++j) blk[j] = in[i + j] ^ iv[j];
        aes128_ecb_encrypt(key, blk, out + i);
        memcpy(iv, out + i, 16);
    }
}

void aes128_cbc_decrypt(const uint8_t key[16], uint8_t iv[16],
                        const uint8_t *in, uint8_t *out, size_t len) {
    for (size_t i = 0; i < len; i += 16) {
        uint8_t blk[16];
        aes128_ecb_decrypt(key, in + i, blk);
        for (int j = 0; j < 16; ++j) out[i + j] = blk[j] ^ iv[j];
        memcpy(iv, in + i, 16);
    }
}
