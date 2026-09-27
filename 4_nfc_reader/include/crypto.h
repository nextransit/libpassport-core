#ifndef CRYPTO_H
#define CRYPTO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* SHA-1 (RFC 3174). */
void sha1(const uint8_t *in, size_t len, uint8_t out[20]);

/* Single DES block (FIPS 46-3). key is 8 bytes; enc != 0 encrypts. */
void des_block(const uint8_t key[8], const uint8_t in[8], uint8_t out[8],
               int enc);

/* 3DES-EDE block. key is 16 bytes (K1||K2, K3 = K1). */
void tdes_ede_block(const uint8_t key[16], const uint8_t in[8],
                    uint8_t out[8], int enc);

/* 3DES-EDE-CBC. len must be a multiple of 8. IV is 8 bytes. */
void tdes_cbc_encrypt(const uint8_t key[16], const uint8_t iv[8],
                      const uint8_t *in, size_t len, uint8_t *out);
void tdes_cbc_decrypt(const uint8_t key[16], const uint8_t iv[8],
                      const uint8_t *in, size_t len, uint8_t *out);

/* 3DES-EDE-CBC-MAC with zero IV (ICAO 9303 uses the last block).
 * If len is not a multiple of 8, ISO 9797-1 padding method 2 is applied
 * internally (0x80 0x00 ...). */
void tdes_cbc_mac(const uint8_t key[16], const uint8_t *in, size_t len,
                  uint8_t out[8]);

/* Cryptographically-strong random bytes (/dev/urandom). Returns 0 on ok. */
int rand_bytes(uint8_t *out, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* CRYPTO_H */