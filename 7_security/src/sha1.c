/* SHA-1 (RFC 3174). */
#include "security.h"

#include <string.h>

#define ROL32(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

static void sha1_compress(uint32_t state[5], const uint8_t block[64]) {
    uint32_t w[80];
    for (int i = 0; i < 16; ++i) {
        w[i] = ((uint32_t)block[i*4] << 24) |
               ((uint32_t)block[i*4+1] << 16) |
               ((uint32_t)block[i*4+2] << 8) |
               ((uint32_t)block[i*4+3]);
    }
    for (int i = 16; i < 80; ++i) {
        w[i] = ROL32(w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16], 1);
    }
    uint32_t a = state[0], b = state[1], c = state[2],
             d = state[3], e = state[4];
    for (int i = 0; i < 80; ++i) {
        uint32_t f, k;
        if (i < 20)      { f = (b & c) | (~b & d);       k = 0x5A827999; }
        else if (i < 40) { f = b ^ c ^ d;                k = 0x6ED9EBA1; }
        else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDC; }
        else             { f = b ^ c ^ d;                k = 0xCA62C1D6; }
        uint32_t t = ROL32(a, 5) + f + e + k + w[i];
        e = d; d = c; c = ROL32(b, 30); b = a; a = t;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d; state[4] += e;
}

void sha1(const uint8_t *data, size_t n, uint8_t out[20]) {
    uint32_t state[5] = {
        0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0
    };
    uint64_t bitlen = (uint64_t)n * 8;
    uint8_t block[64];
    size_t i = 0;
    while (n - i >= 64) {
        sha1_compress(state, data + i);
        i += 64;
    }
    size_t rem = n - i;
    memcpy(block, data + i, rem);
    block[rem] = 0x80;
    if (rem >= 56) {
        memset(block + rem + 1, 0, 64 - rem - 1);
        sha1_compress(state, block);
        memset(block, 0, 56);
    } else {
        memset(block + rem + 1, 0, 56 - rem - 1);
    }
    for (int j = 0; j < 8; ++j) {
        block[56 + j] = (uint8_t)(bitlen >> (56 - 8 * j));
    }
    sha1_compress(state, block);
    for (int j = 0; j < 5; ++j) {
        out[j*4]     = (uint8_t)(state[j] >> 24);
        out[j*4 + 1] = (uint8_t)(state[j] >> 16);
        out[j*4 + 2] = (uint8_t)(state[j] >> 8);
        out[j*4 + 3] = (uint8_t)(state[j]);
    }
}
