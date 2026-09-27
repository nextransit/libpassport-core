/* Minimal, dependency-free crypto primitives for the BAC eMRTD flow:
 * SHA-1, single DES, 3DES-EDE, 3DES-CBC and 3DES-CBC-MAC.
 * All self-contained; verified against RFC/FIPS/NIST test vectors in
 * tests/test_crypto.py.
 */
#include "crypto.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ----------------------------- SHA-1 ----------------------------------- */

static uint32_t rotl32(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

void sha1(const uint8_t *in, size_t len, uint8_t out[20]) {
    uint32_t h0 = 0x67452301, h1 = 0xEFCDAB89;
    uint32_t h2 = 0x98BADCFE, h3 = 0x10325476, h4 = 0xC3D2E1F0;

    size_t padlen = ((len + 8) / 64 + 1) * 64;
    uint8_t *buf = (uint8_t *)calloc(padlen, 1);
    if (!buf) return;
    memcpy(buf, in, len);
    buf[len] = 0x80;
    uint64_t bits = (uint64_t)len * 8;
    for (int i = 0; i < 8; ++i)
        buf[padlen - 1 - i] = (uint8_t)(bits >> (8 * i));

    for (size_t off = 0; off < padlen; off += 64) {
        uint32_t w[80];
        for (int i = 0; i < 16; ++i)
            w[i] = ((uint32_t)buf[off + i * 4] << 24) |
                   ((uint32_t)buf[off + i * 4 + 1] << 16) |
                   ((uint32_t)buf[off + i * 4 + 2] << 8) |
                   (uint32_t)buf[off + i * 4 + 3];
        for (int i = 16; i < 80; ++i)
            w[i] = rotl32(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);

        uint32_t a = h0, b = h1, c = h2, d = h3, e = h4;
        for (int i = 0; i < 80; ++i) {
            uint32_t f, k;
            if (i < 20)      { f = (b & c) | (~b & d); k = 0x5A827999; }
            else if (i < 40) { f = b ^ c ^ d;          k = 0x6ED9EBA1; }
            else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDC; }
            else             { f = b ^ c ^ d;          k = 0xCA62C1D6; }
            uint32_t t = rotl32(a, 5) + f + e + k + w[i];
            e = d; d = c; c = rotl32(b, 30); b = a; a = t;
        }
        h0 += a; h1 += b; h2 += c; h3 += d; h4 += e;
    }
    free(buf);

    uint32_t h[5] = {h0, h1, h2, h3, h4};
    for (int i = 0; i < 5; ++i) {
        out[i * 4]     = (uint8_t)(h[i] >> 24);
        out[i * 4 + 1] = (uint8_t)(h[i] >> 16);
        out[i * 4 + 2] = (uint8_t)(h[i] >> 8);
        out[i * 4 + 3] = (uint8_t)(h[i]);
    }
}

/* ------------------------------- DES ----------------------------------- */

static const uint8_t IPTAB[64] = {
    58,50,42,34,26,18,10,2, 60,52,44,36,28,20,12,4,
    62,54,46,38,30,22,14,6, 64,56,48,40,32,24,16,8,
    57,49,41,33,25,17,9,1,  59,51,43,35,27,19,11,3,
    61,53,45,37,29,21,13,5, 63,55,47,39,31,23,15,7
};
static const uint8_t FPTAB[64] = {
    40,8,48,16,56,24,64,32, 39,7,47,15,55,23,63,31,
    38,6,46,14,54,22,62,30, 37,5,45,13,53,21,61,29,
    36,4,44,12,52,20,60,28, 35,3,43,11,51,19,59,27,
    34,2,42,10,50,18,58,26, 33,1,41,9,49,17,57,25
};
static const uint8_t ETAB[48] = {
    32,1,2,3,4,5, 4,5,6,7,8,9, 8,9,10,11,12,13, 12,13,14,15,16,17,
    16,17,18,19,20,21, 20,21,22,23,24,25, 24,25,26,27,28,29, 28,29,30,31,32,1
};
static const uint8_t PTAB[32] = {
    16,7,20,21,29,12,28,17, 1,15,23,26,5,18,31,10,
    2,8,24,14,32,27,3,9, 19,13,30,6,22,11,4,25
};
static const uint8_t PC1TAB[56] = {
    57,49,41,33,25,17,9, 1,58,50,42,34,26,18,
    10,2,59,51,43,35,27, 19,11,3,60,52,44,36,
    63,55,47,39,31,23,15, 7,62,54,46,38,30,22,
    14,6,61,53,45,37,29, 21,13,5,28,20,12,4
};
static const uint8_t PC2TAB[48] = {
    14,17,11,24,1,5, 3,28,15,6,21,10,
    23,19,12,4,26,8, 16,7,27,20,13,2,
    41,52,31,37,47,55, 30,40,51,45,33,48,
    44,49,39,56,34,53, 46,42,50,36,29,32
};
static const uint8_t SHIFTS[16] = {1,1,2,2,2,2,2,2,1,2,2,2,2,2,2,1};

static const uint8_t SBOX[8][64] = {
    {14,4,13,1,2,15,11,8,3,10,6,12,5,9,0,7,
     0,15,7,4,14,2,13,1,10,6,12,11,9,5,3,8,
     4,1,14,8,13,6,2,11,15,12,9,7,3,10,5,0,
     15,12,8,2,4,9,1,7,5,11,3,14,10,0,6,13},
    {15,1,8,14,6,11,3,4,9,7,2,13,12,0,5,10,
     3,13,4,7,15,2,8,14,12,0,1,10,6,9,11,5,
     0,14,7,11,10,4,13,1,5,8,12,6,9,3,2,15,
     13,8,10,1,3,15,4,2,11,6,7,12,0,5,14,9},
    {10,0,9,14,6,3,15,5,1,13,12,7,11,4,2,8,
     13,7,0,9,3,4,6,10,2,8,5,14,12,11,15,1,
     13,6,4,9,8,15,3,0,11,1,2,12,5,10,14,7,
     1,10,13,0,6,9,8,7,4,15,14,3,11,5,2,12},
    {7,13,14,3,0,6,9,10,1,2,8,5,11,12,4,15,
     13,8,11,5,6,15,0,3,4,7,2,12,1,10,14,9,
     10,6,9,0,12,11,7,13,15,1,3,14,5,2,8,4,
     3,15,0,6,10,1,13,8,9,4,5,11,12,7,2,14},
    {2,12,4,1,7,10,11,6,8,5,3,15,13,0,14,9,
     14,11,2,12,4,7,13,1,5,0,15,10,3,9,8,6,
     4,2,1,11,10,13,7,8,15,9,12,5,6,3,0,14,
     11,8,12,7,1,14,2,13,6,15,0,9,10,4,5,3},
    {12,1,10,15,9,2,6,8,0,13,3,4,14,7,5,11,
     10,15,4,2,7,12,9,5,6,1,13,14,0,11,3,8,
     9,14,15,5,2,8,12,3,7,0,4,10,1,13,11,6,
     4,3,2,12,9,5,15,10,11,14,1,7,6,0,8,13},
    {4,11,2,14,15,0,8,13,3,12,9,7,5,10,6,1,
     13,0,11,7,4,9,1,10,14,3,5,12,2,15,8,6,
     1,4,11,13,12,3,7,14,10,15,6,8,0,5,9,2,
     6,11,13,8,1,4,10,7,9,5,0,15,14,2,3,12},
    {13,2,8,4,6,15,11,1,10,9,3,14,5,0,12,7,
     1,15,13,8,10,3,7,4,12,5,6,11,0,14,9,2,
     7,11,4,1,9,12,14,2,0,6,10,13,15,3,5,8,
     2,1,14,7,4,10,8,13,15,12,9,0,3,5,6,11}
};

static uint64_t permute(uint64_t v, const uint8_t *tab, int n) {
    uint64_t r = 0;
    for (int i = 0; i < n; ++i) {
        int pos = tab[i]; /* 1-based, MSB-first */
        r = (r << 1) | ((v >> (64 - pos)) & 1);
    }
    return r;
}

static void schedule(const uint8_t key[8], uint64_t subkeys[16]) {
    uint64_t k = 0;
    for (int i = 0; i < 8; ++i)
        k = (k << 8) | key[i];
    uint64_t cd = permute(k, PC1TAB, 56); /* 56 bits */
    uint32_t C = (uint32_t)(cd >> 28) & 0x0FFFFFFF;
    uint32_t D = (uint32_t)(cd) & 0x0FFFFFFF;
    for (int r = 0; r < 16; ++r) {
        int s = SHIFTS[r];
        C = ((C << s) | (C >> (28 - s))) & 0x0FFFFFFF;
        D = ((D << s) | (D >> (28 - s))) & 0x0FFFFFFF;
        uint64_t c56 = ((uint64_t)C << 28) | D;
        subkeys[r] = permute(c56 << 8, PC2TAB, 48);
    }
}

static void des_crypt(const uint64_t subkeys[16], const uint8_t in[8],
                      uint8_t out[8], int enc) {
    uint64_t b = 0;
    for (int i = 0; i < 8; ++i) b = (b << 8) | in[i];
    b = permute(b, IPTAB, 64);
    uint32_t L = (uint32_t)(b >> 32);
    uint32_t R = (uint32_t)(b & 0xFFFFFFFF);

    for (int r = 0; r < 16; ++r) {
        int idx = enc ? r : 15 - r;
        uint64_t er = permute((uint64_t)R << 32, ETAB, 48);
        er ^= subkeys[idx];
        uint32_t s = 0;
        for (int i = 0; i < 8; ++i) {
            uint8_t six = (uint8_t)((er >> (42 - 6 * i)) & 0x3F);
            uint8_t row = (uint8_t)(((six >> 4) & 2) | (six & 1));
            uint8_t col = (uint8_t)((six >> 1) & 0x0F);
            s = (s << 4) | SBOX[i][row * 16 + col];
        }
        uint32_t p = (uint32_t)permute((uint64_t)s << 32, PTAB, 32);
        uint32_t nr = L ^ p;
        L = R; R = nr;
    }
    b = ((uint64_t)R << 32) | L; /* swap halves */
    b = permute(b, FPTAB, 64);
    for (int i = 7; i >= 0; --i) { out[i] = (uint8_t)(b & 0xFF); b >>= 8; }
}

void des_block(const uint8_t key[8], const uint8_t in[8], uint8_t out[8],
               int enc) {
    uint64_t subkeys[16];
    schedule(key, subkeys);
    des_crypt(subkeys, in, out, enc);
}

void tdes_ede_block(const uint8_t key[16], const uint8_t in[8],
                    uint8_t out[8], int enc) {
    uint8_t k1[8], k2[8], t[8];
    memcpy(k1, key, 8);
    memcpy(k2, key + 8, 8);
    uint64_t s1[16], s2[16];
    schedule(k1, s1);
    schedule(k2, s2);
    if (enc) {
        des_crypt(s1, in, out, 1);
        des_crypt(s2, out, t, 0);
        des_crypt(s1, t, out, 1);
    } else {
        des_crypt(s1, in, out, 0);
        des_crypt(s2, out, t, 1);
        des_crypt(s1, t, out, 0);
    }
}

static void xor8(uint8_t *a, const uint8_t *b) {
    for (int i = 0; i < 8; ++i) a[i] ^= b[i];
}

void tdes_cbc_encrypt(const uint8_t key[16], const uint8_t iv[8],
                      const uint8_t *in, size_t len, uint8_t *out) {
    uint8_t prev[8];
    memcpy(prev, iv, 8);
    for (size_t off = 0; off < len; off += 8) {
        uint8_t blk[8];
        memcpy(blk, in + off, 8);
        xor8(blk, prev);
        tdes_ede_block(key, blk, out + off, 1);
        memcpy(prev, out + off, 8);
    }
}

void tdes_cbc_decrypt(const uint8_t key[16], const uint8_t iv[8],
                      const uint8_t *in, size_t len, uint8_t *out) {
    uint8_t prev[8];
    memcpy(prev, iv, 8);
    for (size_t off = 0; off < len; off += 8) {
        uint8_t blk[8];
        memcpy(blk, in + off, 8);
        tdes_ede_block(key, blk, out + off, 0);
        xor8(out + off, prev);
        memcpy(prev, blk, 8);
    }
}

void tdes_cbc_mac(const uint8_t key[16], const uint8_t *in, size_t len,
                  uint8_t out[8]) {
    uint8_t buf[512];
    size_t n = len;
    if (n % 8 != 0) {
        n = ((len / 8) + 1) * 8;
        if (n > sizeof(buf)) n = sizeof(buf);
        memset(buf, 0, n);
        memcpy(buf, in, len);
        buf[len] = 0x80;
        in = buf;
    }
    uint8_t iv[8] = {0};
    size_t padded = (n % 8 == 0) ? n : ((n / 8) + 1) * 8;
    if (padded > sizeof(buf)) padded = sizeof(buf);
    uint8_t tmp[512];
    tdes_cbc_encrypt(key, iv, in, padded, tmp);
    memcpy(out, tmp + padded - 8, 8);
}

int rand_bytes(uint8_t *out, size_t n) {
    FILE *f = fopen("/dev/urandom", "rb");
    if (!f) return -1;
    size_t got = fread(out, 1, n, f);
    fclose(f);
    return got == n ? 0 : -1;
}