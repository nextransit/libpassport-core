/* Standalone verification CLI for crypto.c / bac.c primitives.
 * Build (from repo root):
 *   gcc -I 4_nfc_reader/include 4_nfc_reader/src/crypto.c \
 *       4_nfc_reader/src/bac.c tests/test_crypto_cli.c -o /tmp/test_crypto_cli
 * Usage: test_crypto_cli <cmd> <args>  -- prints hex result on stdout.
 */
#include "crypto.h"
#include "bac.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int hex_to_buf(const char *hex, uint8_t *out, size_t cap) {
    size_t n = strlen(hex);
    if (n % 2 != 0) return -1;
    size_t bytes = n / 2;
    if (bytes > cap) return -1;
    for (size_t i = 0; i < bytes; ++i) {
        char hi = hex[i * 2], lo = hex[i * 2 + 1];
        int h = (hi >= '0' && hi <= '9') ? hi - '0' :
                (hi >= 'a' && hi <= 'f') ? 10 + hi - 'a' :
                (hi >= 'A' && hi <= 'F') ? 10 + hi - 'A' : -1;
        int l = (lo >= '0' && lo <= '9') ? lo - '0' :
                (lo >= 'a' && lo <= 'f') ? 10 + lo - 'a' :
                (lo >= 'A' && lo <= 'F') ? 10 + lo - 'A' : -1;
        if (h < 0 || l < 0) return -1;
        out[i] = (uint8_t)((h << 4) | l);
    }
    return (int)bytes;
}

static void print_hex(const uint8_t *b, size_t n) {
    for (size_t i = 0; i < n; ++i) printf("%02X", b[i]);
    printf("\n");
}

int main(int argc, char **argv) {
    if (argc < 2) return 2;
    const char *cmd = argv[1];

    if (strcmp(cmd, "sha1") == 0 && argc == 3) {
        uint8_t out[20];
        size_t n = strlen(argv[2]);
        sha1((const uint8_t *)argv[2], n, out);
        print_hex(out, 20);
        return 0;
    }
    if (strcmp(cmd, "tdes") == 0 && argc == 5) {
        uint8_t key[16], in[8], out[8];
        if (hex_to_buf(argv[2], key, 16) != 16) return 2;
        if (hex_to_buf(argv[3], in, 8) != 8) return 2;
        tdes_ede_block(key, in, out, atoi(argv[4]) ? 1 : 0);
        print_hex(out, 8);
        return 0;
    }
    if (strcmp(cmd, "cbc") == 0 && argc == 6) {
        uint8_t key[16], iv[8], in[256], out[256];
        int inlen = hex_to_buf(argv[3], in, 256);
        if (hex_to_buf(argv[2], key, 16) != 16) return 2;
        if (hex_to_buf(argv[4], iv, 8) != 8) return 2;
        if (inlen < 0 || inlen % 8 != 0) return 2;
        if (atoi(argv[5])) tdes_cbc_encrypt(key, iv, in, (size_t)inlen, out);
        else               tdes_cbc_decrypt(key, iv, in, (size_t)inlen, out);
        print_hex(out, (size_t)inlen);
        return 0;
    }
    if (strcmp(cmd, "mac") == 0 && argc == 4) {
        uint8_t key[16], in[256], out[8];
        int inlen = hex_to_buf(argv[3], in, 256);
        if (hex_to_buf(argv[2], key, 16) != 16) return 2;
        if (inlen < 0) return 2;
        tdes_cbc_mac(key, in, (size_t)inlen, out);
        print_hex(out, 8);
        return 0;
    }
    if (strcmp(cmd, "derive") == 0 && argc == 3) {
        uint8_t kseed[16], kenc[16], kmac[16];
        bac_derive_keys(argv[2], kseed, kenc, kmac);
        print_hex(kseed, 16);
        print_hex(kenc, 16);
        print_hex(kmac, 16);
        return 0;
    }
    if (strcmp(cmd, "mrzinfo") == 0 && argc == 3) {
        char out[BAC_MRZ_INFO_LEN + 1];
        if (bac_mrz_info(argv[2], out) != 0) return 2;
        printf("%s\n", out);
        return 0;
    }
    fprintf(stderr, "usage: test_crypto_cli sha1|tdes|cbc|mac|derive|mrzinfo\n");
    return 2;
}