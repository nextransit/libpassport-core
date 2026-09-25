/* Unit tests for security primitives. */
#include "security.h"
#include "face.h"
#include "mrz.h"

#include <stdio.h>
#include <string.h>

static int g_fail = 0, g_total = 0;
static const char *g_cur = "";
#define EXPECT(c) do {                                                  \
    g_total++;                                                          \
    if (!(c)) {                                                         \
        g_fail++;                                                       \
        fprintf(stderr, "[FAIL] %s:%d in %s : %s\n",                    \
                __FILE__, __LINE__, g_cur, #c);                         \
    }                                                                   \
} while (0)
#define EXPECT_EQ_INT(a, b) do {                                        \
    long _a = (long)(a), _b = (long)(b);                                \
    g_total++;                                                          \
    if (_a != _b) {                                                     \
        g_fail++;                                                       \
        fprintf(stderr, "[FAIL] %s:%d in %s : %ld vs %ld\n",            \
                __FILE__, __LINE__, g_cur, _a, _b);                     \
    }                                                                   \
} while (0)

static int hex_eq(const uint8_t *buf, size_t n, const char *expect) {
    char hx[128];
    if (n > 63) n = 63;
    for (size_t i = 0; i < n; ++i) snprintf(hx + i*2, 3, "%02X", buf[i]);
    return strcmp(hx, expect) == 0;
}
static void expect_hex(const uint8_t *buf, size_t n, const char *expect,
                       const char *name) {
    g_total++;
    if (!hex_eq(buf, n, expect)) {
        g_fail++;
        char hx[128];
        for (size_t i = 0; i < n && i < 63; ++i) snprintf(hx + i*2, 3, "%02X", buf[i]);
        fprintf(stderr, "[FAIL] %s : %s\n", name, hx);
    }
}
#define EXPECT_EQ_HEX20(buf, expect) expect_hex((buf), 20, (expect), #buf)
#define EXPECT_EQ_HEX16(buf, expect) expect_hex((buf), 16, (expect), #buf)
#define EXPECT_EQ_HEX8(buf, expect)  expect_hex((buf), 8,  (expect), #buf)

#define RUN(fn) do { g_cur = #fn; fn(); } while (0)

static void test_sha1_known_vector(void) {
    uint8_t h[20];
    sha1((const uint8_t *)"abc", 3, h);
    EXPECT_EQ_HEX20(h, "A9993E364706816ABA3E25717850C26C9CD0D89D");
}

static void test_sha1_long_vector(void) {
    const char *s = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    uint8_t h[20];
    sha1((const uint8_t *)s, strlen(s), h);
    EXPECT_EQ_HEX20(h, "84983E441C3BD26EBAAE4AA1F95129E5E54670F1");
}

static void test_des_known_vector(void) {
    /* NIST SP 800-17 official vectors. */
    uint8_t k0[8] = {0}, p0[8] = {0}, c0[8];
    des_ecb_encrypt(k0, p0, c0);
    EXPECT_EQ_HEX8(c0, "8CA64DE9C1B123A7");
    uint8_t back0[8];
    des_ecb_decrypt(k0, c0, back0);
    EXPECT_EQ_INT(memcmp(back0, p0, 8), 0);
    uint8_t k1[8] = {0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF};
    uint8_t c1[8];
    des_ecb_encrypt(k1, p0, c1);
    EXPECT_EQ_HEX8(c1, "CAAAAF4DEAF1DBAE");
    uint8_t k2[8] = {0x01,0x23,0x45,0x67,0x89,0xAB,0xCD,0xEF};
    uint8_t c2[8];
    des_ecb_encrypt(k2, p0, c2);
    EXPECT_EQ_HEX8(c2, "D5D44FF720683D0D");
}

static void test_aes128_known_vector(void) {
    uint8_t k[16]  = {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
                     0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
    uint8_t in[16] = {0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,
                     0x88,0x99,0xAA,0xBB,0xCC,0xDD,0xEE,0xFF};
    uint8_t out[16];
    aes128_ecb_encrypt(k, in, out);
    EXPECT_EQ_HEX16(out, "69C4E0D86A7B0430D8CDB78070B4C55A");
}

static void test_aes128_cbc_roundtrip(void) {
    uint8_t k[16] = {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
                     0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
    uint8_t iv[16] = {0};
    uint8_t plain[16];
    for (int i = 0; i < 16; ++i) plain[i] = (uint8_t)(i * 7 + 3);
    uint8_t enc[16];
    uint8_t iv_save[16];
    memcpy(iv_save, iv, 16);
    aes128_cbc_encrypt(k, iv, plain, enc, 16);
    uint8_t back[16];
    memcpy(iv, iv_save, 16);
    aes128_cbc_decrypt(k, iv, enc, back, 16);
    EXPECT_EQ_INT(memcmp(back, plain, 16), 0);
}

static void test_mac3_deterministic(void) {
    uint8_t key[16] = {0x01,0x23,0x45,0x67,0x89,0xAB,0xCD,0xEF,
                       0xFE,0xDC,0xBA,0x98,0x76,0x54,0x32,0x10};
    const char *msg = "hello world";
    uint8_t mac1[8], mac2[8];
    mac3_des(key, (const uint8_t *)msg, strlen(msg), mac1);
    mac3_des(key, (const uint8_t *)msg, strlen(msg), mac2);
    EXPECT_EQ_INT(memcmp(mac1, mac2, 8), 0);
}

static void test_bac_derive_keys(void) {
    bac_keys_t k1 = bac_derive_keys("L898902C3", "690806", "940623");
    int zero = 1;
    for (int i = 0; i < 16; ++i) if (k1.k_enc[i] || k1.k_mac[i]) { zero = 0; break; }
    EXPECT_EQ_INT(zero, 0);
    bac_keys_t k2 = bac_derive_keys("L898902C3", "690806", "940623");
    EXPECT_EQ_INT(memcmp(k1.k_enc, k2.k_enc, 16), 0);
    EXPECT_EQ_INT(memcmp(k1.k_mac, k2.k_mac, 16), 0);
}

static void test_crosscheck_match(void) {
    cross_modal_report_t r = security_cross_check(
        "P<UTOERIKSSON<ANNA<MARIA<<<<<<<<<<<<<<<<<<<<\n"
        "L898902C36UTO6908061F9406236ZE184226B<<<<<18\n",
        "ERIKSSON ANNA MARIA, UTO, Passport L898902C3, "
        "Born 1969-08-06, Exp 1994-06-23");
    EXPECT_EQ_INT(r.passport_no, AC_FIELD_OK);
    EXPECT_EQ_INT(r.birth_date,  AC_FIELD_OK);
    EXPECT_EQ_INT(r.expiry_date, AC_FIELD_OK);
    EXPECT_EQ_INT(r.nationality, AC_FIELD_OK);
    EXPECT_EQ_INT(r.name,        AC_FIELD_OK);
}

static void test_crosscheck_mismatch(void) {
    cross_modal_report_t r = security_cross_check(
        "P<UTOERIKSSON<ANNA<MARIA<<<<<<<<<<<<<<<<<<<<\n"
        "L898902C36UTO6908061F9406236ZE184226B<<<<<18\n",
        "ERIKSSON ANNA MARIA, UTO, Passport X99999999");
    EXPECT_EQ_INT(r.passport_no, AC_FIELD_MISMATCH);
}

int main(void) {

    RUN(test_sha1_known_vector);
    RUN(test_sha1_long_vector);
    RUN(test_des_known_vector);
    RUN(test_aes128_known_vector);
    RUN(test_aes128_cbc_roundtrip);
    RUN(test_mac3_deterministic);
    RUN(test_bac_derive_keys);
    RUN(test_crosscheck_match);
    RUN(test_crosscheck_mismatch);
    fprintf(stderr, "\n[TEST] %d total, %d failed\n", g_total, g_fail);
    return g_fail == 0 ? 0 : 1;
}
