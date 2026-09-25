/* Unit tests for the anti-counterfeit verifier. */
#include "ac.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int g_fail = 0;
static int g_total = 0;
static const char *g_cur = "";

#define EXPECT(cond) do {                                              \
    g_total++;                                                          \
    if (!(cond)) {                                                      \
        g_fail++;                                                       \
        fprintf(stderr, "[FAIL] %s:%d in %s : %s\n",                    \
                __FILE__, __LINE__, g_cur, #cond);                      \
    }                                                                   \
} while (0)

#define EXPECT_EQ_INT(a, b) do {                                        \
    long _a = (long)(a), _b = (long)(b);                                \
    g_total++;                                                          \
    if (_a != _b) {                                                     \
        g_fail++;                                                       \
        fprintf(stderr, "[FAIL] %s:%d in %s : %s == %s (got %ld vs %ld)\n", \
                __FILE__, __LINE__, g_cur, #a, #b, _a, _b);             \
    }                                                                   \
} while (0)

#define RUN(fn) do { g_cur = #fn; fn(); } while (0)

/* The canonical sample we already round-tripped in the MRZ module. */
static const char *GOOD_MRZ =
    "P<UTOERIKSSON<ANNA<MARIA<<<<<<<<<<<<<<<<<<<<\n"
    "L898902C36UTO6908061F9406236ZE184226B<<<<<18\n";

/* Helper: tamper a single byte in a copy of GOOD_MRZ. */
static void tamper(char *buf, size_t idx, char c) { buf[idx] = c; }

static void test_good_passport(void) {
    ac_report_t r; ac_report_init(&r);
    ac_date_t today = {2026, 9, 25};
    ac_status_t s = ac_verify_td3(GOOD_MRZ, &today, &r);
    EXPECT_EQ_INT(s, (long)AC_OK);
    EXPECT_EQ_INT(r.fails, 0);
    /* Should have OK summary finding. */
    int saw_summary = 0;
    for (size_t i = 0; i < r.count; ++i)
        if (strcmp(r.items[i].name, "summary") == 0 &&
            r.items[i].severity == AC_SEVERITY_OK) saw_summary = 1;
    EXPECT(saw_summary);
    ac_report_free(&r);
}

static void test_check_digit_fail(void) {
    char bad[256];
    strcpy(bad, GOOD_MRZ);
    /* flip passport_no check digit (line2 position 9 in joined view,
     * which is at buffer index MRZ_TD3_LINE_LEN + 1 + 9 = 54) */
    tamper(bad, 54, (bad[54] == '6') ? '7' : '6');
    ac_report_t r; ac_report_init(&r);
    ac_date_t today = {2026, 9, 25};
    ac_status_t s = ac_verify_td3(bad, &today, &r);
    EXPECT(s != AC_OK);
    EXPECT(r.fails > 0);
    int saw_decode_fail = 0;
    for (size_t i = 0; i < r.count; ++i)
        if (strcmp(r.items[i].name, "decode") == 0 &&
            r.items[i].severity == AC_SEVERITY_FAIL) saw_decode_fail = 1;
    EXPECT(saw_decode_fail);
    ac_report_free(&r);
}

static void test_bad_doc_type(void) {
    char bad[256];
    strcpy(bad, GOOD_MRZ);
    /* Replace P< with Q< (invalid doc type). */
    tamper(bad, 0, 'Q');
    ac_report_t r; ac_report_init(&r);
    ac_date_t today = {2026, 9, 25};
    ac_status_t s = ac_verify_td3(bad, &today, &r);
    EXPECT(s != AC_OK);
    int saw_doc_type_fail = 0;
    for (size_t i = 0; i < r.count; ++i)
        if (strcmp(r.items[i].name, "doc_type") == 0 &&
            r.items[i].severity == AC_SEVERITY_FAIL) saw_doc_type_fail = 1;
    EXPECT(saw_doc_type_fail);
    ac_report_free(&r);
}

static void test_bad_nationality(void) {
    char bad[256];
    strcpy(bad, GOOD_MRZ);
    /* Inject lowercase char into nationality: position 55 (line2[10]).
     * Lowercase isn't in the MRZ alphabet so the decoder will fail. */
    tamper(bad, 55, 'u');
    ac_report_t r; ac_report_init(&r);
    ac_date_t today = {2026, 9, 25};
    ac_status_t s = ac_verify_td3(bad, &today, &r);
    EXPECT(s != AC_OK);
    ac_report_free(&r);
}

static void test_invalid_birth_month(void) {
    /* Construct a known-good-looking MRZ with month '13'. We override
     * just the YYMMDD positions while preserving check digits. The
     * decoder will accept (since check digits still match the encoded
     * values), but our verifier will catch the impossible date. */
    char bad[256];
    strcpy(bad, GOOD_MRZ);
    /* birth date at line2 positions 13..18 (buffer offset 58..63).
     * We replace '690806' with '691306' and recompute the check digit. */
    memcpy(bad + 58, "691306", 6);
    /* Recompute ck_birth: it is at buffer offset 64. */
    int ck = mrz_check_digit("691306", 6);
    bad[64] = (char)('0' + ck);
    /* Composite check is over line2 first 43 chars -> recompute. */
    int comp = mrz_td3_composite_check(bad + MRZ_TD3_LINE_LEN + 1);
    bad[MRZ_TD3_LINE_LEN + 1 + 43] = (char)('0' + comp);
    ac_report_t r; ac_report_init(&r);
    ac_date_t today = {2026, 9, 25};
    ac_status_t s = ac_verify_td3(bad, &today, &r);
    EXPECT(s != AC_OK);
    int saw_birth_fail = 0;
    for (size_t i = 0; i < r.count; ++i)
        if (strcmp(r.items[i].name, "birth_date") == 0 &&
            r.items[i].severity == AC_SEVERITY_FAIL) saw_birth_fail = 1;
    EXPECT(saw_birth_fail);
    ac_report_free(&r);
}

static void test_expiry_before_birth(void) {
    char bad[256];
    strcpy(bad, GOOD_MRZ);
    /* Make expiry < birth: change birth to 950101 and expiry to 950101
     * (equal) -> our rule says expiry must be > birth. */
    memcpy(bad + 58, "950101", 6);
    int ck = mrz_check_digit("950101", 6);
    bad[64] = (char)('0' + ck);
    memcpy(bad + 66, "950101", 6);
    int ck2 = mrz_check_digit("950101", 6);
    bad[72] = (char)('0' + ck2);
    int comp = mrz_td3_composite_check(bad + MRZ_TD3_LINE_LEN + 1);
    bad[MRZ_TD3_LINE_LEN + 1 + 43] = (char)('0' + comp);
    ac_report_t r; ac_report_init(&r);
    ac_date_t today = {2026, 9, 25};
    ac_status_t s = ac_verify_td3(bad, &today, &r);
    EXPECT(s != AC_OK);
    int saw_order_fail = 0;
    for (size_t i = 0; i < r.count; ++i)
        if (strcmp(r.items[i].name, "date_order") == 0 &&
            r.items[i].severity == AC_SEVERITY_FAIL) saw_order_fail = 1;
    EXPECT(saw_order_fail);
    ac_report_free(&r);
}

static void test_empty_personal_no_info(void) {
    char mrz[256];
    strcpy(mrz, GOOD_MRZ);
    /* Fill personal_no with '<'. ck_personal_no for 14 '<' is 0. */
    for (int i = 0; i < 14; ++i) mrz[73 + i] = '<';
    int ck = mrz_check_digit("<<<<<<<<<<<<<<", 14);
    mrz[87] = (char)('0' + ck);
    int comp = mrz_td3_composite_check(mrz + MRZ_TD3_LINE_LEN + 1);
    mrz[MRZ_TD3_LINE_LEN + 1 + 43] = (char)('0' + comp);
    ac_report_t r; ac_report_init(&r);
    ac_date_t today = {2026, 9, 25};
    ac_status_t s = ac_verify_td3(mrz, &today, &r);
    EXPECT_EQ_INT(s, (long)AC_OK);
    int saw_info = 0;
    for (size_t i = 0; i < r.count; ++i)
        if (strcmp(r.items[i].name, "personal_no") == 0 &&
            r.items[i].severity == AC_SEVERITY_INFO) saw_info = 1;
    EXPECT(saw_info);
    ac_report_free(&r);
}

static void test_low_alpha_name_warn(void) {
    /* Construct a name of nearly all '<' so the entropy check warns.
     * Line 1 layout: P<UTO + 'Z' + 38 '<'  = 44 chars. */
    char mrz[256];
    char l1[45], l2_good[45];
    memset(l1, '<', 44);
    memcpy(l1, "P<UTOZ", 6);
    l1[44] = '\0';
    memcpy(l2_good, "L898902C36UTO6908061F9406236ZE184226B<<<<<18", 44);
    l2_good[44] = '\0';
    snprintf(mrz, sizeof(mrz), "%s\n%s\n", l1, l2_good);
    ac_report_t r; ac_report_init(&r);
    ac_date_t today = {2026, 9, 25};
    ac_status_t s = ac_verify_td3(mrz, &today, &r);
    /* Decode will pass because composite check digit is unchanged,
     * but we should see a WARN for name_entropy. */
    int saw_warn = 0;
    for (size_t i = 0; i < r.count; ++i)
        if (strcmp(r.items[i].name, "name_entropy") == 0 &&
            r.items[i].severity == AC_SEVERITY_WARN) saw_warn = 1;
    EXPECT(saw_warn);
    (void)s;
    ac_report_free(&r);
}

int main(void) {
    RUN(test_good_passport);
    RUN(test_check_digit_fail);
    RUN(test_bad_doc_type);
    RUN(test_bad_nationality);
    RUN(test_invalid_birth_month);
    RUN(test_expiry_before_birth);
    RUN(test_empty_personal_no_info);
    RUN(test_low_alpha_name_warn);
    fprintf(stderr, "\n[TEST] %d total, %d failed\n", g_total, g_fail);
    return g_fail == 0 ? 0 : 1;
}
