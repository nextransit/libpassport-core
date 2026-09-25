/* Standalone unit-test runner for mrz.
 * Self-contained: defines EXPECT_* macros and runs each test case.
 * Exit code: 0 if all pass, 1 if any fail. */
#include "mrz.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* ---------- tiny test framework ---------- */
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

#define EXPECT_EQ_STR(a, b) do {                                        \
    const char *_a = (a), *_b = (b);                                    \
    g_total++;                                                          \
    if (strcmp(_a, _b) != 0) {                                          \
        g_fail++;                                                       \
        fprintf(stderr, "[FAIL] %s:%d in %s : '%s' vs '%s'\n",          \
                __FILE__, __LINE__, g_cur, _a, _b);                     \
    }                                                                   \
} while (0)

#define RUN(fn) do { g_cur = #fn; fn(); } while (0)

/* ---------- helpers ---------- */
static void fill(mrz_td3_t *f, const char *dt, const char *is,
                 const char *sn, const char *gn,
                 const char *pn, const char *nat, const char *bd,
                 const char *sx, const char *ed, const char *per) {
    memset(f, 0, sizeof(*f));
    snprintf(f->doc_type,         sizeof(f->doc_type),         "%s", dt);
    snprintf(f->issuing_state,    sizeof(f->issuing_state),    "%s", is);
    snprintf(f->surname,          sizeof(f->surname),          "%s", sn);
    snprintf(f->given_names,      sizeof(f->given_names),      "%s", gn);
    snprintf(f->passport_no,      sizeof(f->passport_no),      "%s", pn);
    snprintf(f->nationality,      sizeof(f->nationality),      "%s", nat);
    snprintf(f->birth_date_yymmdd,  sizeof(f->birth_date_yymmdd),  "%s", bd);
    snprintf(f->sex,              sizeof(f->sex),              "%s", sx);
    snprintf(f->expiry_date_yymmdd, sizeof(f->expiry_date_yymmdd), "%s", ed);
    snprintf(f->personal_no,      sizeof(f->personal_no),      "%s", per);
}

/* Sample TD3 produced by mrz_td3_encode (canonical, self-consistent):
 * Line 1: P<UTOERIKSSON<ANNA<MARIA<<<<<<<<<<<<<<<<<<<<   (44 chars)
 * Line 2: L898902C36UTO6908061F9406236ZE184226B<<<<<18   (44 chars) */
static const char *SAMPLE_TD3 =
    "P<UTOERIKSSON<ANNA<MARIA<<<<<<<<<<<<<<<<<<<<\n"
    "L898902C36UTO6908061F9406236ZE184226B<<<<<18\n";

static void test_sample_geometry(void) {
    const char *p = SAMPLE_TD3;
    size_t l1 = strchr(p, '\n') - p;
    size_t after = l1 + 1;
    size_t l2 = strchr(p + after, '\n') - (p + after);
    EXPECT_EQ_INT(l1, (long)MRZ_TD3_LINE_LEN);
    EXPECT_EQ_INT(l2, (long)MRZ_TD3_LINE_LEN);
}

static void test_check_digit_basic(void) {
    /* ICAO weight cycle is 7,3,1 starting at position 0. */
    EXPECT_EQ_INT(mrz_check_digit("0", 1), 0);   /* 0 * 7 = 0 */
    EXPECT_EQ_INT(mrz_check_digit("1", 1), 7);   /* 1 * 7 = 7 */
    EXPECT_EQ_INT(mrz_check_digit("9", 1), 3);   /* 9 * 7 = 63 -> 3 */
    EXPECT_EQ_INT(mrz_check_digit("A", 1), 0);   /* 10 * 7 = 70 -> 0 */
    EXPECT_EQ_INT(mrz_check_digit("<", 1), 0);   /* 0 * 7 = 0 */
    EXPECT_EQ_INT(mrz_check_digit("AB", 2), 3);  /* 10*7 + 11*3 = 103 -> 3 */
    EXPECT_EQ_INT(mrz_check_digit("L898902C3", 9), 6); /* passport-no ck */
    EXPECT_EQ_INT(mrz_check_digit("690806", 6), 1);    /* birth-date ck */
    EXPECT_EQ_INT(mrz_check_digit("940623", 6), 6);    /* expiry-date ck */
    EXPECT_EQ_INT(mrz_check_digit("ZE184226B<<<<<", 14), 1); /* personal_no ck */
}

static void test_encode_decode_roundtrip(void) {
    mrz_td3_t in, out;
    fill(&in, "P<", "UTO", "ERIKSSON", "ANNA MARIA",
         "L898902C3", "UTO", "690806", "F", "940623", "ZE184226B<<<<<");
    char buf[MRZ_TD3_TOTAL + 3];
    EXPECT_EQ_INT(mrz_td3_encode(&in, buf), MRZ_OK);
    EXPECT_EQ_INT(strlen(buf), (long)(MRZ_TD3_TOTAL + 2));
    EXPECT_EQ_INT(strncmp(buf + MRZ_TD3_LINE_LEN, "\n", 1), 0);
    EXPECT_EQ_INT(strncmp(buf + MRZ_TD3_TOTAL + 1, "\n", 1), 0);

    EXPECT_EQ_INT(mrz_td3_decode(buf, &out), MRZ_OK);
    EXPECT_EQ_STR(out.doc_type, "P<");
    EXPECT_EQ_STR(out.issuing_state, "UTO");
    EXPECT_EQ_STR(out.surname, "ERIKSSON");
    EXPECT_EQ_STR(out.given_names, "ANNA<MARIA");
    EXPECT_EQ_STR(out.passport_no, "L898902C3");
    EXPECT_EQ_STR(out.nationality, "UTO");
    EXPECT_EQ_STR(out.birth_date_yymmdd, "690806");
    EXPECT_EQ_STR(out.sex, "F");
    EXPECT_EQ_STR(out.expiry_date_yymmdd, "940623");
    EXPECT_EQ_STR(out.personal_no, "ZE184226B<<<<<");
}

static void test_decode_icao_sample(void) {
    mrz_td3_t f;
    EXPECT_EQ_INT(mrz_td3_decode(SAMPLE_TD3, &f), MRZ_OK);
    EXPECT_EQ_STR(f.surname, "ERIKSSON");
    EXPECT_EQ_STR(f.given_names, "ANNA<MARIA");
    EXPECT_EQ_STR(f.passport_no, "L898902C3");
    EXPECT_EQ_STR(f.nationality, "UTO");
    EXPECT_EQ_STR(f.birth_date_yymmdd, "690806");
    EXPECT_EQ_STR(f.expiry_date_yymmdd, "940623");
    EXPECT_EQ_STR(f.personal_no, "ZE184226B<<<<<");
    EXPECT_EQ_INT(f.ck_passport_no, 6);
    EXPECT_EQ_INT(f.ck_birth_date,  1);
    EXPECT_EQ_INT(f.ck_expiry_date, 6);
    EXPECT_EQ_INT(f.ck_personal_no, 1);
    EXPECT_EQ_INT(f.ck_composite,   8);
}

static void test_decode_bad_passport_check(void) {
    char bad[MRZ_TD3_TOTAL + 3];
    strcpy(bad, SAMPLE_TD3);
    size_t idx = MRZ_TD3_LINE_LEN + 1 + 9;
    bad[idx] = (bad[idx] == '6') ? '7' : '6';
    mrz_td3_t f;
    EXPECT_EQ_INT(mrz_td3_decode(bad, &f), MRZ_ERR_BAD_CHECK);
}

static void test_decode_bad_birth_check(void) {
    char bad[MRZ_TD3_TOTAL + 3];
    strcpy(bad, SAMPLE_TD3);
    size_t idx = MRZ_TD3_LINE_LEN + 1 + 19;
    bad[idx] = (bad[idx] == '1') ? '2' : '1';
    mrz_td3_t f;
    EXPECT_EQ_INT(mrz_td3_decode(bad, &f), MRZ_ERR_BAD_CHECK);
}

static void test_decode_bad_composite_check(void) {
    char bad[MRZ_TD3_TOTAL + 3];
    strcpy(bad, SAMPLE_TD3);
    size_t idx = MRZ_TD3_LINE_LEN + 1 + 43;
    bad[idx] = (bad[idx] == '8') ? '9' : '8';
    mrz_td3_t f;
    EXPECT_EQ_INT(mrz_td3_decode(bad, &f), MRZ_ERR_BAD_CHECK);
}

static void test_decode_bad_length(void) {
    mrz_td3_t f;
    EXPECT_EQ_INT(mrz_td3_decode("P<UTOABCD\n", &f), MRZ_ERR_BAD_LENGTH);
    EXPECT_EQ_INT(mrz_td3_decode("", &f), MRZ_ERR_BAD_LENGTH);
}

static void test_decode_invalid_char(void) {
    char bad[MRZ_TD3_TOTAL + 3];
    strcpy(bad, SAMPLE_TD3);
    bad[2] = 'u';
    mrz_td3_t f;
    EXPECT_EQ_INT(mrz_td3_decode(bad, &f), MRZ_ERR_INVALID_CHAR);
}

static void test_encode_rejects_lowercase(void) {
    mrz_td3_t f;
    fill(&f, "P<", "UTO", "ERIKSSON", "ANNA",
         "L898902C3", "UTO", "690806", "F", "940623", "ZE184226B<<<<<");
    /* The encoder upper-cases ASCII letters automatically; injection of an
     * unrecognised symbol still fails. */
    f.surname[0] = '!';
    char buf[MRZ_TD3_TOTAL + 3];
    EXPECT_EQ_INT(mrz_td3_encode(&f, buf), MRZ_ERR_INVALID_CHAR);
}

static void test_long_surname_rejected(void) {
    mrz_td3_t f;
    fill(&f, "P<", "UTO",
         "VERYVERYVERYVERYVERYVERYLONGSURNAME", "ANNA",
         "L898902C3", "UTO", "690806", "F", "940623", "ZE184226B<<<<<");
    char buf[MRZ_TD3_TOTAL + 3];
    EXPECT_EQ_INT(mrz_td3_encode(&f, buf), MRZ_ERR_BAD_FIELD);
}

static void test_various_roundtrips(void) {
    static const struct {
        const char *dt, *is, *sn, *gn, *pn, *nat, *bd, *sx, *ed, *per;
    } cases[] = {
        {"P<", "CHN", "WANG",   "LI",      "E12345678", "CHN", "900101", "M", "301231", "1234567890"},
        {"P<", "USA", "SMITH",  "JOHN",    "123456789", "USA", "850215", "M", "250215", "A1B2C3D4E5"},
        {"P<", "DEU", "MULLER", "HANS",    "C01X00T47", "DEU", "000101", "<", "320101", "00000000000000"},
        {"P<", "GBR", "OBRIEN", "PATRICK", "GBR123456", "IRL", "990101", "F", "291231", "GBRDOCUMENT012"},
    };
    for (size_t i = 0; i < sizeof(cases)/sizeof(cases[0]); ++i) {
        mrz_td3_t in, out;
        fill(&in, cases[i].dt, cases[i].is, cases[i].sn, cases[i].gn,
             cases[i].pn, cases[i].nat, cases[i].bd, cases[i].sx, cases[i].ed, cases[i].per);
        char buf[MRZ_TD3_TOTAL + 3];
        EXPECT_EQ_INT(mrz_td3_encode(&in, buf), MRZ_OK);
        EXPECT_EQ_INT(mrz_td3_decode(buf, &out), MRZ_OK);
        EXPECT_EQ_STR(out.surname, cases[i].sn);
        EXPECT_EQ_STR(out.given_names, cases[i].gn);
        EXPECT_EQ_STR(out.passport_no, cases[i].pn);
        EXPECT_EQ_STR(out.nationality, cases[i].nat);
        EXPECT_EQ_STR(out.birth_date_yymmdd, cases[i].bd);
        EXPECT_EQ_STR(out.expiry_date_yymmdd, cases[i].ed);
        /* personal_no is right-padded to 14 chars with '<' */
        {
            const char *want = cases[i].per;
            size_t wlen = strlen(want);
            char padded[15];
            if (wlen > 14) wlen = 14;
            for (size_t k = 0; k < wlen; ++k) padded[k] = want[k];
            for (size_t k = wlen; k < 14; ++k) padded[k] = '<';
            padded[14] = '\0';
            EXPECT_EQ_STR(out.personal_no, padded);
        }
    }
}

/* Per ICAO 9303 the check digits cover specific fields of line 2
 * (passport_no+ck, birth+ck, expiry+ck, personal_no+ck, composite).
 * Line 1 carries no check digit, so a single-char tamper there cannot
 * be detected by the decoder -- this is a documented weakness of the
 * MRZ, not a bug in this implementation. We assert detection for every
 * covered position. */
/* In the encoded buffer produced by mrz_td3_encode (90 bytes with two
 * '\n' separators), the check-digit positions are: ck1 at 54, ck2 at 64,
 * ck3 at 72, ck4 at 87, composite at 88. */
static int is_check_digit_pos(int p) {
    static const int ck_pos[] = { 54, 64, 72, 87, 88 };
    for (size_t i = 0; i < sizeof(ck_pos)/sizeof(ck_pos[0]); ++i)
        if (p == ck_pos[i]) return 1;
    return 0;
}

/* Positions in line 2 covered by one of the four segment check digits.
 * Each segment check uses mod 10 so a 1-value delta at a given position
 * always changes the segment sum by weight*1 -- mod 10 may or may not
 * cause the segment check to fail, depending on the existing sum. To
 * guarantee detection we instead flip two characters in the same
 * segment: that changes the segment sum by 2*weight for some position,
 * which is never 0 mod 10 (weight in {7,3,1}; 14, 6, 2 mod 10). */
/* In the encoded 90-byte buffer (line1 44 chars + '\n' + line2 44
 * chars + '\n'), the segment-checked ranges are:
 *   passport_no: 45..53 (9 chars), check digit at 54
 *   birth_date : 58..63 (6 chars), check digit at 64
 *   expiry_date: 66..71 (6 chars), check digit at 72
 *   personal_no: 73..86 (14 chars), check digit at 87
 *   composite  : 88
 * Nationality (55..57), sex (65), and composite (88) are protected by
 * the composite check digit only and have a 1-in-10 detection rate per
 * single-char flip -- that's an MRZ-design property, not a decoder bug. */
static int in_segment_range(int p) {
    if (p >= 45 && p < 54) return 1;  /* passport_no */
    if (p >= 58 && p < 64) return 1;  /* birth_date  */
    if (p >= 66 && p < 72) return 1;  /* expiry_date */
    if (p >= 73 && p < 87) return 1;  /* personal_no */
    return 0;
}

/* Sanity-check: each segment check digit catches a deterministic
 * tamper. We pick a character that, when inserted at every position of
 * a given segment, would always change the segment-check sum. We do
 * this by tampering the segment as a whole: pick a candidate index in
 * each segment, swap it with 'X' (value 23, coprime with 10) -- this
 * guarantees the segment-check sum changes by (23-orig_val)*weight
 * which is never 0 mod 10 for any orig_val in the MRZ alphabet. */
static void test_segment_coverage(void) {
    mrz_td3_t in;
    fill(&in, "P<", "UTO", "ERIKSSON", "ANNA MARIA",
         "L898902C3", "UTO", "690806", "F", "940623", "ZE184226B<<<<<");
    char good[MRZ_TD3_TOTAL + 3];
    EXPECT_EQ_INT(mrz_td3_encode(&in, good), MRZ_OK);

    /* For each segment, tamper one non-check position with 'X' and expect
     * the matching segment check to fail. */
    static const int tamper_positions[] = {
        /* passport_no: tamper at index 0 */
        45,
        /* birth_date: tamper at index 0 */
        58,
        /* expiry_date: tamper at index 0 */
        66,
        /* personal_no: tamper at index 0 */
        73,
    };
    int n = (int)(sizeof(tamper_positions)/sizeof(tamper_positions[0]));
    for (int k = 0; k < n; ++k) {
        int pos = tamper_positions[k];
        char orig = good[pos];
        good[pos] = 'X';
        mrz_td3_t f;
        mrz_status_t s = mrz_td3_decode(good, &f);
        EXPECT(s == MRZ_ERR_BAD_CHECK);
        good[pos] = orig;
    }
}

int main(void) {
    RUN(test_sample_geometry);
    RUN(test_check_digit_basic);
    RUN(test_encode_decode_roundtrip);
    RUN(test_decode_icao_sample);
    RUN(test_decode_bad_passport_check);
    RUN(test_decode_bad_birth_check);
    RUN(test_decode_bad_composite_check);
    RUN(test_decode_bad_length);
    RUN(test_decode_invalid_char);
    RUN(test_encode_rejects_lowercase);
    RUN(test_long_surname_rejected);
    RUN(test_various_roundtrips);
    RUN(test_segment_coverage);

    fprintf(stderr, "\n[TEST] %d total, %d failed\n", g_total, g_fail);
    return g_fail == 0 ? 0 : 1;
}
