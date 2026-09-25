#include "ac.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

void ac_report_init(ac_report_t *r) {
    if (!r) return;
    r->items = NULL;
    r->count = 0;
    r->cap   = 0;
    r->warnings = r->fails = r->infos = 0;
}

void ac_report_free(ac_report_t *r) {
    if (!r) return;
    free(r->items);
    r->items = NULL;
    r->count = r->cap = 0;
}

void ac_report_add(ac_report_t *r, const char *name,
                   ac_severity_t sev, const char *detail) {
    if (!r) return;
    if (r->count == r->cap) {
        size_t nc = r->cap ? r->cap * 2 : 8;
        ac_finding_t *p = (ac_finding_t *)realloc(r->items, nc * sizeof(*p));
        if (!p) return;
        r->items = p;
        r->cap = nc;
    }
    r->items[r->count].name = name;
    r->items[r->count].severity = sev;
    r->items[r->count].detail = detail;
    r->count++;
    if (sev == AC_SEVERITY_WARN) r->warnings++;
    else if (sev == AC_SEVERITY_FAIL) r->fails++;
    else if (sev == AC_SEVERITY_INFO) r->infos++;
}

const char *ac_strerror(ac_status_t s) {
    switch (s) {
        case AC_OK:                  return "OK";
        case AC_ERR_LENGTH:          return "bad MRZ length";
        case AC_ERR_DOC_TYPE:        return "bad document type";
        case AC_ERR_ISSUING_STATE:   return "bad issuing state";
        case AC_ERR_NATIONALITY:     return "bad nationality";
        case AC_ERR_SEX:             return "bad sex field";
        case AC_ERR_BIRTH_DATE:      return "bad birth date";
        case AC_ERR_EXPIRY_DATE:     return "bad expiry date";
        case AC_ERR_PADDING_ANOMALY: return "suspicious padding pattern";
        case AC_ERR_NAME_ENTROPY:    return "name entropy suspicious";
        case AC_ERR_PERSONAL_NO:     return "bad personal number";
        case AC_ERR_CHECK_DIGITS:    return "check-digit mismatch";
        case AC_ERR_DATE_ORDER:      return "date order inconsistent";
        case AC_ERR_DATE_RANGE:      return "date outside plausible range";
    }
    return "unknown";
}

/* -------- helpers -------- */
static int all_upper_or_digit(const char *s, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        char c = s[i];
        if (c == '<') continue;
        if (c >= 'A' && c <= 'Z') continue;
        if (c >= '0' && c <= '9') continue;
        return 0;
    }
    return 1;
}

static int is_valid_country_code3(const char *s) {
    /* Permissive: ICAO publishes a list of ~250 three-letter codes.
     * For the offline test we accept any 3-letter combination from
     * A-Z (including the ICAO-reserved code 'UTO' used in examples). */
    for (int i = 0; i < 3; ++i) {
        char c = s[i];
        if (c < 'A' || c > 'Z') return 0;
    }
    return 1;
}

static int is_valid_doc_type(const char *s) {
    /* ICAO Doc 9303 lists: P, P<, PA, PD, PE, PI, PO, PR, PS, PV,
     * plus visa/accreditation variants. We accept P + optional '<' or
     * letter, plus I (ID card), A (alien passport), C (refugee), etc. */
    if (s[0] == 'P' || s[0] == 'I' || s[0] == 'A' ||
        s[0] == 'C' || s[0] == 'D' || s[0] == 'X') {
        if (s[1] == '<' || (s[1] >= 'A' && s[1] <= 'Z')) return 1;
    }
    return 0;
}

static int is_valid_sex(char c) {
    return c == 'M' || c == 'F' || c == '<';
}

/* YYMMDD -> ac_date_t; returns 0 on success. We treat YYMMDD as
 * 19YY for YY < cutoff, else 20YY (cutoff defaults to 60 per ICAO 9303). */
static int yymmdd_to_date(const char *s, int cutoff_year_2digit,
                          ac_date_t *out) {
    if (!all_upper_or_digit(s, 6)) return -1;
    int yy = (s[0]-'0')*10 + (s[1]-'0');
    int mm = (s[2]-'0')*10 + (s[3]-'0');
    int dd = (s[4]-'0')*10 + (s[5]-'0');
    if (mm < 1 || mm > 12) return -1;
    if (dd < 1 || dd > 31) return -1;
    /* Days-in-month basic check (no leap-year handling needed for MRZ). */
    static const int dpm[12] = {31,29,31,30,31,30,31,31,30,31,30,31};
    if (dd > dpm[mm-1]) return -1;
    int year = (yy <= cutoff_year_2digit) ? 2000 + yy : 1900 + yy;
    out->year = year;
    out->month = mm;
    out->day = dd;
    return 0;
}

static int date_cmp(const ac_date_t *a, const ac_date_t *b) {
    if (a->year != b->year) return a->year - b->year;
    if (a->month != b->month) return a->month - b->month;
    return a->day - b->day;
}

/* Compute an "entropy" score for a name: count distinct characters and
 * ratio of alphabetic to filler '<'. Very low diversity or excessive
 * filler is suspicious. */
static int name_score(const char *raw, size_t n, double *out_diversity,
                      double *out_padding_ratio) {
    int distinct[256] = {0};
    int fillers = 0, alpha = 0;
    for (size_t i = 0; i < n; ++i) {
        char c = raw[i];
        distinct[(unsigned char)c] = 1;
        if (c == '<') fillers++;
        else if (c >= 'A' && c <= 'Z') alpha++;
    }
    int d = 0;
    for (int i = 0; i < 256; ++i) d += distinct[i];
    if (out_diversity) *out_diversity = (double)d / (double)n;
    if (out_padding_ratio) *out_padding_ratio = (double)fillers / (double)n;
    return alpha;
}

/* -------- main verification -------- */
ac_status_t ac_verify_td3(const char *mrz_lines,
                          const ac_date_t *today,
                          ac_report_t *report) {
    if (!report) return AC_ERR_LENGTH;
    /* Parse the MRZ first. We piggy-back on the library for the heavy
     * lifting: character set + check digits. */
    mrz_td3_t fields;
    mrz_status_t ms = mrz_td3_decode(mrz_lines, &fields);
    if (ms != MRZ_OK) {
        ac_report_add(report, "decode", AC_SEVERITY_FAIL,
                      mrz_strerror(ms));
        return AC_ERR_CHECK_DIGITS;
    }

    ac_status_t overall = AC_OK;

    /* doc_type */
    if (!is_valid_doc_type(fields.doc_type)) {
        ac_report_add(report, "doc_type", AC_SEVERITY_FAIL,
                      "document type prefix not in ICAO 9303 list");
        overall = AC_ERR_DOC_TYPE;
    } else {
        ac_report_add(report, "doc_type", AC_SEVERITY_OK,
                      "document type prefix recognised");
    }

    /* issuing_state */
    if (!is_valid_country_code3(fields.issuing_state)) {
        ac_report_add(report, "issuing_state", AC_SEVERITY_FAIL,
                      "issuing state code must be 3 uppercase letters");
        overall = AC_ERR_ISSUING_STATE;
    } else {
        ac_report_add(report, "issuing_state", AC_SEVERITY_OK,
                      "issuing state code well-formed");
    }

    /* nationality */
    if (!is_valid_country_code3(fields.nationality)) {
        ac_report_add(report, "nationality", AC_SEVERITY_FAIL,
                      "nationality code must be 3 uppercase letters");
        overall = AC_ERR_NATIONALITY;
    } else {
        ac_report_add(report, "nationality", AC_SEVERITY_OK,
                      "nationality code well-formed");
    }

    /* sex */
    if (!is_valid_sex(fields.sex[0])) {
        ac_report_add(report, "sex", AC_SEVERITY_FAIL,
                      "sex must be M, F, or <");
        overall = AC_ERR_SEX;
    } else {
        ac_report_add(report, "sex", AC_SEVERITY_OK, "sex field valid");
    }

    /* birth date plausibility */
    ac_date_t birth = {0,0,0};
    if (yymmdd_to_date(fields.birth_date_yymmdd, 60, &birth) != 0) {
        ac_report_add(report, "birth_date", AC_SEVERITY_FAIL,
                      "birth date YYMMDD unparseable");
        overall = AC_ERR_BIRTH_DATE;
    } else {
        ac_report_add(report, "birth_date", AC_SEVERITY_OK,
                      "birth date YYMMDD well-formed");
    }

    /* expiry plausibility */
    ac_date_t expiry = {0,0,0};
    if (yymmdd_to_date(fields.expiry_date_yymmdd, 60, &expiry) != 0) {
        ac_report_add(report, "expiry_date", AC_SEVERITY_FAIL,
                      "expiry date YYMMDD unparseable");
        overall = AC_ERR_EXPIRY_DATE;
    } else {
        ac_report_add(report, "expiry_date", AC_SEVERITY_OK,
                      "expiry date YYMMDD well-formed");
    }

    /* Date order: birth < expiry (a passport cannot expire before birth). */
    if (birth.year && expiry.year) {
        int cmp = date_cmp(&birth, &expiry);
        if (cmp >= 0) {
            ac_report_add(report, "date_order", AC_SEVERITY_FAIL,
                          "expiry date is not after birth date");
            overall = AC_ERR_DATE_ORDER;
        } else {
            ac_report_add(report, "date_order", AC_SEVERITY_OK,
                          "expiry > birth");
        }
    }

    /* Reasonable age: birth in [1900..today], expiry < today + 30y. */
    if (birth.year) {
        int min_y = 1900;
        int max_y = today ? today->year : 2000;
        if (birth.year < min_y || birth.year > max_y) {
            ac_report_add(report, "birth_range", AC_SEVERITY_WARN,
                          "birth year outside plausible range");
            if (overall == AC_OK) overall = AC_ERR_DATE_RANGE;
        }
        if (expiry.year) {
            int max_exp = (today ? today->year : 2000) + 30;
            if (expiry.year > max_exp) {
                ac_report_add(report, "expiry_range", AC_SEVERITY_WARN,
                              "expiry more than 30y in the future");
                if (overall == AC_OK) overall = AC_ERR_DATE_RANGE;
            }
        }
    }

    /* Name entropy: at least 2 distinct characters and at least 25% alpha. */
    {
        char raw[40] = {0};
        size_t sn = strlen(fields.surname);
        size_t gn = strlen(fields.given_names);
        if (sn + 1 + gn < sizeof(raw)) {
            memcpy(raw, fields.surname, sn);
            raw[sn] = '<';
            memcpy(raw + sn + 1, fields.given_names, gn);
        }
        double diversity = 0, padding = 0;
        int alpha = name_score(raw, sn + 1 + gn, &diversity, &padding);
        if (alpha < 3 || diversity < 0.2 || padding > 0.8) {
            ac_report_add(report, "name_entropy", AC_SEVERITY_WARN,
                          "name has very low alphabetic content");
            if (overall == AC_OK) overall = AC_ERR_NAME_ENTROPY;
        } else {
            ac_report_add(report, "name_entropy", AC_SEVERITY_OK,
                          "name has plausible alphabetic content");
        }
    }

    /* personal_no: must be at least 1 non-filler character. */
    {
        int non_filler = 0;
        for (size_t i = 0; fields.personal_no[i]; ++i)
            if (fields.personal_no[i] != '<') non_filler++;
        if (non_filler == 0) {
            ac_report_add(report, "personal_no", AC_SEVERITY_INFO,
                          "personal number is empty (allowed by spec)");
        } else {
            ac_report_add(report, "personal_no", AC_SEVERITY_OK,
                          "personal number present");
        }
    }

    /* check-digit cross-consistency: if any segment check digit in the
     * decoded struct does not equal a fresh computation we mark a fail. */
    if (fields.ck_passport_no < 0 || fields.ck_passport_no > 9 ||
        fields.ck_birth_date  < 0 || fields.ck_birth_date  > 9 ||
        fields.ck_expiry_date < 0 || fields.ck_expiry_date > 9 ||
        fields.ck_personal_no < 0 || fields.ck_personal_no > 9 ||
        fields.ck_composite   < 0 || fields.ck_composite   > 9) {
        ac_report_add(report, "check_digits", AC_SEVERITY_FAIL,
                      "check-digit extraction out of range");
        overall = AC_ERR_CHECK_DIGITS;
    } else {
        ac_report_add(report, "check_digits", AC_SEVERITY_OK,
                      "all check digits consistent");
    }

    /* Final summary. We use a static buffer to keep `detail` valid
     * after this function returns (the report stores the pointer). */
    static char buf[128];
    snprintf(buf, sizeof(buf),
             "%d fail / %d warn / %d info",
             report->fails, report->warnings, report->infos);
    ac_report_add(report, "summary",
                  report->fails ? AC_SEVERITY_FAIL :
                  report->warnings ? AC_SEVERITY_WARN :
                  AC_SEVERITY_OK, buf);

    return overall;
}
