#include "mrz.h"

#include <string.h>
#include <ctype.h>
#include <stdio.h>

/* Map ASCII char -> numeric value per ICAO 9303.
 * '0'-'9' -> 0-9, 'A'-'Z' -> 10-35, '<' -> 0.
 * Anything else is invalid and returns -1. */
static int char_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'Z') return c - 'A' + 10;
    if (c == '<')              return 0;
    return -1;
}

static int is_valid_mrz_char(char c) {
    if (c >= '0' && c <= '9') return 1;
    if (c >= 'A' && c <= 'Z') return 1;
    if (c == '<')              return 1;
    return 0;
}

const char *mrz_strerror(mrz_status_t s) {
    switch (s) {
        case MRZ_OK:                return "OK";
        case MRZ_ERR_INVALID_CHAR:  return "invalid MRZ character";
        case MRZ_ERR_BAD_LENGTH:    return "bad MRZ length";
        case MRZ_ERR_BAD_CHECK:     return "check digit mismatch";
        case MRZ_ERR_BAD_FIELD:     return "invalid field content";
        case MRZ_ERR_BUFFER_SMALL:  return "output buffer too small";
    }
    return "unknown";
}

/* Cycle weights 7,3,1. */
int mrz_check_digit(const char *s, size_t n) {
    int sum = 0;
    static const int w[3] = { MRZ_W0, MRZ_W1, MRZ_W2 };
    for (size_t i = 0; i < n; ++i) {
        int v = char_value(s[i]);
        if (v < 0) return -1;
        sum += v * w[i % 3];
    }
    return sum % 10;
}

/* Compute the composite check digit over line 2 of a TD3 record.
 * Per ICAO 9303, the composite check digit is calculated over the
 * 43 data+segment-check characters of line 2 (positions 0..42 of l2),
 * i.e. passport_no(9) + ck1 + nationality(3) + birth(6) + ck2 + sex +
 * expiry(6) + ck3 + personal_no(14) + ck4.
 * The function expects a buffer of at least MRZ_TD3_LINE_LEN + 1 chars,
 * where line 2 begins at index MRZ_TD3_LINE_LEN. */
int mrz_td3_composite_check(const char *line2) {
    if (line2 == NULL) return -1;
    int sum = 0;
    static const int w[3] = { MRZ_W0, MRZ_W1, MRZ_W2 };
    for (size_t i = 0; i < 43; ++i) {
        int v = char_value(line2[i]);
        if (v < 0) return -1;
        sum += v * w[i % 3];
    }
    return sum % 10;
}

/* Uppercase ASCII letters in place. */
static void auto_upper_inplace(char *s) {
    for (; *s; ++s) if (*s >= 'a' && *s <= 'z') *s = (char)(*s - 'a' + 'A');
}

/* Copy up to (size-1) bytes of src into dst, padding with '<'.
 * src may be NULL/empty -> all padding. */
static void copy_field(char *dst, size_t size, const char *src) {
    size_t i;
    if (size == 0) return;
    for (i = 0; i + 1 < size && src && src[i] != '\0'; ++i) dst[i] = src[i];
    for (; i + 1 < size; ++i) dst[i] = '<';
    dst[size - 1] = '\0';
}

/* Extract a substring from the joined 88-char buffer.
 * Writes a NUL-terminated string into out (size must be at least span+1). */
static void slice(const char *src, size_t start, size_t span,
                  char *out, size_t out_size) {
    if (out_size == 0) return;
    size_t n = (span < out_size - 1) ? span : (out_size - 1);
    for (size_t i = 0; i < n; ++i) out[i] = src[start + i];
    out[n] = '\0';
}

mrz_status_t mrz_td3_encode(const mrz_td3_t *f, char *out_lines) {
    if (f == NULL || out_lines == NULL) return MRZ_ERR_BAD_FIELD;

    char l1[MRZ_TD3_LINE_LEN + 1];
    char l2[MRZ_TD3_LINE_LEN + 1];

    /* In MRZ the character set is [A-Z0-9<]. Many real-world MRZ readers
     * accept lowercase or spaces and canonicalise them. We do the same:
     * uppercase ASCII letters and replace spaces with '<' so that callers
     * may pass human-readable names like "ANNA MARIA". Any other
     * character (e.g. punctuation) is still rejected. */
    auto_upper_inplace(f->doc_type);
    auto_upper_inplace(f->issuing_state);
    auto_upper_inplace(f->surname);
    auto_upper_inplace(f->given_names);
    for (char *q = f->given_names; *q; ++q) if (*q == ' ') *q = '<';
    /* Multiple spaces / runs in given_names would now produce runs of '<'
     * which decode collapses back into a single separator, so we're OK. */
    for (const char *q = f->doc_type; *q; ++q)
        if (!is_valid_mrz_char(*q)) return MRZ_ERR_INVALID_CHAR;
    for (const char *q = f->issuing_state; *q; ++q)
        if (!is_valid_mrz_char(*q)) return MRZ_ERR_INVALID_CHAR;
    for (const char *q = f->surname; *q; ++q)
        if (!is_valid_mrz_char(*q)) return MRZ_ERR_INVALID_CHAR;
    for (const char *q = f->given_names; *q; ++q)
        if (!is_valid_mrz_char(*q)) return MRZ_ERR_INVALID_CHAR;

    /* Doc type: 2 chars */
    {
        char dt[3]; copy_field(dt, sizeof(dt), f->doc_type);
        memcpy(l1, dt, 2);
    }
    /* Issuing state: 3 chars, positions [2..4] */
    {
        char is[4]; copy_field(is, sizeof(is), f->issuing_state);
        memcpy(l1 + 2, is, 3);
    }
    /* Surname<<Given, positions [5..43], max 39 chars */
    {
        char name[40];
        size_t sn = strlen(f->surname);
        size_t gn = strlen(f->given_names);
        if (sn + 1 + gn > 39) return MRZ_ERR_BAD_FIELD;
        memcpy(name, f->surname, sn);
        name[sn] = '<';
        memcpy(name + sn + 1, f->given_names, gn);
        for (size_t i = sn + 1 + gn; i < 39; ++i) name[i] = '<';
        name[39] = '\0';
        memcpy(l1 + 5, name, 39);
    }
    l1[MRZ_TD3_LINE_LEN] = '\0';

    /* Line 2: passport_no(9) + ck(1) + nationality(3) + birth(6) + ck(1) +
     *          sex(1) + expiry(6) + ck(1) + personal_no(14) + ck(1) + composite(1)
     */
    {
        char pn[10], nat[4], bd[7], sx[2], ed[7], per[15];
        copy_field(pn, sizeof(pn), f->passport_no);
        copy_field(nat, sizeof(nat), f->nationality);
        copy_field(bd, sizeof(bd), f->birth_date_yymmdd);
        copy_field(sx, sizeof(sx), f->sex);
        copy_field(ed, sizeof(ed), f->expiry_date_yymmdd);
        copy_field(per, sizeof(per), f->personal_no);

        for (size_t i = 0; i < 9; ++i)  l2[i] = pn[i];
        int ck1 = mrz_check_digit(pn, 9);
        if (ck1 < 0) return MRZ_ERR_INVALID_CHAR;
        l2[9] = (char)('0' + ck1);

        for (size_t i = 0; i < 3; ++i)  l2[10 + i] = nat[i];
        for (size_t i = 0; i < 6; ++i)  l2[13 + i] = bd[i];
        int ck2 = mrz_check_digit(bd, 6);
        if (ck2 < 0) return MRZ_ERR_INVALID_CHAR;
        l2[19] = (char)('0' + ck2);

        l2[20] = sx[0];

        for (size_t i = 0; i < 6; ++i)  l2[21 + i] = ed[i];
        int ck3 = mrz_check_digit(ed, 6);
        if (ck3 < 0) return MRZ_ERR_INVALID_CHAR;
        l2[27] = (char)('0' + ck3);

        for (size_t i = 0; i < 14; ++i) l2[28 + i] = per[i];
        int ck4 = mrz_check_digit(per, 14);
        if (ck4 < 0) return MRZ_ERR_INVALID_CHAR;
        l2[42] = (char)('0' + ck4);

        /* Build composite input: passport_no + ck1 + nat + bd + ck2 + sex +
         * expiry + ck3 + personal_no + ck4 = 9+1+3+6+1+1+6+1+14+1 = 43 */
        char comp[43];
        memcpy(comp,       pn, 9);
        comp[9] = l2[9];
        memcpy(comp + 10,  nat, 3);
        memcpy(comp + 13,  bd, 6);
        comp[19] = l2[19];
        comp[20] = sx[0];
        memcpy(comp + 21,  ed, 6);
        comp[27] = l2[27];
        memcpy(comp + 28,  per, 14);
        comp[42] = l2[42];
        int ck5 = mrz_check_digit(comp, 43);
        if (ck5 < 0) return MRZ_ERR_INVALID_CHAR;
        l2[43] = (char)('0' + ck5);

        l2[MRZ_TD3_LINE_LEN] = '\0';
    }

    /* Concatenate as: l1\nl2\n\0 */
    memcpy(out_lines, l1, MRZ_TD3_LINE_LEN);
    out_lines[MRZ_TD3_LINE_LEN] = '\n';
    memcpy(out_lines + MRZ_TD3_LINE_LEN + 1, l2, MRZ_TD3_LINE_LEN);
    out_lines[MRZ_TD3_TOTAL + 1] = '\n';
    out_lines[MRZ_TD3_TOTAL + 2] = '\0';
    return MRZ_OK;
}

mrz_status_t mrz_td3_decode(const char *lines, mrz_td3_t *out) {
    if (lines == NULL || out == NULL) return MRZ_ERR_BAD_FIELD;
    memset(out, 0, sizeof(*out));

    /* Build a flat 88-char buffer, dropping any '\n'. */
    char flat[MRZ_TD3_TOTAL + 1];
    size_t w = 0;
    for (size_t i = 0; lines[i] != '\0' && w < MRZ_TD3_TOTAL; ++i) {
        char c = lines[i];
        if (c == '\n' || c == '\r') continue;
        flat[w++] = c;
    }
    if (w != MRZ_TD3_TOTAL) return MRZ_ERR_BAD_LENGTH;
    flat[MRZ_TD3_TOTAL] = '\0';

    /* Validate every char. */
    for (size_t i = 0; i < MRZ_TD3_TOTAL; ++i)
        if (!is_valid_mrz_char(flat[i])) return MRZ_ERR_INVALID_CHAR;

    /* Slice fields. */
    slice(flat, 0, 2,  out->doc_type,          sizeof(out->doc_type));
    slice(flat, 2, 3,  out->issuing_state,     sizeof(out->issuing_state));
    {
        /* The name field (39 chars) is laid out as:
         *   surname '<<' given1 '<' given2 '<' ... '<' + '<' padding.
         * Surname ends at the first '<'. Each given name ends at a later
         * '<' or at the end of the field. */
        char raw[40];
        slice(flat, 5, 39, raw, sizeof(raw));
        char *first_lt = strchr(raw, '<');
        if (first_lt == NULL) return MRZ_ERR_BAD_FIELD;
        size_t sn = (size_t)(first_lt - raw);
        memcpy(out->surname, raw, sn);
        out->surname[sn] = '\0';

        /* Build given_names by joining subsequent non-empty name tokens
         * separated by single '<'. Trailing padding '<' is dropped. */
        out->given_names[0] = '\0';
        const char *p = first_lt + 1;
        const char *end = raw + 39;
        while (p < end) {
            const char *sep = strchr(p, '<');
            const char *tok_end = (sep == NULL) ? end : sep;
            size_t tlen = (size_t)(tok_end - p);
            if (tlen > 0) {
                size_t cur = strlen(out->given_names);
                if (cur > 0 && cur + 1 < sizeof(out->given_names)) {
                    out->given_names[cur++] = '<';
                }
                if (cur + tlen < sizeof(out->given_names)) {
                    memcpy(out->given_names + cur, p, tlen);
                    out->given_names[cur + tlen] = '\0';
                }
            }
            if (sep == NULL) break;
            p = sep + 1;
        }
    }
    slice(flat, 0  + MRZ_TD3_LINE_LEN, 9,  out->passport_no,      sizeof(out->passport_no));
    slice(flat, 10 + MRZ_TD3_LINE_LEN, 3,  out->nationality,      sizeof(out->nationality));
    slice(flat, 13 + MRZ_TD3_LINE_LEN, 6,  out->birth_date_yymmdd,sizeof(out->birth_date_yymmdd));
    slice(flat, 20 + MRZ_TD3_LINE_LEN, 1,  out->sex,              sizeof(out->sex));
    slice(flat, 21 + MRZ_TD3_LINE_LEN, 6,  out->expiry_date_yymmdd,sizeof(out->expiry_date_yymmdd));
    slice(flat, 28 + MRZ_TD3_LINE_LEN, 14, out->personal_no,      sizeof(out->personal_no));

    /* Extract per-segment check digits. */
    out->ck_passport_no = flat[9  + MRZ_TD3_LINE_LEN] - '0';
    out->ck_birth_date  = flat[19 + MRZ_TD3_LINE_LEN] - '0';
    out->ck_expiry_date = flat[27 + MRZ_TD3_LINE_LEN] - '0';
    out->ck_personal_no = flat[42 + MRZ_TD3_LINE_LEN] - '0';
    out->ck_composite   = flat[43 + MRZ_TD3_LINE_LEN] - '0';

    /* Verify each check digit. */
    int got;
    got = mrz_check_digit(flat + 0  + MRZ_TD3_LINE_LEN, 9);
    if (got < 0 || got != out->ck_passport_no) return MRZ_ERR_BAD_CHECK;
    got = mrz_check_digit(flat + 13 + MRZ_TD3_LINE_LEN, 6);
    if (got < 0 || got != out->ck_birth_date)   return MRZ_ERR_BAD_CHECK;
    got = mrz_check_digit(flat + 21 + MRZ_TD3_LINE_LEN, 6);
    if (got < 0 || got != out->ck_expiry_date)  return MRZ_ERR_BAD_CHECK;
    got = mrz_check_digit(flat + 28 + MRZ_TD3_LINE_LEN, 14);
    if (got < 0 || got != out->ck_personal_no)  return MRZ_ERR_BAD_CHECK;

    int comp = mrz_td3_composite_check(flat + MRZ_TD3_LINE_LEN);
    if (comp < 0 || comp != out->ck_composite) return MRZ_ERR_BAD_CHECK;

    return MRZ_OK;
}
