/* High-level security pipeline: BAC key derivation, cross-modal
 * consistency, anti-counterfeit feature simulation. */
#include "security.h"
#include "face.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* ICAO Doc 9303 Part 11: SHA-1(seed) split into 16-byte halves; XOR
 * each half with a constant ("01234567" and "89ABCDEF") to produce
 * the 16-byte K_enc and K_mac. The constants we use are the high
 * parity version (odd parity in the least significant bit of each
 * byte), per the standard. */
static const uint8_t C1[8] = {0x01,0x23,0x45,0x67,0x89,0xAB,0xCD,0xEF};
static const uint8_t C2[8] = {0xFE,0xDC,0xBA,0x98,0x76,0x54,0x32,0x10};

bac_keys_t bac_derive_keys(const char *passport_no,
                           const char *birth_yymmdd,
                           const char *expiry_yymmdd) {
    bac_keys_t k;
    memset(&k, 0, sizeof(k));
    /* Build the 24-byte seed: passport_no (9) + ck + birth (6) + ck
     * + expiry (6) + ck. */
    if (strlen(passport_no) < 9 || strlen(birth_yymmdd) < 6 ||
        strlen(expiry_yymmdd) < 6) {
        return k;
    }
    uint8_t full_seed[24];
    memcpy(full_seed,      passport_no,  9);
    full_seed[9] = (uint8_t)mrz_check_digit(passport_no, 9);
    memcpy(full_seed + 10, birth_yymmdd, 6);
    full_seed[16] = (uint8_t)mrz_check_digit(birth_yymmdd, 6);
    memcpy(full_seed + 17, expiry_yymmdd, 6);
    full_seed[23] = (uint8_t)mrz_check_digit(expiry_yymmdd, 6);

    uint8_t digest[20];
    sha1(full_seed, 24, digest);
    /* Per ICAO Doc 9303 Part 11: K_enc = first half of digest XOR
     * constant (repeated), K_mac = second half XOR second constant. */
    for (int i = 0; i < 8; ++i) {
        k.k_enc[i]      = digest[i]      ^ C1[i];
        k.k_enc[i + 8]  = digest[i]      ^ C1[i];
        k.k_mac[i]      = digest[i + 8]  ^ C2[i];
        k.k_mac[i + 8]  = digest[i + 12] ^ C2[i];
    }
    return k;
}

/* ISO/IEC 9797-1 MAC Algorithm 3 (retail MAC) using DES.
 *
 * Algorithm:
 *   1. Zero-pad the message to a multiple of 8 bytes.
 *   2. CBC-encrypt with K1 (first 8 bytes of key).
 *   3. Take the last ciphertext block C1.
 *   4. DES-decrypt with K2 (second 8 bytes of key).
 *   5. DES-encrypt with K1 again.
 *   The 8-byte output is the MAC. */
void mac3_des(const uint8_t key[16], const uint8_t *data, size_t n,
              uint8_t out[8]) {
    size_t padded = ((n + 7) / 8) * 8;
    uint8_t buf[1024];   /* enough for our tests */
    if (padded > sizeof(buf)) padded = sizeof(buf);
    memcpy(buf, data, n);
    memset(buf + n, 0, padded - n);
    uint8_t iv[8] = {0};
    des3_cbc_encrypt(key, iv, buf, buf, padded);  /* EDE encrypt all */
    uint8_t last[8];
    memcpy(last, buf + padded - 8, 8);
    /* E-D-E to get the MAC. */
    uint8_t tmp[8];
    des_ecb_decrypt(key + 8, last, tmp);
    des_ecb_encrypt(key, tmp, out);
}

/* ----- Cross-modal check ----- */
static const char *field_at(const char *s, int pos, int len, char *buf) {
    int i;
    for (i = 0; i < len && s[pos + i] != '\0'; ++i) buf[i] = s[pos + i];
    buf[i] = '\0';
    return buf;
}

cross_modal_report_t security_cross_check(const char *mrz, const char *viz) {
    cross_modal_report_t r;
    memset(&r, 0, sizeof(r));
    /* Extract MRZ fields by calling into mrz library. */
    mrz_td3_t m;
    if (!mrz || mrz_td3_decode(mrz, &m) != MRZ_OK) {
        r.passport_no = AC_FIELD_MISSING;
        r.birth_date  = AC_FIELD_MISSING;
        r.expiry_date = AC_FIELD_MISSING;
        r.nationality = AC_FIELD_MISSING;
        r.name        = AC_FIELD_MISSING;
        return r;
    }
    /* If viz is NULL, we just verify that the mrz itself decodes
     * properly and count fields present. */
    if (!viz) {
        if (m.passport_no[0] != '\0') r.passport_no = AC_FIELD_OK, r.ok_count++;
        else r.passport_no = AC_FIELD_MISSING, r.fail_count++;
        if (m.birth_date_yymmdd[0] != '\0') r.birth_date = AC_FIELD_OK, r.ok_count++;
        else r.birth_date = AC_FIELD_MISSING, r.fail_count++;
        if (m.expiry_date_yymmdd[0] != '\0') r.expiry_date = AC_FIELD_OK, r.ok_count++;
        else r.expiry_date = AC_FIELD_MISSING, r.fail_count++;
        if (m.nationality[0] != '\0') r.nationality = AC_FIELD_OK, r.ok_count++;
        else r.nationality = AC_FIELD_MISSING, r.fail_count++;
        if (m.surname[0] != '\0' || m.given_names[0] != '\0')
            r.name = AC_FIELD_OK, r.ok_count++;
        else
            r.name = AC_FIELD_MISSING, r.fail_count++;
        return r;
    }
    /* Compare fields: the VIZ text is plain human-readable; the
     * passport_no, birth_date, expiry_date, nationality should
     * appear in both. We do a case-insensitive substring search. */
    char vbuf[256];
    /* Convert viz to upper-case for comparison. */
    strncpy(vbuf, viz, sizeof(vbuf) - 1);
    vbuf[sizeof(vbuf) - 1] = '\0';
    for (char *c = vbuf; *c; ++c) if (*c >= 'a' && *c <= 'z') *c -= 32;
    /* passport_no: mrz.passport_no. */
    {
        char pn[16]; snprintf(pn, sizeof(pn), "%s", m.passport_no);
        int ok = strstr(vbuf, pn) != NULL;
        r.passport_no = ok ? AC_FIELD_OK : AC_FIELD_MISMATCH;
        if (ok) r.ok_count++; else r.fail_count++;
    }
    /* birth YYMMDD -- viz usually has full YYYY-MM-DD; we accept
     * any of: 6-digit date, or 8-digit with dashes. */
    {
        /* Try several representations: YYMMDD, 19YY-MM-DD, 20YY-MM-DD. */
        char bd[16]; snprintf(bd, sizeof(bd), "%s", m.birth_date_yymmdd);
        /* Pivot: YY >= 60 -> 19YY, else 20YY. */
        int yy_bd = (m.birth_date_yymmdd[0] - '0') * 10 +
                     (m.birth_date_yymmdd[1] - '0');
        int cent_bd = yy_bd >= 60 ? 1900 : 2000;
        char bd_iso[16]; snprintf(bd_iso, sizeof(bd_iso), "%04d-%.2s-%.2s",
                                  cent_bd + yy_bd,
                                  m.birth_date_yymmdd + 2,
                                  m.birth_date_yymmdd + 4);
        char bd_long[16]; snprintf(bd_long, sizeof(bd_long), "19%s", m.birth_date_yymmdd);
        int ok = (strstr(vbuf, bd) != NULL) ||
                 (strstr(vbuf, bd_long) != NULL) ||
                 (strstr(vbuf, bd_iso) != NULL);
        r.birth_date = ok ? AC_FIELD_OK : AC_FIELD_MISMATCH;
        if (ok) r.ok_count++; else r.fail_count++;
    }
    {
        char ed[16]; snprintf(ed, sizeof(ed), "%s", m.expiry_date_yymmdd);
        int yy_ed = (m.expiry_date_yymmdd[0] - '0') * 10 +
                     (m.expiry_date_yymmdd[1] - '0');
        int cent_ed = yy_ed >= 60 ? 1900 : 2000;
        char ed_iso[16]; snprintf(ed_iso, sizeof(ed_iso), "%04d-%.2s-%.2s",
                                  cent_ed + yy_ed,
                                  m.expiry_date_yymmdd + 2,
                                  m.expiry_date_yymmdd + 4);
        char ed_long[16]; snprintf(ed_long, sizeof(ed_long), "20%s", m.expiry_date_yymmdd);
        int ok = (strstr(vbuf, ed) != NULL) ||
                 (strstr(vbuf, ed_long) != NULL) ||
                 (strstr(vbuf, ed_iso) != NULL);
        r.expiry_date = ok ? AC_FIELD_OK : AC_FIELD_MISMATCH;
        if (ok) r.ok_count++; else r.fail_count++;
    }
    {
        int ok = strstr(vbuf, m.nationality) != NULL;
        r.nationality = ok ? AC_FIELD_OK : AC_FIELD_MISMATCH;
        if (ok) r.ok_count++; else r.fail_count++;
    }
    {
        char name[80]; snprintf(name, sizeof(name), "%s", m.surname);
        int ok = strstr(vbuf, name) != NULL;
        r.name = ok ? AC_FIELD_OK : AC_FIELD_MISMATCH;
        if (ok) r.ok_count++; else r.fail_count++;
    }
    return r;
}

/* ----- Anti-counterfeit feature simulation ----- */
ir_absorption_t security_ir_absorption(const face_image_t *visible,
                                       const face_image_t *ir) {
    ir_absorption_t r = {0, 0, 0.0, 0};
    if (!visible || !ir) return r;
    if (visible->width != ir->width || visible->height != ir->height)
        return r;
    int v_dark = 0, i_dark = 0;
    int n = visible->width * visible->height;
    for (int i = 0; i < n; ++i) {
        int vy = (visible->pixels[i*3] +
                  visible->pixels[i*3+1] +
                  visible->pixels[i*3+2]) / 3;
        int iy = (ir->pixels[i*3] +
                  ir->pixels[i*3+1] +
                  ir->pixels[i*3+2]) / 3;
        if (vy < 100) v_dark++;
        if (iy < 100) i_dark++;
    }
    r.visible_dark_count = v_dark;
    r.ir_dark_count = i_dark;
    r.ratio = v_dark > 0 ? (double)i_dark / v_dark : 0;
    r.passes = (r.ratio >= 0.85);
    return r;
}

ssim_match_t security_ssim(const face_image_t *a, const face_image_t *b) {
    ssim_match_t r = {0, 0};
    if (!a || !b) return r;
    if (a->width != b->width || a->height != b->height) return r;
    int n = a->width * a->height;
    double sa = 0, sb = 0;
    for (int i = 0; i < n; ++i) {
        int y = (a->pixels[i*3] + a->pixels[i*3+1] + a->pixels[i*3+2]) / 3;
        int y2 = (b->pixels[i*3] + b->pixels[i*3+1] + b->pixels[i*3+2]) / 3;
        sa += y; sb += y2;
    }
    sa /= n; sb /= n;
    double num = 0, da = 0, db = 0;
    for (int i = 0; i < n; ++i) {
        int y  = (a->pixels[i*3] + a->pixels[i*3+1] + a->pixels[i*3+2]) / 3;
        int y2 = (b->pixels[i*3] + b->pixels[i*3+1] + b->pixels[i*3+2]) / 3;
        double x = y  - sa;
        double y_= y2 - sb;
        num += x * y_;
        da  += x * x;
        db  += y_ * y_;
    }
    double denom = (da + db);
    r.ssim = denom > 0 ? (2 * num) / denom : 0;
    if (r.ssim < 0) r.ssim = 0;
    r.passes = r.ssim >= 0.70;
    return r;
}

uv_check_t security_uv_luminance(const face_image_t *uv) {
    uv_check_t r = {0, 0};
    if (!uv) return r;
    int n = uv->width * uv->height;
    double sum = 0;
    for (int i = 0; i < n; ++i)
        sum += (uv->pixels[i*3] + uv->pixels[i*3+1] + uv->pixels[i*3+2]) / 3;
    r.mean_luminance = sum / n;
    r.passes = r.mean_luminance < 100;
    return r;
}
