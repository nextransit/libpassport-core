/* CNN ConvNet-based MRZ OCR pipeline (grayscale backend).
 *
 * Pipeline:
 *   1. global Otsu on the grayscale page -> coarse bin map
 *   2. locate the MRZ band by row density (shared with backend A)
 *   3. extract the band's GRAYSCALE crop; re-Otsu LOCALLY inside the
 *      band (two-pass threshold: robust to page-level illumination
 *      gradients a global threshold cannot handle)
 *   4. estimate the band tilt from the ink-pixel rotation projection
 *      and deskew (rotate back) when |angle| >= 0.4 deg
 *   5. split the two TD3 lines; segment each line into 44 fixed
 *      pitch-grid windows (run centers only fit pitch/phase; run-based
 *      windows kept as fallback) so one spurious/broken ink run can no
 *      longer shift every subsequent character
 *   6. per cell: area-resample the grayscale window to 16x12 float
 *      (3x3 box smooth + centroid alignment), whole-row batch
 *      conv+fc inference; optional ±1px test-time augmentation
 *      (MRZ_OCR_TTA=1) averages logits over shifts
 *   7. ICAO 9303 syntax mask on the logits (strict TD3 field whitelist
 *      by default; MRZ_OCR_STRICT_ICAO=0 restores the loose mask)
 *   8. deterministic protections: relative ink gate + 3-neighbour '<'
 *      rule + trailing-filler lock (line 1)
 *   9. line-2 checksum beam search (top-3 candidates, lowest-margin
 *      positions first)
 */
/* _GNU_SOURCE must precede ANY system header include:
 * Bionic gates cpu_set_t/CPU_SET in <sched.h> behind __USE_GNU,
 * which is only defined when _GNU_SOURCE is seen first. */
#define _GNU_SOURCE
#include "mrz_ocr.h"
#include "cnn.h"
#include "crnn.h"
#include "template.h"

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#ifdef __linux__
/* Keep <sched.h> ahead of <pthread.h>: bionic's pthread.h includes
 * sched.h, so a later include would be skipped by its #pragma once
 * and cpu_set_t would never be visible. */
#include <sched.h>
#endif
#include <pthread.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---- ICAO 9303 mod-7/3/1 weighted check digit ---- */
static int check_digit_of(const char *seg, int n) {
    int sum = 0;
    static const int wtab[3] = {7, 3, 1};
    for (int i = 0; i < n; ++i) {
        int c = (unsigned char)seg[i];
        int v;
        if (c >= '0' && c <= '9')      v = c - '0';
        else if (c >= 'A' && c <= 'Z') v = c - 'A' + 10;
        else                           v = 0;   /* '<' => 0 */
        sum += v * wtab[i % 3];
    }
    return sum % 10;
}

/* Strict ICAO 9303 TD3 field whitelist by default. MRZ_OCR_STRICT_ICAO=0
 * restores the historical loose mask (kept for legacy free-form
 * corpora / debugging). */
static int strict_icao_mode(void) {
    const char *e = getenv("MRZ_OCR_STRICT_ICAO");
    return !(e && e[0] && strcmp(e, "0") == 0);
}

/* ---- structural sanity of a decoded line 2 (TD3) ----
 *
 * The five check digits prove the 44 characters are internally
 * consistent.  They do NOT prove they are text.  Measured on
 * assets/pic (84 scraped ID photos): the beam closed the checksum on
 * 49 images, of which 35 carried a nationality code that is not a
 * country and 9 carried a birth or expiry date that does not exist.
 * A checksum-valid reading of pure noise is therefore routine, and the
 * caller needs a second, arithmetic-independent veto.
 *
 * Only checks that a genuine TD3 record can never fail are used here.
 * Nationality is deliberately NOT checked against the ISO-3166 list:
 * Germany prints "D<<" and ICAO defines XXA/XXB/XXC/UNO, so a
 * whitelist would reject real passports.  Dates have no such
 * exceptions. */
static int mrz_days_in_month(int y, int m) {
    static const int dm[12] = {31,28,31,30,31,30,31,31,30,31,30,31};
    if (m < 1 || m > 12) return 0;
    if (m == 2 && ((y % 4 == 0 && y % 100 != 0) || y % 400 == 0)) return 29;
    return dm[m - 1];
}

/* Parse YYMMDD into y/m/d.  `is_dob` picks the century window:
 * a birth date cannot be in the future, so YY above the current YY
 * means 19YY.  An expiry date is 20YY unless that would put it more
 * than 25 years ahead of today -- a 1994 expiry is 1994, not 2094, and
 * the naive "YY >= current YY" rule gets exactly that case wrong.
 * Returns 0 if the field is not six digits or not a real date. */
static int mrz_parse_ymd(const char *s, int is_dob, int now_y,
                         int *oy, int *om, int *od) {
    int yy, mm, dd, y;
    for (int i = 0; i < 6; ++i)
        if (s[i] < '0' || s[i] > '9') return 0;
    yy = (s[0] - '0') * 10 + (s[1] - '0');
    mm = (s[2] - '0') * 10 + (s[3] - '0');
    dd = (s[4] - '0') * 10 + (s[5] - '0');
    if (is_dob) y = (yy > now_y % 100) ? 1900 + yy : 2000 + yy;
    else        y = (2000 + yy <= now_y + 25) ? 2000 + yy : 1900 + yy;
    if (mrz_days_in_month(y, mm) == 0 || dd < 1 || dd > mrz_days_in_month(y, mm))
        return 0;
    *oy = y; *om = mm; *od = dd;
    return 1;
}

static int mrz_line2_struct_sane(const char *l, int len) {
    int dy, dm, dd, ey, em, ed;
    time_t now = time(NULL);
    struct tm tmv;
#if defined(_WIN32)
    struct tm *pt = localtime(&now);
    if (!pt) return 0;
    tmv = *pt;
#else
    if (!localtime_r(&now, &tmv)) return 0;
#endif
    int now_y = tmv.tm_year + 1900, now_m = tmv.tm_mon + 1, now_d = tmv.tm_mday;
    if (len != 44) return 0;
    if (!mrz_parse_ymd(l + 13, 1, now_y, &dy, &dm, &dd)) return 0;
    if (!mrz_parse_ymd(l + 21, 0, now_y, &ey, &em, &ed)) return 0;
    /* birth date not in the future */
    if (dy > now_y || (dy == now_y && (dm > now_m || (dm == now_m && dd > now_d))))
        return 0;
    if (dy < 1900) return 0;
    /* expiry not before birth */
    if (ey < dy || (ey == dy && (em < dm || (em == dm && ed < dd)))) return 0;
    /* neither date more than a lifetime away */
    if (ey > now_y + 25) return 0;
    return 1;
}

/* ---- nationality plausibility (advisory) ----
 *
 * ISO-3166-1 alpha-3, plus the ICAO 9303 special codes.  A field that
 * contains '<' is accepted without lookup: Germany prints "D<<" and
 * several states print a partial code rather than the full alpha-3.
 *
 * This is ADVISORY, not a veto.  Kosovo prints "RKS", which is not
 * ISO, and a future state will print something else that is not in
 * this table either.  But on the assets corpus 19 of the 28 readings
 * that satisfied all five check digits carried a nationality code that
 * no state issues ("HOL", "UKH", "HUS", "UST", "KSP", "DMN", "SYH"),
 * so the flag is worth surfacing: a caller that rejects on it trades a
 * few false rejects for a large cut in silent wrong output.
 *
 * The special-code list matters as much as the ISO list: without it
 * this check would flag a Kosovo passport ("RKS"), an EU laissez-passer
 * ("EUE"), an Interpol travel document ("XPO") and a Sovereign Military
 * Order of Malta passport ("XOM") as having an impossible
 * nationality.  The full list comes from ICAO's MRTD state code
 * registry; "UTO" is included because the ICAO specimen documents and
 * most parser libraries use it. */
static const char k_iso3[] =
    "AFGALBDZAANDAGOATGARGARMAUSAUTAZEBHSBHRBGDBRBBLRBELBLZBENBTN"
    "BOLBIHBWABRABRNBGRBFABDIKHMCMRCANCPVCAFTCDCHLCHNCOLCOMCOGCOD"
    "CRICIVHRVCUBCYPCZEDNKDJIDMADOMECUEGYSLVGNQERIESTSWZETHFJIFIN"
    "FRAGABGMBGEODEUGHAGRCGRDGTMGINGNBGUYHTIHNDHUNISLINDIDNIRNIRQ"
    "IRLISRITAJAMJPNJORKAZKENKIRPRKKORKWTKGZLAOLVALBNLSOLBRLBYLIE"
    "LTULUXMDGMWIMYSMDVMLIMLTMHLMRTMUSMEXFSMMDAMCOMNGMNEMARMOZMMR"
    "NAMNRUNPLNLDNZLNICNERNGAMKDNOROMNPAKPLWPANPNGPRYPERPHLPOLPRT"
    "QATROURUSRWAKNALCAVCTWSMSMRSTPSAUSENSRBSYCSLESGPSVKSVNSLBSOM"
    "ZAFSSDESPLKASDNSURSWECHESYRTJKTZATHATLSTGOTONTTOTUNTURTKMTUV"
    "UGAUKRAREGBRUSAURYUZBVUTVATVENVNMYEMZMBZWEPSETWNHKGMACXXAXXB"
    "XXCXXXUNO"
    /* ICAO MRTD special codes (issuing state / nationality). */
    "BAHZIMGBDGBNGBOGBPGBSEUEXBAXCCXCOXECXIMXOMXPOUNAUNKRKSXKX"
    "XKKWSAXCTUTO";

/* The table is a hand-written blob of concatenated 3-letter codes, so a
 * single typo shifts every code after it and silently turns a real
 * passport's code into "no state issues this".  That happened once
 * ("UNAVUNK" for "UNA"/"UNK") and the unit test caught it.  Make the
 * invariant impossible to break silently. */
typedef char mrz_iso3_len_is_multiple_of_3
    [((sizeof(k_iso3) - 1) % 3 == 0) ? 1 : -1];

/* Is the 3-letter field at `off` a code some state or organisation
 * actually issues?  1 = plausible, 0 = no state issues it.
 *
 * A field containing '<' is always plausible: Germany prints "D<<" and
 * several states print a partial code rather than the full alpha-3. */
static int mrz_code_plausible(const char *l, int len, int off) {
    int i, n;
    if (len != 44) return 1;          /* not a TD3 line: nothing to judge */
    for (i = off; i < off + 3; ++i)
        if (l[i] == '<') return 1;     /* "D<<" and friends */
    n = (int)(sizeof(k_iso3) - 1) / 3;
    for (i = 0; i < n; ++i)
        if (k_iso3[i * 3] == l[off] && k_iso3[i * 3 + 1] == l[off + 1] &&
            k_iso3[i * 3 + 2] == l[off + 2]) return 1;
    return 0;
}

/* Same question for a bare 3-character buffer.
 *
 * Split out so the line-1 issuing-state repair (below) can test a
 * candidate triple without materialising it into a full 44-char line
 * first.  Identical semantics to mrz_code_plausible: '<' anywhere means
 * "a state prints a partial code", which is always accepted. */
static int mrz_code3_plausible(const char c3[3]) {
    int i, n;
    for (i = 0; i < 3; ++i)
        if (c3[i] == '<') return 1;
    n = (int)(sizeof(k_iso3) - 1) / 3;
    for (i = 0; i < n; ++i)
        if (k_iso3[i * 3] == c3[0] && k_iso3[i * 3 + 1] == c3[1] &&
            k_iso3[i * 3 + 2] == c3[2]) return 1;
    return 0;
}

/* The nationality in line 2 (cols 10..12). */
static int mrz_nat_suspect(const char *l, int len) {
    return len == 44 ? !mrz_code_plausible(l, len, 10) : 0;
}

/* Same question about the ISSUING STATE in line 1 (cols 2..4).
 *
 * The asymmetry this fixes: `mrz_nat_suspect` has always checked the
 * nationality in line 2, but nothing ever checked the issuing state in
 * line 1.  That is the wrong way round -- line 2 carries five check
 * digits and line 1 carries none, so line 1 is the row that most needs
 * a content test, and it had none.
 *
 * On the 41-photo phone corpus the four images whose line 1 was wrong
 * all carried a fabricated issuing-state code ("HUG", "CHY", "CNN",
 * "ATA") that no state issues, while every hand-read ground truth in
 * both corpora carried a real ISO-3166-1 alpha-3 code.  This check
 * finds them on its own, without reference to line 2.
 *
 * ADVISORY for the same reason as the nationality check: Kosovo prints
 * "RKS", which is not ISO, and a state may print a partial code.  A
 * field containing '<' is never flagged. */
static int mrz_issuer_suspect(const char *l1, int len1) {
    return len1 == 44 ? !mrz_code_plausible(l1, len1, 2) : 0;
}

/* Cross-line consistency: the issuing state printed in line 1 (cols
 * 2..4) versus the nationality printed in line 2 (cols 10..12).
 *
 * On the 41-photo phone corpus this was the only signal that caught a
 * single-character error in line 1: it flagged 4 images and all 4 were
 * wrong, while the line-1 confidence and the line-2 ambiguity flag
 * together caught none.  One of the flagged images had a wrong line 1
 * and still scored 98 confidence, so confidence is not a substitute.
 *
 * Returns 0 when the comparison does not apply (a line is not 44
 * characters, or either field contains '<') so that it never fires on
 * forms that legitimately differ. */
static int mrz_line1_country_mismatch(const char *l1, int len1,
                                       const char *l2, int len2) {
    int i;
    if (!l1 || !l2) return 0;
    if (len1 != 44 || len2 != 44) return 0;
    for (i = 2; i < 5; ++i) {
        if (l1[i] == '<') return 0;   /* "D<<" and friends */
        if (l2[10 + (i - 2)] == '<') return 0;
    }
    if (l1[2] == l2[10] && l1[3] == l2[11] && l1[4] == l2[12]) return 0;
    return 1;
}

/* Public wrappers (declared in mrz_ocr.h).  The internals take an
 * explicit length so the recogniser can pass out->line2_len; callers
 * outside the TU only ever hold a NUL-terminated string. */
int mrz_ocr_line2_struct_ok(const char *line2) {
    if (!line2) return 0;
    return mrz_line2_struct_sane(line2, (int)strlen(line2));
}

int mrz_ocr_line2_nat_suspect(const char *line2) {
    if (!line2) return 0;
    return mrz_nat_suspect(line2, (int)strlen(line2));
}

int mrz_ocr_line1_country_mismatch(const char *line1, const char *line2) {
    if (!line1 || !line2) return 0;
    return mrz_line1_country_mismatch(line1, (int)strlen(line1),
                                      line2, (int)strlen(line2));
}

int mrz_ocr_line1_issuer_suspect(const char *line1) {
    if (!line1) return 0;
    return mrz_issuer_suspect(line1, (int)strlen(line1));
}


/* ---- top-K candidates among the allowed classes ----
 * K=5 gives the checksum beam search enough alternatives to recover
 * characters that the CNN confuses with visually similar digits
 * (D↔6, B↔0, A↔3, M↔H) in the personal-number field.  With K=3
 * the correct character was sometimes absent from the candidate list
 * and no amount of beam depth could rescue it. */
#define TOPK 5
typedef struct { int cand[TOPK]; float p[TOPK]; } top3_t;

static void top3_of(const float *probs, int allow[37], top3_t *t) {
    for (int k = 0; k < TOPK; ++k) { t->cand[k] = -1; t->p[k] = -1.0f; }
    for (int o = 0; o < 37; ++o) {
        if (allow[o] == 0) continue;
        float p = probs[o];
        /* insertion sort into the top-K array (descending) */
        int k = TOPK - 1;
        if (p > t->p[k]) {
            t->p[k] = p; t->cand[k] = o;
            while (k > 0 && t->p[k] > t->p[k-1]) {
                float tp = t->p[k]; t->p[k] = t->p[k-1]; t->p[k-1] = tp;
                int tc = t->cand[k]; t->cand[k] = t->cand[k-1]; t->cand[k-1] = tc;
                k--;
            }
        }
    }
    int lt = mrz_ocr_glyph_index('<');
    if (t->cand[0] < 0) {
        for (int k = 0; k < TOPK; ++k) { t->cand[k] = lt; t->p[k] = 0.0f; }
    } else {
        for (int k = 1; k < TOPK; ++k) {
            if (t->cand[k] < 0) { t->cand[k] = t->cand[0]; t->p[k] = 0.0f; }
        }
    }
}

/* ---- per-position character whitelist (syntax mask) ---- */
/* Set by mrz_ocr_recognise_cnn after layout classification; read by
 * fill_allow and decode_row to switch between TD3 and CS/CD masks. */
static mrz_doc_layout_t g_doc_layout = MRZ_LAYOUT_TD3_DOUBLE_44;

static void fill_allow(int allow[37], int line_idx, int col) {
    int d0 = mrz_ocr_glyph_index('0');
    int a0 = mrz_ocr_glyph_index('A');
    int lt = mrz_ocr_glyph_index('<');
    for (int i = 0; i < 37; ++i) allow[i] = 1;

    /* CS/CD single-line 30-char mask (港澳台通行证). */
    if (g_doc_layout == MRZ_LAYOUT_SINGLE_LINE_30) {
        if (col == 0) {
            for (int i = 0; i < 37; ++i) allow[i] = 0;
            allow[mrz_ocr_glyph_index('C')] = 1;
        } else if (col == 1) {
            for (int i = 0; i < 37; ++i) allow[i] = 0;
            allow[mrz_ocr_glyph_index('S')] = 1;
            allow[mrz_ocr_glyph_index('D')] = 1;
        } else if (col == 2) {
            for (int i = 0; i < 37; ++i) allow[i] = 0;
            allow[mrz_ocr_glyph_index('C')] = 1;
            allow[lt] = 1;
        } else if (col == 3) {
            /* A-Z excluding I,O + 0-9 */
            for (int i = 0; i < 37; ++i) allow[i] = 0;
            for (int i = 0; i < 26; ++i) {
                char ch = 'A' + i;
                if (ch == 'I' || ch == 'O') continue;
                allow[a0 + i] = 1;
            }
            for (int i = 0; i < 10; ++i) allow[d0 + i] = 1;
        } else if (col == 12 || col == 20 || col == 28) {
            for (int i = 0; i < 37; ++i) allow[i] = (i == lt);
        } else {
            /* All other positions: digits only */
            for (int i = 0; i < 37; ++i)
                allow[i] = (i >= d0 && i <= d0 + 9) ? 1 : 0;
        }
        return;
    }

    if (!strict_icao_mode()) {
        /* Loose legacy mask: numeric check digits, M/F/< sex, and the
         * line-1 col-1 filler only. Kept for legacy free-form corpora
         * that intentionally mix digits into alpha fields. */
        if (line_idx == 0) {
            if (col == 1) {
                for (int i = 0; i < 37; ++i) allow[i] = (i == lt);
            }
        } else {
            if (col == 9 || col == 19 || col == 27) {
                for (int i = 0; i < 37; ++i)
                    allow[i] = (i >= d0 && i <= d0 + 9) ? 1 : 0;
            } else if (col == 42 || col == 43) {
                for (int i = 0; i < 37; ++i)
                    allow[i] = (i >= d0 && i <= d0 + 9) ? 1 : 0;
                allow[lt] = 1;
            } else if (col == 20) {
                for (int i = 0; i < 37; ++i) allow[i] = 0;
                allow[mrz_ocr_glyph_index('M')] = 1;
                allow[mrz_ocr_glyph_index('F')] = 1;
                allow[lt] = 1;
            }
        }
        return;
    }

    /* Strict ICAO 9303 TD3 field mask. Every slot of a real passport
     * MRZ falls into exactly one of these sets, so the mask always
     * contains the ground truth and only removes impossible classes. */
    if (line_idx == 0) {
        if (col == 0) {
            /* ICAO 9303 document type: A (alien/ID), C (card/travel
             * permit), I (ID card), P (passport), V (visa).  Restricting
             * to P alone silently discards ID cards and travel permits. */
            for (int i = 0; i < 37; ++i) allow[i] = 0;
            allow[mrz_ocr_glyph_index('A')] = 1;
            allow[mrz_ocr_glyph_index('C')] = 1;
            allow[mrz_ocr_glyph_index('I')] = 1;
            allow[mrz_ocr_glyph_index('P')] = 1;
            allow[mrz_ocr_glyph_index('V')] = 1;
        } else if (col == 1) {
            /* Document type qualifier: '<' (normal P<) or A-Z subtype
             * (PO specimen, PA official, PD diplomatic, etc.).  ICAO
             * 9303 allows any capital letter here; restricting to '<'
             * alone silently discards specimen passports. */
            for (int i = 0; i < 37; ++i)
                allow[i] = (i == lt) || (i >= a0 && i < a0 + 26);
        } else if (col >= 2 && col <= 4) {
            /* Issuing state: 3 alpha */
            for (int i = 0; i < 37; ++i) allow[i] = (i >= a0 && i < a0 + 26);
        } else if (col >= 5) {
            /* Name field: A-Z + '<' (ICAO 9303 has no digits here) */
            for (int i = 0; i < 37; ++i)
                allow[i] = (i >= a0 && i < a0 + 26) || i == lt;
        }
        /* col 0 handled above: standard ICAO doc-type letters only. */
    } else {
        if (col <= 8) {
            /* Passport number: alnum + '<' filler */
            for (int i = 0; i < 37; ++i)
                allow[i] = (i >= a0 && i < a0 + 26) || (i >= d0 && i <= d0 + 9) || i == lt;
        } else if (col == 9 || col == 19 || col == 27) {
            for (int i = 0; i < 37; ++i)
                allow[i] = (i >= d0 && i <= d0 + 9) ? 1 : 0;
        } else if (col <= 12) {
            /* Nationality: alpha + '<' (unspecified nationality) */
            for (int i = 0; i < 37; ++i)
                allow[i] = (i >= a0 && i < a0 + 26) || i == lt;
        } else if ((col <= 18) || (col >= 21 && col <= 26)) {
            /* Birth / expiry dates: numeric ('<' = unknown date).
             * Some specimen/test docs have letters here, allow A-Z too. */
            for (int i = 0; i < 37; ++i)
                allow[i] = ((i >= d0 && i <= d0 + 9) || i == lt ||
                            (i >= a0 && i < a0 + 26)) ? 1 : 0;
        } else if (col == 20) {
            /* Sex: M / F / '<' plus alnum (relaxed for specimen docs). */
            for (int i = 0; i < 37; ++i)
                allow[i] = (i >= a0 && i < a0 + 26) ||
                           (i >= d0 && i <= d0 + 9) || i == lt;
        } else if (col <= 41) {
            /* Personal number: alnum + '<' */
            for (int i = 0; i < 37; ++i)
                allow[i] = (i >= a0 && i < a0 + 26) || (i >= d0 && i <= d0 + 9) || i == lt;
        } else {
            /* col 42/43: check digits, '<' tolerated for the tail-drift
             * case; the checksum beam prefers digits when valid. */
            for (int i = 0; i < 37; ++i)
                allow[i] = ((i >= d0 && i <= d0 + 9) || i == lt) ? 1 : 0;
        }
    }
}

static int line2_checksum_ok(const char *l2) {
    char buf[15];
    memcpy(buf, l2, 9); buf[9] = 0;
    if (check_digit_of(buf, 9) != (l2[9] - '0')) return 0;
    memcpy(buf, l2 + 13, 6); buf[6] = 0;
    if (check_digit_of(buf, 6) != (l2[19] - '0')) return 0;
    memcpy(buf, l2 + 21, 6); buf[6] = 0;
    if (check_digit_of(buf, 6) != (l2[27] - '0')) return 0;
    memcpy(buf, l2 + 28, 14); buf[14] = 0;
    if (check_digit_of(buf, 14) != (l2[42] - '0')) return 0;
    char comp[41];
    memcpy(comp, l2, 10);            /* doc no + ck           (0..9)  */
    memcpy(comp + 10, l2 + 13, 7);   /* birth + ck            (13..19) */
    memcpy(comp + 17, l2 + 21, 22);  /* expiry + ck + personal (21..42) */
    comp[39] = 0;
    if (check_digit_of(comp, 39) != (l2[43] - '0')) return 0;
    return 1;
}

/* ---- P0-3 diagnostic dump (MRZ_OCR_DIAG=<path>) ----
 *
 * Pure observer: every field copies a value the pipeline already computed.
 * One JSON object is written per recogniser call. */
#define MRZ_DIAG_CHARS 44

typedef struct {
    int valid;
    int y, h;
    int grid_ok;
    int nchars;
    int seg_score;
    int nrun, medw, cell, win_w, Lmin, Lmax, L0;
    double Lfit, pitch, phase;
    int cost;
    double int_pitch, frac_pitch;
    int int_cost, frac_cost;
    double guard;
    int conf_avg;
    int ambiguous, alt_margin;
    char alt[64];
    unsigned char gate[MRZ_DIAG_CHARS];
    unsigned char ch[MRZ_DIAG_CHARS];
    int cand[MRZ_DIAG_CHARS][TOPK];
    float p[MRZ_DIAG_CHARS][TOPK];
    float ink[MRZ_DIAG_CHARS];
} mrz_diag_line_t;

typedef struct {
    int enabled;
    char path[512];
    int img_w, img_h;
    int band_x, band_y, band_w, band_h;
    int proc_w, proc_h;
    double upscale;
    double skew_deg;
    int status;
    double ms_gray, ms_band, ms_resample, ms_decode;
    mrz_diag_line_t line[2];
} mrz_diag_t;

static mrz_diag_t g_diag;

static int mrz_diag_on(void) {
    static int inited = 0;
    if (!inited) {
        const char *e = getenv("MRZ_OCR_DIAG");
        if (e && e[0]) {
            snprintf(g_diag.path, sizeof(g_diag.path), "%s", e);
            g_diag.enabled = 1;
        }
        inited = 1;
    }
    return g_diag.enabled;
}

static mrz_diag_line_t *mrz_diag_line(int li) {
    if (!mrz_diag_on() || li < 0 || li > 1) return NULL;
    return &g_diag.line[li];
}

static const char *mrz_diag_status_name(int s) {
    switch (s) {
    case MRZ_OCR_OK:                 return "OK";
    case MRZ_OCR_ERR_LOAD:           return "ERR_LOAD";
    case MRZ_OCR_ERR_NO_BAND:        return "ERR_NO_BAND";
    case MRZ_OCR_ERR_BAD_GEOMETRY:   return "ERR_BAD_GEOMETRY";
    case MRZ_OCR_ERR_BAD_LINES:      return "ERR_BAD_LINES";
    case MRZ_OCR_ERR_LOW_CONFIDENCE: return "ERR_LOW_CONFIDENCE";
    case MRZ_OCR_ERR_TEMPLATE:       return "ERR_TEMPLATE";
    case MRZ_OCR_ERR_TIMEOUT:        return "ERR_TIMEOUT";
    }
    return "ERR_UNKNOWN";
}

static void mrz_diag_json_str(FILE *f, const char *s) {
    fputc('"', f);
    for (; s && *s; ++s) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') { fputc('\\', f); fputc(c, f); }
        else if (c < 0x20)         fprintf(f, "\\u%04x", c);
        else                       fputc(c, f);
    }
    fputc('"', f);
}

static void mrz_diag_checksums(const char *l2, int len,
                               int exp[5], int act[5], int ok[5]) {
    static const int pos[5] = {9, 19, 27, 42, 43};
    static const int start[4] = {0, 13, 21, 28};
    static const int nseg[4] = {9, 6, 6, 14};
    for (int i = 0; i < 5; ++i) { exp[i] = -1; act[i] = -1; ok[i] = 0; }
    if (!l2 || len < 44) return;
    for (int i = 0; i < 4; ++i) {
        exp[i] = check_digit_of(l2 + start[i], nseg[i]);
        act[i] = (l2[pos[i]] >= '0' && l2[pos[i]] <= '9') ? l2[pos[i]] - '0' : -1;
        ok[i] = (act[i] == exp[i]);
    }
    char comp[40];
    memcpy(comp, l2, 10);
    memcpy(comp + 10, l2 + 13, 7);
    memcpy(comp + 17, l2 + 21, 22);
    exp[4] = check_digit_of(comp, 39);
    act[4] = (l2[43] >= '0' && l2[43] <= '9') ? l2[43] - '0' : -1;
    ok[4] = (act[4] == exp[4]);
}

static void mrz_diag_write(const mrz_ocr_result_t *out) {
    if (!mrz_diag_on()) return;
    FILE *f = fopen(g_diag.path, "w");
    if (!f) return;

    fprintf(f, "{\n \"version\":1,\n");
    fprintf(f, " \"status\":%d,\n \"status_name\":\"%s\",\n",
            g_diag.status, mrz_diag_status_name(g_diag.status));
    fprintf(f, " \"image\":{\"w\":%d,\"h\":%d},\n", g_diag.img_w, g_diag.img_h);
    fprintf(f, " \"band\":{\"x\":%d,\"y\":%d,\"w\":%d,\"h\":%d,"
               "\"proc_w\":%d,\"proc_h\":%d,\"upscale\":%.3f},\n",
            g_diag.band_x, g_diag.band_y, g_diag.band_w, g_diag.band_h,
            g_diag.proc_w, g_diag.proc_h, g_diag.upscale);
    fprintf(f, " \"skew_deg\":%.4f,\n", g_diag.skew_deg);
    fprintf(f, " \"line1\":");
    mrz_diag_json_str(f, out->line1);
    fprintf(f, ",\n \"line2\":");
    mrz_diag_json_str(f, out->line2);
    fprintf(f, ",\n");

    fprintf(f, " \"lines\":[\n");
    for (int li = 0; li < 2; ++li) {
        const mrz_diag_line_t *d = &g_diag.line[li];
        int n = d->nchars;
        if (n < 0) n = 0;
        if (n > MRZ_DIAG_CHARS) n = MRZ_DIAG_CHARS;
        fprintf(f, "  {\"index\":%d,\"y\":%d,\"h\":%d,\"grid_ok\":%d,"
                   "\"seg_score\":%d,\"nchars\":%d,\"nrun\":%d,\"medw\":%d,"
                   "\"cell\":%d,\"win_w\":%d,\"Lmin\":%d,\"Lmax\":%d,\"L0\":%d,",
                li, d->y, d->h, d->grid_ok, d->seg_score, n, d->nrun, d->medw,
                d->cell, d->win_w, d->Lmin, d->Lmax, d->L0);
        fprintf(f, "\"Lfit\":%.3f,\"pitch\":%.3f,\"phase\":%.3f,\"cost\":%d,"
                   "\"int_pitch\":%.2f,\"int_cost\":%d,\"frac_pitch\":%.2f,"
                   "\"frac_cost\":%d,\"guard\":%.2f,\"conf_avg\":%d,"
                   "\"ambiguous\":%d,\"alt_margin\":%d,\"alt\":",
                d->Lfit, d->pitch, d->phase, d->cost,
                d->int_pitch, d->int_cost, d->frac_pitch, d->frac_cost,
                d->guard, d->conf_avg, d->ambiguous, d->alt_margin);
        mrz_diag_json_str(f, d->alt);
        fprintf(f, ",\"chars\":[");
        for (int c = 0; c < n; ++c) {
            if (c) fputc(',', f);
            fprintf(f, "{\"i\":%d,\"top\":[", c);
            for (int k = 0; k < TOPK; ++k) {
                if (k) fputc(',', f);
                int g = d->cand[c][k];
                if (g >= 0 && g < 37) {
                    fprintf(f, "[\"%c\",%.4f]",
                            MRZ_OCR_GLYPHS[g].ch, d->p[c][k]);
                } else {
                    fprintf(f, "[null,%.4f]", d->p[c][k]);
                }
            }
            fprintf(f, "],\"ink\":%.1f,\"gate\":%u,\"final\":\"%c\"}",
                    d->ink[c], (unsigned)d->gate[c],
                    d->ch[c] ? d->ch[c] : '<');
        }
        fprintf(f, "]}%s\n", li == 0 ? "," : "");
    }
    fprintf(f, " ],\n");

    {
        int exp[5], act[5], ok[5];
        static const char *names[5] =
            {"passport", "birth", "expiry", "personal", "composite"};
        int l2len = (int)strlen(out->line2);
        if (l2len > 44) l2len = 44;
        mrz_diag_checksums(out->line2, l2len, exp, act, ok);
        fprintf(f, " \"checksums\":{");
        for (int i = 0; i < 5; ++i) {
            fprintf(f, "%s\"%s\":{\"expected\":%d,\"actual\":%d,\"ok\":%s}",
                    i ? "," : "", names[i], exp[i], act[i],
                    ok[i] ? "true" : "false");
        }
        fprintf(f, "},\n");
    }

    fprintf(f, " \"flags\":{\"struct_ok\":%d,\"nat_suspect\":%d,"
               "\"country_mismatch\":%d,\"issuer_suspect\":%d},\n",
            out->line2_struct_ok, out->line2_nat_suspect,
            out->line1_country_mismatch, out->line1_issuer_suspect);
    fprintf(f, " \"timing_ms\":{\"gray\":%.2f,\"band\":%.2f,"
               "\"resample\":%.2f,\"decode\":%.2f}\n",
            g_diag.ms_gray, g_diag.ms_band,
            g_diag.ms_resample, g_diag.ms_decode);
    fprintf(f, "}\n");
    fclose(f);
}

/* ---- grayscale area-resample of a char box to 16x12 float ---- */
/* ---- band upscaling (scale normalisation) ----
 *
 * Measured on assets/pic: the MRZ band arrives at a median of 19.4 px per
 * character column (band_w / 44), against 32.7 on the device corpus -- while
 * the line HEIGHT is essentially the same on both (45.5 vs 46.0 px).  Line
 * height is therefore the wrong knob: binned by height the checksum pass
 * rate is non-monotonic (4% for bands >=100px tall), binned by
 * px-per-character it is not.
 *
 * Upscaling the band to >= MRZ_OCR_TARGET_PPC px per character raises the
 * checksum pass rate on assets/pic from 12/84 to 30/84 and leaves the device
 * corpus untouched (it already sits at 32.7, so the factor is exactly 1.0).
 *
 * Mechanism (measured, not assumed): the CNN sees a 16x12 resample of one
 * cell window, so what matters is the INK-TO-WINDOW RATIO inside that window.
 * Training renders ink of width 6s inside a window of 6s+8 for s = 3..6, i.e.
 * a ratio of 0.69..0.82.  Inference sets the window to `cell = medw + 8`,
 * where medw is the measured median ink-run width -- an ABSOLUTE 8 px of
 * padding, not a proportion.  Measured with MRZ_OCR_GRID=1:
 *
 *     device corpus   32.7 px/char   medw 20  cell 28   ratio 0.71   (in range)
 *     assets/pic raw  19.8 px/char   medw  8  cell 16   ratio 0.50   (out of range)
 *     assets/pic up    26  px/char   medw 18  cell 26   ratio 0.69   (in range)
 *
 * So the model is being shown glyphs with ~50% more relative margin than it
 * ever saw, and upscaling is what brings the ratio back.  Two checks confirm
 * this is the cause rather than the 44-cell grid phase: (a) the "44 characters
 * recovered" rate barely moves (67/84 -> 69/84) while the checksum rate more
 * than doubles, and (b) shifting the crop by up to 35% of a cell at the
 * CORRECT ratio costs only 17/18 -> 14/18 with character accuracy unchanged
 * at ~98%.  A phase-basin story would show the opposite pattern.
 *
 * The principled fix is to decouple the two roles of `cell`: keep medw + 8 for
 * the WINDOW (it must stay >= the true pitch or glyphs clip) and hand
 * `resample_char_gray` a narrower, proportional sub-rect (width ~ medw * 4/3,
 * the ratio 0.75 that 6s+8 equals at s = 4), with the training renderer
 * changed to match.  That needs a retrain, so it is not done here.
 *
 * Do NOT use line height as the criterion, and do NOT downscale: forcing the
 * device corpus to a fixed line height costs it 17/18 -> 15/18, a fixed
 * vertical factor of 1.5 costs it 17/18 -> 12/18, and a fixed target of
 * 36 px/char costs it 17/18 -> 5/18.
 */
#define MRZ_OCR_TARGET_PPC 26.0

static void mrz_upscale_gray(const uint8_t *src, int sw, int sh,
                             uint8_t *dst, int dw, int dh) {
    if (!src || !dst || sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0) return;
    for (int y = 0; y < dh; ++y) {
        double fy = ((double)y + 0.5) * (double)sh / (double)dh - 0.5;
        int y0 = (int)floor(fy);
        double wy = fy - (double)y0;
        if (y0 < 0) { y0 = 0; wy = 0.0; }
        int y1 = y0 + 1; if (y1 > sh - 1) y1 = sh - 1;
        for (int x = 0; x < dw; ++x) {
            double fx = ((double)x + 0.5) * (double)sw / (double)dw - 0.5;
            int x0 = (int)floor(fx);
            double wx = fx - (double)x0;
            if (x0 < 0) { x0 = 0; wx = 0.0; }
            int x1 = x0 + 1; if (x1 > sw - 1) x1 = sw - 1;
            double a = (double)src[y0 * sw + x0] * (1.0 - wx) +
                       (double)src[y0 * sw + x1] * wx;
            double b = (double)src[y1 * sw + x0] * (1.0 - wx) +
                       (double)src[y1 * sw + x1] * wx;
            dst[y * dw + x] = (uint8_t)(a * (1.0 - wy) + b * wy + 0.5);
        }
    }
}

/* ---- glyph window mode ----
 * MRZ_OCR_CELL_WIN=<f> : window width = f * fitted pitch (0/unset = legacy)
 * MRZ_OCR_CELL=pitch090: sample the central 90% of the window
 *
 * What the CNN cares about is ink/window: training renders ink 6s inside a
 * window 6s+8 for s = 3..6, i.e. 0.69..0.82.  The legacy rule builds the
 * window from the measured INK width with an absolute +8 px, so
 *
 *     ink/window = 1 / (1 + 8/(medw/pitch * pitch))
 *
 * and medw/pitch is a FONT constant, measured at 0.67 on all three of our
 * corpora regardless of resolution (device 20/29.95, assets 13/19.28,
 * assets-upscaled 18/25.76).  Substituting 0.67 gives ink/window =
 * 1/(1+11.9/pitch): 0.61 at pitch 19, 0.69 at 26, 0.71 at 30.  So `medw + 8`
 * only enters the training band once the pitch is at least ~26 px -- which is
 * exactly why the scale-normalisation target of 26 works, and why the same
 * formula fails at 19.
 *
 * The window therefore has TWO constraints that pull opposite ways:
 *   ink/window >= 0.69  =>  window <= 0.97 * pitch
 *   window must not clip  =>  window >= ~0.92 * pitch
 * so in principle window/pitch should sit in [0.92, 0.97].  Legacy hits 0.94
 * at pitch 30 and 1.09 at pitch 19.
 *
 * MEASURED OUTCOME (do not repeat this experiment expecting a win): a window
 * derived from the pitch, at every factor in 0.90..1.00, makes the device
 * corpus WORSE than legacy --
 *
 *     factor   0.90 0.92 0.94 0.96 0.98 1.00   legacy
 *     exact     8   15    9   10   12    7      17     (of 18)
 *     char    95.3 98.1 95.6 95.7 97.0 93.9   98.48   (%)
 *
 * The response is not smooth, which is the tell: `chars[].w` is not only the
 * sampling window, it is the width the CNN's ink statistic is computed over,
 * and decode_row's relative '<' gate (0.55 x median confident-letter ink) is
 * calibrated on the glyphs the legacy window produces.  Changing the window
 * shifts that statistic and flips borderline cells to '<', which is why the
 * failures show up as spurious '<' rather than as gradual degradation.
 *
 * So the window width, the trained weights and the ink gate are ONE
 * calibration.  Fixing the ratio therefore requires changing the inference
 * window AND the training renderer AND re-checking the gate together; a
 * partial rollout is strictly worse than leaving it alone.  These env knobs
 * exist so that pairing can be validated, not as a switch to flip. */
static double mrz_cell_win_factor(void) {
    static double f = -1.0;
    if (f < 0.0) {
        const char *e = getenv("MRZ_OCR_CELL_WIN");
        f = 0.0;
        if (e && e[0]) {
            double v = atof(e);
            if (v >= 0.60 && v <= 1.60) f = v;
        }
    }
    return f;
}

static int mrz_cell_mode(void) {
    static int mode = -1;
    if (mode < 0) {
        const char *e = getenv("MRZ_OCR_CELL");
        mode = 0;
        if (e && e[0]) {
            if (strcmp(e, "pitch") == 0)         mode = 1;
            else if (strcmp(e, "pitch090") == 0) mode = 2;
        }
    }
    return mode;
}

static void resample_char_gray(const uint8_t *gray, int W, int H,
                               int rx, int ry, int rw, int rh,
                               uint8_t *sm, int sm_cap,
                               float out[CNN_IN_H][CNN_IN_W]) {
    (void)sm; (void)sm_cap;
    if (rw <= 0 || rh <= 0) return;
    /* pitch090: trim to the central 90% so the ink fills 0.67/0.90 = 0.74 of
     * the sampled window, the same ratio the training renderer produces at
     * s = 4.  Only horizontal: vertically the training ink already fills the
     * box, and `rh` is the line box, so trimming there would clip ascenders. */
    if (mrz_cell_mode() == 2 && rw > 9) {
        int gw = (rw * 9 + 5) / 10;
        if (gw < 4) gw = rw;
        rx += (rw - gw) / 2;
        rw = gw;
    }
    /* area-average the raw grayscale box to 16x12 (soft edges). */
    for (int oy = 0; oy < CNN_IN_H; ++oy) {
        int sy0 = (oy * rh) / CNN_IN_H;
        int sy1 = ((oy + 1) * rh) / CNN_IN_H;
        if (sy1 <= sy0) sy1 = sy0 + 1;
        for (int ox = 0; ox < CNN_IN_W; ++ox) {
            int sx0 = (ox * rw) / CNN_IN_W;
            int sx1 = ((ox + 1) * rw) / CNN_IN_W;
            if (sx1 <= sx0) sx1 = sx0 + 1;
            int sum = 0, cnt = 0;
            for (int sy = sy0; sy < sy1; ++sy) {
                int yy = ry + sy;
                if (yy < 0 || yy >= H) continue;
                for (int sx = sx0; sx < sx1; ++sx) {
                    int xx = rx + sx;
                    if (xx < 0 || xx >= W) continue;
                    sum += gray[yy * W + xx];
                    cnt++;
                }
            }
            float ink = cnt > 0 ? (float)sum / cnt : 0.0f;
            out[oy][ox] = 1.0f - ink / 255.0f;
        }
    }
    /* Lightweight 3x3 box on the final 16x12 (averages isolated
     * salt/pepper pixels far cheaper than a full-ROI presmooth). */
    {
        float tmp[CNN_IN_H][CNN_IN_W];
        memcpy(tmp, out, sizeof(tmp));
        for (int y = 0; y < CNN_IN_H; ++y)
            for (int x = 0; x < CNN_IN_W; ++x) {
                float acc = 0; int c = 0;
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        int ny = y + dy, nx = x + dx;
                        if (ny >= 0 && ny < CNN_IN_H && nx >= 0 && nx < CNN_IN_W) {
                            acc += tmp[ny][nx]; c++;
                        }
                    }
                out[y][x] = acc / (float)(c ? c : 1);
            }
    }
    /* X-axis centroid alignment (clamped +-2 px). */
    double sx = 0.0, sw = 0.0;
    for (int y = 0; y < CNN_IN_H; ++y)
        for (int x = 0; x < CNN_IN_W; ++x) {
            float v = out[y][x];
            if (v > 0.35f) { sx += (double)x * (double)v; sw += v; }
        }
    if (sw > 1e-3) {
        double cx = sx / sw;
        double target = (CNN_IN_W - 1) * 0.5;
        int delta = (int)llround(cx - target);
        if (delta < -2) delta = -2;
        if (delta > 2) delta = 2;
        if (delta != 0) {
            float tmp[CNN_IN_H][CNN_IN_W];
            memcpy(tmp, out, sizeof(tmp));
            memset(out, 0, sizeof(float) * CNN_IN_H * CNN_IN_W);
            for (int y = 0; y < CNN_IN_H; ++y)
                for (int x = 0; x < CNN_IN_W; ++x) {
                    int nx = x - delta;
                    if (nx >= 0 && nx < CNN_IN_W)
                        out[y][x] = tmp[y][nx];
                }
        }
    }
    /* Y-axis centroid alignment (clamped +-2 px). */
    double sy = 0.0, syw = 0.0;
    for (int y = 0; y < CNN_IN_H; ++y)
        for (int x = 0; x < CNN_IN_W; ++x) {
            float v = out[y][x];
            if (v > 0.35f) { sy += (double)y * (double)v; syw += v; }
        }
    if (syw > 1e-3) {
        double cy = sy / syw;
        double target = (CNN_IN_H - 1) * 0.5;
        int delta = (int)llround(cy - target);
        if (delta < -2) delta = -2;
        if (delta > 2) delta = 2;
        if (delta != 0) {
            float tmp[CNN_IN_H][CNN_IN_W];
            memcpy(tmp, out, sizeof(tmp));
            memset(out, 0, sizeof(float) * CNN_IN_H * CNN_IN_W);
            for (int y = 0; y < CNN_IN_H; ++y)
                for (int x = 0; x < CNN_IN_W; ++x) {
                    int ny = y - delta;
                    if (ny >= 0 && ny < CNN_IN_H)
                        out[y][x] = tmp[ny][x];
                }
        }
    }
}

/* ---- shared ink-run finder over a line ----
 * Returns the number of ink runs (<=96) and fills centers/widths.
 * Used by both the run-based fallback segmenter and the pitch-grid
 * segmenter (which only consumes centers + median width). */
typedef struct {
    int s[96], e[96];
    int center[96];
    int width[96];
    int n;
    int medw;   /* median run width */
} runs_t;

static int find_runs(const uint8_t *bin, int W, int H, runs_t *r) {
    int *col_dark = (int *)calloc((size_t)W, sizeof(int));
    if (!col_dark) return -1;
    for (int x = 0; x < W; ++x) {
        int c = 0;
        for (int y = 0; y < H; ++y) if (bin[y * W + x]) c++;
        col_dark[x] = c;
    }
    int thr = H / 12; if (thr < 1) thr = 1;
    int in_run = 0, rs = 0;
    r->n = 0;
    for (int x = 0; x <= W; ++x) {
        int ink = (x < W && col_dark[x] >= thr);
        if (ink && !in_run) { in_run = 1; rs = x; }
        else if (!ink && in_run) {
            if (r->n < 96) {
                r->s[r->n] = rs; r->e[r->n] = x;
                r->center[r->n] = (rs + x) / 2;
                r->width[r->n] = x - rs;
                r->n++;
            }
            in_run = 0;
        }
    }
    free(col_dark);
    if (r->n == 0) { r->medw = 8; return 0; }
    int w[96]; int nw = 0;
    for (int i = 0; i < r->n; ++i) w[nw++] = r->width[i];
    for (int i = 1; i < nw; ++i) {
        int k = w[i], j = i - 1;
        while (j >= 0 && w[j] > k) { w[j+1] = w[j]; j--; }
        w[j+1] = k;
    }
    r->medw = w[nw / 2];
    if (r->medw < 6) r->medw = 6;
    return r->n;
}

/* Debug helper: dump the 16x12 glyphs a given image produces.
 * Enabled by MRZ_OCR_DUMP=<tag>. Writes ppm to /tmp/mrz_dump_<tag>_<l><c>.ppm */
static void mrz_ocr_dump_glyphs(const char *tag, int line_idx,
                                const float (*glyphs)[CNN_IN_H][CNN_IN_W],
                                int nchars, const mrz_ocr_rect_t *chars) {
    const char *env = getenv("MRZ_OCR_DUMP");
    if (!env || env[0] == 0 || strcmp(env, "0") == 0) return;
    /* The env value is the dump TAG: it lets several producers write
     * their own glyph set without clobbering each other's files.  The
     * old call sites passed a hardcoded "img", which made the tag
     * unreachable. */
    const char *use_tag = (env[0] && strcmp(env, "0") != 0) ? env : tag;
    const char *dir_env = getenv("MRZ_OCR_DUMP_DIR");
    const char *dir = (dir_env && dir_env[0]) ? dir_env : "/tmp";
    char name[256];
    for (int c = 0; c < nchars; ++c) {
        snprintf(name, sizeof(name), "%s/mrz_dump_%s_l%d_c%02d_x%d_r%d_w%d.ppm",
                 dir, use_tag, line_idx, c, chars[c].x, chars[c].y, chars[c].w);
        FILE *f = fopen(name, "wb");
        if (!f) continue;
        fprintf(f, "P6\n16 12\n255\n");
        for (int y = 0; y < CNN_IN_H; ++y)
            for (int x = 0; x < CNN_IN_W; ++x) {
                uint8_t v = (uint8_t)(glyphs[c][y][x] * 255.0f);
                fputc(v, f); fputc(v, f); fputc(v, f);
            }
        fclose(f);
    }
}

/* CRNN sequence-feature dump. Writes the exact post-conv CNN_FLAT feature
 * vectors the row feeds to the classifier, as raw float32 binary, so the
 * Python trainer trains on the identical numeric distribution the C device
 * runtime consumes. Enabled by MRZ_OCR_FEAT_DUMP=<tag>; the caller writes
 * one file per line: /tmp/mrz_feat_<tag>_l<line>.bin */
static void mrz_ocr_dump_line_features(const char *tag, int line_idx,
                                       const float *feat, int nchars) {
    const char *env = getenv("MRZ_OCR_FEAT_DUMP");
    if (!env || env[0] == 0 || strcmp(env, "0") == 0) return;
    const char *use_tag = (env[0] && strcmp(env, "0") != 0) ? env : tag;
    const char *dir_env = getenv("MRZ_OCR_DUMP_DIR");
    const char *dir = (dir_env && dir_env[0]) ? dir_env : "/tmp";
    char name[256];
    snprintf(name, sizeof(name), "%s/mrz_feat_%s_l%d.bin", dir, use_tag, line_idx);
    FILE *f = fopen(name, "wb");
    if (!f) return;
    int n = nchars < 44 ? nchars : 44;
    fwrite(&n, sizeof(int), 1, f);
    fwrite(feat, sizeof(float), (size_t)n * CNN_FLAT, f);
    fclose(f);
}

/* ---- run-based fallback segmenter (original behaviour) ----
 * One window per ink run, width medw+8 centred on the run. Kept as
 * the fallback when the pitch grid cannot be fitted. */
static int segment_line_cnn(const uint8_t *bin, int W, int H,
                            mrz_ocr_rect_t *chars, int max_chars) {
    runs_t runs;
    if (find_runs(bin, W, H, &runs) < 0) return -1;
    if (runs.n < 1) return 0;
    int medw = runs.medw;
    int cell = medw + 8;                 /* ~ full pitch */
    if (cell < medw + 4) cell = medw + 4;
    int n = 0;
    for (int i = 0; i < runs.n; ++i) {
        if (n >= max_chars) break;
        int c = runs.center[i];
        int half = cell / 2;
        int x0 = c - half;
        int ww = cell;
        if (x0 < 0) { ww += x0; x0 = 0; }
        if (x0 + ww > W) ww = W - x0;
        if (ww < 4) ww = 4;
        chars[n].x = x0; chars[n].w = ww;
        chars[n].y = 0; chars[n].h = H;
        n++;
    }
    return n;
}

/* Latest pitch-grid alignment score (0..100) written by segment_line_grid.
 * 100 = every ink run sat at the centre of its 44-cell grid slot. Low values
 * mean the monospaced-cell assumption is violated, which is the dominant
 * failure mode for the sequence head on real photos. */
static int g_last_seg_score = -1;
int mrz_ocr_last_seg_score(void) { return g_last_seg_score; }

/* TD3 two-line grid sharing.
 *
 * A TD3 MRZ is 2 x 44 monospaced cells rendered from the SAME font at
 * the SAME scale, so both rows necessarily share one pitch L and (up to
 * the row's own x origin) one phase c0.  segment_line_grid() used to fit
 * L and c0 independently on each row from that row's ink runs.  On real
 * photos the second row has a different mix of glyphs (digits plus the
 * personal-number letters), so its run endpoints and median run spacing
 * differ: measured on Photo/003 the two fits were L=44.50/c0=292.5 vs
 * L=46.25/c0=255.75, a 75 px drift by cell 43; on Photo/008 it was
 * L=45.0 vs L=52.0, a 300 px drift that destroyed line 2 completely
 * while line 1 read 43/44.  The first row's fit is the reliable one
 * because it is mostly a long run of uniform '<' fillers.
 *
 * Both call sites (sequential and MRZ_OCR_PAR=1) serialise the two
 * segment_line_grid() calls before starting any worker thread, so a
 * file-static hand-off from line 0 to line 1 is race-free.  The values
 * are reset on every line-0 call, so a failed line-0 fit can never leak
 * a stale grid from the previous image into line 1.
 *
 * MEASURED RESULT (2026-11-05, 50-photo corpus): this hypothesis is WRONG
 * for photographs.  The two rows really do have different pitches -- a
 * page photographed at an angle is trapezoidally distorted, so the lower
 * row is wider.  Forcing line 1's grid onto line 2 cut Photo/003 line-2
 * character accuracy from 37/44 to 16/44.  The independent per-row fit is
 * therefore kept as the default; the switch remains OFF for diagnosis.
 *
 * MRZ_OCR_SHARED_GRID=1 enables full sharing, "pitch" shares only the
 * pitch and keeps the row's own phase; 0 / unset keeps the shipped
 * independent fit. */
static double g_grid_shared_c0 = -1.0;
static double g_grid_shared_L  = 0.0;
static int    g_grid_shared_W  = 0;
static int    g_grid_shared_ok = 0;
static int    g_grid_shared_mode = -1;   /* 0 off, 1 pitch+phase, 2 pitch only */

/* ---- pitch-grid segmenter (primary) ----
 * TD3 lines are exactly 44 monospaced cells. Instead of trusting each
 * ink run to be one character (fragile: one noise run shifts every
 * later window; one merged run loses characters), we fit the global
 * grid parameters:
 *
 *   pitch L  — median of consecutive run-center differences (plausible
 *              range only), autocorrelation fallback;
 *   phase c0 — scan around the first observed glyph, scored by weighted
 *              ink at the window edges (edges must sit in the
 *              inter-glyph gaps).
 *
 * Window POSITIONS come from the fitted grid; window WIDTH stays
 * medw+8 (the distribution the CNN was trained on). When no good grid
 * exists the caller falls back to the run-based segmenter. */
static int segment_line_grid(const uint8_t *bin, int W, int H,
                             mrz_ocr_rect_t *chars, int n_cells,
                             int line_idx) {
    mrz_diag_line_t *dg = mrz_diag_line(line_idx);
    if (dg) {
        dg->valid = 1;
        dg->grid_ok = 0;
        dg->nchars = 0;
        dg->seg_score = -1;
    }
    if (g_grid_shared_mode < 0) {
        const char *e = getenv("MRZ_OCR_SHARED_GRID");
        g_grid_shared_mode = 0;                       /* default: independent */
        if (e && e[0]) {
            if (strcmp(e, "pitch") == 0) g_grid_shared_mode = 2;
            else if (!(e[0] == '0' && e[1] == 0)) g_grid_shared_mode = 1;
        }
    }
    /* Every image starts with a clean hand-off from its own line 0. */
    if (line_idx == 0) g_grid_shared_ok = 0;
    runs_t runs;
    if (find_runs(bin, W, H, &runs) < 0) return -1;
    int nrun = runs.n;
    if (nrun < 2) return -1;              /* not enough structure to fit */
    int medw = runs.medw;
    int cell = medw + 8;
    if (cell < medw + 4) cell = medw + 4;
    if (n_cells < 30) return -1;        /* grid mode needs at least 30 slots */
    if (dg) { dg->nrun = nrun; dg->medw = medw; dg->cell = cell; }

    /* column projection, computed once and shared by pitch search and
     * phase scoring */
    int *col = (int *)calloc((size_t)W, sizeof(int));
    if (!col) return -1;
    for (int x = 0; x < W; ++x)
        for (int y = 0; y < H; ++y) col[x] += bin[y * W + x] ? 1 : 0;

    /* --- pitch estimate --- */
    int Lmin = W / (n_cells + 8); if (Lmin < 10) Lmin = 10;
    int Lmax = W / (n_cells - 12); if (Lmax < Lmin + 2) Lmax = Lmin + 2;
    if (dg) { dg->Lmin = Lmin; dg->Lmax = Lmax; }
    int diffs[95]; int nd = 0;
    for (int i = 1; i < nrun; ++i) {
        int d = runs.center[i] - runs.center[i-1];
        if (d >= Lmin && d <= Lmax) diffs[nd++] = d;
    }
    int L0 = 0;
    if (nd > 0) {
        for (int i = 1; i < nd; ++i) {
            int k = diffs[i], j = i - 1;
            while (j >= 0 && diffs[j] > k) { diffs[j+1] = diffs[j]; j--; }
            diffs[j+1] = k;
        }
        L0 = diffs[nd / 2];
    }
    if (L0 < Lmin || L0 > Lmax) {
        /* Autocorrelation of the column projection. */
        double best = -1; int bestL = (Lmin + Lmax) / 2;
        for (int L = Lmin; L <= Lmax; ++L) {
            double acc = 0;
            for (int x = 0; x + L < W; ++x) acc += (double)col[x] * col[x + L];
            if (acc > best) { best = acc; bestL = L; }
        }
        L0 = bestL;
    }
    /*
     * The old phase search only considered c0 in [0, L). That is not a
     * phase search for a cropped band: it silently assumes that the first
     * MRZ cell starts near x=0. Real photos commonly have 20-40px of
     * horizontal margin, and the previous code then shifted every glyph
     * into the wrong cell.
     *
     * Estimate the endpoints from the run list instead. A narrow run at
     * the extreme page edge, followed by a large gap, is usually the
     * black border/reflection and is excluded. The endpoint span also
     * captures the fractional pitch caused by perspective.
     */
    int left_run = 0;
    int right_run = nrun - 1;
    while (left_run + 1 < right_run &&
           runs.center[left_run] < W / 20 &&
           runs.center[left_run + 1] - runs.center[left_run] > Lmin * 2)
        left_run++;
    while (right_run - 1 > left_run &&
           runs.center[right_run] > (W * 19) / 20 &&
           runs.center[right_run] - runs.center[right_run - 1] > Lmin * 3 / 2)
        right_run--;

    double first_center = (double)runs.center[left_run];
    double last_center = (double)runs.center[right_run];
    double Lfit = (last_center - first_center) / (n_cells - 1.0);
    if (dg) { dg->L0 = L0; dg->Lfit = Lfit; }


    /* MRZ_OCR_GRID_L0=0 restores the legacy pitch search; the default is ON.
     *
     * Lfit is derived from the two EXTREME runs only, so a single dark
     * border / reflection run at the right edge inflates it by ~10%
     * (measured 47 -> 52.07 on a photo whose real pitch is ~47).  The
     * legacy search window (Lfit +- 1) then never contains the correct
     * pitch, and window centre 43 drifts four whole cells into the
     * margin.  L0 is a median over all run gaps and is immune to one
     * outlier.
     *
     * Gated on 2026-11-05: 114-photo corpus, TD3 exact 38/108 unchanged,
     * character accuracy 86.57% -> 87.51%, no photo regressed. */
    static int grid_l0 = -1;
    if (grid_l0 < 0) {
        const char *e = getenv("MRZ_OCR_GRID_L0");
        grid_l0 = (e && e[0] && strcmp(e, "0") == 0) ? 0 : 1;
    }
    /* Only engage when the endpoint fit is pulled HIGH by a border run.
     * A small Lfit/L0 gap means both estimates agree and the endpoints
     * are trustworthy; offering L0 there merely lets a lower edge-ink
     * cost win with a wrong phase (measured on 2026-9-30-H line 1:
     * L0=29 Lfit=30.05 -> c0 17.5 became 0.0 and the name field
     * garbled).  The polluted cases are 1.08x and 1.11x. */
    int lfit_suspect = (L0 >= Lmin && L0 <= Lmax && L0 > 0 &&
                        Lfit > (double)L0 * 1.05);
    int use_l0 = grid_l0 && lfit_suspect;
    double Lcand[32]; int nL = 0;
    /* When Lfit falls outside [Lmin,Lmax] the text doesn't fill the band
     * (e.g. CS/CD with border/shadow at the edges).  Use Lfit-centred
     * candidates instead of L0, which was derived from autocorrelation
     * of the full band including borders and is wrong. */
    double lo = (Lfit >= Lmin && Lfit <= Lmax) ? (double)Lmin : Lfit * 0.7;
    double hi = (Lfit >= Lmin && Lfit <= Lmax) ? (double)Lmax : Lfit * 1.3;
    if (!use_l0) {
        double lstart = floor((Lfit - 1.0) * 4.0) / 4.0;
        double lend = ceil((Lfit + 1.0) * 4.0) / 4.0;
        for (double L = lstart; L <= lend + 0.001 && nL < 32; L += 0.25)
            if (L >= lo && L <= hi) Lcand[nL++] = L;
    } else {
        /* Two compact neighbourhoods rather than a single wide sweep:
         * fewer pitch candidates keeps the phase scan inside the budget. */
        for (int pass = 0; pass < 2 && nL < 32; ++pass) {
            double centre = pass == 0 ? (double)L0 : Lfit;
            if (pass == 0 && (L0 < Lmin || L0 > Lmax)) continue;
            double lstart = floor((centre - 1.0) * 4.0) / 4.0;
            double lend = ceil((centre + 1.0) * 4.0) / 4.0;
            for (double L = lstart; L <= lend + 0.001 && nL < 32; L += 0.25) {
                if (L < lo || L > hi) continue;
                int dup = 0;
                for (int q = 0; q < nL; ++q)
                    if (fabs(Lcand[q] - L) < 1e-9) { dup = 1; break; }
                if (!dup) Lcand[nL++] = L;
            }
        }
    }
    if (nL == 0) {
        Lcand[nL++] = (Lfit >= Lmin && Lfit <= Lmax) ? (double)L0 : Lfit;
    }

    /* --- phase search ---
     * The TRUE grid places every window edge inside an inter-glyph
     * gap. Cost = weighted ink near the 88 window edges (both edges,
     * +-2 px, distance-weighted); the true phase minimises it and the
     * weights break the zero-plateau towards the gap centre. An
     * earlier "ink in the middle half" metric was actively WRONG: a
     * 3-4 px shifted grid pulls strokes into the middle half and
     * scores HIGHER than the aligned one. */
    static const int wt[5] = {1, 2, 3, 2, 1};
    /* Sub-pixel phase scan: the true grid can sit on a fractional
     * offset, and a 0.25px slip at the first cell accumulates into a
     * full-cell mis-slice at the last one. Scanning c0 at quarter-px
     * steps keeps the whole 44-cell grid aligned; integer pitch keeps
     * synthetic corpora (integral pitch) bit-identical. */
    const double PSTEP = 0.25;
    /* Fractional-pitch guard.  A fractional pitch candidate is only
     * accepted when it beats the best INTEGRAL candidate by a clear
     * margin.  On images whose true pitch is integral (synthetic
     * corpora; most >=300dpi scans) a fractional L can win by a hair
     * purely by fitting noise, and a 0.25px pitch error accumulates
     * into a ~10px (1/3 cell) mis-slice at cell 43: measured 14/238
     * fits on the realistic eval subset, -6.9pp full-match.  A genuine
     * fractional pitch (real-photo crops render at e.g. 12.43 px) wins
     * by a wide margin because every integral alternative
     * progressively misaligns the far end.  MRZ_OCR_FRAC_GUARD
     * overrides the margin ratio (0 disables the guard). */
    double guard = 0.90;
    {
        static double g = -2.0;
        if (g < -1.0) {
            const char *e = getenv("MRZ_OCR_FRAC_GUARD");
            g = 0.90;
            if (e && e[0]) {
                double v = atof(e);
                if (v >= 0.0 && v <= 1.0) g = v;
            }
        }
        guard = g;
    }
    double best_c0 = -1; double best_L = 0; double best_cost = 0;
    /* Per-track winners: integral pitch vs fractional pitch. */
    double int_c0 = -1, int_L = 0, int_cost = 0;
    double fr_c0 = -1, fr_L = 0, fr_cost = 0;
    for (int li = 0; li < nL; ++li) {
        double L = Lcand[li];
        if (L <= 0) continue;
        int is_frac = fabs(L - round(L)) > 1e-9;
        double c0_min = first_center - (double)cell / 2.0;
        double c0_max = first_center + (double)cell / 2.0;
        int nsteps = (int)((c0_max - c0_min) / PSTEP) + 1;
        for (int si = 0; si < nsteps; ++si) {
            double c0d = c0_min + si * PSTEP;
            double last = c0d + (double)(n_cells - 1) * L;
            if (use_l0) {
                /* last_center can BE the polluted edge run, so using it as
                 * a hard window would reject every L0 candidate.  Keep only
                 * a bound that the grid stays inside the band; the edge-ink
                 * cost then decides which pitch really aligns 44 windows. */
                if (last > W - (double)cell / 2.0) continue;
            } else if (last < last_center - cell ||
                       last > last_center + cell) {
                continue;
            }
            int cost = 0;
            for (int k = 0; k < n_cells; ++k) {
                double cx = c0d + (double)k * L;
                int edges[2] = {
                    (int)(cx - (double)(cell / 2) + 0.5),
                    (int)(cx + (double)(cell / 2) - 1.0 + 0.5) };
                for (int e = 0; e < 2; ++e)
                    for (int dx = -2; dx <= 2; ++dx) {
                        int x = edges[e] + dx;
                        if (x < 0 || x >= W) continue;
                        cost += wt[dx + 2] * col[x];
                    }
            }
            if (is_frac) {
                if (fr_c0 < 0 || cost < fr_cost) { fr_cost = cost; fr_c0 = c0d; fr_L = L; }
            } else {
                if (int_c0 < 0 || cost < int_cost) { int_cost = cost; int_c0 = c0d; int_L = L; }
            }
        }
    }
    /* Global winner across the two tracks... */
    if (int_c0 < 0)              { best_c0 = fr_c0;  best_L = fr_L;  best_cost = fr_cost; }
    else if (fr_c0 < 0)          { best_c0 = int_c0; best_L = int_L; best_cost = int_cost; }
    else if (fr_cost < int_cost) { best_c0 = fr_c0;  best_L = fr_L;  best_cost = fr_cost; }
    else                         { best_c0 = int_c0; best_L = int_L; best_cost = int_cost; }
    /* ...then the guard: a fractional win that is not decisively
     * cheaper than the integral track is noise fitting; fall back to
     * the integral grid.  When no integral candidate exists (a crop
     * whose fitted pitch admits none) the fractional winner stands. */
    if (fr_c0 >= 0 && int_c0 >= 0 && fr_cost < int_cost &&
        (double)fr_cost > (double)int_cost * guard) {
        best_c0 = int_c0; best_L = int_L; best_cost = int_cost;
    }
    if (dg) {
        dg->pitch = best_L;
        dg->phase = best_c0;
        dg->cost = best_cost;
        dg->int_pitch = int_L;
        dg->int_cost = int_c0 >= 0 ? int_cost : -1;
        dg->frac_pitch = fr_L;
        dg->frac_cost = fr_c0 >= 0 ? fr_cost : -1;
        dg->guard = guard;
    }
    free(col);
    if (best_c0 < 0) return -1;

    /* Hand-off from line 0 to line 1.  Only apply when both rows were
     * measured in the same width coordinate space; otherwise the phase
     * is not comparable. */
    if (line_idx == 0) {
        g_grid_shared_c0 = best_c0;
        g_grid_shared_L  = best_L;
        g_grid_shared_W  = W;
        g_grid_shared_ok = 1;
    } else if (g_grid_shared_mode && g_grid_shared_ok &&
               W == g_grid_shared_W) {
        double fitted_L = best_L, fitted_c0 = best_c0;
        best_L = g_grid_shared_L;
        if (g_grid_shared_mode == 1) best_c0 = g_grid_shared_c0;
        if (getenv("MRZ_OCR_GRID"))
            fprintf(stderr, "GRID SHARED li=1 fitted L=%.3f c0=%.3f "
                            "-> L=%.3f c0=%.3f mode=%d\n",
                    fitted_L, fitted_c0, best_L, best_c0, g_grid_shared_mode);
        if (dg) { dg->pitch = best_L; dg->phase = best_c0; }
    }

    {   /* MRZ_OCR_GRID=1: report the fitted grid for offline diagnosis. */
        static int griddbg = -1;
        if (griddbg < 0) { const char *e = getenv("MRZ_OCR_GRID"); griddbg = (e && e[0] && strcmp(e,"0") != 0) ? 1 : 0; }
        if (griddbg)
            fprintf(stderr, "GRID W=%d H=%d nrun=%d medw=%d cell=%d Lmin=%d Lmax=%d "
                            "L0=%d Lfit=%.3f lr=%d rr=%d fc=%.1f lc=%.1f "
                            "-> c0=%.3f L=%.3f span43=%.1f "
                            "intL=%.2f/%.0f frL=%.2f/%.0f guard=%.2f nL=%d\n",
                    W, H, nrun, medw, cell, Lmin, Lmax, L0, Lfit,
                    left_run, right_run, first_center, last_center,
                    best_c0, best_L, 43.0 * best_L,
                    int_L, int_c0 >= 0 ? int_cost : -1,
                    fr_L, fr_c0 >= 0 ? fr_cost : -1, guard, nL);
    }

    int n = 0;
    /* Window width: `cell` (medw + 8) by default, or a factor of the fitted
     * pitch.  See mrz_cell_win_factor() -- at pitch 30 the legacy window is
     * 0.94 of a cell (correct) and at pitch 19 it is 1.09 (too wide, which
     * pushes ink/window below the training band). */
    int cell_use = cell;
    {
        double f = mrz_cell_win_factor();
        if (f > 0.0 && best_L >= 6.0) cell_use = (int)(f * best_L + 0.5);
        else if (mrz_cell_mode() >= 1 && best_L >= 6.0) cell_use = (int)(best_L + 0.5);
    }
    if (cell_use < 6) cell_use = 6;
    if (dg) dg->win_w = cell_use;
    for (int k = 0; k < n_cells; ++k) {
        double cx = best_c0 + (double)k * best_L;
        int x0 = (int)(cx - (double)(cell_use / 2) + 0.5);
        int ww = cell_use;
        if (x0 < 0) { ww += x0; x0 = 0; }
        if (x0 + ww > W) ww = W - x0;
        if (ww < 4) ww = 4;
        chars[n].x = x0; chars[n].w = ww;
        chars[n].y = 0; chars[n].h = H;
        n++;
    }
    /* Segmentation quality: how well the fitted grid cells line up with the
     * observed ink runs. The sequence head assumes each of the 44 cells
     * holds exactly one character, so a misaligned grid is the single best
     * predictor of CRNN failure on real photos. Score 0..100 where 100 means
     * every run centre sits near its cell centre.
     *
     * Exposed through MRZ_OCR_SEGDBG=1 and, on Android, surfaced through the
     * recogniser log so a low score can gate a fallback engine. */
    {
        double sum = 0.0; int cnt = 0;
        for (int i = 0; i < nrun; ++i) {
            int c = runs.center[i];
            double pos = ((double)c - best_c0) / best_L;
            if (pos < -0.6 || pos > (double)(n_cells - 1) + 0.6) continue;   /* outside the row */
            int k = (int)floor(pos + 0.5);
            if (k < 0) k = 0;
            if (k > n_cells - 1) k = n_cells - 1;
            double d = fabs(pos - (double)k);
            sum += d;
            cnt++;
        }
        int score = (cnt >= 30)
                  ? (int)((1.0 - (sum / cnt) / 0.5) * 100.0)
                  : 0;
        if (score < 0) score = 0;
        if (score > 100) score = 100;
        static int segdbg = -1;
        if (segdbg < 0) { const char *e = getenv("MRZ_OCR_SEGDBG"); segdbg = (e && e[0] && strcmp(e,"0") != 0) ? 1 : 0; }
        if (segdbg)
            fprintf(stderr, "SEG W=%d H=%d runs=%d scored=%d mean_off=%.3f pitch=%.3f score=%d\n",
                    W, H, nrun, cnt, cnt ? sum / cnt : 0.0, best_L, score);
        if (cnt >= 30) g_last_seg_score = score; else g_last_seg_score = 0;
        if (dg) dg->seg_score = g_last_seg_score;
    }
    if (dg) { dg->grid_ok = 1; dg->nchars = n; }
    return n;
}

/* ---- checksum-beam candidate scoring ----
 *
 * A candidate line is scored by the total log probability of its 44
 * characters.  This is the whole point of the beam: two candidates can
 * both satisfy all five ICAO check digits, and only their likelihoods
 * tell them apart.  The previous search accepted the first
 * checksum-valid row it reached while walking the least-confident
 * columns outward, which is a traversal order, not a quality ranking --
 * it could lock in a 0.02-probability glyph while a 0.14-probability
 * glyph elsewhere would have satisfied the same digit.
 *
 * Kept as a standalone function so the ranking rule is directly
 * testable (see tests/test_postproc.c) instead of only observable
 * through a full image run. */
static double mrz_candidate_logprob(const char *line, const top3_t *t3) {
    double s = 0.0;
    for (int c = 0; c < 44; ++c) {
        int g = mrz_ocr_glyph_index(line[c]);
        float p = 0.0f;
        for (int k = 0; k < TOPK; ++k)
            if (t3[c].cand[k] == g) { p = t3[c].p[k]; break; }
        if (p <= 1e-6f) p = 1e-6f;
        s += log((double)p);
    }
    return s;
}

/* ---- deep checksum beam --------------------------------------------
 *
 * Search the top-3 candidates of the least confident columns for the
 * highest-probability string that satisfies all five ICAO check
 * digits.
 *
 * Two properties make this affordable inside the frame budget:
 *
 *  * the column set is bounded (MRZ_BEAM_COLS) and the depth is
 *    bounded (MRZ_BEAM_DEPTH), so the number of leaves is
 *    C(cols, depth) * 2^depth -- a few thousand at most;
 *  * the probability bound prunes: a partial assignment cannot beat
 *    the best complete candidate found so far, so branches that lost
 *    too much probability are abandoned before they are expanded.
 *
 * The bound is computed on the log-probability of the columns that
 * are still FREE, using the best alternative each of them could
 * still contribute.  That is an admissible (never optimistic) bound,
 * so pruning cannot discard the optimum.
 */
#define MRZ_BEAM_COLS  14
#define MRZ_BEAM_DEPTH 4

/* Runtime A/B knobs for the checksum beam.  The shipped constants are the
 * defaults; MRZ_BEAM_DEPTH / MRZ_BEAM_COLS override them so the search
 * space can be measured on a corpus without rebuilding.  A reading whose
 * GT needs five or six simultaneous corrections (e.g. a passport-number
 * confusion that also drags the following optional-field letters) is
 * mathematically unreachable at depth 4 no matter how good the top-5
 * candidates are. */
static int mrz_beam_depth(void) {
    static int d = -1;
    if (d < 0) {
        const char *e = getenv("MRZ_BEAM_DEPTH");
        d = MRZ_BEAM_DEPTH;
        if (e && e[0]) { int v = atoi(e); if (v >= 1 && v <= 10) d = v; }
    }
    return d;
}

static int mrz_beam_cols(void) {
    static int c = -1;
    if (c < 0) {
        const char *e = getenv("MRZ_BEAM_COLS");
        c = MRZ_BEAM_COLS;
        if (e && e[0]) { int v = atoi(e); if (v >= 4 && v <= 44) c = v; }
    }
    return c;
}

/* Log-probability gap below which two checksum-valid readings are
 * considered indistinguishable.
 *
 * Measured on the 41-photo corpus: the readings that flip between the
 * host (libjpeg) and the device (Skia) differ by less than 0.3 in
 * summed log-probability, while the readings that are stable differ by
 * more than 1.  0.5 sits between the two groups: it flags the readings
 * that a one-level pixel change can flip, and leaves the stable ones
 * alone. */
#define MRZ_AMBIGUITY_MARGIN 0.5

/* How many characters must differ between the best and the runner-up
 * reading before the result is called ambiguous.  One is enough: the
 * measurements showed that a single-character disagreement on a
 * protected field can flip under a one-level pixel change (the device
 * read 'L' where the host read 'V' in the document-type column). */
#define MRZ_AMBIGUITY_MIN_DIFF 1

/* Skip the relaxed-bound runner-up beam walk when every candidate column's
 * top-2 probability margin is at least this.  With margin m, any
 * single-column flip costs >= log((1+m)/(1-m)) log-prob (the two-class
 * worst case), i.e. 0.62 at m=0.3 -- already above MRZ_AMBIGUITY_MARGIN
 * (0.5), so a reportable runner-up is unreachable and the second walk
 * can only burn time.  Outputs are bit-identical with and without the
 * skip; this is a pure tail-latency cut. */
#define MRZ_RUNNERUP_SKIP_MARGIN 0.3

/* Does the difference between two readings fall on a field that
 * matters to the caller?
 *
 * The five check digits cover the document number (cols 0-8), the
 * birth date (13-18), the expiry date (21-26) and the personal number
 * (28-41), plus the composite digit (43) which covers everything
 * except the check digits themselves.  A difference that sits ONLY on
 * a check-digit column, on the nationality (10-12) or on the sex (20)
 * is not a risk: those columns are either determined by the checksum
 * or are not protected by it, so both readings carry the same data.
 *
 * A difference on any other column means the two readings disagree
 * about a field the checksum validates and could not tell apart --
 * that is the case worth reporting.
 */
static int mrz_alt_diff_count(const char *a, const char *b) {
    int n = 0;
    for (int c = 0; c < 44; ++c)
        if (a[c] != b[c]) n++;
    return n;
}

static int mrz_alt_is_material(const char *a, const char *b) {
    for (int c = 0; c < 44; ++c) {
        if (a[c] == b[c]) continue;
        if (c == 9 || c == 19 || c == 27 || c == 42 || c == 43) continue; /* check digits */
        if (c >= 10 && c <= 12) continue;                                  /* nationality */
        if (c == 20) continue;                                             /* sex */
        return 1;
    }
    return 0;
}

typedef struct {
    const char *base;      /* the greedy decode, 44 chars            */
    const top3_t *t3;      /* per-column candidates                   */
    const int *cols;       /* columns the search may change           */
    int ncols;
    int depth;
    char *best;
    double best_score;
    double *suffix;        /* optimistic bound for columns >= i       */
    /* Runner-up tracking.  A checksum-valid reading that scores close
     * to the winner means the line is ambiguous: neither the checksum
     * nor the classifier could tell the two apart.  The caller has to
     * know that, because a wrong-but-checksum-valid line looks
     * exactly like a correct one downstream. */
    char *alt;
    double alt_score;
    /* Slack added to the incumbent when the bound is compared against
     * it.  Zero means "prune as aggressively as possible", which is
     * what the WINNER search must use: relaxing the bound there lets
     * the walk reach leaves that the strict bound would have pruned,
     * and the accumulated log-probability of a leaf then depends on
     * the order the leaves are visited in.  That changed the winner on
     * real photos (Photo/lv went from the correct LV90008223LVA... to
     * LV90008223LWA...), so the slack is confined to the separate
     * runner-up pass. */
    double prune_slack;
} mrz_beam_ctx;

/* Log-probability of the greedy (best) class of a column. */
static double mrz_beam_lp(const top3_t *t, int j) {
    double p = t->p[j];
    return log(p > 1e-9 ? p : 1e-9);
}

static double mrz_beam_opt_suffix(const mrz_beam_ctx *ctx, int k) {
    /* Best DELTA still available from column k onwards: each remaining
     * column may still switch to its most likely alternative.  The
     * score is tracked as a delta against the greedy decode, so the
     * bound is the sum of those deltas. */
    double s = 0.0;
    for (int i = k; i < ctx->ncols; ++i) {
        int c = ctx->cols[i];
        const top3_t *t = &ctx->t3[c];
        double best = 0.0;
        for (int j = 1; j < TOPK; ++j) {
            if (t->cand[j] < 0) continue;
            if (j > 1 && t->cand[j] == t->cand[1]) continue;
            if (t->cand[j] == t->cand[0]) continue;
            double d = mrz_beam_lp(t, j) - mrz_beam_lp(t, 0);
            if (d > best) best = d;
        }
        s += best;
    }
    return s;
}

static void mrz_beam_recurse(mrz_beam_ctx *ctx, int k, int used,
                             double score, char *work) {
    if (k >= ctx->ncols) {
        if (!line2_checksum_ok(work)) return;
        if (score > ctx->best_score) {
            /* The old winner becomes the runner-up -- but only if there
             * WAS a winner.  Before the first legal reading is found
             * the incumbent is still the sentinel, and copying it would
             * put a non-reading into the runner-up slot. */
            if (ctx->alt && ctx->best_score > -1e299) {
                memcpy(ctx->alt, ctx->best, 45);
                ctx->alt_score = ctx->best_score;
            }
            ctx->best_score = score;
            memcpy(ctx->best, work, 44);
            ctx->best[44] = 0;
        } else if (ctx->alt && score > ctx->alt_score &&
                   memcmp(work, ctx->best, 44) != 0) {
            /* A reading that did not beat the incumbent, and is not
             * the incumbent itself.  When the incumbent is the
             * already-known winner (the runner-up pass) this records
             * the best reading that is strictly worse than it, which is
             * the runner-up. */
            ctx->alt_score = score;
            memcpy(ctx->alt, work, 44);
            ctx->alt[44] = 0;
        }
        return;
    }

    /* Optimistic bound for the rest of the search.  The greedy decode
     * is not a valid candidate, so the bound is only meaningful once
     * at least one valid leaf has been found. */
    if (ctx->best_score > -1e299) {
        double bound = score + mrz_beam_opt_suffix(ctx, k);
        if (bound <= ctx->best_score - ctx->prune_slack) return;
    }

    int c = ctx->cols[k];
    const top3_t *t = &ctx->t3[c];
    /* The depth limit restricts how many columns may be CHANGED.  It
     * must not stop the walk: once the budget is used up the search
     * still has to carry the remaining columns unchanged to the end of
     * the line, or the leaf is never reached and the candidate is
     * silently dropped.  (Measured: with the budget consumed at the
     * fourth flip the walk aborted early and every reading that
     * needed four corrections was lost.) */
    for (int j = 1; used < ctx->depth && j < TOPK; ++j) {
        int g = t->cand[j];
        if (g < 0) continue;
        if (j > 1 && g == t->cand[1]) continue;   /* duplicate */
        if (g == t->cand[0]) continue;
        char keep = work[c];
        work[c] = MRZ_OCR_GLYPHS[g].ch;
        /* The score is a DELTA against the greedy decode: the greedy
         * class already contributed its own log-probability, so
         * switching the column replaces that contribution. */
        double delta = mrz_beam_lp(t, j) - mrz_beam_lp(t, 0);
        mrz_beam_recurse(ctx, k + 1, used + 1, score + delta, work);
        work[c] = keep;
    }
    /* keep this column unchanged and move on */
    mrz_beam_recurse(ctx, k + 1, used, score, work);
}

static void mrz_beam_search(const char *base, const top3_t *t3,
                            const int *cols, int ncols,
                            char *best, double *best_score,
                            char *alt, double *alt_score,
                            double prune_slack) {
    char work[45];
    memcpy(work, base, 44); work[44] = 0;
    static int bdbg = -1;
    if (bdbg < 0) { const char *e = getenv("MRZ_OCR_BEAMDBG"); bdbg = (e && e[0] && strcmp(e,"0") != 0) ? 1 : 0; }
    mrz_beam_ctx ctx;
    ctx.base = work;
    ctx.t3 = t3;
    ctx.cols = cols;
    ctx.ncols = ncols;
    ctx.depth = mrz_beam_depth();
    ctx.best = best;
    ctx.best_score = *best_score;
    ctx.alt = alt;
    ctx.alt_score = alt ? *alt_score : -1e300;
    ctx.prune_slack = prune_slack;
    double _s0 = mrz_candidate_logprob(work, t3);
    mrz_beam_recurse(&ctx, 0, 0, 0.0, work);
    *best_score = ctx.best_score;
    if (alt) *alt_score = ctx.alt_score;
    if (bdbg)
        fprintf(stderr, "BEAM base=%s (%.3f) -> best=%s (%.3f) cols=%d\n",
                work, _s0, ctx.best, ctx.best_score, ncols);
}

/* ---- whole-line decode: syntax mask + protections + checksum beam ----
 * `probs` holds n rows of softmaxed class probabilities (possibly
 * TTA-averaged by the caller). Glyphs are needed for the ink gate. */
/* Ambiguity of the decoded line, reported to the caller.
 *
 * `margin` is the log-probability gap between the best checksum-valid
 * reading and the runner-up; a small gap means the classifier could
 * not tell them apart and the checksum could not either.  `alt` holds
 * the runner-up so a caller can show both readings. */
typedef struct {
    int  ambiguous;
    int  margin;      /* 0 when there is no runner-up */
    char alt[45];
} mrz_decode_amb_t;

/* Confidence-gated fusion of the per-character CNN head and the CRNN
 * sequence head.
 *
 * The two heads fail on DIFFERENT cells.  Measured on a strictly
 * held-out split (train on pic_scan1, evaluate on pic_scan2/3, the
 * SAME exported features for both heads):
 *
 *     line 1   CNN 94.62%   CRNN 92.94%
 *     line 2   CNN 97.73%   CRNN 82.66%
 *
 * and on the network-scraped assets set:
 *
 *     line 2   CNN 76.78%   CRNN 54.40%
 *
 * On the held-out split the CRNN head was RIGHT on every cell where it
 * disagreed with the CNN (45/45 on line 1, 19/19 on line 2).  On the
 * assets set the opposite held: the CNN was right on 386 of the 643
 * disagreements, and there the CRNN's confidence was clearly lower
 * (0.55 mean when the CNN was right, versus 0.66-0.81 when the CRNN
 * was right).  That is the signal the gate uses: the CRNN voice only
 * counts on the cells where it is confident.
 *
 * gate is the absolute CRNN-confidence threshold (0 = no gate, i.e.
 * plain weighted log-sum).  alpha scales the CRNN log-probability
 * term for the cells that pass the gate.
 *
 * Measured character accuracy with this rule:
 *     gate 0.00   held-out 100.00%   assets 48.33%
 *     gate 0.70   held-out 100.00%   assets 57.88%
 *     gate 0.80   held-out  99.58%   assets 59.27%
 *     gate 0.90   held-out  98.80%   assets 59.80%
 *     (CNN only)  held-out  96.17%   assets 58.38%
 */
static void mrz_fuse_probs(float *cnn_probs, const float *crnn_probs,
                           int nchars, float alpha, float gate, float ratio) {
    for (int c = 0; c < nchars; ++c) {
        const float *pc = cnn_probs + c * CNN_OUT;
        const float *ps = crnn_probs + c * CNN_OUT;
        float *row = cnn_probs + c * CNN_OUT;
        float cmax = 0.0f, smax = 0.0f;
        for (int o = 0; o < CNN_OUT; ++o) {
            if (pc[o] > cmax) cmax = pc[o];
            if (ps[o] > smax) smax = ps[o];
        }
        /* The CRNN voice counts on a cell only when it is BOTH
         * confident in absolute terms AND clearly more confident than
         * the CNN on that same cell.  The relative test is what makes
         * the rule work on both real sets: on the held-out photos the
         * CRNN was right on every cell where the two heads disagreed
         * (45/45 and 19/19) and its confidence was the higher of the
         * two, while on the scraped assets set the CNN was right on
         * 386 of 643 disagreements and there the CRNN's confidence was
         * the lower one. */
        int use_seq = (smax >= gate) && (smax >= cmax * ratio);
        for (int o = 0; o < CNN_OUT; ++o) {
            float a = pc[o] > 1e-9f ? logf(pc[o]) : -20.7f;
            float b = ps[o] > 1e-9f ? logf(ps[o]) : -20.7f;
            row[o] = use_seq ? (a + alpha * b) : a;
        }
        /* Re-normalise so the downstream code sees a probability
         * distribution again. */
        float mx = -INFINITY;
        for (int o = 0; o < CNN_OUT; ++o)
            if (row[o] > mx) mx = row[o];
        float sum = 0.0f;
        for (int o = 0; o < CNN_OUT; ++o) {
            row[o] = expf(row[o] - mx);
            sum += row[o];
        }
        for (int o = 0; o < CNN_OUT; ++o)
            row[o] /= sum > 0.0f ? sum : 1.0f;
    }
}

static void decode_row(const float (*glyphs)[CNN_IN_H][CNN_IN_W],
                       int n, int line_idx, char *dest, int *conf_avg,
                       const float *probs, mrz_decode_amb_t *amb,
                       const float *seq_probs,
                       float arb_gap, float arb_margin, float arb_conf) {
    int total = n < 44 ? n : 44;
    if (amb) { amb->ambiguous = 0; amb->margin = 0; amb->alt[0] = 0; }
    mrz_diag_line_t *dg = mrz_diag_line(line_idx);
    if (dg) {
        dg->nchars = total;
        dg->conf_avg = 0;
        dg->ambiguous = 0;
        dg->alt_margin = 0;
        dg->alt[0] = 0;
        memset(dg->gate, 0, sizeof(dg->gate));
        memset(dg->ch, 0, sizeof(dg->ch));
    }
    top3_t t3[44];
    int allow[37];
    static int dbg = -1;
    if (dbg < 0) { const char *e = getenv("MRZ_OCR_TOPK"); dbg = (e && e[0] && strcmp(e,"0")!=0) ? 1 : 0; }
    for (int c = 0; c < n && c < 44; ++c) {
        fill_allow(allow, line_idx, c);
        top3_of(probs + c * CNN_OUT, allow, &t3[c]);
        if (dg && c < MRZ_DIAG_CHARS) {
            for (int k = 0; k < TOPK; ++k) {
                dg->cand[c][k] = t3[c].cand[k];
                dg->p[c][k] = t3[c].p[k];
            }
        }
        if (dbg) {
            fprintf(stderr, "TOPK L%d c%02d:", line_idx, c);
            for (int k = 0; k < TOPK; ++k)
                fprintf(stderr, " %c(%.3f)", MRZ_OCR_GLYPHS[t3[c].cand[k]].ch, t3[c].p[k]);
            fprintf(stderr, "\n");
        }
    }

    /* patch ink per cell (relative '<' gate statistic) */
    float patch_ink[44];
    for (int c = 0; c < n && c < 44; ++c) {
        float ink = 0.0f;
        for (int y = 0; y < CNN_IN_H; ++y)
            for (int x = 0; x < CNN_IN_W; ++x)
                ink += glyphs[c][y][x];
        patch_ink[c] = ink;
        dest[c] = MRZ_OCR_GLYPHS[t3[c].cand[0]].ch;
        if (dg && c < MRZ_DIAG_CHARS) dg->ink[c] = ink;
    }
    if (n < 44) dest[n] = '\0'; else dest[44] = '\0';

    /* (a) Ink gate: '<' is the lowest-ink OCR-B glyph. If the model
     * says X/R but the patch carries far less ink than the line's
     * confident letters, it cannot be X/R. The threshold is RELATIVE
     * (0.55 x median ink of confident letter cells, p0 >= 0.85) so it
     * follows contrast/resolution instead of a hand-tuned constant;
     * falls back to the historical absolute 60 when the line has too
     * few confident letters. T/I/L/M are NOT gated (ink too close to
     * '<' to be safe). */
    {
        float letter_ink[44]; int nl = 0;
        for (int c = 0; c < total; ++c) {
            int g = t3[c].cand[0];
            if (g >= 10 && g <= 35 && t3[c].p[0] >= 0.85f)
                letter_ink[nl++] = patch_ink[c];
        }
        float gate = -1.0f;
        if (nl >= 5) {
            for (int i = 1; i < nl; ++i) {
                float k = letter_ink[i]; int j = i - 1;
                while (j >= 0 && letter_ink[j] > k) { letter_ink[j+1] = letter_ink[j]; j--; }
                letter_ink[j+1] = k;
            }
            gate = 0.55f * letter_ink[nl / 2];
        }
        for (int c = 0; c < total; ++c) {
            char ch0 = dest[c];
            if (ch0 == 'X' || ch0 == 'R') {
                if (t3[c].p[0] >= 0.95f) continue;
                float thr = (gate > 0.0f) ? gate : 60.0f;
                if (patch_ink[c] < thr) {
                    dest[c] = '<';
                    if (dg) dg->gate[c] |= 1;
                }
            }
        }
    }

    /* (b) 3-neighbour smoothing: an isolated X/R/T sandwiched between
     * '<' on both sides is syntactically impossible in ICAO 9303. */
    for (int c = 1; c + 1 < total; ++c) {
        if (dest[c-1] == '<' && dest[c+1] == '<' &&
            (dest[c] == 'X' || dest[c] == 'R' || dest[c] == 'T')) {
            dest[c] = '<';
            if (dg) dg->gate[c] |= 2;
        }
    }

    /* (c) Trailing "<" filler lock on Line 1: once >=5 consecutive '<'
     * the rest of the name field is guaranteed filler by ICAO 9303.
     * Line 2's tail keeps two numeric check digits, so only Line 1
     * locks. */
    if (line_idx == 0) {
        int locked = -1, run = 0;
        for (int c = 0; c < total; ++c) {
            if (dest[c] == '<') { run++; if (run >= 5) locked = c - run + 1; }
            else run = 0;
        }
        if (locked >= 0)
            for (int c = locked; c < total; ++c) {
                dest[c] = '<';
                if (dg) dg->gate[c] |= 4;
            }
    }

    /* (c2) Line-1 issuing-state repair against the ICAO state registry.
     *
     * Line 1 has NO check digit, so unlike line 2 it has no arithmetic
     * constraint to search against -- which is why it has never had a
     * beam at all, only the three heuristics above.  But its columns
     * 2..4 are not free text: ICAO 9303 requires them to be an issuing
     * state code from the MRTD registry.  That is a hard, verifiable
     * constraint on exactly three characters, and it is the only
     * content constraint line 1 has ever had.
     *
     * On the phone corpus this is what separates a misread state code
     * from a correctly read one: the four images whose line 1 was wrong
     * all carried codes no state issues ("HUG", "CHY", "CNN", "ATA"),
     * while every hand-read ground truth carried a real one.  The same
     * failure shows up on device-only differences: the emulator reads
     * "P<CHNBU" where the handset reads "P<GHNBU", and only "CHN" is a
     * country.
     *
     * The search is exhaustive over the top-3 of the three columns (27
     * combinations, each a single table lookup) and only runs when the
     * greedy triple is NOT a valid code.  So:
     *   - a reading that is already valid is never touched (zero
     *     regression risk on correct output);
     *   - the repair only ever replaces "impossible" with "possible",
     *     and among the possible ones it takes the highest-probability
     *     combination, which is the same maximum-likelihood rule the
     *     line-2 beam uses.
     *
     * MRZ_OCR_NOISSUERFIX=1 disables it for A/B measurement. */
    if (line_idx == 0 && total == 44) {
        static int issuer_fix = -1;
        if (issuer_fix < 0) {
            const char *e = getenv("MRZ_OCR_NOISSUERFIX");
            issuer_fix = !(e && e[0] == '1' && e[1] == 0);
        }
        char cur[3] = { dest[2], dest[3], dest[4] };
        if (issuer_fix && !mrz_code3_plausible(cur)) {
            double best_lp = -1e300;
            char best3[3];
            int found = 0;
            for (int a = 0; a < TOPK; ++a) {
                for (int b = 0; b < TOPK; ++b) {
                    for (int c = 0; c < TOPK; ++c) {
                        char cand[3] = {
                            MRZ_OCR_GLYPHS[t3[2].cand[a]].ch,
                            MRZ_OCR_GLYPHS[t3[3].cand[b]].ch,
                            MRZ_OCR_GLYPHS[t3[4].cand[c]].ch,
                        };
                        if (!mrz_code3_plausible(cand)) continue;
                        double lp = mrz_beam_lp(&t3[2], a) +
                                    mrz_beam_lp(&t3[3], b) +
                                    mrz_beam_lp(&t3[4], c);
                        if (lp > best_lp) {
                            best_lp = lp;
                            best3[0] = cand[0];
                            best3[1] = cand[1];
                            best3[2] = cand[2];
                            found = 1;
                        }
                    }
                }
            }
            if (found) {
                dest[2] = best3[0];
                dest[3] = best3[1];
                dest[4] = best3[2];
                if (dg) { dg->gate[2] |= 8; dg->gate[3] |= 8; dg->gate[4] |= 8; }
            }
        }
    }

    /* (d) Line-2 checksum beam search.
     *
     * Every checksum-valid candidate is scored by the SUM OF LOG
     * PROBABILITIES of the 44 characters, and the HIGHEST-scoring
     * candidate wins.  The previous implementation took the FIRST
     * checksum-valid candidate it stumbled upon while walking the
     * positions from the smallest margin (least confident) upwards.
     * That ordering is a search heuristic, not a quality ranking: it
     * happily accepted a 0.02-probability character in order to reach a
     * valid checksum while a 0.14-probability character elsewhere would
     * have satisfied the same check digit.  Both results are
     * checksum-valid, so the wrong one was indistinguishable from the
     * right one downstream (real photos: AD's birth date read as
     * 950118 instead of 960118).
     *
     * Scoring by log-prob keeps the beam's recall (every column stays
     * in play through the composite check digit) while making the
     * choice deterministic and probability-aware. */
    /* MRZ_OCR_NOBEAM=1 disables the search entirely (diagnostic A/B:
     * measures whether the beam helps or hurts on a given corpus). */
    static int beam_enabled = -1;
    if (beam_enabled < 0) {
        const char *e = getenv("MRZ_OCR_NOBEAM");
        beam_enabled = !(e && e[0] == '1' && e[1] == 0);
    }
    if (line_idx == 1 && total == 44 && beam_enabled) {
        char best[45];
        memcpy(best, dest, 44); best[44] = 0;
        /* The beam runs whether or not the greedy decode already
         * satisfies the checksum.
         *
         * A checksum-valid decode is NOT automatically the right one:
         * a misread 'B' that happens to keep the check digit valid
         * produces a line the checksum happily accepts.  Restricting
         * the search to the failing case therefore left exactly those
         * wrong-but-valid readings untouched.  Running the search
         * unconditionally and letting the probability pick the winner
         * was measured on the 41-photo corpus: it never broke a
         * reading that was already correct, and it changed the two
         * readings that were not. */
        {
            /* The beam works in DELTA space against the greedy decode.
             * The greedy decode itself is NOT a candidate -- it failed
             * the checksum -- so the incumbent starts below any
             * reachable delta. */
            double best_score = -1e300;

            int idxs[44]; float marg[44]; int picks = 0;
            for (int c = 0; c < 44; ++c) {
                if (t3[c].cand[0] != t3[c].cand[1]) {
                    idxs[picks] = c;
                    marg[picks] = t3[c].p[0] - t3[c].p[1];
                    picks++;
                }
            }
            for (int a = 1; a < picks; ++a) {
                int ki = idxs[a]; float km = marg[a];
                int j = a - 1;
                while (j >= 0 && marg[j] > km) { idxs[j+1] = idxs[j]; marg[j+1] = marg[j]; j--; }
                idxs[j+1] = ki; marg[j+1] = km;
            }
            /* Candidate columns, least confident first.  The margin
             * between the best and the second-best class is the best
             * available proxy for "this cell might be wrong": a cell
             * the network is sure about does not need to be searched. */
            if (picks > mrz_beam_cols()) picks = mrz_beam_cols();

            /* Deep beam.
             *
             * The previous search enumerated ONE-position flips and
             * TWO-position flips over the six least confident columns.
             * That is not enough: on the 41-photo corpus the residual
             * failures need three to four simultaneous corrections
             * (measured on the real photos: 3 of the 4 failures become
             * reachable at depth 4, none of them at depth 2), because
             * the confusions cluster -- a passport number whose
             * 'B' was read as '8' also drags the following '0' to 'O'
             * and the check digit with it.
             *
             * The enumeration is a depth-first walk over the candidate
             * columns with a checksum test at every leaf and a
             * probability bound that prunes whole subtrees.  The cost
             * is bounded by the number of candidate columns and the
             * depth, both of which are small constants, so the decode
             * stays inside the time budget. */
            /* First pass: the winner.  The bound is NOT relaxed here,
             * so the set of leaves the walk visits -- and therefore the
             * order the log-probabilities are accumulated in -- is
             * exactly the one the pre-ambiguity build used.  Relaxing
             * it in a single combined pass changed the winner on real
             * photos (Photo/lv went from the correct LV90008223LVA...
             * to LV90008223LWA...), so the relaxation is confined to
             * the runner-up pass below. */
            char pre_beam[45];
            memcpy(pre_beam, dest, 45);
            mrz_beam_search(dest, t3, idxs, picks, best, &best_score,
                            NULL, NULL, 0.0);

            /* Second pass: the runner-up.  The bound is relaxed by the
             * ambiguity margin so that a candidate within the margin of
             * the winner survives the pruning and can be reported.
             * This pass never changes the winner.
             *
             * Skip it entirely when every candidate column is confident
             * (MRZ_RUNNERUP_SKIP_MARGIN): a reportable runner-up is then
             * mathematically unreachable, and this walk is the beam's
             * main worst-case cost.
             *
             * With CRNN arbitration active (seq_probs != NULL), the
             * relaxation window can be widened via arb_gap so a runner-up
             * beyond the strict ambiguity margin exists for the tie-break
             * to consider. */
            static float amb_margin_cfg = -1.0f;
            if (amb_margin_cfg < 0.0f) {
                const char *e = getenv("MRZ_OCR_AMB_MARGIN");
                amb_margin_cfg = (e && e[0]) ? (float)atof(e) : MRZ_AMBIGUITY_MARGIN;
            }
            float relax = amb_margin_cfg;
            if (seq_probs && arb_gap > amb_margin_cfg) relax = arb_gap;
            char alt[45];
            double alt_score = -1e300;
            alt[0] = 0;
            if (best_score > -1e299 && picks > 0 && marg[0] >= MRZ_RUNNERUP_SKIP_MARGIN) {
                /* all candidate columns confident -> no alt possible */
            } else if (best_score > -1e299) {
                /* The runner-up pass starts with the WINNER already in
                 * the incumbent slot, so what it records is the best
                 * reading that is WORSE than the winner -- which is
                 * exactly the runner-up.  Starting from the sentinel
                 * instead would make it record its own best, i.e. the
                 * winner again, and the flag would never fire. */
                char best_copy[45];
                memcpy(best_copy, best, 45);
                double best_score_copy = best_score;
                mrz_beam_search(dest, t3, idxs, picks, best_copy, &best_score_copy,
                                alt, &alt_score, relax);
            }

            /* The search only replaces the decode when it actually
             * found a checksum-valid reading.  When it found none
             * (the greedy decode is kept, which may be invalid) the
             * result is left exactly as it was. */
            if (best_score > -1e299) {
                memcpy(dest, best, 44);
                if (dg)
                    for (int c = 0; c < 44; ++c)
                        if (dest[c] != pre_beam[c]) dg->gate[c] |= 8;
                if (n < 44) dest[n] = '\0'; else dest[44] = '\0';
                /* A runner-up within the margin means the reading is a
                 * near coin flip between two checksum-valid candidates.
                 * Report it: a wrong-but-checksum-valid line is
                 * indistinguishable from a correct one downstream, so
                 * the caller needs to know the result is not
                 * trustworthy. */
                if (amb && alt_score > -1e299) {
                    double gap = best_score - alt_score;
                    int ndiff = mrz_alt_diff_count(best, alt);
                    if (gap < amb_margin_cfg &&
                        ndiff >= MRZ_AMBIGUITY_MIN_DIFF &&
                        mrz_alt_is_material(best, alt)) {
                        amb->ambiguous = 1;
                        amb->margin = (int)(gap * 1000.0);
                        memcpy(amb->alt, alt, 45);
                    }
                }
                /* CRNN arbitration (MRZ_OCR_FUSE_MODE=arb, the default):
                 * the beam above ran on the PURE CNN probabilities, so the
                 * fused engine can no longer overwrite the checksum-beam
                 * repairs (the measured 10-25 failure on E/G/K/Q/T/AJ).
                 * When a runner-up exists within the relaxation window,
                 * the sequence head votes ONLY on the columns where the
                 * two readings differ: if its log-prob mass clearly
                 * favours the alt, take it.  Ties stay with the CNN, so
                 * the fused engine can only differ from the CNN where the
                 * CNN itself saw a near-tie -- the "fuse must not cost"
                 * contract.  On real photos the CRNN was right on 78 of
                 * the 105 disagreement cells vs 27 for the CNN (assets),
                 * so the vote is worth having exactly there. */
                if (seq_probs && alt[0] && alt_score > -1e299 &&
                    (best_score - alt_score) < relax) {
                    double s_best = 0.0, s_alt = 0.0;
                    int arbdiff = 0, low_conf = 0;
                    for (int c = 0; c < total; ++c) {
                        if (best[c] == alt[c]) continue;
                        int ib = mrz_ocr_glyph_index(best[c]);
                        int ia = mrz_ocr_glyph_index(alt[c]);
                        if (ib < 0 || ia < 0) continue;
                        float pb = seq_probs[c * CNN_OUT + ib];
                        float pa = seq_probs[c * CNN_OUT + ia];
                        /* Per-column guard: the sequence head must be
                         * CONFIDENT about the character it favours on
                         * every differing column.  The measured real-photo
                         * profile is 0.66-0.81 mean confidence when the
                         * CRNN is right vs 0.55 when it is wrong; a noisy
                         * or domain-collapsed head must not get a vote.
                         * MRZ_OCR_FUSE_ARB_CONF, 0 disables the guard. */
                        if (arb_conf > 0.0f && pa < arb_conf) { low_conf = 1; break; }
                        s_best += (pb > 1e-9f) ? logf(pb) : -20.7f;
                        s_alt  += (pa > 1e-9f) ? logf(pa) : -20.7f;
                        arbdiff++;
                    }
                    if (arbdiff > 0 && !low_conf && s_alt - s_best >= arb_margin) {
                        memcpy(dest, alt, (size_t)total);
                        dest[total] = '\0';
                        if (dg)
                            for (int c = 0; c < total; ++c)
                                if (dest[c] != pre_beam[c]) dg->gate[c] |= 8;
                    }
                }
            }
        }
    }

    /* (e) Tail check-digit rescue.
     *
     * Positions 42 and 43 are BOTH deterministic functions of the
     * preceding characters:
     *   42 = check digit over the 14-char personal number (28..41)
     *   43 = composite check digit over cols 0..9, 13..19, 21..42
     * When the classifier drops one of them to '<' (tail drift in a long
     * run of filler), neither the greedy decode nor a top-3 beam can
     * recover it: the digit is outside the candidate set and the line
     * stays checksum-invalid.  Recompute it instead of asking a slower
     * external engine to rescue the read.
     *
     * Guard: only fire when the three leading field check digits already
     * close, i.e. the identity fields are trustworthy.  A '<' in a
     * check-digit slot can NEVER be checksum-valid, so this path cannot
     * overwrite a reading the decoder would otherwise have accepted.
     * Set MRZ_OCR_NO_CD_RESCUE=1 for A/B measurement. */
    if (line_idx == 1 && total == 44) {
        static int cd_rescue_enabled = -1;
        if (cd_rescue_enabled < 0) {
            const char *e = getenv("MRZ_OCR_NO_CD_RESCUE");
            cd_rescue_enabled = !(e && e[0] == '1' && e[1] == 0);
        }
        if (cd_rescue_enabled
            && dest[9]  >= '0' && dest[9]  <= '9'
            && dest[19] >= '0' && dest[19] <= '9'
            && dest[27] >= '0' && dest[27] <= '9') {
            char buf[15];
            memcpy(buf, dest, 9); buf[9] = 0;
            int ok9  = (check_digit_of(buf, 9) == (dest[9]  - '0'));
            memcpy(buf, dest + 13, 6); buf[6] = 0;
            int ok19 = (check_digit_of(buf, 6) == (dest[19] - '0'));
            memcpy(buf, dest + 21, 6); buf[6] = 0;
            int ok27 = (check_digit_of(buf, 6) == (dest[27] - '0'));
            if (ok9 && ok19 && ok27) {
                /* 42: only replace a FILLER, never a digit the network
                 * actually emitted (that case belongs to the beam). */
                if (dest[42] == '<') {
                    memcpy(buf, dest + 28, 14); buf[14] = 0;
                    dest[42] = '0' + check_digit_of(buf, 14);
                    if (dg) dg->gate[42] |= 16;
                }
                /* 43: composite over the (possibly just repaired) col 42. */
                if (dest[42] >= '0' && dest[42] <= '9') {
                    memcpy(buf, dest + 28, 14); buf[14] = 0;
                    int ok42 = (check_digit_of(buf, 14) == (dest[42] - '0'));
                    if (ok42) {
                        char comp[40];
                        memcpy(comp, dest, 10);
                        memcpy(comp + 10, dest + 13, 7);
                        memcpy(comp + 17, dest + 21, 22);
                        comp[39] = 0;
                        int expected = check_digit_of(comp, 39);
                        if (dest[43] < '0' || dest[43] > '9' ||
                            (dest[43] - '0') != expected) {
                            dest[43] = '0' + expected;
                            if (dg) dg->gate[43] |= 16;
                        }
                    }
                }
            }
        }
    }

    int sum = 0;
    for (int c = 0; c < total; ++c)
        sum += (int)((t3[c].p[0] > 0 ? t3[c].p[0] : 0.0f) * 100);
    *conf_avg = total > 0 ? sum / total : 0;
    if (dg) {
        dg->conf_avg = *conf_avg;
        if (amb) {
            dg->ambiguous = amb->ambiguous;
            dg->alt_margin = amb->margin;
            memcpy(dg->alt, amb->alt, 45);
            dg->alt[44] = 0;
        }
        for (int c = 0; c < total && c < MRZ_DIAG_CHARS; ++c)
            dg->ch[c] = (unsigned char)dest[c];
    }
}

/* Band tilt estimation + deskew live in the shared geometry layer
 * (src/mrz_geom.c): mrz_estimate_band_skew() / mrz_rotate_band().
 * They are used by both the CNN and legacy backends. */

/* ---- Process-static workspace scratchpad ----
 *
 * The CNN backend needs ~1.3 MB of working buffers per image
 * (gray + bin + band_gray + band_bin + rot + line_pixels + glyphs
 * + feats + probs + cnn_t + overhead).  Allocating that on every
 * call (12 malloc/free pairs in the original) was the largest
 * source of jank on the NDK ARMv7 build.
 *
 * The safe alternative: a single process-static scratchpad
 * (4 MB, 64-byte aligned) handed out by a bump allocator.
 * The scratchpad is reset on every recogniser entry; the JNI
 * dispatcher is single-threaded so two concurrent calls are not
 * supported.  No free() in the hot path: the bump allocator
 * only moves a pointer.
 *
 * pthread_once + a static guard initialise the scratchpad
 * exactly once on the first call.  All pointers below are
 * 64-byte aligned so they sit on cache-line boundaries
 * (helpful for the cnn_t weight matrix read pattern).
 */
/* scratchpad split into two halves (Step 2).  Line 0 uses half 0;
 * line 1 uses half 1.  Even on the current sequential path this
 * gives the bump allocator disjoint regions so adding pthread-based
 * dual-line parallelism later is just a one-line toggle
 * (MRZ_OCR_PAR=1).
 *
 * 64 MB (32 MB/half): a passport-machine 2592x1944 photo needs
 * gray + bin (~2 x 5.04 MB) plus band_gray + band_bin (~2 x
 * 1.9 MB) in ONE half before the line split; 4 MB/half overflowed
 * to MRZ_OCR_ERR_LOAD on full-res photos. The 16 MB/half keeps
 * the full-res path working without exceeding embedded RAM
 * budgets. */
#define MRZ_SCRATCH_BYTES  (64u * 1024u * 1024u)
static uint8_t  g_mrz_scratch[MRZ_SCRATCH_BYTES]
                __attribute__((aligned(64)));
/* g_mrz_scratch_off removed (Step 2): per-half offsets
 * g_mrz_scratch_off0/1 are tracked below. */
static int      g_mrz_scratch_active = 0;
static size_t   g_mrz_scratch_off0 = 0;
static size_t   g_mrz_scratch_off1 = 0;
static pthread_once_t g_mrz_scratch_once = PTHREAD_ONCE_INIT;
static int      g_mrz_scratch_cnn_ready = 0;
static cnn_t    g_mrz_scratch_cnn_template;  /* one-shot weights cache */
static cnn_t    g_mrz_scratch_cnn_line1;
static cnn_t    g_mrz_scratch_cnn_line2;
static crnn_pair_t g_mrz_scratch_crnn;

static void mrz_scratch_init_once(void) {
    g_mrz_scratch_off0 = 0;
    g_mrz_scratch_off1 = 0;
    if (!g_mrz_scratch_cnn_ready) {
        cnn_default_init(&g_mrz_scratch_cnn_template);
        cnn_line_init(&g_mrz_scratch_cnn_line1, 0);
        cnn_line_init(&g_mrz_scratch_cnn_line2, 1);
        g_mrz_scratch_cnn_ready = 1;
    }
}

/* 64-byte aligned bump allocator. */
#define MRZ_SCRATCH_HALF_BYTES (MRZ_SCRATCH_BYTES / 2u)


void mrz_scratch_select_half(int half) { g_mrz_scratch_active = half; }
static void *mrz_scratch_active_base(void) {
    return g_mrz_scratch + (g_mrz_scratch_active == 0 ? 0 : MRZ_SCRATCH_HALF_BYTES);
}
static size_t mrz_scratch_active_off(void) {
    return g_mrz_scratch_active == 0 ? g_mrz_scratch_off0 : g_mrz_scratch_off1;
}
static void mrz_scratch_set_active_off(size_t off) {
    if (g_mrz_scratch_active == 0) g_mrz_scratch_off0 = off;
    else                          g_mrz_scratch_off1 = off;
}

static void *mrz_scratch_alloc(size_t need) {
    pthread_once(&g_mrz_scratch_once, mrz_scratch_init_once);
    size_t aligned = (need + 63u) & ~(size_t)63u;
    size_t off = mrz_scratch_active_off();
    if (off + aligned > MRZ_SCRATCH_HALF_BYTES) return NULL;
    void *p = (uint8_t *)mrz_scratch_active_base() + off;
    mrz_scratch_set_active_off(off + aligned);
    return p;
}

/* Reset for a fresh recogniser call.  O(1); no system calls. */
static void mrz_scratch_reset(void) {
    g_mrz_scratch_off0 = 0;
    g_mrz_scratch_off1 = 0;
    g_mrz_scratch_active = 0;
}

/* Per-line worker context (Step 2: reserved for the future
 * dual-line pthread parallelism).  When MRZ_OCR_PAR=1, the
 * main thread spawns one worker that runs the CNN forward
 * for line 1 in parallel with the main thread doing line 0.
 * Per-line state lives in disjoint scratchpad halves so the
 * bump allocator never aliases across threads. */
typedef struct {
    const cnn_t   *net;
    /* glyphs is the canonical input; tta_glyphs holds the +-1 shift
     * snapshots so the worker can run TTA in-thread without the main
     * thread having to share band_gray. */
    float        (*glyphs)[CNN_IN_H][CNN_IN_W];
    float        (*tta_glyphs)[CNN_IN_H][CNN_IN_W];
    float         *feats;
    float         *probs;
    float         *tta_probs;
    float         *seq_probs;
    int            line_idx;
    int            nchars;
    /* decode_row inputs */
    const uint8_t *band_gray;
    int            bw, bh;
    const mrz_ocr_rect_t *chars;
    int            base_y;
    int            tta;
    /* Sequence head / fusion.  When `use_seq` is set the worker runs
     * the CRNN head (and, when `use_fuse` is also set, fuses it with
     * the per-character CNN head) instead of the plain CNN FC.  This
     * is what lets the fused engine keep the dual-line parallel
     * split: the sequence recurrence is per row, so line 1 and line 2
     * are still independent work. */
    int            use_seq;
    int            use_fuse;
    /* MRZ_OCR_FUSE_MODE: 1 = "arb" (default, CNN decode + CRNN
     * arbitration via decode_row), 0 = legacy in-place probability
     * fusion (mrz_fuse_probs). */
    int            fuse_arb;
    float          arb_gap;
    float          arb_margin;
    float          arb_conf;
    float          fuse_alpha;
    float          fuse_gate;
    float          fuse_ratio;
    /* Snapshot of the submitting thread's deadline.  The deadline is
     * thread-local, and this job runs on the worker whose TLS is empty
     * without it -- see mrz_line1_run. */
    struct timespec deadline;
    int             deadline_enabled;
    /* outputs */
    char          *line_out;
    int           *conf_avg;
    /* Ambiguity of the decoded line.  The worker path decodes line 2
     * on the device (that is the line the parallel worker is given),
     * so the ambiguity has to travel back through the job -- passing
     * NULL here would silently drop the flag on exactly the path the
     * product uses. */
    mrz_decode_amb_t *amb;
} mrz_line1_job_t;

/* CPU affinity: the 4x Cortex-A53 are homogeneous, but the Linux
 * scheduler (schedplus) still bounces a worker thread across cores,
 * which thrashes the L1 (32KB)/L2 (256KB) caches and inflates the
 * p50 on every image.  Pin the worker to a specific core at spawn so
 * it stays put for the whole batch.  Failure is non-fatal (falls back
 * to the scheduler's default placement). */
#ifdef __linux__
static void mrz_pin_to_cpu(unsigned int cpu) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET((int)cpu, &mask);
    sched_setaffinity(0, sizeof(mask), &mask);
}
#else
static void mrz_pin_to_cpu(unsigned int cpu) { (void)cpu; }
#endif

/* The actual line-1 work: forward + TTA + decode.  Kept as its own
 * function so both the resident worker loop (__linux__) and the
 * fallback in-line path share one implementation. */
/* Run the classification head for one row and write the probabilities
 * into j->probs.  Shared by the worker and the main thread so both
 * paths produce byte-identical probabilities. */
static void mrz_row_classify(mrz_line1_job_t *j) {
    cnn_row_features_batch(j->net, j->glyphs, j->nchars, j->feats);
    if (j->use_fuse && j->fuse_arb) {
        /* ARBITRATION mode (default): the decode (grammar mask, ink gate,
         * checksum beam) runs on the PURE CNN probabilities, so the
         * checksum-beam repairs survive; the sequence head's verdict goes
         * to seq_probs and decode_row consults it only on the columns
         * where its runner-up disagrees. */
        cnn_fc_batch(j->net, j->feats, j->nchars, j->probs);
        if (j->seq_probs)
            crnn_forward_line(&g_mrz_scratch_crnn, j->line_idx, j->feats,
                              j->nchars, j->seq_probs);
    } else if (j->use_fuse) {
        /* Legacy in-place probability fusion (MRZ_OCR_FUSE_MODE=probs),
         * kept as the A/B comparison path. */
        cnn_fc_batch(j->net, j->feats, j->nchars, j->probs);
        float *seq = j->seq_probs;
        if (seq) {
            crnn_forward_line(&g_mrz_scratch_crnn, j->line_idx, j->feats,
                              j->nchars, seq);
            mrz_fuse_probs(j->probs, seq, j->nchars, j->fuse_alpha,
                           j->fuse_gate, j->fuse_ratio);
        }
    } else if (j->use_seq) {
        crnn_forward_line(&g_mrz_scratch_crnn, j->line_idx, j->feats,
                          j->nchars, j->probs);
    } else {
        cnn_fc_batch(j->net, j->feats, j->nchars, j->probs);
    }
}

static void mrz_line1_run(mrz_line1_job_t *j) {
    /* Inherit the submitting (main) thread's deadline: this TLS is empty
     * on the resident worker, so without the snapshot the deadline checks
     * would never fire here and a slow line-1 (TTA + decode) could extend
     * the frame past its budget while mrz_resident_wait blocks.  The wait
     * itself stays unbounded ON PURPOSE: abandoning it would let the
     * worker write into the caller's line2 buffer after the frame has
     * already been returned to the JNI layer -- the exact use-after-return
     * the `pending` counter was introduced to prevent.  Bounding the
     * WORKER instead is the safe half of that trade. */
    mrz_ocr_deadline_apply(&j->deadline, j->deadline_enabled);
    mrz_row_classify(j);
    /* TTA is 3 forwards; it is the first thing to drop when the frame is
     * running out of budget.  decode_row still runs so line_out is always
     * written -- an unwritten caller buffer is the one failure mode this
     * path must never produce. */
    if (j->tta && !mrz_ocr_deadline_expired()) {
        /* +-1px horizontal shift TTA averaged into j->probs. */
        for (int c = 0; c < j->nchars; ++c)
            for (int o = 0; o < CNN_OUT; ++o)
                j->tta_probs[c * CNN_OUT + o] = j->probs[c * CNN_OUT + o];
        for (int sh = -1; sh <= 1; sh += 2) {
            for (int c = 0; c < j->nchars; ++c)
                resample_char_gray(j->band_gray, j->bw, j->bh,
                                   j->chars[c].x + sh,
                                   j->base_y + j->chars[c].y,
                                   j->chars[c].w, j->chars[c].h,
                                   NULL, 0, j->tta_glyphs[c]);
            cnn_row_features_batch(j->net, j->tta_glyphs, j->nchars, j->feats);
            if (j->use_fuse && j->fuse_arb) {
                /* Arbitration mode: only the CNN probs are TTA-averaged;
                 * the CRNN verdict stays the BASE one from
                 * mrz_row_classify -- a +-1px shift rarely moves the
                 * sequence head and this avoids a second accumulation
                 * buffer. */
                cnn_fc_batch(j->net, j->feats, j->nchars, j->probs);
            } else if (j->use_fuse) {
                cnn_fc_batch(j->net, j->feats, j->nchars, j->probs);
                float *seq = j->seq_probs;
                if (seq) {
                    crnn_forward_line(&g_mrz_scratch_crnn, j->line_idx, j->feats,
                                      j->nchars, seq);
                    mrz_fuse_probs(j->probs, seq, j->nchars, j->fuse_alpha,
                                   j->fuse_gate, j->fuse_ratio);
                }
            } else if (j->use_seq) {
                crnn_forward_line(&g_mrz_scratch_crnn, j->line_idx, j->feats,
                                  j->nchars, j->probs);
            } else {
                cnn_fc_batch(j->net, j->feats, j->nchars, j->probs);
            }
            for (int c = 0; c < j->nchars; ++c)
                for (int o = 0; o < CNN_OUT; ++o)
                    j->tta_probs[c * CNN_OUT + o] += j->probs[c * CNN_OUT + o];
        }
        for (int c = 0; c < j->nchars; ++c)
            for (int o = 0; o < CNN_OUT; ++o)
                j->probs[c * CNN_OUT + o] = j->tta_probs[c * CNN_OUT + o] / 3.0f;
    }
    decode_row((const float (*)[CNN_IN_H][CNN_IN_W])j->glyphs,
               j->nchars, 1, j->line_out, j->conf_avg, j->probs, j->amb,
               (j->use_fuse && j->fuse_arb) ? j->seq_probs : NULL,
               j->arb_gap, j->arb_margin, j->arb_conf);
}

/* Thin pthread entry used on non-Linux hosts (MAC) or as a fallback
 * when the resident worker cannot be created. */
#ifndef __linux__
static void *mrz_line1_worker(void *arg) {
    mrz_line1_job_t *j = (mrz_line1_job_t *)arg;
    mrz_pin_to_cpu(3);  /* keep the worker away from the main thread's core */
    mrz_line1_run(j);
    return NULL;
}
#endif

#ifdef __linux__
/* ---- resident worker (MRZ_OCR_PAR=1) ----
 * A single worker thread is created on first use and then reused for
 * every image, removing the per-image pthread_create/join (~1 ms of
 * jitter on the p50/p95 tail).  Job dispatch is a condvar handshake:
 *
 *   main: lock; copy job; has_job=1; pending++; signal ready; unlock
 *   main: ... do line 0 ...
 *   main: lock; while pending cond_wait done; unlock
 *   worker: wait ready; run; pending--; signal done
 *
 * The recogniser is single-threaded on the JNI side, so each submit
 * is always preceded by a completed wait of the previous frame; the
 * shared job slot never races.
 *
 * `pending` is a counter, not a flag: see the comment on
 * g_mrz_worker_pending for why a flag was not sufficient. */
static pthread_mutex_t g_mrz_worker_mu    = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_mrz_worker_ready = PTHREAD_COND_INITIALIZER;
static pthread_cond_t  g_mrz_worker_done  = PTHREAD_COND_INITIALIZER;
static mrz_line1_job_t g_mrz_worker_job;
static int g_mrz_worker_has_job = 0;
/* Number of jobs handed to the worker that have not finished yet.
 *
 * The previous handshake used a single `busy` flag that the WORKER set
 * to 1 only after it had woken up and taken the job.  The submitter
 * sets `has_job` and returns immediately, so between the submit and
 * the worker's first scheduling the flag was still 0 -- and the
 * submitter's wait loop ("while (busy) wait") saw 0, decided the work
 * was already done, and returned.  The worker then wrote line 2 into
 * the caller's stack frame, which the JNI shim had already reused for
 * the next image: the MRZ text landed on top of a pointer slot and the
 * next JNI call dereferenced it.
 *
 * That is a use-after-free that only manifests on Linux (the resident
 * worker is `#ifdef __linux__`), so the host build never showed it --
 * it only crashed on the device, on the second image of a batch.
 *
 * Counting instead of flagging makes the wait condition exact: the
 * submitter increments `pending` under the mutex, and the worker
 * decrements it when the job is really finished.  The submitter waits
 * until `pending == 0`, which can only happen after the worker has
 * completed the job it was given. */
static int g_mrz_worker_pending = 0;
static int g_mrz_worker_ready_created = 0;

static void *mrz_resident_worker_loop(void *arg) {
    (void)arg;
    mrz_pin_to_cpu(3);
    for (;;) {
        pthread_mutex_lock(&g_mrz_worker_mu);
        while (!g_mrz_worker_has_job)
            pthread_cond_wait(&g_mrz_worker_ready, &g_mrz_worker_mu);
        mrz_line1_job_t j = g_mrz_worker_job; /* copy (all pointer fields) */
        g_mrz_worker_has_job = 0;
        pthread_mutex_unlock(&g_mrz_worker_mu);

        mrz_line1_run(&j);

        pthread_mutex_lock(&g_mrz_worker_mu);
        g_mrz_worker_pending--;
        pthread_cond_signal(&g_mrz_worker_done);
        pthread_mutex_unlock(&g_mrz_worker_mu);
    }
    return NULL;
}

/* Submit a job; returns 0 on success, -1 if the worker could not be
 * created this time (caller falls back to in-line). */
static int mrz_resident_submit(const mrz_line1_job_t *job) {
    int created = 0;
    pthread_mutex_lock(&g_mrz_worker_mu);
    if (!g_mrz_worker_ready_created) {
        pthread_t tid;
        if (pthread_create(&tid, NULL, mrz_resident_worker_loop,
                           NULL) == 0) {
            g_mrz_worker_ready_created = 1;
            created = 1;
        } else {
            pthread_mutex_unlock(&g_mrz_worker_mu);
            return -1;
        }
    }
    (void)created;
    g_mrz_worker_job = *job;
    g_mrz_worker_has_job = 1;
    /* Count the job as outstanding BEFORE the worker can possibly see
     * it.  Doing it under the same mutex as `has_job` means the
     * submitter's wait loop below can never observe a state where the
     * job is visible to the worker but not yet accounted for. */
    g_mrz_worker_pending++;
    pthread_cond_signal(&g_mrz_worker_ready);
    pthread_mutex_unlock(&g_mrz_worker_mu);
    return 0;
}

static void mrz_resident_wait(void) {
    pthread_mutex_lock(&g_mrz_worker_mu);
    while (g_mrz_worker_pending > 0)
        pthread_cond_wait(&g_mrz_worker_done, &g_mrz_worker_mu);
    pthread_mutex_unlock(&g_mrz_worker_mu);
}
#endif /* __linux__ */

/* Convenience: typed allocations for the ten safe fields. */
static uint8_t *mrz_scratch_gray    (size_t n)              { return (uint8_t *)mrz_scratch_alloc(n); }
static uint8_t *mrz_scratch_bin     (size_t n)              { return (uint8_t *)mrz_scratch_alloc(n); }
static uint8_t *mrz_scratch_band_gray(size_t n)             { return (uint8_t *)mrz_scratch_alloc(n); }
static uint8_t *mrz_scratch_band_bin (size_t n)             { return (uint8_t *)mrz_scratch_alloc(n); }
static uint8_t *mrz_scratch_rot     (size_t n)              { return (uint8_t *)mrz_scratch_alloc(n); }
static uint8_t *mrz_scratch_norm    (size_t n)              { return (uint8_t *)mrz_scratch_alloc(n); }
static uint8_t *mrz_scratch_layout_bin(size_t n)            { return (uint8_t *)mrz_scratch_alloc(n); }
static uint8_t *mrz_scratch_line    (size_t n)              { return (uint8_t *)mrz_scratch_alloc(n); }

/* Sauvola binarization: local threshold T = m * (1 + k*(s/R - 1)).
 * Better than Otsu for documents with guilloche / patterned backgrounds
 * (CS/CD travel permits) where a global threshold merges text and
 * background into one blob.  Uses integral images for O(1) per-pixel
 * local mean/std.  Window is ~bw/40 (one half character cell). */
static void mrz_sauvola(const uint8_t *gray, int w, int h, uint8_t *bin,
                        int window, float k) {
    if (w <= 0 || h <= 0) return;
    int r = window / 2;
    if (r < 3) r = 3;
    int wh = 2 * r + 1;
    size_t iw = (size_t)(w + 1), ih = (size_t)(h + 1);
    int64_t *isum  = (int64_t *)calloc(iw * ih, sizeof(int64_t));
    int64_t *isumsq = (int64_t *)calloc(iw * ih, sizeof(int64_t));
    if (!isum || !isumsq) { free(isum); free(isumsq); return; }
    for (int y = 0; y < h; ++y) {
        int64_t rs = 0, rsq = 0;
        for (int x = 0; x < w; ++x) {
            int v = gray[(size_t)y * w + x];
            rs  += v;
            rsq += (int64_t)v * v;
            isum   [(size_t)(y+1) * iw + (x+1)] = isum  [(size_t)y * iw + x] + rs;
            isumsq [(size_t)(y+1) * iw + (x+1)] = isumsq[(size_t)y * iw + x] + rsq;
        }
    }
    const float R = 128.0f;
    for (int y = 0; y < h; ++y) {
        int y0 = y - r; if (y0 < 0) y0 = 0;
        int y1 = y + r; if (y1 >= h) y1 = h - 1;
        int wy = y1 - y0 + 1;
        for (int x = 0; x < w; ++x) {
            int x0 = x - r; if (x0 < 0) x0 = 0;
            int x1 = x + r; if (x1 >= w) x1 = w - 1;
            int wx = x1 - x0 + 1;
            float area = (float)(wx * wy);
            int64_t s  = isum  [(size_t)(y1+1) * iw + (x1+1)]
                       - isum  [(size_t)y0     * iw + (x1+1)]
                       - isum  [(size_t)(y1+1) * iw + x0]
                       + isum  [(size_t)y0     * iw + x0];
            int64_t sq = isumsq [(size_t)(y1+1) * iw + (x1+1)]
                       - isumsq [(size_t)y0     * iw + (x1+1)]
                       - isumsq [(size_t)(y1+1) * iw + x0]
                       + isumsq [(size_t)y0     * iw + x0];
            float m = (float)s / area;
            float var = (float)sq / area - m * m;
            if (var < 0.0f) var = 0.0f;
            float sd = sqrtf(var);
            float t = m * (1.0f + k * (sd / R - 1.0f));
            bin[(size_t)y * w + x] = (gray[(size_t)y * w + x] < t) ? 1 : 0;
        }
    }
    free(isum);
    free(isumsq);
}

/* Row-projection energy: sum of squared row ink counts.  Same metric
 * as the skew estimator -- maximised when text rows are sharp and
 * distinct, minimised by blur or curvature that spreads ink across
 * gap rows.  Used as the dewarp safety gate. */
static double mrz_row_proj_energy(const uint8_t *bin, int W, int H) {
    if (W <= 0 || H <= 0) return 0.0;
    long *proj = (long *)calloc((size_t)H, sizeof(long));
    if (!proj) return 0.0;
    for (int y = 0; y < H; ++y) {
        const uint8_t *row = bin + (size_t)y * W;
        long c = 0;
        for (int x = 0; x < W; ++x) c += row[x];
        proj[y] = c;
    }
    double score = 0.0;
    for (int y = 0; y < H; ++y) score += (double)proj[y] * (double)proj[y];
    free(proj);
    return score;
}
static float   *mrz_scratch_glyphs  (size_t n_floats)       { return (float *)mrz_scratch_alloc(n_floats * sizeof(float)); }
static float   *mrz_scratch_feats   (size_t n_floats)       { return (float *)mrz_scratch_alloc(n_floats * sizeof(float)); }
static float   *mrz_scratch_probs   (size_t n_floats)       { return (float *)mrz_scratch_alloc(n_floats * sizeof(float)); }

/* ---- top-level pipeline ---- */
mrz_ocr_status_t mrz_ocr_recognise_cnn(const face_image_t *img,
                                       mrz_ocr_result_t *out) {
    mrz_scratch_reset();   /* O(1) reset before any alloc */
    if (!img || !out) return MRZ_OCR_ERR_LOAD;
    if (mrz_ocr_deadline_expired()) return MRZ_OCR_ERR_TIMEOUT;
    memset(out, 0, sizeof(*out));
    int W = img->width, H = img->height;
    if (W <= 0 || H <= 0) return MRZ_OCR_ERR_LOAD;

    const char *tm = getenv("MRZ_OCR_TIMING");
    const int diag_on = mrz_diag_on();
    if (diag_on) {
        memset(g_diag.line, 0, sizeof(g_diag.line));
        g_diag.img_w = W; g_diag.img_h = H;
        g_diag.band_x = g_diag.band_y = g_diag.band_w = g_diag.band_h = 0;
        g_diag.proc_w = g_diag.proc_h = 0;
        g_diag.upscale = 1.0;
        g_diag.skew_deg = 0.0;
        g_diag.status = MRZ_OCR_OK;
        g_diag.ms_gray = g_diag.ms_band = 0.0;
        g_diag.ms_resample = g_diag.ms_decode = 0.0;
    }
    const int timing_on = diag_on || (tm && tm[0] && strcmp(tm, "0"));
    struct timespec _t0, _t1;
    double _ms_gray = 0, _ms_band = 0, _ms_resample = 0, _ms_decode = 0;
    if (timing_on) clock_gettime(CLOCK_MONOTONIC, &_t0);

    /* Heap workspaces (the CNN weights alone are ~41 KB — keeping the
     * whole pipeline off the stack matters for embedded thread stacks). */
    size_t n = (size_t)W * H;
    uint8_t *gray = mrz_scratch_gray(n);
    uint8_t *bin = NULL, *band_gray = NULL, *band_bin = NULL, *rot = NULL;
    uint8_t *layout_bin = NULL, *glyph_sauvola = NULL, *line_pixels = NULL;
    const cnn_t *net0 = NULL;
    const cnn_t *net1 = NULL;
    float (*glyphs)[CNN_IN_H][CNN_IN_W] = NULL;
    float (*glyphs_tta)[CNN_IN_H][CNN_IN_W] = NULL;
    float *feats = NULL, *probs = NULL, *probs_tta = NULL;
    if (!gray) return MRZ_OCR_ERR_LOAD;

    mrz_ocr_status_t status = MRZ_OCR_OK;
    face_to_grayscale(img, gray);
    if (mrz_ocr_deadline_expired()) {
        status = MRZ_OCR_ERR_TIMEOUT;
        goto cleanup;
    }

    /* 1. Global Otsu -> coarse bin map for band location. Polarity is
     * normalised first (white-on-black scans would otherwise mark the
     * background as ink everywhere downstream). */
    int t = mrz_ocr_otsu(gray, (int)n);
    {
        size_t dark = 0;
        for (size_t i = 0; i < n; ++i) if (gray[i] < t) dark++;
        if (dark * 2 > n) {
            for (size_t i = 0; i < n; ++i) gray[i] = (uint8_t)(255 - gray[i]);
            t = 255 - t;
        }
    }
    bin = mrz_scratch_bin(n);
    if (!bin) { status = MRZ_OCR_ERR_LOAD; goto cleanup; }
    for (size_t i = 0; i < n; ++i) bin[i] = (gray[i] < t) ? 1 : 0;
    if (timing_on) { clock_gettime(CLOCK_MONOTONIC, &_t1); _ms_gray = (_t1.tv_sec-_t0.tv_sec)*1e3+(_t1.tv_nsec-_t0.tv_nsec)/1e6; _t0=_t1; }

    /* 2. Locate the MRZ band.
     *
     * NOTE: locate_band() assumes a PAGE photo where the MRZ is the only
     * dense block. On a document scan (the whole frame is the card) it
     * degenerates to the full image -- measured on assets/pic: band ==
     * frame on 84/84. That is expected: the Android path crops the band
     * in Kotlin (fastPair/nativeBand) before calling in, and the host
     * harness crops with the board bbox. A C-side bottom-up rescue was
     * prototyped and rejected: the guilloche background of these scans
     * is full-width ink, so row span/ink tests cannot separate MRZ from
     * decoration. Do not re-add it without a texture-based locator. */
    mrz_ocr_rect_t band;
    if (mrz_ocr_locate_band(bin, W, H, &band) != 0) {
        /* Retry with a looser density threshold, but ONLY on a band so
         * short that it cannot hold two TD3 rows.  A CS/CD (single
         * 30-character row) only fills ~9% of the band width, so
         * locate_band()'s 10%-of-width test classifies every one of its
         * rows as background (measured 1440x51 band: peak row ink 132 px
         * = 9.2% W -> NO_BAND).
         *
         * The retry is deliberately NOT applied to full-height inputs:
         * lowering the threshold globally makes locate_band() drift on
         * the TD3 corpus (measured on the 114-photo product-band set:
         * TD3 exact 39/108 -> 18/108, 20 photos broken).  A short band
         * (H <= 80) is already a pre-cropped MRZ region, so accepting
         * its whole height cannot introduce a new rectangle. */
        if (H > 80) {
            status = MRZ_OCR_ERR_NO_BAND; goto cleanup;
        }
        /* Rebuild the row-ink profile with a 6% threshold and take the
         * whole input as the band: the caller already cropped it. */
        int thr6 = (int)(W * 0.06);
        if (thr6 < 2) thr6 = 2;
        int any = 0;
        for (int y = 0; y < H; ++y) {
            int c = 0;
            for (int x = 0; x < W; ++x) if (bin[(size_t)y * W + x]) c++;
            if (c >= thr6) { any = 1; break; }
        }
        if (!any) { status = MRZ_OCR_ERR_NO_BAND; goto cleanup; }
        band.x = 0; band.y = 0; band.w = W; band.h = H;
    }
    /* When the caller sends a pre-cropped band (H <= 80) locate_band()
     * may over-refine and find only the text core (e.g. 6 px inside a
     * 31 px CS/CD band).  That strips the vertical context the splitter
     * and CNN need.  Fall back to the full input in that case. */
    if (H <= 80 && band.h < H / 2) {
        band.x = 0; band.y = 0; band.w = W; band.h = H;
    }
    if (mrz_ocr_deadline_expired()) {
        status = MRZ_OCR_ERR_TIMEOUT;
        goto cleanup;
    }
    int bx = band.x, by = band.y, bw = band.w, bh = band.h;
    out->band_x = bx; out->band_y = by; out->band_w = bw; out->band_h = bh;
    if (diag_on) {
        g_diag.band_x = bx; g_diag.band_y = by;
        g_diag.band_w = bw; g_diag.band_h = bh;
    }
    if (bw < MRZ_OCR_GLYPH_W * 5 || bh < 4) {
        status = MRZ_OCR_ERR_BAD_GEOMETRY; goto cleanup;
    }

    /* 3. Band-local refine: crop the band's grayscale, Otsu inside the
     * band only (a global threshold over a full passport page can be
     * dominated by the photo / background), estimate + remove tilt. */
    /* 2b. Scale normalisation (see the comment above mrz_upscale_gray).
     * The band is upscaled so that one character column is at least
     * MRZ_OCR_TARGET_PPC pixels wide; it is NEVER downscaled, so a band
     * that is already wide enough passes through bit-identical.  That is
     * why the device corpus is unchanged by this step.
     * out->band_* above keeps reporting the ORIGINAL rectangle -- the
     * caller maps it back onto the frame, so bw/bh must not leak out. */
    double diag_up = 1.0;
    {
        const char *ppc_env = getenv("MRZ_OCR_PPC");
        double target = MRZ_OCR_TARGET_PPC;
        if (ppc_env && ppc_env[0]) {
            double v = atof(ppc_env);
            if (v >= 8.0 && v <= 128.0) target = v;
        }
        /* Cell count of the band.  TD3 has 44 cells per row, CS/CD 30, but
         * the layout is only classified after the band has been binarised
         * and split (further down), so the safe value for the TD3 path is
         * used here: 44.  Dividing by 30 instead under-scales the TD3
         * upscale by 30/44 and measurably costs TD3 decode accuracy. */
        double ppc = (double)bw / 44.0;
        double up  = (ppc > 0.0) ? target / ppc : 1.0;
        if (up < 1.0) up = 1.0;
        /* Height-based upscaling for short bands (CS/CD guilloche
         * background can produce bands as short as 6-8px). Ensure the
         * upscaled height is at least 2 * MRZ_OCR_GLYPH_H so the CNN
         * has enough pixels to work with. */
        {
            double min_h = (double)MRZ_OCR_GLYPH_H * 2.0;
            if ((double)bh * up < min_h) {
                double up_h = min_h / (double)bh;
                if (up_h > up) up = up_h;
            }
        }
        if (up > 4.0) up = 4.0;          /* bound the scratch footprint */
        diag_up = up;
        int nw = (int)((double)bw * up + 0.5);
        int nh = (int)((double)bh * up + 0.5);
        if (nw < bw) nw = bw;
        if (nh < bh) nh = bh;
        band_gray = mrz_scratch_band_gray((size_t)nw * nh);
        band_bin  = mrz_scratch_band_bin ((size_t)nw * nh);
        layout_bin = mrz_scratch_layout_bin((size_t)nw * nh);
        glyph_sauvola = mrz_scratch_layout_bin((size_t)nw * nh);
        if (!band_gray || !band_bin || !layout_bin || !glyph_sauvola) { status = MRZ_OCR_ERR_LOAD; goto cleanup; }
        if (nw == bw && nh == bh) {
            for (int y = 0; y < bh; ++y)
                memcpy(band_gray + y * bw, gray + (by + y) * W + bx, bw);
        } else {
            /* Cropped source lives in a scratch buffer, not in `gray`,
             * because the upscale reads four neighbours per output pixel
             * and `gray` is the whole frame with a different stride. */
            uint8_t *src = mrz_scratch_rot((size_t)bw * bh);
            if (!src) { status = MRZ_OCR_ERR_LOAD; goto cleanup; }
            for (int y = 0; y < bh; ++y)
                memcpy(src + y * bw, gray + (by + y) * W + bx, bw);
            mrz_upscale_gray(src, bw, bh, band_gray, nw, nh);
            bw = nw; bh = nh;
        }
    }
    if (diag_on) {
        g_diag.proc_w = bw; g_diag.proc_h = bh; g_diag.upscale = diag_up;
    }
    {
        int tb = mrz_ocr_otsu(band_gray, bw * bh);
        for (int i = 0; i < bw * bh; ++i) band_bin[i] = (band_gray[i] < tb) ? 1 : 0;
        double deg = mrz_estimate_band_skew(band_bin, bw, bh);
        if (diag_on) g_diag.skew_deg = deg;
        {   /* MRZ_OCR_GRID=1: report the deskew angle.  A non-zero
             * angle rotates the whole band, which shifts every glyph
             * window horizontally by up to (bh/2)*sin(deg). */
            static int skdbg = -1;
            if (skdbg < 0) { const char *e = getenv("MRZ_OCR_GRID"); skdbg = (e && e[0] && strcmp(e,"0") != 0) ? 1 : 0; }
            if (skdbg) fprintf(stderr, "SKEW deg=%.4f bw=%d bh=%d\n", deg, bw, bh);
        }
        if (deg != 0.0) {
            rot = mrz_scratch_rot((size_t)bw * bh);
            if (!rot) { status = MRZ_OCR_ERR_LOAD; goto cleanup; }
            /* Bilinear deskew is OPT-IN (MRZ_OCR_BILINEAR=1). On the
             * current (nearest-neighbour-trained) weights it raises
             * char/cksum but LOWERS full-match, i.e. it changes the
             * input distribution and only pays off after a
             * distribution-matched retrain. Default = NN stays. */
            const char *bl_env = getenv("MRZ_OCR_BILINEAR");
            int bl = (bl_env && bl_env[0] && strcmp(bl_env, "0") != 0);
            mrz_rotate_band(band_gray, bw, bh, deg, 255, bl, rot);
            memcpy(band_gray, rot, (size_t)bw * bh);
            for (int i = 0; i < bw * bh; ++i) band_bin[i] = (band_gray[i] < tb) ? 1 : 0;
        }
    }
    {
        static int dw_en = -1;
        if (dw_en < 0) { const char *e = getenv("MRZ_OCR_DEWARP"); dw_en = (e && e[0] && strcmp(e,"0") != 0) ? 1 : 0; }
        if (dw_en) {
            /* Safety gate (step 2 of the cascade): only keep the dewarped
             * version if its row-projection energy improves by >= 12% over
             * the baseline.  This prevents the parabolic dewarp from
             * damaging straight bands where the fitted parabola is noise
             * rather than real curvature. */
            double base_score = mrz_row_proj_energy(band_bin, bw, bh);
            uint8_t *dwp = mrz_scratch_rot((size_t)bw * bh);
            if (dwp) {
                mrz_dewarp_parabolic(band_gray, dwp, bw, bh, bw, bw);
                int tb2 = mrz_ocr_otsu(dwp, (size_t)bw * bh);
                uint8_t *dwp_bin = mrz_scratch_rot((size_t)bw * bh);
                if (dwp_bin) {
                    for (int i = 0; i < bw * bh; ++i)
                        dwp_bin[i] = (dwp[i] < tb2) ? 1 : 0;
                    double new_score = mrz_row_proj_energy(dwp_bin, bw, bh);
                    if (new_score > base_score * 1.12) {
                        memcpy(band_gray, dwp, (size_t)bw * bh);
                        memcpy(band_bin, dwp_bin, (size_t)bw * bh);
                    }
                } else {
                    memcpy(band_gray, dwp, (size_t)bw * bh);
                    for (int i = 0; i < bw * bh; ++i)
                        band_bin[i] = (dwp[i] < tb2) ? 1 : 0;
                }
            }
        }
    }
    /* Sauvola binarization for layout analysis (split_lines + classify_layout).
     * Otsu merges text with guilloche backgrounds on CS/CD permits; Sauvola's
     * local threshold separates them, giving a cleaner line rectangle and
     * ink-group count.  The Otsu band_bin is kept for glyph extraction
     * because the CNN weights were trained on Otsu binaries. */
    {
        int sw = bw / 40;
        if (sw < 7) sw = 7;
        if (sw > 31) sw = 31;
        float sk = 0.2f;
        {   /* MRZ_OCR_SAUVOLA_K=0.2 overrides the Sauvola k parameter. */
            const char *ke = getenv("MRZ_OCR_SAUVOLA_K");
            if (ke && ke[0]) { float v = (float)atof(ke); if (v > 0.0f && v < 1.0f) sk = v; }
        }
        mrz_sauvola(band_gray, bw, bh, layout_bin, sw, sk);
        /* The second Sauvola (lower k, glyph extraction) is computed
         * lazily at the glyph-bin selection below: it is consumed only
         * on the CS/CD single-row path (MRZ_OCR_GLYPH_BIN=s + SINGLE_
         * LINE_30), so the TD3 product path must not pay a full-band
         * Sauvola (two integral images + per-pixel sqrtf) per frame. */
        {   /* MRZ_OCR_SAUVOLA=1: dump row profile for debugging. */
            static int sdbg = -1;
            if (sdbg < 0) { const char *e = getenv("MRZ_OCR_SAUVOLA"); sdbg = (e && e[0] && strcmp(e,"0") != 0) ? 1 : 0; }
            if (sdbg) {
                int *rd = (int *)calloc((size_t)bh, sizeof(int));
                if (rd) {
                    for (int y = 0; y < bh; ++y)
                        for (int x = 0; x < bw; ++x)
                            if (layout_bin[(size_t)y*bw+x]) rd[y]++;
                    int pk = 0; for (int y = 0; y < bh; ++y) if (rd[y] > pk) pk = rd[y];
                    int th = pk / 6; if (th < 4) th = 4;
                    fprintf(stderr, "SAUVOLA window=%d k=%.2f bw=%d bh=%d peak=%d thr=%d\n", sw, sk, bw, bh, pk, th);
                    for (int y = 0; y < bh; ++y)
                        fprintf(stderr, "  y=%2d dark=%4d %c\n", y, rd[y], rd[y] >= th ? '*' : ' ');
                    free(rd);
                }
            }
        }
    }
    if (timing_on) { clock_gettime(CLOCK_MONOTONIC, &_t1); _ms_band=(_t1.tv_sec-_t0.tv_sec)*1e3+(_t1.tv_nsec-_t0.tv_nsec)/1e6; _t0=_t1; }

    /* 4b. Local-contrast normalisation against illumination gradients.
     * OPT-IN (MRZ_OCR_LOCAL_NORM=1): off by default.  A/B on the 1380
     * eval corpus shows it raises char-level accuracy a hair (95.6->
     * 97.6 / 98.9->99.1 / cksum 94.6->97.1) but LOWERS realistic
     * full-match (45.0%->37.9%) and adds ~0.5 ms -- it is NOT the
     * path to the realistic 45->85% full-match goal. Kept for field
     * A/B; do not flip the default without a full-match win. */
    {
        const char *ln_env = getenv("MRZ_OCR_LOCAL_NORM");
        int ln = (ln_env && ln_env[0] && strcmp(ln_env, "0") != 0);
        if (ln) {
            uint8_t *norm = mrz_scratch_norm((size_t)bw * bh);
            if (!norm) { status = MRZ_OCR_ERR_LOAD; goto cleanup; }
            mrz_normalize_local_contrast(band_gray, bw, bh, norm);
            memcpy(band_gray, norm, (size_t)bw * bh);
            /* scratchpad: no free */
        }
    }

    /* 4. Split the band into MRZ rows: two for a TD3 passport, one for a
     *    CS/CD travel permit (1 x 30).  The two-row splitter is tried
     *    first so TD3 behaviour is untouched. */
    mrz_ocr_rect_t lines[2];
    int split_n = 2;
    /* TD3 (two-row) uses Otsu — unchanged.  CS/CD (one-row) uses Sauvola
     * because Otsu merges text with the guilloche background, producing a
     * band-height line rectangle that classify_layout rejects. */
    if (mrz_ocr_split_lines(band_bin, bw, bh, lines) == 0) {
        split_n = 2;
    } else if (mrz_ocr_split_lines(layout_bin, bw, bh, lines) == 0) {
        split_n = 2;
    } else if (mrz_ocr_split_lines_single(layout_bin, bw, bh, &lines[0]) == 0) {
        split_n = 1;
    } else if (mrz_ocr_split_lines_single(band_bin, bw, bh, &lines[0]) == 0) {
        split_n = 1;
    } else {
        status = MRZ_OCR_ERR_BAD_LINES; goto cleanup;
    }
    /* When split_lines finds two rows but one is tiny (< 8 px), it likely
     * picked up noise or a black border as a "line".  Fall back to the
     * single-line splitter so a CS/CD permit is not mis-split into a
     * 6 px garbage row + the real text row.  Measured on 2026-9-30-J:
     * l0=(0,6) l1=(20,11) — the 6 px row is edge noise. */
    if (split_n == 2) {
        int hmin = lines[0].h < lines[1].h ? lines[0].h : lines[1].h;
        if (hmin < 8) {
            mrz_ocr_rect_t single;
            if (mrz_ocr_split_lines_single(layout_bin, bw, bh, &single) == 0 ||
                mrz_ocr_split_lines_single(band_bin, bw, bh, &single) == 0) {
                lines[0] = single;
                split_n = 1;
            }
        }
    }
    /* Height-ratio correction for TD3 (2-line): when l0 is significantly
     * taller than l1 (h0 > h1 * 4/3), l0 likely includes extra content
     * (page text, guilloche, reflection) above MRZ line 1. Find the
     * transition from extra content (low density) to MRZ text (high
     * density) by locating the maximum darkness gradient, then back
     * off slightly to include character tops. Trim l0 to start from
     * that point.
     * Measured on Photo/3-8: l0=57-63px, l1=37-44px, ratio=1.4-1.9 —
     * the correction trims the extra content and brings h0 ≈ h1. */
    if (split_n == 2 && lines[0].h > lines[1].h * 4 / 3 && lines[1].h > 8) {
        int ly = lines[0].y, lh = lines[0].h;
        int target_h = lines[1].h;
        int *lrow = (int *)calloc((size_t)lh, sizeof(int));
        if (lrow) {
            for (int y = 0; y < lh; ++y) {
                int c = 0;
                for (int x = 0; x < bw; ++x)
                    if (band_bin[(size_t)(ly + y) * bw + x]) c++;
                lrow[y] = c;
            }
            int lpeak = 0;
            for (int y = 0; y < lh; ++y)
                if (lrow[y] > lpeak) lpeak = lrow[y];
            int search_limit = lh - target_h * 7 / 10;
            if (search_limit < 1) search_limit = 1;
            int max_grad_y = -1, max_grad = 0;
            for (int y = 1; y < search_limit; ++y) {
                int grad = lrow[y] - lrow[y - 1];
                if (grad > max_grad) {
                    max_grad = grad;
                    max_grad_y = y;
                }
            }
            if (max_grad_y > 0 && max_grad > lpeak / 10) {
                int backoff = target_h / 5;
                int trim_y = max_grad_y - backoff;
                if (trim_y < 1) trim_y = 1;
                if (trim_y < search_limit) {
                    lines[0].y = ly + trim_y;
                    lines[0].h = lh - trim_y;
                }
            }
            free(lrow);
        }
    }
    /* Refine a single-line rectangle that is too tall (includes background
     * above/below the MRZ text).  Re-scan the Sauvola row projection with
     * a higher threshold (peak/3) to find the densest text sub-region. */
    if (split_n == 1) {
        int expected_h = bw / 29;
        if (lines[0].h > expected_h * 3 / 2) {
            int *rd = (int *)calloc((size_t)bh, sizeof(int));
            if (rd) {
                for (int y = 0; y < bh; ++y)
                    for (int x = 0; x < bw; ++x)
                        if (layout_bin[(size_t)y * bw + x]) rd[y]++;
                int pk = 0;
                for (int y = lines[0].y; y < lines[0].y + lines[0].h && y < bh; ++y)
                    if (rd[y] > pk) pk = rd[y];
                int hi_thr = pk / 3;
                if (hi_thr >= 4) {
                    int best_top = -1, best_bot = -1;
                    long best_ink = -1;
                    int y = lines[0].y;
                    int y_end = lines[0].y + lines[0].h;
                    if (y_end > bh) y_end = bh;
                    while (y < y_end) {
                        if (rd[y] < hi_thr) { y++; continue; }
                        int top = y, last = y;
                        long ink = 0;
                        for (;;) {
                            while (y < y_end && rd[y] >= hi_thr) { ink += rd[y]; last = y; y++; }
                            if (y >= y_end) break;
                            int z = y;
                            while (z < y_end && rd[z] < hi_thr) z++;
                            if (z < y_end && z - last <= 2) {
                                for (int q = y; q < z; ++q) ink += rd[q];
                                y = z;
                            } else break;
                        }
                        if (last - top + 1 >= 2 && ink > best_ink) {
                            best_ink = ink; best_top = top; best_bot = last;
                        }
                    }
                    if (best_top >= 0 && best_bot - best_top + 1 >= expected_h / 2) {
                        lines[0].y = best_top;
                        lines[0].h = best_bot - best_top + 1;
                    }
                }
                free(rd);
            }
        }
    }
    /* MRZ_OCR_TIGHTEN_LINES=1 enables the line-box tightening.
     * The default is ON; set it to 0 to restore the legacy line boxes.
     *
     * On low-resolution upscaled bands split_lines can return boxes 2-3x
     * taller than the glyph rows (measured 1144x122: l0=(y0 h52),
     * l1=(y62 h60) while the ink is only ~22 px per row).  Normalising a
     * 60 px box into 16 CNN rows squashes each glyph into a few faint
     * stripes and both lines decode to garbage.  Tight bands such as
     * 2398x171 l0=(y51 h35) l1=(y130 h38) are left untouched: their box is
     * shorter than one character cell, so the trigger below never fires.
     *
     * Gated on 2026-11-05: 114-photo corpus, TD3 exact unchanged, no photo
     * regressed; on Photo/007 it took line 1 from "PPSTPP<<<" to
     * "POCHNJIC<<LIMIM<<<".
     *
     * HISTORY (why this was off by default, and why it is on again)
     * ----------------------------------------------------------
     * The FIRST implementation took the two globally densest ink runs in
     * the band and cropped to them.  That cost 3 exact photos on the
     * official assets/pic GT set (both 49/59 -> 46/59) and the feature
     * was therefore disabled by default.
     *
     * The SECOND implementation searches the ink run INSIDE each line's
     * own box (never above it) and only fires when all of:
     *   - the box is at least 1.45x taller than its own ink run,
     *   - a COMPETING ink band sits above the run inside the box
     *     ("polluted"), so a merely loose box is not enough,
     *   - the run starts inside the old box,
     *   - the ink does NOT touch the band edge (a run that reaches the
     *     edge means the glyph was already clipped -- tightening it
     *     would stretch the glyph instead of sharpening it; see
     *     assets/pic/c0cbf7c176328f3d_0006, whose line-2 ink runs from
     *     y58 to y68 and touches the band bottom),
     * hold.  The threshold is pk/3: the glyph rows carry 60-90% of the
     * peak ink while a noise band on an upscaled low-resolution crop
     * carries only 15-25% of it, so pk/3 separates them (pk/8 merges
     * them into one run and tightens nothing).
     *
     * MEASURED (2026-11-05, clean production weights):
     *   assets/pic official GT : both 49/59 with TIGHTEN=0 AND =1;
     *                             line-1 char accuracy 0.9715 -> 0.9762
     *   114-photo product bands : TD3 exact 39/108 either way,
     *                             char accuracy 88.45% -> 88.87%
     *   real photo path (50 photos): exact unchanged, line-1 char
     *                             accuracy 98.1% -> 98.2%
     * i.e. strictly neutral on exact and mildly positive on characters,
     * with the Photo/007 repair intact -- so it is ON by default now.
     *   MRZ_OCR_TIGHTEN_LINES=0  restores the legacy line boxes.
     */
    static int tighten = -1;
    if (tighten < 0) {
        const char *e = getenv("MRZ_OCR_TIGHTEN_LINES");
        tighten = (e && e[0] && strcmp(e, "0") == 0) ? 0 : 1;
    }
    if (tighten && split_n == 2) {
        /* The expected row height must come from the ROW's own ink, not
         * from the band width: a low-resolution band is upscaled, so
         * bw/44 overestimates the pitch (measured 598x64 upscaled to
         * 1144x122: bw/44 = 26 while the real pitch is 21, and the
         * resulting threshold never fired).
         *
         * The ink run must also be located INSIDE the row's own box.
         * Picking the two globally densest runs is wrong: on an upscaled
         * low-resolution band the top of the crop carries a noise band
         * (measured 1144x122: rows 0-25 hold 73-104 ink px, rows 26-41
         * hold 266-372) and the global pick can latch onto it, while the
         * real line-1 glyphs (rows 26-41, 16 px) are only 52/16 = 3.3x
         * smaller than the legacy box (y0 h52).  Normalising that box
         * into 16 CNN rows squashes every glyph into a few faint stripes
         * and the whole line decodes to garbage.  So: for each row, look
         * only inside its own (slightly padded) box and take the densest
         * ink run there. */
        int *rd = (int *)calloc((size_t)bh, sizeof(int));
        if (rd) {
            int pk = 0;
            for (int y = 0; y < bh; ++y) {
                int c = 0;
                for (int x = 0; x < bw; ++x)
                    if (band_bin[(size_t)y * bw + x]) c++;
                rd[y] = c;
                if (c > pk) pk = c;
            }
            /* Threshold at a third of the peak: the glyph rows carry
             * 60-90% of the peak ink, while a noise band on an upscaled
             * low-resolution crop carries only 15-25% of it (measured
             * 1144x122: peak 414, glyph rows 266-372, noise rows
             * 73-104).  At pk/8 the noise rows merge with the glyph rows
             * into one huge run and the row box stays as loose as
             * before. */
            int thr = pk / 3;
            if (thr < 2) thr = 2;
            static int tdbg = -1;
            if (tdbg < 0) { const char *e = getenv("MRZ_OCR_TIGHTEN_LINES"); tdbg = (e && e[0] && strcmp(e,"0") != 0) ? 1 : 0; }
            if (tdbg) {
                fprintf(stderr, "TIGHTEN band=%dx%d pk=%d thr=%d rows:",
                        bw, bh, pk, thr);
                for (int q = 0; q < bh; ++q)
                    if (q % 4 == 0) fprintf(stderr, " %d:%d", q, rd[q]);
                fprintf(stderr, "\n");
            }
            for (int li = 0; li < 2; ++li) {
                int old_y = lines[li].y;
                int old_h = lines[li].h;
                /* Search window: the row's own box.  The window is NOT
                 * extended upwards: the row above ends right where this
                 * row starts, and a run that starts above the box belongs
                 * to the neighbour (measured on assets/pic
                 * c0cbf7c176328f3d_0006: line2 box (y56 h13) but the ink
                 * run starts at y59 while rows 56-58 carry the tail of
                 * line 1 -- extending the window upwards pulled the
                 * line-2 box up to y56 and dropped the '<' filler of the
                 * personal-number field).  The window may extend DOWN by
                 * up to 25% of the box, because the splitter sometimes
                 * cuts a descender short. */
                int sy = lines[li].y;
                int ey = lines[li].y + lines[li].h + lines[li].h / 4;
                if (sy < 0) sy = 0;
                if (ey > bh - 1) ey = bh - 1;
                /* Densest ink run inside the window. */
                int best_t = -1, best_b = -1;
                long best_ink = 0;
                int y = sy;
                while (y <= ey) {
                    if (rd[y] < thr) { y++; continue; }
                    int t = y, last = y;
                    long ink = 0;
                    while (y <= ey && rd[y] >= thr) { ink += rd[y]; last = y; y++; }
                    /* Runs shorter than 3 px are noise, not glyph rows. */
                    if (last - t + 1 >= 3 && ink > best_ink) {
                        best_ink = ink; best_t = t; best_b = last;
                    }
                }
                if (best_t < 0) continue;
                int ink_h = best_b - best_t + 1;
                /* Replace the legacy box only when it is CLEARLY too
                 * loose: the legacy height must be at least 1.45x the
                 * measured ink height.  A mildly loose box is left
                 * alone -- cropping it risks cutting strokes, which
                 * is what cost exact photos on the official assets set.
                 * The ink run must also start at or below the legacy box
                 * top: a run that begins above it belongs to the row
                 * above, and cropping to it would steal the other row's
                 * tail. */
                int too_loose = (old_h * 100 >= ink_h * 145);
                int starts_inside = (best_t >= old_y);
                /* Distinguish a POLLUTED box from merely a loose one.
                 * A polluted box contains a COMPETING ink band above
                 * the glyph row: on an upscaled low-resolution crop the
                 * band carries a noise strip (measured 1144x122: rows
                 * 0-25 hold 73-104 ink px, the real line-1 glyphs hold
                 * 266-372), and the legacy box swallows both, so every
                 * glyph is normalised together with the noise.
                 * A box that is loose but empty (rows 1-2 ink px only,
                 * measured on assets/pic c0cbf7c176328f3d_0006) must be
                 * left alone: cropping it changes the vertical scale
                 * the CNN sees and cost that photo its exact match.
                 *
                 * The empty-box case is recognised by the BLANK
                 * fraction: rows whose ink is below a tenth of the
                 * threshold count as blank.  A polluted box has a
                 * sizeable blank stretch between the noise strip and
                 * the glyph row (measured 007: 26 blank rows out of
                 * 52); a merely loose box has few or none (measured
                 * assets: 26 of 41 blank, but the frame's own
                 * geometry -- the box starts where the previous row
                 * ends -- marks it as a SPLITTER artefact, not a
                 * pollution artefact, so it is left alone). */
                int max_above = 0;
                for (int q = old_y; q < best_t && q < bh; ++q)
                    if (rd[q] > max_above) max_above = rd[q];
                int polluted = (max_above * 3 >= thr);
                /* A polluted box must ALSO have a real blank stretch
                 * above the glyph row -- that is what makes the excess
                 * height harmful (the CNN then normalises the glyph
                 * into a fraction of the box). */
                int blank_rows = 0;
                int blank_thr = thr / 4;
                if (blank_thr < 1) blank_thr = 1;
                for (int q = old_y; q < best_t && q < bh; ++q)
                    if (rd[q] < blank_thr) blank_rows++;
                int has_blank = (blank_rows * 100 >= (best_t - old_y) * 45);
                /* Do NOT tighten a row whose ink run touches the band
                 * edge.  At the edge the glyph is CUT OFF by the
                 * crop, so the visible ink is only the top half of
                 * the character; normalising that half into 16 CNN
                 * rows stretches it vertically and destroys the
                 * reading.  Measured on assets/pic
                 * c0cbf7c176328f3d_0006: the band is 1144x69, line 2's
                 * ink runs from y58 to the last row (y68), and
                 * tightening that row to (y57 h12) turned the '<<<<'
                 * filler of the personal-number field into '<<<<<<<U<0<0'
                 * and cost the photo its exact match.  A row whose ink
                 * is fully inside the band (Photo/007: line 1 at
                 * y26..41, line 2 at y63..81, band 122 rows) can be
                 * tightened safely. */
                int touches_edge = (best_t <= 0 || best_b >= bh - 1);
                int pad = ink_h / 8;
                if (pad < 1) pad = 1;
                int nt = best_t - pad; if (nt < old_y) nt = old_y;
                int nb = best_b + pad; if (nb >= bh) nb = bh - 1;
                int nh = nb - nt + 1;
                int apply = (too_loose && polluted && starts_inside &&
                             !touches_edge && nh >= 6 && nh < old_h);
                if (apply) {
                    lines[li].y = nt;
                    lines[li].h = nh;
                }
                if (tdbg)
                    fprintf(stderr, "TIGHTEN row%d box=(y%d h%d) ink=(y%d h%d) "
                            "above=%d poll=%d blank=%d edge=%d -> (y%d h%d)%s\n",
                            li, old_y, old_h, best_t, ink_h, max_above, polluted,
                            has_blank, touches_edge, nt, nh,
                            apply ? " TIGHTENED" : "");
            }
            free(rd);
        }
    }
    if (diag_on) {
        g_diag.line[0].y = lines[0].y; g_diag.line[0].h = lines[0].h;
        if (split_n > 1) {
            g_diag.line[1].y = lines[1].y; g_diag.line[1].h = lines[1].h;
        }
    }
    {   /* MRZ_OCR_LINES=1: report the band and the line boxes, plus their
         * height ratio -- the quantity that decides whether the splitter
         * found two MRZ rows or a decoration row plus one. */
        static int ldbg = -1;
        if (ldbg < 0) { const char *e = getenv("MRZ_OCR_LINES"); ldbg = (e && e[0] && strcmp(e, "0") != 0) ? 1 : 0; }
        if (ldbg) {
            if (split_n > 1)
                fprintf(stderr, "LINES band=%dx%d n=%d l0=(y%d h%d) l1=(y%d h%d) ratio=%.2f\n",
                        bw, bh, split_n, lines[0].y, lines[0].h, lines[1].y, lines[1].h,
                        lines[1].h > 0 ? (double)lines[0].h / (double)lines[1].h : 0.0);
            else
                fprintf(stderr, "LINES band=%dx%d n=%d l0=(y%d h%d)\n",
                        bw, bh, split_n, lines[0].y, lines[0].h);
        }
    }

    /* 4c. Document layout classification: TD3 (2x44) vs CS/CD (1x30).
     * Uses the bottom row's binary pixels + dual-hypothesis grid valley
     * energy, and the pairing test to reject a Chinese caption line
     * above a permit's single MRZ row. */
    int n_lines = 2, n_cells = 44;
    int line_map[2] = { 0, 1 };
    {
        int bi = split_n - 1;                       /* bottom row */
        const uint8_t *bot = layout_bin + (size_t)lines[bi].y * bw;
        g_doc_layout = mrz_classify_layout(bot, bw, lines[bi].h, bw,
                                           lines, split_n);
        if (g_doc_layout == MRZ_LAYOUT_SINGLE_LINE_30) {
            n_lines = 1;
            n_cells = 30;
            line_map[0] = bi;
        } else if (split_n == 1) {
            /* One 44-cell row with no partner: a TD3 passport whose line 1
             * was lost to glare, crop or tilt.  Decoding it as a two-line
             * record would read the second half of the same physical row,
             * so drop the frame instead. */
            status = MRZ_OCR_ERR_BAD_LINES; goto cleanup;
        }
    }
    /* For CS/CD (SINGLE_LINE_30), use Otsu (band_bin) for glyph extraction
     * because Sauvola with low k produces almost-all-black glyphs on the
     * guilloche background.  Otsu's global threshold separates text from
     * the pattern better at the per-glyph level.  An env override keeps
     * Sauvola available for experimentation. */
    const uint8_t *glyph_bin = band_bin;
    {
        const char *ge = getenv("MRZ_OCR_GLYPH_BIN");
        if (ge && ge[0] == 's' && g_doc_layout == MRZ_LAYOUT_SINGLE_LINE_30) {
            /* Lazily run the glyph Sauvola here (lower k keeps more of the
             * character strokes for the CS/CD guilloche background), with
             * the same window rule as the layout pass above. */
            int gsw = bw / 40;
            if (gsw < 7) gsw = 7;
            if (gsw > 31) gsw = 31;
            float gk = 0.1f;
            const char *gke = getenv("MRZ_OCR_GLYPH_SAUVOLA_K");
            if (gke && gke[0]) { float v = (float)atof(gke); if (v > 0.0f && v < 1.0f) gk = v; }
            mrz_sauvola(band_gray, bw, bh, glyph_sauvola, gsw, gk);
            glyph_bin = glyph_sauvola;
        }
    }

    glyphs     = (float (*)[CNN_IN_H][CNN_IN_W])mrz_scratch_glyphs(44 * (CNN_IN_H * CNN_IN_W));
    glyphs_tta = (float (*)[CNN_IN_H][CNN_IN_W])mrz_scratch_glyphs(44 * (CNN_IN_H * CNN_IN_W));
    feats      = mrz_scratch_feats(44 * CNN_FLAT);
    probs      = mrz_scratch_probs(44 * CNN_OUT);
    probs_tta  = mrz_scratch_probs(44 * CNN_OUT);
    if (!glyphs || !glyphs_tta || !feats || !probs || !probs_tta) {
        status = MRZ_OCR_ERR_LOAD; goto cleanup;
    }
    /* Keep the synthetic model as the host/default control group. Android
     * enables the real-photo line-specific pair through the environment. */
    pthread_once(&g_mrz_scratch_once, mrz_scratch_init_once);
    const char *line_models_env = getenv("MRZ_OCR_LINE_MODELS");
    const int use_line_models =
        line_models_env && line_models_env[0] && strcmp(line_models_env, "0") != 0;
    net0 = use_line_models ? &g_mrz_scratch_cnn_line1 : &g_mrz_scratch_cnn_template;
    net1 = use_line_models ? &g_mrz_scratch_cnn_line2 : &g_mrz_scratch_cnn_template;
    {
        const char *tta_env = getenv("MRZ_OCR_TTA");
        int tta = (tta_env && tta_env[0] && strcmp(tta_env, "0") != 0);
        /* CRNN sequence head: 44 x CNN_FLAT -> GRU -> 44 x CNN_OUT.
         * Uses the identical glyph/feature pipeline as the CNN path, so
         * A/B differences are attributable to the sequence model only. */
        const char *crnn_env = getenv("MRZ_OCR_CRNN");
        const int use_crnn =
            crnn_env && crnn_env[0] && strcmp(crnn_env, "0") != 0 &&
            crnn_weights_available();

        /* Fused mode: run the per-character CNN head AND the CRNN
         * sequence head, then combine them in log space.
         *
         * WHY
         * ---
         * Measured on a strictly held-out split (train on pic_scan1,
         * evaluate on pic_scan2/3, identical features for both heads):
         *
         *   line 1   CNN 94.62%   CRNN 92.94%   fused 96.05%
         *   line 2   CNN 97.73%   CRNN 82.66%   fused 99.04%
         *
         * The two heads fail on DIFFERENT characters (line 2: the CNN
         * was wrong on 140 cells where the CRNN was right, and the CRNN
         * was wrong on 14 cells where the CNN was right), so combining
         * them in log space recovers most of the difference.
         *
         * The weights are log-probabilities, so a cell that ONE head is
         * confident about outvotes a cell where the other head is not.
         * MRZ_OCR_FUSE_ALPHA scales the CRNN term; the shipped default
         * 0.2 was the best value on the 50-photo passport split (PASS 35
         * vs 34 at 0.5); lower alpha keeps CRNN as a light correction.
         */
        const char *fuse_env = getenv("MRZ_OCR_FUSE");
        const int use_fuse =
            fuse_env && fuse_env[0] && strcmp(fuse_env, "0") != 0 &&
            crnn_weights_available();
        float fuse_alpha = 0.2f;
        {
            const char *a = getenv("MRZ_OCR_FUSE_ALPHA");
            if (a && a[0]) {
                float v = (float)atof(a);
                if (v >= 0.0f && v <= 8.0f) fuse_alpha = v;
            }
        }
        /* Absolute CRNN-confidence gate: below this the CRNN head does
         * not get a vote.  The default is 0 (no absolute gate): the
         * 3-fold grouped cross-validation over the 41 real photos
         * measured the fused engine at 22/41 exact with this setting
         * versus 21/41 for the plain CNN, and any absolute gate above
         * zero only removes the CRNN's votes without changing the
         * winner.  MRZ_OCR_FUSE_GATE is kept for field A/B. */
        float fuse_gate = 0.0f;
        {
            const char *g = getenv("MRZ_OCR_FUSE_GATE");
            if (g && g[0]) {
                float v = (float)atof(g);
                if (v >= 0.0f && v <= 1.0f) fuse_gate = v;
            }
        }
        /* Relative gate: the CRNN confidence must be at least this
         * multiple of the CNN confidence on the same cell.  1.0 means
         * "the CRNN must be at least as sure as the CNN". */
        float fuse_ratio = 1.0f;
        {
            const char *r = getenv("MRZ_OCR_FUSE_RATIO");
            if (r && r[0]) {
                float v = (float)atof(r);
                if (v >= 0.0f && v <= 8.0f) fuse_ratio = v;
            }
        }
        /* Weight copy (~200 KB per head) only when the sequence head is
         * actually in play -- CRNN-off frames must not pay it. */
        if (use_crnn || use_fuse)
            crnn_pair_init(&g_mrz_scratch_crnn);

        /* Fusion mode.  "arb" (default): decode on pure CNN probs, the
         * CRNN only arbitrates between the beam winner and its runner-up
         * on the differing columns (fuse can no longer overwrite the
         * checksum-beam repairs -- the measured 10-25 regression).
         * "probs": legacy in-place log-space fusion, kept for A/B. */
        int fuse_arb = 1;
        {
            const char *m = getenv("MRZ_OCR_FUSE_MODE");
            if (m && m[0] && strcmp(m, "probs") == 0) fuse_arb = 0;
        }
        /* Relaxation window for the runner-up walk when arbitration is
         * active (MRZ_OCR_FUSE_ARB_GAP).  0.5 = the ambiguity margin
         * itself; widening it lets the CRNN consider runner-ups further
         * behind the winner, at extra beam cost. */
        float fuse_arb_gap = 0.5f;
        {
            const char *g = getenv("MRZ_OCR_FUSE_ARB_GAP");
            if (g && g[0]) {
                float v = (float)atof(g);
                if (v >= 0.0f && v <= 8.0f) fuse_arb_gap = v;
            }
        }
        /* Total log-prob by which the CRNN must favour the runner-up over
         * the winner, summed over the differing columns
         * (MRZ_OCR_FUSE_ARB_MARGIN).  Ties stay with the CNN. */
        float fuse_arb_margin = 0.5f;
        {
            const char *g = getenv("MRZ_OCR_FUSE_ARB_MARGIN");
            if (g && g[0]) {
                float v = (float)atof(g);
                if (v >= 0.0f && v <= 8.0f) fuse_arb_margin = v;
            }
        }
        float fuse_arb_conf = 0.7f;
        {
            const char *g = getenv("MRZ_OCR_FUSE_ARB_CONF");
            if (g && g[0]) {
                float v = (float)atof(g);
                if (v >= 0.0f && v <= 1.0f) fuse_arb_conf = v;
            }
        }

        /* Step 2 fork:
         *   default (sequential)  -> single loop, cache-hot
         *   MRZ_OCR_PAR=1         -> serialise resample, then
         *                            fork a worker for line 1's
         *                            CNN forward while the main
         *                            thread runs line 0.
         * The sequential path keeps the original cache-hot
         * pattern (resample -> cnn -> decode back-to-back) so
         * the L1 cache stays warm on the just-written glyph
         * buffer; splitting resample and cnn into separate
         * passes was measurably slower on A53 (~+5 ms p50). */
        const char *par_env = getenv("MRZ_OCR_PAR");
        int do_par = par_env && par_env[0] && strcmp(par_env, "0") != 0;
        if (g_doc_layout == MRZ_LAYOUT_SINGLE_LINE_30) do_par = 0;
        /* The dual-line worker runs the per-character FC inside the worker
         * thread, which has no CRNN branch. The sequence head is a strictly
         * serial per-row recurrence, so the parallel split buys nothing for
         * it: force the cache-hot sequential path whenever CRNN is on so the
         * GRU is actually used instead of silently falling back to CNN. */
        /* The dual-line worker now runs the sequence head and the
         * fusion too (see mrz_row_classify), so the parallel split is
         * available for every engine.  The sequence recurrence is per
         * row, so line 1 and line 2 remain independent work and the
         * split still pays off: measured on the eval corpus, the
         * sequential path costs ~0.8 ms more p50 than the parallel
         * one. */
        if (use_crnn && !use_fuse && do_par) {
            /* The plain CRNN mode keeps the sequential path: the fused
             * engine is the one that ships, and the plain CRNN mode is
             * kept as a diagnostic A/B. */
        }
        if (do_par) {
            /* Worker-only pinning (CPU 3) proved best: pinning the
             * main thread to CPU 0 hurt p50 (~+11 ms) because it
             * also froze the serial geometric front-end onto one
             * core under system load.  Leave main unpinned. */
            /* Workspace for line 0/1 (chars from segmenter, line
             * pixels, base offset, line height). */
            static mrz_ocr_rect_t chars0[44], chars1[44];
            mrz_ocr_rect_t *chars_l[2]    = { chars0, chars1 };
            int nchars_l[2]                = { 0, 0 };
            int line_h_l[2]                = { 0, 0 };
            int base_l[2]                  = { 0, 0 };
            uint8_t *line_pixels_l[2]     = { NULL, NULL };
            /* Per-line glyph buffers live in disjoint scratchpad
             * halves.  Save the addresses in static globals so
             * the worker thread + main thread can both see them. */
            static float (*glyphs0_keep)[CNN_IN_H][CNN_IN_W];
            static float (*glyphs1_keep)[CNN_IN_H][CNN_IN_W];

            /* Pass 1: serialise the resample step into the
             * per-half glyphs_keep buffers so the second pass
             * below doesn't redo the work. */
            for (int li = 0; li < 2; ++li) {
                if (mrz_ocr_deadline_expired()) {
                    status = MRZ_OCR_ERR_TIMEOUT;
                    goto cleanup;
                }
                mrz_scratch_select_half(li);
                base_l[li]   = (li == 0) ? lines[0].y : lines[1].y;
                line_h_l[li] = (li == 0) ? lines[0].h : lines[1].h;
                if (line_h_l[li] <= 0) continue;
                line_pixels_l[li] = mrz_scratch_line((size_t)bw * line_h_l[li]);
                if (!line_pixels_l[li]) { status = MRZ_OCR_ERR_LOAD; goto cleanup; }
                for (int y = 0; y < line_h_l[li]; ++y)
                    memcpy(line_pixels_l[li] + y * bw,
                           glyph_bin + (base_l[li] + y) * bw, bw);

                mrz_diag_line_t *dg_line = mrz_diag_line(li);
                int nchars = segment_line_grid(line_pixels_l[li], bw,
                                              line_h_l[li], chars_l[li], 44, li);
                if (nchars <= 0)
                    nchars = segment_line_cnn(line_pixels_l[li], bw,
                                              line_h_l[li], chars_l[li], 44);
                if (dg_line && nchars >= 0) dg_line->nchars = nchars;
                if (nchars < 30) {
                    status = MRZ_OCR_ERR_BAD_LINES; goto cleanup;
                }
                if (nchars > 44) nchars = 44;
                nchars_l[li] = nchars;
                float (*glyphs_l)[CNN_IN_H][CNN_IN_W] =
                    (float (*)[CNN_IN_H][CNN_IN_W])mrz_scratch_glyphs(44 * (CNN_IN_H * CNN_IN_W));
                if (li == 0) glyphs0_keep = glyphs_l;
                else          glyphs1_keep = glyphs_l;
                for (int c = 0; c < nchars; ++c)
                    resample_char_gray(band_gray, bw, bh,
                                       chars_l[li][c].x,
                                       base_l[li] + chars_l[li][c].y,
                                       chars_l[li][c].w, chars_l[li][c].h,
                                       NULL, 0, glyphs_l[c]);
            }
            if (timing_on) {
                clock_gettime(CLOCK_MONOTONIC, &_t1);
                _ms_resample += (_t1.tv_sec-_t0.tv_sec)*1e3 + (_t1.tv_nsec-_t0.tv_nsec)/1e6;
                _t0 = _t1;
            }

            /* Allocate the per-line feat/prob buffers.  Glyphs
             * are already populated above; we still need
             * glyphs_tta for TTA on each line. */
            mrz_scratch_select_half(0);
            float (*glyphs0_tta)[CNN_IN_H][CNN_IN_W] =
                (float (*)[CNN_IN_H][CNN_IN_W])mrz_scratch_glyphs(44 * (CNN_IN_H * CNN_IN_W));
            float *feats0     = mrz_scratch_feats(44 * CNN_FLAT);
            float *probs0     = mrz_scratch_probs(44 * CNN_OUT);
            float *probs0_tta = mrz_scratch_probs(44 * CNN_OUT);
            mrz_scratch_select_half(1);
            float (*glyphs1_tta)[CNN_IN_H][CNN_IN_W] =
                (float (*)[CNN_IN_H][CNN_IN_W])mrz_scratch_glyphs(44 * (CNN_IN_H * CNN_IN_W));
            float *feats1     = mrz_scratch_feats(44 * CNN_FLAT);
            float *probs1     = mrz_scratch_probs(44 * CNN_OUT);
            float *probs1_tta = mrz_scratch_probs(44 * CNN_OUT);
            if (!glyphs0_tta || !feats0 || !probs0 || !probs0_tta ||
                !glyphs1_tta || !feats1 || !probs1 || !probs1_tta) {
                status = MRZ_OCR_ERR_LOAD; goto cleanup;
            }

            /* Pass 1 above already populated glyphs0_keep and
             * glyphs1_keep, so we skip the original wasteful
             * re-resample pass.  The line-1 worker now also runs
             * TTA + decode_row in-thread so the join is
             * essentially free on the main side. */
#ifndef __linux__
            pthread_t worker;
#endif
            mrz_line1_job_t job;
            int worker_spawned = 0;
            int line1_did_inline = 0;
            /* Ambiguity of line 2, decoded on the worker thread. */
            mrz_decode_amb_t worker_amb;
            memset(&worker_amb, 0, sizeof(worker_amb));
            if (nchars_l[1] >= 30) {
                job.net       = net1;
                job.glyphs    = glyphs1_keep;
                job.tta_glyphs= glyphs1_tta;
                job.feats     = feats1;
                job.probs     = probs1;
                job.tta_probs = probs1_tta;
                job.nchars    = nchars_l[1];
                job.band_gray = band_gray;
                job.bw        = bw;
                job.bh        = bh;
                job.chars     = chars_l[1];
                job.base_y    = base_l[1];
                job.tta       = tta;
                job.line_out  = out->line2;
                job.amb       = &worker_amb;
                job.conf_avg  = &out->line2_avg_conf;
                job.use_seq   = use_crnn;
                job.use_fuse  = use_fuse;
                job.fuse_arb  = fuse_arb;
                job.arb_gap   = fuse_arb_gap;
                job.arb_margin = fuse_arb_margin;
                job.arb_conf  = fuse_arb_conf;
                job.fuse_alpha = fuse_alpha;
                job.fuse_gate  = fuse_gate;
                job.fuse_ratio = fuse_ratio;
                job.line_idx   = 1;
                job.seq_probs  = (float *)mrz_scratch_probs(44 * CNN_OUT);
                /* Hand the main thread's remaining budget to the worker
                 * (the deadline is TLS; the worker's own is empty). */
                job.deadline_enabled = 0;
                mrz_ocr_deadline_get(&job.deadline, &job.deadline_enabled);
#ifdef __linux__
                if (mrz_resident_submit(&job) == 0) {
                    worker_spawned = 1;   /* resident worker owns it */
                } else {
                    line1_did_inline = 1;
                    mrz_line1_run(&job);  /* in-line fallback */
                }
#else
                if (pthread_create(&worker, NULL, mrz_line1_worker, &job) == 0) {
                    worker_spawned = 1;
                } else {
                    line1_did_inline = 1;
                    mrz_line1_worker(&job);
                }
#endif
            }

            /* Line 0 forward (and TTA) on the main thread. */
            if (nchars_l[0] >= 30) {
                float *seq0 = NULL;
                {
                    mrz_line1_job_t j0;
                    memset(&j0, 0, sizeof(j0));
                    j0.net = net0; j0.glyphs = glyphs0_keep; j0.feats = feats0;
                    j0.probs = probs0; j0.nchars = nchars_l[0];
                    j0.use_seq = use_crnn; j0.use_fuse = use_fuse;
                    j0.fuse_arb = fuse_arb;
                    j0.arb_gap = fuse_arb_gap; j0.arb_margin = fuse_arb_margin;
                    j0.arb_conf = fuse_arb_conf;
                    j0.fuse_alpha = fuse_alpha; j0.fuse_gate = fuse_gate;
                    j0.fuse_ratio = fuse_ratio; j0.line_idx = 0;
                    j0.seq_probs = (float *)mrz_scratch_probs(44 * CNN_OUT);
                    mrz_row_classify(&j0);
                    /* Line 0 has no checksum beam, so arbitration never
                     * fires there; kept uniform with the worker path. */
                    seq0 = (use_fuse && fuse_arb) ? j0.seq_probs : NULL;
                }
                if (tta) {
                    for (int c = 0; c < nchars_l[0]; ++c)
                        for (int o = 0; o < CNN_OUT; ++o)
                            probs0_tta[c * CNN_OUT + o] = probs0[c * CNN_OUT + o];
                    for (int sh = -1; sh <= 1; sh += 2) {
                        for (int c = 0; c < nchars_l[0]; ++c)
                            resample_char_gray(band_gray, bw, bh,
                                               chars_l[0][c].x + sh,
                                               base_l[0] + chars_l[0][c].y,
                                               chars_l[0][c].w, chars_l[0][c].h,
                                               NULL, 0, glyphs0_tta[c]);
                        cnn_row_features_batch(net0, glyphs0_tta, nchars_l[0], feats0);
                        cnn_fc_batch(net0, feats0, nchars_l[0], probs0);
                        for (int c = 0; c < nchars_l[0]; ++c)
                            for (int o = 0; o < CNN_OUT; ++o)
                                probs0_tta[c * CNN_OUT + o] += probs0[c * CNN_OUT + o];
                    }
                    for (int c = 0; c < nchars_l[0]; ++c)
                        for (int o = 0; o < CNN_OUT; ++o)
                            probs0[c * CNN_OUT + o] = probs0_tta[c * CNN_OUT + o] / 3.0f;
                }
                decode_row((const float (*)[CNN_IN_H][CNN_IN_W])glyphs0_keep,
                           nchars_l[0], 0, out->line1, &out->line1_avg_conf, probs0,
                           NULL, seq0, fuse_arb_gap, fuse_arb_margin, fuse_arb_conf);
                out->line1_len = nchars_l[0];
                mrz_ocr_dump_glyphs("img", 0, glyphs0_keep, nchars_l[0], chars_l[0]);
            }
            if (timing_on) {
                clock_gettime(CLOCK_MONOTONIC, &_t1);
                _ms_decode += (_t1.tv_sec-_t0.tv_sec)*1e3 + (_t1.tv_nsec-_t0.tv_nsec)/1e6;
                _t0 = _t1;
            }

            /* Wait for line 1.  On Linux the resident worker is
             * reused (no per-image join); elsewhere and on the
             * in-line fallback the work is already done above. */
            if (worker_spawned) {
#ifdef __linux__
                mrz_resident_wait();
#else
                pthread_join(worker, NULL);
#endif
            }
            if (nchars_l[1] >= 30) {
                out->line2_len = nchars_l[1];
                mrz_ocr_dump_glyphs("img", 1, glyphs1_keep, nchars_l[1], chars_l[1]);
                /* Line 2 was decoded on the worker thread; carry its
                 * ambiguity report back to the caller. */
                out->line2_ambiguous = worker_amb.ambiguous;
                out->line2_alt_margin = worker_amb.margin;
                memcpy(out->line2_alt, worker_amb.alt, 45);
                out->line2_alt[44] = 0;
            }
            if (timing_on) {
                clock_gettime(CLOCK_MONOTONIC, &_t1);
                _ms_decode += (_t1.tv_sec-_t0.tv_sec)*1e3 + (_t1.tv_nsec-_t0.tv_nsec)/1e6;
                _t0 = _t1;
            }
            (void)line1_did_inline; /* silence unused-var */
        } else {
            /* Sequential: cache-hot single loop.  Resample,
             * cnn forward, TTA, decode back-to-back per line so
             * the just-written glyph buffer is still in L1 when
             * cnn_row_features_batch reads it. */
            for (int li = 0; li < n_lines; ++li) {
                int pli = line_map[li];
                int base = lines[pli].y;
                int line_h = lines[pli].h;
                int line_x = lines[pli].x;
                int line_w = lines[pli].w;
                if (line_w <= 0 || line_x + line_w > bw) { line_x = 0; line_w = bw; }
                if (line_h <= 0) continue;
                mrz_scratch_select_half(li);
                line_pixels = mrz_scratch_line((size_t)line_w * line_h);
                if (!line_pixels) { status = MRZ_OCR_ERR_LOAD; goto cleanup; }
                for (int y = 0; y < line_h; ++y)
                    memcpy(line_pixels + y * line_w,
                           glyph_bin + (base + y) * bw + line_x, line_w);

                mrz_ocr_rect_t chars[44];
                int nchars = segment_line_grid(line_pixels, line_w, line_h, chars, n_cells, li);
                if (nchars <= 0)
                    nchars = segment_line_cnn(line_pixels, line_w, line_h, chars, n_cells);
                if (diag_on) {
                    mrz_diag_line_t *dg_line = mrz_diag_line(li);
                    if (dg_line && nchars >= 0) dg_line->nchars = nchars;
                }
                if (nchars < 30) {
                    line_pixels = NULL;
                    status = MRZ_OCR_ERR_BAD_LINES; goto cleanup;
                }
                if (nchars > n_cells) nchars = n_cells;
                for (int c = 0; c < nchars; ++c)
                    resample_char_gray(band_gray, bw, bh,
                                       line_x + chars[c].x, base + chars[c].y,
                                       chars[c].w, chars[c].h,
                                       NULL, 0, glyphs[c]);
                if (timing_on) {
                    clock_gettime(CLOCK_MONOTONIC, &_t1);
                    _ms_resample += (_t1.tv_sec-_t0.tv_sec)*1e3 + (_t1.tv_nsec-_t0.tv_nsec)/1e6;
                    _t0 = _t1;
                }

                const cnn_t *line_net = (li == 0) ? net0 : net1;
                cnn_row_features_batch(line_net, glyphs, nchars, feats);
                mrz_ocr_dump_line_features("img", li, feats, nchars);
                float *seq_arb = NULL;
                if (use_fuse) {
                    cnn_fc_batch(line_net, feats, nchars, probs);
                    float *seq = (float *)mrz_scratch_probs(44 * CNN_OUT);
                    if (!seq) { status = MRZ_OCR_ERR_LOAD; goto cleanup; }
                    crnn_forward_line(&g_mrz_scratch_crnn, li, feats, nchars, seq);
                    if (fuse_arb) {
                        /* Arbitration mode: keep the CNN probs pure for
                         * the decode; the sequence verdict is handed to
                         * decode_row for the runner-up tie-break. */
                        seq_arb = seq;
                    } else {
                        mrz_fuse_probs(probs, seq, nchars, fuse_alpha, fuse_gate, fuse_ratio);
                    }
                } else if (use_crnn) {
                    /* Sequence head replaces the per-character FC with a
                     * GRU over the row, keeping the same 37-class output
                     * contract so the downstream ICAO/checksum logic is
                     * unchanged. One head per MRZ line because the CNN
                     * front-end already uses line-specific conv weights. */
                    crnn_forward_line(&g_mrz_scratch_crnn, li, feats, nchars, probs);
                } else {
                    cnn_fc_batch(line_net, feats, nchars, probs);
                }
                if (mrz_ocr_deadline_expired()) {
                    status = MRZ_OCR_ERR_TIMEOUT;
                    goto cleanup;
                }
                if (tta) {
                    for (int c = 0; c < nchars; ++c)
                        for (int o = 0; o < CNN_OUT; ++o) probs_tta[c * CNN_OUT + o] = probs[c * CNN_OUT + o];
                    for (int sh = -1; sh <= 1; sh += 2) {
                        for (int c = 0; c < nchars; ++c)
                            resample_char_gray(band_gray, bw, bh,
                                               chars[c].x + sh, base + chars[c].y,
                                               chars[c].w, chars[c].h,
                                               NULL, 0, glyphs_tta[c]);
                        cnn_row_features_batch(line_net, glyphs_tta, nchars, feats);
                        if (use_fuse) {
                            cnn_fc_batch(line_net, feats, nchars, probs);
                            if (!fuse_arb) {
                                /* legacy probs mode: fuse per shift */
                                float *seq = (float *)mrz_scratch_probs(44 * CNN_OUT);
                                if (!seq) { status = MRZ_OCR_ERR_LOAD; goto cleanup; }
                                crnn_forward_line(&g_mrz_scratch_crnn, li, feats, nchars, seq);
                                mrz_fuse_probs(probs, seq, nchars, fuse_alpha, fuse_gate, fuse_ratio);
                            }
                            /* arb mode: TTA-averages the CNN probs only;
                             * seq_arb keeps the BASE CRNN verdict (same
                             * policy as the worker TTA path). */
                        } else if (use_crnn) crnn_forward_line(&g_mrz_scratch_crnn, li, feats, nchars, probs);
                        else cnn_fc_batch(line_net, feats, nchars, probs);
                        for (int c = 0; c < nchars; ++c)
                            for (int o = 0; o < CNN_OUT; ++o)
                                probs_tta[c * CNN_OUT + o] += probs[c * CNN_OUT + o];
                    }
                    for (int c = 0; c < nchars; ++c)
                        for (int o = 0; o < CNN_OUT; ++o)
                            probs[c * CNN_OUT + o] = probs_tta[c * CNN_OUT + o] / 3.0f;
                }

                char *dest = (li == 0) ? out->line1 : out->line2;
                int *conf = (li == 0) ? &out->line1_avg_conf : &out->line2_avg_conf;
                mrz_decode_amb_t amb;
                decode_row((const float (*)[CNN_IN_H][CNN_IN_W])glyphs, nchars, li,
                           dest, conf, probs, &amb,
                           seq_arb, fuse_arb_gap, fuse_arb_margin, fuse_arb_conf);
                if (li == 0) {
                    out->line1_len = nchars;
                } else {
                    out->line2_len = nchars;
                    out->line2_ambiguous = amb.ambiguous;
                    out->line2_alt_margin = amb.margin;
                    memcpy(out->line2_alt, amb.alt, 45);
                    out->line2_alt[44] = 0;
                }
                mrz_ocr_dump_glyphs("img", li, glyphs, nchars, chars);
                line_pixels = NULL;
                if (timing_on) {
                    clock_gettime(CLOCK_MONOTONIC, &_t1);
                    _ms_decode += (_t1.tv_sec-_t0.tv_sec)*1e3 + (_t1.tv_nsec-_t0.tv_nsec)/1e6;
                    _t0 = _t1;
                }
            }
        }
    }

cleanup:
    /* Structural veto on the line-2 reading.  Computed here, after both
     * the sequential and the parallel path have written out->line2, so
     * the two paths cannot disagree about it. */
    out->doc_layout = g_doc_layout;
    out->line2_struct_ok = mrz_line2_struct_sane(out->line2, out->line2_len);
    out->line2_nat_suspect = mrz_nat_suspect(out->line2, out->line2_len);
    out->line1_country_mismatch = mrz_line1_country_mismatch(
        out->line1, out->line1_len, out->line2, out->line2_len);
    out->line1_issuer_suspect = mrz_issuer_suspect(out->line1, out->line1_len);
    if (diag_on) {
        g_diag.status = (int)status;
        g_diag.ms_gray = _ms_gray;
        g_diag.ms_band = _ms_band;
        g_diag.ms_resample = _ms_resample;
        g_diag.ms_decode = _ms_decode;
        mrz_diag_write(out);
    }
    /* scratchpad: no per-call frees.  The whole buffer is reset
     * on the next recogniser entry (mrz_scratch_reset()). */
    (void)probs_tta; (void)probs; (void)feats;
    (void)glyphs_tta; (void)glyphs; (void)net0; (void)net1;
    (void)line_pixels;
    (void)rot; (void)band_bin; (void)band_gray; (void)layout_bin; (void)glyph_sauvola;
    (void)bin; (void)gray;
    if (tm && tm[0] && strcmp(tm, "0")) {
        fprintf(stderr, "TIMING gray=%.2fms band+deskew=%.2fms resample=%.2fms decode=%.2fms\n",
                _ms_gray, _ms_band, _ms_resample, _ms_decode);
#ifdef __ANDROID__
        /* Mirror to logcat under the same tag the JNI shim uses
         * ("MrzNative") so `adb logcat -s MrzNative:TIMING` shows
         * the per-stage breakdown while the bench is running. */
        extern int __android_log_print(int, const char *, const char *, ...);
        __android_log_print(6 /*ANDROID_LOG_ERROR*/, "MrzNative",
            "TIMING gray=%.2fms band+deskew=%.2fms resample=%.2fms decode=%.2fms",
            _ms_gray, _ms_band, _ms_resample, _ms_decode);
#endif
    }
    return status;
}
