#ifndef MRZ_H
#define MRZ_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ICAO 9303 check-digit weight table */
#define MRZ_W0 7
#define MRZ_W1 3
#define MRZ_W2 1

/* TD3 (passport) fixed dimensions */
#define MRZ_TD3_LINE_LEN    44
#define MRZ_TD3_LINE_COUNT  2
#define MRZ_TD3_TOTAL       (MRZ_TD3_LINE_LEN * MRZ_TD3_LINE_COUNT)  /* 88 */

/* TD2 (ID card) fixed dimensions */
#define MRZ_TD2_LINE_LEN    36
#define MRZ_TD2_LINE_COUNT  2
#define MRZ_TD2_TOTAL       (MRZ_TD2_LINE_LEN * MRZ_TD2_LINE_COUNT)  /* 72 */

/* Status codes */
typedef enum {
    MRZ_OK = 0,
    MRZ_ERR_INVALID_CHAR = -1,
    MRZ_ERR_BAD_LENGTH   = -2,
    MRZ_ERR_BAD_CHECK    = -3,
    MRZ_ERR_BAD_FIELD    = -4,
    MRZ_ERR_BUFFER_SMALL = -5
} mrz_status_t;

/* Decoded TD3 fields */
typedef struct {
    char doc_type[3];          /* "P<" typically; 2 chars + NUL */
    char issuing_state[4];     /* 3 chars + NUL */
    char surname[40];
    char given_names[40];
    char passport_no[10];      /* 9 + NUL */
    char nationality[4];       /* 3 + NUL */
    char birth_date_yymmdd[7]; /* 6 + NUL */
    char sex[2];
    char expiry_date_yymmdd[7];
    char personal_no[15];      /* 14 + NUL */
    /* raw per-segment check results */
    int  ck_passport_no;
    int  ck_birth_date;
    int  ck_expiry_date;
    int  ck_personal_no;
    int  ck_composite;
} mrz_td3_t;

/* Compute single check digit for a substring. weight cycle: 7,3,1. */
int mrz_check_digit(const char *s, size_t n);

/* Compute composite check digit over line 2's first 43 chars (line2 must point at line 2 of a TD3 record). */
int mrz_td3_composite_check(const char *line2);

/* Encode TD3 fields into a 2-line buffer.
 * Returns MRZ_OK on success, fills lines (must hold MRZ_TD3_TOTAL+1 bytes)
 * with two trailing '\n' and a terminating NUL.
 */
mrz_status_t mrz_td3_encode(const mrz_td3_t *f, char *out_lines);

/* Decode and verify a TD3 record (must be exactly MRZ_TD3_TOTAL chars).
 * Lines may be concatenated with or without '\n' between them.
 */
mrz_status_t mrz_td3_decode(const char *lines, mrz_td3_t *out);

/* Helper: returns human-readable error string. */
const char *mrz_strerror(mrz_status_t s);

#ifdef __cplusplus
}
#endif
#endif /* MRZ_H */
