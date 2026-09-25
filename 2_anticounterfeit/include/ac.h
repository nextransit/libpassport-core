#ifndef AC_H
#define AC_H

#include <stddef.h>
#include "mrz.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    AC_OK = 0,
    AC_ERR_LENGTH         = -1,
    AC_ERR_DOC_TYPE       = -2,
    AC_ERR_ISSUING_STATE  = -3,
    AC_ERR_NATIONALITY    = -4,
    AC_ERR_SEX            = -5,
    AC_ERR_BIRTH_DATE     = -6,
    AC_ERR_EXPIRY_DATE    = -7,
    AC_ERR_PADDING_ANOMALY= -8,
    AC_ERR_NAME_ENTROPY   = -9,
    AC_ERR_PERSONAL_NO    = -10,
    AC_ERR_CHECK_DIGITS   = -11,
    AC_ERR_DATE_ORDER     = -12,
    AC_ERR_DATE_RANGE     = -13
} ac_status_t;

typedef enum {
    AC_SEVERITY_OK         = 0,
    AC_SEVERITY_INFO       = 1,
    AC_SEVERITY_WARN       = 2,
    AC_SEVERITY_FAIL       = 3,
} ac_severity_t;

/* Per-check diagnostic record. */
typedef struct {
    const char   *name;
    ac_severity_t severity;
    const char   *detail;
} ac_finding_t;

typedef struct {
    ac_finding_t *items;
    size_t        count;
    size_t        cap;
    int           warnings;
    int           fails;
    int           infos;
} ac_report_t;

void ac_report_init(ac_report_t *r);
void ac_report_free(ac_report_t *r);
void ac_report_add(ac_report_t *r, const char *name,
                   ac_severity_t sev, const char *detail);

/* Anchor "today" for date validation. Tests pass an explicit value. */
typedef struct {
    int year, month, day;
} ac_date_t;

/* Master verification entry. Returns AC_OK if every check has
 * severity <= WARN. The report is always populated with findings.
 *
 *  'mrz_lines' is the encoded TD3 string (88 chars or 88+\n chars).
 *  'today' is used for relative date validation; pass NULL to use 2000-01-01
 *  (so tests are deterministic). */
ac_status_t ac_verify_td3(const char *mrz_lines,
                          const ac_date_t *today,
                          ac_report_t *report);

const char *ac_strerror(ac_status_t s);

#ifdef __cplusplus
}
#endif
#endif /* AC_H */
