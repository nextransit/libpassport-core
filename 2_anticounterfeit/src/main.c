/* ac_tool -- run anti-counterfeit verification on a TD3 MRZ string.
 * Usage:
 *   ac_tool <path-to-mrz-file>
 *   ac_tool -        (read from stdin)
 * The mrz file should contain two 44-character lines, optionally
 * separated by a newline.
 * Exit codes:
 *   0  verification passed (no FAIL severity)
 *   1  verification produced WARN/INFO findings only
 *   2  verification FAILED (one or more FAIL findings)
 *   3  bad input / internal error
 */
#include "ac.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_all(FILE *f, char *buf, size_t cap) {
    size_t n = fread(buf, 1, cap - 1, f);
    buf[n] = '\0';
    /* strip trailing whitespace */
    while (n > 0 && (buf[n-1] == '\n' || buf[n-1] == '\r' || buf[n-1] == ' '))
        buf[--n] = '\0';
    return (int)n;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <mrz-file> | -\n", argv[0]);
        return 3;
    }
    char buf[256];
    if (strcmp(argv[1], "-") == 0) {
        if (read_all(stdin, buf, sizeof(buf)) <= 0) return 3;
    } else {
        FILE *f = fopen(argv[1], "r");
        if (!f) { perror("fopen"); return 3; }
        if (read_all(f, buf, sizeof(buf)) <= 0) { fclose(f); return 3; }
        fclose(f);
    }

    /* Anchor "today" as 2026-09-25 (matches the current date in the
     * runtime context). The user can edit AC_TODAY in ac.c to override. */
    ac_date_t today = {2026, 9, 25};

    ac_report_t rep;
    ac_report_init(&rep);

    ac_status_t s = ac_verify_td3(buf, &today, &rep);

    printf("verification: %s\n", ac_strerror(s));
    for (size_t i = 0; i < rep.count; ++i) {
        const char *sev = "??";
        switch (rep.items[i].severity) {
            case AC_SEVERITY_OK:   sev = "OK  "; break;
            case AC_SEVERITY_INFO: sev = "INFO"; break;
            case AC_SEVERITY_WARN: sev = "WARN"; break;
            case AC_SEVERITY_FAIL: sev = "FAIL"; break;
        }
        printf("  [%s] %-16s %s\n", sev,
               rep.items[i].name, rep.items[i].detail);
    }

    int rc;
    if (rep.fails > 0) rc = 2;
    else if (rep.warnings > 0) rc = 1;
    else rc = 0;
    ac_report_free(&rep);
    return rc;
}
