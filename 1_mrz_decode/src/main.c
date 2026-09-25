/* mrz_tool -- tiny CLI for the MRZ library.
 * Usage:
 *   mrz_tool encode  <doc_type> <issuing_state> <surname> <given_names>
 *                   <passport_no> <nationality> <yymmdd_birth>
 *                   <sex> <yymmdd_expiry> <personal_no>
 *   mrz_tool decode  "<two 44-char lines>" */
#include "mrz.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int do_encode(int argc, char **argv) {
    if (argc != 11) {
        fprintf(stderr, "encode needs 10 fields\n");
        return 2;
    }
    mrz_td3_t f;
    memset(&f, 0, sizeof(f));
    snprintf(f.doc_type,        sizeof(f.doc_type),        "%s", argv[1]);
    snprintf(f.issuing_state,   sizeof(f.issuing_state),   "%s", argv[2]);
    snprintf(f.surname,         sizeof(f.surname),         "%s", argv[3]);
    snprintf(f.given_names,     sizeof(f.given_names),     "%s", argv[4]);
    snprintf(f.passport_no,     sizeof(f.passport_no),     "%s", argv[5]);
    snprintf(f.nationality,     sizeof(f.nationality),     "%s", argv[6]);
    snprintf(f.birth_date_yymmdd,  sizeof(f.birth_date_yymmdd),  "%s", argv[7]);
    snprintf(f.sex,             sizeof(f.sex),             "%s", argv[8]);
    snprintf(f.expiry_date_yymmdd, sizeof(f.expiry_date_yymmdd), "%s", argv[9]);
    snprintf(f.personal_no,     sizeof(f.personal_no),     "%s", argv[10]);

    char out[MRZ_TD3_TOTAL + 3];
    mrz_status_t s = mrz_td3_encode(&f, out);
    if (s != MRZ_OK) {
        fprintf(stderr, "encode failed: %s\n", mrz_strerror(s));
        return 1;
    }
    fputs(out, stdout);
    return 0;
}

static int do_decode(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "decode needs 1 argument (the MRZ string)\n");
        return 2;
    }
    mrz_td3_t f;
    mrz_status_t s = mrz_td3_decode(argv[1], &f);
    if (s != MRZ_OK) {
        fprintf(stderr, "decode failed: %s\n", mrz_strerror(s));
        return 1;
    }
    printf("doc_type       : '%s'\n",  f.doc_type);
    printf("issuing_state  : '%s'\n",  f.issuing_state);
    printf("surname        : '%s'\n",  f.surname);
    printf("given_names    : '%s'\n",  f.given_names);
    printf("passport_no    : '%s'  ck=%d\n", f.passport_no,  f.ck_passport_no);
    printf("nationality    : '%s'\n",  f.nationality);
    printf("birth_yymmdd   : '%s'  ck=%d\n", f.birth_date_yymmdd, f.ck_birth_date);
    printf("sex            : '%s'\n",  f.sex);
    printf("expiry_yymmdd  : '%s'  ck=%d\n", f.expiry_date_yymmdd, f.ck_expiry_date);
    printf("personal_no    : '%s'  ck=%d\n", f.personal_no, f.ck_personal_no);
    printf("composite_ck   : %d\n",     f.ck_composite);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
            "usage:\n"
            "  %s encode <doc_type> <issuing_state> <surname> <given_names>"
            " <passport_no> <nationality> <yymmdd_birth> <sex> <yymmdd_expiry>"
            " <personal_no>\n"
            "  %s decode \"<two 44-char lines>\"\n",
            argv[0], argv[0]);
        return 2;
    }
    if (strcmp(argv[1], "encode") == 0) return do_encode(argc - 1, argv + 1);
    if (strcmp(argv[1], "decode") == 0) return do_decode(argc - 1, argv + 1);
    fprintf(stderr, "unknown subcommand: %s\n", argv[1]);
    return 2;
}
