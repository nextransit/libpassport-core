/* nfc_tool -- drive the BAC eMRTD read sequence against a mock backend
 * loaded from a JSON script. Prints the result on stdout in a
 * human-readable form.
 *
 * Usage:
 *   nfc_tool <mock-script.json> [mrz_line1 mrz_line2]
 *
 * The MRZ (both 44-char lines) feeds the BAC key derivation. When
 * omitted, the MRZ embedded in the script's top-level "mrz" field is
 * used. Exit codes:
 *   0  success (BAC complete)
 *   1  protocol error
 *   2  script parse error / file missing
 */
#include "nfc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    setvbuf(stderr, NULL, _IONBF, 0);
    if (argc != 2 && argc != 4) {
        fprintf(stderr, "usage: %s <mock-script.json> [mrz_line1 mrz_line2]\n",
                argv[0]);
        return 2;
    }
    nfc_backend_t *be = nfc_backend_mock_from_json(argv[1]);
    if (!be) {
        fprintf(stderr, "failed to load script %s\n", argv[1]);
        return 2;
    }

    const char *l1 = NULL, *l2 = NULL;
    if (argc == 4) {
        l1 = argv[2];
        l2 = argv[3];
    } else {
        if (nfc_mock_get_mrz(be, &l1, &l2) != 0) {
            fprintf(stderr, "no MRZ in script; pass one on the command line\n");
            be->vt->destroy(be);
            return 2;
        }
    }

    nfc_result_t r = nfc_read_emrtd(be, l1, l2);
    printf("result.ok        : %d\n", r.ok);
    printf("result.step      : %s\n", nfc_step_name(r.step));
    printf("result.detail    : %s\n", r.detail);
    if (r.crypto_done) {
        printf("result.mrz_info  : %s\n", r.mrz_info);
        printf("result.kseed     : %s\n", r.kseed_hex);
        printf("result.kenc      : %s\n", r.kenc_hex);
        printf("result.kmac      : %s\n", r.kmac_hex);
        printf("result.ksenc     : %s\n", r.ksenc_hex);
        printf("result.ksmac     : %s\n", r.ksmac_hex);
        printf("result.rnd_icc   : %s\n", r.rnd_icc_hex);
        printf("result.auth1     : %s\n", r.auth1_hex);
        printf("result.auth2     : %s\n", r.auth2_hex);
    }
    printf("result.dg1_mrz   : %s\n", r.dg1_mrz);
    printf("result.sod_head  : ");
    for (int i = 0; i < r.sod_head_len; ++i) printf("%02X", r.sod_head[i]);
    printf("\n");
    printf("mock.unexpected  : %d\n", nfc_mock_unexpected_count(be));
    printf("mock.remaining   : %d\n", nfc_mock_remaining(be));

    /* Emit the per-APDU trace as a single JSON line for the GUI. */
    printf("trace.json       : ");
    printf("{\"steps\":[");
    for (int i = 0; i < r.trace_count; ++i) {
        nfc_trace_t *t = &r.trace[i];
        if (i) printf(",");
        printf("{\"name\":\"%s\",\"capdu\":\"%s\",\"rapdu\":\"%s\","
               "\"sw\":\"%04X\",\"ms\":%u,\"plain\":\"%s\",\"mac_ok\":%d,"
               "\"err\":%d}",
               t->name, t->capdu, t->rapdu, t->sw, t->ms, t->plain,
               t->mac_ok, t->err);
    }
    printf("]}\n");

    be->vt->destroy(be);
    return r.ok ? 0 : 1;
}