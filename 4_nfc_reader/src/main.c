/* nfc_tool -- drive the BAC eMRTD read sequence against a mock backend
 * loaded from a JSON script. Prints the result on stdout in a
 * human-readable form.
 *
 * Usage:
 *   nfc_tool <mock-script.json>
 *
 * Exit codes:
 *   0  success (BAC complete)
 *   1  protocol error
 *   2  script parse error / file missing
 */
#include "nfc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int hex_to_buf(const char *hex, uint8_t *out, size_t cap) {
    size_t n = strlen(hex);
    if (n % 2 != 0) return -1;
    size_t bytes = n / 2;
    if (bytes > cap) return -1;
    for (size_t i = 0; i < bytes; ++i) {
        char hi = hex[i*2], lo = hex[i*2+1];
        int h = (hi >= '0' && hi <= '9') ? hi - '0' :
                (hi >= 'a' && hi <= 'f') ? 10 + hi - 'a' :
                (hi >= 'A' && hi <= 'F') ? 10 + hi - 'A' : -1;
        int l = (lo >= '0' && lo <= '9') ? lo - '0' :
                (lo >= 'a' && lo <= 'f') ? 10 + lo - 'a' :
                (lo >= 'A' && lo <= 'F') ? 10 + lo - 'A' : -1;
        if (h < 0 || l < 0) return -1;
        out[i] = (uint8_t)((h << 4) | l);
    }
    return (int)bytes;
}

int main(int argc, char **argv) {
    setvbuf(stderr, NULL, _IONBF, 0);
    if (argc != 2) {
        fprintf(stderr, "usage: %s <mock-script.json>\n", argv[0]);
        return 2;
    }
    nfc_backend_t *be = nfc_backend_mock_from_json(argv[1]);
    if (!be) {
        fprintf(stderr, "failed to load script %s\n", argv[1]);
        return 2;
    }
    nfc_result_t r = nfc_read_emrtd(be);
    printf("result.ok       : %d\n", r.ok);
    printf("result.step     : %s\n", nfc_step_name(r.step));
    printf("result.detail   : %s\n", r.detail);
    printf("result.dg1_mrz  : %s\n", r.dg1_mrz);
    printf("result.sod_head : ");
    for (int i = 0; i < r.sod_head_len; ++i) printf("%02X", r.sod_head[i]);
    printf("\n");
    printf("mock.unexpected : %d\n", nfc_mock_unexpected_count(be));
    printf("mock.remaining  : %d\n", nfc_mock_remaining(be));
    be->vt->destroy(be);
    return r.ok ? 0 : 1;
}
