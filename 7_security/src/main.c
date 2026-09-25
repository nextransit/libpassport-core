/* security_tool -- run the ICAO security pipeline.
 *
 * Subcommands:
 *   bac <passport_no> <birth_yymmdd> <expiry_yymmdd>
 *   mac <hex_key16> <hex_data>
 *   crosscheck <mrz_file> <viz_text>
 *   ir <visible.ppm> <ir.ppm>
 *   ssim <a.ppm> <b.ppm>
 *   uv <uv.ppm>
 */
#include "security.h"
#include "face.h"

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

static int do_bac(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: bac <passport_no> <birth_yymmdd> <expiry_yymmdd>\n");
        return 2;
    }
    bac_keys_t k = bac_derive_keys(argv[1], argv[2], argv[3]);
    printf("K_enc = ");
    for (int i = 0; i < 16; ++i) printf("%02X", k.k_enc[i]);
    printf("\nK_mac = ");
    for (int i = 0; i < 16; ++i) printf("%02X", k.k_mac[i]);
    printf("\n");
    return 0;
}

static int do_mac(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: mac <hex_key16> <hex_data>\n");
        return 2;
    }
    uint8_t key[16];
    if (hex_to_buf(argv[1], key, 16) != 16) {
        fprintf(stderr, "bad key\n"); return 2;
    }
    uint8_t data[2048];
    int n = hex_to_buf(argv[2], data, sizeof(data));
    if (n < 0) { fprintf(stderr, "bad data\n"); return 2; }
    uint8_t mac[8];
    mac3_des(key, data, (size_t)n, mac);
    printf("mac = ");
    for (int i = 0; i < 8; ++i) printf("%02X", mac[i]);
    printf("\n");
    return 0;
}

static int do_crosscheck(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: crosscheck <mrz_file> <viz_text>\n");
        return 2;
    }
    FILE *f = fopen(argv[1], "r");
    if (!f) { perror("fopen"); return 2; }
    char mrz[256];
    size_t n = fread(mrz, 1, sizeof(mrz) - 1, f);
    fclose(f);
    mrz[n] = '\0';
    /* Strip trailing newlines. */
    while (n > 0 && (mrz[n-1] == '\n' || mrz[n-1] == '\r' || mrz[n-1] == ' '))
        mrz[--n] = '\0';
    cross_modal_report_t r = security_cross_check(mrz, argv[2]);
    printf("cross_check: %d ok, %d fail\n", r.ok_count, r.fail_count);
    printf("  passport_no : %s\n",
        r.passport_no == AC_FIELD_OK ? "OK" :
        r.passport_no == AC_FIELD_MISMATCH ? "MISMATCH" : "MISSING");
    printf("  birth_date  : %s\n",
        r.birth_date == AC_FIELD_OK ? "OK" :
        r.birth_date == AC_FIELD_MISMATCH ? "MISMATCH" : "MISSING");
    printf("  expiry_date : %s\n",
        r.expiry_date == AC_FIELD_OK ? "OK" :
        r.expiry_date == AC_FIELD_MISMATCH ? "MISMATCH" : "MISSING");
    printf("  nationality : %s\n",
        r.nationality == AC_FIELD_OK ? "OK" :
        r.nationality == AC_FIELD_MISMATCH ? "MISMATCH" : "MISSING");
    printf("  name        : %s\n",
        r.name == AC_FIELD_OK ? "OK" :
        r.name == AC_FIELD_MISMATCH ? "MISMATCH" : "MISSING");
    return r.fail_count == 0 ? 0 : 1;
}

static int do_ir(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: ir <visible.ppm> <ir.ppm>\n");
        return 2;
    }
    face_image_t *v = face_image_load_ppm(argv[1]);
    face_image_t *i = face_image_load_ppm(argv[2]);
    if (!v || !i) { fprintf(stderr, "load failed\n"); return 2; }
    ir_absorption_t r = security_ir_absorption(v, i);
    printf("visible_dark=%d ir_dark=%d ratio=%.2f passes=%d\n",
        r.visible_dark_count, r.ir_dark_count, r.ratio, r.passes);
    face_image_free(v); face_image_free(i);
    return r.passes ? 0 : 1;
}

static int do_ssim(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: ssim <a.ppm> <b.ppm>\n");
        return 2;
    }
    face_image_t *a = face_image_load_ppm(argv[1]);
    face_image_t *b = face_image_load_ppm(argv[2]);
    if (!a || !b) { fprintf(stderr, "load failed\n"); return 2; }
    ssim_match_t r = security_ssim(a, b);
    printf("ssim=%.4f passes=%d\n", r.ssim, r.passes);
    face_image_free(a); face_image_free(b);
    return r.passes ? 0 : 1;
}

static int do_uv(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: uv <uv.ppm>\n");
        return 2;
    }
    face_image_t *u = face_image_load_ppm(argv[1]);
    if (!u) { fprintf(stderr, "load failed\n"); return 2; }
    uv_check_t r = security_uv_luminance(u);
    printf("uv_mean=%.1f passes=%d\n", r.mean_luminance, r.passes);
    face_image_free(u);
    return r.passes ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
            "usage: %s <subcmd> [args...]\n"
            "  bac <passport_no> <birth_yymmdd> <expiry_yymmdd>\n"
            "  mac <hex_key16> <hex_data>\n"
            "  crosscheck <mrz_file> <viz_text>\n"
            "  ir <visible.ppm> <ir.ppm>\n"
            "  ssim <a.ppm> <b.ppm>\n"
            "  uv <uv.ppm>\n", argv[0]);
        return 2;
    }
    if (strcmp(argv[1], "bac") == 0)         return do_bac(argc - 1, argv + 1);
    if (strcmp(argv[1], "mac") == 0)         return do_mac(argc - 1, argv + 1);
    if (strcmp(argv[1], "crosscheck") == 0)  return do_crosscheck(argc - 1, argv + 1);
    if (strcmp(argv[1], "ir") == 0)          return do_ir(argc - 1, argv + 1);
    if (strcmp(argv[1], "ssim") == 0)        return do_ssim(argc - 1, argv + 1);
    if (strcmp(argv[1], "uv") == 0)          return do_uv(argc - 1, argv + 1);
    fprintf(stderr, "unknown subcmd %s\n", argv[1]);
    return 2;
}
