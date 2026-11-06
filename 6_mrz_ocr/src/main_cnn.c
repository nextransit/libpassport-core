/* mrz_ocr_cnn_tool -- like mrz_ocr_tool but uses the CNN recogniser. */
#include "mrz_ocr.h"
#include "face.h"
#include "crnn.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The recognition head the tool ACTUALLY used.
 *
 * MRZ_OCR_CRNN / MRZ_OCR_FUSE select a head, but a weight set that
 * ships no CRNN head (the assets/pic variant) makes both of them fall
 * back to the plain CNN head -- inside the recogniser, silently.  A
 * caller that asked for "CRNN" and got the CNN head has been given a
 * different algorithm than the one it named, so the head that ran has
 * to be reported, not assumed.
 *
 * "cnn(no-crnn)" is the substitution marker: the request named a CRNN
 * head and the plain CNN head ran instead. */
static const char *effective_head(void) {
    const char *fuse_env = getenv("MRZ_OCR_FUSE");
    const char *crnn_env = getenv("MRZ_OCR_CRNN");
    int want_fuse = fuse_env && fuse_env[0] && strcmp(fuse_env, "0") != 0;
    int want_crnn = crnn_env && crnn_env[0] && strcmp(crnn_env, "0") != 0;
    if (want_fuse)
        return crnn_weights_available() ? "fuse" : "cnn(no-crnn)";
    if (want_crnn)
        return crnn_weights_available() ? "crnn" : "cnn(no-crnn)";
    return "cnn";
}

/* The head the caller demanded, or NULL when it did not demand one.
 *
 * MRZ_OCR_REQUIRE_HEAD=cnn|crnn|fuse turns "the head I asked for is not
 * available" into a hard failure instead of a substituted result: the
 * GUI names an algorithm, and naming it must not be silently ignored. */
static const char *required_head(void) {
    const char *e = getenv("MRZ_OCR_REQUIRE_HEAD");
    if (!e || !e[0] || strcmp(e, "0") == 0) return NULL;
    return e;
}

/* Print one recognition result in the shared KV contract. */
static int run_one_cnn(const char *path) {
    face_image_t *img = face_image_load_ppm(path);
    if (!img) img = face_image_load_bmp(path);
    if (!img) { fprintf(stderr, "cannot load %s\n", path); return 2; }
    const char *head = effective_head();
    const char *want = required_head();
    if (want && strcmp(want, head) != 0) {
        printf("method            : %s\n", head);
        printf("result.ok         : HEAD_UNAVAILABLE\n");
        printf("result.head       : %s\n", head);
        printf("result.required   : %s\n", want);
        printf("result.note       : requested head is not available in this "
               "weight set\n");
        printf("result.done     : 1\n");
        face_image_free(img);
        return 3;
    }
    mrz_ocr_result_t r;
    mrz_ocr_status_t s = mrz_ocr_recognise_cnn(img, &r);
    printf("method            : cnn\n");
    printf("result.head       : %s\n", head);
    printf("result.ok         : %s\n", mrz_ocr_strerror(s));
    printf("result.line1      : %s\n", r.line1);
    printf("result.line2      : %s\n", r.line2);
    printf("result.conf1      : %d\n", r.line1_avg_conf);
    printf("result.conf2      : %d\n", r.line2_avg_conf);
    /* Ambiguity of the line-2 reading: a second checksum-valid reading
     * within MRZ_AMBIGUITY_MARGIN means the classifier and the checksum
     * could not tell the two apart.  A wrong-but-checksum-valid line is
     * indistinguishable from a correct one downstream, so the caller
     * needs this to decide whether to trust the result. */
    printf("result.ambiguous2 : %d\n", r.line2_ambiguous);
    printf("result.altmargin2 : %d\n", r.line2_alt_margin);
    if (r.line2_alt[0]) printf("result.alt2       : %s\n", r.line2_alt);
    /* Arithmetic-independent veto on line 2: 1 means every field is at
     * least structurally possible (real calendar dates, expiry not
     * before birth).  0 means the five check digits were satisfied by
     * something that is not a TD3 record. */
    printf("result.struct2    : %d\n", r.line2_struct_ok);
    printf("result.natsuspect : %d\n", r.line2_nat_suspect);
    printf("result.countrymm  : %d\n", r.line1_country_mismatch);
    printf("result.issuersusp : %d\n", r.line1_issuer_suspect);
    printf("band.x band.y band.w band.h : %d %d %d %d\n",
           r.band_x, r.band_y, r.band_w, r.band_h);
    printf("result.line1y line1h line2y line2h : %d %d %d %d\n",
           r.line1_y, r.line1_h, r.line2_y, r.line2_h);
    /* Document layout: TD3 (2x44 passport) vs CS/CD (1x30 travel permit)
     * vs TD1 (3x30 back).  Callers need this to know how many characters
     * each line is supposed to carry -- a 30-character reading compared
     * against a 44-character GT is a guaranteed mismatch, not an error. */
    printf("result.layout     : %d\n", (int)r.doc_layout);
    printf("result.layoutname : %s\n",
           r.doc_layout == MRZ_LAYOUT_TD3_DOUBLE_44 ? "TD3_44" :
           r.doc_layout == MRZ_LAYOUT_SINGLE_LINE_30 ? "CS_CD_30" :
           r.doc_layout == MRZ_LAYOUT_TD1_TRIPLE_30 ? "TD1_30" : "UNKNOWN");
    printf("result.done     : 1\n");
    face_image_free(img);
    return s == MRZ_OCR_OK ? 0 : 1;
}

/* Resident mode: read one image path per line from stdin, print one
 * result block per line.  Keeps the model loaded so the caller pays
 * the recognition cost only, not a process start per image. */
static int serve(void) {
    char line[4096];
    mrz_ocr_init();
    while (fgets(line, sizeof(line), stdin)) {
        size_t n = strlen(line);
        while (n && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = 0;
        if (!n) continue;
        run_one_cnn(line);
        fflush(stdout);
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--serve") == 0) return serve();
    if (argc != 2) {
        fprintf(stderr, "usage: %s <image> | --serve\n", argv[0]);
        return 2;
    }
    /* Optional frame budget, mirroring what the Android JNI passes
     * (recogniseBitmap(bitmap, deadlineMs)).  MRZ_OCR_DEADLINE_MS=0
     * disables it.  This makes the deadline behaviour testable from the
     * command line -- the device measurements need it because the
     * product path's budget is spent partly in Kotlin. */
    {
        const char *e = getenv("MRZ_OCR_DEADLINE_MS");
        if (e && e[0]) mrz_ocr_deadline_set_ms(atoi(e));
    }
    mrz_ocr_init();
    return run_one_cnn(argv[1]);
}
