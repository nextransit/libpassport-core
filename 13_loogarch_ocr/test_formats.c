/*
 * test_formats.c - Verify all pixel format paths work end-to-end
 *                 (both color and Y-only/gray modes)
 */
#include "loogarch_ocr.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int run_format(loogarch_pixel_format_t fmt, const char* name,
                      int w, int h, int buf_size, int gray)
{
    uint8_t* buf = (uint8_t*)malloc(buf_size);
    if (!buf) { printf("FAIL %s: OOM\n", name); return 1; }
    memset(buf, 0x80, buf_size);

    void* handle = loogarch_ocr_init(fmt, w, h, gray ? "gray=1" : NULL);
    if (!handle) { printf("FAIL %s: init NULL\n", name); free(buf); return 1; }

    loogarch_ocr_result_t r = loogarch_ocr_process(handle, buf, 0, 0, 0, 0);

    int failed = 0;
    if (r.text == NULL) {
        printf("FAIL %s: text NULL\n", name);
        failed = 1;
    } else if (strstr(r.text, "Unsupported") != NULL) {
        printf("FAIL %s: %s\n", name, r.text);
        failed = 1;
    } else if (r.width != w || r.height != h || r.format != (int)fmt) {
        printf("FAIL %s: meta w=%d h=%d fmt=%d (exp %dx%d fmt=%d)\n",
               name, r.width, r.height, r.format, w, h, (int)fmt);
        failed = 1;
    } else {
        printf("PASS %s: %s (conf=%d)\n", name, r.text, r.confidence);
    }

    free(r.text);
    loogarch_ocr_cleanup(handle);
    free(buf);
    return failed;
}

int main(void)
{
    int fails = 0;
    const int W = 8, H = 8;

    printf("libloogarch_ocr version: %s\n", loogarch_ocr_get_version());

    struct { loogarch_pixel_format_t fmt; const char* name; int buf; } cases[] = {
        {LOOGARCH_PIXEL_FORMAT_NV21,     "NV21",     W*H*3/2},
        {LOOGARCH_PIXEL_FORMAT_NV12,     "NV12",     W*H*3/2},
        {LOOGARCH_PIXEL_FORMAT_I420,     "I420",     W*H*3/2},
        {LOOGARCH_PIXEL_FORMAT_I422,     "I422",     W*H*2},
        {LOOGARCH_PIXEL_FORMAT_I444,     "I444",     W*H*3},
        {LOOGARCH_PIXEL_FORMAT_YUYV,     "YUYV",     W*H*2},
        {LOOGARCH_PIXEL_FORMAT_ARGB8888, "ARGB8888", W*H*4},
        {LOOGARCH_PIXEL_FORMAT_RGB888,   "RGB888",   W*H*3},
        {LOOGARCH_PIXEL_FORMAT_RGB565,   "RGB565",   W*H*2},
        {LOOGARCH_PIXEL_FORMAT_JPEG,     "JPEG",     1024},
        {LOOGARCH_PIXEL_FORMAT_MJPEG,    "MJPEG",    1024},
        {LOOGARCH_PIXEL_FORMAT_PNG,      "PNG",      1024},
        /* Single-plane luma: the format the loongarch v4l2 camera
         * delivers (CAMERA_PIXEL_FORMAT_GRAY).  One byte per pixel. */
        {LOOGARCH_PIXEL_FORMAT_GRAY,     "GRAY",     W*H},
    };
    const int N = sizeof(cases) / sizeof(cases[0]);

    printf("\n=== COLOR path (3-channel BGR) ===\n");
    for (int i = 0; i < N; i++) {
        fails += run_format(cases[i].fmt, cases[i].name, W, H, cases[i].buf, 0);
    }

    printf("\n=== Y-ONLY path (single-channel luma, gray=1) ===\n");
    for (int i = 0; i < N; i++) {
        fails += run_format(cases[i].fmt, cases[i].name, W, H, cases[i].buf, 1);
    }

    printf("\n=== %d format-path(s) failed, %d passed ===\n",
           fails, N * 2 - fails);
    return fails ? 1 : 0;
}
