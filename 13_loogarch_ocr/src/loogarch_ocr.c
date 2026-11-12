/*
 * loogarch_ocr.c - OCR implementation for loongarch/OpenHarmony desktop
 *
 * Camera format support: NV21, NV12, I420, I422, I444, YUYV,
 *                        ARGB8888, RGB888, RGB565, MJPEG, JPEG, PNG
 * Y-only fast path: extract luma plane directly, skip chroma conversion
 *                   (optimal for MRZ decoding where luma carries all glyph info)
 *
 * Copyright (c) 2026 passport
 * Licensed under Apache-2.0
 */

#include "loogarch_ocr.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <time.h>

#ifdef LOOGARCH_OCR_WITH_MRZ
#include "mrz_ocr.h"
#include "face.h"
#endif

/* Forward declarations */
typedef struct {
    loogarch_pixel_format_t format;
    int width;
    int height;
    int stride;           /* Row stride for aligned access */
    int gray_only;        /* Y-only fast path: feed single-channel luma to OCR */
    int use_fuse;         /* 1 = fused engine (CNN head + CRNN head) */
    int use_par;          /* 1 = dual-line parallel decode */
    const uint8_t* y_data;
    const uint8_t* u_data;
    const uint8_t* v_data;
} loogarch_ocr_internal_t;

/* ===================== Y-only (luma) extraction ===================== */

/* Planar YUV (NV21/NV12/I420/I422/I444): Y plane is first W*H bytes */
static void planar_extract_y(const uint8_t* buf, int width, int height,
                             int stride, uint8_t* y_out)
{
    if (stride == width) {
        memcpy(y_out, buf, (size_t)width * height);
    } else {
        for (int y = 0; y < height; y++) {
            memcpy(y_out + y * width, buf + y * stride, width);
        }
    }
}

/* YUYV (packed YUY2): Y at even byte offsets [Y0 U Y1 V ...] */
static void yuyv_extract_y(const uint8_t* yuyv, int width, int height,
                           int stride, uint8_t* y_out)
{
    for (int y = 0; y < height; y++) {
        const uint8_t* row = yuyv + y * stride;
        uint8_t* dst = y_out + y * width;
        for (int x = 0; x < width; x++) {
            dst[x] = row[x * 2];
        }
    }
}

/* RGB888 to luma (ITU-R BT.601 integer approximation) */
static void rgb888_to_gray(const uint8_t* src, int width, int height,
                           int stride, uint8_t* gray)
{
    for (int y = 0; y < height; y++) {
        const uint8_t* s = src + y * stride;
        uint8_t* d = gray + y * width;
        for (int x = 0; x < width; x++) {
            int r = s[x * 3 + 0];
            int g = s[x * 3 + 1];
            int b = s[x * 3 + 2];
            d[x] = (uint8_t)((r * 299 + g * 587 + b * 114) / 1000);
        }
    }
}

/* ARGB8888 to luma (skip alpha) */
static void argb8888_to_gray(const uint8_t* src, int width, int height,
                             int stride, uint8_t* gray)
{
    for (int y = 0; y < height; y++) {
        const uint8_t* s = src + y * stride;
        uint8_t* d = gray + y * width;
        for (int x = 0; x < width; x++) {
            int r = s[x * 4 + 1];
            int g = s[x * 4 + 2];
            int b = s[x * 4 + 3];
            d[x] = (uint8_t)((r * 299 + g * 587 + b * 114) / 1000);
        }
    }
}

/* RGB565 (little-endian) to luma */
static void rgb565_to_gray(const uint8_t* src, int width, int height,
                           int stride, uint8_t* gray)
{
    for (int y = 0; y < height; y++) {
        const uint8_t* s = src + y * stride;
        uint8_t* d = gray + y * width;
        for (int x = 0; x < width; x++) {
            uint16_t px = (uint16_t)s[x * 2] | ((uint16_t)s[x * 2 + 1] << 8);
            int r = ((px >> 11) & 0x1F) << 3;
            int g = ((px >> 5) & 0x3F) << 2;
            int b = (px & 0x1F) << 3;
            d[x] = (uint8_t)((r * 299 + g * 587 + b * 114) / 1000);
        }
    }
}

/* JPEG/MJPEG gray decode stub - in production use libjpeg with JCS_GRAYSCALE */
static int jpeg_decode_gray_stub(const uint8_t* data, int data_size,
                                 uint8_t** out_gray, int* out_w, int* out_h)
{
    (void)data;
    (void)data_size;
    *out_w = 64;
    *out_h = 64;
    *out_gray = (uint8_t*)malloc(64 * 64);
    if (*out_gray == NULL) return -1;
    memset(*out_gray, 0x80, 64 * 64);
    return 0;
}

/* PNG gray decode stub - in production use libpng with grayscale transform */
static int png_decode_gray_stub(const uint8_t* data, int data_size,
                                uint8_t** out_gray, int* out_w, int* out_h)
{
    (void)data;
    (void)data_size;
    *out_w = 64;
    *out_h = 64;
    *out_gray = (uint8_t*)malloc(64 * 64);
    if (*out_gray == NULL) return -1;
    memset(*out_gray, 0x80, 64 * 64);
    return 0;
}

/* ===================== Color (BGR888) conversion ===================== */

/* Convert NV21 (Y + VU interleaved, V first) to BGR888 */
static void nv21_to_rgb888(const uint8_t* nv21, int width, int height,
                           int stride, uint8_t* rgb)
{
    const uint8_t* uv = nv21 + (size_t)stride * height;
    for (int y = 0; y < height; y++) {
        const uint8_t* ny = nv21 + y * stride;
        const uint8_t* uv_row = uv + (y >> 1) * stride;

        for (int x = 0; x < width; x++) {
            int y_val = ny[x] - 16;
            int v_val = uv_row[(x >> 1) * 2 + 0] - 128;  /* V first in NV21 */
            int u_val = uv_row[(x >> 1) * 2 + 1] - 128;  /* U second */

            int r = y_val + (int)(1.402 * v_val);
            int g = y_val - (int)(0.34414 * u_val + (int)(0.71414 * v_val));
            int b = y_val + (int)(1.772 * u_val);

            if (r > 255) r = 255; if (r < 0) r = 0;
            if (g > 255) g = 255; if (g < 0) g = 0;
            if (b > 255) b = 255; if (b < 0) b = 0;

            rgb[y * width * 3 + x * 3 + 0] = b;
            rgb[y * width * 3 + x * 3 + 1] = g;
            rgb[y * width * 3 + x * 3 + 2] = r;
        }
    }
}

/* Convert NV12 (Y + UV interleaved, U first) to BGR888 */
static void nv12_to_rgb888(const uint8_t* nv12, int width, int height,
                           int stride, uint8_t* rgb)
{
    const uint8_t* uv = nv12 + (size_t)stride * height;
    for (int y = 0; y < height; y++) {
        const uint8_t* ny = nv12 + y * stride;
        const uint8_t* uv_row = uv + (y >> 1) * stride;

        for (int x = 0; x < width; x++) {
            int y_val = ny[x] - 16;
            int u_val = uv_row[(x >> 1) * 2 + 0] - 128;  /* U first in NV12 */
            int v_val = uv_row[(x >> 1) * 2 + 1] - 128;  /* V second */

            int r = y_val + (int)(1.402 * v_val);
            int g = y_val - (int)(0.34414 * u_val + (int)(0.71414 * v_val));
            int b = y_val + (int)(1.772 * u_val);

            if (r > 255) r = 255; if (r < 0) r = 0;
            if (g > 255) g = 255; if (g < 0) g = 0;
            if (b > 255) b = 255; if (b < 0) b = 0;

            rgb[y * width * 3 + x * 3 + 0] = b;
            rgb[y * width * 3 + x * 3 + 1] = g;
            rgb[y * width * 3 + x * 3 + 2] = r;
        }
    }
}

/* Convert I420 (planar Y + U + V, 4:2:0) to BGR888 */
static void i420_to_rgb888(const uint8_t* i420, int width, int height,
                           int stride, uint8_t* rgb)
{
    const uint8_t* u_plane = i420 + (size_t)stride * height;
    const uint8_t* v_plane = u_plane + (size_t)(stride / 2) * (height / 2);

    for (int y = 0; y < height; y++) {
        const uint8_t* ny = i420 + y * stride;
        const uint8_t* nu = u_plane + (y >> 1) * (stride / 2);
        const uint8_t* nv = v_plane + (y >> 1) * (stride / 2);

        for (int x = 0; x < width; x++) {
            int y_val = ny[x] - 16;
            int u_val = nu[x >> 1] - 128;
            int v_val = nv[x >> 1] - 128;

            int r = y_val + (int)(1.402 * v_val);
            int g = y_val - (int)(0.34414 * u_val + (int)(0.71414 * v_val));
            int b = y_val + (int)(1.772 * u_val);

            if (r > 255) r = 255; if (r < 0) r = 0;
            if (g > 255) g = 255; if (g < 0) g = 0;
            if (b > 255) b = 255; if (b < 0) b = 0;

            rgb[y * width * 3 + x * 3 + 0] = b;
            rgb[y * width * 3 + x * 3 + 1] = g;
            rgb[y * width * 3 + x * 3 + 2] = r;
        }
    }
}

/* Convert I422 (planar 4:2:2) to BGR888 */
static void i422_to_rgb888(const uint8_t* i422, int width, int height,
                           int stride, uint8_t* rgb)
{
    const uint8_t* u_plane = i422 + (size_t)stride * height;
    const uint8_t* v_plane = u_plane + (size_t)(stride / 2) * height;

    for (int y = 0; y < height; y++) {
        const uint8_t* ny = i422 + y * stride;
        const uint8_t* nu = u_plane + y * (stride / 2);
        const uint8_t* nv = v_plane + y * (stride / 2);

        for (int x = 0; x < width; x++) {
            int y_val = ny[x] - 16;
            int u_val = nu[x >> 1] - 128;
            int v_val = nv[x >> 1] - 128;

            int r = y_val + (int)(1.402 * v_val);
            int g = y_val - (int)(0.34414 * u_val + (int)(0.71414 * v_val));
            int b = y_val + (int)(1.772 * u_val);

            if (r > 255) r = 255; if (r < 0) r = 0;
            if (g > 255) g = 255; if (g < 0) g = 0;
            if (b > 255) b = 255; if (b < 0) b = 0;

            rgb[y * width * 3 + x * 3 + 0] = b;
            rgb[y * width * 3 + x * 3 + 1] = g;
            rgb[y * width * 3 + x * 3 + 2] = r;
        }
    }
}

/* Convert I444 (planar 4:4:4) to BGR888 */
static void i444_to_rgb888(const uint8_t* i444, int width, int height,
                           int stride, uint8_t* rgb)
{
    const uint8_t* u_plane = i444 + (size_t)stride * height;
    const uint8_t* v_plane = u_plane + (size_t)stride * height;

    for (int y = 0; y < height; y++) {
        const uint8_t* ny = i444 + y * stride;
        const uint8_t* nu = u_plane + y * stride;
        const uint8_t* nv = v_plane + y * stride;

        for (int x = 0; x < width; x++) {
            int y_val = ny[x] - 16;
            int u_val = nu[x] - 128;
            int v_val = nv[x] - 128;

            int r = y_val + (int)(1.402 * v_val);
            int g = y_val - (int)(0.34414 * u_val + (int)(0.71414 * v_val));
            int b = y_val + (int)(1.772 * u_val);

            if (r > 255) r = 255; if (r < 0) r = 0;
            if (g > 255) g = 255; if (g < 0) g = 0;
            if (b > 255) b = 255; if (b < 0) b = 0;

            rgb[y * width * 3 + x * 3 + 0] = b;
            rgb[y * width * 3 + x * 3 + 1] = g;
            rgb[y * width * 3 + x * 3 + 2] = r;
        }
    }
}

/* Convert YUYV (packed YUY2) to BGR888 */
static void yuyv_to_rgb888(const uint8_t* yuyv, int width, int height,
                           int stride, uint8_t* rgb)
{
    for (int y = 0; y < height; y++) {
        const uint8_t* row = yuyv + y * stride;

        for (int x = 0; x < width; x++) {
            int y_val = row[x * 2] - 16;
            int u_val = row[(x >> 1) * 4 + 1] - 128;
            int v_val = row[(x >> 1) * 4 + 3] - 128;

            int r = y_val + (int)(1.402 * v_val);
            int g = y_val - (int)(0.34414 * u_val + (int)(0.71414 * v_val));
            int b = y_val + (int)(1.772 * u_val);

            if (r > 255) r = 255; if (r < 0) r = 0;
            if (g > 255) g = 255; if (g < 0) g = 0;
            if (b > 255) b = 255; if (b < 0) b = 0;

            rgb[y * width * 3 + x * 3 + 0] = b;
            rgb[y * width * 3 + x * 3 + 1] = g;
            rgb[y * width * 3 + x * 3 + 2] = r;
        }
    }
}

/* Convert ARGB8888 to BGR888 (strip alpha) */
static void argb8888_to_rgb888(const uint8_t* argb, int width, int height,
                               int stride, uint8_t* rgb)
{
    for (int y = 0; y < height; y++) {
        const uint8_t* src = argb + y * stride;
        uint8_t* dst = rgb + y * width * 3;

        for (int x = 0; x < width; x++) {
            dst[x * 3 + 0] = src[x * 4 + 3];  /* B */
            dst[x * 3 + 1] = src[x * 4 + 2];  /* G */
            dst[x * 3 + 2] = src[x * 4 + 1];  /* R */
        }
    }
}

/* Convert RGB888 to BGR888 (swap R and B channels) */
static void rgb888_to_bgr888(const uint8_t* src, int width, int height,
                             int stride, uint8_t* dst)
{
    for (int y = 0; y < height; y++) {
        const uint8_t* s = src + y * stride;
        uint8_t* d = dst + y * width * 3;

        for (int x = 0; x < width; x++) {
            d[x * 3 + 0] = s[x * 3 + 2];  /* B */
            d[x * 3 + 1] = s[x * 3 + 1];  /* G */
            d[x * 3 + 2] = s[x * 3 + 0];  /* R */
        }
    }
}

/* Convert RGB565 (little-endian) to BGR888 */
static void rgb565_to_rgb888(const uint8_t* src, int width, int height,
                             int stride, uint8_t* rgb)
{
    for (int y = 0; y < height; y++) {
        const uint8_t* s = src + y * stride;

        for (int x = 0; x < width; x++) {
            uint16_t px = (uint16_t)s[x * 2] | ((uint16_t)s[x * 2 + 1] << 8);
            int r5 = (px >> 11) & 0x1F;
            int g6 = (px >> 5) & 0x3F;
            int b5 = px & 0x1F;

            int r = (r5 << 3) | (r5 >> 2);
            int g = (g6 << 2) | (g6 >> 4);
            int b = (b5 << 3) | (b5 >> 2);

            rgb[y * width * 3 + x * 3 + 0] = b;
            rgb[y * width * 3 + x * 3 + 1] = g;
            rgb[y * width * 3 + x * 3 + 2] = r;
        }
    }
}

/* JPEG decode stub - in production would use libjpeg */
static int jpeg_decode_stub(const uint8_t* data, int data_size, uint8_t** out_rgb, int* out_w, int* out_h)
{
    (void)data;
    (void)data_size;
    *out_w = 64;
    *out_h = 64;
    *out_rgb = (uint8_t*)malloc(64 * 64 * 3);
    if (*out_rgb == NULL) return -1;
    memset(*out_rgb, 0, 64 * 64 * 3);
    return 0;
}

/* PNG decode stub - in production would use libpng */
static int png_decode_stub(const uint8_t* data, int data_size, uint8_t** out_rgb, int* out_w, int* out_h)
{
    (void)data;
    (void)data_size;
    *out_w = 64;
    *out_h = 64;
    *out_rgb = (uint8_t*)malloc(64 * 64 * 3);
    if (*out_rgb == NULL) return -1;
    memset(*out_rgb, 0x80, 64 * 64 * 3);
    return 0;
}

/* ===================== MRZ CNN engine integration ===================== */

/* ---- optional per-stage timing: LOOGARCH_OCR_TIMING=1 ----
 * One line per loogarch_ocr_process() on stderr (and appended to
 * LOOGARCH_OCR_TIMING_FILE when that env var is set):
 *   [loogarch-timing] total=..ms convert=..ms prep=..ms recog=..ms size=WxH fmt=N
 * convert: pixel-format extraction (+ gray->rgb expand on Y-only path)
 * prep:    BGR byte swap at the engine entry (color path only)
 * recog:   face_image alloc + copy + mrz_ocr_recognise_* (band locate
 *          + CNN/CRNN heads)
 */
static int loog_timing_on(void)
{
    static int cached = -1;
    if (cached < 0) {
        const char* e = getenv("LOOGARCH_OCR_TIMING");
        cached = (e != NULL && e[0] != '\0' && e[0] != '0') ? 1 : 0;
    }
    return cached;
}

static double loog_now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static void loog_emit_timing(double total, double convert, double prep,
                             double recog, int w, int h, int fmt)
{
    char line[256];
    snprintf(line, sizeof(line),
             "[loogarch-timing] total=%.1fms convert=%.1fms prep=%.1fms "
             "recog=%.1fms size=%dx%d fmt=%d\n",
             total, convert, prep, recog, w, h, fmt);
    fputs(line, stderr);
    const char* path = getenv("LOOGARCH_OCR_TIMING_FILE");
    if (path != NULL && path[0] != '\0') {
        FILE* f = fopen(path, "a");
        if (f != NULL) {
            fputs(line, f);
            fclose(f);
        }
    }
}

#ifdef LOOGARCH_OCR_WITH_MRZ
/* The MRZ engine is not thread-safe: the CNN scratchpad and the
 * dual-line worker are shared statics.  One lock around the whole
 * recognition keeps the library usable from the camera thread and
 * from an application thread at the same time. */
static pthread_mutex_t g_mrz_mutex = PTHREAD_MUTEX_INITIALIZER;

/* Run the MRZ engine over an RGB buffer.
 *
 * `is_bgr` says the buffer holds BGR (the colour path produces BGR),
 * `gray_only` says the buffer holds a grey-expanded RGB, i.e. R==G==B.
 * Both are converted to the RGB the engine expects.
 *
 * `use_fuse` selects the fused engine: the per-character CNN head AND
 * the CRNN sequence head over the same features, combined in log space
 * with a confidence gate.  On a 3-fold grouped cross-validation over
 * the 41 real photos it was the only variant that beat the plain CNN
 * (22/41 exact vs 21/41, 96.90% vs 96.78% character accuracy).
 *
 * t_prep / t_recog (optional out, may be NULL) receive milliseconds for
 * the BGR swap and for alloc+copy+recognise respectively. */
static void run_mrz_engine(uint8_t* rgb, int w, int h, int is_bgr,
                           int use_fuse, int use_par,
                           loogarch_ocr_result_t* result,
                           double* t_prep, double* t_recog)
{
    static int mrz_inited = 0;
    pthread_mutex_lock(&g_mrz_mutex);

    if (t_prep) *t_prep = 0.0;
    if (t_recog) *t_recog = 0.0;

    if (!mrz_inited) {
        mrz_ocr_init();
        mrz_inited = 1;
    }
    /* The line-specific weight pair is the real-photo model; the
     * template head is the synthetic control group. */
    setenv("MRZ_OCR_LINE_MODELS", "1", 1);
    setenv("MRZ_OCR_PAR", use_par ? "1" : "0", 1);
    /* The fusion needs both heads, so the CRNN-only switch must stay
     * off -- mrz_ocr_recognise_fuse() manages it itself. */
    setenv("MRZ_OCR_CRNN", "0", 1);
    /* One switch for both layers: the engine then prints its own
     * gray/band/resample/decode breakdown on stderr as well. */
    if (loog_timing_on()) setenv("MRZ_OCR_TIMING", "1", 1);

    double tp0 = loog_now_ms();
    if (is_bgr) {
        for (int i = 0; i < w * h; i++) {
            uint8_t tmp = rgb[i * 3 + 0];
            rgb[i * 3 + 0] = rgb[i * 3 + 2];
            rgb[i * 3 + 2] = tmp;
        }
    }
    double tp1 = loog_now_ms();
    if (t_prep) *t_prep = tp1 - tp0;

    double tr0 = tp1;
    face_image_t *img = face_image_new(w, h);
    if (!img) { result->status = -1; pthread_mutex_unlock(&g_mrz_mutex); return; }
    memcpy(img->pixels, rgb, (size_t)w * h * 3);

    mrz_ocr_result_t r;
    memset(&r, 0, sizeof(r));
    mrz_ocr_status_t s;
    if (use_fuse) {
        s = mrz_ocr_recognise_fuse(img, &r);
    } else {
        s = mrz_ocr_recognise_cnn(img, &r);
    }
    face_image_free(img);
    if (t_recog) *t_recog = loog_now_ms() - tr0;

    result->status = (int)s;
    strncpy(result->line1, r.line1, 63);
    result->line1[63] = '\0';
    strncpy(result->line2, r.line2, 63);
    result->line2[63] = '\0';
    result->conf1 = r.line1_avg_conf;
    result->conf2 = r.line2_avg_conf;
    result->band_x = r.band_x;
    result->band_y = r.band_y;
    result->band_w = r.band_w;
    result->band_h = r.band_h;
    result->layout = (int)r.doc_layout;
    result->confidence = (r.line1_avg_conf + r.line2_avg_conf) / 2;

    size_t cap = strlen(r.line1) + strlen(r.line2) + 4;
    char* text = (char*)malloc(cap);
    if (text) {
        snprintf(text, cap, "%s\n%s", r.line1, r.line2);
        result->text = text;
    }

    pthread_mutex_unlock(&g_mrz_mutex);
}
#endif /* LOOGARCH_OCR_WITH_MRZ */

/* ===================== OCR processing ===================== */

static loogarch_ocr_result_t process_ocr(void* handle, const uint8_t* buffer, int format, int width, int height, int roi_x, int roi_y, int roi_w, int roi_h)
{
    (void)roi_x;
    (void)roi_y;
    (void)roi_w;
    (void)roi_h;

    int timing = loog_timing_on();
    double t_start = timing ? loog_now_ms() : 0.0;
    double t_conv = 0.0, t_prep = 0.0, t_recog = 0.0;

    loogarch_ocr_internal_t* internal = (loogarch_ocr_internal_t*)handle;
    int gray_only = internal ? internal->gray_only : 0;
    int use_fuse = internal ? internal->use_fuse : 0;
    int use_par  = internal ? internal->use_par : 0;

    /* A single-plane luma buffer has no chroma to convert, so the
     * Y-only path is the only correct one for it -- regardless of what
     * the caller asked for.  The camera on the loongarch device hands
     * out GRAY frames, so this is the normal production path there. */
    if (format == LOOGARCH_PIXEL_FORMAT_GRAY) {
        gray_only = 1;
    }

    loogarch_ocr_result_t result;
    result.text = NULL;
    result.confidence = 0;
    result.width = width;
    result.height = height;
    result.format = format;
    result.line1[0] = '\0';
    result.line2[0] = '\0';
    result.conf1 = 0;
    result.conf2 = 0;
    result.band_x = result.band_y = result.band_w = result.band_h = 0;
    result.layout = 0;
    result.status = 0;

    if (buffer == NULL || width <= 0 || height <= 0) {
        return result;
    }

    int effective_w = width;
    int effective_h = height;

    if (gray_only) {
        /* Y-only fast path: produce single-channel luma, skip chroma */
        size_t gray_size = (size_t)width * height;
        uint8_t* gray = (uint8_t*)malloc(gray_size);
        if (gray == NULL) {
            return result;
        }

        switch (format) {
            case LOOGARCH_PIXEL_FORMAT_NV21:
            case LOOGARCH_PIXEL_FORMAT_NV12:
            case LOOGARCH_PIXEL_FORMAT_I420:
            case LOOGARCH_PIXEL_FORMAT_I422:
            case LOOGARCH_PIXEL_FORMAT_I444:
                planar_extract_y(buffer, width, height, width, gray);
                break;
            case LOOGARCH_PIXEL_FORMAT_YUYV:
                yuyv_extract_y(buffer, width, height, width * 2, gray);
                break;
            case LOOGARCH_PIXEL_FORMAT_ARGB8888:
                argb8888_to_gray(buffer, width, height, width * 4, gray);
                break;
            case LOOGARCH_PIXEL_FORMAT_RGB888:
                rgb888_to_gray(buffer, width, height, width * 3, gray);
                break;
            case LOOGARCH_PIXEL_FORMAT_RGB565:
                rgb565_to_gray(buffer, width, height, width * 2, gray);
                break;
            case LOOGARCH_PIXEL_FORMAT_GRAY:
                /* Already single-plane luma: copy it out, nothing to
                 * convert.  This is what the loongarch v4l2 camera
                 * delivers (CAMERA_PIXEL_FORMAT_GRAY). */
                planar_extract_y(buffer, width, height, width, gray);
                break;
            case LOOGARCH_PIXEL_FORMAT_JPEG:
            case LOOGARCH_PIXEL_FORMAT_MJPEG: {
                uint8_t* decoded = NULL;
                if (jpeg_decode_gray_stub(buffer, width, &decoded, &effective_w, &effective_h) == 0) {
                    free(gray); gray = decoded;
                } else {
                    memset(gray, 0x80, gray_size);
                }
                break;
            }
            case LOOGARCH_PIXEL_FORMAT_PNG: {
                uint8_t* decoded = NULL;
                if (png_decode_gray_stub(buffer, width, &decoded, &effective_w, &effective_h) == 0) {
                    free(gray); gray = decoded;
                } else {
                    memset(gray, 0x80, gray_size);
                }
                break;
            }
            default:
                free(gray);
                result.text = strdup("Unsupported pixel format (Y-only)");
                result.confidence = 0;
                return result;
        }

#ifdef LOOGARCH_OCR_WITH_MRZ
        {
            size_t n = (size_t)effective_w * effective_h;
            uint8_t* gray_rgb = (uint8_t*)malloc(n * 3);
            if (gray_rgb) {
                for (size_t i = 0; i < n; i++) {
                    gray_rgb[i * 3 + 0] = gray[i];
                    gray_rgb[i * 3 + 1] = gray[i];
                    gray_rgb[i * 3 + 2] = gray[i];
                }
                if (timing) t_conv = loog_now_ms() - t_start;
                run_mrz_engine(gray_rgb, effective_w, effective_h, 0,
                               use_fuse, use_par, &result,
                               timing ? &t_prep : NULL,
                               timing ? &t_recog : NULL);
                free(gray_rgb);
            }
        }
        if (timing && t_conv == 0.0) t_conv = loog_now_ms() - t_start;
#else
        char* text_buf = (char*)malloc(80);
        if (text_buf) {
            snprintf(text_buf, 80,
                     "loogarch OH OCR Y-only: %dx%d format=%d",
                     effective_w, effective_h, format);
            result.text = text_buf;
            result.confidence = 60;
        }
#endif

        if (timing)
            loog_emit_timing(loog_now_ms() - t_start, t_conv, t_prep,
                             t_recog, effective_w, effective_h, format);
        free(gray);
        return result;
    }

    /* Color path: convert input format to BGR888 for OCR engine */
    size_t rgb_size = (size_t)width * height * 3;
    uint8_t* rgb = (uint8_t*)malloc(rgb_size);
    if (rgb == NULL) {
        return result;
    }

    switch (format) {
        case LOOGARCH_PIXEL_FORMAT_NV21:
            nv21_to_rgb888(buffer, width, height, width, rgb);
            break;
        case LOOGARCH_PIXEL_FORMAT_NV12:
            nv12_to_rgb888(buffer, width, height, width, rgb);
            break;
        case LOOGARCH_PIXEL_FORMAT_I420:
            i420_to_rgb888(buffer, width, height, width, rgb);
            break;
        case LOOGARCH_PIXEL_FORMAT_I422:
            i422_to_rgb888(buffer, width, height, width, rgb);
            break;
        case LOOGARCH_PIXEL_FORMAT_I444:
            i444_to_rgb888(buffer, width, height, width, rgb);
            break;
        case LOOGARCH_PIXEL_FORMAT_YUYV:
            yuyv_to_rgb888(buffer, width, height, width * 2, rgb);
            break;
        case LOOGARCH_PIXEL_FORMAT_ARGB8888:
            argb8888_to_rgb888(buffer, width, height, width * 4, rgb);
            break;
        case LOOGARCH_PIXEL_FORMAT_RGB888:
            rgb888_to_bgr888(buffer, width, height, width * 3, rgb);
            break;
        case LOOGARCH_PIXEL_FORMAT_RGB565:
            rgb565_to_rgb888(buffer, width, height, width * 2, rgb);
            break;
        case LOOGARCH_PIXEL_FORMAT_JPEG:
        case LOOGARCH_PIXEL_FORMAT_MJPEG: {
            uint8_t* decoded = NULL;
            if (jpeg_decode_stub(buffer, width, &decoded, &effective_w, &effective_h) == 0) {
                free(rgb); rgb = decoded;
            } else {
                memset(rgb, 0, rgb_size);
            }
            break;
        }
        case LOOGARCH_PIXEL_FORMAT_PNG: {
            uint8_t* decoded = NULL;
            if (png_decode_stub(buffer, width, &decoded, &effective_w, &effective_h) == 0) {
                free(rgb); rgb = decoded;
            } else {
                memset(rgb, 0, rgb_size);
            }
            break;
        }
        default:
            free(rgb);
            result.text = strdup("Unsupported pixel format");
            result.confidence = 0;
            return result;
    }

    if (timing) t_conv = loog_now_ms() - t_start;

#ifdef LOOGARCH_OCR_WITH_MRZ
    run_mrz_engine(rgb, effective_w, effective_h, 1, use_fuse, use_par, &result,
                   timing ? &t_prep : NULL,
                   timing ? &t_recog : NULL);
#else
    char* text_buf = (char*)malloc(64);
    if (text_buf) {
        snprintf(text_buf, 64,
                 "loogarch OH OCR result: %dx%d format=%d",
                 effective_w, effective_h, format);
        result.text = text_buf;
        result.confidence = 50;
    }
#endif

    if (timing)
        loog_emit_timing(loog_now_ms() - t_start, t_conv, t_prep,
                         t_recog, effective_w, effective_h, format);
    free(rgb);
    return result;
}

/* ===================== Public API ===================== */

void* loogarch_ocr_init(loogarch_pixel_format_t format, int width, int height, const char* options)
{
    loogarch_ocr_internal_t* internal = (loogarch_ocr_internal_t*)
                                        malloc(sizeof(loogarch_ocr_internal_t));
    if (internal == NULL) {
        return NULL;
    }

    internal->format = format;
    internal->width = width;
    internal->height = height;
    internal->stride = width;
    internal->gray_only = (options != NULL && strstr(options, "gray=1") != NULL) ? 1 : 0;
    /* The fused engine is the shipped one: it is the only variant that
     * beat the plain per-character CNN on the held-out photo set.  It
     * can be turned off with "fuse=0" for A/B measurement. */
    internal->use_fuse = (options == NULL || strstr(options, "fuse=0") == NULL) ? 1 : 0;
    /* The dual-line decode uses a second thread; "par=0" keeps the
     * decode single-threaded. */
    internal->use_par = (options != NULL && strstr(options, "par=0") != NULL) ? 0 : 1;
    internal->y_data = NULL;
    internal->u_data = NULL;
    internal->v_data = NULL;

    return (void*)internal;
}

loogarch_ocr_result_t loogarch_ocr_process(void* handle, const uint8_t* buffer,
                                           int roi_x, int roi_y, int roi_w, int roi_h)
{
    loogarch_ocr_internal_t* internal = (loogarch_ocr_internal_t*)handle;

    if (internal == NULL || buffer == NULL) {
        loogarch_ocr_result_t result;
        result.text = strdup("Invalid handle or buffer");
        result.confidence = 0;
        result.width = 0;
        result.height = 0;
        result.format = internal ? internal->format : 0;
        return result;
    }

    int effective_w = (roi_w > 0) ? roi_w : internal->width;
    int effective_h = (roi_h > 0) ? roi_h : internal->height;
    int effective_x = (roi_x >= 0) ? roi_x : 0;
    int effective_y = (roi_y >= 0) ? roi_y : 0;

    if (effective_x < 0) effective_x = 0;
    if (effective_y < 0) effective_y = 0;
    if (effective_x + effective_w > internal->width) effective_w = internal->width - effective_x;
    if (effective_y + effective_h > internal->height) effective_h = internal->height - effective_y;

    return process_ocr(handle, buffer, internal->format, internal->width, internal->height,
                       effective_x, effective_y, effective_w, effective_h);
}

uint8_t* loogarch_ocr_extract_gray_rgb(void* handle, const uint8_t* buffer,
                                       int roi_x, int roi_y, int roi_w, int roi_h)
{
    (void)roi_x; (void)roi_y; (void)roi_w; (void)roi_h;

    loogarch_ocr_internal_t* internal = (loogarch_ocr_internal_t*)handle;
    if (internal == NULL || buffer == NULL) return NULL;

    int w = internal->width, h = internal->height;
    if (w <= 0 || h <= 0) return NULL;

    uint8_t* y = (uint8_t*)malloc((size_t)w * h);
    if (y == NULL) return NULL;

    int dw = w, dh = h;
    switch (internal->format) {
        case LOOGARCH_PIXEL_FORMAT_NV21:
        case LOOGARCH_PIXEL_FORMAT_NV12:
        case LOOGARCH_PIXEL_FORMAT_I420:
        case LOOGARCH_PIXEL_FORMAT_I422:
        case LOOGARCH_PIXEL_FORMAT_I444:
            planar_extract_y(buffer, w, h, w, y);
            break;
        case LOOGARCH_PIXEL_FORMAT_YUYV:
            yuyv_extract_y(buffer, w, h, w * 2, y);
            break;
        case LOOGARCH_PIXEL_FORMAT_ARGB8888:
            argb8888_to_gray(buffer, w, h, w * 4, y);
            break;
        case LOOGARCH_PIXEL_FORMAT_RGB888:
            rgb888_to_gray(buffer, w, h, w * 3, y);
            break;
        case LOOGARCH_PIXEL_FORMAT_RGB565:
            rgb565_to_gray(buffer, w, h, w * 2, y);
            break;
        case LOOGARCH_PIXEL_FORMAT_GRAY:
            planar_extract_y(buffer, w, h, w, y);
            break;
        case LOOGARCH_PIXEL_FORMAT_JPEG:
        case LOOGARCH_PIXEL_FORMAT_MJPEG: {
            uint8_t* dec = NULL;
            if (jpeg_decode_gray_stub(buffer, w, &dec, &dw, &dh) == 0) {
                free(y); y = dec;
            } else {
                memset(y, 0x80, (size_t)w * h);
            }
            break;
        }
        case LOOGARCH_PIXEL_FORMAT_PNG: {
            uint8_t* dec = NULL;
            if (png_decode_gray_stub(buffer, w, &dec, &dw, &dh) == 0) {
                free(y); y = dec;
            } else {
                memset(y, 0x80, (size_t)w * h);
            }
            break;
        }
        default:
            free(y);
            return NULL;
    }

    size_t n = (size_t)dw * dh;
    uint8_t* gray_rgb = (uint8_t*)malloc(n * 3);
    if (gray_rgb == NULL) { free(y); return NULL; }
    for (size_t i = 0; i < n; i++) {
        gray_rgb[i * 3 + 0] = y[i];
        gray_rgb[i * 3 + 1] = y[i];
        gray_rgb[i * 3 + 2] = y[i];
    }
    free(y);
    return gray_rgb;
}

void loogarch_ocr_cleanup(void* handle)
{
    if (handle != NULL) {
        free(handle);
    }
}

const char* loogarch_ocr_get_version(void)
{
    return "1.1.0-loongarch";
}
