/*
 * ohocr.h - OCR library for OpenHarmony desktop
 *
 * Camera format support: NV21, NV12, I420, I422, I444, YUYV,
 *                        ARGB8888, RGB888, RGB565, MJPEG, JPEG, PNG
 * Buffer-based interface for camera integration
 *
 * Copyright (c) 2026 passport
 * Licensed under Apache-2.0
 */

#ifndef OHOCR_H
#define OHOCR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Pixel format enumeration for camera compatibility
 */
typedef enum {
    OHOCR_PIXEL_FORMAT_NV21 = 0,   /* YUV 4:2:0 planar, VU interleaved (camera preview) */
    OHOCR_PIXEL_FORMAT_NV12,       /* YUV 4:2:0 planar, UV interleaved */
    OHOCR_PIXEL_FORMAT_I420,       /* YUV 4:2:0 planar (Y + U + V) */
    OHOCR_PIXEL_FORMAT_I422,       /* YUV 4:2:2 planar */
    OHOCR_PIXEL_FORMAT_I444,       /* YUV 4:4:4 planar */
    OHOCR_PIXEL_FORMAT_ARGB8888,   /* 32-bit ARGB */
    OHOCR_PIXEL_FORMAT_RGB888,     /* 24-bit RGB */
    OHOCR_PIXEL_FORMAT_RGB565,     /* 16-bit RGB565 */
    OHOCR_PIXEL_FORMAT_JPEG,       /* Compressed JPEG data */
    OHOCR_PIXEL_FORMAT_PNG,        /* Compressed PNG data */
    OHOCR_PIXEL_FORMAT_YUYV,       /* Packed YUV 4:2:2 (YUY2), camera format */
    OHOCR_PIXEL_FORMAT_MJPEG,      /* Motion JPEG, camera compressed format */
    OHOCR_PIXEL_FORMAT_GRAY,       /* Single-plane luma, Y only (1 byte/pixel) */
    OHOCR_PIXEL_FORMAT_LAST
} ohocr_pixel_format_t;

/*
 * OCR result structure
 */
typedef struct {
    char* text;          /* MRZ text "line1\nline2" (caller frees) */
    int  confidence;     /* Overall confidence score (0-100) */
    int  width;          /* Image width in pixels */
    int  height;         /* Image height in pixels */
    int  format;         /* Input pixel format */
    /* MRZ-specific fields (filled when compiled with OHOCR_WITH_MRZ).
     * When the MRZ engine is not compiled in, line1/line2 are empty and
     * status is 0.  layout: 0=unknown, 1=CS/CD single-30, 2=TD3 double-44,
     * 3=TD1 triple-30.  status: 0=OK, negative=mrz_ocr_status_t error. */
    char line1[64];
    char line2[64];
    int  conf1;
    int  conf2;
    int  band_x, band_y, band_w, band_h;
    int  layout;
    int  status;
} ohocr_result_t;

/*
 * Initialize OCR with options
 * @format: Input pixel format (NV21, NV12, I420, I422, I444, YUYV,
 *          ARGB8888, RGB888, RGB565, MJPEG, JPEG, PNG)
 * @width: Image width
 * @height: Image height
 * @options: Null-terminated options string (e.g., "lang=eng;psm=3;gray=1")
 *           gray=1 enables Y-only fast path: skip chroma conversion and feed
 *           single-channel luma to the OCR engine (optimal for MRZ decoding).
 *
 * Returns: OCR handle or NULL on failure
 */
void* ohocr_init(ohocr_pixel_format_t format, int width, int height, const char* options);

/*
 * Perform OCR on image buffer
 * @handle: OCR handle from ohocr_init()
 * @buffer: Raw image data buffer
 * @roi_x: Region of interest X offset (0 for full image)
 * @roi_y: Region of interest Y offset (0 for full image)
 * @roi_w: Region of interest width (0 for full width)
 * @roi_h: Region of interest height (0 for full height)
 *
 * Returns: ohocr_result_t with extracted text and metadata
 */
ohocr_result_t ohocr_process(void* handle, const uint8_t* buffer,
                                           int roi_x, int roi_y, int roi_w, int roi_h);

/*
 * Extract luma as grayscale RGB (Y,Y,Y) for direct feed to a single-channel
 * OCR engine (e.g. 6_mrz_ocr mrz_ocr_recognise_cnn).  Equivalent to the
 * gray=1 fast path but returns the image buffer instead of running the
 * (placeholder) OCR engine.  Caller frees the returned buffer with free().
 *
 * Returns: malloc'd W*H*3 buffer with R==G==B==Y, or NULL on failure.
 */
uint8_t* ohocr_extract_gray_rgb(void* handle, const uint8_t* buffer,
                                       int roi_x, int roi_y, int roi_w, int roi_h);

/*
 * Clean up OCR handle
 * @handle: OCR handle from ohocr_init()
 */
void ohocr_cleanup(void* handle);

/*
 * Get OCR version string
 */
const char* ohocr_get_version(void);

#ifdef __cplusplus
}
#endif

#endif /* OHOCR_H */
