# OpenHarmony OCR library -- CNN + fused MRZ engine

The MRZ engine is the CNN head plus the CRNN sequence head fused in log
space with a confidence gate (`mrz_ocr_recognise_fuse`).  The camera
path accepts a single-plane luma buffer, which is what the loongarch
v4l2 camera delivers.

## Layout

```
src/                     the library itself
  ohocr.h           public API and the pixel format enum
  ohocr.c           the library
  ohocr_optimized.c the cache-optimised variant
engine/                  the MRZ engine (CNN head + CRNN head + fusion)
  include/                mrz_ocr.h, cnn.h, crnn.h
  src/                    the engine sources and the exported weights
  face/                   the image helpers the engine needs
```

The tree is self-contained: it builds without the rest of the passport
repository.

## Build

Native build:

```bash
make mrz
```

Cross build for loongarch64:

```bash
make CROSS=loongarch64-linux-gnu- mrz
```

OHOS NDK build for loongarch64:

```bash
make OHOS_NDK=/path/to/sdk/native ohos
```

OHOS NDK build for arm64-v8a:

```bash
make OHOS_NDK=/path/to/sdk/native OHOS_ARCH=arm64 ohos
```

`make` alone builds the two libraries without the MRZ engine, which
only serves the pixel format conversion API.

## Pixel formats

`ohocr_pixel_format_t` covers NV21, NV12, I420, I422, I444, YUYV,
ARGB8888, RGB888, RGB565, JPEG, PNG, MJPEG and GRAY.

`OHOCR_PIXEL_FORMAT_GRAY` is a single-plane luma buffer, one byte per
pixel.  A GRAY buffer has no chroma, so it always takes the Y-only
path, whatever the options say.  This is the format the loongarch v4l2
camera produces (`CAMERA_PIXEL_FORMAT_GRAY`).

## Options

`ohocr_init()` takes a semicolon separated option string:

| option | effect |
|---|---|
| `gray=1` | force the Y-only path: extract luma and skip chroma conversion |
| `fuse=0` | use the plain per-character CNN head instead of the fused engine |
| `par=0` | decode the two MRZ lines single-threaded |

The fused engine is the default.  On a 3-fold grouped cross-validation
over the 41 real photos it was the only variant that beat the plain CNN
(22/41 exact against 21/41).  On the 44-photo set measured here it is
on par with the plain CNN: 38 of 44 readings are identical, and of the
6 that differ the fused engine is better on 2, worse on 1 and equal on 3.

## API

```c
void* ohocr_init(ohocr_pixel_format_t format, int w, int h,
                        const char* options);
ohocr_result_t ohocr_process(void* handle, const uint8_t* buf,
                                           int roi_x, int roi_y, int roi_w, int roi_h);
uint8_t* ohocr_extract_gray_rgb(void* handle, const uint8_t* buf,
                                       int roi_x, int roi_y, int roi_w, int roi_h);
void ohocr_cleanup(void* handle);
const char* ohocr_get_version(void);
```

`ohocr_process()` fills `line1`, `line2`, `conf1`, `conf2`, the
band rectangle, the document layout and the engine status.  `text` is
malloc'd and the caller frees it.

The recognition path is guarded by a mutex, so the camera thread and an
application thread can call the library at the same time.
