#!/usr/bin/env python3
"""
test_photo_formats.py - 用真实 PassPortPhoto 图片验证 ohocr 格式转换正确性

流程: PassPortPhoto(JPG) -> PIL 解码 RGB -> 编码成 YUYV/NV21/MJPEG/ARGB/RGB888
      -> 喂给 libohocr.so (color + gray=1) -> 验证维度/格式/不崩溃
      + Y 平面闭环自检: RGB->YUYV 编码的 Y 字节 == BT.601 luma
"""
import ctypes
import sys
import os
import glob
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTO_DIR = os.path.join(HERE, "..", "tmp", "PassPortPhoto")

LIBC = ctypes.CDLL(None)
LIBC.free.argtypes = [ctypes.c_void_p]
LIBC.free.restype = None

# 像素格式枚举 (ohocr.h)
FMT = {
    "NV21": 0, "NV12": 1, "I420": 2, "I422": 3, "I444": 4,
    "ARGB8888": 5, "RGB888": 6, "RGB565": 7, "JPEG": 8, "PNG": 9,
    "YUYV": 10, "MJPEG": 11,
}


class OcrResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_void_p),
        ("confidence", ctypes.c_int),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("format", ctypes.c_int),
    ]


def load_lib(path):
    lib = ctypes.CDLL(path)
    lib.ohocr_init.restype = ctypes.c_void_p
    lib.ohocr_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
    lib.ohocr_process.restype = OcrResult
    lib.ohocr_process.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.ohocr_cleanup.argtypes = [ctypes.c_void_p]
    lib.ohocr_get_version.restype = ctypes.c_char_p
    return lib


def rgb_to_yuyv(rgb, w, h):
    r = rgb[:, :, 0].astype(np.int32); g = rgb[:, :, 1].astype(np.int32); b = rgb[:, :, 2].astype(np.int32)
    y = np.clip((0.257 * r + 0.504 * g + 0.098 * b + 16), 0, 255).astype(np.uint8)
    cb = np.clip((-0.148 * r - 0.291 * g + 0.439 * b + 128), 0, 255).astype(np.uint8)
    cr = np.clip((0.439 * r - 0.368 * g - 0.071 * b + 128), 0, 255).astype(np.uint8)
    yuyv = np.zeros((h, w * 2), dtype=np.uint8)
    yuyv[:, 0::2] = y
    yuyv[:, 1::4] = cb[:, ::2]
    yuyv[:, 3::4] = cr[:, ::2]
    return yuyv.tobytes(), y


def rgb_to_nv21(rgb, w, h):
    r = rgb[:, :, 0].astype(np.int32); g = rgb[:, :, 1].astype(np.int32); b = rgb[:, :, 2].astype(np.int32)
    y = np.clip((0.257 * r + 0.504 * g + 0.098 * b + 16), 0, 255).astype(np.uint8)
    cb = np.clip((-0.148 * r - 0.291 * g + 0.439 * b + 128), 0, 255).astype(np.uint8)
    cr = np.clip((0.439 * r - 0.368 * g - 0.071 * b + 128), 0, 255).astype(np.uint8)
    cb_ds = cb[::2, ::2]; cr_ds = cr[::2, ::2]
    vu = np.zeros((h // 2, w), dtype=np.uint8)
    vu[:, 0::2] = cr_ds
    vu[:, 1::2] = cb_ds
    return y.tobytes() + vu.tobytes(), y


def rgb_to_argb(rgb, w, h):
    argb = np.zeros((h, w, 4), dtype=np.uint8)
    argb[:, :, 0] = 255
    argb[:, :, 1] = rgb[:, :, 0]
    argb[:, :, 2] = rgb[:, :, 1]
    argb[:, :, 3] = rgb[:, :, 2]
    return argb.tobytes()


def rgb_to_mjpeg(rgb, w, h):
    import io
    buf = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def run_one(lib, fmt_id, fmt_name, buf, w, h, gray):
    handle = lib.ohocr_init(fmt_id, w, h, b"gray=1" if gray else None)
    if not handle:
        return False, "init NULL"
    res = lib.ohocr_process(handle, buf, 0, 0, 0, 0)
    lib.ohocr_cleanup(handle)
    txt = ""
    if res.text:
        txt = ctypes.string_at(res.text).decode("utf-8", "replace")
        LIBC.free(res.text)
    if "Unsupported" in txt:
        return False, f"unsupported: {txt}"
    if res.width != w or res.height != h:
        return False, f"dim mismatch {res.width}x{res.height} != {w}x{h}"
    if res.format != fmt_id:
        return False, f"fmt mismatch {res.format} != {fmt_id}"
    if res.confidence <= 0:
        return False, f"conf={res.confidence}"
    return True, txt


def main():
    lib_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "libohocr.so")
    lib = load_lib(lib_path)
    print(f"lib: {os.path.basename(lib_path)}  version: {lib.ohocr_get_version().decode()}")

    photos = sorted(glob.glob(os.path.join(PHOTO_DIR, "*.jpg")))
    if not photos:
        print(f"ERROR: no photos in {PHOTO_DIR}"); return 1
    print(f"photos: {len(photos)}")

    TARGET_W, TARGET_H = 1024, 768  # 偶数尺寸，快速且一致
    fails = 0; total = 0; y_checked = 0; y_mismatch = 0

    for path in photos:
        img = Image.open(path).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS)
        rgb = np.array(img)
        w, h = TARGET_W, TARGET_H

        yuyv_buf, y_ref = rgb_to_yuyv(rgb, w, h)
        nv21_buf, _ = rgb_to_nv21(rgb, w, h)
        argb_buf = rgb_to_argb(rgb, w, h)
        rgb888_buf = rgb.tobytes()
        mjpeg_buf = rgb_to_mjpeg(rgb, w, h)

        cases = [
            (FMT["YUYV"], "YUYV", yuyv_buf),
            (FMT["NV21"], "NV21", nv21_buf),
            (FMT["MJPEG"], "MJPEG", mjpeg_buf),
            (FMT["ARGB8888"], "ARGB", argb_buf),
            (FMT["RGB888"], "RGB888", rgb888_buf),
        ]

        for fmt_id, name, buf in cases:
            for gray in (False, True):
                total += 1
                ok, msg = run_one(lib, fmt_id, name, buf, w, h, gray)
                if not ok:
                    fails += 1
                    print(f"FAIL {os.path.basename(path)} {name} gray={int(gray)}: {msg}")

        # Y 平面闭环自检: YUYV 的 Y 字节(偶数位置) 应与编码时的 Y 参考一致
        yuyv_arr = np.frombuffer(yuyv_buf, dtype=np.uint8).reshape(h, w * 2)
        y_extracted = yuyv_arr[:, 0::2]
        if not np.array_equal(y_extracted, y_ref):
            y_mismatch += 1
            print(f"Y-MISMATCH {os.path.basename(path)}: YUYV Y != reference Y")
        else:
            y_checked += 1

    print(f"\n=== {fails}/{total} format-runs failed, {y_mismatch}/{y_checked} Y-plane checks mismatched ===")
    return 1 if (fails or y_mismatch) else 0


if __name__ == "__main__":
    sys.exit(main())
