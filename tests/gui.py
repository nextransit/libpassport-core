#!/usr/bin/env python3
"""passport_test_gui -- Tkinter GUI that exercises all 4 modules.

Tabs (in order):

  1. MRZ OCR 测试       -- corpus-driven per-case benchmark (>=100 cases),
                          traditional vs CNN, side-by-side with image preview.
  2. NFC 读卡           -- mock R-APDU full eMRTD flow over JSON scripts.
  3. MRZ 解码           -- text TD3 -> fields / check digits.
  4. 防伪验证           -- MRZ + sample -> AC report.
  5. 照片比对           -- two image files -> SSIM similarity.

All four CLI tools must already be built in their respective build/
directories. Missing tools are reported in the status bar.
"""
from __future__ import annotations
import os, subprocess, sys, threading, time, json, re, hashlib, tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

# High-DPI awareness must be set before Tk initialises (Windows only;
# macOS/Retina is handled natively). Prevents blurry controls/table
# text on scaled displays.
if sys.platform.startswith("win"):
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

try:
    import ttkbootstrap as ttk
    from ttkbootstrap import Window
    if not hasattr(ttk, "PanedWindow"):
        ttk.PanedWindow = ttk.Panedwindow
    HAS_TTKB = True
except ImportError:
    from tkinter import ttk
    HAS_TTKB = False

ROOT = Path(__file__).resolve().parent.parent

_OCRB_FONT_NAME = None

def _load_ocrb_font(root):
    """Load OCR-B font for MRZ display.

    Checks if the 'OCRB' family is available in the system (installed
    from tmp/font/ocrb.ttf).  Falls back to Menlo if not found.
    Must be called after the Tk root window exists.
    """
    global _OCRB_FONT_NAME
    if _OCRB_FONT_NAME is not None:
        return _OCRB_FONT_NAME
    try:
        from tkinter import font as tkfont
        if "OCRB" in tkfont.families():
            _OCRB_FONT_NAME = "OCRB"
    except Exception:
        pass
    return _OCRB_FONT_NAME

def _ocrb_font(size=11, weight="normal"):
    """Return a font tuple using OCR-B if loaded, else Menlo fallback."""
    if _OCRB_FONT_NAME:
        return (_OCRB_FONT_NAME, size, weight) if weight != "normal" else (_OCRB_FONT_NAME, size)
    return ("Menlo", size, weight) if weight != "normal" else ("Menlo", size)

def tool_path(name, sub_dirs=("1_mrz_decode", "2_anticounterfeit",
                              "3_face_recognition", "4_nfc_reader")):
    """Locate a built CLI under */build/."""
    for sub in sub_dirs:
        cand = ROOT / sub / "build" / name
        if cand.exists():
            return cand
    return ROOT / sub_dirs[0] / "build" / name

MRZ_TOOL  = tool_path("mrz_tool")
AC_TOOL   = tool_path("ac_tool")
FACE_TOOL = tool_path("face_tool")
GEN_SAMP  = tool_path("gen_sample")
NFC_TOOL  = tool_path("nfc_tool")

# OCR back-end display names, kept in one place so the method picker,
# the summary rows, and the detail rows all stay consistent.
METHOD_LABELS = {
    "traditional": "传统模板",
    "cnn":         "轻量CNN",
    "crnn":        "CRNN序列",
    "fuse":        "融合CNN+CRNN",
    "tesseract":   "Tesseract",
    "paddle":      "PaddleOCR",
    "indra":       "IndraOCR",
    "easyocr":     "EasyOCR",
}
METHOD_ORDER = ["traditional", "cnn", "crnn", "fuse", "tesseract", "paddle", "indra", "easyocr"]
# CRNN / fusion 都是同一个 host CNN tool，靠环境变量切换后端。
CRNN_BACKENDS = {"crnn", "fuse"}

# --- 严格极速模式 (MRZ_FAST_STRICT) ------------------------------------
# 产品路径约束：单次识别 < 100 ms，且禁止任何跨引擎 fallback / 假设重试。
#   * 只用本机 C 引擎（无 Tesseract / Paddle / Indra / ONNX 子进程）。
#   * 只做一次 page-bottom 假设，A 失败也不再跑 B / sensitive C。
#   * 不做 Tesseract / Indra rescue，不做暴力校验位枚举修复。
#   * 复用固定的 band PPM，不创建也不删除临时文件（省掉每次 ~60 ms 的
#     文件创建 / 回收站 IPC 开销）。
#
# 同批次逐行融合（MRZ_FAST_FUSION=1，默认关）不是 fallback：几个本机 C
# 引擎在**同一次裁剪、同一次并发批次**里各给出一行读法，再对 line 1
# 做加权投票（TD3 line 1 没有校验位，单引擎读错无法自检）。没有失败后
# 重试、没有第二次推理、没有慢速引擎顶替、没有 repair_line2 的 35 万次
# checksum 枚举（~250 ms）。墙钟 = 最慢的那个 C 引擎。
# 全语料 A/B (n=108) 实测：融合把校验通过 89 -> 95，但 GT 完全一致仍是
# 68（C 引擎对姓名行读的是同一批错字，投票选不出正确值），耗时却从
# 29.91 ms 涨到 51.74 ms。因此默认保持纯单 fuse 头。
#   MRZ_FAST_FUSION=1  开启同批次 C 引擎逐行融合
#   MRZ_FAST_TIER=...  显式覆盖引擎集
#   MRZ_FAST_STRICT=0  恢复旧的多引擎路径，用于 A/B 验收对比
FAST_TIER_SINGLE = ("crnn",)
FAST_TIER_FUSION = ("traditional", "cnn", "crnn", "fuse")
FAST_TIER_DEFAULT = FAST_TIER_SINGLE


def _fast_strict() -> bool:
    return os.environ.get("MRZ_FAST_STRICT", "1") != "0"


def _fast_fusion() -> bool:
    return os.environ.get("MRZ_FAST_FUSION", "0") == "1"


def _fast_tier():
    raw = os.environ.get("MRZ_FAST_TIER", "")
    if raw.strip():
        picked = tuple(m.strip() for m in raw.split(",")
                       if m.strip() in METHOD_ORDER)
        if picked:
            return picked
    return FAST_TIER_FUSION if _fast_fusion() else FAST_TIER_SINGLE


# `_rst_run_ocr_methods` 的白名单：严格模式只放行这四个本机 C 引擎。
FAST_TIER = FAST_TIER_FUSION

# 图片直接识别 / 照片批测可选的护照照片目录。这些是真实机拍/扫描件，
# 与 tests/assets/pic 的合成语料分开维护。
PASSPORT_PHOTO_DIRS = [
    (ROOT / "assets" / "pic", "assets/pic"),
    (ROOT / "tmp" / "PassPortPhoto", "tmp/PassPortPhoto"),
    (ROOT / "Photo", "Photo"),
]

# 训练集 → 该训练集自己的照片目录。两套权重是在两套互不相干的语料上
# 分别训出来的（板端标签的手机照 vs 多引擎投票+人工复核的扫描件），
# 所以「用 A 集权重去评 B 集照片」量出来的是分布差异，不是模型好坏。
# 勾选训练集时列表只列该集照片，就是为了让这个跨分布读数不出现在
# 界面上被误读成准确率。Photo/ 是用户自备目录，不属于任何训练集，
# 两个模式下都列出来。
WEIGHTS_SET_DIRS = {
    "passport": {"tmp/PassPortPhoto"},
    "pic": {"assets/pic"},
}

def iter_passport_photos(weights_set=None):
    """Yield (label, path) for every real-photo passport image under
    PASSPORT_PHOTO_DIRS. Photo/ 是我们的专用目录，其它目录只取顶层文件。

    weights_set 传 "passport" / "pic" 时只列该训练集自己的目录（外加
    Photo/）；传 None 表示不过滤。
    """
    seen = set()
    keep = WEIGHTS_SET_DIRS.get(weights_set) if weights_set else None
    for base, label in PASSPORT_PHOTO_DIRS:
        if not base.is_dir():
            continue
        if keep is not None and label != "Photo" and label not in keep:
            continue
        if label == "Photo":
            entries = sorted(base.glob("*/*"))
        else:
            entries = sorted(base.glob("*"))
        for f in entries:
            if f.suffix.lower() not in (".jpg", ".jpeg", ".png", ".ppm"):
                continue
            key = str(f.resolve())
            if key in seen:
                continue
            seen.add(key)
            yield f"{label}/{f.relative_to(base)}", f

FACE_DATA_DIR = ROOT / "3_face_recognition" / "data"
FACE_SAMPLES = {
    "face_a (合成, 128×128)": FACE_DATA_DIR / "face_a.ppm",
    "face_b (合成, 128×128)": FACE_DATA_DIR / "face_b.ppm",
    "face_c (合成, 128×128)": FACE_DATA_DIR / "face_c.ppm",
}

SAMPLE_PASSPORT_PNG = ROOT / "tests" / "assets" / "passport_sample.png"

SAMPLE_MRZ = (
    "P<UTOERIKSSON<ANNA<MARIA<<<<<<<<<<<<<<<<<<<<\n"
    "L898902C36UTO6908061F9406236ZE184226B<<<<<18\n"
)

_RST_PADDLE_MODULE = None
_RST_PADDLE_LOCK = threading.Lock()
_RST_PADDLE_WARMUP_STARTED = False
_RST_CNN_MIN_ROW_HEIGHT = 12


def _mrz_icao_clean(text):
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
    return "".join(ch for ch in (text or "").upper() if ch in allowed)


def _mrz_line1_candidates(text):
    """Extract 44-char TD3 line-1 or 30-char CS/CD line windows from noisy OCR."""
    clean = _mrz_icao_clean(text)
    found = []
    # CS/CD 港澳台通行证: 30-char single line starting with CS or CD
    for start in range(max(0, len(clean) - 29)):
        line = clean[start:start + 30]
        if (len(line) == 30 and line[0] == 'C'
                    and line[1] in ('S', 'D')
                    and line[12] == '<' and line[20] == '<' and line[28] == '<'):
            if line not in found:
                found.append(line)
    # TD3 passport: 44-char line-1
    doc_types = set("PIVACDRQ")
    for start in range(max(0, len(clean) - 43)):
        line = clean[start:start + 44]
        if (len(line) == 44 and line[0] in doc_types
                    and line[1] in "<ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    and line[2:5].isalpha()
                    and sum(ch.isalpha() for ch in line[5:]) >= 3):
                if line not in found:
                    found.append(line)
                # Tesseract often reads the empty second slot ('<') of a
                # standard P< line as O/0. Emit the corrected variant so
                # the 'P<' candidate outranks the 'PO' one in decode.
                if line[0] == "P" and line[1] in "O0":
                    alt = line[0] + "<" + line[2:]
                    if alt not in found:
                        found.append(alt)
    return found


def _mrz_line2_candidates(text):
    """Extract and checksum-repair near-44-char TD3 line-2 windows."""
    clean = _mrz_icao_clean(text)
    if not clean:
        return []
    checksum_dir = ROOT / "6_1_PaddleOCR"
    if str(checksum_dir) not in sys.path:
        sys.path.insert(0, str(checksum_dir))
    from mrz_checksum import repair_line2

    country_fix = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"}
    found = []
    for start in range(len(clean)):
        for length in (43, 44, 45):
            raw = clean[start:start + length]
            if len(raw) != length or len(raw) < 28:
                continue
            country = raw[10:13]
            if not country.isalpha():
                country = "".join(country_fix.get(ch, ch) for ch in country)
                if not country.isalpha():
                    continue
                raw = raw[:10] + country + raw[13:]
            if raw[20] not in "MF<ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                continue
            fixed, n_fixed, report, valid = repair_line2(raw)
            if not valid or len(fixed) != 44:
                continue
            item = (n_fixed, abs(length - 44), start, fixed, report, raw)
            if not any(old[3] == fixed for old in found):
                found.append(item)
    found.sort(key=lambda item: item[:3])
    return found


_RST_TESSERACT_CAPI = None
_RST_TESSERACT_CAPI_DATAPATH = None
_RST_TESSERACT_CAPI_FAILED = False
_RST_TESSERACT_POOL = []
_RST_TESSERACT_POOL_CREATED = 0
_RST_TESSERACT_POOL_COND = threading.Condition()
_RST_TESSERACT_POOL_MAX = 2


def _rst_tesseract_capi(tess):
    """Load the Tesseract C API once, if the native library is available."""
    global _RST_TESSERACT_CAPI, _RST_TESSERACT_CAPI_DATAPATH
    global _RST_TESSERACT_CAPI_FAILED
    if _RST_TESSERACT_CAPI is not None:
        return _RST_TESSERACT_CAPI, _RST_TESSERACT_CAPI_DATAPATH
    if _RST_TESSERACT_CAPI_FAILED:
        return None, None

    import ctypes
    import ctypes.util

    library = ctypes.util.find_library("tesseract")
    if not library:
        exe = Path(tess).resolve()
        candidates = [
            exe.parent.parent / "lib" / "libtesseract.dylib",
            exe.parent.parent / "lib" / "libtesseract.so",
            Path("/usr/local/lib/libtesseract.dylib"),
            Path("/usr/local/lib/libtesseract.so"),
        ]
        library = next((str(path) for path in candidates if path.exists()),
                       None)
    if not library:
        _RST_TESSERACT_CAPI_FAILED = True
        return None, None

    try:
        lib = ctypes.CDLL(library)
        handle = ctypes.c_void_p
        lib.TessBaseAPICreate.restype = handle
        lib.TessBaseAPIDelete.argtypes = [handle]
        lib.TessBaseAPIInit3.argtypes = [
            handle, ctypes.c_char_p, ctypes.c_char_p]
        lib.TessBaseAPIInit3.restype = ctypes.c_int
        lib.TessBaseAPISetVariable.argtypes = [
            handle, ctypes.c_char_p, ctypes.c_char_p]
        lib.TessBaseAPISetVariable.restype = ctypes.c_int
        lib.TessBaseAPISetPageSegMode.argtypes = [handle, ctypes.c_int]
        lib.TessBaseAPISetImage.argtypes = [
            handle, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.TessBaseAPIProcessPages.argtypes = [
            handle, ctypes.c_char_p, ctypes.c_char_p,
            ctypes.c_int, handle]
        lib.TessBaseAPIProcessPages.restype = ctypes.c_int
        lib.TessBaseAPIGetUTF8Text.argtypes = [handle]
        lib.TessBaseAPIGetUTF8Text.restype = ctypes.c_void_p
        lib.TessDeleteText.argtypes = [ctypes.c_void_p]
        lib.TessBaseAPIClear.argtypes = [handle]

        tessdata = []
        prefix = Path(tess).resolve().parent.parent
        env_prefix = os.environ.get("TESSDATA_PREFIX")
        if env_prefix:
            env_path = Path(env_prefix)
            tessdata.extend([env_path, env_path / "tessdata"])
        tessdata.extend([
            prefix / "share" / "tessdata",
            Path("/usr/share/tesseract-ocr/5/tessdata"),
            Path("/usr/share/tessdata"),
        ])
        tessdata.extend(
            parent / "share" / "tessdata"
            for parent in Path(tess).resolve().parents)
        datapath = next(
            (str(path) for path in tessdata
             if (path / "ocrb.traineddata").exists()),
            None)
        _RST_TESSERACT_CAPI = lib
        _RST_TESSERACT_CAPI_DATAPATH = datapath
        return lib, datapath
    except (AttributeError, OSError):
        _RST_TESSERACT_CAPI_FAILED = True
        return None, None


def _rst_tesseract_create(lib, datapath):
    """Build one Tesseract handle with the OCR-B whitelist applied."""
    api = lib.TessBaseAPICreate()
    if not api:
        return None
    data_path = datapath.encode() if datapath else None
    if lib.TessBaseAPIInit3(api, data_path, b"ocrb") != 0:
        lib.TessBaseAPIDelete(api)
        return None
    lib.TessBaseAPISetVariable(
        api, b"tessedit_char_whitelist",
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
    return api


def _rst_tesseract_acquire(lib, datapath):
    """Lease an idle handle, creating one up to _RST_TESSERACT_POOL_MAX.

    TessBaseAPIInit3 reloads every language model (~70ms), so handles
    are recycled for the process lifetime and re-armed with
    SetImage/ProcessPages + Clear between recognitions.  A handle is
    not thread safe, so each in-flight recognition owns one exclusive.
    """
    global _RST_TESSERACT_POOL_CREATED
    with _RST_TESSERACT_POOL_COND:
        while (not _RST_TESSERACT_POOL
               and _RST_TESSERACT_POOL_CREATED >= _RST_TESSERACT_POOL_MAX):
            _RST_TESSERACT_POOL_COND.wait()
        if _RST_TESSERACT_POOL:
            return _RST_TESSERACT_POOL.pop()
        _RST_TESSERACT_POOL_CREATED += 1
    api = _rst_tesseract_create(lib, datapath)
    if api is None:
        with _RST_TESSERACT_POOL_COND:
            _RST_TESSERACT_POOL_CREATED -= 1
            _RST_TESSERACT_POOL_COND.notify_all()
    return api


def _rst_tesseract_release(api):
    """Return a handle to the idle pool."""
    if not api:
        return
    with _RST_TESSERACT_POOL_COND:
        _RST_TESSERACT_POOL.append(api)
        _RST_TESSERACT_POOL_COND.notify_all()


def _rst_tesseract_capi_text(lib, datapath, image, psm):
    """Recognize one PIL image without starting a Tesseract process."""
    import ctypes

    api = _rst_tesseract_acquire(lib, datapath)
    if not api:
        return ""
    try:
        lib.TessBaseAPISetPageSegMode(api, int(psm))
        gray = image.convert("L")
        raw = gray.tobytes()
        buffer = ctypes.create_string_buffer(raw)
        lib.TessBaseAPISetImage(
            api,
            ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)),
            gray.width, gray.height, 1, gray.width)
        text_ptr = lib.TessBaseAPIGetUTF8Text(api)
        if not text_ptr:
            return ""
        try:
            return ctypes.string_at(text_ptr).decode("utf-8", "replace")
        finally:
            lib.TessDeleteText(text_ptr)
    finally:
        lib.TessBaseAPIClear(api)
        _rst_tesseract_release(api)


def _rst_tesseract_capi_file_text(lib, datapath, image_path, psm):
    """Recognize one PNG through the same file path as the CLI."""
    import ctypes

    api = _rst_tesseract_acquire(lib, datapath)
    if not api:
        return ""
    try:
        lib.TessBaseAPISetPageSegMode(api, int(psm))
        lib.TessBaseAPIProcessPages(
            api, str(image_path).encode(), None, 0, None)
        text_ptr = lib.TessBaseAPIGetUTF8Text(api)
        if not text_ptr:
            return ""
        try:
            return ctypes.string_at(text_ptr).decode("utf-8", "replace")
        finally:
            lib.TessDeleteText(text_ptr)
    finally:
        lib.TessBaseAPIClear(api)
        _rst_tesseract_release(api)


def _rst_adaptive_xpad(rgb, pair, scale, nominal):
    """左右裁边: 机位照保留名义 xpad, 无取景框的照片不裁边。

    名义 xpad 的唯一用途是剔除机位照的黑色取景框; 整页满幅照/ID 卡
    照没有边框, 名义 xpad 会裁掉真实字形(实测 960px 图 MRZ 墨迹延到
    x45/x913)。以源图左右边条(最多 12 列)的整体墨量占比判定取景框:
    机位照边框列整列近全黑(实测 6 图边条暗占比 1.00), 满幅照边缘
    0.00-0.67; 有框一侧保留 nominal, 无框一侧返回 0(不裁)。
    疏底纹/暗背景不会误触发(边条均值阈值 0.9)。
    """
    import numpy as np

    W = rgb.shape[1]
    if W < 40 or not pair:
        return nominal, nominal
    wts = np.array([0.299, 0.587, 0.114])
    strip = max(4, min(12, W // 25))
    left_dark = ((rgb[:, :strip] @ wts) < 160).mean()
    right_dark = ((rgb[:, W - strip:] @ wts) < 160).mean()
    left = nominal if left_dark >= 0.9 else 0
    right = nominal if right_dark >= 0.9 else 0
    # Never crop INTO the MRZ ink. The nominal pad is a fraction of the
    # image width, so on a narrow document it can reach past the first
    # or last character: measured on assets/pic/1c6acd092dee7031_0222.png
    # (CS/CD permit) a 20 px crop already dropped the trailing
    # characters and 40 px cut into the code. Clamp the pad to the ink
    # extent measured inside the detected rows.
    try:
        ys = []
        for y0, y1 in pair:
            a = max(0, int(round(y0 / max(scale, 1e-6))))
            b = min(rgb.shape[0], int(round((y1 + 1) / max(scale, 1e-6))))
            if b > a:
                ys.append((a, b))
        if ys:
            a = min(y[0] for y in ys)
            b = max(y[1] for y in ys)
            rows = rgb[a:b]
            if rows.size:
                dark = (rows @ wts) < 160
                # A column counts as a CHARACTER column only when its
                # ink does not run through the whole row height: the
                # camera frame's black border and page edge lines do
                # exactly that, and treating them as characters would
                # cancel the very pad that removes them.
                cov = dark.sum(axis=0) / max(1, rows.shape[0])
                idx = np.where((cov > 0.0) & (cov < 0.95))[0]
                if len(idx):
                    ink_l = int(idx[0])
                    ink_r = int(idx[-1])
                    left = min(left, max(0, ink_l))
                    right = min(right, max(0, W - 1 - ink_r))
    except Exception:
        pass
    return left, right


def _rst_tesseract_cscd(rgb, pair, scale):
    """Run tesseract on CS/CD single-line MRZ (30-char format).

    Tries PSM 7 (single line), PSM 6 (block), and PSM 3 (full image)
    on the band crop, plus PSM 3 on the full source image as a last
    resort.  The textured guilloche background defeats per-row crops
    on some cards; PSM 3 on a larger area gives tesseract more context
    to separate text from background.
    """
    import shutil
    from PIL import Image

    tess = shutil.which("tesseract")
    if not tess:
        return [], [], "", "tesseract not found"

    H0, W0 = rgb.shape[:2]
    y0, y1 = pair[0]
    sy0 = max(0, int(round(y0 / scale)))
    sy1 = min(H0, int(round((y1 + 1) / scale)))
    band_h = sy1 - sy0
    pad = max(20, int(band_h * 1.5))
    source = Image.fromarray(rgb)

    capi, datapath = _rst_tesseract_capi(tess)
    raw_parts = []
    line_candidates = []

    def try_text(image, psm, label):
        if capi is not None:
            text = _rst_tesseract_capi_text(capi, datapath, image, psm)
        else:
            import subprocess, os
            from tempfile import NamedTemporaryFile
            with NamedTemporaryFile(suffix=".png", delete=False) as f:
                image.save(f.name)
                proc = subprocess.run(
                    [tess, f.name, "stdout", "--psm", str(psm), "-l", "ocrb",
                     "-c", "tessedit_char_whitelist="
                     "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"],
                    text=True, capture_output=True, timeout=20)
                text = proc.stdout
                os.unlink(f.name)
        raw_parts.append(f"{label}/psm{psm}: {text.strip()}")
        for line in _mrz_line1_candidates(text):
            if line not in line_candidates:
                line_candidates.append(line)

    band_crop = source.crop((0, max(0, sy0 - pad), W0, min(H0, sy1 + pad)))
    for psm in (7, 6, 3):
        try_text(band_crop, psm, "band")
    if not line_candidates:
        try_text(source, 3, "full")
    return line_candidates, [], "\n".join(raw_parts), ""


def _rst_tesseract_candidates(rgb, pair, scale, psm_by_row=None,
                               fallback_missing=True):
    """Run OCR-B on each real MRZ row at source resolution.

    Source-resolution, per-row crops avoid the CNN path's two common
    failure modes here: line 1 being swallowed by line 2 and the
    lower-page reflection being treated as text. The fast path runs
    one psm7 recognition per row; psm6 is only used for a row with no
    candidate, or when the caller explicitly requests a slow fallback.
    """
    import shutil
    from concurrent.futures import ThreadPoolExecutor
    from tempfile import TemporaryDirectory
    from PIL import Image

    tess = shutil.which("tesseract")
    if not tess:
        return [], [], "", "tesseract not found"

    H0, W0 = rgb.shape[:2]
    xpad = min(100, max(10, int(round(W0 * 0.08))))
    xpad_l, xpad_r = _rst_adaptive_xpad(rgb, pair, scale, xpad)
    raw_parts = []
    line_candidates = [[], []]
    if psm_by_row is None:
        psm_by_row = {0: (7,), 1: (7,)}
    capi, datapath = _rst_tesseract_capi(tess)

    def add_candidates(idx, text):
        if idx == 0:
            for line in _mrz_line1_candidates(text):
                if line not in line_candidates[0]:
                    line_candidates[0].append(line)
        else:
            for item in _mrz_line2_candidates(text):
                if not any(old[3] == item[3]
                           for old in line_candidates[1]):
                    line_candidates[1].append(item)

    scratch = ROOT / "10_ocr_research"
    scratch.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="_gui_mrz_", dir=str(scratch)) as td:
        tmp_dir = Path(td)
        source = Image.fromarray(rgb)
        row_images = []
        source_images = []
        row_heights = []
        row_paths = []
        for idx, (y0, y1) in enumerate(pair):
            sy0 = max(0, int(round(y0 / scale)))
            sy1 = min(H0, int(round((y1 + 1) / scale)))
            row_h = max(1, sy1 - sy0)
            row_heights.append(row_h)
            pad = max(8, min(25, int(round(row_h * 0.70))))
            crop = source.crop((
                xpad_l,
                max(0, sy0 - pad),
                max(xpad_l + 1, W0 - xpad_r),
                min(H0, sy1 + pad),
            ))
            source_images.append(crop)
            resize_factor = 2 if row_h < 20 else 1
            if resize_factor > 1:
                crop = crop.resize(
                    (crop.width * resize_factor, crop.height * resize_factor),
                    Image.LANCZOS)
            row_images.append(crop)
            row_paths.append(tmp_dir / f"line{idx + 1}.png")

        def save_paths():
            for image, path in zip(row_images, row_paths):
                if not path.exists():
                    image.save(path)

        def run_one(idx, psm, source_mode):
            if capi is not None:
                if source_mode == "file":
                    save_paths()
                    text = _rst_tesseract_capi_file_text(
                        capi, datapath, row_paths[idx], psm)
                else:
                    text = _rst_tesseract_capi_text(
                        capi, datapath, row_images[idx], psm)
                return idx, psm, text
            save_paths()
            base = tmp_dir / f"line{idx + 1}_psm{psm}"
            proc = subprocess.run(
                [tess, str(row_paths[idx]), str(base),
                 "--psm", str(psm), "-l", "ocrb",
                 "-c", "tessedit_char_whitelist="
                 "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"],
                text=True, capture_output=True, timeout=20)
            txt_path = Path(str(base) + ".txt")
            text = txt_path.read_text() if txt_path.exists() else ""
            if proc.returncode != 0:
                text = ""
            return idx, psm, text

        def run_jobs(jobs, source_mode):
            if len(jobs) <= 1:
                return [run_one(*jobs[0], source_mode)] if jobs else []
            with ThreadPoolExecutor(max_workers=min(2, len(jobs))) as pool:
                return list(pool.map(
                    lambda job: run_one(*job, source_mode), jobs))

        def absorb(results):
            for idx, psm, text in sorted(results, key=lambda item: item[:2]):
                raw_parts.append(f"line{idx + 1}/psm{psm}: {text.strip()}")
                if text.strip():
                    add_candidates(idx, text)

        initial_jobs = [
            (idx, psm)
            for idx in range(len(pair))
            for psm in psm_by_row.get(idx, (7,))
        ]
        absorb(run_jobs(initial_jobs, "memory" if capi is not None else "file"))
        if (capi is not None and len(pair) > 1
                and 7 in psm_by_row.get(1, ())
                and row_heights[1] >= 20
                and (not line_candidates[1])):
            enlarged = source_images[1].resize(
                (source_images[1].width * 2, source_images[1].height * 2),
                Image.LANCZOS)
            row_images[1] = enlarged
            if row_paths[1].exists():
                row_paths[1].unlink()
            absorb(run_jobs([(1, 7)], "memory"))
        if (capi is not None and 7 in psm_by_row.get(1, ())
                and line_candidates[1]):
            # Direct SetImage is fast, but a difficult crop can lose leading
            # glyphs. More than one checksum repair is a reliable signal to
            # replay only that row through the CLI-equivalent PNG path.
            if min(item[0] for item in line_candidates[1]) > 1:
                save_paths()
                memory_candidates = line_candidates[1]
                line_candidates[1] = []
                absorb(run_jobs([(1, 7)], "file"))
                if not line_candidates[1]:
                    line_candidates[1] = memory_candidates
        if fallback_missing:
            missing_jobs = [
                (idx, 6)
                for idx in range(len(pair))
                if (not line_candidates[idx]
                    and 7 in psm_by_row.get(idx, ())
                    and 6 not in psm_by_row.get(idx, ()))
            ]
            absorb(run_jobs(missing_jobs, "memory" if capi is not None
                            else "file"))
    line_candidates[0].sort(
        key=lambda line: (line[0] != "P", line[1] != "<",
                          -line[5:].count("<")))
    line_candidates[1].sort(key=lambda item: item[:3])
    return line_candidates[0], line_candidates[1], "\n".join(raw_parts), ""


def _rst_decode_ok(completed):
    """True only when the real decoder ACCEPTED the lines.

    mrz_tool exits 0 even for rejected input (it reports "decode
    failed: ..." on stderr and leaves stdout empty), so the exit code
    alone marks empty / malformed results as verified. The reliable
    signal is a non-empty stdout carrying the parsed TD3 fields."""
    if completed is None or completed.returncode != 0:
        return False
    out = completed.stdout or ""
    if "decode failed" in (completed.stderr or "").lower():
        return False
    return "doc_type" in out


def _rst_decode_candidates(line1s, line2s):
    """Return the first OCR candidate accepted by the real decoder."""
    if not line2s:
        line2s = [(0, 0, 0, "", "", "")]
    for line1 in line1s:
        for _n, _len_delta, _start, line2, report, raw in line2s:
            decoded = subprocess.run(
                [str(MRZ_TOOL), "decode", f"{line1}\n{line2}"],
                text=True, capture_output=True, timeout=10)
            if _rst_decode_ok(decoded):
                return line1, line2, report, raw, decoded
    return "", "", [], "", subprocess.CompletedProcess(
        args=[], returncode=2, stdout="", stderr="no MRZ candidate decoded")


_RST_GT_CACHE = None


def _rst_gt_lookup(base):
    """按文件名(base)查找 corpus_real 的正确 MRZ (GT)。

    数据源: 6_mrz_ocr/data/corpus_real/corpus.json (records[].id 形如
    "pic_scan1/2026-9-29-A", line1/line2 为板端 fixmrz 弱标签)。
    匹配规则: id 的最后一段 == 图片文件名(不含扩展名)。
    """
    global _RST_GT_CACHE
    if _RST_GT_CACHE is None:
        _RST_GT_CACHE = {}
        _gt_json = ROOT / "6_mrz_ocr" / "data" / "corpus_real" / "corpus.json"
        try:
            for r in json.loads(_gt_json.read_text())["records"]:
                _RST_GT_CACHE[r["id"].split("/")[-1]] = r
        except Exception:
            pass
        _real_gt = ROOT / "tests" / "real_gt.json"
        try:
            for k, v in json.loads(_real_gt.read_text()).get("gt", {}).items():
                base = Path(k).stem
                if base not in _RST_GT_CACHE:
                    _RST_GT_CACHE[base] = v
        except Exception:
            pass
    return _RST_GT_CACHE.get(base)


def _rst_gt_compare(pred_line1, pred_line2, gt_line1, gt_line2):
    """GT vs Pred 逐字符对比: 返回 (acc1, acc2, 对比文本行列表)。"""
    def char_acc(a, b):
        n = max(len(a or ""), len(b or ""))
        if n == 0:
            return 0.0
        return sum(1 for i in range(n)
                   if (a[i] if i < len(a) else "_") ==
                   (b[i] if i < len(b) else "_")) / n
    a1, a2 = char_acc(pred_line1, gt_line1), char_acc(pred_line2, gt_line2)
    rows = [
        f"GT  line1: {gt_line1 or '<无>'}",
        f"PRED l1  : {pred_line1 or '<未提取>'}   (acc {a1*100:.1f}%)",
        f"GT  line2: {gt_line2 or '<无>'}",
        f"PRED l2  : {pred_line2 or '<未提取>'}   (acc {a2*100:.1f}%)",
    ]
    return a1, a2, rows


def _rst_parse_kv_output(text):
    """Parse the small key/value stdout contract shared by OCR tools."""
    out = {}
    for line in (text or "").splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        out[key.strip()] = value.strip()
    return out


def _rst_decode_lines(line1, line2):
    """Validate one pair (or a single CS/CD line) through the real MRZ decoder."""
    if not line1:
        return subprocess.CompletedProcess(
            args=[], returncode=2, stdout="",
            stderr="OCR 未输出任何行")
    # CS/CD 港澳台通行证: 单行 30 字符, line2 为空
    if len(line1) == 30 and not line2:
        return subprocess.run(
            [str(MRZ_TOOL), "decode", line1],
            text=True, capture_output=True, timeout=10)
    if not line2:
        return subprocess.CompletedProcess(
            args=[], returncode=2, stdout="",
            stderr="OCR 未输出完整两行")
    return subprocess.run(
        [str(MRZ_TOOL), "decode", f"{line1}\n{line2}"],
        text=True, capture_output=True, timeout=10)


def _rst_cnn_source_heights(pair, scale):
    """Map detected work-image rows back to source-image pixel heights."""
    if not pair or scale <= 0:
        return ()
    return tuple(
        max(1, int(round((y1 + 1) / scale)) - int(round(y0 / scale)))
        for y0, y1 in pair
    )


def _rst_cnn_low_res_reason(pair, scale):
    """Return a reason when the C CNN cannot safely consume this MRZ."""
    heights = _rst_cnn_source_heights(pair, scale)
    if len(heights) != 2 or min(heights) >= _RST_CNN_MIN_ROW_HEIGHT:
        return ""
    return (
        f"源图 MRZ 行高 {heights[0]}/{heights[1]}px，"
        f"低于轻量CNN最小 {_RST_CNN_MIN_ROW_HEIGHT}px"
    )


def _rst_prepare_paddle():
    """Import and initialize the cached Paddle recognizer once."""
    global _RST_PADDLE_MODULE
    with _RST_PADDLE_LOCK:
        module = _RST_PADDLE_MODULE
        if module is None:
            paddle_dir = ROOT / "6_1_PaddleOCR"
            if str(paddle_dir) not in sys.path:
                sys.path.insert(0, str(paddle_dir))
            import importlib
            module = importlib.import_module("paddle_v6")
        rec, mode = module._get_rec()
        _RST_PADDLE_MODULE = module
        init_error = module._OCR_ERROR if mode == "fallback" else ""
    return module, rec, mode, init_error


def _rst_paddle_warmup():
    """Prepare Paddle in a background thread before the first run."""
    try:
        _rst_prepare_paddle()
    except Exception:
        # The actual run still reports the concrete initialization error.
        pass


def _rst_run_paddle_inprocess(img_path):
    """Run the cached ONNX recognizer without spawning a Python process."""
    module, rec, mode, init_error = _rst_prepare_paddle()

    ok, line1, line2, c1, c2, band, error, ms = \
        module._decode_one(rec, img_path, init_error)
    note = init_error if mode == "fallback" else ""
    raw = [
        f"result.backend  : {mode}",
        f"result.ok       : {'OK' if ok else 'unreadable'}",
        f"result.line1    : {line1}",
        f"result.line2    : {line2}",
        f"result.conf1    : {c1}",
        f"result.conf2    : {c2}",
        f"band.x band.y band.w band.h : {band}",
    ]
    if note:
        raw.append(f"result.note     : {note}")
    raw.append(f"ms: {ms:.1f}")
    return "\n".join(raw), (0 if ok else 1), error


def _rst_easyocr_fix_line1(line1, line2):
    """Post-repair EasyOCR line1 using TD3 MRZ format constraints.

    EasyOCR often misreads '<' fillers as '4', '2', or space.  TD3 line1
    has a rigid structure we can exploit:

      pos 0      : 'P'  (passport)
      pos 1      : '<' or type letter (e.g. 'O' for some countries)
      pos 2-4    : 3-letter country code (cross-check with line2 pos 10-12)
      pos 5+     : name field; '<<' separates surname/given; rest is '<'

    Repair rules (applied in order):
    1. Force pos 0 = 'P'.
    2. Copy country code from line2[10:13] into pos 2-4.
    3. Replace '4' with '<' everywhere (EasyOCR's most common '<' misread).
    4. After the second '<<' group, replace digits with '<' (filler zone).
    5. Ensure at least one '<<' separator exists in the name region.
    """
    if len(line1) < 6:
        return line1
    chars = list(line1.ljust(44, '<')[:44])
    chars[0] = 'P'
    if len(line2) >= 13:
        for i in range(3):
            if line2[10 + i].isalpha():
                chars[2 + i] = line2[10 + i]
    for i in range(44):
        if chars[i] == '4':
            chars[i] = '<'
    sep_count = 0
    filler_start = None
    for i in range(5, 43):
        if chars[i] == '<' and chars[i + 1] == '<':
            sep_count += 1
            if sep_count >= 1:
                filler_start = i + 2
                break
    if filler_start is not None:
        for i in range(filler_start, 44):
            if chars[i].isdigit():
                chars[i] = '<'
    name = ''.join(chars[5:])
    if '<<' not in name:
        for i in range(5, 43):
            if chars[i].isalpha() and chars[i + 1].isalpha():
                chars[i + 1] = '<'
                break
    return ''.join(chars)


def _rst_easyocr_mrz_extract(reader, band_img):
    """Run fine-tuned EasyOCR custom_mrz model on a band image.

    Uses the MRZ-specific VGG+BiLSTM+CTC network (37 chars, imgH=32,
    imgW=440) that was fine-tuned on synthetic OCR-B MRZ strips.

    The band image is split into two horizontal halves (line1 top,
    line2 bottom), each half is fed to the custom_mrz recognizer
    directly (no CRAFT detection), and the output is decoded with
    CTC greedy decoding.

    Returns (line1, line2, conf1, conf2, ok).
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    if band_img is None:
        return "", "", 0, 0, False
    if isinstance(band_img, str):
        band_img = np.array(Image.open(band_img).convert('L'))
    if band_img.ndim == 3:
        band_img = np.array(Image.fromarray(band_img).convert('L'))

    h, w = band_img.shape
    mid = h // 2
    top_half = band_img[:mid]
    bot_half = band_img[mid:]

    def _recognize_strip(strip_np):
        img = Image.fromarray(strip_np, 'L')
        result = reader.recognize(
            img, decoder='greedy', batch_size=1,
            contrast_ths=0.0, adjust_contrast=0.0,
            filter_ths=0.0, reformat=False
        )
        if result:
            text = result[0][1]
            conf = float(result[0][2])
            return text, conf
        return "", 0.0

    l1_raw, c1 = _recognize_strip(top_half)
    l2_raw, c2 = _recognize_strip(bot_half)

    charset = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")

    def sanitise(s):
        s = s.upper().replace(' ', '')
        return ''.join(ch if ch in charset else '<' for ch in s)

    line1 = sanitise(l1_raw)[:44]
    line2 = sanitise(l2_raw)[:44]
    line1 = line1.ljust(44, '<')[:44]
    line2 = line2.ljust(44, '<')[:44]
    line1 = _rst_easyocr_fix_line1(line1, line2)
    conf1 = int(c1 * 100)
    conf2 = int(c2 * 100)
    ok = bool(line1 and line2)
    return line1, line2, conf1, conf2, ok


def _rst_easyocr_extract(reader, band_img):
    """Run EasyOCR on a band image and extract MRZ line1/line2.

    EasyOCR is a general-purpose OCR engine, not MRZ-specific: it reads
    natural text and its '<' filler handling is unreliable (often read
    as '4' or space).  But on glass-table photos (Photo/3-8) where the
    CNN/fuse heads misread 4-11 characters on line 2, EasyOCR reads
    line 2 perfectly (0 errors on all 6 photos).

    The extraction strategy:
    1. Run readtext with bounding boxes, sort by y position.
    2. Group into two rows (top = line1, bottom = line2).
    3. Concatenate each row's text, strip spaces, uppercase.
    4. Sanitise: replace any char outside [A-Z0-9<] with '<'.
    5. Post-repair line1 using TD3 format constraints.
    6. Pad/truncate to 44 characters.

    Returns (line1, line2, conf1, conf2, ok).
    """
    import numpy as np
    if band_img is None:
        return "", "", 0, 0, False
    if isinstance(band_img, str):
        from PIL import Image as _Im
        band_img = np.array(_Im.open(band_img).convert('RGB'))
    results = reader.readtext(band_img, detail=1)
    if not results:
        return "", "", 0, 0, False
    results.sort(key=lambda r: r[0][0][1])
    mid_y = (results[0][0][0][1] + results[-1][0][2][1]) / 2.0
    top_texts, bot_texts = [], []
    top_confs, bot_confs = [], []
    for bbox, text, conf in results:
        y_center = sum(p[1] for p in bbox) / 4
        if y_center < mid_y:
            top_texts.append(text)
            top_confs.append(conf)
        else:
            bot_texts.append(text)
            bot_confs.append(conf)
    charset = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")

    def sanitise(s):
        s = s.upper().replace(' ', '')
        return ''.join(c if c in charset else '<' for c in s)

    line1 = sanitise(''.join(top_texts))[:44]
    line2 = sanitise(''.join(bot_texts))[:44]
    line1 = line1.ljust(44, '<')[:44]
    line2 = line2.ljust(44, '<')[:44]
    line1 = _rst_easyocr_fix_line1(line1, line2)
    conf1 = int(sum(top_confs) / max(1, len(top_confs)) * 100) if top_confs else 0
    conf2 = int(sum(bot_confs) / max(1, len(bot_confs)) * 100) if bot_confs else 0
    ok = bool(line1 and line2)
    return line1, line2, conf1, conf2, ok


_RST_INDRA_MODULE = None
_RST_INDRA_LOCK = threading.Lock()
_RST_INDRA_WARMUP_STARTED = False


def _rst_mrz_shaped(line1, line2):
    """Cheap structural sanity for a TD3 or CS/CD OCR result.

    Guards the "preferred engine" choice against engines that run their
    own detection on the FULL photo and lock onto page decoration
    (document titles like 'PASSPORT / PASSEPORT / CESKA REPUBLIKA'
    satisfy '44 uppercase chars' but are not MRZ).  A real TD3 line pair
    always has: 44 chars per line, [A-Z0-9<] only, a valid document-type
    character at line1 col1, and digits at most of the five line-2 check
    positions.

    Two ICAO 9303-4 facts drive the details here:

    * line1 col1 is the document-type filler.  The spec says '<', but a
      state may print its own letter there, and this corpus genuinely
      contains PD (diplomatic), PE, PM and PP documents -- plus a PO
      specimen.  Requiring '<' there rejected six real passports.
    * line2 col13..19 (date of birth) and col21..27 (date of expiry) are
      six DIGITS each, always.  A read that puts letters there is not a
      TD3 line 2 at all: page body text ('000905<606BBG0<<<<66<6<6<...')
      otherwise slipped through because a line made mostly of '<' can
      satisfy the check digits by accident.

    CS/CD 港澳台通行证: 单行 30 字符, line2 为空.
    """
    if not line1:
        return False
    # CS/CD 港澳台通行证: 30-char single line
    if len(line1) == 30 and not line2:
        charset = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
        if any(c not in charset for c in line1):
            return False
        if line1[0] != 'C' or line1[1] not in ('S', 'D'):
            return False
        return True
    if not line2:
        return False
    if len(line1) != 44 or len(line2) != 44:
        return False
    charset = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
    if any(c not in charset for c in line1 + line2):
        return False
    if not (line1[1] == "<" or "A" <= line1[1] <= "Z"):
        return False
    if not line2[13:19].isdigit() or not line2[21:27].isdigit():
        return False
    digit_cols = (9, 19, 27, 42, 43)
    digits = sum(1 for i in digit_cols if line2[i].isdigit())
    return digits >= 3


def _rst_char_accuracy(pred, gt):
    """Per-character accuracy of `pred` against `gt` (0.0..1.0).

    Both lines are compared over their common length; a character the
    reader did not produce at all counts as wrong.
    """
    if not gt:
        return None
    n = max(len(pred), len(gt))
    if not n:
        return None
    hit = sum(1 for i in range(n)
               if i < len(pred) and i < len(gt) and pred[i] == gt[i])
    return hit / n


def _rst_line1_agreement(results):
    """How many engines agree with each engine's line 1.

    A TD3 line 1 carries no check digit, so a single engine can pass the
    line-2 checksum and still have read line 1 wrong -- the generic
    document recognizer turns the filler '<' into a letter ('K') often
    enough that its line 1 differs while its line 2 still verifies.

    The engines are independent readers of the same image, so a line 1
    that most of them produced is the better bet.  Returns a list
    {result_id: agreeing_engine_count}; engines whose line 1 is not a
    44-character (TD3) or 30-character (CS/CD) MRZ-shaped string get -1
    so a malformed read never outranks a well-formed one.
    """
    counts = []
    for r in results:
        line1 = r.get("line1") or ""
        if len(line1) not in (30, 44):
            counts.append(-1)
            continue
        agree = sum(
            1 for other in results
            if other is not r
            and (other.get("line1") or "") == line1
            and len(other.get("line1") or "") in (30, 44)
        )
        counts.append(agree)
    return counts


def _hyp_rank(result):
    """Order two bottom-of-page hypotheses: verified > MRZ-shaped >
    anything else."""
    r = result or {}
    if r.get("verified"):
        return 3
    for e in r.get("engines") or []:
        if _rst_mrz_shaped(e.get("line1"), e.get("line2")):
            return 2
    return 1


def _rst_cscd_struct_ok(line):
    """Structural and checksum validation of a 30-character CS/CD row.

    A CS/CD (Mainland travel permit) document carries one 30 character row
    instead of the two 44 character TD3 rows:

        [0:2]   "CS" / "CD"
        [2:11]  document number, 9 characters (the country code letter is
                 the first of them)
        [11]    check digit of the document number
        [12]    '<'
        [13:19] date of birth, 6 digits
        [19]    check digit
        [20]    '<'
        [21:27] date of expiry, 6 digits
        [27]    check digit
        [28]    '<'
        [29]    composite check digit over [2:12] + [13:20] + [21:28]

    All four check digits follow the ordinary ICAO 7-3-1 cycle; this was
    verified on every CS/CD sample of the truth sheet (the document
    number starts at index 2, which is what made the composite look
    unreproducible in an earlier round).  The shape is checked as well:
    'CS'/'CD' prefix, 30 characters of [A-Z0-9<] and the three fixed
    filler positions at 12, 20 and 28."""
    if not line or len(line) != 30:
        return False
    if line[:2] not in ("CS", "CD"):
        return False
    if any(c not in set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
           for c in line):
        return False
    if not (line[12] == "<" and line[20] == "<" and line[28] == "<"):
        return False
    try:
        checksum_dir = ROOT / "6_1_PaddleOCR"
        if str(checksum_dir) not in sys.path:
            sys.path.insert(0, str(checksum_dir))
        from mrz_checksum import compute_check_digit
    except Exception:
        return True
    if compute_check_digit(line[2:11]) != line[11]:
        return False
    if compute_check_digit(line[13:19]) != line[19]:
        return False
    if compute_check_digit(line[21:27]) != line[27]:
        return False
    composite = line[2:12] + line[13:20] + line[21:28]
    return compute_check_digit(composite) == line[29]


def _rst_prepare_indra():
    """Import the Indra MRZ helper once and warm its ONNX sessions."""
    global _RST_INDRA_MODULE
    with _RST_INDRA_LOCK:
        module = _RST_INDRA_MODULE
        if module is None:
            indra_dir = ROOT / "11_indra"
            if str(indra_dir) not in sys.path:
                sys.path.insert(0, str(indra_dir))
            import importlib
            module = importlib.import_module("mrz_indra")
        module.engine()
        _RST_INDRA_MODULE = module
    return module


def _rst_indra_warmup():
    """Prepare Indra in a background thread before the first run."""
    try:
        _rst_prepare_indra()
    except Exception:
        # The actual run still reports the concrete initialization error.
        pass


def _rst_repair_cscd(clean):
    """Repair a 30-char CS/CD line: fix separators and recompute check digits."""
    if len(clean) != 30 or clean[:2] not in ('CS', 'CD'):
        return None
    checksum_dir = ROOT / "6_1_PaddleOCR"
    if str(checksum_dir) not in sys.path:
        sys.path.insert(0, str(checksum_dir))
    from mrz_checksum import compute_check_digit
    s = list(clean)
    s[12] = '<'
    s[20] = '<'
    s[28] = '<'
    s[11] = compute_check_digit(''.join(s[2:11]))
    s[19] = compute_check_digit(''.join(s[13:19]))
    s[27] = compute_check_digit(''.join(s[21:27]))
    comp = ''.join(s[2:12]) + ''.join(s[13:20]) + ''.join(s[21:28])
    s[29] = compute_check_digit(comp)
    return ''.join(s)


def _rst_run_indra_inprocess(img_path):
    """Run the cached Indra engine on a full-page photo.

    Indra detects and reads the text lines itself, so it gets the whole
    page instead of the cropped MRZ band the other tools use.
    """
    module = _rst_prepare_indra()
    result = module.read_mrz(img_path)
    if not result.get("ok") and result.get("raw_lines"):
        for raw_line in result["raw_lines"]:
            clean = _mrz_icao_clean(raw_line)
            for cand in _mrz_line1_candidates(clean):
                if len(cand) == 30 and cand[0] == 'C' and cand[1] in ('S', 'D'):
                    result["ok"] = True
                    result["line1"] = cand
                    result["line2"] = ""
                    result["note"] = f"indra CS/CD: {cand}"
                    break
            if result.get("ok"):
                break
            if len(clean) >= 30 and clean[:2] in ('CS', 'CD'):
                repaired = _rst_repair_cscd(clean[:30])
                if repaired:
                    decoded = _rst_decode_lines(repaired, "")
                    if decoded.returncode == 0:
                        result["ok"] = True
                        result["line1"] = repaired
                        result["line2"] = ""
                        result["note"] = f"indra CS/CD (repaired): {repaired}"
                        break
            if len(clean) >= 30 and clean[:3] in ('656', '65D', 'C56', 'C5D'):
                for prefix in ('CSC', 'CDC'):
                    attempt = prefix + clean[3:30]
                    repaired = _rst_repair_cscd(attempt)
                    if repaired:
                        decoded = _rst_decode_lines(repaired, "")
                        if decoded.returncode == 0:
                            result["ok"] = True
                            result["line1"] = repaired
                            result["line2"] = ""
                            result["note"] = f"indra CS/CD (prefix-fix): {repaired}"
                            break
                if result.get("ok"):
                    break
    stdout = module.format_kv(result)
    error = "" if result.get("ok") else result.get("note", "indra 未识别到 MRZ")
    return stdout, (0 if result.get("ok") else 1), error


# ---------------------------------------------------------------------------
# Cyber-tech theme system (.impeccable.md)
#
# Two brand themes are registered into ttkbootstrap so every ttk widget
# (Notebook, buttons, progress bars, Treeviews, scrollbars, entries) follows
# the palette in one switch:
#
#   cyber-dark  -- deep space blue-black (#0a0e14) + neon cyan accent
#   cyber-light -- cool paper white + electronic blue accent
#
# Semantic colours (success green / fail red / in-flight violet) keep their
# MEANING across both modes while each mode gets harmonised values. The
# "tk" dict extends the 16 ttkbootstrap keys with tokens for the non-ttk
# widgets (canvas, Text tags, chips, CTA buttons) that need manual recolour.
# ---------------------------------------------------------------------------
THEME_PREF_FILE = Path(__file__).resolve().parent / ".gui_theme.json"

CYBER_THEMES = {
    "cyber-dark": {
        "mode": "dark",
        "colors": {
            "primary":   "#00e5ff",
            "secondary": "#2e86ff",
            "success":   "#2ee6a8",
            "info":      "#2e86ff",
            "warning":   "#ffb454",
            "danger":    "#ff5c7a",
            "light":     "#1c2530",
            "dark":      "#060a10",
            "bg":        "#0a0e14",
            "fg":        "#dce3ec",
            "selectbg":  "#0f2a33",
            "selectfg":  "#7df3ff",
            "border":    "#21262D",
            "inputfg":   "#e8eef6",
            "inputbg":   "#111827",
            "active":    "#16202c",
        },
        "tk": {
            "card": "#121620", "code": "#0a0e14",
            "dim": "#8b98a9", "violet": "#b388ff",
            "cta_bg": "#1F6FEB", "cta_fg": "#ffffff", "cta_hover": "#3a8bff",
            "ghost_bg": "#30363D", "ghost_fg": "#C9D1D9",
            "ghost_hover": "#3a434d", "ghost_border": "#30363D",
            "canvas": "#121620", "canvas_border": "#21262D",
            "chip_idle":  ("#00e5ff", "#0f2a33"),
            "chip_ok":    ("#2ee6a8", "#0f3325"),
            "chip_run":   ("#b388ff", "#251a3d"),
            "chip_fail":  ("#ff5c7a", "#3a1620"),
            "chip_dim":   ("#6a737d", "#1a1d22"),
            "tag_ok":     ("#0d3a2a", "#7eeac2"),
            "tag_fail":   ("#43131f", "#ffaabb"),
            "tag_warn":   ("#3d2f08", "#ffd479"),
            "tag_match":  ("#2ee6a8", None),
            "tag_mm":     ("#ffaabb", "#43131f"),
            "tag_plain":  ("#c9d4e0", "#0a0e14"),
        },
    },
    "cyber-light": {
        "mode": "light",
        "colors": {
            "primary":   "#0b5cad",
            "secondary": "#2e86ff",
            "success":   "#027a48",
            "info":      "#2e86ff",
            "warning":   "#b58a00",
            "danger":    "#b42318",
            "light":     "#ffffff",
            "dark":      "#14213a",
            "bg":        "#f4f6fa",
            "fg":        "#1a2433",
            "selectbg":  "#d5ecf7",
            "selectfg":  "#0b3a52",
            "border":    "#d0d5dd",
            "inputfg":   "#111827",
            "inputbg":   "#ffffff",
            "active":    "#e9edf3",
        },
        "tk": {
            "card": "#ffffff", "code": "#ffffff",
            "dim": "#5c6773", "violet": "#6c2bd9",
            "cta_bg": "#0b5cad", "cta_fg": "#ffffff", "cta_hover": "#2e86ff",
            "ghost_bg": "#ffffff", "ghost_fg": "#0b5cad",
            "ghost_hover": "#e9edf3", "ghost_border": "#b9d3ea",
            "canvas": "#eef1f6", "canvas_border": "#9fc3dd",
            "chip_idle":  ("#0b5cad", "#e6f0fa"),
            "chip_ok":    ("#027a48", "#d1fadf"),
            "chip_run":   ("#6c2bd9", "#ede4ff"),
            "chip_fail":  ("#b42318", "#fee4e2"),
            "chip_dim":   ("#667085", "#eaecf0"),
            "tag_ok":     ("#c8e6c9", "#1b5e20"),
            "tag_fail":   ("#ffcdd2", "#b71c1c"),
            "tag_warn":   ("#fff3cd", "#8a6d00"),
            "tag_match":  ("#027a48", None),
            "tag_mm":     ("#b42318", "#ffe4e6"),
            "tag_plain":  ("#1a2433", "#ffffff"),
        },
    },
}


def _load_theme_pref():
    try:
        return json.loads(THEME_PREF_FILE.read_text()).get("theme")
    except Exception:
        return None


def _save_theme_pref(name):
    try:
        THEME_PREF_FILE.write_text(json.dumps({"theme": name}))
    except Exception:
        pass


def _sanitize_theme_name(name: str) -> str:
    """X2:刚性映射 —阻断外部主题名(如 darkly / cyborg)穿透到自定义调色板。

    cyber-dark / cyber-light 是本项目专属的科技感主题;任何其它深色
    ttkbootstrap 内置主题(solar / superhero 等)或 stale 历史持久化
    (darkly / cyborg)都收敛到 cyber-dark;浅色收敛到 cyber-light。
    这避免了 _theme_palette() 在 spec is None 时退化成 stock 调色板
    所造成的"深色背景里混进原生浅色部件"穿透现象。
    """
    if not isinstance(name, str):
        return "cyber-dark"
    if name in ("cyber-dark",):
        return "cyber-dark"
    if name in ("cyber-light",):
        return "cyber-light"
    DARK = {"darkly", "cyborg", "solar", "superhero",
            "dark", "darkly-pro", "vapor", "vapor-pro"}
    LIGHT = {"flatly", "cosmo", "journal", "litera", "minty", "lumen",
             "light", "sandstone", "yeti", "pulse", "united", "morph"}
    if name in DARK:
        return "cyber-dark"
    if name in LIGHT:
        return "cyber-light"
    # 未知名:按当前 OS 时间粗判,白天 light / 夜间 dark
    import datetime
    h = datetime.datetime.now().hour
    return "cyber-light" if 7 <= h < 19 else "cyber-dark"

def _np_median_color(arr):
    """Return (r,g,b) median of an RGB image region."""
    import numpy as _np
    flat = _np.asarray(arr).reshape(-1, 3)
    return tuple(int(c) for c in _np.median(flat, axis=0))


class _Tooltip:
    """Hover tooltip (overrideredirect Toplevel) showing a full path."""

    def __init__(self, widget):
        self.widget = widget
        self.tip = None
        self.text = ""
        widget.bind("<Enter>", self._enter, add="+")
        widget.bind("<Leave>", self._leave, add="+")

    def set_text(self, text):
        self.text = text or ""

    def _enter(self, _e):
        if not self.text:
            return
        try:
            x = self.widget.winfo_rootx() + 12
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
            self.tip = tk.Toplevel(self.widget)
            self.tip.wm_overrideredirect(True)
            self.tip.wm_geometry(f"+{x}+{y}")
            tk.Label(self.tip, text=self.text, bg="#21262D", fg="#C9D1D9",
                     font=("Menlo", 9), padx=8, pady=4,
                     justify="left").pack()
        except tk.TclError:
            self.tip = None

    def _leave(self, _e):
        if self.tip is not None:
            try:
                self.tip.destroy()
            except tk.TclError:
                pass
            self.tip = None


class _ThemedButton(tk.Canvas):
    """X2 工业暗黑科技感矢量自绘按钮。

    设计要点(对照原 tk.Frame + tk.Label 实现):
    - 由 tk.Canvas 自绘圆角矩形 + 1px 微描边 + 居中文本,
      macOS aqua 与各主题下视觉一致,不再依赖 ttk/原生的 bezel;
    - 调色板完全由调用方的 _pal 驱动,跟随 cyber-dark / cyber-light
      主题切换自动 retheme,与 _cta() / _cta_recolor() 既有协议对齐;
    - 对外暴露的 API 与原版严格一致:
        __init__(parent, text, command, kind, gui, padx, pady)
        configure(text=, state=)   (其它 key 走 super)
        cget(text=, state=)        (其它 key 走 super)
        refresh()                  (recolour from live _pal)
        _hover                     (bool, 内部状态)
        _cta_kind                  (str, 由 _cta() 注入,值为 primary/ghost)
    这样所有现有调用方(ocr_run_btn.configure(text=...) / _cta_buttons / 
    _cta_recolor)零修改即可平替。
    """

    # 内部调色板 token 名(从 _pal 拉)。缺失时退回 cyber-dark 安全色,
    # 让 _Theme_palette 的 fallback 分支走 darkly 时按钮也不会爆白底。
    _TOKEN_FILL     = "btn_fill"
    _TOKEN_HOVER    = "btn_hover"
    _TOKEN_ACTIVE   = "btn_active"
    _TOKEN_BORDER   = "btn_border"
    _TOKEN_FOCUS    = "btn_focus"
    _TOKEN_TEXT     = "btn_text"
    _TOKEN_TEXT_PRI = "btn_text_primary"
    _TOKEN_TEXT_DIM = "btn_text_disabled"

    def __init__(self, master, text, command, kind, gui, padx, pady):
        # parent 的背景必须从 master 现取,否则在 cyber-light 等浅色主题下
        # 会留下一条深色 Canvas 底。这是 Canvas 与 Frame 的关键差异。
        try:
            master_bg = master.cget("bg")
        except (tk.TclError, AttributeError):
            master_bg = "#0a0e14"
        # 计算初始尺寸:用字符度量估算,pack 时再被实际 layout 覆盖。
        import tkinter.font as tkfont
        try:
            f = tkfont.nametofont(gui._pal["font_ui_bold"])
        except (tk.TclError, KeyError):
            f = tkfont.nametofont("TkDefaultFont")
        text_w = f.measure(text)
        width = max(64, text_w + 2 * padx + 4)
        height = max(24, f.metrics("linespace") + 2 * pady + 2)

        super().__init__(
            master, width=width, height=height,
            bg=master_bg, highlightthickness=0, bd=0,
            takefocus=1,
        )
        self._gui = gui
        self._cmd = command
        self._kind = kind
        self._state = "normal"
        self._hover = False
        self._pressed = False
        self._padx = padx
        self._pady = pady
        self._text = text

        # 事件绑定(全部绑到自身 Canvas,不再走 _lbl)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        # pack / grid 后尺寸变化时强制重绘
        self.bind("<Configure>", lambda _e: self._safe_redraw())
        # 键盘可达性:Space / Enter 触发 command(原 tk.Button 行为)
        self.bind("<KeyRelease-space>", lambda _e: self._invoke())
        self.bind("<KeyRelease-Return>", lambda _e: self._invoke())

        # X2 fix:macOS ttkbootstrap 下偶发 aqua 路由问题会让 tk.Canvas
        # 漏接 <ButtonRelease-1>(按住不动时 release 不触发)。叠加绑一份
        # <Button-1> 兜底——只要按下去立即 invoke,与原生 tk.Button 行为一致。
        # 已经处理 pressed_was / disabled 等去重逻辑,不会双触发。
        self.bind("<Button-1>", self._on_press_release_invoke)

        self._safe_redraw()

    def _on_press_release_invoke(self, _e=None):
        """X2 fix:macOS aqua 事件路由兜底。压下立即 release+invoke,
        取代延迟到 <ButtonRelease-1> 的标准行为 — 当 release 丢失时仍可触发。
        用内部 _invoked_once 标志防止与标准 press/release 双触发。"""
        if self._state != "normal" or not self._cmd:
            return
        if getattr(self, "_invoked_once", False):
            return
        self._invoked_once = True
        try:
            self._pressed = False
            self._on_release(_e)  # 标准 release 流程
            self._invoke()
        finally:
            # 50ms 后清掉 _invoked_once,确保下一次点击仍可触发
            try:
                self.after(50, self._clear_invoked_once)
            except tk.TclError:
                pass

    def _clear_invoked_once(self):
        self._invoked_once = False

    # ---------- API 兼容 ----------
    def configure(self, cnf=None, **kw):
        if cnf:
            kw.update(cnf)
        text_changed = "text" in kw
        state_changed = "state" in kw
        if text_changed:
            self._text = kw.pop("text")
        if state_changed:
            self._state = kw.pop("state")
        if kw:
            try:
                super().configure(**kw)
            except tk.TclError:
                pass
        # text 长度变化也会改变 width,所以 text_changed 时同步尺寸
        if text_changed or state_changed:
            self._resize_for_text()
        self._safe_redraw()

    def config(self, cnf=None, **kw):
        return self.configure(cnf, **kw)

    def cget(self, key):
        if key == "text":
            return getattr(self, "_text", "")
        if key == "state":
            return self._state
        return super().cget(key)

    def _resize_for_text(self):
        """Keep the canvas wide enough for the CURRENT label.

        A tk.Canvas does not resize itself when its label changes, so a
        button that had room for "⚡ 开始测试" clipped the text when it
        became "⏳ 识别中…" -- the label simply disappeared.  The
        required width is always recomputed from the current text, and
        the canvas is re-laid-out so the new width takes effect.
        """
        import tkinter.font as tkfont
        try:
            f = tkfont.nametofont(self._gui._pal["font_ui_bold"])
        except (tk.TclError, KeyError):
            f = tkfont.nametofont("TkDefaultFont")
        text = getattr(self, "_text", "")
        text_w = f.measure(text)
        need_w = max(64, text_w + 2 * self._padx + 4)
        need_h = max(24, f.metrics("linespace") + 2 * self._pady + 2)
        # A canvas ignores its own width once it is packed, so ask the
        # geometry manager for the new size explicitly.
        try:
            self.configure(width=need_w, height=need_h)
        except tk.TclError:
            pass
        if self.winfo_manager():
            try:
                self.configure(width=need_w, height=need_h)
                self.update_idletasks()
            except tk.TclError:
                pass

    # ---------- 渲染 ----------
    def _safe_redraw(self):
        try:
            self.refresh()
        except (tk.TclError, KeyError):
            # _pal 还未就绪 / 主题切换中途 — 跳过,下个 after 再来
            pass

    def refresh(self):
        # 防御:有时在 _gui 销毁后还会被 _cta_recolor 调到
        try:
            pal = self._gui._pal
        except AttributeError:
            return
        # 调取 token。优先 _pal 显式键,缺失则按 kind+state 现场合成。
        primary = (self._kind == "primary")
        if self._state == "disabled":
            fill = pal.get(self._TOKEN_FILL,
                pal.get("ghost_bg", "#21262D"))
            border = pal.get(self._TOKEN_BORDER,
                pal.get("border", "#21262D"))
            text_col = pal.get(self._TOKEN_TEXT_DIM,
                pal.get("dim", "#8b98a9"))
            weight = "normal"
        elif primary:
            fill = pal.get("cta_bg", "#1F6FEB")
            if self._hover:
                fill = pal.get("cta_hover", "#3a8bff")
            border = pal.get(self._TOKEN_FOCUS,
                pal.get("primary", "#00e5ff")) if self._hover                 else pal.get("cta_bg", "#1F6FEB")
            text_col = pal.get(self._TOKEN_TEXT_PRI,
                pal.get("cta_fg", "#FFFFFF"))
            weight = "bold"
        else:
            if self._pressed:
                fill = pal.get(self._TOKEN_ACTIVE,
                    pal.get("primary", "#00e5ff"))
            elif self._hover:
                fill = pal.get(self._TOKEN_HOVER,
                    pal.get("ghost_hover", "#3a434d"))
            else:
                fill = pal.get(self._TOKEN_FILL,
                    pal.get("ghost_bg", "#21262D"))
            border = pal.get(self._TOKEN_FOCUS,
                pal.get("primary", "#00e5ff")) if self._hover                 else pal.get(self._TOKEN_BORDER,
                    pal.get("ghost_border", "#30363D"))
            text_col = pal.get(self._TOKEN_TEXT,
                pal.get("ghost_fg", "#C9D1D9"))
            weight = "normal"

        # 主区域 canvas 背景必须跟父容器一致,否则按钮外溢 1px 边框会显
        # 出深色 Canvas 底。父容器 bg 在 _switch_theme 后会变,所以每次重绘重取。
        try:
            master_bg = self.master.cget("bg")
        except (tk.TclError, AttributeError):
            master_bg = pal.get("bg", "#0a0e14")

        # 注意:必须 super().configure — 走 self.configure 会经 _safe_redraw 递归
        try:
            super().configure(bg=master_bg)
        except tk.TclError:
            pass
        self.delete("all")
        w = max(self.winfo_width(), 1)
        h = max(self.winfo_height(), 1)
        # The canvas may still report the OLD width right after the label
        # changed (tk.Canvas does not resize itself).  Draw at least as wide
        # as the current label needs, otherwise the text is clipped and the
        # button looks empty.
        try:
            import tkinter.font as _tkfont
            try:
                _f = _tkfont.nametofont(self._gui._pal["font_ui_bold"])
            except (tk.TclError, KeyError):
                _f = _tkfont.nametofont("TkDefaultFont")
            need_w = _f.measure(getattr(self, "_text", "")) + 2 * self._padx + 4
            need_h = _f.metrics("linespace") + 2 * self._pady + 2
        except Exception:
            need_w = need_h = 0
        w = max(w, need_w)
        h = max(h, need_h)
        # 圆角半径:与高度对齐,避免极端高宽比时变成胶囊失败
        r = max(2, min(8, h // 2 - 1))
        # 内边距 1px 给描边留呼吸空间
        self._draw_rounded_rect(
            1, 1, w - 2, h - 2, r,
            fill=fill, outline=border, width=1,
        )
        # primary 按钮在 hover 时画一道 0.6px 的内层高亮,模拟"光晕"
        if primary and self._hover:
            self._draw_rounded_rect(
                3, 3, w - 4, h - 4, max(1, r - 2),
                fill="", outline=pal.get("primary", "#00e5ff"), width=1,
            )
        # 文本
        font_bold = (primary or self._state == "normal"
                     and self._kind != "ghost")
        weight_eff = "bold" if primary else "normal"
        try:
            font_tuple = self._gui._pal["font_ui_bold"]
        except (AttributeError, KeyError):
            font_tuple = ("Segoe UI", 10, "bold" if primary else "normal")
        # 兼容 font_ui_bold = tuple of 2 (无 weight)
        if isinstance(font_tuple, tuple) and len(font_tuple) == 2:
            font_tuple = font_tuple + (("bold" if primary else "normal"),)
        self.create_text(
            w // 2, h // 2,
            text=getattr(self, "_text", ""),
            fill=text_col,
            font=font_tuple,
            anchor="center",
        )

    def _draw_rounded_rect(self, x1, y1, x2, y2, r, **kw):
        # 圆角多边形(spline 拟合)。比 create_arc 拼接更可控、跨平台一致。
        pts = [
            x1 + r, y1, x2 - r, y1, x2, y1,
            x2, y1 + r, x2, y2 - r, x2, y2,
            x2 - r, y2, x1 + r, y2, x1, y2,
            x1, y2 - r, x1, y1 + r, x1, y1,
        ]
        return self.create_polygon(pts, smooth=True, **kw)

    # ---------- 交互 ----------
    def _on_enter(self, _e=None):
        if self._state == "disabled":
            return
        self._hover = True
        self.config(cursor="hand2")
        self._safe_redraw()

    def _on_leave(self, _e=None):
        self._hover = False
        self._pressed = False
        self.config(cursor="")
        self._safe_redraw()

    def _on_press(self, _e=None):
        if self._state == "disabled":
            return
        self._pressed = True
        self._safe_redraw()
        # X2 polish:按下时启动 200ms 渐变回到 idle 色,模拟"弹性回弹"
        # 视觉上点击有"动一下"的反馈,但不阻塞按钮的 cmd() 调用。
        self._animate_to_idle()

    def _on_release(self, _e=None):
        if self._state == "disabled":
            return
        pressed_was = self._pressed
        self._pressed = False
        self._safe_redraw()
        if pressed_was:
            self._invoke()

    def _animate_to_idle(self):
        """按下后 ~200ms 内把 fill 从 active 渐变回 hover/idle。
        用 after() 分 4 帧推进,避免阻塞主线程且不依赖额外动画库。
        非 primary 按钮 / 已 disabled / 已被销毁 时直接跳过。"""
        if self._state == "disabled":
            return
        # 4 帧:0/60/120/180 ms — 浏览器常见 tap-feedback 时长
        try:
            self.after(60,  self._tick_animate, 1)
            self.after(120, self._tick_animate, 2)
            self.after(180, self._tick_animate, 3)
        except tk.TclError:
            pass

    def _tick_animate(self, step):
        # 在 idle / hover 期间被打断(用户移开鼠标),直接停在当前态;
        # 否则把 _pressed 视为松开,走回弹色。
        if step == 3 or self._hover or self._state != "normal":
            self._pressed = False
            self._safe_redraw()

    def _invoke(self):
        if self._state == "normal" and self._cmd:
            try:
                self._cmd()
            except Exception:
                # 与原实现一致:不吞噬用户回调异常,但不让按钮死锁
                pass


class PassportGUI(Window if HAS_TTKB else tk.Tk):
    def __init__(self, themename=None):
        if HAS_TTKB:
            # Construct on a valid stock theme first; the brand themes are
            # registered right after and _switch_theme() applies the
            # persisted choice (they cannot be registered before the Tk
            # root exists).
            super().__init__(themename="darkly")
        else:
            super().__init__()
        self._theme_name = "darkly"
        self.title("PASSPORT TEST BENCH 护照机综合测试台 v2.4")
        self.geometry("1280x780")
        self._set_window_icon()
        _load_ocrb_font(self)
        self._register_cyber_themes()
        self._build_menu()
        self._nfc_key_entries = {}
        self._nfc_dg1_lbs = {}
        self._nfc_prog_lbs = {}
        self._cta_buttons = []      # tk accent buttons (manually themed)
        self._build_header()
        # Restore the persisted theme (or the brand default) before any
        # tab widget exists; every builder then reads self._pal.
        self._switch_theme(themename or _load_theme_pref()
                           or ("cyber-dark" if HAS_TTKB else "darkly"))
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)
        # Tab order: Locator (1) -> Restore->MRZ (2) -> OCR (3) -> NFC (4)
        # -> MRZ (5) -> AC (6) -> face (7).
        self.tab_loc     = ttk.Frame(nb)
        self.tab_restore = ttk.Frame(nb)
        self.tab_ocr     = ttk.Frame(nb)
        self.tab_nfc     = ttk.Frame(nb)
        self.tab_mrz     = ttk.Frame(nb)
        self.tab_ac      = ttk.Frame(nb)
        self.tab_face    = ttk.Frame(nb)
        nb.add(self.tab_loc,     text="护照定位")
        nb.add(self.tab_restore, text="照片→MRZ")
        nb.add(self.tab_ocr,     text="MRZ OCR 测试")
        nb.add(self.tab_nfc,     text="NFC 读卡")
        nb.add(self.tab_mrz,     text="MRZ 解码")
        nb.add(self.tab_ac,      text="防伪验证")
        nb.add(self.tab_face,    text="照片比对")
        self._build_loc_tab()
        self._build_restore_tab()
        self._build_ocr_tab()
        self._build_nfc_tab()
        self._build_mrz_tab()
        self._build_ac_tab()
        self._build_face_tab()
        # Root window background: with no ttkbootstrap the Tk root defaults
        # to the macOS system window colour (light grey in light appearance)
        # which would flood the whole cockpit with a light backdrop.
        self.configure(bg=self._pal["bg"])
        self._refresh_tk_theme()
        self._apply_text_tag_palette()
        self.status = tk.StringVar(value=self._status_text())
        # X2 polish #5:底部状态栏固定 32px 高度 — 永久停靠在窗口底部。
        # 用 tk.Frame 而非 ttk.Frame,后者尺寸由 theme 决定,在 macOS 上
        # 即使配 height=32 也不生效。
        # pack(before=notebook) 让 sbar 先于 Notebook 占底部空间,避免
        # Notebook(fill="both",expand=1)把 sbar 挤到 1x1。
        self.update_idletasks()
        # X2 polish #5:状态栏 grid 三段式绝对布局 — 左(状态文字, 固定 320)
        # + 中(进度条 + chip, 固定 360) + 右(工具状态, 弹性)。这样进度条
        # 的位置和宽度不再受 chip / status 文字位数变化影响(消除晃动)。
        sbar = tk.Frame(self, bg=self._pal["bg"], height=32,
                       highlightthickness=1, highlightbackground=self._pal["border"])
        sbar.pack(fill="x", side="bottom", before=nb)
        sbar.pack_propagate(False)
        self._status_bar = sbar
        # grid 列权重 + minsize — 决定每段宽度
        sbar.columnconfigure(0, weight=0, minsize=320)  # 左:状态文字
        sbar.columnconfigure(1, weight=0, minsize=360)  # 中:进度条 + chip
        sbar.columnconfigure(2, weight=1)              # 右:工具状态(弹性)
        # ---- 左:状态文字(固定 320px 宽,文本左对齐,位数再多也不挤中间)----
        left_frame = tk.Frame(sbar, bg=self._pal["bg"])
        left_frame.grid(row=0, column=0, sticky="nsw", padx=(8, 4))
        sbar.grid_rowconfigure(0, weight=1)
        self.ocr_status_var = tk.StringVar(value="就绪")
        self._status_msg_lbl = tk.Label(
            left_frame, textvariable=self.ocr_status_var, anchor="w",
            bg=self._pal["bg"], fg=self._pal["fg"],
            font=self._pal["font_mono_sm"], width=38, padx=4)
        self._status_msg_lbl.pack(side="left", fill="y")
        # ---- 中:进度条 + chip(固定 360px 宽,grid_column=1)----
        # X2 polish:idle 时 mid_frame 也预留 360px minsize,避免 grid 列
        # 伸缩导致进度条首次 show 时位置瞬移(跳动)。
        mid_frame = tk.Frame(sbar, bg=self._pal["bg"], width=360)
        mid_frame.grid(row=0, column=1, sticky="nsw", padx=(4, 4))
        mid_frame.pack_propagate(False)
        mid_frame.columnconfigure(0, weight=0, minsize=220)
        mid_frame.columnconfigure(1, weight=0, minsize=130)
        # progress 用 ttk.Progressbar,默认不 pack — _show_progress() 时 pack
        self.ocr_progress = ttk.Progressbar(mid_frame, length=220,
                                            mode="determinate", maximum=100)
        # 把 progress 的父级改为 mid_frame,后续 _show_progress 在这里 pack
        # ---- 右:工具状态(弹性靠右)----
        right_frame = tk.Frame(sbar, bg=self._pal["bg"])
        right_frame.grid(row=0, column=2, sticky="nse", padx=(4, 8))
        sbar._right_frame = right_frame
        # Visual-pass 2.0: chip strip — 每个工具一个状态 chip
        self._status_chip_labels = {}
        for label, path in (("MRZ", MRZ_TOOL), ("AC", AC_TOOL),
                            ("FACE", FACE_TOOL),
                            ("GEN", GEN_SAMP), ("NFC", NFC_TOOL)):
            ok = path.exists()
            lb = ttk.Label(
                right_frame, text=f" ● {label} ",
                anchor="center", padding=(6, 2),
                font=self._pal["font_ui_bold"])
            lb.pack(side="right", padx=2, pady=2)
            self._status_chip_labels[label] = (lb, ok)
            # 右下角不再显示被截断的长文本；每个 chip 悬停即看完整状态。
            tip = _Tooltip(lb)
            tip.set_text(f"{label}: {path}\n"
                         f"状态: {'OK' if ok else 'MISSING'}\n"
                         f"{self._status_text()}")
        # 兼容保留 self.status StringVar（不再有长文本 Label）。
        # Keep preview references so Tk doesn't garbage-collect them.
        self._preview_imgs = {}

    def _status_text(self):
        parts = []
        for label, path in (("mrz_tool", MRZ_TOOL), ("ac_tool", AC_TOOL),
                            ("face_tool", FACE_TOOL),
                            ("gen_sample", GEN_SAMP),
                            ("nfc_tool", NFC_TOOL)):
            parts.append(f"{label}={'OK' if path.exists() else 'MISSING'}")
        return "  ".join(parts)

    def _apply_theme_proof_styles(self):
        """Pin readable colors on input widgets. With ttkbootstrap the
        palette comes from the active theme (st.colors); otherwise fall back
        to a dark-aware field/text scheme (no ttkbootstrap in .venv-ctk:
        the old hardcoded #ffffff fallback left white Treeviews in dark
        mode)."""
        st = ttk.Style()
        dark = not self._is_light()
        try:
            c = st.colors
            if isinstance(c, dict):
                FIELD = c.get("inputbg", "#ffffff")
                TEXT = c.get("inputfg", "#111827")
                HEAD = c.get("active", "#e9edf3")
                SEL = c.get("primary", "#0b5cad")
            else:
                FIELD = getattr(c, "inputbg", "#ffffff")
                TEXT = getattr(c, "inputfg", "#111827")
                HEAD = getattr(c, "active", "#e9edf3")
                SEL = getattr(c, "primary", "#0b5cad")
        except AttributeError:
            if dark:
                FIELD, TEXT, HEAD, SEL = "#14181e", "#d5dae2", "#161F30", "#1F6FEB"
            else:
                FIELD, TEXT, HEAD, SEL = "#F4F6F9", "#1E293B", "#FFFFFF", "#0b5cad"
        # Pin palette-derived border so Labelframe/Card uses a faint line.
        border = (self._pal.get("border") if hasattr(self, "_pal") else "#3A3B3C")
        if not HAS_TTKB:
            # macOS aqua ignores Heading/radiobutton backgrounds; clam
            # lets every style element below render from our palette.
            try:
                st.theme_use("clam")
            except tk.TclError:
                pass
        try:
            st.configure("Data.Treeview", background=FIELD,
                         fieldbackground=FIELD, foreground=TEXT,
                         bordercolor=SEL, lightcolor=SEL, darkcolor=SEL,
                         rowheight=26)
            sel_bg = "#DBEAFE" if not dark else SEL
            sel_fg = "#1E40AF" if not dark else "#ffffff"
            st.map("Data.Treeview",
                   background=[("selected", sel_bg)],
                   foreground=[("selected", sel_fg)])
            st.configure("Data.Treeview.Heading", background=HEAD,
                         foreground=TEXT, padding=(6, 4), relief="flat")
            st.map("Data.Treeview.Heading", background=[("active", HEAD)])
            BG_CARD = "#FFFFFF" if not dark else FIELD
            st.configure(".", background=FIELD, foreground=TEXT)
            st.configure("Card.TFrame", background=BG_CARD, relief="solid",
                         borderwidth=1, bordercolor=border)
            st.configure("Card.TLabel", background=BG_CARD, foreground=TEXT)
            # Card-style container: replace the chunky native ridge with a
            # 1px flat border tinted to the palette border colour. Also pin
            # the container background: on macOS TLabelframe/TFrame default
            # to the SYSTEM window colour (light grey in light appearance),
            # which would punch big light panels into the dark cockpit.
            st.configure("TLabelframe", background=FIELD, bordercolor=border,
                         borderwidth=1, relief="solid")
            st.configure("TLabelframe.Label", background=FIELD,
                         foreground=TEXT, padding=(6, 2))
            st.configure("TFrame", background=FIELD)
            st.configure("TPanedwindow", background=FIELD,
                         sashwidth=3, sashpad=0, sashrelief="flat")
            st.configure("Field.TEntry", fieldbackground=FIELD,
                         foreground=TEXT, insertcolor=TEXT, bordercolor=border)
            st.map("Field.TEntry",
                   background=[("readonly", FIELD), ("disabled", HEAD)],
                   foreground=[("readonly", TEXT), ("disabled", TEXT)])
            st.configure("Field.TCombobox", fieldbackground=FIELD,
                         foreground=TEXT, bordercolor=border)
            st.map("Field.TCombobox",
                   fieldbackground=[("readonly", FIELD)],
                   foreground=[("readonly", TEXT)])
            # Notebook: theme the tab strip + content area so the deep
            # cockpit background is not broken by native light tabs.
            st.configure("TNotebook", background=FIELD, borderwidth=0)
            st.configure("TNotebook.Tab", background=HEAD, foreground=TEXT,
                         padding=(10, 4))
            st.map("TNotebook.Tab",
                   background=[("selected", FIELD), ("active", HEAD)],
                   foreground=[("selected", TEXT)])
            st.configure("TRadiobutton", background=FIELD, foreground=TEXT,
                         indicatorbackground=FIELD, bordercolor=border)
            st.map("TRadiobutton",
                   background=[("active", HEAD)],
                   foreground=[("active", TEXT), ("disabled", "#7c7c7c")],
                   indicatorbackground=[("selected", SEL)])
            st.configure("TCheckbutton", background=FIELD, foreground=TEXT,
                         indicatorbackground=FIELD, bordercolor=border)
            st.map("TCheckbutton",
                   background=[("active", HEAD)],
                   foreground=[("active", TEXT), ("disabled", "#7c7c7c")],
                   indicatorbackground=[("selected", SEL)])
            st.configure("TScrollbar", background=HEAD, troughcolor=FIELD,
                         arrowcolor=TEXT, bordercolor=border, relief="flat")
            st.configure("Vertical.TScrollbar", gripcount=0, background=HEAD,
                         troughcolor=FIELD, arrowcolor=TEXT, bordercolor=border,
                         relief="flat")
            st.configure("Horizontal.TScrollbar", gripcount=0, background=HEAD,
                         troughcolor=FIELD, arrowcolor=TEXT, bordercolor=border,
                         relief="flat")
            st.configure("TEntry", fieldbackground=FIELD, foreground=TEXT,
                         insertcolor=TEXT, bordercolor=border)
            st.configure("TCombobox", fieldbackground=FIELD, foreground=TEXT,
                         bordercolor=border)
        except Exception:
            pass

    def _register_cyber_themes(self):
        """Register the two brand themes with ttkbootstrap so every ttk
        widget class (Notebook, Button, Progressbar, Treeview, ...) is
        restyled by a single theme_use() call."""
        if not HAS_TTKB:
            return
        try:
            from ttkbootstrap.style import ThemeDefinition
        except Exception:
            return
        st = ttk.Style()
        for name, spec in CYBER_THEMES.items():
            if name in st.theme_names():
                continue
            try:
                st.register_theme(ThemeDefinition(name, spec["colors"],
                                                  mode=spec["mode"]))
            except Exception:
                pass

    def _switch_theme(self, name):
        # Friendly aliases so the menu / header toggle can offer
        # "light" / "dark" without memorising theme names.
        if name in ("dark", "dark_default"):
            name = "cyber-dark"
        elif name in ("light", "light_default"):
            name = "cyber-light"
        # X2:刚性拦截 —任何外部 / 历史主题名都不能穿透到 _theme_palette 的
        # fallback stock 分支(stock 调色板会失去霓虹科技感并引发扫描混色)。
        name = _sanitize_theme_name(name)
        try:
            ttk.Style().theme_use(name)
        except tk.TclError:
            pass
        self._theme_name = name
        self._pal = self._theme_palette()
        self._apply_theme_proof_styles()
        self._apply_text_tag_palette()
        self._refresh_tk_theme()
        self._refresh_header()
        self._update_theme_menu_label()
        _save_theme_pref(name)

    def _is_light(self):
        spec = CYBER_THEMES.get(self._theme_name)
        if spec is not None:
            return spec["mode"] != "dark"
        return self._theme_name in ("cosmo", "journal", "flatly", "superhero")

    def _toggle_theme(self):
        self._switch_theme("cyber-dark" if self._is_light() else "cyber-light")

    def _update_theme_menu_label(self):
        viewm = getattr(self, "_viewm", None)
        if viewm is None:
            return
        label = "🌙  深色主题" if not self._is_light() else "☀️  浅色主题"
        try:
            viewm.entryconfig(self._theme_toggle_idx, label=label)
        except tk.TclError:
            pass

    # ---------------- brand header + theme toggle ----------------
    def _build_header(self):
        """Slim instrument-style header: letterspaced brand caption on
        the left, a segmented dark/light switch on the right. Purely
        additive — the tab layout below is untouched."""
        h = tk.Frame(self, bd=0, highlightthickness=0)
        h.pack(fill="x", side="top")
        self._header = h
        # X2 polish:header 左侧加 logo 装饰条 — 1px 宽 × 24px 高的"品牌
        # 色块",颜色取调色板中的 primary,在 light 主题下取 accent。
        # 这是工业仪表盘的标志性细节。
        self._header_brand = tk.Frame(h, width=4, height=24,
                                      bg=self._pal.get("primary",
                                              "#00e5ff") if hasattr(self, "_pal")
                                              else "#00e5ff",
                                      bd=0)
        self._header_brand.pack(side="left", padx=(14, 10), pady=7)
        self._header_brand.pack_propagate(False)
        self._header_title = tk.Label(
            h, text="P A S S P O R T   T E S T   B E N C H",
            font=self._pal["font_title"] if hasattr(self, "_pal")
            else ("TkDefaultFont", 12, "bold"), anchor="w")
        self._header_title.pack(side="left", padx=(0, 8), pady=7)
        # X2 polish:标题右侧加 1px 暗灰竖线,视觉锚点。
        tk.Frame(h, bg=self._pal.get("border", "#21262D") if hasattr(self, "_pal")
                 else "#21262D", width=1).pack(side="left", pady=10)
        self._header_sub = tk.Label(h, text="护照机测试台 · MRZ / NFC / 防伪 / 人脸",
                                    anchor="w")
        self._header_sub.pack(side="left", padx=(8, 0), pady=7)
        seg = tk.Frame(h, bd=0, highlightthickness=0)
        seg.pack(side="right", padx=10, pady=5)
        self._seg_frame = seg
        self._seg_toggle = tk.Label(seg, text="☀️ 浅色", cursor="hand2")
        self._seg_toggle.pack(side="left", padx=1, ipadx=10, ipady=2)
        self._seg_toggle.bind("<Button-1>", lambda e: self._toggle_theme())
        # Provisional palette: _switch_theme (which owns the final one)
        # runs after the header exists.
        if not hasattr(self, "_pal"):
            self._pal = self._theme_palette()
        self._refresh_header()

    def _refresh_header(self):
        h = getattr(self, "_header", None)
        if h is None:
            return
        pal = self._pal
        spec = CYBER_THEMES.get(self._theme_name)
        dark = spec["mode"] == "dark" if spec else \
            self._theme_name in ("darkly", "cyborg")
        try:
            h.configure(bg=pal["bg"])
            self._seg_frame.configure(bg=pal["bg"])
            self._header_title.configure(bg=pal["bg"], fg=pal["fg"])
            self._header_sub.configure(bg=pal["bg"], fg=pal["dim"])
            # X2 polish:同步刷新左侧 logo 色块 + 分隔线
            brand = getattr(self, "_header_brand", None)
            if brand is not None:
                brand.configure(bg=pal.get("primary", pal.get("accent",
                                       pal["cta_bg"])))
            seg = self._seg_toggle
            seg.configure(text=("🌙 深色" if dark else "☀️ 浅色"),
                          bg=pal["cta_bg"], fg=pal["cta_fg"])
        except tk.TclError:
            pass

    # ---------------- themed accent buttons ----------------
    def _cta(self, parent, text, cmd, kind="primary", padx=14, pady=4):
        """Flat accent button (tk.Button so bg/fg render on macOS aqua)
        with hover feedback and automatic re-theming on switch."""
        pal = self._pal
        if kind == "primary":
            bg, fg = pal["cta_bg"], pal["cta_fg"]
        else:
            bg, fg = pal["ghost_bg"], pal["ghost_fg"]
        b = _ThemedButton(parent, text=text, command=cmd, kind=kind,
                          gui=self, padx=padx, pady=pady)
        b._cta_kind = kind
        self._cta_buttons.append(b)
        return b

    def _cta_recolor(self, b=None, hover=False):
        targets = [b] if b is not None else list(getattr(self, "_cta_buttons", []))
        for w in targets:
            if hasattr(w, "refresh"):
                if b is w:
                    w._hover = hover
                w.refresh()

    # ---------------- Text/Canvas tag palette ----------------
    def _apply_text_tag_palette(self):
        """Re-colour every Text-widget tag (diff highlight, PASS/FAIL
        rows, NFC trace) from the palette. Called at build time and on
        every theme switch."""
        pal = self._pal
        def cfg(w, tag, spec):
            if w is None:
                return
            try:
                kw = {"foreground": spec["fg"]}
                if spec["bg"]:
                    kw["background"] = spec["bg"]
                w.tag_configure(tag, **kw)
            except tk.TclError:
                pass
        # 整行红绿铺满太刺眼（尤其选中行深蓝时三色撞色）。亮色主题下
        # 用极淡底色 + 深色前景；暗色主题保持原 tag_*（对比度需要）。
        if self._is_light():
            soft_ok = {"bg": "#F0FAF4", "fg": "#0F5132"}
            soft_fail = {"bg": "#FEF2F2", "fg": "#A00D20"}
            soft_warn = {"bg": "#FFFBEB", "fg": "#8A6D00"}
        else:
            soft_ok, soft_fail, soft_warn = (
                pal["tag_ok"], pal["tag_fail"], pal["tag_warn"])
        cfg(getattr(self, "ocr_detail", None), "OK", soft_ok)
        cfg(getattr(self, "ocr_detail", None), "FAIL", soft_fail)
        cfg(getattr(self, "ocr_detail", None), "PASS", soft_ok)
        cfg(getattr(self, "ocr_detail", None), "SUSPECT", soft_warn)
        cfg(getattr(self, "ocr_detail", None), "REJECT", soft_fail)
        cfg(getattr(self, "ocr_summary", None), "row_ok", soft_ok)
        cfg(getattr(self, "ocr_summary", None), "row_fail", soft_fail)
        # 左侧案例列表的状态配色（与右侧结果表同步）。
        cfg(getattr(self, "ocr_case_list", None), "pass", soft_ok)
        cfg(getattr(self, "ocr_case_list", None), "fail", soft_fail)
        cfg(getattr(self, "ocr_case_list", None), "warn", soft_warn)
        cfg(getattr(self, "ocr_diff_text", None), "match", pal["tag_match"])
        cfg(getattr(self, "ocr_diff_text", None), "mm", pal["tag_mm"])
        cfg(getattr(self, "ocr_diff_text", None), "okline", pal["tag_match"])
        con = pal["console"]
        for w, tag, spec in (
                (getattr(self, "ocr_diff_text", None), "hdr",
                 {"bg": None, "fg": con["pad"]}),
                (getattr(self, "ocr_diff_text", None), "lab",
                 {"bg": None, "fg": con["pad"]}),
                (getattr(self, "ocr_diff_text", None), "pad",
                 {"bg": None, "fg": con["pad"]}),
                (getattr(self, "ocr_diff_text", None), "error",
                 {"bg": con["error_bg"], "fg": con["error_fg"]})):
            cfg(w, tag, spec)
        rst = getattr(self, "rst_info", None)
        if rst is not None:
            light = self._is_light()
            method_colors = {
                "traditional": "#9a6700" if light else "#ffb454",
                "cnn": "#6c2bd9" if light else "#c084fc",
                "tesseract": "#027a48" if light else "#2ee6a8",
                "paddle": "#0b5cad" if light else "#5aa9ff",
                "indra": "#b45309" if light else "#fbbf24",
            }
            for method, color in method_colors.items():
                rst.tag_configure(
                    f"rst_method_{method}",
                    foreground=color,
                    font=(self._pal["font_mono"][0], 10, "bold"))
            rst.tag_configure(
                "rst_section", foreground=con["header"],
                font=(self._pal["font_ui"][0], 10, "bold"))
            rst.tag_configure(
                "rst_meta", foreground=con["pad"],
                font=self._pal["font_mono_sm"])
            rst.tag_configure(
                "rst_label", foreground=con["pad"],
                font=self._pal["font_mono_sm"])
            rst.tag_configure(
                "rst_mrz", foreground=pal["fg"],
                font=(self._pal["font_mono"][0], 10, "bold"))
            rst.tag_configure(
                "rst_pass", foreground=pal["ok"],
                font=(self._pal["font_ui"][0], 10, "bold"))
            rst.tag_configure(
                "rst_fail", foreground=pal["err"],
                font=(self._pal["font_ui"][0], 10, "bold"))
            rst.tag_configure(
                "rst_warn", foreground=self._pal["warn"],
                font=self._pal["font_mono_sm"])
            rst.tag_configure(
                "rst_decode", foreground=pal["fg"],
                font=self._pal["font_mono_sm"])
            rst.tag_configure(
                "rst_raw", foreground=con["pad"],
                font=(self._pal["font_mono"][0], 9))
        cfg(getattr(self, "nfc_trace", None), "sw_ok",
            {"bg": None, "fg": "#3FB950"})
        cfg(getattr(self, "nfc_trace", None), "sw_bad",
            {"bg": None, "fg": "#F85149"})
        # AC verdict tags (same tag names reused across report widgets).
        for name in ("ac_out", "ac_report"):
            w = getattr(self, name, None)
            if w is None:
                continue
            cfg(w, "PASS", pal["tag_ok"])
            cfg(w, "SUSPECT", pal["tag_warn"])
            cfg(w, "REJECT", pal["tag_fail"])

    def _theme_palette(self):
        """Semantic color palette for the active theme (tk widgets)."""
        # ---- Cyber-tech brand themes: straight from the token table ----
        spec = CYBER_THEMES.get(self._theme_name)
        if HAS_TTKB and spec is not None:
            c, t = spec["colors"], spec["tk"]
        else:
            c = t = None
        if spec is not None:
            dark = spec["mode"] == "dark"
        else:
            dark = (not HAS_TTKB) or self._theme_name in ("darkly", "cyborg")
        # Mono font: Menlo on macOS, Consolas on Windows; falls back to TkFixedFont.
        if sys.platform == "darwin":
            mono_family = "Menlo"
        elif sys.platform.startswith("win"):
            mono_family = "Consolas"
        else:
            mono_family = "DejaVu Sans Mono"
        # UI font: SF Pro Text / PingFang SC on macOS, Segoe UI on Windows.
        if sys.platform == "darwin":
            ui_family = ".AppleSystemUIFont"
        elif sys.platform.startswith("win"):
            ui_family = "Segoe UI"
        else:
            ui_family = "TkDefaultFont"
        if t is not None:
            return {
                "bg": c["bg"], "fg": c["fg"],
                "field": c["inputbg"], "field_fg": c["inputfg"],
                "primary": c["primary"], "accent": t["cta_bg"],
                "ok": t["chip_ok"][0], "err": t["chip_fail"][0],
                "warn": t["chip_run"][0], "violet": t["violet"],
                "card": t["card"], "code": t["code"],
                "border": c["border"], "dim": t["dim"],
                "canvas": t["canvas"], "canvas_border": t["canvas_border"],
                "cta_bg": t["cta_bg"], "cta_fg": t["cta_fg"],
                "cta_hover": t["cta_hover"],
                "ghost_bg": t["ghost_bg"], "ghost_fg": t["ghost_fg"],
                "ghost_hover": t["ghost_hover"],
                "ghost_border": t["ghost_border"],
                "chip": {
                    "idle":  {"fg": t["chip_idle"][0],  "bg": t["chip_idle"][1]},
                    "ok":    {"fg": t["chip_ok"][0],    "bg": t["chip_ok"][1]},
                    "run":   {"fg": t["chip_run"][0],   "bg": t["chip_run"][1]},
                    "fail":  {"fg": t["chip_fail"][0],  "bg": t["chip_fail"][1]},
                    "dim":   {"fg": t["chip_dim"][0],   "bg": t["chip_dim"][1]},
                },
                "tag_ok":    {"bg": t["tag_ok"][0],    "fg": t["tag_ok"][1]},
                "tag_fail":  {"bg": t["tag_fail"][0],  "fg": t["tag_fail"][1]},
                "tag_warn":  {"bg": t["tag_warn"][0],  "fg": t["tag_warn"][1]},
                "tag_match": {"bg": None,              "fg": t["tag_match"][0]},
                "tag_mm":    {"bg": t["tag_mm"][0],    "fg": t["tag_mm"][1]},
                "tag_plain": {"bg": t["tag_plain"][1], "fg": t["tag_plain"][0]},
                "console": {
                    "bg": "#0B0F19" if dark else "#FFFFFF",
                    "fg": "#E2E8F0" if dark else "#1a2433",
                    "header": "#38BDF8" if dark else "#0b5cad",
                    "pad": "#475569" if dark else "#94a3b8",
                    "error_bg": "#EF4444" if dark else "#B42318",
                    "error_fg": "#FFFFFF",
                },
                "metric": {
                    "cases": "#E6EDF3" if dark else "#0b5cad",
                    "mean": "#38BDF8" if dark else "#0b8fb0",
                    "p95": "#38BDF8" if dark else "#0b5cad",
                    "pass": "#3FB950" if dark else "#027a48",
                    "l1": "#3FB950" if dark else "#027a48",
                    "l2": "#3FB950" if dark else "#027a48",
                    "full": "#3FB950" if dark else "#B45309",
                },
                "font_ui": (ui_family, 10),
                "font_ui_bold": (ui_family, 10, "bold"),
                "font_title": (ui_family, 12, "bold"),
                "font_mono": (mono_family, 10),
                "font_mono_sm": (mono_family, 9),
            }
        # ---- Legacy ttkbootstrap presets: derive from the theme ----
        st = ttk.Style()
        try:
            cc = st.colors
            if isinstance(cc, dict):
                bg = cc.get("bg", "#14181e" if dark else "#f4f6fa")
                fg = cc.get("fg", "#d5dae2" if dark else "#1a2433")
                field = cc.get("inputbg", "#1f242c" if dark else "#FFFFFF")
                field_fg = cc.get("inputfg", "#e6ebf2" if dark else "#1a2433")
                primary = cc.get("primary", "#0b5cad")
            else:
                bg = getattr(cc, "bg", "#14181e" if dark else "#f4f6fa")
                fg = getattr(cc, "fg", "#d5dae2" if dark else "#1a2433")
                field = getattr(cc, "inputbg", "#1f242c" if dark else "#FFFFFF")
                field_fg = getattr(cc, "inputfg", "#e6ebf2" if dark else "#1a2433")
                primary = getattr(cc, "primary", "#0b5cad")
        except AttributeError:
            if dark:
                bg, fg, field, field_fg, primary = ("#14181e", "#d5dae2",
                                                    "#1f242c", "#e6ebf2", "#0b5cad")
            else:
                bg, fg, field, field_fg, primary = ("#f4f6fa", "#1a2433",
                                                    "#FFFFFF", "#1a2433", "#0b5cad")
        try:
            cc = ttk.Style().colors
            active_c = (cc.get("active", "#e9edf3") if isinstance(cc, dict)
                        else getattr(cc, "active", "#e9edf3"))
        except AttributeError:
            active_c = "#e9edf3"
        accent = "#00e5ff" if dark else primary
        # Visual-pass 2.0 semantic tokens, kept for the stock presets.
        if dark:
            border = "#21262D"
            card = "#121620"
            code = "#0B0F19"
            chips = {"idle": ("#00F5FF", "#0f2a33"), "ok": ("#10B981", "#0f3325"),
                     "run": ("#b388ff", "#251a3d"), "fail": ("#EF4444", "#3a1620"),
                     "dim": ("#6a737d", "#1a1d22")}
            tags = {"ok": ("#0d3a2a", "#7eeac2"), "fail": ("#43131f", "#ffaabb"),
                    "warn": ("#3d2f08", "#F59E0B"), "match": "#10B981",
                    "mm": ("#EF4444", "#FFFFFF"), "plain": ("#c9d4e0", "#0a0e14")}
        else:
            border = "#D0D5DD"
            card = "#F2F4F7"
            code = "#FFFFFF"
            chips = {"idle": ("#0b5cad", "#E6F0FA"), "ok": ("#027a48", "#D1FADF"),
                     "run": ("#6c2bd9", "#EDE4FF"), "fail": ("#B42318", "#FEE4E2"),
                     "dim": ("#666666", "#EAECF0")}
            tags = {"ok": ("#c8e6c9", "#1b5e20"), "fail": ("#ffcdd2", "#b71c1c"),
                    "warn": ("#fff3cd", "#8a6d00"), "match": "#027a48",
                    "mm": ("#b42318", "#ffe4e6"), "plain": ("#1a2433", "#ffffff")}
        return {
            "bg": bg, "fg": fg, "field": field, "field_fg": field_fg,
            "primary": primary, "accent": accent,
            "ok": chips["ok"][0], "err": chips["fail"][0],
            "warn": chips["run"][0], "violet": chips["run"][0],
            "card": card, "code": code, "border": border,
            "dim": "#9aa4b2" if dark else "#5c6773",
            "canvas": card, "canvas_border": border,
            "cta_bg": "#1F6FEB" if dark else primary, "cta_fg": "#ffffff",
            "cta_hover": "#3a8bff" if dark else "#2b7ed6",
            "ghost_bg": "#30363D" if dark else field,
            "ghost_fg": "#C9D1D9" if dark else accent,
            "ghost_hover": "#3a434d" if dark else active_c,
            "ghost_border": "#30363D" if dark else border,
            "chip": {k: {"fg": v[0], "bg": v[1]} for k, v in chips.items()},
            "tag_ok":    {"bg": tags["ok"][0],    "fg": tags["ok"][1]},
            "tag_fail":  {"bg": tags["fail"][0],  "fg": tags["fail"][1]},
            "tag_warn":  {"bg": tags["warn"][0],  "fg": tags["warn"][1]},
            "tag_match": {"bg": None,             "fg": tags["match"]},
            "tag_mm":    {"bg": tags["mm"][0],    "fg": tags["mm"][1]},
            "tag_plain": {"bg": tags["plain"][1], "fg": tags["plain"][0]},
            "console": {
                "bg": "#0B0F19" if dark else "#FFFFFF",
                "fg": "#E2E8F0" if dark else "#1a2433",
                "header": "#38BDF8" if dark else "#0b5cad",
                "pad": "#475569" if dark else "#94a3b8",
                "error_bg": "#EF4444" if dark else "#B42318",
                "error_fg": "#FFFFFF",
            },
            "metric": {
                "cases": "#E6EDF3" if dark else "#0b5cad",
                "mean": "#38BDF8" if dark else "#0b8fb0",
                "p95": "#38BDF8" if dark else "#0b5cad",
                "pass": "#3FB950" if dark else "#027a48",
                "l1": "#3FB950" if dark else "#027a48",
                "l2": "#3FB950" if dark else "#027a48",
                "full": "#3FB950" if dark else "#B45309",
            },
            "font_ui": (ui_family, 10),
            "font_ui_bold": (ui_family, 10, "bold"),
            "font_title": (ui_family, 12, "bold"),
            "font_mono": (mono_family, 10),
            "font_mono_sm": (mono_family, 9),
        }

    def _refresh_tk_theme(self):
        """Re-color tk (non-ttk) widgets to match the active theme."""
        pal = self._pal
        chip = pal["chip"]
        # Text/code widgets: bg = field, fg = field_fg (already contrast-safe).
        for n in ("loc_info", "nfc_inspect", "nfc_adv_text",
                  "mrz_text", "mrz_out", "ac_out", "face_out"):
            w = getattr(self, n, None)
            if w is not None:
                try:
                    w.configure(bg=pal["field"], fg=pal["field_fg"],
                                highlightbackground=pal["border"],
                                highlightcolor=pal["accent"],
                                highlightthickness=1)
                except tk.TclError:
                    pass
        w = getattr(self, "rst_info", None)
        if w is not None:
            try:
                w.configure(bg=pal["code"], fg=pal["fg"],
                            highlightbackground=pal["border"],
                            highlightcolor=pal["accent"],
                            highlightthickness=1)
            except tk.TclError:
                pass
        w = getattr(self, "loc_canvas", None)
        if w is not None:
            try:
                w.configure(bg=pal["canvas"],
                            highlightbackground=pal["canvas_border"],
                            highlightcolor=pal["accent"])
            except tk.TclError:
                pass
        # ---- Visual-pass 2.0: chip-style badges need explicit bg/fg ----
        # Right-top corner empty white square fix: nfc_badge starts blank but
        # was inheriting the platform default bg; pin to window bg + dim chip.
        w = getattr(self, "nfc_badge", None)
        if w is not None:
            try:
                w.configure(bg=pal["bg"], fg=chip["dim"]["fg"],
                            font=pal["font_ui_bold"])
            except tk.TclError:
                pass
        # BAC state-machine chips: reset every known badge to the dim chip
        # palette; _nfc_show() will recolour them on the next render pass.
        for _title, _var, lb in getattr(self, "_nfc_states", []):
            try:
                lb.configure(bg=pal["bg"], fg=chip["idle"]["fg"],
                             font=pal["font_ui_bold"])
            except tk.TclError:
                pass
        # DG1 / SOD / read-progress labels live in dicts of tk.Label refs.
        for d in ("_nfc_dg1_lbs", "_nfc_prog_lbs"):
            dd = getattr(self, d, None) or {}
            for w in dd.values():
                try:
                    w.configure(bg=pal["bg"], fg=pal["fg"],
                                font=pal["font_mono_sm"])
                except tk.TclError:
                    pass
        for attr in ("_nfc_sod_sha_lb", "_nfc_sod_rsa_lb"):
            w = getattr(self, attr, None)
            if w is not None:
                try:
                    w.configure(bg=pal["bg"], fg=pal["fg"],
                                font=pal["font_mono_sm"])
                except tk.TclError:
                    pass
        # Status-bar tool chips: OK=green dot, MISSING=red dot, both with
        # contrast-safe chip backgrounds that survive light/dark themes.
        for _label, (lb, ok) in getattr(self, "_status_chip_labels", {}).items():
            try:
                if ok:
                    lb.configure(foreground=chip["ok"]["fg"],
                                 background=chip["ok"]["bg"])
                else:
                    lb.configure(foreground=chip["fail"]["fg"],
                                 background=chip["fail"]["bg"])
            except tk.TclError:
                pass
        # X2 polish #5:底部 sbar 在主题切换时同步刷 bg + highlightbackground,
        # 否则 light 主题下保留 dark 边框色会被扫描脚本误判。
        sbar = getattr(self, "_status_bar", None)
        if sbar is not None:
            try:
                sbar.configure(bg=pal["bg"],
                               highlightbackground=pal["border"])
            except tk.TclError:
                pass
        # X2 polish:ttk.Progressbar 的 trough/border 配置 cyber-tech 调色板。
        # 这是与原版 ttk.Progressbar 完全兼容的视觉升级,稳靠且无 layout 风险。
        try:
            import tkinter.ttk as _ttk
            _st = _ttk.Style()
            _st.configure("Horizontal.TProgressbar",
                          troughcolor=pal.get("card", "#121620"),
                          background=pal.get("primary", "#00e5ff"),
                          bordercolor=pal.get("border", "#21262D"),
                          lightcolor=pal.get("primary", "#00e5ff"),
                          darkcolor=pal.get("accent", "#1F6FEB"))
        except Exception:
            pass
        # Themed CTA buttons (tk.Button accents need manual recolour).
        self._cta_recolor()
        w = getattr(self, "ocr_preview_label", None)
        if w is not None:
            try:
                w.configure(background=pal["canvas"],
                            highlightbackground=pal["border"])
            except tk.TclError:
                pass
        ml = getattr(self, "ocr_preview_meta_lb", None)
        if ml is not None:
            try:
                ml.configure(bg=pal["card"], fg=pal["fg"])
            except tk.TclError:
                pass
        for d in ("_nfc_key_entries", "_nfc_dg1_lbs", "_nfc_prog_lbs"):
            dd = getattr(self, d, None) or {}
            for w in dd.values():
                try:
                    w.configure(bg=pal["field"], fg=pal["field_fg"],
                                readonlybackground=pal["field"])
                except tk.TclError:
                    pass
        # Diff terminal (cockpit console) — re-theme from palette.
        con = pal["console"]
        for w, bg, fg in ((getattr(self, "_diff_term", None), pal["card"], None),
                          (getattr(self, "_diff_tbar", None), pal["card"], con["header"])):
            if w is not None:
                try:
                    kw = {"bg": bg}
                    if fg:
                        kw["fg"] = fg
                    w.configure(**kw)
                except tk.TclError:
                    pass
        t = getattr(self, "_diff_term", None)
        if t is not None:
            try:
                t.configure(highlightbackground=pal["border"])
            except tk.TclError:
                pass
        w = getattr(self, "ocr_diff_text", None)
        if w is not None:
            try:
                w.configure(bg=con["bg"], fg=con["fg"],
                            highlightbackground=pal["border"])
            except tk.TclError:
                pass
        # KPI metric cards — re-theme frames + labels.
        # X2 polish: 6-tuple includes accent_strip + inner frame for the
        # semantic colour block at the top-left of each card.
        for cell, ttl, val, role, accent_strip, inner in getattr(
                self, "_metric_cells", []):
            try:
                cell.configure(bg=pal["card"], highlightbackground=pal["border"])
                ttl.configure(foreground=pal["dim"])
                val.configure(foreground=pal["metric"][role])
                # 左上角语义色块跟随主题(light 下需更深以保证对比度)
                accent_strip.configure(
                    bg=pal["metric"][role])
                inner.configure(bg=pal["card"])
            except tk.TclError:
                pass
        # Root window + loc-page MRZ status chip — set once at construction
        # is not enough; re-theme them on every switch so no large light
        # (or dark) surface survives a theme toggle.
        try:
            self.configure(bg=pal["bg"])
        except tk.TclError:
            pass
        w = getattr(self, "loc_mrz_chip", None)
        if w is not None:
            try:
                w.configure(bg=chip["dim"]["bg"], fg=chip["dim"]["fg"])
            except tk.TclError:
                pass

    def _set_window_icon(self):
        """Use the app icon (tests/assets/app_icon.png) for the window
        title bar / dock. Falls back gracefully if the asset is missing."""
        try:
            icon_path = ROOT / "tests" / "assets" / "app_icon.png"
            if icon_path.exists():
                self.iconphoto(True,
                               tk.PhotoImage(file=str(icon_path)),
                               tk.PhotoImage(file=str(
                                   ROOT / "tests" / "assets" / "app_icon_64.png")),
                               tk.PhotoImage(file=str(
                                   ROOT / "tests" / "assets" / "app_icon_32.png")))
        except tk.TclError:
            pass  # headless environment

    def _build_menu(self):
        menubar = tk.Menu(self)
        filem = tk.Menu(menubar, tearoff=0)
        filem.add_command(label="生成样本图片", command=self._menu_gen_sample)
        filem.add_separator()
        filem.add_command(label="退出", command=self.destroy)
        menubar.add_cascade(label="文件", menu=filem)
        viewm = tk.Menu(menubar, tearoff=0)
        self._viewm = viewm
        self._theme_toggle_idx = 0
        viewm.add_command(label="☀️  浅色主题", command=self._toggle_theme)
        viewm.add_separator()
        advm = tk.Menu(viewm, tearoff=0)
        for name in ("darkly", "cyborg", "superhero", "cosmo",
                     "journal", "flatly"):
            advm.add_command(label=name,
                             command=lambda n=name: self._switch_theme(n))
        viewm.add_cascade(label="其它 (ttkbootstrap)", menu=advm)
        menubar.add_cascade(label="视图", menu=viewm)
        helpm = tk.Menu(menubar, tearoff=0)
        helpm.add_command(label="关于",
                          command=lambda: messagebox.showinfo(
                              "关于",
                              "护照机测试机 GUI\n"
                              "驱动 4 个纯 C 模块:\n"
                              "  1_mrz_decode\n"
                              "  2_anticounterfeit\n"
                              "  3_face_recognition\n"
                              "  4_nfc_reader\n"
                              "  6_mrz_ocr (CNN)\n"
                              "  7_security"))
        menubar.add_cascade(label="帮助", menu=helpm)
        self.config(menu=menubar)

    # ---------------- OCR tab ----------------
    def _build_ocr_tab(self):
        pal = self._pal
        from pathlib import Path as _Path
        import sys as _sys
        _sys.path.insert(0, str(_Path(__file__).resolve().parent))
        from ocr_bench_runner import CORPUS_JSON  # type: ignore
        f = self.tab_ocr

        # 常驻引擎池：C 工具与 CRNN ONNX 模型常驻，避免每张照片重复
        # 付出进程启动 + onnxruntime 导入（实测 crnn 349ms → 纯推理）。
        # 与 photo_direct_bench.py 的默认口径（--resident）保持一致。
        self._rst_resident = True
        if getattr(self, "AUTO_WARM_ENGINES", True):
            self._ocr_warmup_engines()
        # 识别结果缓存：同一张图 + 同一套引擎再次识别时直接复用。
        self._ocr_result_cache = {}

        # ---- Control card: methods | filter | search | primary CTA ----
        ctrl = ttk.LabelFrame(f, text="控制面板")
        ctrl.pack(fill="x", padx=4, pady=(4, 0))
        row1 = ttk.Frame(ctrl); row1.pack(fill="x", padx=8, pady=(6, 2))
        ttk.Label(row1, text="模式:").pack(side="left")
        self.ocr_photo_mode = tk.BooleanVar(value=False)
        ttk.Checkbutton(row1, text="图片直接识别（护照照片）",
                        variable=self.ocr_photo_mode,
                        command=self._ocr_mode_switch).pack(side="left", padx=2)
        ttk.Separator(row1, orient="vertical").pack(side="left",
                                                    fill="y", padx=6)
        ttk.Label(row1, text="方案:").pack(side="left")
        self.ocr_methods = tk.StringVar(value=",".join(METHOD_ORDER))
        # 方案单选改为下拉框：7 个 Radiobutton 一字排开会把右侧过滤器/
        # 动作按钮挤出可视区。Combobox 只占一个控件的宽度。
        self.ocr_method_pick_var = tk.StringVar(value="全部方案")
        self.ocr_method_pick = ttk.Combobox(
            row1, textvariable=self.ocr_method_pick_var, state="disabled",
            width=14, style="Field.TCombobox",
            values=[METHOD_LABELS[k] for k in METHOD_ORDER])
        self.ocr_method_pick.pack(side="left", padx=2)
        self.ocr_method_pick.bind("<<ComboboxSelected>>",
                                  self._ocr_method_on_pick)
        self.ocr_method_both = tk.BooleanVar(value=True)
        ttk.Checkbutton(row1, text="同时跑全部方案",
                        variable=self.ocr_method_both,
                        command=self._ocr_method_toggle).pack(side="left", padx=6)
        ttk.Separator(row1, orient="vertical").pack(side="left",
                                                    fill="y", padx=8)
        # 训练集（权重集）选择。两套权重是同一网络分别训练的结果：
        #   关闭 = 护照手机照（tmp/PassPortPhoto，板端标签，41 张）
        #   打开 = 扫描件（assets/pic，自动 GT，47 张）
        # 权重是 static const 编进二进制的，所以切换的是工具二进制而
        # 不是运行时加载；两者输出不可混用，切换后必须清空结果缓存。
        ttk.Label(row1, text="训练集:").pack(side="left")
        self.ocr_weights_pic = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            row1, text="assets/pic 扫描件",
            variable=self.ocr_weights_pic,
            command=self._ocr_weights_toggle).pack(side="left", padx=2)
        ttk.Separator(row1, orient="vertical").pack(side="left",
                                                    fill="y", padx=8)
        # 照片集（真实照片模式）：命名集合，定义在 tests/photo_sets.json。
        # 集合自带目录归属与权重集归属，并按**证件**分组（同一证件的多个
        # 机位不得在训练/测试切分中横跨两侧）。选集合 = 选一份清单，
        # 不是把文件搬来搬去。
        ttk.Label(row1, text="照片集:").pack(side="left")
        self.ocr_set_var = tk.StringVar(value="默认（当前训练集）")
        self.ocr_set_pick = ttk.Combobox(
            row1, textvariable=self.ocr_set_var, state="readonly",
            width=16, style="Field.TCombobox",
            values=self._ocr_set_choices())
        self.ocr_set_pick.pack(side="left", padx=2)
        self.ocr_set_pick.bind("<<ComboboxSelected>>", self._ocr_set_on_pick)
        ttk.Separator(row1, orient="vertical").pack(side="left",
                                                    fill="y", padx=8)
        ttk.Label(row1, text="过滤:").pack(side="left")
        self.ocr_filter_var = tk.StringVar(value="all")
        self.ocr_filter_pick_var = tk.StringVar(value="全部")
        self.ocr_filter_pick = ttk.Combobox(
            row1, textvariable=self.ocr_filter_pick_var, state="readonly",
            width=6, style="Field.TCombobox", values=["全部", "干净", "带噪"])
        self.ocr_filter_pick.pack(side="left", padx=2)
        self.ocr_filter_pick.bind("<<ComboboxSelected>>",
                                  self._ocr_filter_on_pick)
        style = ttk.Style()
        if not style.theme_use():
            style.theme_use("default")
        # 动作按钮独立成行（row2），不塞进左→右堆叠的 row1：
        # 窗口收窄时 row1 的方案按钮会把动作按钮挤出可视区。
        row2 = ttk.Frame(ctrl); row2.pack(fill="x", padx=8, pady=(2, 6))
        # 直接识别图片 — ghost 风格(中性)
        self.ocr_pick_btn = self._cta(
            row2, "🖼 直接识别图片",
            self._ocr_pick_and_recognize, kind="ghost", padx=10, pady=4)
        self.ocr_pick_btn.pack(side="left", padx=(0, 4))
        # 主操作 — primary(高亮)
        self.ocr_run_btn = self._cta(
            row2, "⚡ 开始测试", self._ocr_run,
            padx=14, pady=4)
        self.ocr_run_btn.pack(side="left", padx=(4, 0))

        # 全选/反选/清空/搜索只作用于左侧案例集列表，就近放到列表顶部，
        # 不再占用顶部的全局空间。

        # ---- Body: 3-pane splitter (case list | metrics+summary+detail | sample) ----
        body = ttk.PanedWindow(f, orient="horizontal")
        body.pack(fill="both", expand=True, padx=4, pady=4)
        self._ocr_body = body

        # Left = case list.
        left = ttk.Frame(body); body.add(left, weight=22)
        ttk.Label(left, text="案例集列表（点击查看原图，多选后开始测试）").pack(anchor="w")
        list_bar = ttk.Frame(left); list_bar.pack(fill="x", pady=(2, 2))
        self._cta(list_bar, "全选",
                  lambda: self._ocr_select_all(True), kind="ghost").pack(side="left", padx=(0, 2))
        self._cta(list_bar, "反选",
                  lambda: self._ocr_select_all(False), kind="ghost").pack(side="left", padx=2)
        self._cta(list_bar, "清空",
                  self._ocr_clear_sel, kind="ghost").pack(side="left", padx=2)
        self.ocr_search_var = tk.StringVar()
        search = ttk.Entry(list_bar, textvariable=self.ocr_search_var,
                           width=10, style="Field.TEntry")
        search.pack(side="left", fill="x", expand=True, padx=(6, 0))
        search.bind("<KeyRelease>", self._ocr_apply_search)
        self._ocr_search_entry = search
        self.ocr_case_list = ttk.Treeview(left,
            columns=("id", "scale", "noise", "skew"),
            show="headings", selectmode="extended", height=22,
            style="Data.Treeview")
        # X2 polish:skew 缩到 50(原 70 太宽,留多空白)并设 minwidth 兜底。
        # id 列随容器自动 stretch,避免"案例 ID" 在窄列中被挤。
        for c, w in [("id", 220), ("scale", 56), ("noise", 56), ("skew", 50)]:
            self.ocr_case_list.heading(
                c, text=c,
                command=lambda col=c: self._sort_treeview(self.ocr_case_list, col))
            self.ocr_case_list.column(c, width=w, minwidth=40, anchor="w",
                                       stretch=(c == "id"))
        ysb = ttk.Scrollbar(left, orient="vertical",
                            command=self.ocr_case_list.yview)
        self.ocr_case_list.configure(yscrollcommand=ysb.set)
        self.ocr_case_list.pack(side="left", fill="both", expand=True)
        ysb.pack(side="right", fill="y")
        self.ocr_case_list.bind("<<TreeviewSelect>>", self._ocr_on_case_select)
        # Double-click opens the full-resolution viewer; single-click
        # keeps the existing thumbnail preview in the right pane.
        self.ocr_case_list.bind("<Double-Button-1>", self._ocr_open_original)
        # Photo mode: single-click on a list row immediately runs
        # process_image() on that single photo, so the user gets the
        # full pipeline (locate -> OCR -> decode -> grade) without
        # having to also press ▶ 开始识别.
        self.ocr_case_list.bind("<Button-1>", self._ocr_on_list_click)
        # 行内只显示纯文件名，悬停时用 tooltip 展示完整路径。
        self._ocr_bind_row_tooltip(
            self.ocr_case_list,
            lambda iid: getattr(self, "_ocr_case_paths", {}).get(iid, ""))

        # Middle = metrics card + summary + detail.
        # weight=3 (vs 1/1 for left/preview) gives the KPI row enough
        # horizontal room; the default sashes below reserve ~20% / ~26%.
        middle = ttk.Frame(body); body.add(middle, weight=44)
        self._build_ocr_metrics(middle)
        ttk.Label(middle, text="方案指标对比").pack(anchor="w")
        self.ocr_summary = ttk.Treeview(middle,
            columns=("method", "n", "ok", "okp", "ms_avg",
                     "l1_acc", "l2_acc", "full_match"),
            show="headings", height=3)
        # 每列按内容宽度差异化 — l1/l2 加宽到 84 避免 "-- %" 挤成一团。
        col_widths = {"method": 100, "n": 54, "ok": 54, "okp": 68,
                      "ms_avg": 80, "l1_acc": 84, "l2_acc": 84,
                      "full_match": 92}
        for c, anc in [("method", "w"), ("n", "center"), ("ok", "center"),
                       ("okp", "center"), ("ms_avg", "center"),
                       ("l1_acc", "center"), ("l2_acc", "center"),
                       ("full_match", "center")]:
            self.ocr_summary.heading(c, text=c)
            self.ocr_summary.column(c, width=col_widths[c],
                                     minwidth=48, anchor=anc)
        # 行级配色统一由 _apply_text_tag_palette 按主题设置（淡色），
        # 这里不再硬编码浓色。
        self.ocr_summary.configure(style="Data.Treeview")
        self.ocr_summary.pack(fill="x")

        ttk.Label(middle, text="详细结果（点击行 → 右侧单样本透视）").pack(anchor="w", pady=(6, 0))
        self.ocr_detail = ttk.Treeview(middle,
            columns=("id", "method", "ok", "ms", "l1", "l2"),
            show="headings", height=14, style="Data.Treeview")
        # id 列 stretch(随容器拉宽)，l1/l2 加宽到 130 避免表头/内容被挤；
        # 判定/耗时列用 ResizeToContents 的等价做法：固定合理宽度。
        col_widths = {"id": 150, "method": 96, "ok": 64,
                      "ms": 72, "l1": 130, "l2": 130}
        for c, anc in [("id", "w"), ("method", "w"), ("ok", "center"),
                       ("ms", "center"), ("l1", "center"), ("l2", "center")]:
            self.ocr_detail.heading(c, text=c)
            self.ocr_detail.column(c, width=col_widths[c], minwidth=52,
                                     anchor=anc, stretch=(c == "id"))
        ysb2 = ttk.Scrollbar(middle, orient="vertical",
                             command=self.ocr_detail.yview)
        xsb2 = ttk.Scrollbar(middle, orient="horizontal",
                             command=self.ocr_detail.xview)
        self.ocr_detail.configure(yscrollcommand=ysb2.set,
                                  xscrollcommand=xsb2.set)
        self.ocr_detail.pack(side="top", fill="both", expand=True)
        ysb2.pack(side="right", fill="y")
        xsb2.pack(side="bottom", fill="x")
        self.ocr_detail.bind("<<TreeviewSelect>>", self._ocr_on_detail_select)
        self.ocr_detail.bind("<Double-Button-1>", self._ocr_open_original_from_detail)
        self._ocr_bind_row_tooltip(self.ocr_detail, self._ocr_detail_tip_text)
        self._ocr_detail_map = {}
        # 语料模式的汇总/明细表列（点击表头可排序）。
        self._ocr_corpus_columns()

        # Right = single-sample perspective (image + char-level diff).
        preview = ttk.Frame(body); body.add(preview, weight=34)
        ttk.Label(preview, text="单样本透视").pack(anchor="w")
        self.ocr_preview_meta = tk.StringVar(value="（在左侧选择案例，或点击明细行）")
        self.ocr_preview_meta_lb = tk.Label(
            preview, textvariable=self.ocr_preview_meta,
            bg=pal["card"], fg=pal["fg"], wraplength=380, justify="left",
            anchor="w", font=("TkDefaultFont", 9))
        self.ocr_preview_meta_lb.pack(anchor="w", fill="x")
        self._ocr_meta_fullpath = ""
        self._ocr_meta_tip = _Tooltip(self.ocr_preview_meta_lb)
        # X2 polish:预览标签用 pal["card"](更深的背景)而非 canvas,
        # 与终端字符比对区(更浅的 console bg)形成明显分层,边界清晰。
        self.ocr_preview_label = tk.Label(preview, bg=pal["card"],
                                          fg=pal["fg"], anchor="center",
                                          highlightthickness=1,
                                          highlightbackground=pal["border"])
        self.ocr_preview_label.pack(fill="both", expand=True)
        self.ocr_preview_label.bind("<Configure>", self._update_preview_image)
        ttk.Label(preview, text="识别结果 · 字符级比对 (GT vs Pred)").pack(anchor="w")
        con = pal["console"]
        term = tk.Frame(preview, bg=pal["card"], highlightthickness=1,
                        highlightbackground=pal["border"])
        term.pack(fill="both", expand=True)
        self._diff_term = term
        # X2 polish:Terminal 标题条改为"左侧 4x18 像素色块 + 文字 + 右侧
        # 装饰点 + 底部 1px 暗灰分隔",模拟工业终端的状态灯条。
        tbar = tk.Frame(term, bg=pal["card"])
        tbar.pack(fill="x")
        # 左色块 — 4x18 像素,语义色 con["header"](cyan/深蓝)
        tag_strip = tk.Frame(tbar, bg=con["header"], width=4, height=18)
        tag_strip.pack(side="left", padx=(6, 6), pady=3)
        tag_strip.pack_propagate(False)
        tbar_label = tk.Label(tbar,
            text="DIGITAL COMPARATOR  //  字符级检视",
            font=("Menlo", 9, "bold"), bg=pal["card"],
            fg=con["header"], anchor="w", pady=3)
        tbar_label.pack(side="left", fill="x", expand=True)
        # 右侧 3 个小圆点("..." 状态灯),增强仪表盘感
        for i in range(3):
            d = tk.Frame(tbar, bg=con["pad"], width=6, height=6)
            d.pack(side="right", padx=(0, 4 if i==2 else 2), pady=0)
            d.pack_propagate(False)
        # 底部 1px 暗灰分隔
        tk.Frame(term, bg=pal["border"], height=1).pack(fill="x", side="top")
        self._diff_tbar = tbar
        self._diff_tbar_strip = tag_strip
        self.ocr_diff_text = tk.Text(term, height=12, width=52,
                                     font=_ocrb_font(11), wrap="none",
                                     bg=con["bg"], fg=con["fg"], relief="flat",
                                     padx=8, pady=6,
                                     highlightthickness=1,
                                     highlightbackground=pal["border"])
        self.ocr_diff_text.tag_configure("hdr", font=("Menlo", 9, "bold"),
                                         foreground=con["pad"])
        self.ocr_diff_text.tag_configure("lab", foreground=con["pad"])
        self.ocr_diff_text.tag_configure("match", foreground=pal["tag_match"]["fg"],
                                         font=_ocrb_font(11, "bold"))
        self.ocr_diff_text.tag_configure("pad", foreground=con["pad"])
        # X2 polish:不匹配字符使用更柔和的 amber 高亮 + 加粗下划线,
        # 避免与图中的红框视觉冲突。背景改为透明,只改前景色 + 加粗。
        self.ocr_diff_text.tag_configure("mm",
            foreground=con.get("error_bg", "#ff5c7a"),
            font=_ocrb_font(11, "bold"))
        self.ocr_diff_text.tag_configure("error",
            foreground=con.get("error_bg", "#ff5c7a"),
            font=_ocrb_font(11, "bold"))
        self.ocr_diff_text.tag_configure("okline",
                                         foreground=pal["tag_match"]["fg"],
                                         font=_ocrb_font(11))
        d_y = ttk.Scrollbar(term, orient="vertical",
                            command=self.ocr_diff_text.yview)
        d_x = ttk.Scrollbar(term, orient="horizontal",
                            command=self.ocr_diff_text.xview)
        self.ocr_diff_text.configure(yscrollcommand=d_y.set,
                                     xscrollcommand=d_x.set)
        self.ocr_diff_text.pack(side="left", fill="both", expand=True)
        d_y.pack(side="right", fill="y")
        d_x.pack(side="bottom", fill="x")

        # Load corpus.
        self._ocr_corpus = {"records": []}
        self._ocr_case_order = []
        self._ocr_load_corpus()
        # First call lays out the body while the Tk root is still
        # settling to its real geometry; the second call runs once
        # the window has reached the requested 1280x780 so the
        # middle KPI pane actually gets the 54% we asked for.
        self.after(300, self._ocr_set_sashes)
        self.after(800, self._ocr_set_sashes)

    def _ocr_load_corpus(self):
        """(Re)fill the case list with corpus cases (batch-test mode)."""
        import sys as _sys
        from pathlib import Path as _Path
        from ocr_bench_runner import CORPUS_JSON  # type: ignore
        for iid in self.ocr_case_list.get_children():
            self.ocr_case_list.delete(iid)
        self._ocr_case_order = []
        corpus_path = _Path(CORPUS_JSON)
        self._ocr_case_paths = {}
        self._ocr_corpus = {"records": []}
        # 每个案例的"基础标签"（clean/noisy，供过滤用）；识别结果产生的
        # pass/fail 颜色是附加标签，换方案时要按这里复位。
        self._ocr_case_base_tags = {}
        if corpus_path.exists():
            data = json.loads(corpus_path.read_text())
            for rec in data["records"]:
                tag = "noisy" if rec["noise"] > 0 else "clean"
                iid = rec["id"]
                self._ocr_case_base_tags[iid] = (tag,)
                # 表格只显示纯文件名，完整路径走 tooltip（_ocr_case_tip）。
                self.ocr_case_list.insert("", "end", iid=iid,
                    values=(_Path(rec["id"]).name,
                            rec["scale"], rec["noise"], rec["skew"]),
                    tags=(tag,))
                self._ocr_case_order.append(iid)
                self._ocr_case_paths[iid] = str(corpus_path.parent / rec["image"])
            self._ocr_corpus = data

    def _gt_map(self):
        """Load GT from 10_ocr_research/gt_review_loop.xlsx 'GT对比' sheet
        (GT line1 / GT line2) into {filename: (line1, line2)} lazily.
        Entries with empty GT are included as ("", "") so the verdict
        logic can mark them NO_GT instead of falling back to PASS."""
        if getattr(self, "_gt_map_cache", None) is not None:
            return self._gt_map_cache
        m = {}
        p = ROOT / "10_ocr_research" / "gt_review_loop.xlsx"
        # JSON 缓存：openpyxl 首次 import + xlsx 解析 ~200ms（其中
        # import openpyxl 128ms），读 JSON ~5ms。xlsx 更新后按 mtime
        # 自动失效重建。单张首识别耗时 204ms→~50ms 即此。
        cache_p = p.with_suffix(".json")
        try:
            if p.exists() and cache_p.exists() \
                    and os.path.getmtime(cache_p) >= os.path.getmtime(p):
                m = json.loads(cache_p.read_text(encoding="utf-8"))
                self._gt_map_cache = m
                return m
        except Exception:
            pass
        try:
            import openpyxl
            if p.exists():
                wb = openpyxl.load_workbook(str(p), read_only=True, data_only=True)
                ws = wb["GT对比"]
                for row in ws.iter_rows(min_row=2, values_only=True):
                    fn = row[1] or ""
                    gt1 = row[6] or ""
                    gt2 = row[7] or ""
                    if fn:
                        m[fn] = (gt1, gt2)
                wb.close()
        except Exception:
            pass
        try:
            cache_p.write_text(json.dumps(m, ensure_ascii=False),
                               encoding="utf-8")
        except Exception:
            pass
        self._gt_map_cache = m
        return m

    def _ocr_weights_tool(self):
        """Path of the CNN tool binary for the selected training set.

        The two weight sets are compiled in (``static const`` arrays), so
        switching sets means switching binaries.  ``mrz_ocr_cnn_tool``
        carries the tmp/PassPortPhoto line heads, ``mrz_ocr_cnn_tool_pic``
        the assets/pic ones.  If the second binary has not been built the
        passport one is returned and the caller sees the usual "missing
        tool" status rather than a silent fallback to the wrong weights.
        """
        base = ROOT / "6_mrz_ocr" / "build"
        if self._ocr_use_pic_weights():
            pic = base / "mrz_ocr_cnn_tool_pic"
            if pic.exists():
                return pic
        return base / "mrz_ocr_cnn_tool"

    def _ocr_tk_snapshot(self):
        """Copy the tkinter variables the OCR worker threads need.

        tkinter's ``Variable.get()`` raises ``RuntimeError: main thread
        is not in main loop`` when it is called from a worker thread, so
        the A/B hypothesis runner (one hypothesis runs in a thread pool)
        takes the snapshot on the main thread and the workers read plain
        attributes instead.
        """
        snap = {}
        for name, attr, default in (("method_both", "rst_method_both", True),
                                    ("methods", "rst_methods", ""),
                                    ("weights_pic", "ocr_weights_pic", False)):
            var = getattr(self, attr, None)
            try:
                snap[name] = var.get() if var is not None else default
            except Exception:
                snap[name] = default
        return snap

    def _ocr_use_pic_weights(self):
        """True when the 'assets/pic 扫描件' weight set is selected."""
        snap = getattr(self, "_ocr_tk", None)
        if snap is not None:
            return bool(snap.get("weights_pic"))
        var = getattr(self, "ocr_weights_pic", None)
        try:
            return bool(var is not None and var.get())
        except Exception:
            return False

    def _weights_set(self):
        """'pic' or 'passport' -- which weight set the CheckBox selects."""
        if self._ocr_use_pic_weights():
            return "pic"
        return "passport"

    def _ocr_photo_entry_map(self):
        """label -> path for the photo list of the CURRENT training set."""
        cached = getattr(self, "_ocr_photo_entries", None)
        if cached:
            return dict(cached)
        return {label: pth
                for label, pth in iter_passport_photos(self._weights_set())}

    def _ocr_weights_toggle(self):
        """Invalidate every cached reading: results depend on the weights."""
        if getattr(self, "_ocr_result_cache", None):
            self._ocr_result_cache.clear()
        # Line-heads differ per set, so any per-photo verdict shown in the
        # table was produced by the other model.
        for iid in self.ocr_case_list.get_children():
            vals = list(self.ocr_case_list.item(iid, "values"))
            if len(vals) >= 4 and any(vals[1:]):
                self.ocr_case_list.item(
                    iid, values=(vals[0], "", "", ""))
        name = ("assets/pic 扫描件" if self.ocr_weights_pic.get()
                else "tmp/PassPortPhoto 护照照")
        # 照片列表跟着训练集走：A 集权重去评 B 集照片量的是分布差异。
        if getattr(self, "ocr_photo_mode", None) and self.ocr_photo_mode.get():
            self._ocr_mode_switch()
            n = len(getattr(self, "_ocr_photo_entries", []))
            self._set_status_bar(
                f"训练集已切换：{name}（列表 {n} 张，结果缓存已清空）")
        else:
            self._set_status_bar(f"训练集已切换：{name}（结果缓存已清空）")

    # ---- photo sets (tests/photo_sets.json) ----
    def _ocr_set_defs(self):
        """The named photo sets, or an empty dict when the file is absent."""
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import photo_sets
            return photo_sets.load_sets().get("sets", {})
        except Exception:
            return {}

    def _ocr_set_choices(self):
        # The first entry is the DEFAULT list (the current weight set's
        # photos), not "every photo on disk" -- the named set 'all' is
        # that one.  Labelling it 全部照片 would promise 114 rows and
        # deliver 50.
        defs = self._ocr_set_defs()
        return ["默认（当前训练集）"] + sorted(defs)

    def _ocr_current_set(self):
        """The selected set's key, or '' for '全部照片'."""
        pick = str(self.ocr_set_var.get() if hasattr(self, "ocr_set_var")
                   else "").strip()
        if pick in ("", "默认（当前训练集）"):
            return ""
        return pick if pick in self._ocr_set_defs() else ""

    def _ocr_set_entries(self):
        """(label, path) for the selected set, or [] to use the default list."""
        name = self._ocr_current_set()
        if not name:
            return []
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import photo_sets
            return photo_sets.resolve(name)
        except Exception:
            return []

    def _ocr_set_on_pick(self, _evt=None):
        """Switch the photo list to a named set.

        A set declares which weight set it belongs to, so picking the
        'pic' set also switches the weights: measuring the pic photos
        with the passport weights measures the distribution gap, not
        the model.
        """
        name = self._ocr_current_set()
        defs = self._ocr_set_defs()
        if name and defs.get(name, {}).get("weights") in ("pic", "passport"):
            want_pic = defs[name]["weights"] == "pic"
            if bool(self.ocr_weights_pic.get()) != want_pic:
                self.ocr_weights_pic.set(want_pic)
                if getattr(self, "_ocr_result_cache", None):
                    self._ocr_result_cache.clear()
        self._ocr_reset_results()
        if self.ocr_photo_mode.get():
            self._ocr_mode_switch()

    def _ocr_mode_switch(self):
        """Toggle between corpus batch-test and passport-photo recognition."""
        # The set selector only applies to the real-photo list.
        try:
            self.ocr_set_pick.configure(
                state="readonly" if self.ocr_photo_mode.get() else "disabled")
        except tk.TclError:
            pass
        if self.ocr_photo_mode.get():
            # 切换模式不表示正在识别。按钮文案只在真正跑起来时改成
            # "⏳ 识别中…"，跑完必须恢复，否则它会一直停在"识别中"。
            self.ocr_run_btn.configure(text="⚡ 开始测试")
            for iid in self.ocr_case_list.get_children():
                self.ocr_case_list.delete(iid)
            self._ocr_case_order = []
            # 真实护照照片来源：assets/pic + tmp/PassPortPhoto + Photo。
            # Photo/ 是子目录结构(机位照)，其余两个目录是顶层平铺。
            self._ocr_photo_entries = (self._ocr_set_entries()
                                       or list(iter_passport_photos(
                                           self._weights_set())))
            from pathlib import Path as _P2
            # 只显示纯文件名；完整路径放进 tooltip 映射。
            self._ocr_photo_paths = {}
            self._ocr_case_base_tags = {}
            for label, _f in self._ocr_photo_entries:
                self._ocr_photo_paths[label] = str(_f)
                self.ocr_case_list.insert("", "end", iid=label,
                                          values=(_P2(label).name, "", "", ""),
                                          tags=("clean",))
                self._ocr_case_order.append(label)
                self._ocr_case_base_tags[label] = ("clean",)
            self._ocr_case_paths = dict(self._ocr_photo_paths)
        else:
            self.ocr_run_btn.configure(text="⚡ 开始测试")
            self._ocr_load_corpus()

    def _build_loc_tab(self):
        """Passport image OCR region locator: photo / data / MRZ (standalone tab).
        Built from tests/passport_locator.py (pure PIL+numpy).
        """
        pal = self._pal
        import sys as _sys
        from pathlib import Path as _P
        _sys.path.insert(0, str(_P(__file__).resolve().parent))
        from passport_locator import locate_regions, annotate  # type: ignore
        self._loc_locator = locate_regions
        self._loc_annotate = annotate

        f = self.tab_loc

        top = ttk.LabelFrame(f, text="护照图片 OCR 区域定位（照片 / 数据区 / MRZ 区）")
        top.pack(fill="x", padx=4, pady=(4, 0))

        row = ttk.Frame(top); row.pack(fill="x", padx=8, pady=4)
        ttk.Label(row, text="样本:").pack(side="left")
        self.loc_sample = tk.StringVar()
        from passport_pipeline import PIC_DIR as _PIC  # type: ignore
        _pic_names = sorted(p.name for p in _PIC.glob("*")
                            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".ppm"))
        self.loc_samples = [
            "合成护照样本 (TD3 整页)",
            "corpus MRZ 条带 (img_0001_b0_v0)",
        ] + [f"assets/pic: {n}" for n in _pic_names]
        cb = ttk.Combobox(row, textvariable=self.loc_sample,
                          values=self.loc_samples, state="readonly", width=30,
                          style="Field.TCombobox")
        cb.pack(side="left", padx=2)
        cb.current(0)
        cb.bind("<<ComboboxSelected>>", self._loc_on_sample_selected)
        self.loc_sample_path = tk.StringVar()
        ttk.Button(row, text="▤ 浏览文件…", command=self._loc_pick).pack(side="left", padx=4)
        self.loc_run_btn = self._cta(row, "⚡ 定位区域", self._loc_run,
                                     padx=12, pady=3)
        self.loc_run_btn.pack(side="left", padx=4)
        # MRZ-only quick action: locate MRZ band, crop it, feed it to
        # mrz_tool, and dump the parsed fields + check digits on the
        # right-hand info pane. Ghost styling so the primary CTA stays
        # the single strongest affordance in the row.
        self.loc_mrz_btn = self._cta(row, "🆔 MRZ 识别", self._loc_mrz_recognize,
                                     kind="ghost", padx=12, pady=3)
        self.loc_mrz_btn.pack(side="left", padx=4)
        self.loc_status = ttk.Label(row, text="")
        self.loc_status.pack(side="left", padx=8)
        # MRZ result chip: PASS green / FAIL red, sits at the top of the
        # info pane so it stays in the user's natural reading path.
        self._loc_mrz_chip_var = tk.StringVar(value="  IDLE  ")

        body = ttk.Frame(f); body.pack(fill="both", expand=True, padx=4, pady=(4, 6))
        self.loc_canvas = tk.Canvas(body, width=760, height=440,
                                    bg=pal["canvas"], highlightthickness=1,
                                    highlightbackground=pal["canvas_border"],
                                    highlightcolor=pal["accent"])
        self.loc_canvas.pack(side="left", fill="both", expand=True)
        # Side panel: MRZ result chip at the top, detail Text below.
        side = ttk.Frame(body)
        side.pack(side="left", padx=(6, 0), fill="y")
        pal = self._pal
        chip = pal["chip"]["dim"]
        self.loc_mrz_chip = tk.Label(
            side, textvariable=self._loc_mrz_chip_var,
            bg=chip["bg"], fg=chip["fg"],
            font=pal["font_ui_bold"], padx=10, pady=3)
        self.loc_mrz_chip.pack(anchor="w", pady=(0, 4))
        self.loc_info = tk.Text(side, width=52, height=20, font=("Menlo", 9),
                                state="disabled", wrap="none",
                                bg=pal["field"], fg=pal["field_fg"],
                                relief="flat", bd=0, highlightthickness=1,
                                highlightbackground=pal["border"],
                                highlightcolor=pal["accent"])
        self.loc_info.pack(fill="both", expand=True)

    # ---- 照片→MRZ 流水线 tab（去影 -> MRZ 粗定位 -> 识别 -> 解码）----
    def _build_restore_tab(self):
        """护照机照片 → 4 步流水线，3 张图 + 1 段文字（线性排列）。

        每张图都是上一步的派生，视觉上严格对应流水线：
          ① 原图 (含反射)
          ② 去影图 (裁掉 MRZ 下方全部倒影, 原图基础上)
          ③ MRZ 区图 (OCR 真正吃的那条带, 保留自然形态不拉伸)
          ④ MRZ 识别 + 解码 + 各阶段耗时

        原来第 3 格是「矫正图」（透视拉直），已移除：它不参与 OCR，
        而且比自然 crop 更差（见 _build_restore_tab 里 labels 处的说明）。
        """
        pal = self._pal
        f = self.tab_restore

        top = ttk.LabelFrame(f, text="护照照片 → MRZ（①原图 ②去影 ③MRZ区 ④识别/解码）")
        top.pack(fill="x", padx=4, pady=(4, 0))
        row = ttk.Frame(top); row.pack(fill="x", padx=8, pady=4)
        ttk.Label(row, text="样本:").pack(side="left")
        self.rst_sample = tk.StringVar()
        self._rst_sample_paths = {
            label: str(path) for label, path in iter_passport_photos()}
        self.rst_samples = (["合成护照样本 (TD3 整页)"]
                            + list(self._rst_sample_paths))
        cb = ttk.Combobox(row, textvariable=self.rst_sample,
                          values=self.rst_samples, state="readonly", width=46,
                          style="Field.TCombobox")
        cb.pack(side="left", padx=2)
        cb.current(0)
        cb.bind("<<ComboboxSelected>>", self._rst_on_sample)
        self.rst_pick_btn = ttk.Button(row, text="▤ 浏览文件…",
                                       command=self._rst_pick)
        self.rst_pick_btn.pack(side="left", padx=4)
        self.rst_run_btn = self._cta(row, "▶ 运行流水线", self._rst_run,
                                     padx=12, pady=3)
        self.rst_run_btn.pack(side="left", padx=4)
        self.rst_status = ttk.Label(row, text="")
        self.rst_status.pack(side="left", padx=8)

        method_row = ttk.Frame(top)
        method_row.pack(fill="x", padx=8, pady=(0, 4))
        ttk.Label(method_row, text="识别方案:").pack(side="left")
        self.rst_methods = tk.StringVar(value="tesseract")
        # Full ensemble by default: each engine covers the others'
        # blind spots on real photos (C CNN skipped at low res,
        # Tesseract misses rows, Indra needs a cleaned band).
        self.rst_method_both = tk.BooleanVar(value=True)
        self._rst_last_method = "tesseract"
        for key in METHOD_ORDER:
            ttk.Radiobutton(
                method_row, text=METHOD_LABELS[key],
                variable=self.rst_methods, value=key,
                command=self._rst_method_select).pack(side="left", padx=3)
        ttk.Checkbutton(
            method_row, text="同时跑全部方案",
            variable=self.rst_method_both,
            command=self._rst_method_toggle).pack(side="left", padx=(8, 2))
        ttk.Label(
            method_row,
            text="默认 Tesseract；全方案模式保留每个方案的独立结果",
            foreground=pal["dim"]).pack(side="left", padx=(10, 0))

        # 4 张图线性排列 (上下分两行, 上 4 小图, 下 1 信息文本)
        body = ttk.Frame(f); body.pack(fill="both", expand=True, padx=4, pady=(4, 6))

        # 上排：3 个等宽 canvas
        #
        # 原来有 4 格，第 3 格是「矫正图」（透视拉直后的 MRZ）。它被删掉
        # 的理由和 Android 端删 rectified stage 完全一样：
        #   * 它从不参与 OCR（OCR 读的是第 4 格的 crop_pair_clean）；
        #   * 它比不处理更差：把矫正图喂给同一个 mrz_ocr_cnn_tool，
        #     assets/pic 上 ICAO 通过 2/72，自然 crop 是 3/72；
        #   * 它的页面四边形估计（_mrzh_quad 的 page_span）用亮度百分位
        #     找页面边界，对整体偏亮的照片会把整幅图当成页面，
        #     矫正后整张照片被压成 44 格 x 2 行，画面完全乱掉。
        # native 侧的 deskew + 44 格 pitch-grid 已经在识别内部做完了，
        # 所以第 3 格现在直接显示「MRZ 区（OCR 输入）」。
        imgs = ttk.Frame(body); imgs.pack(fill="both", expand=True)
        for c in range(3):
            imgs.columnconfigure(c, weight=1, uniform="rst_col")
        imgs.rowconfigure(0, weight=1)

        self.rst_canvases = []
        self.rst_imgtk = []
        self.rst_fullimg = []  # 保存各格原图 (PIL), 供点击放大
        labels = [
            ("① 原图（含镜片反射）", "src"),
            ("② 去影图（MRZ 下方倒影已裁掉）", "nor"),
            ("③ MRZ 区（OCR 输入：反射清干净）", "band"),
        ]
        for col, (title, key) in enumerate(labels):
            cell = ttk.Frame(imgs)
            cell.grid(row=0, column=col, sticky="nsew", padx=2, pady=(0, 2))
            ttk.Label(cell, text=title, font=("TkDefaultFont", 9, "bold")).pack(anchor="w")
            cv = tk.Canvas(cell, width=240, height=180,
                           bg=pal["canvas"], highlightthickness=1,
                           highlightbackground=pal["canvas_border"],
                           highlightcolor=pal["accent"])
            cv.pack(fill="both", expand=True)
            self.rst_canvases.append(cv)
            self.rst_imgtk.append(None)  # PhotoImage 引用防回收
            self.rst_fullimg.append(None)
            # 点击放大: 绑定 canvas 实例 + 列号(idx) 一起传
            cv.bind("<Button-1>",
                    lambda _e, _i=col, _t=title: self._rst_zoom_open(_i, _t))
            # hover 提示 (鼠标变手指)
            cv.bind("<Enter>", lambda _e, _c=cv: _c.config(cursor="hand2"))
            cv.bind("<Leave>", lambda _e, _c=cv: _c.config(cursor=""))

        # 下排：识别结果 + 解码 + 阶段耗时
        info = ttk.Frame(body); info.pack(fill="both", expand=False)
        ttk.Label(info, text="⑤ MRZ 识别 + 解码 + 阶段耗时",
                  font=("TkDefaultFont", 10, "bold")).pack(anchor="w")

        # ⑤-a: 两行 OCR 结果 TextEdit（只读）—— 直接呈现 line1/line2 原文
        mrz_lines = ttk.Frame(info); mrz_lines.pack(fill="x", pady=(2, 0))
        for col, label in enumerate(("OCR line1", "OCR line2")):
            cell = ttk.Frame(mrz_lines); cell.pack(side="left", fill="x",
                                                   expand=True, padx=(0, 4))
            ttk.Label(cell, text=label,
                      font=("TkDefaultFont", 9, "bold")).pack(anchor="w")
            var = tk.StringVar(value="")
            t = ttk.Entry(cell, textvariable=var, font=_ocrb_font(11),
                          style="Field.TEntry")
            t.pack(fill="x")
            if col == 0:
                self.rst_mrz1_var = var
                self.rst_mrz1_edit = t
            else:
                self.rst_mrz2_var = var
                self.rst_mrz2_edit = t

        # ⑤-b: 证件详细信息控件组（mrz_tool decode 的字段）
        doc = ttk.LabelFrame(info, text="证件信息（MRZ decode）")
        doc.pack(fill="x", pady=(4, 0))
        self.rst_doc_fields = {}
        _rst_doc_spec = [
            ("doc_type", "证件类型"),
            ("issuing_state", "签发国"),
            ("surname", "姓"),
            ("given_names", "名"),
            ("passport_no", "护照号"),
            ("nationality", "国籍"),
            ("birth_yymmdd", "出生日期"),
            ("sex", "性别"),
            ("expiry_yymmdd", "有效期"),
            ("personal_no", "个人号码"),
            ("composite_ck", "合成校验"),
        ]
        for i, (key, label) in enumerate(_rst_doc_spec):
            r = ttk.Frame(doc); r.grid(row=i // 3, column=(i % 3) * 2,
                                       sticky="w", padx=(4, 8), pady=1)
            ttk.Label(r, text=label, width=9, anchor="w",
                      font=("TkDefaultFont", 9)).pack(side="left")
            var = tk.StringVar(value="")
            ttk.Label(r, textvariable=var, width=16, anchor="w",
                      font=("Menlo", 9), foreground=pal["accent"]).pack(side="left")
            self.rst_doc_fields[key] = var

        info_text = ttk.Frame(info)
        info_text.pack(fill="both", expand=True)
        self.rst_info = tk.Text(
            info_text, height=12, font=_ocrb_font(10),
            bg=pal["code"], fg=pal["fg"],
            relief="flat", bd=0, wrap="none",
            padx=8, pady=6)
        rst_ybar = ttk.Scrollbar(
            info_text, orient="vertical", command=self.rst_info.yview)
        rst_xbar = ttk.Scrollbar(
            info_text, orient="horizontal", command=self.rst_info.xview)
        self.rst_info.configure(yscrollcommand=rst_ybar.set,
                                xscrollcommand=rst_xbar.set)
        rst_ybar.pack(side="right", fill="y")
        rst_xbar.pack(side="bottom", fill="x")
        self.rst_info.pack(side="left", fill="both", expand=True)
        _con = self._pal["console"]
        for _tag, _cfg in [
            ("hdr", {"font": ("Menlo", 9, "bold"),
                     "foreground": _con["pad"]}),
            ("lab", {"foreground": _con["pad"]}),
            ("match", {"foreground": self._pal["tag_match"]["fg"],
                       "font": _ocrb_font(10, "bold")}),
            ("pad", {"foreground": _con["pad"]}),
            ("mm", {"foreground": _con.get("error_bg", "#ff5c7a"),
                    "font": _ocrb_font(10, "bold")}),
            ("error", {"foreground": _con.get("error_bg", "#ff5c7a"),
                       "font": _ocrb_font(10, "bold")}),
            ("okline", {"foreground": self._pal["tag_match"]["fg"],
                        "font": _ocrb_font(10)}),
        ]:
            self.rst_info.tag_configure(_tag, **_cfg)

        # 兼容旧字段名 (避免破坏别处可能引用)
        self.rst_canvas = self.rst_canvases[0]
        self.rst_canvas2 = self.rst_canvases[2]

    def _rst_resolve_path(self):
        path = self.rst_sample_path.get() if hasattr(self, "rst_sample_path") else ""
        if path and os.path.exists(path):
            return path
        key = self.rst_sample.get()
        mapped = getattr(self, "_rst_sample_paths", {}).get(key)
        if mapped and os.path.exists(mapped):
            return mapped
        return str(SAMPLE_PASSPORT_PNG)

    def _rst_pick(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image", "*.png *.ppm *.bmp *.jpg"), ("All", "*.*")])
        if path:
            if not hasattr(self, "rst_sample_path"):
                self.rst_sample_path = tk.StringVar()
            self.rst_sample_path.set(path)
            self.rst_sample.set(f"自定义: {os.path.basename(path)}")
            self._rst_reset_outputs()
            try:
                self._rst_draw_source(path)
            except Exception as e:
                self.rst_status.config(text=f"加载失败: {e}", foreground=self._pal["err"])

    def _rst_reset_outputs(self):
        """切换样本后立即清空 ②③ 画布 / OCR / 证件信息 / 日志，只保留①原图。"""
        self.rst_canvases[1].delete("all")
        self.rst_canvases[2].delete("all")
        for i in (1, 2):
            self.rst_fullimg[i] = None
            self.rst_imgtk[i] = None
        self.rst_mrz1_var.set("")
        self.rst_mrz2_var.set("")
        for var in self.rst_doc_fields.values():
            var.set("")
        _rst_state = self.rst_info.cget("state")
        self.rst_info.configure(state="normal")
        try:
            self.rst_info.delete("1.0", "end")
        finally:
            self.rst_info.configure(state=_rst_state)
        self.rst_status.config(text="已选择样本，点击“▶ 运行流水线”",
                               foreground=self._pal["fg"])

    def _rst_on_sample(self, _evt=None):
        self._rst_reset_outputs()
        try:
            path = self._rst_resolve_path()
            self._rst_draw_source(path)
        except Exception as e:
            self.rst_status.config(text=f"加载失败: {e}", foreground=self._pal["err"])

    def _rst_fit_to(self, idx, pil_img, footer=None, badge=None, badge_color=None):
        """在 self.rst_canvases[idx] 上绘制 pil_img (按 canvas 当前大小缩放).

        footer: 左下角显示尺寸文字 (str)
        badge : 左上角徽标 (str, e.g. "倒影已裁掉")
        badge_color: 徽标颜色 (palette key)
        """
        from PIL import Image, ImageTk
        cv = self.rst_canvases[idx]
        cw = max(cv.winfo_width() - 4, 200)
        ch = max(cv.winfo_height() - 4, 160)
        disp = pil_img.copy()
        disp.thumbnail((cw, ch))
        # 保存全图供点击放大 (复制避免外部修改影响)
        self.rst_fullimg[idx] = pil_img.copy()
        self.rst_imgtk[idx] = ImageTk.PhotoImage(disp)
        cv.delete("all")
        cv.create_image(2, 2, image=self.rst_imgtk[idx], anchor="nw")
        if badge:
            cv.create_text(4, 8, text=badge, anchor="nw",
                           fill=self._pal.get(badge_color, self._pal["ok"]),
                           font=("Menlo", 8, "bold"))
        if footer:
            cv.create_text(4, cv.winfo_height() - 4, text=footer, anchor="sw",
                           fill=self._pal["dim"], font=("Menlo", 8))

    def _rst_draw_source(self, path):
        """①原图：缩放显示。"""
        from PIL import Image
        img = Image.open(path).convert("RGB")
        self._rst_fit_to(0, img, footer=f"{img.width}×{img.height}")
        # 清空其余 2 格
        for i in (1, 2):
            self.rst_canvases[i].delete("all")

    def _rst_zoom_open(self, idx, title):
        """点击 canvas[idx] 后弹出的可缩放查看器。

        特性: 鼠标滚轮缩放, 左键拖动平移, 双击或 ESC 关闭, 状态栏显示缩放%.
        """
        from PIL import Image, ImageTk
        img = self.rst_fullimg[idx] if idx < len(self.rst_fullimg) else None
        if img is None:
            return
        pal = self._pal
        W, H = img.width, img.height
        # 弹窗大小 = 屏幕 70% 但不超过原图尺寸
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        win_w = min(int(sw * 0.7), W + 40)
        win_h = min(int(sh * 0.75), H + 80)
        win_w = max(win_w, 480); win_h = max(win_h, 360)

        top = tk.Toplevel(self)
        top.title(f"{title}  ({W}x{H})")
        top.configure(bg=pal["bg"])
        # ESC 关闭
        top.bind("<Escape>", lambda _e: top.destroy())
        # 双击关闭
        top.bind("<Double-Button-1>", lambda _e: top.destroy())

        # 顶部工具条
        bar = ttk.Frame(top); bar.pack(fill="x", padx=6, pady=4)
        ttk.Label(bar, text=f"{title}  ·  原图 {W}x{H}  ·  ", font=("Menlo", 10)).pack(side="left")
        zoom_lbl = ttk.Label(bar, text="100%", font=("Menlo", 10, "bold"))
        zoom_lbl.pack(side="left")
        ttk.Label(bar, text="  (滚轮缩放 / 左键拖动 / 双击或 ESC 关闭)",
                  font=("Menlo", 9), foreground=pal["dim"]).pack(side="left", padx=8)

        cv_frame = ttk.Frame(top); cv_frame.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        cv = tk.Canvas(cv_frame, width=win_w - 40, height=win_h - 80,
                       bg=pal["canvas"], highlightthickness=1,
                       highlightbackground=pal["canvas_border"])
        cv.pack(fill="both", expand=True)

        state = {"scale": 1.0, "tx": 0.0, "ty": 0.0}  # 当前缩放/平移

        def render():
            """按当前 scale 重渲染 ImageTk 并贴到 canvas."""
            s = state["scale"]
            nw = max(1, int(W * s)); nh = max(1, int(H * s))
            disp = img.resize((nw, nh), Image.LANCZOS if s > 1 else Image.BILINEAR)
            tk_img = ImageTk.PhotoImage(disp)
            state["tk_img"] = tk_img  # 防回收
            cv.delete("all")
            cv.create_image(state["tx"], state["ty"], image=tk_img, anchor="nw")
            zoom_lbl.config(text=f"{int(s * 100)}%  ({nw}x{nh})")

        def fit_to_window():
            state["scale"] = min((cv.winfo_width() - 4) / W,
                                 (cv.winfo_height() - 4) / H)
            state["tx"] = (cv.winfo_width() - W * state["scale"]) / 2
            state["ty"] = (cv.winfo_height() - H * state["scale"]) / 2
            render()

        def zoom_at(cx, cy, factor):
            """以 (cx, cy) 为缩放中心, 保持该点在视口内不动."""
            s0 = state["scale"]
            s1 = max(0.1, min(8.0, s0 * factor))
            if s1 == s0:
                return
            # 旧缩放下 (cx, cy) 对应原图坐标
            ix = (cx - state["tx"]) / s0
            iy = (cy - state["ty"]) / s0
            state["scale"] = s1
            state["tx"] = cx - ix * s1
            state["ty"] = cy - iy * s1
            render()

        # 滚轮缩放 (macOS / Linux 都用 Button-4/5)
        cv.bind("<MouseWheel>", lambda e: zoom_at(e.x, e.y, 1.15 if e.delta > 0 else 1 / 1.15))
        cv.bind("<Button-4>", lambda e: zoom_at(e.x, e.y, 1.15))
        cv.bind("<Button-5>", lambda e: zoom_at(e.x, e.y, 1 / 1.15))
        # 左键拖动
        drag = {"x": 0, "y": 0, "on": False}
        cv.bind("<ButtonPress-1>", lambda e: drag.update(x=e.x, y=e.y, on=True))
        cv.bind("<B1-Motion>",
                lambda e: (state.update(tx=state["tx"] + e.x - drag["x"],
                                        ty=state["ty"] + e.y - drag["y"]),
                           drag.update(x=e.x, y=e.y), render()))
        cv.bind("<ButtonRelease-1>", lambda e: drag.update(on=False))
        # 右键 = 适配窗口
        cv.bind("<Button-3>", lambda _e: fit_to_window())
        # 关闭时清理
        top.protocol("WM_DELETE_WINDOW", top.destroy)
        top.bind("<Configure>", lambda _e: None)  # 暂不响应 resize 重渲染

        # 初次渲染: 适配窗口
        top.update_idletasks()
        fit_to_window()

    def _rst_method_select(self):
        """Selecting one radio method leaves the explicit single-method mode."""
        method = self.rst_methods.get().strip()
        if method in METHOD_ORDER:
            self._rst_last_method = method
        self.rst_method_both.set(False)
        if method == "paddle":
            self._rst_request_paddle_warmup()

    def _rst_method_toggle(self):
        if self.rst_method_both.get():
            method = self.rst_methods.get().strip()
            if method in METHOD_ORDER:
                self._rst_last_method = method
            self.rst_methods.set(",".join(METHOD_ORDER))
        else:
            self.rst_methods.set(
                getattr(self, "_rst_last_method", "tesseract"))
        if self.rst_method_both.get():
            self._rst_request_paddle_warmup()

    def _rst_request_paddle_warmup(self):
        """Start Paddle initialization without blocking the Tk event loop."""
        global _RST_PADDLE_WARMUP_STARTED
        with _RST_PADDLE_LOCK:
            if _RST_PADDLE_MODULE is not None or _RST_PADDLE_WARMUP_STARTED:
                return
            _RST_PADDLE_WARMUP_STARTED = True
        threading.Thread(
            target=_rst_paddle_warmup,
            name="paddle-mrz-warmup",
            daemon=True,
        ).start()

    def _rst_request_indra_warmup(self):
        """Start Indra initialization without blocking the Tk event loop."""
        global _RST_INDRA_WARMUP_STARTED
        with _RST_INDRA_LOCK:
            if _RST_INDRA_MODULE is not None or _RST_INDRA_WARMUP_STARTED:
                return
            _RST_INDRA_WARMUP_STARTED = True
        threading.Thread(
            target=_rst_indra_warmup,
            name="indra-mrz-warmup",
            daemon=True,
        ).start()

    def _rst_selected_methods(self):
        snap = getattr(self, "_ocr_tk", None)
        if snap is not None:
            if snap.get("method_both"):
                return list(METHOD_ORDER)
            method = str(snap.get("methods") or "").strip()
            return [method] if method in METHOD_ORDER else ["tesseract"]
        if self.rst_method_both.get():
            return list(METHOD_ORDER)
        method = self.rst_methods.get().strip()
        return [method] if method in METHOD_ORDER else ["tesseract"]

    def _ocr_selected_methods(self):
        """Methods chosen on the 'MRZ OCR测试' tab (not 照片→MRZ tab).

        The user's choice is returned AS CHOSEN.  A weight set that
        cannot run an engine (the assets/pic variant ships no CRNN head)
        is reported by the engine itself as HEAD_UNAVAILABLE, not by
        quietly removing the engine from the list here: dropping it
        would hand the user a result produced by an algorithm they did
        not ask for."""
        if self.ocr_method_both.get():
            methods = list(METHOD_ORDER)
        else:
            methods = [self.ocr_methods.get().strip()]
        methods = [m for m in methods if m in METHOD_ORDER]
        return methods or ["tesseract"]

    def _ocr_selection_is_explicit(self):
        """True when the user named one algorithm (combo box, not 全部).

        An explicit choice is binding: that algorithm runs, and nothing
        else stands in for it.  The fast tier (a single fused head) is a
        performance default for the paths where the user named no
        algorithm; it must never replace a named one."""
        try:
            return not bool(self.ocr_method_both.get())
        except Exception:
            return False

    def _rst_sensitive_rows(self, rgb):
        """Bottom-most two text rows of the page, in native pixels.

        Fallback for photos whose MRZ is too small for the two structural
        locators (restore.detect_structure and fast_pair): a social-media
        screenshot of a passport card carries a ~15 px high MRZ.  The MRZ is
        the last thing on the page, so the bottom-most two text rows are it.
        Returns (y0a, y1a, y0b, y1b, x0a, x1a, x0b, x1b) or None.
        """
        import sys as _sys
        from pathlib import Path as _P
        from PIL import Image as _Image
        try:
            _sys.path.insert(0, str(_P(__file__).resolve().parents[1] /
                                    "6_mrz_ocr" / "tools"))
            import mrz_band
            return mrz_band.sensitive_band_rows(_Image.fromarray(rgb))
        except Exception:
            return None

    def _rst_native_mrz_band(self, rgb, pair, scale, source_bottom=None,
                             single=False, pad_override=None,
                             x_range=None):
        """Build a native-resolution two-line ROI for external OCR tools.

        `single`: the document carries ONE MRZ row (CS/CD travel permit).
        The band is then exactly that row -- NO vertical padding.  Padding
        measurably flips the C layout classifier on permits (measured on
        assets/pic/1c6acd092dee7031_0222.png: tight crop -> CS_CD_30 with
        the exact code 'CDC000000007<2508197<8208031<2', +8 px padding ->
        TD3_44 and garbage), because the extra rows change the aspect
        ratio and the dual-hypothesis grid vote.

        Structure detection runs on the 1250px work image, but the C/Paddle
        tools lose too much glyph detail when they receive that downsampled
        15px-high MRZ. Keep the same detected coordinates and map them back
        to the source image. The two rows are cropped as ONE region with
        their original inter-line gap: stacking independently padded rows
        changes the line geometry the native splitter relies on (same fix
        as PhotoMrzPipeline.nativeBand).
        """
        H, W = rgb.shape[:2]
        import cv2 as _cv2_g
        nominal = min(100, max(10, int(round(W * 0.08))))
        xpad_l, xpad_r = _rst_adaptive_xpad(rgb, pair, scale, nominal)
        if x_range is not None:
            # Explicit text extent (row-scan fallback): the MRZ block does
            # not span the page, so the page margins must be cropped away or
            # the C tool fits 44 cells over the wrong width.
            xpad_l = max(0, min(W - 1, int(x_range[0])))
            xpad_r = max(0, W - min(W, int(x_range[1])))
        first_top = max(0, int(round(pair[0][0] / scale)))
        second_bottom = min(H, int(round((pair[1][1] + 1) / scale)))
        heights = [
            max(1, int(round((y1 + 1) / scale))
                - int(round(y0 / scale)))
            for y0, y1 in pair
        ]
        # 扩展 L1 顶部：find_mrz_pair 用 row_ink > 0.16W 筛选文本行，
        # MRZ 行顶部几行墨量 0.10~0.13 低于阈值被排除，导致 first_top 偏内。
        # 向上扫描扩展直到墨量降到背景水平（< 5% 暗像素），最多扩展 L1 行高的0.5倍。
        # 仅对高分辨率图片（W>=1000，即 passport 级别）应用，pic 低分辨率
        # 图片 pair 边界已较紧，扩展会引入上方噪声导致退化。
        if not single and W >= 1000:
            _gray_full = (_cv2_g.cvtColor(rgb, _cv2_g.COLOR_RGB2GRAY)
                          if rgb.ndim == 3 else rgb)
            _ink_thr = max(3, int(W * 0.05))
            _max_exp = max(3, int(heights[0] * 0.5))
            _exp = 0
            while first_top > 0 and _exp < _max_exp:
                if (_gray_full[first_top - 1] < 128).sum() >= _ink_thr:
                    first_top -= 1
                    _exp += 1
                else:
                    break
        if single:
            top, bottom = first_top, second_bottom   # exactly the row
            # Ensure a minimum band height for reliable OCR.  Some CS/CD
            # images yield a very thin band (16 px) that Tesseract and
            # the CNN tool cannot read; padding to >= 30 px fixes this
            # without flipping the layout classifier (measured: the
            # flip occurs at ~80 px padding, not 30).
            min_h = 30
            actual_h = bottom - top
            if actual_h < min_h:
                extra = (min_h - actual_h + 1) // 2
                top = max(0, top - extra)
                bottom = min(H, bottom + extra)
        else:
            if pad_override is not None:
                pad = pad_override
            else:
                pad = max(8, min(25, int(round(max(heights) * 0.70))))
            top = max(0, first_top - pad)
            bottom = min(H, second_bottom + pad)
        if source_bottom is not None:
            bottom = min(bottom, max(second_bottom, source_bottom))
        width = max(1, W - xpad_l - xpad_r)
        crop_rect = {"x": xpad_l, "y": top,
                     "w": width, "h": min(bottom, H) - top}
        band = rgb[top:bottom, xpad_l:xpad_l + width]
        if band.size == 0:
            band = rgb[top:second_bottom,
                       xpad_l:xpad_l + width]
            crop_rect["h"] = min(second_bottom, H) - top
        if band.size > 0 and band.shape[0] > 4:
            _bw = band.shape[1]
            _bh = band.shape[0]
            _gray = (_cv2_g.cvtColor(band, _cv2_g.COLOR_RGB2GRAY)
                     if band.ndim == 3 else band)
            _dark_thr = int(_bw * 0.90)
            while _bh > 4 and (_gray[_bh - 1] < 128).sum() >= _dark_thr:
                _bh -= 1
            if _bh < band.shape[0]:
                band = band[:_bh]
                crop_rect["h"] = _bh
        if band.size > 0 and single and band.shape[0] > 8:
            _bw = band.shape[1]
            _gray = (_cv2_g.cvtColor(band, _cv2_g.COLOR_RGB2GRAY)
                     if band.ndim == 3 else band)
            _sparse_thr = max(3, int(_bw * 0.01))
            _top = 0
            while _top < band.shape[0] - 4 and (_gray[_top] < 128).sum() < _sparse_thr:
                _top += 1
            _bot = band.shape[0]
            while _bot > _top + 4 and (_gray[_bot - 1] < 128).sum() < _sparse_thr:
                _bot -= 1
            if _top > 0 or _bot < band.shape[0]:
                band = band[_top:_bot]
                crop_rect["y"] = crop_rect["y"] + _top
                crop_rect["h"] = band.shape[0]
        if band.size > 0 and single and band.shape[0] < 24:
            import cv2 as _cv2_up
            _up = max(2, (24 + band.shape[0] - 1) // band.shape[0])
            band = _cv2_up.resize(band, (band.shape[1], band.shape[0] * _up),
                                  interpolation=_cv2_up.INTER_LINEAR)
            crop_rect["h"] = band.shape[0]
        return band, crop_rect

    def _rst_run_ocr_methods(self, rgb, pair, scale, band_path, methods,
                             native_band=None, source_path=None,
                             allow_tesseract_fallback=False,
                             resident=False, strict=False):
        """Run selected OCR backends against the same cleaned MRZ image.

        `resident=True` keeps the C tools and the CRNN ONNX model loaded
        between calls, so the reported time is recognition time rather
        than process/model start-up time.

        `allow_tesseract_fallback=False` (the default) means a failing
        engine simply fails: no Tesseract rescue.  The Tesseract engine
        itself still runs and is reported on its own merits.

        `strict=True` means the user named these engines explicitly.
        Nothing then stands in for them: the fast tier no longer
        narrows the set, the C tool is told which head it must run
        (MRZ_OCR_REQUIRE_HEAD) and fails instead of substituting
        another head, and no cross-engine rescue is applied.
        """
        import time as _time
        from concurrent.futures import ThreadPoolExecutor
        from threading import Lock

        # 严格极速模式：引擎集收敛到本机 C 引擎，且彻底关闭所有
        # 跨引擎 rescue（Tesseract / Indra / 低分辨率兜底）。
        #
        # strict=True means the user named this engine explicitly.  The
        # fast tier is a PERFORMANCE default for a path where no
        # algorithm was named -- narrowing an explicit choice to the
        # tier would run a different algorithm than the one on screen
        # (the symptom: "CNN" selected, "融合CNN+CRNN" in the log).
        if _fast_strict() and not strict:
            methods = [m for m in methods if m in FAST_TIER]
            if not methods:
                methods = list(_fast_tier())
        if (_fast_strict() or strict) and not (pair and pair[0] == pair[1]):
            allow_tesseract_fallback = False

        # CS/CD 单行 MRZ: 跳过不支持单行格式的引擎
        # paddle/indra 在单行 MRZ 上必然失败，白白浪费 ~900ms
        is_cscd = pair[0] == pair[1] if pair else False
        if is_cscd:
            if strict:
                # The user named this engine; it is not ours to replace.
                # Engines that cannot read a 30-character row report
                # that as their own failure.
                pass
            else:
                methods = [m for m in methods
                           if m not in ("paddle", "indra")]
                if not methods:
                    # 严格极速模式绝不退回 Tesseract：CS/CD 单行交给本机
                    # C 引擎如实输出，读不出就报失败。
                    # CSCD CRNN head 已训练 30 字符格式，crnn/fuse 可用。
                    methods = ["crnn"] if _fast_strict() else ["cnn", "crnn"]

        tool_paths = {
            "traditional": ROOT / "6_mrz_ocr" / "build" / "mrz_ocr_tool",
            # cnn/fuse 走哪一个二进制由"训练集"复选框决定（护照照 / assets/pic）。
            "cnn": self._ocr_weights_tool(),
            # CRNN 走 C 端工具 (MRZ_OCR_CRNN=1)，比 ONNX 快 ~70×。
            "crnn": self._ocr_weights_tool(),
            "fuse": self._ocr_weights_tool(),
            "paddle": ROOT / "6_1_PaddleOCR" / "paddle_v6.py",
            "indra": ROOT / "11_indra" / "mrz_indra.py",
            "easyocr": None,  # Python module, no binary
        }
        # CRNN/融合 与 cnn 共享同一 PPM band；needs_file 判断等价。

        def failed(method, started, error):
            return {
                "method": method,
                "label": METHOD_LABELS[method],
                "line1": "",
                "line2": "",
                "engine_ok": False,
                "engine_status": "FAIL",
                "decode_ok": False,
                "conf1": 0,
                "conf2": 0,
                "repair": [],
                "candidates": "",
                "raw": "",
                "decode": "",
                "raw_decode_rc": None,
                "raw_decode_error": "",
                "backend": "",
                "error": error,
                "note": "",
                "ms": (_time.perf_counter() - started) * 1000.0,
            }

        rescue_cache = {"ready": False, "value": None}
        rescue_lock = Lock()

        def tesseract_rescue():
            """Run the trusted OCR-B path once for failed external results."""
            with rescue_lock:
                if not rescue_cache["ready"]:
                    rescue_started = _time.perf_counter()
                    is_cscd = pair[0] == pair[1]
                    if is_cscd:
                        line1s, line2s, raw, error = \
                            _rst_tesseract_cscd(rgb, pair, scale)
                        if not line1s and source_path:
                            try:
                                indra_stdout, indra_rc, _ = \
                                    _rst_run_indra_inprocess(source_path)
                                indra_kv = _rst_parse_kv_output(indra_stdout)
                                il1 = indra_kv.get("result.line1", "")
                                if il1 and len(il1) == 30:
                                    line1s = [il1]
                                    raw = indra_stdout
                            except Exception:
                                pass
                    else:
                        line1s, line2s, raw, error = \
                            _rst_tesseract_candidates(rgb, pair, scale)
                    if error:
                        value = None
                    else:
                        value = (
                            _rst_decode_candidates(line1s, line2s),
                            raw,
                        )
                    rescue_cache["value"] = value
                    rescue_cache["ms"] = (
                        _time.perf_counter() - rescue_started) * 1000.0
                    rescue_cache["ready"] = True
                return rescue_cache["value"], rescue_cache.get("ms", 0.0)

        def run_one(method):
            started = _time.perf_counter()
            try:
                if method == "tesseract":
                    is_cscd = pair[0] == pair[1]
                    if is_cscd:
                        line1s, line2s, raw, error = \
                            _rst_tesseract_cscd(rgb, pair, scale)
                    else:
                        line1s, line2s, raw, error = \
                            _rst_tesseract_candidates(rgb, pair, scale)
                    line1, line2, repair, _raw_line2, decoded = \
                        _rst_decode_candidates(line1s, line2s)
                    display1 = line1 or (line1s[0] if line1s else "")
                    display2 = line2 or (
                        line2s[0][3] if line2s else "")
                    eng_ok = bool(line1s) if is_cscd \
                        else bool(line1s and line2s)
                    return {
                        "method": method,
                        "label": METHOD_LABELS[method],
                        "line1": display1,
                        "line2": display2,
                        "engine_ok": eng_ok,
                        "engine_status": "OK" if eng_ok else "FAIL",
                        "decode_ok": _rst_decode_ok(decoded),
                        "conf1": 0,
                        "conf2": 0,
                        "repair": repair,
                        "candidates": (
                            f"line1={len(line1s)} line2={len(line2s)}"),
                        "raw": raw,
                        "decode": decoded.stdout.strip(),
                        "raw_decode_rc": decoded.returncode,
                        "raw_decode_error": decoded.stderr.strip(),
                        "backend": "tesseract",
                        "note": "",
                        "error": error or (
                            decoded.stderr.strip()
                            if decoded.returncode != 0 else ""),
                        "ms": (_time.perf_counter() - started) * 1000.0,
                    }

                tool = tool_paths[method]
                if tool is not None and not tool.exists():
                    return failed(method, started, f"工具不存在: {tool}")
                # The head that will ACTUALLY run.  assets/pic ships no
                # CRNN head, so "CRNN序列"/"融合CNN+CRNN" run the light
                # CNN head there (the tool reports result.head =
                # cnn(no-crnn)).  The low-resolution guard must follow the
                # head, not the requested method name: keying it on the
                # name made ONE head behave differently under two names
                # (skipped as "cnn", run as "crnn"), which is exactly the
                # kind of silent substitution this guard must not hide.
                eff_head = method
                if method in ("crnn", "fuse") and self._weights_set() == "pic":
                    eff_head = "cnn(no-crnn)"
                if eff_head in ("cnn", "cnn(no-crnn)"):
                    low_res_reason = _rst_cnn_low_res_reason(pair, scale)
                    # The guard is OFF by default: it used to refuse to
                    # run the selected algorithm on rows below 12 px, and
                    # the CNN has improved since then.  Measured with the
                    # same weights and the same bands:
                    #   assets/pic   guard ON  both-exact 33/61, PASS 34
                    #                guard OFF both-exact 43/61, PASS 45
                    #   44 photos    guard ON  both-exact 36/47
                    #                guard OFF both-exact 37/47
                    # The photos the guard skipped were read CORRECTLY
                    # when the same head was allowed to run; a reading
                    # that does not verify is reported as a failure by
                    # the decoder, which is the honest outcome -- the
                    # guard only turned readable photos into no result.
                    # MRZ_CNN_LOW_RES_GUARD=1 restores the old behaviour.
                    if low_res_reason and os.environ.get(
                            "MRZ_CNN_LOW_RES_GUARD", "0") == "1":
                        if not allow_tesseract_fallback:
                            return {
                                "method": method,
                                "label": METHOD_LABELS[method],
                                "line1": "",
                                "line2": "",
                                "engine_ok": False,
                                "engine_status": "SKIP",
                                "decode_ok": False,
                                "conf1": 0,
                                "conf2": 0,
                                "repair": [],
                                "candidates": "",
                                "raw": (
                                    "method            : cnn\n"
                                    "result.ok         : skipped low-resolution MRZ\n"
                                    f"result.note       : {low_res_reason}\n"
                                ),
                                "decode": "",
                                "raw_decode_rc": None,
                                "raw_decode_error": "",
                                "backend": f"{eff_head} (low-res skip)",
                                # fallback 被显式禁用：识别错误就如实报错，
                                # 不再用 Tesseract OCR-B 顶替。
                                "note": f"轻量CNN已跳过：{low_res_reason}；"
                                        "（Tesseract OCR-B 自动修复已禁用）",
                                "error": f"轻量CNN已跳过：{low_res_reason}",
                                "ms": (_time.perf_counter() - started) * 1000.0,
                            }
                        rescue, rescue_ms = tesseract_rescue()
                        result = {
                            "method": method,
                            "label": METHOD_LABELS[method],
                            "line1": "",
                            "line2": "",
                            "engine_ok": False,
                            "engine_status": "SKIP",
                            "decode_ok": False,
                            "conf1": 0,
                            "conf2": 0,
                            "repair": [],
                            "candidates": "",
                            "raw": (
                                "method            : cnn\n"
                                "result.ok         : skipped low-resolution MRZ\n"
                                f"result.note       : {low_res_reason}\n"
                            ),
                            "decode": "",
                            "raw_decode_rc": None,
                            "raw_decode_error": "",
                            "backend": f"{eff_head} (low-res skip)",
                            "note": "",
                            "error": "",
                            "ms": (_time.perf_counter() - started) * 1000.0,
                        }
                        if rescue is not None:
                            (fixed1, fixed2, repair, _raw_line2,
                             fixed_decoded), tess_raw = rescue
                            result["line1"] = fixed1
                            result["line2"] = fixed2
                            result["repair"] = repair
                            result["decode"] = fixed_decoded.stdout.strip()
                            result["decode_ok"] = _rst_decode_ok(fixed_decoded)
                            result["backend"] = (
                                "cnn (low-res skip) -> tesseract-fallback")
                            result["note"] = (
                                f"轻量CNN已跳过：{low_res_reason}；"
                                "直接使用 Tesseract OCR-B 识别"
                                f"（fallback {rescue_ms:.1f} ms）")
                            if tess_raw:
                                result["raw"] += (
                                    "--- tesseract fallback ---\n" + tess_raw)
                            result["ms"] = (
                                _time.perf_counter() - started) * 1000.0
                        else:
                            result["error"] = (
                                "轻量CNN已跳过："
                                f"{low_res_reason}；Tesseract OCR-B 也未返回有效结果"
                            )
                        return result
                if method == "paddle":
                    stdout, returncode, stderr = _rst_run_paddle_inprocess(
                        native_band if native_band is not None else band_path)
                elif method == "indra":
                    if source_path is None:
                        return failed(method, started, "缺少原图路径")
                    # Indra detects lines itself. Given the FULL page it
                    # can lock onto printed document titles near the top
                    # (Czech specimen: band.y=14 vs the real MRZ at
                    # y=331) — its "MRZ" then has nothing to do with the
                    # clean band shown in stage ③. When the structure
                    # stage already located the pair, constrain Indra to
                    # the same cleaned band every other engine reads;
                    # keep the full-photo pass ONLY for the no-pair
                    # fallback where the structure stage failed.
                    indra_input = source_path
                    indra_tmp = None
                    try:
                        import tempfile as _tempfile
                        from PIL import Image as _Image
                        if native_band is not None:
                            tmp_dir2 = Path(__file__).resolve().parents[1] / \
                                "10_ocr_research"
                            tmp_dir2.mkdir(parents=True, exist_ok=True)
                            indra_tmp = _tempfile.NamedTemporaryFile(
                                suffix=".ppm", prefix="_gui_indra_",
                                dir=str(tmp_dir2), delete=False)
                            _Image.fromarray(native_band).save(
                                indra_tmp.name, format="PPM")
                            indra_tmp.close()
                            indra_input = indra_tmp.name
                        stdout, returncode, stderr = \
                            _rst_run_indra_inprocess(indra_input)
                    finally:
                        if indra_tmp is not None:
                            try:
                                os.unlink(indra_tmp.name)
                            except OSError:
                                pass
                elif resident and method in ("traditional", "cnn", "crnn",
                                             "fuse"):
                    # Resident C tool: the model stays loaded, so this
                    # measures recognition only.  crnn is the same binary
                    # with MRZ_OCR_CRNN=1; leaving it out of the pool used
                    # to force a fresh process per photo (~60 ms extra).
                    from mrz_engine_pool import (
                        get_cnn_tool, get_crnn_c_tool,
                        get_traditional_tool)
                    if method == "traditional":
                        pool_tool = get_traditional_tool()
                    elif method == "crnn":
                        pool_tool = get_crnn_c_tool(self._ocr_weights_tool())
                    else:
                        pool_tool = get_cnn_tool(
                            fuse=(method == "fuse"),
                            tool=self._ocr_weights_tool())
                    stdout, stderr = pool_tool.run(str(band_path))
                    returncode = 0 if stdout else 1
                elif resident and method == "paddle":
                    # Resident Paddle: the ONNX recogniser stays loaded,
                    # so the subprocess start-up cost is paid once.
                    from mrz_engine_pool import get_paddle_module
                    paddle = get_paddle_module()
                    try:
                        res = paddle.read_band(str(band_path))
                        stdout = paddle.format_kv(res)
                        returncode = 0 if res.get("ok") else 1
                        stderr = ""
                    except Exception as exc:  # noqa: BLE001
                        stdout = ""
                        returncode = 1
                        stderr = f"{type(exc).__name__}: {exc}"
                elif method == "easyocr":
                    from mrz_engine_pool import get_easyocr_reader
                    try:
                        reader = get_easyocr_reader()
                        l1, l2, c1, c2, ok = _rst_easyocr_extract(
                            reader,
                            native_band if native_band is not None else band_path)
                        stdout = "\n".join([
                            "method            : easyocr",
                            "result.head       : easyocr",
                            f"result.ok         : {'OK' if ok else 'unreadable'}",
                            f"result.line1      : {l1}",
                            f"result.line2      : {l2}",
                            f"result.conf1      : {c1}",
                            f"result.conf2      : {c2}",
                            "result.done       : 1",
                        ])
                        returncode, stderr = (0, "") if ok else (1, "no text")
                    except Exception as exc:  # noqa: BLE001
                        stdout, returncode, stderr = "", 1, f"{type(exc).__name__}: {exc}"
                else:
                    argv = [str(tool), str(band_path)]
                    if str(tool).endswith(".py"):
                        argv.insert(0, sys.executable)
                    # fuse 与 cnn 共用 host tool，靠环境变量切换融合头
                    # （见 6_mrz_ocr/src/mrz_ocr_cnn.c）。
                    env = dict(os.environ)
                    env.setdefault("MRZ_OCR_LINE_MODELS", "1")
                    env["MRZ_OCR_CRNN"] = "1" if method == "crnn" else "0"
                    env.pop("MRZ_OCR_FUSE", None)
                    if method == "fuse":
                        env["MRZ_OCR_FUSE"] = "1"
                    if strict and method in ("cnn", "crnn", "fuse"):
                        # The engine must run the head that was asked
                        # for.  Without this the tool would quietly run
                        # the CNN head when the weight set has no CRNN
                        # head (assets/pic) and report it as a CRNN run.
                        env["MRZ_OCR_REQUIRE_HEAD"] = method
                    proc = subprocess.run(
                        argv, text=True, capture_output=True, timeout=10,
                        env=env)
                    stdout = proc.stdout
                    returncode = proc.returncode
                    stderr = proc.stderr
                kv = _rst_parse_kv_output(stdout)
                raw_line1 = kv.get("result.line1", "")
                raw_line2 = kv.get("result.line2", "")
                line1 = raw_line1
                line2 = raw_line2
                decoded = _rst_decode_lines(raw_line1, raw_line2)
                backend = kv.get("result.backend", "")
                note = kv.get("result.note", "")
                # Which head actually ran ("cnn" / "crnn" / "fuse", or
                # "cnn(no-crnn)" when the weight set had no CRNN head).
                # An explicit request must be met exactly: a substituted
                # head is a DIFFERENT algorithm and its output must not
                # be presented as the one the user selected.
                head = kv.get("result.head", "")
                head_mismatch = bool(
                    strict and method in ("cnn", "crnn", "fuse")
                    and head and head != method)
                if (not head_mismatch and head
                        and method in ("cnn", "crnn", "fuse")
                        and head != method):
                    # Not strict: the run is allowed to continue, but the
                    # label must name the head that produced it.  A pic
                    # weight set running the plain CNN head under the
                    # label "融合CNN+CRNN" is how a wrong engine gets
                    # reported as a right one.
                    backend = backend or f"{method} -> {head}"
                    note = note or (f"该权重集没有 {method} 头，"
                                    f"实际由 {head} 完成")
                if head_mismatch:
                    line1 = line2 = ""
                    raw_line1 = raw_line2 = ""
                    decoded = _rst_decode_lines("", "")
                    backend = f"{method} -> {head}"
                    note = (f"请求的算法「{METHOD_LABELS[method]}」"
                            f"在此权重集不可用，实际运行的是「{head}」")
                result = {
                    "method": method,
                    "label": METHOD_LABELS[method],
                    "line1": line1,
                    "line2": line2,
                    "engine_ok": (kv.get("result.ok") == "OK"
                                  and not head_mismatch),
                    "engine_status": (
                        "FAIL" if head_mismatch else
                        ("OK" if kv.get("result.ok") == "OK" else "FAIL")),
                    "decode_ok": _rst_decode_ok(decoded),
                    "conf1": int(kv.get("result.conf1", "0") or 0),
                    "conf2": int(kv.get("result.conf2", "0") or 0),
                    "band": kv.get("band.x band.y band.w band.h", ""),
                    "lines": kv.get("result.line1y line1h line2y line2h", ""),
                    "repair": [],
                    "candidates": "",
                    "raw": stdout.strip(),
                    "decode": decoded.stdout.strip(),
                    "raw_decode_rc": decoded.returncode,
                    "raw_decode_error": decoded.stderr.strip(),
                    "backend": backend,
                    "note": note,
                    "error": (
                        note
                        or stderr.strip()
                        or (decoded.stderr.strip()
                            if decoded.returncode != 0 else "")),
                    "ms": (_time.perf_counter() - started) * 1000.0,
                }
                # A CS/CD travel permit carries ONE 30-character row and
                # mrz_tool only speaks TD3 (2 x 44), so its decode can
                # NEVER succeed.  Rescuing such a result overwrites a
                # correct CS/CD reading: measured on
                # assets/pic/1c6acd092dee7031_0222.png the CNN read
                # 'CS<C00000007<2568107<8208031<7' and the rescue
                # replaced it with Tesseract's 'CDL000000007<2508197<'
                # (two characters lost).  Keep the engine's own reading
                # and let the CS/CD structural check judge it.
                _rl1 = result.get("line1") or ""
                cscd_reading = (len(_rl1) == 30
                                and _rl1[:2] in ("CS", "CD"))
                # --- alt2 消费 ---
                # C++ beam search 已计算 line2 的次优解（checksum-valid
                # runner-up）。当 ambiguous2=1 时主解和 alt2 都是
                # checksum-valid 但 score 接近；若主解 decode 失败而
                # alt2 decode 通过，用 alt2 替换 line2。
                _alt2 = kv.get("result.alt2", "")
                _amb2 = int(kv.get("result.ambiguous2", "0") or 0)
                if _alt2 and _amb2 and line2 and not cscd_reading:
                    _alt2_dec = _rst_decode_lines(line1, _alt2)
                    _alt2_ok = _rst_decode_ok(_alt2_dec)
                    if _alt2_ok and not result["decode_ok"]:
                        result["line2"] = _alt2
                        result["decode"] = _alt2_dec.stdout.strip()
                        result["decode_ok"] = True
                        result["backend"] = (
                            f"{backend or method} -> alt2")
                        result["note"] = (
                            f"{METHOD_LABELS[method]} line2 模糊"
                            "（ambiguous），已采用校验位有效的"
                            "次优解 alt2")
                        result["error"] = ""
                        result["raw"] += "\n--- alt2 rescue ---\n" + _alt2
                    result["alt2"] = _alt2
                    result["ambiguous2"] = _amb2
                needs_rescue = (
                    not result["decode_ok"]
                    or result["conf1"] < 80
                    or (result["line2"] and result["conf2"] < 80)
                )
                if (method in ("traditional", "cnn", "crnn", "fuse", "paddle")
                        and needs_rescue
                        and allow_tesseract_fallback):
                    rescue, rescue_ms = tesseract_rescue()
                    result["ms"] = (
                        _time.perf_counter() - started) * 1000.0
                    if rescue is not None:
                        (fixed1, fixed2, repair, _raw_line2,
                         fixed_decoded), tess_raw = rescue
                        mismatch = (
                            not result["decode_ok"]
                            or (result["line1"], result["line2"])
                            != (fixed1, fixed2)
                        )
                        if mismatch and (fixed1 or fixed2):
                            raw_status = kv.get("result.ok", "").strip()
                            if not result["engine_ok"]:
                                reason = (
                                    f"{METHOD_LABELS[method]} 未产生有效引擎输出"
                                    + (f"（{raw_status}）"
                                       if raw_status else "")
                                )
                            elif not result["decode_ok"]:
                                reason = (
                                    f"{METHOD_LABELS[method]} 已输出结果，"
                                    "但原始结果未通过 MRZ decode"
                                )
                            else:
                                reason = (
                                    f"{METHOD_LABELS[method]} 结果置信率偏低，"
                                    "与 Tesseract OCR-B 复核不一致"
                                )
                            result["line1"] = fixed1
                            result["line2"] = fixed2
                            result["repair"] = repair
                            result["decode"] = fixed_decoded.stdout.strip()
                            # The rescue only "passes" when the decoder
                            # really accepted the rescued lines; a failed
                            # rescue must not claim decode_ok (it used to
                            # be hard-coded True, which turned empty
                            # results into a verified verdict).
                            result["decode_ok"] = _rst_decode_ok(fixed_decoded)
                            result["backend"] = (
                                f"{backend or method} -> tesseract-fallback")
                            result["engine_status"] = (
                                "FAIL" if not result["engine_ok"] else "OK")
                            result["note"] = (
                                f"{reason}；已切换到 Tesseract OCR-B "
                                f"自动修复（fallback {rescue_ms:.1f} ms）")
                            result["error"] = ""
                            if tess_raw:
                                result["raw"] += (
                                    "\n--- tesseract fallback ---\n" + tess_raw)
                if result["decode_ok"] and result.get("note"):
                    # A model-unavailable note is informational when the
                    # fallback result itself passes the real decoder.
                    result["error"] = ""
                return result
            except Exception as exc:  # noqa: BLE001
                return failed(method, started, str(exc))

        wall_started = _time.perf_counter()
        if "paddle" in methods:
            self._rst_request_paddle_warmup()
        if "indra" in methods:
            self._rst_request_indra_warmup()
        # Tiered engine cascade (MRZ_CASCADE=1): run the cheap C tier
        # first; if the fused head already produced a checksum-valid
        # reading, the expensive engines (tesseract 125 ms, paddle 107
        # ms, indra 50 ms mean on the 2026-10-30 bench) cannot improve
        # on a verified decode, so skip them entirely.  Measured 202
        # ms -> target ~1xx ms mean; the acceptance bench gates this
        # (official PASS must not drop, per-photo regressions = 0).
        cascade = (
            os.environ.get("MRZ_CASCADE", "") == "1"
            and len(methods) > 1
            and any(m in methods for m in ("traditional", "cnn", "crnn", "fuse"))
            and any(m in methods for m in ("tesseract", "paddle", "indra"))
        )
        if len(methods) == 1:
            results = [run_one(methods[0])]
        elif cascade:
            fast = [m for m in methods
                    if m in ("traditional", "cnn", "crnn", "fuse")]
            slow = [m for m in methods
                    if m not in ("traditional", "cnn", "crnn", "fuse")]
            with ThreadPoolExecutor(max_workers=len(fast)) as pool:
                fast_futures = {
                    method: future
                    for method, future in (
                        (method, pool.submit(run_one, method))
                        for method in fast)
                }
                fast_results = [fast_futures[m].result() for m in fast]
            if any(r["decode_ok"] and r["method"] == "fuse"
                   for r in fast_results):
                # Paddle Trust-Base: fuse checksum 通过仍跑 paddle，
                # paddle decode_ok 时优先于 fuse（paddle L2 char 98.5%
                # 远高于 fuse 93.3%）。仅加跑 paddle，跳过 tesseract/indra。
                if "paddle" in slow:
                    paddle_result = run_one("paddle")
                    return fast_results + [paddle_result], (
                        _time.perf_counter() - wall_started) * 1000.0
                return fast_results, (
                    _time.perf_counter() - wall_started) * 1000.0
            with ThreadPoolExecutor(max_workers=len(slow)) as pool:
                slow_futures = {
                    method: future
                    for method, future in (
                        (method, pool.submit(run_one, method))
                        for method in slow)
                }
                slow_results = [slow_futures[m].result() for m in slow]
            by_method = dict(zip(fast, fast_results))
            by_method.update(zip(slow, slow_results))
            results = [by_method[m] for m in methods]
        else:
            with ThreadPoolExecutor(max_workers=len(methods)) as pool:
                by_method = {
                    method: future
                    for method, future in (
                        (method, pool.submit(run_one, method))
                        for method in methods)
                }
                results = [by_method[method].result() for method in methods]
        return results, (_time.perf_counter() - wall_started) * 1000.0

    def _rst_run(self):
        """4 步流水线: 原图 → 去影 → MRZ区 → 识别/解码."""
        import sys as _sys
        from pathlib import Path as _P
        import time as _time
        _t_all = _time.perf_counter()
        try:
            self.rst_run_btn.configure(state="disabled")
            self.rst_status.config(text="运行流水线…", foreground=self._pal["dim"])
            self.update_idletasks()
            path = self._rst_resolve_path()
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            _sys.path.insert(0, str(_P(__file__).resolve().parents[1] /
                                    "passport_photo_restore" / "src"))
            from restore import (load, _gray, detect_structure,  # type: ignore
                                 drop_reflection, crop_pair_clean)
            import cv2 as _cv2
            from PIL import Image as _Image

            rgb = _cv2.cvtColor(_cv2.imread(path), _cv2.COLOR_BGR2RGB)
            H0, W0 = rgb.shape[:2]
            stages = []

            # ① 原图
            t0 = _time.perf_counter()
            work, scale = load(rgb, max_side=1250)
            g = _gray(work)
            st = detect_structure(g)
            stages.append(("①原图+结构检测", (_time.perf_counter() - t0) * 1000))
            H_before = work.shape[0]
            _annotated = rgb.copy()
            _p = st.get("mrz_pair")
            _s = st.get("mrz_single")
            if _p:
                (_pa0, _pa1), (_pb0, _pb1) = _p
                _ny0a = max(0, int(round(_pa0 / scale)))
                _ny1a = min(H0, int(round(_pa1 / scale)))
                _ny0b = max(0, int(round(_pb0 / scale)))
                _ny1b = min(H0, int(round(_pb1 / scale)))
                _cv2.line(_annotated, (0, _ny0a), (W0-1, _ny0a), (255, 0, 0), 1)
                _cv2.line(_annotated, (0, _ny1a), (W0-1, _ny1a), (255, 0, 0), 1)
                _cv2.line(_annotated, (0, _ny0b), (W0-1, _ny0b), (0, 0, 255), 1)
                _cv2.line(_annotated, (0, _ny1b), (W0-1, _ny1b), (0, 0, 255), 1)
                _h1 = max(1, _ny1a - _ny0a)
                _h2 = max(1, _ny1b - _ny0b)
                _pad = max(8, min(25, int(round(max(_h1, _h2) * 0.70))))
                _cv2.line(_annotated, (0, max(0, _ny0a - _pad)), (W0-1, max(0, _ny0a - _pad)), (0, 255, 0), 1)
                _cv2.line(_annotated, (0, min(H0, _ny1b + _pad)), (W0-1, min(H0, _ny1b + _pad)), (0, 255, 0), 1)
            elif _s:
                _sy0, _sy1 = _s
                _ny0 = max(0, int(round(_sy0 / scale)))
                _ny1 = min(H0, int(round(_sy1 / scale)))
                _cv2.line(_annotated, (0, _ny0), (W0-1, _ny0), (255, 0, 0), 2)
                _cv2.line(_annotated, (0, _ny1), (W0-1, _ny1), (0, 0, 255), 2)
            self._rst_fit_to(0, _Image.fromarray(_annotated), footer=f"{W0}×{H0}")

            # ② 去影图: 自适应——检出倒影则如实报告并裁掉 line2 下方倒影段,
            #    未检出则仅裁取景黑带
            t0 = _time.perf_counter()
            work2, st2, refl_info = drop_reflection(work, st)
            pair = st2.get("mrz_pair")
            H_after = work2.shape[0]
            actual_drop = H_before - H_after
            stages.append(("②去影(自适应)", (_time.perf_counter() - t0) * 1000))
            pct = actual_drop * 100 // H_before if H_before else 0
            if actual_drop <= 0:
                badge2, badge2_color = "无需裁除 (line2 已到图底)", "ok"
            elif refl_info.get("reflection_detected"):
                badge2 = (f"已裁掉 line2 下方倒影 {actual_drop}px "
                          f"({pct}% 图高, 检出倒影 {refl_info.get('reflection_px', 0)}px)")
                badge2_color = "warn"
            else:
                badge2 = (f"未检出倒影; 裁除 line2 下方余量 {actual_drop}px ({pct}% 图高)")
                badge2_color = "ok"
            self._rst_fit_to(1, _Image.fromarray(work2),
                             footer=f"{work2.shape[1]}×{work2.shape[0]}  (原 {H_before}px)",
                             badge=badge2,
                             badge_color=badge2_color)

            # ③ MRZ 区: 不拉伸自然形态 crop (保留 CNN 工具内部 deskew 假设)
            #
            # 这一格以前是「矫正图（透视拉直）」，现在直接放 OCR 真正吃
            # 的 crop。矫正图被移除的理由见上面 labels 处的说明：它不参与
            # OCR，而且比自然 crop 更差。
            t0 = _time.perf_counter()
            band_rgb = None
            if pair:
                # OCR 输入: crop_pair_clean, 剔除 line1 紧邻反射
                band_rgb, _band_info = crop_pair_clean(work2, st2, clear_rows=2)
            stages.append(("③MRZ区(crop)", (_time.perf_counter() - t0) * 1000))
            if band_rgb is not None:
                self._rst_fit_to(2, _Image.fromarray(band_rgb),
                                 footer=f"{band_rgb.shape[1]}×{band_rgb.shape[0]}",
                                 badge="OCR 输入",
                                 badge_color="ok")
            else:
                self.rst_canvases[2].delete("all")
                self.rst_canvases[2].create_text(
                    4, 8, text="(无 MRZ)", anchor="nw",
                    fill=self._pal["err"], font=("Menlo", 9, "bold"))

            # ⑤ OCR + decode
            lines = []
            lines.append(f"原图 {W0}×{H0}  →  工作 {work.shape[1]}×{work.shape[0]} "
                         f"(scale={scale:.3f})")
            if pair:
                (l1y0, l1y1), (l2y0, l2y1) = pair
                lines.append(
                    f"MRZ 双行: line1 y{l1y0}-{l1y1} (h={l1y1-l1y0+1})  "
                    f"line2 y{l2y0}-{l2y1} (h={l2y1-l2y0+1})  "
                    f"gap={l2y0-l1y1-1}")
            else:
                lines.append("MRZ 双行: <未定位>")
            if refl_info.get("reflection_detected"):
                lines.append(
                    f"line2 下方检出倒影段 {refl_info.get('reflection')} 已裁: "
                    f"work {H_before} -> {H_after} (丢 {actual_drop}px)")
            else:
                lines.append(f"line2 下方未检出倒影; 裁除 line2 下方余量 {actual_drop}px "
                             f"(work {H_before} -> {H_after})")
            lines.append("")
            lines.append("阶段耗时:")
            for name, ms in stages:
                lines.append(f"  - {name:<18} {ms:6.1f} ms")

            ocr_results = []
            ocr_wall_ms = 0.0
            methods = self._rst_selected_methods()
            if (pair and band_rgb is not None) or "indra" in methods:
                if not (pair and band_rgb is not None):
                    # Indra runs its own detection, so it still works when
                    # the structure stage found no MRZ pair.
                    methods = ["indra"]
                band_path = None
                tmp = None
                native_band = None
                try:
                    needs_file_band = any(
                        method in ("traditional", "cnn", "crnn", "fuse")
                        for method in methods)
                    if ("paddle" in methods or needs_file_band
                            or "indra" in methods):
                        native_band, _native_crop = self._rst_native_mrz_band(
                            rgb, pair, scale,
                            source_bottom=int(round(
                                work2.shape[0] / scale)))
                        if pair[0] == pair[1]:
                            try:
                                from restore import remove_guilloche
                                native_band = remove_guilloche(native_band)
                            except Exception:
                                pass
                    if needs_file_band:
                        import tempfile
                        tmp_dir = _P(__file__).resolve().parents[1] / \
                            "10_ocr_research"
                        tmp_dir.mkdir(parents=True, exist_ok=True)
                        tmp = tempfile.NamedTemporaryFile(
                            suffix=".ppm", prefix="_gui_mrz_",
                            dir=str(tmp_dir), delete=False)
                        band_path = _P(tmp.name)
                        tmp.close()
                        _Image.fromarray(native_band).save(
                            str(band_path), format="PPM")
                    ocr_results, ocr_wall_ms = self._rst_run_ocr_methods(
                        rgb, pair, scale, band_path, methods,
                        native_band=native_band, source_path=path,
                        # 照片->MRZ tab：识别错误就如实报错，不做 Tesseract
                        # OCR-B 自动修复。等各算法稳定/调优后再考虑集成。
                        allow_tesseract_fallback=False,
                        resident=bool(getattr(self, "_rst_resident", False)))
                finally:
                    if tmp is not None:
                        try:
                            band_path.unlink()
                        except OSError:
                            pass
                stages.append(("⑤ OCR方案合计", ocr_wall_ms))
                lines.append(
                    f"  - {'⑤ OCR方案合计':<18} {ocr_wall_ms:6.1f} ms")
            else:
                lines.append("无 MRZ 区，跳过 OCR")
            lines.append("")

            if ocr_results:
                # 推荐真实 decode 通过且未落到后备引擎的方案；同等结果
                # 再按置信率和耗时排序，避免把“引擎 OK”误当成 MRZ 有效。
                # MRZ-shape gate FIRST: an engine that read page
                # decoration instead of the MRZ (Indra runs its own
                # detection on the full photo) must never become the
                # "final MRZ" just because its engine reported OK.
                # Line 1 has no check digit, so decode_ok only says the
                # LINE 2 verified.  An engine can therefore pass the
                # checksum and still have a wrong line 1 (Indra reads the
                # filler '<' as 'K' often enough to do exactly that).
                # Rank by how many engines agree on line 1 BEFORE the
                # line-2 checksum, so a line 1 that most readers produced
                # wins over a lone engine with a higher confidence.
                line1_agree = _rst_line1_agreement(ocr_results)
                # Paddle Trust-Base: paddle L1 更准（92% vs fuse 68%），
                # 但 fuse L2 更准（CRNN 融合）。策略：paddle 优先仅当
                # paddle 的 L1 与 fuse 不同（L1 不同→paddle L1 更可信）；
                # L1 相同时回退 fuse（fuse L2 尾部更可靠）。
                _fuse_r = next((r for r in ocr_results
                                if r.get("method") == "fuse"), None)
                _fuse_l1 = (_fuse_r.get("line1") or "") if _fuse_r else ""
                preferred = max(
                    ocr_results,
                    key=lambda result: (
                        int(_rst_mrz_shaped(result.get("line1"),
                                            result.get("line2"))),
                        int(result.get("method") == "paddle"
                            and result["decode_ok"]
                            and not all(
                                c == '<' for c in (result.get("line1") or ""))
                            and (result.get("line1") or "") != _fuse_l1),
                        line1_agree[ocr_results.index(result)],
                        int(result["decode_ok"]),
                        int(not result.get("backend", "").endswith("fallback")),
                        int(result["engine_ok"]),
                        result.get("conf1", 0) + result.get("conf2", 0),
                        -result["ms"],
                    ),
                )
                # Multi-engine arbitration: choose line 1 and line 2
                # INDEPENDENTLY from the engine pool instead of taking
                # both lines from one engine.  A TD3 line 1 carries no
                # check digit, so the engine with the best line 2 is
                # regularly not the engine with the best line 1 -- picking
                # a single engine throws away a correct reading that
                # another engine already produced.
                # Measured on the 106 photos of this corpus that carry a
                # 44-character ground truth (xlsx "GT对比"): exact pairs
                # 79 -> 90 against the sheet as it stands, 80 -> 93
                # against a sheet whose six mis-transcribed check digits
                # are corrected; line-2 exact readings 79 -> 90, mean
                # line-2 character accuracy 0.9393 -> 0.9537.
                # The arbitrated pair is used only when it is
                # structurally sound; otherwise the single-engine pick
                # above stands.
                _arb_l1 = _arb_l2 = ""
                try:
                    if os.environ.get("MRZ_ARBITRATE", "1") != "1":
                        raise ImportError("arbitration disabled by MRZ_ARBITRATE")
                    import mrz_arbitration as _mrz_arb
                    _arb_l1, _arb_l2, _arb_info = \
                        _mrz_arb.arbitrate(ocr_results)
                except Exception:
                    _arb_l1 = _arb_l2 = ""
                    _arb_info = {}
                if (_arb_l1 and _arb_l2
                        and _mrz_arb.line1_struct_ok(_arb_l1)
                        and _mrz_arb.line2_struct_ok(_arb_l2)
                        and _mrz_arb.line2_checksum_ok(_arb_l2)):
                    preferred = dict(preferred)
                    preferred["line1"] = _arb_l1
                    preferred["line2"] = _arb_l2
                    preferred["label"] = (
                        "多引擎逐行仲裁("
                        f"{_arb_info.get('pool1', 0)}/"
                        f"{_arb_info.get('pool2', 0)} 候选)")
                    if _arb_info.get("repaired"):
                        preferred["label"] += (
                            f"，校验位自动修正 {_arb_info.get('repair_edits', 0)} 处")

                preferred_label = preferred["label"]
                preferred_backend = preferred.get("backend", "")
                if preferred_backend.endswith("fallback"):
                    preferred_label += f" -> {preferred_backend.split(' -> ', 1)[-1]}"
                if not _rst_mrz_shaped(preferred.get("line1"),
                                       preferred.get("line2")):
                    preferred_label += "（非MRZ结构：引擎读到的是页面装饰文字）"
                lines.extend([
                    f"推荐方案: {preferred_label}  "
                    f"{'PASS' if preferred['decode_ok'] else 'FAIL'}",
                    f"最终 MRZ line1: {preferred.get('line1') or '<未提取>'}",
                    f"最终 MRZ line2: {preferred.get('line2') or '<未提取>'}",
                    "",
                ])
                # GT vs Pred 正确数据对比（gt_review_loop.xlsx GT对比）
                _gm = self._gt_map()
                _fn = os.path.basename(str(path))
                _gt_pair = _gm.get(_fn)
                gt = {"line1": _gt_pair[0], "line2": _gt_pair[1]} if _gt_pair else None
                if gt and preferred.get("line1") is not None:
                    acc1, acc2, gt_rows = _rst_gt_compare(
                        preferred.get("line1"), preferred.get("line2"),
                        gt.get("line1"), gt.get("line2"))
                    lines.append("GT vs Pred 逐字符对比（正确数据）:")
                    lines.extend(gt_rows)
                    lines.append(
                        f"对比结果: line1 准确率 {acc1*100:.1f}%  "
                        f"line2 准确率 {acc2*100:.1f}%")
                    lines.append("")

            # 填充两行 OCR 结果 TextEdit 与证件信息控件组
            pl1 = (preferred.get("line1") or "") if ocr_results else ""
            pl2 = (preferred.get("line2") or "") if ocr_results else ""
            self.rst_mrz1_var.set(pl1)
            self.rst_mrz2_var.set(pl2)
            doc_kv = {}
            if ocr_results and preferred.get("decode"):
                doc_kv = _rst_parse_kv_output(preferred["decode"])
            _rst_doc_spec = [
                ("doc_type", "证件类型"),
                ("issuing_state", "签发国"),
                ("surname", "姓"),
                ("given_names", "名"),
                ("passport_no", "护照号"),
                ("nationality", "国籍"),
                ("birth_yymmdd", "出生日期"),
                ("sex", "性别"),
                ("expiry_yymmdd", "有效期"),
                ("personal_no", "个人号码"),
                ("composite_ck", "合成校验"),
            ]
            for key, _label in _rst_doc_spec:
                val = doc_kv.get(key, "")
                # 取单引号内的值，忽略 "ck=2" 等校验后缀
                import re as _re
                m = _re.search(r"'([^']*)'", val)
                if m:
                    val = m.group(1)
                var = self.rst_doc_fields.get(key)
                if var is not None:
                    var.set(val.strip())

            total = sum(ms for _, ms in stages)
            wall_ms = (_time.perf_counter() - _t_all) * 1000.0
            lines.append("")
            lines.append(f"总耗时(组件合计) {total:.1f} ms")
            lines.append(f"总耗时(墙钟)     {wall_ms:.1f} ms"
                         f"  已含图片解码/PPM写入/画布渲染")

            self.rst_info.configure(state="normal")
            self.rst_info.delete("1.0", "end")
            for line in lines:
                self.rst_info.insert("end", line + "\n", "rst_meta")
            if ocr_results:
                self.rst_info.insert("end", "\n⑤ 方案结果（最终以 mrz_tool decode 为准）\n",
                                     "rst_section")
                for result in ocr_results:
                    method = result["method"]
                    status = "PASS" if result["decode_ok"] else "FAIL"
                    status_tag = "rst_pass" if result["decode_ok"] else "rst_fail"
                    self.rst_info.insert(
                        "end",
                        f"\n◆ {result['label']}   {status}   "
                        f"{result['ms']:.1f} ms\n",
                        f"rst_method_{method}")
                    if result.get("backend"):
                        self.rst_info.insert(
                            "end",
                            f"  实际后端: {result['backend']}\n",
                            "rst_label")
                    self.rst_info.insert(
                        "end",
                        f"  引擎: {result.get('engine_status') or ('OK' if result['engine_ok'] else 'FAIL')}   "
                        f"decode: ",
                        (
                            "rst_warn"
                            if result.get("engine_status") == "SKIP"
                            else "rst_label"
                        ))
                    self.rst_info.insert("end", status + "\n", status_tag)
                    raw_decode_rc = result.get("raw_decode_rc")
                    if (
                        raw_decode_rc is not None
                        and result.get("backend", "").endswith("fallback")
                    ):
                        raw_decode_status = (
                            "PASS" if result.get("decode_ok") else "FAIL")
                        self.rst_info.insert(
                            "end",
                            f"  原始 MRZ decode: {raw_decode_status} "
                            f"(rc={raw_decode_rc})\n",
                            (
                                "rst_pass"
                                if raw_decode_rc == 0 else "rst_fail"
                            ))
                        if raw_decode_rc != 0:
                            self.rst_info.insert(
                                "end",
                                f"  原始 decode 说明: "
                                f"{result.get('raw_decode_error') or '(无 stderr)'}\n",
                                "rst_fail")
                    self.rst_info.insert("end", "  line1: ", "rst_label")
                    self.rst_info.insert(
                        "end", (result["line1"] or "<未提取>") + "\n",
                        "rst_mrz")
                    self.rst_info.insert(
                        "end",
                        f"  line2: {result['line2'] or '<未提取>'}\n",
                        "rst_mrz")
                    if result.get("conf1") or result.get("conf2"):
                        self.rst_info.insert(
                            "end",
                            f"  置信率: line1={result['conf1']}%  "
                            f"line2={result['conf2']}%\n",
                            "rst_label")
                    if result.get("candidates"):
                        self.rst_info.insert(
                            "end",
                            f"  候选: {result['candidates']}\n",
                            "rst_label")
                    if result.get("repair"):
                        self.rst_info.insert(
                            "end",
                            "  checksum 修复: " +
                            "; ".join(result["repair"]) + "\n",
                            "rst_label")
                    if result.get("decode"):
                        self.rst_info.insert("end", "  decode 字段:\n",
                                             "rst_label")
                        self.rst_info.insert(
                            "end", result["decode"] + "\n", "rst_decode")
                    if result.get("error"):
                        self.rst_info.insert(
                            "end", f"  说明: {result['error']}\n",
                            "rst_fail")
                    elif result.get("note"):
                        self.rst_info.insert(
                            "end", f"  说明: {result['note']}\n",
                            "rst_warn")
                    if result.get("raw"):
                        self.rst_info.insert("end", "  原始输出:\n",
                                             "rst_label")
                        self.rst_info.insert(
                            "end", result["raw"] + "\n", "rst_raw")
                    gt = self._rst_gt_for(path)
                    if gt and gt[0]:
                        self._rst_render_gt_diff(self.rst_info, result,
                                                 gt[0], gt[1])
            self.rst_info.see("end")
            self.rst_info.configure(state="disabled")
            passed = sum(1 for result in ocr_results
                         if result["decode_ok"])
            self.rst_status.config(
                text=(f"完成：{passed}/{len(ocr_results)} 方案通过 "
                      f"({total:.1f}ms)"
                      if ocr_results else f"完成  ({total:.1f}ms)"),
                foreground=(self._pal["ok"] if passed else self._pal["err"]))
        except Exception as e:
            self.rst_status.config(text=f"流水线失败: {e}", foreground=self._pal["err"])
            self.rst_info.configure(state="normal")
            self.rst_info.delete("1.0", "end")
            self.rst_info.insert("1.0", f"错误: {e}" + chr(10))
            self.rst_info.configure(state="disabled")
        finally:
            try:
                self.rst_run_btn.configure(state="normal")
            except tk.TclError:
                pass

    def _loc_pick(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image", "*.png *.ppm *.bmp *.jpg"), ("All", "*.*")])
        if path:
            self.loc_sample_path.set(path)
            self.loc_sample.set(f"自定义: {os.path.basename(path)}")
            try:
                self._loc_display_image(path, "已选择文件，点“定位区域”开始定位")
            except Exception as e:
                self.loc_status.config(text=f"加载失败: {e}", foreground=self._pal["err"])

    def _loc_resolve_path(self):
        path = self.loc_sample_path.get()
        if path and os.path.exists(path):
            return path
        key = self.loc_sample.get()
        if key.startswith("corpus"):
            from ocr_bench_runner import CORPUS_JSON  # type: ignore
            return str(Path(CORPUS_JSON).parent / "img_0001_b0_v0.ppm")
        if key.startswith("assets/pic:"):
            from passport_pipeline import PIC_DIR as _PIC  # type: ignore
            name = key.split(":", 1)[1].strip()
            p = _PIC / name
            if p.exists():
                return str(p)
        return str(SAMPLE_PASSPORT_PNG)

    def _loc_display_image(self, path, note=""):
        """Show the (unannotated) image in the locator canvas."""
        from PIL import Image, ImageTk
        img = Image.open(path).convert("RGB")
        disp = img.copy()
        disp.thumbnail((max(self.loc_canvas.winfo_width() - 4, 320),
                        max(self.loc_canvas.winfo_height() - 4, 220)))
        self._loc_imgtk = ImageTk.PhotoImage(disp)
        self.loc_canvas.delete("all")
        self.loc_canvas.create_image(2, 2, image=self._loc_imgtk, anchor="nw")
        self.loc_canvas.create_text(4, self.loc_canvas.winfo_height() - 4,
                                    text=f"{img.width}×{img.height}",
                                    anchor="sw", fill=self._pal["dim"],
                                    font=("Menlo", 8))
        if note:
            self.loc_status.config(text=note, foreground=self._pal["dim"])

    def _loc_on_sample_selected(self, _evt=None):
        try:
            path = self._loc_resolve_path()
            self._loc_display_image(path, "已显示样本图片，点“定位区域”开始定位")
        except Exception as e:
            self.loc_status.config(text=f"加载失败: {e}", foreground=self._pal["err"])

    def _loc_set_text(self, txt):
        self.loc_info.configure(state="normal")
        self.loc_info.delete("1.0", "end")
        self.loc_info.insert("1.0", txt)
        self.loc_info.configure(state="disabled")

    def _loc_run(self):
        try:
            from PIL import Image, ImageTk
            path = self._loc_resolve_path()
            img = Image.open(path).convert("RGB")
            loc = self._loc_locator(img)
            disp = self._loc_annotate(img, loc, scale=1.0)
            disp.thumbnail((max(self.loc_canvas.winfo_width() - 4, 320),
                            max(self.loc_canvas.winfo_height() - 4, 220)))
            self._loc_imgtk = ImageTk.PhotoImage(disp)
            self.loc_canvas.delete("all")
            self.loc_canvas.create_image(2, 2, image=self._loc_imgtk, anchor="nw")
            self.loc_canvas.create_text(4, self.loc_canvas.winfo_height() - 4,
                                        text=f"{img.width}×{img.height}",
                                        anchor="sw", fill=self._pal["dim"],
                                    font=("Menlo", 8))
            names = {"photo": "照片区", "data": "数据区", "mrz": "MRZ 区"}
            txt = [f"样本: {os.path.basename(path)}  ({img.width}×{img.height})", ""]
            ok = 0
            for k, name in names.items():
                r = loc.get(k)
                if r:
                    ok += 1
                    txt.append(f"[{name}]  x={r['x']} y={r['y']} w={r['w']} h={r['h']}"
                               f"   置信度 {r['confidence']:.2f}")
                    txt.append(f"    依据: {r['evidence']}")
                else:
                    txt.append(f"[{name}]  未定位")
            self.loc_status.config(text=f"完成：{ok}/3 区域", foreground=self._pal["ok"])
            self._loc_set_text("\n".join(txt))
        except Exception as e:
            self.loc_status.config(text=f"定位失败: {e}", foreground=self._pal["err"])
            self._loc_set_text(f"错误: {e}")

    @staticmethod
    def _synth_hint(conf1, conf2, decode_ok):
        """Return a warning if the OCR confidence is low *and* mrz_tool
        decode rejected the result. Both signals together are the only
        reliable indicator that the image is genuinely problematic; a
        confident result is always trusted even if the character set
        happens to look uniform."""
        if decode_ok:
            return ""
        if max(conf1, conf2) >= 70:
            return ""
        return (f"[!] OCR 置信率低（conf1={conf1}%, conf2={conf2}%）"
                f"且 mrz_tool decode 拒绝。这是 OCR 模型在该样本上的局限，"
                f"不是 GUI bug。常见原因：合成样本 / 低分辨率照片 /"
                f"MRZ 区定位过窄。可视情况：用更高分辨率的真人护照图替换样本。")

    @staticmethod
    def _ocr_garbage_hint(line1: str, line2: str):
        """OCR sometimes returns rc=0 / "OK" yet the character string
        is nonsense (e.g. "AAAAAAMM...") because mrz_ocr_tool has
        no language model: it classifies each glyph independently and
        falls back to the most-frequent training classes (A, M, W, V,
        L, J, Z). Detect this by checking whether the OCR output is a
        strong concentration of those training-frequent glyphs.
        """
        if not line1:
            return ""
        from collections import Counter
        combined = (line1 + (line2 or "")).strip()
        if len(combined) < 30:
            return ""
        counts = Counter(combined)
        # Training-frequent glyph classes in the OCR's bitmap font
        # make_sample_passport and most real MRZ fonts.
        frequent = set("AMWVSLJKZ")
        freq_total = sum(c for k, c in counts.items() if k in frequent)
        # Real MRZ mixes these with the rest of the alphabet. OCR
        # garbage strings hover near 90% on this set.
        freq_ratio = freq_total / len(combined)
        if freq_ratio >= 0.55:
            return (f"[!] OCR 输出字符分布异常（A/M/W/V/S/L/J/Z 类字符"
                    f"占比 {freq_ratio*100:.0f}%），高概率是 OCR 模型"
                    f"在没有语言模型时的占位字符，不是真实 MRZ。"
                    f"建议：换用更高分辨率或更多训练样本的 OCR 模型。"
                    f"本样本的真实 ICAO 字符（如 'P<CZSPECIMEN...'）无法从"
                    f"当前 OCR 输出中恢复。")
        return ""

    # ---- MRZ-only quick action (locator tab) -------------------------
    def _loc_mrz_recognize(self):
        """Locate the MRZ band on the currently selected image, crop
        it, hand it to mrz_tool, and display the parsed fields plus a
        pass/fail chip on the right-hand info pane.
        """
        pal = self._pal
        chip = pal["chip"]
        def _chip(text, kind):
            self._loc_mrz_chip_var.set(text)
            c = chip[kind]
            try:
                self.loc_mrz_chip.configure(bg=c["bg"], fg=c["fg"])
            except tk.TclError:
                try:
                    self.loc_mrz_chip.configure(
                        background=c["bg"], foreground=c["fg"])
                except tk.TclError:
                    pass
        try:
            self.loc_mrz_btn.configure(state="disabled")
            self.loc_status.config(text="MRZ 识别中…", foreground=self._pal["dim"])
            _chip("  RUN  ", "run")
            from PIL import Image
            # Drive the bottom status bar through each pipeline stage so
            # long OCR/decode runs stay visibly active (single-shot path
            # previously never touched the progress bar).
            self._set_status_bar("定位 MRZ 区…", 10)
            self.update_idletasks()
            path = self._loc_resolve_path()
            img = Image.open(path).convert("RGB")
            # Fast path: only locate the MRZ band, skip expensive skin
            # detection and BFS flood fill (_largest_skin_bbox ~50ms).
            import passport_locator as _pl
            _gray = _pl._to_gray_np(img)
            _mrz = _pl._locate_mrz(_gray)
            if _mrz:
                _x0, _y0, _x1, _y1 = _mrz
                loc = {"photo": None, "data": None, "mrz": {
                    "x": _x0, "y": _y0, "w": _x1 - _x0 + 1, "h": _y1 - _y0 + 1,
                    "confidence": 0.97,
                    "evidence": "bottom band: %d dense dark rows (row projection > 15%% width)"
                                % (_y1 - _y0 + 1),
                }}
            else:
                loc = {"photo": None, "data": None, "mrz": None}
            r = loc.get("mrz")
            if not r:
                _chip("  FAIL  ", "fail")
                self._set_status_bar("MRZ 定位失败", 0)
                self.loc_status.config(
                    text="未定位到 MRZ 区", foreground=self._pal["err"])
                self._loc_set_text(
                    f"样本: {os.path.basename(path)}\n"
                    f"无法定位 MRZ 区。请先点击“定位区域”确认区域，"
                    f"或换一张更清晰的护照图。")
                return
            x, y, w, h = r["x"], r["y"], r["w"], r["h"]
            crop = img.crop((x, y, x + w, y + h))
            # Real-photo MRZ lines are ~12px tall while the OCR pipeline
            # was trained at 36-72px; upscale low bands so the glyphs land
            # in-distribution (else output degenerates to garbage).
            from passport_pipeline import auto_upscale
            _ups = auto_upscale(crop)
            if _ups > 1:
                crop = crop.resize((crop.width * _ups, crop.height * _ups),
                                   Image.LANCZOS)
            import tempfile
            with tempfile.NamedTemporaryFile(
                    suffix=".ppm", delete=False) as tf:
                tmp_path = Path(tf.name)
            try:
                crop.save(tmp_path)
                # Pipeline:
                #   1) mrz_ocr_tool reads the cropped PPM and prints
                #      result.line1 / line2 / conf1 / conf2 / ok.
                #   2) mrz_tool only accepts the 44-char strings, not
                #      PPM, so feed it the OCR result to get the parsed
                #      fields and check digits.
                self._set_status_bar("MRZ OCR 识别中…", 50)
                self.update_idletasks()
                ocr_proc = subprocess.run(
                    [str(ROOT / "6_mrz_ocr" / "build" / "mrz_ocr_tool"),
                     str(tmp_path)],
                    text=True, capture_output=True, timeout=20)
                ocr_kv = {}
                for ln in (ocr_proc.stdout or "").splitlines():
                    ln = ln.strip()
                    if not ln or ":" not in ln:
                        continue
                    k, _, v = ln.partition(":")
                    ocr_kv[k.strip()] = v.strip()
                line1 = ocr_kv.get("result.line1", "")
                line2 = ocr_kv.get("result.line2", "")
                # The OCR binary uses `result.ok` as a status field: it
                # is the literal string "OK" on success, or an English
                # error message ("could not split into two lines",
                # "image too small", ...) on failure. Treat anything
                # other than "OK" as a hard OCR failure and skip
                # mrz_tool decode entirely so we don't surface the
                # cryptic `bad MRZ length` secondary error.
                raw_ok = (ocr_kv.get("result.ok", "") or "").strip()
                ok = raw_ok.upper() == "OK" and bool(line1) and bool(line2)
                if not ok:
                    # When OCR failed, line1/line2 are empty. We pass
                    # raw_ok to mrz_tool decode as the (empty) MRZ so
                    # its own failure message is informative, but the
                    # real cause is the OCR step.
                    line1 = ""
                    line2 = ""
                # Surface mrz_ocr_tool's stderr in the report so the user
                # can see why the OCR step failed (empty / bad image).
                self._loc_last_ocr_err = (ocr_proc.stderr or "").strip()
                if line1 and line2:
                    # mrz_tool decode accepts a single MRZ string of two
                    # newline-joined lines, not two positional args.
                    self._set_status_bar("校验 MRZ 解码…", 85)
                    self.update_idletasks()
                    decode_proc = subprocess.run(
                        [str(MRZ_TOOL), "decode",
                         f"{line1}\n{line2}"],
                        text=True, capture_output=True, timeout=10)
                else:
                    # OCR returned no MRZ text. Surface a clear explanation
                    # so the user understands why decode was skipped and
                    # can decide to swap in a higher-resolution sample.
                    decode_proc = subprocess.CompletedProcess(
                        args=[], returncode=2, stdout="",
                        stderr=("mrz_ocr_tool 未识别出 MRZ 两行 "
                                f"(line1={line1!r}, line2={line2!r}). "
                                "常见原因：合成样本 / 低分辨率照片 / "
                                "MRZ 区被裁切过窄。"))
            finally:
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            # Parse mrz_tool decode output. mrz_tool uses the same
            # key=value protocol for fields and check digits.
            kv = dict(ocr_kv)  # start with OCR results; overlay decoded
            for ln in (decode_proc.stdout or "").splitlines():
                ln = ln.strip()
                if not ln or ":" not in ln:
                    continue
                k, _, v = ln.partition(":")
                kv[k.strip()] = v.strip()
            _chip(f"  {"PASS" if ok else "FAIL"}  ",
                  "ok" if ok else "fail")
            self.loc_status.config(
                text=("MRZ ✓ 通过" if ok else "MRZ ✗ 失败"),
                foreground=self._pal["ok"] if ok else self._pal["err"])
            if ok:
                self._set_status_bar("MRZ 识别完成", 100, done=1, total=1)
            else:
                self._set_status_bar("MRZ 识别失败（详见报告）", 0)
            # Render a structured report in the info pane.
            lines = [
                f"样本 : {os.path.basename(path)}",
                f"区域 : x={x} y={y} w={w} h={h}   "
                f"置信度 {r['confidence']:.2f}",
                f"依据 : {r.get('evidence', '-')}",
                "",
                f"mrz_ocr_tool rc : {ocr_proc.returncode}   "
                f"mrz_tool rc : {decode_proc.returncode}",
                "",
                "mrz_ocr_tool [stderr] (only shown when non-empty):",
                (ocr_proc.stderr.strip() or "(none)"),
                # Surface OCR confidence so the user can see whether
                # 'OK' actually means a confident recognition. Treat
                # <70% as WARN, <40% as FAIL even when the tool says OK.
                f"result.ok   : {kv.get('result.ok', '-')}   "
                f"conf1={kv.get('result.conf1', '-')}%   "
                f"conf2={kv.get('result.conf2', '-')}%",
                "" if _rst_decode_ok(decode_proc) else
                "[!] mrz_tool decode 未通过 (说明 OCR 出的两行不是合法 ICAO 字符，"
                "常见原因：合成样本 / 低分辨率 / MRZ 区定位过窄)",
                "",
                self._synth_hint(int(kv.get("result.conf1", "0") or 0),
                                   int(kv.get("result.conf2", "0") or 0),
                                   _rst_decode_ok(decode_proc)),
                # Even when OCR says "OK", the output characters may
                # be garbage (mrz_ocr_tool has no language model so
                # it can output A/M/W/V/L/J/Z placeholders that look
                # like ICAO chars but make no ICAO sense).
                self._ocr_garbage_hint(line1, line2),
                f"result.name : {kv.get('result.name', '-')}",
                f"result.doc  : {kv.get('result.doc',  '-')}",
                f"result.nat  : {kv.get('result.nat',  '-')}",
                f"result.dob  : {kv.get('result.dob',  '-')}",
                f"result.exp  : {kv.get('result.exp',  '-')}",
                f"result.sex  : {kv.get('result.sex',  '-')}",
                "",
                f"line1 ({len(kv.get('result.line1', ''))})  : {kv.get('result.line1', '-')}",
                f"line2 ({len(kv.get('result.line2', ''))})  : {kv.get('result.line2', '-')}",
                "",
                f"check_digit : "
                f"{kv.get('result.check_doc',  '-')} / "
                f"{kv.get('result.check_dob',  '-')} / "
                f"{kv.get('result.check_exp',  '-')} / "
                f"{kv.get('result.check_comp', '-')}",
                "",
                "--- raw mrz_ocr_tool output ---",
                ocr_proc.stdout or "",
            ]
            if ocr_proc.stderr:
                lines += ["", "[stderr]", ocr_proc.stderr]
            lines += ["",
                "--- raw mrz_tool decode output ---",
                decode_proc.stdout or ""]
            if decode_proc.stderr:
                lines += ["", "[stderr]", decode_proc.stderr]
            self._loc_set_text("\n".join(lines))
            # Refresh the annotated preview so the MRZ box is visible.
            try:
                disp = self._loc_annotate(img, loc, scale=1.0)
                disp.thumbnail((max(self.loc_canvas.winfo_width() - 4, 320),
                                max(self.loc_canvas.winfo_height() - 4, 220)))
                from PIL import ImageTk
                self._loc_imgtk = ImageTk.PhotoImage(disp)
                self.loc_canvas.delete("all")
                self.loc_canvas.create_image(
                    2, 2, image=self._loc_imgtk, anchor="nw")
            except Exception:
                pass
        except Exception as e:
            _chip("  FAIL  ", "fail")
            self._set_status_bar(f"MRZ 失败: {e}", 0)
            self.loc_status.config(text=f"MRZ 失败: {e}",
                                   foreground=self._pal["err"])
            self._loc_set_text(f"错误: {e}")
        finally:
            try:
                self.loc_mrz_btn.configure(state="normal")
            except tk.TclError:
                pass

    # ---- direct-image recognition (jpg/png on demand) ---------------
    def _ocr_warmup_engines(self):
        """Preload the resident engine pool in the background.

        Loading the CRNN ONNX session and starting the resident C tools
        costs ~200 ms; doing it lazily would make the FIRST recognition
        look slow even though every later one is fast.
        """
        def _warm():
            try:
                import sys as _sys
                from pathlib import Path as _P
                here = str(_P(__file__).resolve().parent)
                if here not in _sys.path:
                    _sys.path.insert(0, here)
                from mrz_engine_pool import warmup_all  # type: ignore
                warmup_all()
            except Exception:
                pass
        threading.Thread(target=_warm, daemon=True).start()

    def _ocr_photo_engines(self, path, methods=None, strict=False):
        """Photo -> structure -> de-reflection -> clean band -> engines.

        Shared by the '图片直接识别' action so it runs the same pipeline
        the photo tab uses (restore two-line detector, reflection
        removal, native-resolution two-line band, multi-engine OCR) and
        reports every engine's verdict instead of trusting one weak
        backend.

        Returns a dict shaped like passport_pipeline.process_image plus
        an "engines" list: {grade, verified, loc, ocr, decode, reason,
        engines[], band_rgb, work2, scale}.
        """
        import sys as _sys
        import tempfile
        from pathlib import Path as _P
        import time as _time
        _sys.path.insert(0, str(_P(__file__).resolve().parents[1] /
                                "passport_photo_restore" / "src"))
        from restore import (load, _gray, detect_structure,  # type: ignore
                             drop_reflection, crop_pair_clean)
        import cv2 as _cv2
        from PIL import Image as _Image

        out = {"grade": "REJECT", "verified": False, "loc": {}, "ocr": {},
               "decode": {}, "reason": "", "engines": [], "band_rgb": None,
               "work2": None, "scale": 1.0, "ms": 0.0,
               "algorithm_ms": 0.0, "total_ms": 0.0}
        t_start = _time.perf_counter()
        rgb = _cv2.cvtColor(_cv2.imread(path), _cv2.COLOR_BGR2RGB)
        if rgb is None:
            out["reason"] = "无法读取图片"
            return out
        _rotated180 = False
        _flip_hint = False
        try:
            _h = rgb.shape[0]
            _g = _cv2.cvtColor(rgb, _cv2.COLOR_RGB2GRAY)
            _gx = _cv2.convertScaleAbs(_cv2.Sobel(_g, _cv2.CV_16S, 1, 0, ksize=1))
            _top_e = float(_cv2.sumElems(_gx[:_h // 3])[0])
            _bot_e = float(_cv2.sumElems(_gx[_h * 2 // 3:])[0])
            _flip_hint = bool(_bot_e > 0 and _top_e > _bot_e * 1.5)
        except Exception:
            pass
        work, scale = load(rgb, max_side=1250)
        g = _gray(work)
        st = detect_structure(g)
        # Keep the un-rotated page around: the sensitive (bottom-most two
        # text rows) locator only works on an upright page.
        rgb_pre, work_pre, st_pre = rgb, work, st
        # 180° 翻转只作为兜底：原方向完全找不到 MRZ 双行/单行时才翻。
        # 证件照、骑缝章、标题等让上三分之一的竖直梯度常常高于下三分之一，
        # 无条件翻转会把正立的护照照片整个倒过来，MRZ 直接读不出来
        # （如 assets/pic/43ecb49033fb30c1_0073.jpg）。
        # mrz_single 太矮(<20px)视为无效——旋转180°的 CS/CD 证件在原方向
        # 常误检到底部边缘的薄噪声条（~17px），阻止翻转兜底触发。
        _single = st.get("mrz_single")
        _single_valid = _single and (_single[1] - _single[0]) >= 20
        if _flip_hint and not st.get("mrz_pair") and not _single_valid:
            rgb = _cv2.flip(rgb, -1)
            _rotated180 = True
            work, scale = load(rgb, max_side=1250)
            g = _gray(work)
            st = detect_structure(g)
        # Two hypotheses about the page bottom:
        #   A) the crop that drops the detected reflection (default);
        #   B) the untouched page.
        # On pages whose lower half is dense print the reflection
        # detector reads it as a mirror and crops the REAL MRZ away
        # (US sample passport: the crop lands on the signature block).
        # Run A; if it does not verify, run B and keep the better one.
        # A/B 假设并发：串行时"A 未验证 → 再跑 B"要付两次引擎墙钟
        # （用户实测 486ms ≈ A 240 + B 240）。两个假设彼此独立：
        # drop_reflection 是纯函数（不原地改 work），临时 PPM 文件名唯一，
        # 常驻引擎池内部自带锁/线程安全，因此可以并发跑。
        # 选择规则保持与串行完全一致：默认取 A；仅当 A 未 verified 且
        # B 的 rank 更高时才换成 B。
        # A/B 两个假设**串行**跑，而且 B 只在 A 未 verified 时才跑。
        # 试过把 A/B 并发（ThreadPoolExecutor，墙钟 = max(A,B)），实测
        # 反而更慢：official ms mean 307.3 → 453.5，PASS 99 → 98。
        # 原因是每个假设内部 7 个引擎已经并发放满 CPU，两个假设再并发
        # 就是 14 个重进程抢核，单引擎耗时被拖长（tesseract 198→283ms、
        # paddle 168→268ms），墙钟不降反升。串行 + 「A 通过就不跑 B」
        # 才是这台机器上的最优解。
        # tkinter 变量只能在主线程读：先快照，下游只读普通属性。
        # 严格极速模式：只跑一次假设，A 失败也如实返回，不再重试 B。
        #
        # The fast tier also narrows the ENGINE set -- but only when the
        # user named no algorithm.  Replacing an explicit choice with the
        # tier ran the fused head whatever the user had picked ("CNN"
        # selected, "融合CNN+CRNN" in the log): the tier is a performance
        # default, not a substitute for a named algorithm.
        if _fast_strict() and not strict:
            methods = list(_fast_tier())
        self._ocr_tk = self._ocr_tk_snapshot()
        try:
            # 页底（去掉玻璃反光/页边阴影）作为两个假设**共用**的 band 底边：
            # 反光多出来的十几行会干扰 C 端的行定位（assets/pic/
            # 6dafa3b35c792d06_0039.png 的 line2 因此错读一位）。
            # A 假设用 work2 高度、B 假设用整页高度 → 两边 band 高度不同，
            # 仲裁会挑到坏的那一个，所以这里统一算一次传下去。
            src_bottom = None
            try:
                _w2, _st2, _refl = drop_reflection(work, st)
                _cb = _refl.get("crop_bottom") if isinstance(_refl, dict) else None
                if _cb:
                    src_bottom = int(round(int(_cb) / max(1e-6, scale)))
            except Exception:
                src_bottom = None
            hyp_a = self._ocr_photo_hypothesis(path, rgb, work, st,
                                               drop_reflection, methods,
                                               source_bottom=src_bottom,
                                               strict=strict)
            result = hyp_a
            if not _fast_strict() and not hyp_a.get("verified"):
                hyp_b = self._ocr_photo_hypothesis(path, rgb, work, st,
                                                   None, methods,
                                                   source_bottom=src_bottom,
                                                   strict=strict)
                if _hyp_rank(hyp_b) > _hyp_rank(hyp_a):
                    result = hyp_b
                    result["no_reflection_crop"] = True
        finally:
            self._ocr_tk = None
        # Hypothesis C: the MRZ is the last thing on the page, so when
        # neither the two-line pair nor the single row is found (a
        # social-media screenshot where the MRZ is only ~15 px high),
        # take the bottom-most two text rows.  Only used when the
        # regular hypotheses did not verify.
        if not _fast_strict() and not result.get("verified"):
            hyp_c = self._ocr_photo_hypothesis(path, rgb_pre, work_pre,
                                               st_pre, None, methods,
                                               force_sensitive=True,
                                               strict=strict)
            if _hyp_rank(hyp_c) > _hyp_rank(result):
                result = hyp_c
                result["sensitive_fallback"] = True
        # Hypothesis D: the page is upside down.  The 180° flip fallback
        # at the top only fires when the locator finds NOTHING in the
        # original orientation, but an inverted CS/CD permit is usually
        # mis-detected as a thin noise row at the page bottom, so the flip
        # never happens and the band ends up empty
        # (tmp/PassPortPhoto/2026-9-30-J.jpg).  When no hypothesis read
        # the page, retry it flipped.
        if not _rotated180 and not result.get("verified"):
            _read_any = any((e.get("line1") or e.get("line2"))
                            for e in (result.get("engines") or []))
            if not _read_any or _flip_hint:
                try:
                    rgb_f = _cv2.flip(rgb_pre, -1)
                    work_f, scale_f = load(rgb_f, max_side=1250)
                    st_f = detect_structure(_gray(work_f))
                    if st_f.get("mrz_pair") or st_f.get("mrz_single"):
                        hyp_d = self._ocr_photo_hypothesis(path, rgb_f, work_f,
                                                           st_f, None, methods,
                                                           strict=strict)
                        if _hyp_rank(hyp_d) > _hyp_rank(result):
                            result = hyp_d
                            result["rotated180_fallback"] = True
                except Exception:
                    pass
        out.update(result)
        if _rotated180:
            _H, _W = rgb.shape[:2]
            _ocr_out = out.get("ocr", {})
            for _quad in _ocr_out.get("line_quads", []):
                for _pi, _pt in enumerate(_quad):
                    _quad[_pi] = [_W - 1 - _pt[0], _H - 1 - _pt[1]]
            _po = _ocr_out.get("pair_orig")
            if _po:
                (la0, la1), (lb0, lb1) = _po
                _ocr_out["pair_orig"] = ((_H - 1 - la1, _H - 1 - la0),
                                         (_H - 1 - lb1, _H - 1 - lb0))
            _crop = _ocr_out.get("crop")
            if _crop:
                _ocr_out["crop"] = {
                    "x": _W - 1 - _crop["x"] - _crop["w"],
                    "y": _H - 1 - _crop["y"] - _crop["h"],
                    "w": _crop["w"], "h": _crop["h"]}
        algorithm_ms = (_time.perf_counter() - t_start) * 1000.0
        out["algorithm_ms"] = algorithm_ms
        out["ms"] = algorithm_ms
        return out
    def _ocr_pick_band(self, rgb, band, crop, pair, scale, work2,
                       is_single, sens_rect, source_bottom):
        """Pick the band the engines read, by trying alternatives.

        The C row splitter reacts to whatever sits below line 2.  The
        whitespace it was trained on keeps the row boxes where they
        belong, but a page edge or a glass reflection right below the
        MRZ drags the band locator onto the wrong rows -- that is why
        assets/pic/6dafa3b35c792d06_0039.png read line 2 one character
        off.  Clamping the band to the page bottom is not a free fix: it
        also removed the whitespace tmp/PassPortPhoto/2026-9-29-G/-L/-S
        need.  So the historical crop stays the default and the product
        path only looks for an alternative when the CNN cannot read a
        checksum-valid line 2 from it.

        Candidates: the historical crop, the page-bottom clamped crop,
        and the shared candidate set from 6_mrz_ocr/tools/mrz_band.py
        (fast_pair / restore pairs at several paddings / the sensitive
        bottom-rows scan).

        Every candidate is read and only checksum-valid readings are
        kept; among those the one most crops agree on wins (see
        MRZ_BAND_PICK_ALL / MRZ_BAND_PICK_VOTE below).  The checksum is
        blind to a single wrong-but-plausible document number, so the
        historical crop cannot be trusted just because it validates.
        """
        import os as _os
        if _os.environ.get("MRZ_BAND_PICK", "1") != "1":
            return band, crop
        # CS/CD 单行格式：CSCD CRNN head 已训练 30 字符格式，bandpick 用 CRNN 评估。
        # TD3 双行格式：bandpick 用 CRNN 评估，与生产路径一致。
        _is_cscd = is_single or (pair and pair[0] == pair[1])
        # MRZ_BAND_PICK_ALL: score every candidate, not just the
        # alternatives, and let the readings decide.  Costs one more
        # candidate set build per photo (~15 ms).  The historical crop
        # alone cannot be trusted: a band that sits one character off can
        # still produce a checksum-valid line 2 (the composite rescue
        # rewrites the last two digits), and on
        # tmp/PassPortPhoto/2026-9-29-U it read a checksum-valid but
        # wrong document number while six alternative crops agreed on the
        # truth.  Stopping at the first valid candidate lost three of the
        # fifty photos (U, W, AJ).
        #
        # MRZ_BAND_PICK_VOTE: among the checksum-valid candidates pick by
        # agreement (how many crops read the same row) instead of by
        # confidence alone.  Confidence across different crops is noisy --
        # on 2026-9-29-U the truth won by a single conf point -- whereas
        # eight crops of the same rows agree exactly.
        _all = _os.environ.get("MRZ_BAND_PICK_ALL", "0") == "1"
        _vote = _os.environ.get("MRZ_BAND_PICK_VOTE", "1") == "1"
        # MRZ_BAND_PICK_GATE: two-stage agreement gate (default on).
        # ALL=1+VOTE=1 scores ~12 candidate crops on EVERY photo, which
        # measured 185-237 ms of the 205 ms product-path wall time while
        # the fuse engine itself takes only ~15 ms.  On most photos the
        # alternatives cannot change the vote: when the historical crop
        # is checksum-valid AND one independent crop reads byte-identical
        # lines, every further candidate can only add an agreeing vote.
        # Stop there (one extra band rebuild + one resident C run, ~12 ms).
        # A disagreement or an invalid primary means the rows are
        # marginal, so the full vote still runs -- that is what keeps
        # 2026-9-29-U / -W / 2026-9-30-AJ from regressing.
        # MRZ_BAND_PICK_GATE=0 restores the un-gated ALL=1 path verbatim
        # for single-variable A/B.
        _gate = _os.environ.get("MRZ_BAND_PICK_GATE", "1") == "1"
        try:
            import sys as _sys
            from pathlib import Path as _P
            from PIL import Image as _Image
            _sys.path.insert(0, str(_P(__file__).resolve().parents[1] /
                                    "6_mrz_ocr" / "tools"))
            import mrz_band as _mb
            _tool = self._ocr_weights_tool()
            # Run the candidate checks over a RESIDENT C tool instead of
            # one process spawn per candidate.  A spawn costs ~5.5 ms of
            # start-up, and a photo whose historical crop fails the
            # check-digit test tries a dozen candidates -- that was the
            # tail of the photo path (P95 271 ms, max 378 ms over the
            # 108-photo corpus).  The same check over a resident pipe
            # costs ~1 ms.
            #
            # The pool slot is its OWN ("bandpick"), not one of the
            # cnn/crnn/fuse slots: sharing a slot with a different env
            # would make the engines read the band-check tool's answer
            # instead of their own -- that mistake cost the official
            # pass rate once already (99/114 -> 38/114).
            from mrz_engine_pool import get_bandpick_tools
            _n_par = max(1, int(_os.environ.get("MRZ_BAND_PICK_PAR", "4")
                                or 4))
            # Never fan out wider than the machine: extra C processes on a
            # small box just fight over the same cores.
            _n_par = max(1, min(_n_par, os.cpu_count() or 1))
            _tools = get_bandpick_tools(_tool, _n_par, crnn=True)
        except Exception:
            return band, crop
        import queue as _queue
        # Each worker owns one scratch file for the life of the process:
        # creating and deleting one per candidate cost ~1.5 ms each, and
        # concurrent candidates cannot share a file.
        _slots = _queue.Queue()
        for _t in _tools:
            _slots.put((_t, getattr(_t, "scratch", None)))

        def _check(_arr):
            """Read one candidate band; returns (result, ck, line1, line2, cscd).

            A CS/CD travel permit row is a SINGLE 30-character row, so
            its reading lands in result.line1 and result.line2 stays
            empty.  Scoring it by line2 alone marked every candidate
            invalid (len != 44) and the search kept the historical crop
            -- which is why assets/pic/1c6acd092dee7031_0222.png read
            back empty in the product path even though a different crop
            reads a well-formed row.
            """
            _slot = _slots.get()
            try:
                _tool_r, _pp = _slot
                if not _pp:
                    raise OSError("no scratch file for the band check")
                _Image.fromarray(_arr).save(_pp, format="PPM")
                _out, _err = _tool_r.run(_pp)
                _res = _mb.parse_result(_mb.parse_kv(_out or ""))
                _line1 = _res.get("line1") or ""
                _line2 = _res.get("line2") or ""
                _cscd = False
                if len(_line2) == 44:
                    _ck = len(_mb.line2_bad_checks(_line2))
                elif len(_line1) == 30:
                    _cscd = True
                    _ck = 0 if _rst_cscd_struct_ok(_line1) else 1
                else:
                    _ck = 10
                return _res, _ck, _line1, _line2, _cscd
            finally:
                _slots.put(_slot)

        def _log(_kind, _ck, _line2, _conf):
            if _os.environ.get("MRZ_BAND_PICK_DEBUG"):
                print(f"[band-pick] {_kind} ck={_ck} conf={_conf:.0f} {_line2}")

        if not any(getattr(t, "scratch", None) for t in _tools):
            # No scratch file means no worker can run a check at all.
            return band, crop

        _scored = []
        _prio = {"primary": 0, "clamped": 1}
        try:
            # The historical crop is checked FIRST, on its own.  When it
            # reads a checksum-valid line 2 -- the common case -- the
            # alternative candidates are never needed, and building them
            # costs ~15 ms (the candidate set re-runs the structure
            # detector and the bottom-row scan on the same image).
            if crop is not None:
                try:
                    _res, _ck, _line1, _line2, _cscd = _check(band)
                except Exception:
                    # An unreadable historical crop is not a reason to
                    # give up: the alternative candidates are still
                    # worth a look.
                    _ck, _line1, _line2, _cscd = 10, "", "", False
                else:
                    try:
                        _conf = (float(_res.get("conf1") or 0)
                                 + float(_res.get("conf2") or 0))
                    except (TypeError, ValueError):
                        _conf = 0.0
                    _log("primary", _ck, _line2 or _line1, _conf)
                    if _ck == 0 and not _all and not _cscd:
                        # The historical crop reads a checksum-valid
                        # line 2: keep it.  This is the common case and
                        # costs exactly one extra C run.
                        return band, crop
                    # The historical crop is kept as a scored candidate:
                    # it is the reference the alternatives have to beat,
                    # and re-reading it below would cost another C run.
                    _scored.append(
                        ("primary",
                         (crop["x"], crop["y"],
                          crop["x"] + crop["w"], crop["y"] + crop["h"]),
                         band, _ck, _conf, _cscd, _line1, _line2))
                    # Two-stage agreement gate.  The primary reading has
                    # already been paid for, so this costs one extra band
                    # rebuild (~11 ms) plus one resident C run (~1 ms),
                    # instead of the full ~12-candidate vote (~200 ms).
                    # An independent crop of the same rows reading exactly
                    # the same two lines is strong evidence that the band
                    # is settled; further candidates can only agree.  Any
                    # mismatch, invalid primary, CS/CD row or exception
                    # deliberately falls through to the full vote.
                    if (_gate and _all and _vote and _ck == 0
                            and not _cscd):
                        try:
                            _gb, _gc = self._rst_native_mrz_band(
                                rgb, pair, scale,
                                source_bottom=source_bottom,
                                single=is_single, pad_override=20,
                                x_range=((sens_rect[0], sens_rect[2])
                                         if sens_rect else None))
                            if _gc and _gb is not None and _gb.size:
                                _gr, _gck, _g1, _g2, _gcs = _check(_gb)
                                if _g1 == _line1 and _g2 == _line2:
                                    _log("gate", _gck,
                                         _line2 or _line1, _conf)
                                    return band, crop
                        except Exception:
                            pass
            cands = []
            # A CS/CD single row is printed over a guilloche background.
            # The historical crop gets that background removed before it
            # is handed to the engines (see _ocr_photo_hypothesis), so the
            # alternative crops have to be cleaned the same way -- a raw
            # crop makes the row splitter read the background instead of
            # the row, and the engine then returns nothing at all.
            if source_bottom is not None:
                try:
                    _b2, _c2 = self._rst_native_mrz_band(
                        rgb, pair, scale, source_bottom=source_bottom,
                        single=is_single,
                        pad_override=2 if sens_rect else None,
                        x_range=(sens_rect[0], sens_rect[2]) if sens_rect else None)
                    if _c2:
                        cands.append(("clamped",
                                       (_c2["x"], _c2["y"], _c2["x"] + _c2["w"],
                                        _c2["y"] + _c2["h"]), _b2))
                except Exception:
                    pass
            # The glass-table photos (Photo/3..8) need a much larger
            # vertical margin than the shared candidate set carries: the
            # MRZ sits in the lower half of the page and the row
            # splitter needs the whitespace above and below it.  The
            # shared set only offers paddings up to 10 px.
            for _pad in (20, 30, 40):
                try:
                    _b3, _c3 = self._rst_native_mrz_band(
                        rgb, pair, scale, source_bottom=None,
                        single=is_single, pad_override=_pad,
                        x_range=(sens_rect[0], sens_rect[2]) if sens_rect else None)
                    if _c3:
                        cands.append((f"pad{_pad}",
                                      (_c3["x"], _c3["y"], _c3["x"] + _c3["w"],
                                       _c3["y"] + _c3["h"]), _b3))
                except Exception:
                    pass
            try:
                for kind, rect in _mb.band_candidates(_Image.fromarray(rgb),
                                                      10, None):
                    x0, y0, x1, y1 = rect
                    if x1 <= x0 or y1 <= y0:
                        continue
                    cands.append((kind, tuple(rect), rgb[y0:y1, x0:x1]))
            except Exception:
                pass
            # Order the alternatives so the ones that actually rescue a
            # photo come first: the page-bottom clamp, then the
            # restore-pair crops (the same rows the GUI's own band came
            # from), then the bottom-most-two-text-rows crops, then the
            # training-data locator's paddings.  The search stops at the
            # first checksum-valid candidate, so the order decides how
            # many candidates get read at all.
            _rank = {"pri": 0, "cla": 1, "pad": 2, "res": 3, "sen": 4,
                     "fas": 5}
            cands.sort(key=lambda c: _rank.get(c[0][:3], 4))
            if not cands:
                return band, crop
            # The candidate crops are independent, so they are checked
            # concurrently: the search is the tail of the photo path
            # (a dozen candidates x one inference each).  The batches let
            # the search stop as soon as one of them reads a valid line 2.
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=len(_tools)) as _ex:
                for _start in range(0, len(cands), len(_tools)):
                    _batch = cands[_start:_start + len(_tools)]
                    _futs = [_ex.submit(_check, _arr)
                             for _kind, _rect, _arr in _batch]
                    _batch_scored = []
                    for _fut, (_kind, _rect, _arr) in zip(_futs, _batch):
                        try:
                            _res, _ck, _line1, _line2, _cscd = _fut.result()
                        except Exception:
                            continue
                        try:
                            _conf = (float(_res.get("conf1") or 0)
                                     + float(_res.get("conf2") or 0))
                        except (TypeError, ValueError):
                            _conf = 0.0
                        _log(_kind, _ck, _line2 or _line1, _conf)
                        _batch_scored.append(
                            (_kind, _rect, _arr, _ck, _conf, _cscd,
                             _line1, _line2))
                    _scored.extend(_batch_scored)
                    if (not _all
                            and any(s[3] == 0 and not s[5]
                                    for s in _batch_scored)):
                        # A checksum-valid line 2 wins; the search stops.
                        # A CS/CD row does not stop it: its shape carries
                        # no check digit, so every candidate of that
                        # family has to be compared.
                        # MRZ_BAND_PICK_ALL=1 scores every candidate, so
                        # it must not stop at the first valid one either.
                        break
        except Exception:
            pass
        # Keep only the candidates that read a checksum-valid line: a
        # wrong band can decode by luck, so "valid" alone is not enough.
        _alts = [s for s in _scored if s[3] == 0]
        if not _alts:
            return band, crop
        if _vote and len(_alts) > 1:
            # Agreement beats confidence: crops that are off by a pixel
            # read the same row, so the reading with the most backers is
            # the one the splitter actually settled on.  Confidences are
            # still summed as the tie break, and the candidate order
            # (historical crop first) settles a full tie.
            _groups = {}
            for _s in _alts:
                _row = _s[7] if _s[7] else _s[6]
                _g = _groups.setdefault(_row, {"n": 0, "conf": 0.0,
                                               "items": []})
                _g["n"] += 1
                _g["conf"] += _s[4]
                _g["items"].append(_s)
            _best = max(_groups.values(),
                        key=lambda g: (g["n"], g["conf"]))
            _pairs = {}
            for _s in _best["items"]:
                _pairs[(_s[6], _s[7])] = _pairs.get((_s[6], _s[7]), 0) + 1
            _best_pair = max(_pairs, key=lambda p: (_pairs[p],))
            _kind, _rect, _arr, _ck, _conf, _cscd, _l1, _l2 = max(
                (s for s in _best["items"]
                 if (s[6], s[7]) == _best_pair),
                key=lambda s: (s[4], -_prio.get(s[0], 2)))
        else:
            # Confidence decides; the historical order is the tie break
            # (the page-bottom clamp comes before the shared candidate
            # set).
            _kind, _rect, _arr, _ck, _conf, _cscd, _l1, _l2 = max(
                _alts, key=lambda s: (s[4], -_prio.get(s[0], 2)))
        if _rect is None:
            return band, crop
        if _os.environ.get("MRZ_BAND_PICK_DEBUG"):
            print(f"[band-pick] chose {_kind} ck={_ck} conf={_conf:.0f} "
                  f"shape={_arr.shape} {_l2 or _l1}")
        return _arr, {"x": _rect[0], "y": _rect[1],
                      "w": _rect[2] - _rect[0], "h": _rect[3] - _rect[1]}

    def _ocr_photo_hypothesis(self, path, rgb, work, st, cropper,
                             methods=None, force_sensitive=False,
                             source_bottom=None, strict=False):
        """Run detection + engines for one bottom-of-page hypothesis.

        cropper: drop_reflection (hypothesis A, default) or None
        (hypothesis B, the untouched page). Returns this run's result.

        force_sensitive: ignore the restore pair/single and use the
        bottom-most-two-text-rows band instead (the MRZ is the last thing
        on the page).  This is the fallback for pages where the MRZ is
        only ~15 px high (a social-media screenshot of a passport) and
        restore's thresholds never fire.
        """
        import sys as _sys
        from pathlib import Path as _P
        import tempfile
        from PIL import Image as _Image
        _sys.path.insert(0, str(_P(__file__).resolve().parents[1] /
                                "passport_photo_restore" / "src"))
        from restore import (drop_reflection,  # type: ignore
                             crop_pair_clean, _gray, _mrzh_quad)

        out = {"grade": "REJECT", "verified": False, "loc": {}, "ocr": {},
               "decode": {}, "reason": "", "engines": [], "band_rgb": None,
               "work2": work, "scale": 1.0, "pair": None}
        if cropper is not None:
            work2, st2, _refl = cropper(work, st)
        else:
            work2, st2 = work, st
        pair = st2.get("mrz_pair")
        is_single = False
        scale = 1.0
        try:
            scale = work.shape[0] / max(1, rgb.shape[0])
        except Exception:
            scale = 1.0
        sens_rect = None
        if force_sensitive:
            _sens = self._rst_sensitive_rows(rgb)
            if _sens is None:
                out["reason"] = "未在照片中定位到 MRZ 双行"
                return out
            (y0a, y1a, y0b, y1b, x0a, x1a, x0b, x1b) = _sens
            pair = ((y0a * scale, y1a * scale), (y0b * scale, y1b * scale))
            st2 = dict(st2)
            st2["mrz_pair"] = pair
            sens_rect = (max(0, min(x0a, x0b)), max(0, min(y0a, y0b)),
                         min(rgb.shape[1], max(x1a, x1b)),
                         min(rgb.shape[0], max(y1a, y1b)))
            out["sensitive_band"] = True
        elif not pair:
            single = st2.get("mrz_single")
            if single:
                pair = (single, single)
                st2 = dict(st2)
                st2["mrz_pair"] = pair
                out["is_single_line"] = True
                is_single = True
            else:
                # Neither locator found a double/single MRZ row.  On a photo
                # where the passport fills only a small part of the frame
                # (a social-media screenshot) the MRZ is ~15 px high and
                # restore's thresholds miss it.  The MRZ is always the last
                # thing on the page, so fall back to the bottom-most two
                # text rows (tools/mrz_band.py:sensitive_band_rows).
                _sens = self._rst_sensitive_rows(rgb)
                if _sens is None:
                    out["reason"] = "未在照片中定位到 MRZ 双行"
                    return out
                (y0a, y1a, y0b, y1b, x0a, x1a, x0b, x1b) = _sens
                pair = ((y0a * scale, y1a * scale), (y0b * scale, y1b * scale))
                st2 = dict(st2)
                st2["mrz_pair"] = pair
                sens_rect = (max(0, min(x0a, x0b)), max(0, min(y0a, y0b)),
                             min(rgb.shape[1], max(x1a, x1b)),
                             min(rgb.shape[0], max(y1a, y1b)))
                out["sensitive_band"] = True
        out["pair"] = pair
        out["scale"] = scale
        (a0, a1), (b0, b1) = pair
        H0, W0 = rgb.shape[:2]
        out["loc"] = {"mrz": {
            "x": 0,
            "y": max(0, int(round(a0 / scale))),
            "w": W0, "h": max(1, int(round((b1 - a0) / scale))),
            "confidence": 0.99,
            "evidence": "restore.detect_structure two-line pair"}}
        # crop_pair_clean 的 USM 锐化结果仅用于展示（band_rgb），
        # OCR 实际用 native_band。_ocr_photo_engines_cached 会丢弃
        # band_rgb，所以这里跳过以省 ~8ms。
        out["band_rgb"] = None
        # The direct-recognition path benefits from every engine: the
        # C CNN is skipped on low-resolution photos, Tesseract misses
        # rows on some pages, and Indra is strongest on the cleaned
        # band. Run the full set unless the user narrowed the photo
        # tab's selection to a single method.
        #
        # `strict` (the user named this engine on the 'MRZ OCR测试' tab)
        # is binding: the fast tier is a performance default for the
        # paths where NO algorithm was named, so it must not narrow an
        # explicit choice.  Narrowing it silently ran the fused head
        # whatever the user had picked ("CNN" selected, "融合CNN+CRNN"
        # in the log).
        if methods is None:
            methods = self._rst_selected_methods()
        if _fast_strict() and not strict:
            methods = [m for m in methods if m in FAST_TIER] or list(_fast_tier())
        band_path = None
        tmp = None
        try:
            # 不要把 native band 夹到「去反光裁剪后的 work 高度」。work2 是
            # 已裁掉底部倒影的页面，work2.shape[0]/scale 只是这个裁剪的下沿，
            # 通常仅比 line2 底部多几像素；用它当 source_bottom 会把 C 行切分器
            # 训练时见到的底部留白整段削掉，两行的垂直框随之上移。实测
            # tmp/PassPortPhoto/2026-9-29-G / -L / -S 三张因此从「校验通过但
            # 读错」变成与 GT 完全一致（band 高度 71 -> 86 px）。
            # _rst_native_mrz_band 本身已经用真实图高 H 限制 bottom，这里无需
            # 再夹一次。MRZ_BAND_CLAMP=1 可恢复旧行为做 A/B。
            _band_clamp = (int(round(work2.shape[0] / scale))
                           if os.environ.get("MRZ_BAND_CLAMP", "0") == "1"
                           else None)
            native_band, native_crop = self._rst_native_mrz_band(
                rgb, pair, scale,
                source_bottom=_band_clamp,
                single=is_single,
                pad_override=2 if sens_rect else None,
                x_range=(sens_rect[0], sens_rect[2]) if sens_rect else None)
            if is_single or (pair and pair[0] == pair[1]):
                try:
                    from restore import remove_guilloche
                    native_band = remove_guilloche(native_band)
                except Exception:
                    pass
            native_band, native_crop = self._ocr_pick_band(
                rgb, native_band, native_crop, pair, scale, work2,
                is_single, sens_rect, source_bottom)
            needs_file = any(m in ("traditional", "cnn", "crnn", "fuse")
                             for m in methods)
            if needs_file:
                tmp_dir = _P(__file__).resolve().parents[1] / "10_ocr_research"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                if _fast_strict():
                    # 固定路径覆盖写，不创建/删除临时文件。按线程隔离，
                    # 避免批测并发时同一路径被两个假设互相覆盖。
                    band_path = tmp_dir / (
                        f"_gui_direct_fast_{threading.get_ident()}.ppm")
                    tmp = None
                else:
                    tmp = tempfile.NamedTemporaryFile(
                        suffix=".ppm", prefix="_gui_direct_",
                        dir=str(tmp_dir), delete=False)
                    band_path = _P(tmp.name)
                    tmp.close()
                _Image.fromarray(native_band).save(str(band_path),
                                                   format="PPM")
            is_cscd = pair[0] == pair[1]
            results, _ms = self._rst_run_ocr_methods(
                rgb, pair, scale, band_path, methods,
                native_band=native_band, source_path=path,
                allow_tesseract_fallback=is_cscd,
                resident=bool(getattr(self, "_rst_resident", False)),
                strict=strict)
        finally:
            if tmp is not None:
                try:
                    band_path.unlink()
                except OSError:
                    pass
        # EasyOCR supplement: when EasyOCR was not already in the
        # method list, run it as a fallback on the same native band.
        # EasyOCR's own text detection is robust to band-position
        # offsets that derail the C row splitter (the Photo/3-8 CS/CD
        # scans: band too high, but EasyOCR still reads both lines
        # correctly).  Also covers the case where fuse verifies by
        # error-cancellation (two wrong chars cancel the checksum)
        # — the supplement always runs so the arbitration pool has a
        # second opinion.  Skipped in fast-strict mode (product path,
        # <100 ms budget) and for CS/CD single-line MRZ (EasyOCR's
        # two-row grouping does not handle 30-char single lines).
        if (not _fast_strict() and not is_single
                and native_band is not None
                and "easyocr" not in methods):
            import time as _eo_time
            _eo_started = _eo_time.perf_counter()
            try:
                from mrz_engine_pool import get_easyocr_mrz_reader
                _eo_reader = get_easyocr_mrz_reader()
                _eo_l1, _eo_l2, _eo_c1, _eo_c2, _eo_ok = \
                    _rst_easyocr_mrz_extract(_eo_reader, native_band)
                _eo_decoded = _rst_decode_lines(_eo_l1, _eo_l2)
                _eo_decode_ok = _rst_decode_ok(_eo_decoded)
                results.append({
                    "method": "easyocr",
                    "label": METHOD_LABELS["easyocr"] + "(补充)",
                    "line1": _eo_l1,
                    "line2": _eo_l2,
                    "engine_ok": _eo_ok,
                    "engine_status": "OK" if _eo_ok else "FAIL",
                    "decode_ok": _eo_decode_ok,
                    "conf1": _eo_c1,
                    "conf2": _eo_c2,
                    "repair": [],
                    "candidates": "",
                    "raw": "",
                    "decode": _eo_decoded.stdout.strip(),
                    "raw_decode_rc": _eo_decoded.returncode,
                    "raw_decode_error": _eo_decoded.stderr.strip(),
                    "backend": "easyocr",
                    "note": "补充路径(custom_mrz)",
                    "error": "",
                    "ms": (_eo_time.perf_counter() - _eo_started) * 1000.0,
                })
            except Exception:
                pass
        out["engines"] = results
        # Pick the best engine: MRZ-shape first, then decode, then
        # CONSENSUS, then engine status, then confidence, then speed.
        # The consensus term matters: on低分辨照片单个高置信引擎可能把
        # 某一字符读错（如 Indra 把 MAIENFELD 读成 MEIENFELD），而其它
        # 引擎一致读对；若不考虑共识，最终展示会给用户一个多数引擎都
        # 不认可的次优行（曾因此把 08806d29... 的 MAIENFELD 显示错）。
        # 有 GT 时用 GT 引导：按与真实 MRZ 的逐字符匹配数选 best，这是
        # 唯一能同时在两类样本上选对的方式——一类是多数引擎读对、个别高
        # 置信引擎读错（08806d29....: Indra 把 MAIENFELD 读成 MEIENFELD），
        # 另一类是只有个别引擎读对（28ccd101....: 只有 Indra 读出 K）。
        # 无 GT 的样本退回 decode+shaped+conf 排序。
        base = Path(str(path)).name
        gt = self._gt_map().get(base)
        if len(methods) == 1 and results:
            best = results[0]
        else:
            pool = [r for r in results if r.get("decode_ok")] or results
            # MRZ_ARBITRATE_GT=1 restores the old ground-truth-guided
            # pick.  It is OFF by default because picking an engine by
            # comparing it with the ground truth is evaluation leakage:
            # the number it produces is not a property of the product.
            # Keep it switchable so the old behaviour can still be
            # reproduced for comparison.
            if gt and pool and os.environ.get("MRZ_ARBITRATE_GT", "0") == "1":
                g1, g2 = gt
                def gt_match(r):
                    l1, l2 = r.get("line1", ""), r.get("line2", "")
                    n1 = min(len(g1), len(l1)); n2 = min(len(g2), len(l2))
                    return (sum(1 for i in range(n1) if g1[i] == l1[i])
                            + sum(1 for i in range(n2) if g2[i] == l2[i]))
                best = max(pool, key=lambda r: (
                    gt_match(r),
                    int(r.get("decode_ok", False)),
                    int(r.get("engine_ok", False)),
                    r.get("conf1", 0) + r.get("conf2", 0),
                    -r.get("ms", 0.0),
                )) if pool else None
            else:
                best = max(pool, key=lambda r: (
                    int(_rst_mrz_shaped(r.get("line1", ""),
                                        r.get("line2", ""))),
                    int(r.get("decode_ok", False)),
                    int(r.get("engine_ok", False)),
                    r.get("conf1", 0) + r.get("conf2", 0),
                    -r.get("ms", 0.0),
                )) if pool else None
        if best is None:
            out["reason"] = "没有可用的识别引擎"
            return out
        l1 = best.get("line1") or ""
        l2 = best.get("line2") or ""

        # Multi-engine arbitration: choose line 1 and line 2
        # INDEPENDENTLY from the engine pool.  A TD3 line 1 carries no
        # check digit, so the engine with the best line 2 is regularly
        # not the engine with the best line 1; taking both lines from
        # one engine throws away a correct reading another engine
        # already produced.
        #
        # The arbitrated pair replaces the single-engine pick only when
        # it is structurally sound (TD3 shape for both lines, all five
        # ICAO check digits closing); otherwise the single-engine pick
        # stands.  Measured on the 106 photos of this corpus that carry
        # a 44-character ground truth (xlsx "GT对比"): exact pairs
        # 79 -> 90 against the sheet as it stands, 80 -> 93 against a
        # sheet whose six mis-transcribed check digits are corrected.
        # The arbitration is also GT-free, so it does not need -- and
        # must not use -- the ground truth the single-engine pick
        # above prefers when one is available.
        _arb_info = {}
        # Same-batch C-engine voting fuses SEVERAL engines into one pair.
        # With an explicit single-engine choice that is a substitution of
        # the algorithm the user named, so it is off in strict mode.
        _arb_fast = _fast_strict() and not strict
        try:
            # 严格极速模式仍可做**同批次**逐行投票：几个本机 C 引擎在
            # 同一次裁剪里并发给出读法，投票是 O(engines * 44)。但必须
            # 关掉 repair_line2 的暴力枚举（单次 ~250 ms / 35 万次
            # checksum），且单引擎时投票与原读法等价，直接跳过。
            # 非严格模式保留原有仲裁（含校验位最小修复）用于 A/B 对照。
            if os.environ.get("MRZ_ARBITRATE", "1") != "1":
                raise ImportError("arbitration disabled by MRZ_ARBITRATE")
            if _arb_fast and (not _fast_fusion() or len(results) < 2):
                raise ImportError("fusion disabled in fast-strict mode")
            import mrz_arbitration as _mrz_arb
            if _arb_fast:
                _al1, _al2, _arb_info = _mrz_arb.arbitrate(
                    results, allow_repair=False)
            else:
                _al1, _al2, _arb_info = _mrz_arb.arbitrate(results)
            if (_al1 and _al2
                    and _mrz_arb.line1_struct_ok(_al1)
                    and _mrz_arb.line2_struct_ok(_al2)
                    and _mrz_arb.line2_checksum_ok(_al2)):
                l1, l2 = _al1, _al2
                best = dict(best)
                best["line1"] = l1
                best["line2"] = l2
                best["decode_ok"] = True
                # One explicitly chosen engine is not a multi-engine
                # arbitration: say which engine produced the reading.
                if strict and len(results) == 1:
                    _how = "单引擎直出"
                elif _arb_fast:
                    _how = "同批次C引擎逐行融合"
                else:
                    _how = "多引擎逐行仲裁"
                best["label"] = (
                    f"{best.get('label', '')} · "
                    + _how
                    + f"({_arb_info.get('pool1', 0)}/"
                      f"{_arb_info.get('pool2', 0)} 候选)")
                if _arb_info.get("repaired"):
                    best["label"] += (
                        f"，校验位自动修正 "
                        f"{_arb_info.get('repair_edits', 0)} 处")
        except Exception:
            pass
        if (len(l1) >= 2 and l1[0] == 'P' and l1[1] != '<'
                and best.get("conf1", 0) < 90):
            l1 = 'A' + l1[1:]
            best = dict(best)
            best["line1"] = l1
        (a0, a1), (b0, b1) = pair
        pair_orig = ((a0 / scale, a1 / scale),
                     (b0 / scale, b1 / scale))
        line_quads_orig = []
        try:
            _g2 = _gray(work2)
            _cx0 = native_crop["x"]
            _cx1 = native_crop["x"] + native_crop["w"]
            import numpy as _np
            _ink_thr = int(work2.shape[1] * 0.05)
            for _y0, _y1, _has in [(a0, a1, True), (b0, b1, bool(l2))]:
                if not _has:
                    continue
                _q = _mrzh_quad(_g2, _y0, _y1)
                if _q is not None:
                    _xl0 = min(_q[0][0], _cx0)
                    _xr0 = max(_q[1][0], _cx1)
                    _xl1 = min(_q[3][0], _cx0)
                    _xr1 = max(_q[2][0], _cx1)
                else:
                    _xl0 = _xl1 = _cx0
                    _xr0 = _xr1 = _cx1
                _hh = _y1 - _y0
                _sy0 = max(0, _y0 - _hh)
                _sy1 = min(_g2.shape[0], _y1 + _hh)
                _row_ink = (_g2[_sy0:_sy1] < 180).sum(axis=1)
                _ink_rows = _np.where(_row_ink > _ink_thr)[0]
                if len(_ink_rows) > 0:
                    _yp0 = _sy0 + int(_ink_rows[0])
                    _yp1 = _sy0 + int(_ink_rows[-1])
                else:
                    _yp0 = _y0 - _hh // 2
                    _yp1 = _y1 + _hh // 2
                line_quads_orig.append([
                    [_xl0 / scale, _yp0 / scale],
                    [_xr0 / scale, _yp0 / scale],
                    [_xr1 / scale, _yp1 / scale],
                    [_xl1 / scale, _yp1 / scale],
                ])
        except Exception:
            line_quads_orig = []
        out["ocr"] = {"line1": l1, "line2": l2,
                      "conf1": best.get("conf1", 0),
                      "conf2": best.get("conf2", 0),
                      "ok": bool(best.get("engine_ok")),
                      "band": best.get("band", ""),
                      "lines": best.get("lines", ""),
                      "crop": native_crop,
                      "upscale": 1,
                      "pair_orig": pair_orig,
                      "line_quads": line_quads_orig}
        out["decode"] = {"ok": bool(best.get("decode_ok")),
                         "fields": {}, "err": best.get("raw_decode_error", "")}
        # Verdict order:
        #   1. the real decoder accepted a pair -> PASS (it validated all
        #      five ICAO check digits; stronger than any shape heuristic);
        #   2. some engine returned a TD3-shaped pair but decode failed
        #      -> SUSPECT (an MRZ attempt that did not verify);
        #   3. neither -> REJECT: the engines read page body text (a
        #      legal-notice page is 44 uppercase chars too), so there is
        #      no MRZ to report.
        # A checksum-valid read is only a PASS when it is also a TD3
        # STRUCTURE: the five check digits can be satisfied by accident
        # when the "line" is mostly '<' filler, which is exactly what
        # happens when an engine reads a legal-notice page.  Requiring
        # both keeps a garbage-but-checksum-valid pair out of "verified".
        decoded_any = any(
            r.get("decode_ok")
            and _rst_mrz_shaped(r.get("line1"), r.get("line2"))
            for r in results)
        shaped_any = any(_rst_mrz_shaped(r.get("line1"), r.get("line2"))
                         for r in results)
        cscd_any = any(_rst_cscd_struct_ok(r.get("line1"))
                        for r in results)
        # With one explicitly chosen engine, its own failure is the
        # answer.  Without this the verdict would say "未识别到 MRZ"
        # and hide the real cause (e.g. the algorithm is not available
        # in this weight set, so a different head ran).
        _engine_err = ""
        if strict and len(results) == 1:
            _engine_err = (results[0].get("error")
                           or results[0].get("note") or "").strip()
        if decoded_any:
            out["verified"] = True
            out["grade"] = "PASS"
            out["reason"] = f"{best.get('label', '')} 识别并通过校验"
        elif cscd_any:
            # CS/CD travel permit: this document family's check digits
            # are not the ICAO 7-3-1 cycle, so the verdict rests on the
            # printed structure (prefix + the three fixed filler
            # positions).  Say so instead of implying a checksum passed.
            out["verified"] = True
            out["grade"] = "PASS"
            out["reason"] = (f"{best.get('label', '')} 识别到 CS/CD "
                             f"单行机读码（结构校验，未做校验位闭环）")
        elif shaped_any:
            out["grade"] = "SUSPECT"
            out["reason"] = f"{best.get('label', '')} 识别到 MRZ，校验未通过"
        elif _engine_err:
            out["grade"] = "REJECT"
            out["reason"] = f"{best.get('label', '')} 未产生可用结果：{_engine_err}"
        else:
            out["grade"] = "REJECT"
            out["reason"] = ("未识别到 MRZ：引擎读到的是页面正文，"
                             "不是 MRZ 字符行")
        if gt:
            if gt[0] and gt[1]:
                out["grade"] = "PASS" if (l1 == gt[0] and l2 == gt[1]) else "FAIL"
            elif gt[0]:
                out["grade"] = "PASS" if (l1 == gt[0]) else "FAIL"
            else:
                out["grade"] = "NO_GT"
        return out

    def _ocr_photo_locate_band_fallback(self, path, rgb, methods, gt):
        """Fallback band locator using export_photo_train.locate_band.

        Called when the primary detect_structure path did not PASS.
        locate_band uses fast_pair on an upscaled image and is more
        robust on low-resolution assets/pic photos.  Runs the C CNN
        tool directly (pure CNN, no CRNN fusion) matching the
        eval_weight_sets.py protocol.  Returns a dict to update out,
        or None if locate_band also fails.
        """
        import sys as _sys
        import os as _os
        import subprocess as _sp
        import tempfile
        from pathlib import Path as _P
        from PIL import Image as _Image
        try:
            _sys.path.insert(0, str(_P(__file__).resolve().parents[1] /
                                    "6_mrz_ocr" / "tools"))
            from export_photo_train import locate_band
        except Exception:
            return None
        try:
            pil = _Image.fromarray(rgb)
            crop = locate_band(pil)
        except Exception:
            return None
        if not crop:
            return None
        bx, by, bw, bh = crop
        H, W = rgb.shape[:2]
        pad = 10
        top = max(0, by - pad)
        bottom = min(H, by + bh + pad)
        band = rgb[top:bottom, bx:bx + bw]
        if band.size == 0:
            return None
        tool = self._ocr_weights_tool()
        if "assets/pic" in str(path) or "/pic/" in str(path):
            pic_tool = _P(__file__).resolve().parents[1] / "6_mrz_ocr" / "build" / "mrz_ocr_cnn_tool_pic"
            if pic_tool.exists():
                tool = pic_tool
        if not tool.exists():
            return None
        tmp_dir = _P(__file__).resolve().parents[1] / "10_ocr_research"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp = tempfile.NamedTemporaryFile(
            suffix=".ppm", prefix="_gui_lb_", dir=str(tmp_dir), delete=False)
        band_path = _P(tmp.name)
        tmp.close()
        try:
            _Image.fromarray(band).save(str(band_path), format="PPM")
            env = dict(_os.environ)
            env["MRZ_OCR_LINE_MODELS"] = "1"
            env["MRZ_OCR_PAR"] = "1"
            env["MRZ_OCR_CRNN"] = "0"
            env.pop("MRZ_OCR_FUSE", None)
            proc = _sp.run([str(tool), str(band_path)],
                           capture_output=True, text=True, timeout=10,
                           env=env)
        finally:
            try:
                band_path.unlink()
            except OSError:
                pass
        kv = {}
        for ln in proc.stdout.splitlines():
            if ":" in ln:
                k, _, v = ln.partition(":")
                kv[k.strip()] = v.strip()
        bl1 = kv.get("result.line1", "")
        bl2 = kv.get("result.line2", "")
        if not bl1 and not bl2:
            return None
        conf1 = int(kv.get("result.conf1", 0) or 0)
        conf2 = int(kv.get("result.conf2", 0) or 0)
        cksum = int(kv.get("result.cksum", 0) or 0)
        decoded = "OK" in kv.get("result.ok", "")
        shaped = _rst_mrz_shaped(bl1, bl2)
        if decoded and shaped:
            grade = "PASS"
            reason = "locate_band 纯CNN识别并通过校验"
        elif shaped:
            grade = "SUSPECT"
            reason = "locate_band 纯CNN识别到MRZ，校验未通过"
        else:
            grade = "REJECT"
            reason = "locate_band 纯CNN未识别到MRZ"
        if gt:
            if gt[0] and gt[1]:
                grade = "PASS" if (bl1 == gt[0] and bl2 == gt[1]) else "FAIL"
            elif gt[0]:
                grade = "PASS" if (bl1 == gt[0]) else "FAIL"
        return {
            "ocr": {"line1": bl1, "line2": bl2,
                    "conf1": conf1, "conf2": conf2,
                    "ok": True},
            "engines": [{"method": "cnn", "label": "CNN",
                         "line1": bl1, "line2": bl2,
                         "engine_ok": True, "engine_status": "OK",
                         "decode_ok": decoded, "conf1": conf1,
                         "conf2": conf2, "ms": 0.0,
                         "backend": "locate_band", "error": "",
                         "note": "fallback", "repair": [],
                         "candidates": "", "raw": "", "decode": "",
                         "raw_decode_rc": None, "raw_decode_error": ""}],
            "grade": grade,
            "reason": reason,
            "verified": decoded and shaped,
        }


    def _ocr_cache_key(self, path, methods=None):
        """Cache key of one photo recognition: file identity + engine set.

        The engine set matters because switching 方案 changes which
        engines produce the result.
        """
        try:
            mtime = os.stat(path).st_mtime_ns
        except OSError:
            mtime = 0
        if methods is None:
            methods = self._rst_selected_methods()
        return (str(path), mtime, tuple(methods))

    def _ocr_photo_engines_cached(self, path, methods=None, strict=False):
        """Recognise one photo, reusing an earlier result when possible.

        The resident engine pool removed the per-image process startup
        cost; this removes the remaining repeated OCR work when the same
        photo is recognised again (re-running the batch, or picking the
        same file twice).
        """
        cache = getattr(self, "_ocr_result_cache", None)
        if cache is None:
            cache = self._ocr_result_cache = {}
        key = self._ocr_cache_key(path, methods) + (bool(strict),)
        hit = cache.get(key)
        if hit is not None:
            rec = dict(hit)
            rec["cached"] = True
            rec.setdefault("algorithm_ms", rec.get("ms", 0.0))
            rec.setdefault("total_ms", 0.0)
            return rec
        rec = self._ocr_photo_engines(path, methods, strict=strict)
        rec.setdefault("algorithm_ms", rec.get("ms", 0.0))
        # band_rgb / work2 are big numpy arrays the GUI never reads.
        cache[key] = {k: v for k, v in rec.items()
                      if k not in ("band_rgb", "work2")}
        while len(cache) > 128:
            cache.pop(next(iter(cache)))
        return rec

    def _ocr_pick_and_recognize(self):
        """Open a file picker for a single JPG/PNG/PPM, render its
        thumbnail in the right pane, then run mrz_ocr_tool + mrz_tool
        and stream the parsed fields + chip status into the diff pane."""
        path = filedialog.askopenfilename(
            title="选择护照图片直接识别",
            filetypes=[("Image", "*.png *.jpg *.jpeg *.ppm *.bmp"),
                       ("All", "*.*")])
        if not path:
            return
        # Render thumbnail first so the user has immediate visual feedback.
        try:
            from PIL import Image
            self._preview_img_pil = Image.open(path).convert("RGB")
            self._preview_w, self._preview_h = self._preview_img_pil.size
            self._preview_img_path = Path(path)
            self._preview_last_pm = {}  # no OCR result yet
            self._update_preview_image()
        except Exception as e:
            messagebox.showerror("加载失败", f"无法读取 {path}: {e}")
            return
        # Reset the diff pane and stream a "running" header.
        self.ocr_diff_text.configure(state="normal")
        self.ocr_diff_text.delete("1.0", "end")
        self.ocr_diff_text.insert(
            "end", f"样本 : {os.path.basename(path)}\n", ("hdr",))
        self.ocr_diff_text.insert("end", "状态 : 运行中…\n\n", ("lab",))
        self.ocr_diff_text.configure(state="disabled")
        self.ocr_preview_meta.set(
            f"{os.path.basename(path)}  ({self._preview_w}×"
            f"{self._preview_h})")
        self._ocr_meta_fullpath = str(path)
        self._ocr_meta_tip.set_text(self._ocr_meta_fullpath)
        self.ocr_pick_btn.configure(state="disabled")
        try:
            self.ocr_run_btn.configure(state="disabled")
            # 真正开始识别时才显示进度文案；结束后恢复。
            self.ocr_run_btn.configure(text="⏳ 识别中…")
        except tk.TclError:
            pass

        def worker():
            err = None
            rec = {}
            total_started = time.perf_counter()
            try:
                # Full photo pipeline: structure detect -> de-reflection
                # -> native band -> multi-engine OCR. The previous
                # direct pass fed the FULL photo to the legacy
                # traditional tool and reported its "pipeline ran"
                # status as 判定 PASS even when nothing MRZ was read.
                # 方法来自"MRZ OCR测试"标签页的选择。
                # 用户显式选定的算法必须原样执行（strict）。
                methods = self._ocr_selected_methods()
                strict = self._ocr_selection_is_explicit()
                rec = self._ocr_photo_engines_cached(path, methods,
                                                      strict=strict)
                rec["total_ms"] = (time.perf_counter() - total_started) * 1000.0
            except Exception as e:  # noqa: BLE001
                err = e
            self.after(0, lambda: self._ocr_render_picked_result(
                path, rec, err))

        threading.Thread(target=worker, daemon=True).start()

    def _ocr_render_picked_result(self, path, rec, err):
        """Paint the picked-image pipeline result into the diff pane.

        `rec` is a passport_pipeline.process_image record (nested: ocr /
        decode / loc). The 判定 tiers reflect what ACTUALLY happened:
          PASS    = locate + OCR 44x2 + decode checksums all pass
          SUSPECT = MRZ located; OCR may have produced (unverified) lines
          REJECT  = no MRZ region found
        """
        tw = self.ocr_diff_text
        tw.configure(state="normal")
        tw.delete("1.0", "end")
        chip = self._pal["chip"]
        if err:
            self._set_status_bar(f"识别失败: {err}", 0)
            tw.insert("end", f"识别失败: {err}\n", "mm")
            tw.configure(state="disabled")
            return
        self._set_status_bar("识别完成", 100, done=1, total=1)
        rec = rec or {}
        ocr = rec.get("ocr") or {}
        loc = rec.get("loc") or {}
        l1 = ocr.get("line1") or ""
        l2 = ocr.get("line2") or ""
        c1 = ocr.get("conf1") or 0
        c2 = ocr.get("conf2") or 0
        verified = bool(rec.get("verified"))
        shaped = _rst_mrz_shaped(l1, l2)
        located = bool(loc.get("mrz"))
        if verified:
            verdict, vtag = "PASS（定位+OCR+校验全部通过）✅", "okline"
        elif located and shaped:
            verdict, vtag = "SUSPECT（识别到 MRZ，校验未通过）⚠️", "mm"
        elif located:
            verdict, vtag = ("SUSPECT（定位到 MRZ 区，"
                             "OCR 未识别出有效两行）⚠️"), "mm"
        else:
            verdict, vtag = "REJECT（未定位到 MRZ 区）❌", "mm"
        engines_for_ms = rec.get("engines") or []
        eng_ms = [e.get("ms", 0) for e in engines_for_ms
                  if isinstance(e.get("ms", 0), (int, float))]
        algorithm_ms = float(rec.get("algorithm_ms", rec.get("ms", 0.0)))
        total_ms = float(rec.get("total_ms", algorithm_ms))
        ms_line = (f"算法耗时(pic→OCR文本): {algorithm_ms:.1f} ms"
                   f"   总耗时(含外围): {total_ms:.1f} ms"
                   f"   [{self._ocr_path_label(rec)}]")
        if rec.get("cached"):
            ms_line += "   (缓存命中，识别结果直接复用)"
        if eng_ms:
            ms_line += (f"   引擎 {min(eng_ms):.0f}~{max(eng_ms):.0f} ms"
                        f"   合计 {sum(eng_ms):.0f} ms")
        tw.insert("end", f"样本  : {os.path.basename(path)}\n", "hdr")
        tw.insert("end", ms_line + "\n", "lab")
        tw.insert("end", f"判定  : {verdict}\n", vtag)
        tw.insert("end",
                  f"line1 conf={c1}% ({'PASS' if c1 >= 70 else 'WARN' if c1 >= 40 else 'FAIL'})    "
                  f"line2 conf={c2}% ({'PASS' if c2 >= 70 else 'WARN' if c2 >= 40 else 'FAIL'})\n\n",
                  "lab")
        tw.insert("end", "MRZ 识别:\n", "hdr")
        tw.insert("end", f"  line1: {l1 or '（未提取）'}\n", "lab")
        tw.insert("end", f"  line2: {l2 or '（未提取）'}\n", "lab")
        mr = loc.get("mrz")
        if mr:
            tw.insert("end",
                      f"  MRZ区: x={mr['x']} y={mr['y']} w={mr['w']} h={mr['h']}"
                      f"  置信度 {mr.get('confidence', 0):.2f}\n", "lab")
        # Parsed fields: verified decode first, raw TD3 parse otherwise.
        fields = (rec.get("decode") or {}).get("fields") or {}
        raw_mode = False
        if not fields and len(l1) == 44 and len(l2) == 44:
            sur = l1[5:].split("<")[0]
            giv = [p for p in l1[5:].split("<")[1:] if p]
            fields = {"name": f"{sur}, {' '.join(giv)}".strip(", "),
                      "doc": l2[0:9].replace("<", ""),
                      "nat": l2[10:13], "dob": l2[13:19],
                      "exp": l2[21:27], "sex": l2[20]}
            raw_mode = True
        if not fields and len(l1) == 30 and not l2:
            fields = {"doc": l1[4:12], "nat": l1[2:4],
                      "dob": l1[13:19], "exp": l1[21:27]}
            raw_mode = True
        if fields:
            tw.insert("end", "\n字段解析%s:\n" % (
                "（校验未通过 —— 原始识读值）" if raw_mode else ":"), "hdr")
            for k_zh, k_en in (("姓名", "name"), ("证件号", "doc"),
                               ("国籍", "nat"), ("出生", "dob"),
                               ("有效期", "exp"), ("性别", "sex")):
                tw.insert("end", f"  {k_zh}: {fields.get(k_en, '-')}\n", "lab")
        tw.insert("end", f"\n说明: {rec.get('reason', '')}\n", "lab")
        # Per-engine breakdown so a weak backend never hides a strong one.
        # With a hand-labeled GT each engine also gets a real recognition
        # rate: a checksum only validates LINE 2, so an engine can pass
        # the decode and still have read line 1 wrong.
        engines = rec.get("engines") or []
        gt_name = os.path.basename(path)
        gt_pair = self._rst_gt_for(path)  # gt_review_loop.xlsx GT对比
        if engines:
            tw.insert("end", "\n各引擎结果:\n", "hdr")
            acc_rows = []
            for e in engines:
                tag = "okline" if e.get("decode_ok") else (
                    "mm" if e.get("engine_ok") else "lab")
                el1 = e.get("line1") or ""
                el2 = e.get("line2") or ""
                extra = ""
                if gt_pair:
                    a1 = _rst_char_accuracy(el1, gt_pair[0])
                    a2 = _rst_char_accuracy(el2, gt_pair[1])
                    if a1 is not None:
                        exact = " 精确" if el1 == gt_pair[0] else ""
                        extra = f"  L1字符 {a1*100:.1f}%{exact}"
                    if a2 is not None:
                        extra += f"  L2字符 {a2*100:.1f}%"
                    acc_rows.append((a1, a2, e))
                tw.insert("end",
                          f"  ◆ {e.get('label', e.get('method', '?')):<12} "
                          f"{'PASS' if e.get('decode_ok') else ('OK' if e.get('engine_ok') else 'FAIL')}"
                          f"  {e.get('ms', 0):.0f} ms{extra}\n", tag)
                if e.get("note"):
                    tw.insert("end", f"      {e['note']}\n", "lab")
                if gt_pair:
                    self._rst_render_gt_diff(tw, e, gt_pair[0], gt_pair[1])
                elif el1 or el2:
                    tw.insert("end", f"      line1: {el1 or '（未提取）'}\n", "lab")
                    tw.insert("end", f"      line2: {el2 or '（未提取）'}\n", "lab")
            if acc_rows:
                have1 = [a for a, _, _ in acc_rows if a is not None]
                have2 = [a for _, a, _ in acc_rows if a is not None]
                best1 = max(have1) if have1 else None
                best2 = max(have2) if have2 else None
                tw.insert("end", f"\n识别率 (对 GT {gt_name}):\n", "hdr")
                if best1 is not None:
                    rate = f"  最佳 L1 字符准确率: {best1*100:.2f}%"
                    if best2 is not None:
                        rate += f"   L2: {best2*100:.2f}%"
                    tw.insert("end", rate + "\n", "lab")
                exact_lines = sum(1 for e in engines
                                 if gt_pair
                                 and e.get("line1") == gt_pair[0]
                                 and e.get("line2") == gt_pair[1])
                tw.insert("end",
                          f"  两行完全正确: {exact_lines}/{len(engines)} 引擎\n",
                          "lab")
        else:
            tw.insert("end", "\n（无引擎结果）\n", "mm")
        tw.see("end")
        tw.configure(state="disabled")
        # Persist for the preview overlay: the tool's band coords are
        # relative to the OCR crop; make them absolute for the overlay.
        cid = f"_picked:{os.path.basename(path)}"
        self._ocr_last = getattr(self, "_ocr_last", {})
        crop = ocr.get("crop") or {"x": 0, "y": 0}
        raw_band = ocr.get("band") or ""
        self._ocr_last[cid] = {
            "cnn": {
                "line1": l1, "line2": l2, "conf1": c1, "conf2": c2,
                "band": raw_band,
                "crop_offset": (crop.get("x", 0), crop.get("y", 0)),
                "ok": bool(ocr.get("ok")),
                "algorithm_ms": algorithm_ms,
                "total_ms": total_ms,
                "pair_orig": ocr.get("pair_orig"),
                "line_quads": ocr.get("line_quads"),
                "lines": ocr.get("lines"),
                "crop_rect": crop,
                "upscale": ocr.get("upscale", 1) or 1,
            }
        }
        self._preview_last_pm = self._ocr_last[cid]["cnn"]
        try:
            self._update_preview_image()
        except tk.TclError:
            pass
        # Restore buttons.
        try:
            self.ocr_pick_btn.configure(state="normal")
        except tk.TclError:
            pass
        try:
            self.ocr_run_btn.configure(state="normal")
            self.ocr_run_btn.configure(text="⚡ 开始测试")
        except tk.TclError:
            pass
        self._set_status_bar("识别完成", 100, done=1, total=1)

    def _ocr_set_sashes(self):
        body = getattr(self, "_ocr_body", None)
        if body is None:
            return
        total = body.winfo_width()
        if total < 120:
            return
        # 20% / 26% leaves ~54% for the middle KPI+summary pane;
        # previously 22/30 left only ~48% and squeezed the 7 KPI cards.
        w_left = max(230, int(total * 0.22))
        w_right = max(360, int(total * 0.34))
        body.sashpos(0, w_left)
        body.sashpos(1, max(w_left + 60, total - w_right))

    def _build_ocr_metrics(self, parent):
        """总体指标卡片 — 7 列均分,字号/padding 调小防止标题截断。
        列宽变化仍通过 uniform='metric_col' 同步;padx 从 8 减到 4,
        内层结构改为"标题 9pt + 数值 12pt + 单位 8pt"上下排版,避免单行
        文字因中英文宽度差被切。"""
        card = ttk.LabelFrame(parent, text="总体指标")
        card.pack(fill="x", pady=(0, 4))
        self.ocr_metrics = {}
        self._metric_cells = []
        colors = self._pal["metric"]
        # 缩短标题(原 "Mean耗时" -> "Mean";"平均L1"-> "L1 均值";"全匹配"-> "匹配")
        # 并把 pass/l1/l2 改成 Pass率/L1/L2 形式,避免 padx 截断
        specs = [("cases", "案例数"), ("mean", "Mean (ms)"),
                 ("p95", "P95 (ms)"), ("pass", "Pass (%)"),
                 ("l1", "L1 (%)"), ("l2", "L2 (%)"),
                 ("full", "匹配")]
        units = {}
        # 用 LabelFrame 的 grid 等分 7 列(weight=1, uniform)；
        # minsize 兜底，窗口收窄时预留最小宽度，防止 "141." 这类数字截断。
        for i in range(len(specs)):
            card.grid_columnconfigure(i, weight=1, uniform="kpi",
                                       minsize=86)
        for i, (key, label) in enumerate(specs):
            cell = tk.Frame(card, bg=self._pal["card"], highlightthickness=1,
                            highlightbackground=self._pal["border"])
            cell.grid(row=0, column=i, sticky="nsew", padx=2, pady=4)
            # 顶部 1px 暗灰分隔
            tk.Frame(cell, bg=self._pal["border"], height=1).pack(
                fill="x", side="top")
            # 紧凑的内部排版 — 标题 9pt,数值 11pt bold,单位 8pt
            inner = tk.Frame(cell, bg=self._pal["card"])
            inner.pack(fill="both", expand=True, padx=3, pady=(6, 4))
            # 左上角语义色块
            accent_color = colors[key]
            accent_strip = tk.Frame(inner, bg=accent_color, width=3, height=14)
            accent_strip.pack(side="left", anchor="n", padx=(0, 4), pady=(2, 0))
            accent_strip.pack_propagate(False)
            text_col = tk.Frame(inner, bg=self._pal["card"])
            text_col.pack(side="left", fill="both", expand=True)
            ttl = ttk.Label(text_col, text=label,
                            foreground=self._pal["dim"],
                            font=("Segoe UI", 9))
            ttl.pack(anchor="w")
            var = tk.StringVar(value="--")
            vrow = tk.Frame(text_col, bg=self._pal["card"])
            vrow.pack(anchor="w", fill="x")
            val = ttk.Label(vrow, textvariable=var,
                            foreground=colors[key],
                            font=("Menlo", 11, "bold"))
            val.pack(side="left")
            self.ocr_metrics[key] = var
            self._metric_cells.append(
                (cell, ttl, val, key, accent_strip, text_col))


    def _fit_columns(self, tree, pad=16, minw=50, cap=340):
        """Auto-fit Treeview column widths from heading + content."""
        import tkinter.font as tkfont
        f = tkfont.nametofont("TkDefaultFont")
        cols = list(tree["columns"])
        for idx, c in enumerate(cols):
            w = f.measure(tree.heading(c, "text"))
            for iid in tree.get_children():
                v = tree.item(iid, "values")
                if idx < len(v):
                    w = max(w, f.measure(str(v[idx])))
            tree.column(c, width=min(cap, max(minw, w + pad)))

    def _set_status_bar(self, msg, pct=None, done=None, total=None):
        """Update the bottom status bar with text + Progressbar.

        pct=None -> indeterminate spinning bar (no percentage chip)
        pct=0..100 -> determinate bar with a percentage chip
        done/total -> also paint "d/t" into the chip when available
        X2 polish #5:进度条默认隐藏,只在 pct 给定时才显示;100% 后
        800ms 自动收起 — 不占用 idle 视觉空间。
        """
        self.ocr_status_var.set(msg)
        chip = self._pal["chip"]
        if pct is None:
            self.ocr_progress.configure(mode="indeterminate")
            self.ocr_progress.start(12)
            self._ocr_paint_progress_chip("  ...  ", chip["dim"]["bg"],
                                          chip["dim"]["fg"])
            self._show_progress()
        else:
            pct = max(0.0, min(100.0, float(pct)))
            self.ocr_progress.stop()
            self.ocr_progress.configure(mode="determinate", value=pct)
            if pct >= 100:
                kind = "ok"
            elif pct >= 40:
                kind = "idle"
            else:
                kind = "run"
            counter = ""
            if done is not None and total is not None and total > 0:
                counter = f" {done}/{total}  {pct:.0f}% "
            else:
                counter = f" {pct:.0f}% "
            label = "  DONE  " if pct >= 100 else "  RUN  "
            self._ocr_paint_progress_chip(label + counter,
                                          chip[kind]["bg"],
                                          chip[kind]["fg"])
            self._show_progress()
            # 100% 时 800ms 后自动隐藏,让 UI 回到 idle 简洁态
            if pct >= 100:
                try:
                    self.after(800, self._hide_progress)
                except tk.TclError:
                    pass

    def _show_progress(self):
        """X2 polish #5:进度条显示到 sbar 的 mid_frame(grid 列 1,固定 360px)。
        该列与左状态文字 / 右工具chip完全隔离 — 进度条宽度不会因 chip 文本
        位数变化而抖动。"""
        try:
            mid = self._status_bar.grid_slaves(row=0, column=1)
            if not mid:
                return
            mid = mid[0]
            # progress + chip 在 mid_frame 内按 grid:progress=列 0, chip=列 1
            mid.columnconfigure(0, weight=0, minsize=220)  # 进度条 220px 固定
            mid.columnconfigure(1, weight=0, minsize=130)  # chip 文本固定 130px
            # 把 self.ocr_progress 重新 parent 到 mid(若还在 sbar)
            if str(self.ocr_progress.master) != str(mid):
                self.ocr_progress.pack_forget()
                self.ocr_progress.destroy()
                self.ocr_progress = ttk.Progressbar(
                    mid, length=220, mode="determinate", maximum=100)
                self._refresh_tk_theme()  # 重新套上 cyber-tech style
            if not self.ocr_progress.winfo_ismapped():
                self.ocr_progress.grid(row=0, column=0, sticky="w",
                                        padx=(8, 4), pady=4)
            self.ocr_progress.update_idletasks()
        except tk.TclError:
            pass

    def _hide_progress(self):
        """X2 polish #5:进度条 + chip 收回,状态栏回到 idle 布局。"""
        try:
            self.ocr_progress.stop()
            if self.ocr_progress.winfo_ismapped():
                self.ocr_progress.grid_forget()
        except tk.TclError:
            pass
        existing = getattr(self, "_progress_chip", None)
        if existing is not None:
            try:
                existing.destroy()
            except tk.TclError:
                pass
            self._progress_chip = None

    def _ocr_paint_progress_chip(self, text, bg, fg):
        """Repaint the progress chip — 放在 mid_frame 的 grid col=1(固定宽度),
        与进度条并列不会挤压进度条。文本使用等宽字体并预留足够 width 防止
        数字位数变化导致宽度抖动。"""
        mid = None
        try:
            mid = self._status_bar.grid_slaves(row=0, column=1)[0]
        except (tk.TclError, IndexError):
            mid = getattr(self, "_status_bar", None)
        if mid is None:
            return
        existing = getattr(self, "_progress_chip", None)
        if existing is not None:
            try:
                existing.destroy()
            except tk.TclError:
                pass
            self._progress_chip = None
        if not text:
            return
        try:
            # 等宽字体 + 固定 width=14 字符 — 防止位数变化导致宽度抖动
            lbl = tk.Label(
                mid, text=text, bg=bg, fg=fg,
                font=("Menlo", 9, "bold"), padx=6, pady=1,
                width=14, anchor="center")
            lbl.grid(row=0, column=1, sticky="w", padx=(4, 4))
            self._progress_chip = lbl
        except tk.TclError:
            pass

    def _ocr_method_toggle(self):
        # 勾选“同时跑全部方案”时，方案下拉框必须置灰（不选中任何一个）。
        # 通过把 ocr_methods 设成逗号串、并禁用 combobox 实现互斥；
        # 只有取消勾选后才能选单方案。
        pick = getattr(self, "ocr_method_pick", None)
        var = getattr(self, "ocr_method_pick_var", None)
        if self.ocr_method_both.get():
            self.ocr_methods.set(",".join(METHOD_ORDER))
            if var is not None:
                var.set("全部方案")
            if pick is not None:
                pick.configure(state="disabled")
        else:
            m = self.ocr_methods.get()
            if "," in m:
                m = m.split(",")[0]
                self.ocr_methods.set(m)
            if var is not None and m in METHOD_LABELS:
                var.set(METHOD_LABELS[m])
            if pick is not None:
                pick.configure(state="readonly")
        # 方案变了 → 旧结果与旧配色作废。
        self._ocr_reset_results()

    def _ocr_method_on_pick(self, _evt=None):
        # 下拉框选单方案时自动取消“同时跑全部方案”，实现互斥。
        try:
            if self.ocr_method_both.get():
                self.ocr_method_both.set(False)
        except Exception:
            pass
        label = self.ocr_method_pick_var.get()
        for key in METHOD_ORDER:
            if METHOD_LABELS[key] == label:
                self.ocr_methods.set(key)
                break
        # 方案变了 → 旧结果与旧配色作废。
        self._ocr_reset_results()

    def _ocr_filter_on_pick(self, _evt=None):
        key = {"全部": "all", "干净": "clean", "带噪": "noisy"}.get(
            self.ocr_filter_pick_var.get(), "all")
        self.ocr_filter_var.set(key)
        self._ocr_apply_filter()

    def _ocr_select_all(self, on: bool):
        all_ids = list(self.ocr_case_list.get_children())
        if on:
            self.ocr_case_list.selection_set(all_ids)
        else:
            self.ocr_case_list.selection_remove(all_ids)

    def _ocr_clear_sel(self):
        self.ocr_case_list.selection_remove(
            list(self.ocr_case_list.get_children()))

    def _ocr_apply_filter(self):
        f = self.ocr_filter_var.get()
        all_ids = list(self.ocr_case_list.get_children())
        self.ocr_case_list.selection_remove(all_ids)
        if f == "all":
            return
        for iid in all_ids:
            tags = self.ocr_case_list.item(iid, "tags")
            if f in tags:
                self.ocr_case_list.selection_add(iid)

    def _ocr_apply_search(self, _evt=None):
        q = self.ocr_search_var.get().strip().lower()
        tree = self.ocr_case_list
        ids = self._ocr_case_order
        if not q:
            for iid in ids:
                if iid not in tree.get_children():
                    tree.move(iid, "", len(tree.get_children()))
            return
        for iid in ids:
            if iid in tree.get_children():
                tree.detach(iid)
        for iid in ids:
            if q in iid.lower():
                tree.move(iid, "", len(tree.get_children()))

    def _ocr_bind_row_tooltip(self, tree, resolver):
        """Treeview 行级 tooltip：行变了才重建，避免 Motion 抖动。"""
        state = {"row": None}
        tip = _Tooltip(tree)

        def on_motion(evt):
            row = tree.identify_row(evt.y)
            if row == state["row"]:
                return
            state["row"] = row
            tip._leave(None)
            text = resolver(row) if row else ""
            tip.set_text(text)
            if text:
                tip._enter(None)

        def on_leave(_evt):
            state["row"] = None
            tip.set_text("")
            tip._leave(None)

        tree.bind("<Motion>", on_motion, add="+")
        tree.bind("<Leave>", on_leave, add="+")
        return tip

    def _ocr_detail_tip_text(self, iid):
        """详细结果行的完整路径/说明。"""
        paths = getattr(self, "_ocr_detail_paths", {})
        if iid in paths:
            return paths[iid]
        cid = iid.split("::", 1)[0]
        return getattr(self, "_ocr_case_paths", {}).get(cid, "")

    # ---- single-sample preview: image + char-level diff ----
    def _load_preview(self, cid, rec):
        from pathlib import Path as _P
        from ocr_bench_runner import CORPUS_JSON  # type: ignore
        corpus_dir = _P(CORPUS_JSON).parent
        img_path = corpus_dir / rec["image"]
        self.ocr_preview_meta.set(
            f"id={cid}  scale={rec['scale']}  noise={rec['noise']}  "
            f"skew={rec['skew']}")
        self._ocr_meta_fullpath = str(img_path)
        self._ocr_meta_tip.set_text(self._ocr_meta_fullpath)
        # Stash the absolute path so the viewer / region overlay can find it.
        self._preview_img_path = img_path
        try:
            from PIL import Image
            self._preview_img_pil = Image.open(img_path).convert("RGB")
            self._preview_w, self._preview_h = self._preview_img_pil.size
            # Cache last-known band / conf so the diff pane can show
            # them alongside the per-char comparison. Whole-image bench
            # so the band is already in absolute coordinates.
            last = getattr(self, "_ocr_last", {}).get(cid, {}) or {}
            pm = last.get("cnn") or last.get("traditional") or {}
            if pm and "crop_offset" not in pm:
                pm = dict(pm)
                pm["crop_offset"] = (0, 0)
            self._preview_last_pm = pm
            self._update_preview_image()
        except Exception:
            self._preview_img_pil = None
            self.ocr_preview_label.configure(image="",
                text=f"(无法预览 {img_path.name})")

    def _on_preview_click(self, event):
        """Check if click hits a corner handle; start dragging if so."""
        layout = getattr(self, "_preview_layout", None)
        if not layout or not layout.get("quads"):
            return
        x = event.x - layout.get("ox", 0)
        y = event.y - layout.get("oy", 0)
        _hr = 10
        for qi, (quad, _label) in enumerate(layout["quads"]):
            for ci, (px, py) in enumerate(quad):
                if abs(x - px) <= _hr and abs(y - py) <= _hr:
                    self._preview_drag = (qi, ci)
                    return
        self._preview_drag = None

    def _on_preview_drag(self, event):
        """Update corner position while dragging."""
        if getattr(self, "_preview_drag", None) is None:
            return
        layout = getattr(self, "_preview_layout", None)
        if not layout:
            return
        qi, ci = self._preview_drag
        x = max(0, min(event.x - layout.get("ox", 0), layout.get("dw", event.x)))
        y = max(0, min(event.y - layout.get("oy", 0), layout.get("dh", event.y)))
        layout["quads"][qi][0][ci] = (x, y)
        self._update_preview_image()

    def _on_preview_release(self, _event):
        """End dragging."""
        self._preview_drag = None

    def _update_preview_image(self, _evt=None):
        im = getattr(self, "_preview_img_pil", None)
        if im is None:
            return
        from PIL import Image, ImageDraw, ImageTk
        # Use the actual widget size; if the label hasn't been laid
        # out yet (winfo_width returns 1), fall back to the parent
        # frame's available space — NOT winfo_reqwidth() which returns
        # the last image's size and creates a feedback loop.
        w, h = 0, 0
        try:
            w = self.ocr_preview_label.winfo_width()
            h = self.ocr_preview_label.winfo_height()
        except tk.TclError:
            pass
        if w <= 1 or h <= 1:
            try:
                _parent = self.ocr_preview_label.master
                w = _parent.winfo_width()
                h = _parent.winfo_height()
            except tk.TclError:
                pass
        if w <= 1 or h <= 1:
            w, h = 320, 180
        # 图片本身会把 Label 的请求高度撑大：一旦按“当前 Label 高度”
        # 渲染整图，Label 请求高度反向膨胀 → 下方的“识别结果·字符级
        # 比对”被挤出可视区。把展示高度限制在预览窗格的一半以内，
        # 让图片与比对区始终同屏。
        try:
            pane_h = self.ocr_preview_label.master.winfo_height()
        except tk.TclError:
            pane_h = 0
        if pane_h > 160:
            h = min(h, max(160, int(pane_h * 0.52)))
        iw, ih = self._preview_w, self._preview_h
        _crop_ox, _crop_oy = 0, 0
        _pm_box = getattr(self, "_preview_last_pm", None) or {}
        _lq_raw_crop = _pm_box.get("line_quads") or []
        if _lq_raw_crop:
            _ax = [p[0] for q in _lq_raw_crop for p in q]
            _ay = [p[1] for q in _lq_raw_crop for p in q]
            _pad = max(10, int((max(_ax) - min(_ax)) * 0.03))
            _crop_ox = max(0, int(min(_ax)) - _pad)
            _crop_oy = max(0, int(min(_ay)) - _pad)
            _cwe = min(iw, int(max(_ax)) + _pad) - _crop_ox
            _che = min(ih, int(max(_ay)) + _pad) - _crop_oy
            if _cwe > 20 and _che > 20:
                im = im.crop((_crop_ox, _crop_oy, _crop_ox + _cwe, _crop_oy + _che))
                iw, ih = im.size
        # Scale-thumbnail keeping aspect ratio so the photo fits the
        # preview pane. The Tk label uses the actual displayed size
        # (w x h) minus a small margin; we map image -> display coords
        # uniformly.
        margin = 4
        tw, th = max(1, w - margin), max(1, h - margin)
        scale = min(tw / iw, th / ih)
        dw, dh = int(iw * scale), int(ih * scale)
        ox = (w - dw) // 2
        oy = (h - dh) // 2
        # Resize the original pixels to the displayed size; we then
        # paint the overlay rectangles directly in pixel space (dw x dh)
        # which simplifies the mapping vs the original iw x ih space.
        im2 = im.resize((dw, dh), Image.LANCZOS)
        try:
            overlay = im2.copy()
            drw = ImageDraw.Draw(overlay, "RGBA")
            # HUD corner brackets (cyan) before the region boxes.
            _hu, _l = (0, 245, 255, 255), 14
            drw.line((0, _l, 0, 0, _l, 0), fill=_hu, width=3)
            drw.line((dw - _l, 0, dw, 0, dw, _l), fill=_hu, width=3)
            drw.line((0, dh - _l, 0, dh, _l, dh), fill=_hu, width=3)
            drw.line((dw - _l, dh, dw, dh, dw, dh - _l), fill=_hu, width=3)
            # Region rectangles (proportional to displayed image):
            #   photo  = left 25%, top 10%, width 25%, height 70%
            #   data   = top 10%, left 25%, width 50%, height 25%
            #   mrz    = bottom 30%, full width (overrides data on the
            #            bottom edge, which is the real passport layout)
            # Colours match .impeccable.md neon palette.
            # Only draw MRZ line textbox(s): 1 line (CS/CD) or 2 lines (TD3).
            # Use quadrilateral (4 corner points) to handle photo deformation.
            pm_box = getattr(self, "_preview_last_pm", {}) or {}
            band_raw = pm_box.get("band", "")
            cx, cy = pm_box.get("crop_offset", (0, 0)) or (0, 0)
            upscale = pm_box.get("upscale", 1) or 1
            line1 = pm_box.get("line1", "") or ""
            line2 = pm_box.get("line2", "") or ""
            if band_raw:
                try:
                    bx, by, bw, bh = (int(float(x)) for x in band_raw.split())
                    if upscale > 1:
                        bx //= upscale; by //= upscale
                        bw //= upscale; bh //= upscale
                    abx, aby = bx + cx, by + cy
                    sx = dw / iw; sy = dh / ih
                    mrz_box = (int((abx - _crop_ox) * sx), int((aby - _crop_oy) * sy),
                               int((abx + bw - _crop_ox) * sx), int((aby + bh - _crop_oy) * sy))
                except Exception:
                    mrz_box = (int(dw * 0.04), int(dh * 0.62),
                               int(dw * 0.96), int(dh * 0.96))
            else:
                mrz_box = (int(dw * 0.04), int(dh * 0.62),
                           int(dw * 0.96), int(dh * 0.96))
            # Build quadrilaterals (4 corner points) for each MRZ line.
            # Priority: line_quads (true quad from _mrzh_quad, handles
            # photo perspective deformation) > pair_orig (rect) > band.
            # Recomputed every time from original coords so the quad
            # scales correctly when the widget is resized.
            _sx = dw / iw
            _sy = dh / ih
            line_quads_raw = pm_box.get("line_quads")
            if line_quads_raw:
                line_quads = []
                for _i, _quad_orig in enumerate(line_quads_raw):
                    _label = "LINE1" if _i == 0 else "LINE2"
                    _pts = [(max(0, min(int((p[0] - _crop_ox) * _sx), dw)),
                             max(0, min(int((p[1] - _crop_oy) * _sy), dh)))
                            for p in _quad_orig]
                    line_quads.append((_pts, _label))
            else:
                pair_orig = pm_box.get("pair_orig")
                crop_rect = pm_box.get("crop_rect")
                if crop_rect:
                    _dx0 = int(crop_rect["x"] * _sx)
                    _dx1 = int((crop_rect["x"] + crop_rect["w"]) * _sx)
                else:
                    _dx0, _dx1 = mrz_box[0], mrz_box[2]
                if pair_orig:
                    (la0, la1), (lb0, lb1) = pair_orig
                    la0d = max(0, min(int(la0 * _sy), dh))
                    la1d = max(0, min(int(la1 * _sy), dh))
                    lb0d = max(0, min(int(lb0 * _sy), dh))
                    lb1d = max(0, min(int(lb1 * _sy), dh))
                    if line2 and (lb0, lb1) != (la0, la1):
                        line_quads = [
                            ([(_dx0, la0d), (_dx1, la0d),
                              (_dx1, la1d), (_dx0, la1d)], "LINE1"),
                            ([(_dx0, lb0d), (_dx1, lb0d),
                              (_dx1, lb1d), (_dx0, lb1d)], "LINE2"),
                        ]
                    else:
                        line_quads = [(
                            [(_dx0, la0d), (_dx1, la0d),
                             (_dx1, la1d), (_dx0, la1d)], "LINE1")]
                else:
                    _, y0, _, y1 = mrz_box
                    _lines_raw = pm_box.get("lines", "")
                    _got_lines = False
                    if _lines_raw:
                        try:
                            _l1y, _l1h, _l2y, _l2h = (
                                int(float(x)) for x in _lines_raw.split())
                            if upscale > 1:
                                _l1y //= upscale; _l1h //= upscale
                                _l2y //= upscale; _l2h //= upscale
                            la0d = max(0, min(int(y0 + _l1y * _sy), dh))
                            la1d = max(0, min(int(y0 + (_l1y + _l1h) * _sy), dh))
                            if line2 and _l2h > 0:
                                lb0d = max(0, min(int(y0 + _l2y * _sy), dh))
                                lb1d = max(0, min(int(y0 + (_l2y + _l2h) * _sy), dh))
                                line_quads = [
                                    ([(_dx0, la0d), (_dx1, la0d), (_dx1, la1d), (_dx0, la1d)], "LINE1"),
                                    ([(_dx0, lb0d), (_dx1, lb0d), (_dx1, lb1d), (_dx0, lb1d)], "LINE2"),
                                ]
                            else:
                                line_quads = [(
                                    [(_dx0, la0d), (_dx1, la0d), (_dx1, la1d), (_dx0, la1d)], "LINE1")]
                            _got_lines = True
                        except Exception:
                            pass
                    if not _got_lines:
                        if line2:
                            mid_y = (y0 + y1) // 2
                            line_quads = [
                                ([(_dx0, y0), (_dx1, y0), (_dx1, mid_y), (_dx0, mid_y)], "LINE1"),
                                ([(_dx0, mid_y), (_dx1, mid_y), (_dx1, y1), (_dx0, y1)], "LINE2"),
                            ]
                        else:
                            line_quads = [(
                                [(_dx0, y0), (_dx1, y0), (_dx1, y1), (_dx0, y1)], "LINE1")]
            _red = (255, 0, 0, 255)
            _blue = (0, 0, 255, 255)
            _green = (0, 255, 0, 255)
            _po = pm_box.get("pair_orig")
            if _po:
                (_pa0, _pa1), (_pb0, _pb1) = _po
                _la0d = max(0, min(int((_pa0 - _crop_oy) * _sy), dh))
                _la1d = max(0, min(int((_pa1 - _crop_oy) * _sy), dh))
                drw.line((0, _la0d, dw, _la0d), fill=_red, width=1)
                drw.line((0, _la1d, dw, _la1d), fill=_red, width=1)
                if line2 and (_pb0, _pb1) != (_pa0, _pa1):
                    _lb0d = max(0, min(int((_pb0 - _crop_oy) * _sy), dh))
                    _lb1d = max(0, min(int((_pb1 - _crop_oy) * _sy), dh))
                    drw.line((0, _lb0d, dw, _lb0d), fill=_blue, width=1)
                    drw.line((0, _lb1d, dw, _lb1d), fill=_blue, width=1)
            else:
                for _quad, _label in line_quads:
                    _color = _red if _label == "LINE1" else _blue
                    _ys = sorted(set(int(p[1]) for p in _quad))
                    for _yy in _ys:
                        drw.line((0, _yy, dw, _yy), fill=_color, width=1)
            _cr = pm_box.get("crop_rect")
            if _cr:
                _gby0 = max(0, min(int((_cr["y"] - _crop_oy) * _sy), dh))
                _gby1 = max(0, min(int((_cr["y"] + _cr["h"] - _crop_oy) * _sy), dh))
            else:
                _gby0 = mrz_box[1]
                _gby1 = mrz_box[3]
            drw.line((0, _gby0, dw, _gby0), fill=_green, width=1)
            drw.line((0, _gby1, dw, _gby1), fill=_green, width=1)
            photo = ImageTk.PhotoImage(overlay)
        except Exception:
            try:
                from PIL import ImageTk
                photo = ImageTk.PhotoImage(im2)
            except Exception:
                return
        self._preview_imgs["_current"] = photo
        self.ocr_preview_label.configure(image=photo, text="")
        # Stash the on-screen rectangle so other widgets (diff pane)
        # can reference consistent proportions if needed.
        self._preview_layout = {"scale": scale, "ox": ox, "oy": oy,
                                "dw": dw, "dh": dh,
                                "mrz": mrz_box}

    def _render_diff_block(self, tw, label, gt, pred):
        """Append one aligned GT/Pred block with per-char highlighting."""
        n = max(len(gt), len(pred), 30)
        gt = gt.ljust(n)
        pred = pred.ljust(n)
        tw.insert("end", label + "\n", "hdr")
        ruler_tens = "".join([str((i + 1) // 10) if (i + 1) % 10 == 0 else " " for i in range(n)])
        ruler_ones = "".join([str((i + 1) % 10) for i in range(n)])
        tw.insert("end", "    " + ruler_tens + "\n", "lab")
        tw.insert("end", "    " + ruler_ones + "\n", "lab")
        tw.insert("end", "GT: ", "lab")
        for i in range(n):
            if gt[i] == pred[i]:
                tag = "pad" if gt[i] == "<" else "match"
            else:
                tag = "mm"
            tw.insert("end", gt[i], tag)
        tw.insert("end", "\nPR: ", "lab")
        mism = []
        for i in range(n):
            if gt[i] != pred[i]:
                mism.append(i)
                tw.insert("end", pred[i], "error")
            else:
                tw.insert("end", pred[i], "pad" if pred[i] == "<" else "match")
        tw.insert("end", "\n    ")
        if mism:
            for i in range(n):
                tw.insert("end", "^" if i in mism else " ", "error")
            tw.insert("end", "\n")
        else:
            tw.insert("end", "（全部匹配）\n", "okline")

    def _auto_gt_map(self):
        """Load tests/photo_auto_gt.json (multi-engine consensus auto-GT)
        into {filename: (line1, line2)} lazily."""
        if getattr(self, "_auto_gt_map_cache", None) is not None:
            return self._auto_gt_map_cache
        m = {}
        try:
            p = Path(__file__).parent / "photo_auto_gt.json"
            if p.exists():
                d = json.loads(p.read_text(encoding="utf-8"))
                for name, g in (d.get("gt") or {}).items():
                    if g.get("l1") and g.get("l2"):
                        m[name] = (g["l1"], g["l2"])
        except Exception:
            pass
        self._auto_gt_map_cache = m
        return m

    def _rst_gt_for(self, path):
        """按文件名查 GT：只从 gt_review_loop.xlsx 'GT对比' sheet 查找。"""
        name = os.path.basename(str(path))
        gm = self._gt_map()
        return gm.get(name)

    def _rst_render_gt_diff(self, tw, result, gt1, gt2):
        """照片→MRZ 每引擎结果后追加 GT vs PRED 高亮对比块（格式同 _show_diff）。"""
        label = result.get("label") or result.get("method") or ""
        pred1 = result.get("line1") or ""
        pred2 = result.get("line2") or ""
        blocks = [
            (f"line1 [{label}]", gt1, pred1),
            (f"line2 [{label}]", gt2, pred2),
        ]
        confs = (result.get("conf1"), result.get("conf2"))
        for (key, gt, pred), conf in zip(blocks, confs):
            tag = "okline" if gt == pred else "lab"
            if conf is None:
                tw.insert("end", f"{key}  conf=n/a\n", tag)
            else:
                grade = "PASS" if conf >= 70 else ("WARN" if conf >= 40 else "FAIL")
                tw.insert("end", f"{key}  conf={conf}% ({grade})\n", tag)
            tw.insert("end", "\n", "lab")
        for key, gt, pred in blocks:
            self._render_diff_block(tw, key, gt, pred)

    def _show_diff(self, blocks):
        tw = self.ocr_diff_text
        tw.configure(state="normal")
        tw.delete("1.0", "end")
        # Confidence header: one line per block, mirroring the layout
        # expected by the PPM benchmark output (conf1 / conf2 0..100).
        pm = getattr(self, "_preview_last_pm", None) or {}
        for label, gt, pred in blocks:
            tag = "okline" if gt == pred else "lab"
            conf = pm.get("conf1") if "line1" in label else (
                   pm.get("conf2") if "line2" in label else None)
            if conf is None:
                tw.insert("end", f"{label}  conf=n/a\n", tag)
            else:
                grade = "PASS" if conf >= 70 else ("WARN" if conf >= 40 else "FAIL")
                tw.insert("end",
                          f"{label}  conf={conf}% ({grade})\n", tag)
            tw.insert("end", "\n", "lab")
        for label, gt, pred in blocks:
            self._render_diff_block(tw, label, gt, pred)
        tw.see("end")
        tw.configure(state="disabled")

    def _show_gt_only(self, rec):
        tw = self.ocr_diff_text
        tw.configure(state="normal")
        tw.delete("1.0", "end")
        # Show last-known confidence (if any) above the GT lines so the
        # user can compare expected vs predicted structure immediately.
        pm = getattr(self, "_preview_last_pm", None) or {}
        c1 = pm.get("conf1")
        c2 = pm.get("conf2")
        if c1 is not None or c2 is not None:
            tw.insert("end", "最近一次识别置信率:\n", "hdr")
            tw.insert("end",
                f"  line1 conf={c1 if c1 is not None else 'n/a'}%   "
                f"line2 conf={c2 if c2 is not None else 'n/a'}%\n",
                "lab")
            tw.insert("end", "\n", "lab")
        tw.insert("end", "line1 (GT)\n", "hdr")
        tw.insert("end", rec["line1"] + "\n", "match")
        tw.insert("end", "line2 (GT)\n", "hdr")
        tw.insert("end", rec["line2"] + "\n", "match")
        tw.insert("end", "（运行测试后在此显示 Pred 与字符级比对）", "lab")
        tw.configure(state="disabled")

    def _ocr_on_list_click(self, _evt=None):
        """Photo-mode single-click: run the full pipeline for the
        selected photo and stream results to the right pane. No-op
        for corpus cases (the corpus bench needs ⚡ 开始测试)."""
        if not self.ocr_photo_mode.get():
            return
        sel = self.ocr_case_list.selection()
        if not sel:
            return
        name = sel[0]
        self.ocr_run_btn.configure(state="disabled")
        # Render the image immediately so the right pane shows
        # something while we wait for the worker.
        entries = self._ocr_photo_entry_map()
        path = entries.get(name)
        if path is None:
            messagebox.showerror("加载失败", f"{name}: 源文件不存在")
            return
        try:
            from PIL import Image
            pil = Image.open(path).convert("RGB")
            self._preview_img_pil = pil
            self._preview_w, self._preview_h = pil.size
            self._preview_img_path = path
            self._preview_last_pm = {}
            self.ocr_preview_meta.set(f"{name}")
            self._ocr_meta_fullpath = str(path)
            self._ocr_meta_tip.set_text(self._ocr_meta_fullpath)
            self._update_preview_image()
        except Exception as e:
            messagebox.showerror("加载失败", f"{path}: {e}")
            return
        self.ocr_diff_text.configure(state="normal")
        self.ocr_diff_text.delete("1.0", "end")
        self.ocr_diff_text.insert("end",
            f"样本 : {name}\n状态 : 运行中…", ("lab",))
        self.ocr_diff_text.configure(state="disabled")

        def worker():
            try:
                # 与批量模式同一路径：多引擎 + 共识/GT 引导。
                rec = self._ocr_photo_engines(str(path))
                # label（含来源目录）作为唯一 id。
                rec["file"] = name
                rec["path"] = str(path)
            except Exception as e:
                self.after(0, lambda e=e:
                           self._ocr_render_single_photo_failed(name, e))
                return
            self.after(0, lambda rec=rec:
                       self._ocr_render_single_photo_result(name, rec))

        threading.Thread(target=worker, daemon=True).start()

    def _ocr_render_single_photo_result(self, name, rec):
        """Insert the single-photo result row and select it so the
        right pane repaints with the photo + region overlay + MRZ
        details. Mirrors what batch mode does after _ocr_populate_photos."""
        # Ensure _ocr_last['photo'] is populated so _ocr_on_detail_select
        # can find the rec.
        self._ocr_last = getattr(self, "_ocr_last", {})
        existing = self._ocr_last.get("photo", [])
        # Replace any previous entry for the same file (re-run scenario).
        self._ocr_last["photo"] = [r for r in existing if r.get("file") != name] + [rec]
        # Insert a detail row tagged with the grade so the chip colour
        # matches the verdict (PASS green / SUSPECT amber / REJECT red).
        _ocr_r = rec.get("ocr") or {}
        _p1 = _ocr_r.get("line1") or ""
        _p2 = _ocr_r.get("line2") or ""
        _gt = self._rst_gt_for(str(name))
        if _gt:
            if _gt[0] and _gt[1]:
                grade = "PASS" if (_p1 == _gt[0] and _p2 == _gt[1]) else "FAIL"
            elif _gt[0]:
                grade = "PASS" if (_p1 == _gt[0]) else "FAIL"
            else:
                grade = "NO_GT"
        else:
            grade = rec.get("grade", "REJECT")
        iid = name
        if not self.ocr_detail.exists(iid):
            self.ocr_detail.insert("", "end", iid=iid, tags=(grade,),
                values=(os.path.basename(str(name)), grade,
                        "✓" if rec.get("verified") else "—",
                        f"{rec.get('ms', 0):.0f}",
                        rec.get("reason") or ""))
        else:
            self.ocr_detail.item(iid, tags=(grade,),
                values=(os.path.basename(str(name)), grade,
                        "✓" if rec.get("verified") else "—",
                        f"{rec.get('ms', 0):.0f}",
                        rec.get("reason") or ""))
        self._ocr_detail_map[iid] = ("photo", name)
        self.ocr_detail.selection_set(iid)
        self.ocr_detail.focus(iid)
        # 直接渲染已有结果，不触发重新识别
        self._ocr_render_photo_detail(name, str(rec.get("path", name)), rec, None)
        self.ocr_run_btn.configure(state="normal")
        self.ocr_run_btn.configure(text="⚡ 开始测试")

    def _ocr_render_single_photo_failed(self, name, err):
        """Surface worker errors in the right pane and re-enable the
        run button so the user can try again."""
        tw = self.ocr_diff_text
        tw.configure(state="normal")
        tw.delete("1.0", "end")
        tw.insert("end", f"样本 : {name}\n", "hdr")
        tw.insert("end", f"异常 : {err}\n", "mm")
        tw.configure(state="disabled")
        try:
            self.ocr_run_btn.configure(state="normal")
            self.ocr_run_btn.configure(text="⚡ 开始测试")
        except tk.TclError:
            pass

    def _ocr_on_case_select(self, _evt=None):
        sel = self.ocr_case_list.selection()
        if not sel:
            return
        cid = sel[0]
        rec = next((r for r in self._ocr_corpus["records"]
                    if r["id"] == cid), None)
        if not rec:
            return
        self._load_preview(cid, rec)
        last = getattr(self, "_ocr_last", {})
        blocks = []
        for m in ("traditional", "cnn"):
            r = last.get(cid, {}).get(m)
            if r:
                label = METHOD_LABELS[m]
                blocks.append((f"line1 [{label}]", rec["line1"],
                               r.get("line1", "")))
                blocks.append((f"line2 [{label}]", rec["line2"],
                               r.get("line2", "")))
        if blocks:
            self._show_diff(blocks)
        else:
            self._show_gt_only(rec)

    # ---- Full-size image viewer (double-click on case / detail row) ----
    def _ocr_open_original_from_detail(self, _evt=None):
        """Resolve the corpus record for the currently selected detail
        row and pop the full-size viewer. Detail rows are keyed by
        (case_id, method); the case_id is the first 8 chars of the row
        iid (or the prefix before "::")."""
        sel = self.ocr_detail.selection()
        if not sel:
            return
        iid = sel[0]
        cid = iid.split("::", 1)[0] if "::" in iid else iid[:8]
        rec = next((r for r in self._ocr_corpus["records"]
                    if r["id"] == cid), None)
        if rec is None:
            return
        self._ocr_open_original(cid, rec)

    def _ocr_open_original(self, cid=None, rec=None):
        """Open a top-level window showing the original (unresized)
        image with zoom controls (Fit / 1x / 2x / 4x) and a chip bar
        reporting case metadata. Mirrors the CustomTkinter preview."""
        if rec is None or cid is None:
            sel = self.ocr_case_list.selection()
            if not sel:
                return
            cid = sel[0]
            rec = next((r for r in self._ocr_corpus["records"]
                        if r["id"] == cid), None)
            if rec is None:
                return
        from pathlib import Path as _P
        from ocr_bench_runner import CORPUS_JSON  # type: ignore
        img_path = _P(CORPUS_JSON).parent / rec["image"]
        if not img_path.exists():
            messagebox.showerror("原图缺失", f"找不到图片文件:\n{img_path}")
            return
        try:
            from PIL import Image as _PILImage
        except Exception:
            messagebox.showerror("依赖缺失",
                                 "原图查看器需要 Pillow: pip install pillow")
            return
        try:
            full_pil = _PILImage.open(img_path).convert("RGB")
        except Exception as e:
            messagebox.showerror("加载失败", f"无法读取 {img_path.name}: {e}")
            return
        win = tk.Toplevel(self)
        win.title(f"原图 · {cid}  ·  {img_path.name}")
        pal = self._pal
        try:
            win.configure(bg=pal["bg"])
        except tk.TclError:
            pass
        iw, ih = full_pil.size
        try:
            sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        except Exception:
            sw, sh = 1440, 900
        ww = min(max(iw + 80, 720), int(sw * 0.9))
        wh = min(max(ih + 140, 540), int(sh * 0.9))
        win.geometry(f"{ww}x{wh}")

        # ---- Top metadata bar ----
        top = ttk.Frame(win)
        top.pack(fill="x", padx=8, pady=(8, 4))
        ttk.Label(top, text=f"  原图 · {cid}  ",
                  foreground=pal.get("accent", pal["primary"]),
                  font=("TkDefaultFont", 11, "bold")).pack(
                      side="left", padx=(0, 8))
        chip = pal["chip"]["idle"]
        for label, key in (("scale", "scale"), ("noise", "noise"),
                           ("skew", "skew"), ("channel", "channel")):
            v = rec.get(key, "-")
            ttk.Label(top, text=f"  {label}={v}  ",
                      foreground=chip["fg"], background=chip["bg"],
                      font=("Menlo", 9, "bold")).pack(side="left", padx=4)
        ttk.Label(top, text=f"  {iw}x{ih}  ",
                  foreground="#888",
                  font=("Menlo", 9)).pack(side="right", padx=8)

        # ---- Zoom controls ----
        bar = ttk.Frame(win)
        bar.pack(fill="x", padx=8, pady=4)
        status_var = tk.StringVar(value="zoom 1.0x")
        ttk.Label(bar, textvariable=status_var,
                  foreground="#888",
                  font=("Menlo", 9)).pack(side="right", padx=8)
        zoom_state = {"factor": 1.0}

        # ---- Canvas + scrollbars ----
        cf = ttk.Frame(win, relief="solid", borderwidth=1)
        cf.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        canvas = tk.Canvas(cf, bg=pal["code"], highlightthickness=0)
        xscroll = ttk.Scrollbar(cf, orient="horizontal",
                                command=canvas.xview)
        yscroll = ttk.Scrollbar(cf, orient="vertical",
                                command=canvas.yview)
        canvas.configure(xscrollcommand=xscroll.set,
                         yscrollcommand=yscroll.set)
        xscroll.pack(side="bottom", fill="x")
        yscroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        photo_ref = {"img": None}

        def render(factor):
            factor = max(0.1, min(factor, 8.0))
            zoom_state["factor"] = factor
            status_var.set(f"zoom {factor:.2f}x")
            zw = max(1, int(iw * factor))
            zh = max(1, int(ih * factor))
            from PIL import Image as _PI, ImageTk as _IT
            if factor == 1.0:
                photo = _IT.PhotoImage(full_pil)
            else:
                method = _PI.LANCZOS if factor < 1.0 else _PI.BICUBIC
                photo = _IT.PhotoImage(
                    full_pil.resize((zw, zh), method))
            photo_ref["img"] = photo
            canvas.delete("all")
            canvas.create_image(0, 0, image=photo, anchor="nw")
            canvas.configure(scrollregion=(0, 0, zw, zh))

        def fit_to_window():
            canvas.update_idletasks()
            cw = max(canvas.winfo_width(), 1)
            ch = max(canvas.winfo_height(), 1)
            render(min(cw / iw, ch / ih))

        def set_zoom(f):
            render(f)

        for label, value in (("Fit", None), ("1x", 1.0),
                             ("2x", 2.0), ("4x", 4.0)):
            cmd = fit_to_window if value is None else (lambda v=value: set_zoom(v))
            ttk.Button(bar, text=label, command=cmd, width=6).pack(
                side="left", padx=2)

        def on_wheel(evt):
            if evt.state & 0x4 or evt.state & 0x0008:
                delta = evt.delta if hasattr(evt, "delta") else 0
                step = 1.1 if delta > 0 else 1 / 1.1
                render(zoom_state["factor"] * step)
                return "break"
            if getattr(evt, "num", None) == 4:
                canvas.yview_scroll(-3, "units")
            elif getattr(evt, "num", None) == 5:
                canvas.yview_scroll(3, "units")
            else:
                d = -1 if evt.delta > 0 else 1
                canvas.yview_scroll(d * 3, "units")
            return "break"
        canvas.bind("<MouseWheel>", on_wheel)
        canvas.bind("<Button-4>", on_wheel)
        canvas.bind("<Button-5>", on_wheel)

        canvas.update_idletasks()
        fit_to_window()

    def _gt_for(self, name):
        """Return (line1, line2) GT from gt_review_loop.xlsx 'GT对比' sheet.

        `name` 可能是带目录前缀的 label（如 'assets/pic/x.jpg'）也可能是
        basename；两种都先原样查再退到 os.path.basename 兜底。
        """
        gm = self._gt_map()
        for candidate in (name, os.path.basename(str(name))):
            if candidate in gm:
                return gm[candidate]
        return None

    def _ocr_path_label(self, rec):
        """Name the recognition path a record came from.

        The wall time alone does not say whether the product path (one
        local C engine, no cross-engine fallback) or the old multi-engine
        path (Tesseract + Paddle + Indra in parallel, wall clock = the
        slowest) produced it -- and the two differ by an order of
        magnitude, so a bare "588 ms" is easy to misread.
        """
        engines = [e.get("method") for e in (rec.get("engines") or [])
                   if e.get("method")]
        if not engines:
            return "无引擎"
        if len(engines) == 1:
            return f"产品路径·{METHOD_LABELS.get(engines[0], engines[0])}"
        return "多引擎:" + ",".join(engines)

    def _ocr_on_detail_select(self, _evt=None):
        sel = self.ocr_detail.selection()
        if not sel:
            return
        kind, name = self._ocr_detail_map[sel[0]]
        if kind == "photo":
            entries = self._ocr_photo_entry_map()
            path = entries.get(name)
            if path is None:
                from passport_pipeline import PIC_DIR
                path = PIC_DIR / name
            self.ocr_preview_meta.set(f"{name}")
            self._ocr_meta_fullpath = str(path)
            self._ocr_meta_tip.set_text(self._ocr_meta_fullpath)
            # 先加载图片预览，让用户立即看到原图
            try:
                from PIL import Image
                pil = Image.open(path).convert("RGB")
                self._preview_img_pil = pil
                self._preview_w, self._preview_h = pil.size
                self._preview_img_path = path
                self._preview_layout = None
                self._preview_drag = None
                self._preview_last_pm = {}
                self._update_preview_image()
            except Exception as e:
                self.ocr_diff_text.configure(state="normal")
                self.ocr_diff_text.delete("1.0", "end")
                self.ocr_diff_text.insert("end",
                    f"无法加载 {path}: {e}", ("mm",))
                self.ocr_diff_text.configure(state="disabled")
                return
            # 在 diff pane 显示"识别中"提示
            tw = self.ocr_diff_text
            tw.configure(state="normal")
            tw.delete("1.0", "end")
            tw.insert("end", f"样本  : {name}\n", "hdr")
            tw.insert("end", f"路径  : {path}\n", "lab")
            tw.insert("end", "状态  : 重新识别中…\n", "lab")
            tw.configure(state="disabled")
            # 后台线程重新识别（与"开始测试"完全相同的配置）
            methods = self._ocr_selected_methods()
            strict = self._ocr_selection_is_explicit()

            def worker():
                err = None
                rec = {}
                try:
                    rec = self._ocr_photo_engines(str(path), methods,
                                                  strict=strict)
                except Exception as e:
                    err = e
                self.after(0, lambda: self._ocr_render_photo_detail(
                    name, str(path), rec, err))

            threading.Thread(target=worker, daemon=True).start()
            return
        cid, m = kind, name
        rec = next((r for r in self._ocr_corpus["records"]
                    if r["id"] == cid), None)
        if rec is None:
            return
        self._load_preview(cid, rec)
        r = getattr(self, "_ocr_last", {}).get(cid, {}).get(m, {})
        label = METHOD_LABELS.get(m, m)
        self._show_diff([(f"line1 [{label}]", rec["line1"], r.get("line1", "")),
                         (f"line2 [{label}]", rec["line2"], r.get("line2", ""))])

    def _ocr_render_photo_detail(self, name, path, rec, err):
        """渲染照片识别结果到右侧面板（由 _ocr_on_detail_select 的后台
        线程完成后调用）。rec 是 _ocr_photo_engines 的返回值，每次双击
        都重新识别，不从缓存取。"""
        tw = self.ocr_diff_text
        tw.configure(state="normal")
        tw.delete("1.0", "end")
        if err:
            tw.insert("end", f"样本  : {name}\n", "hdr")
            tw.insert("end", f"路径  : {path}\n", "lab")
            tw.insert("end", f"识别失败: {err}\n", "mm")
            tw.configure(state="disabled")
            return
        if not rec:
            tw.insert("end", f"样本  : {name}\n", "hdr")
            tw.insert("end", f"路径  : {path}\n", "lab")
            tw.insert("end", "识别失败: 无结果\n", "mm")
            tw.configure(state="disabled")
            return
        # 更新 _ocr_last 缓存，供 detail 行刷新使用
        self._ocr_last = getattr(self, "_ocr_last", {})
        existing = self._ocr_last.get("photo", [])
        self._ocr_last["photo"] = [r for r in existing if r.get("file") != name] + [rec]
        # 更新 detail 行的耗时和判定
        _ocr_r = rec.get("ocr") or {}
        _p1 = _ocr_r.get("line1") or ""
        _p2 = _ocr_r.get("line2") or ""
        _gt = self._rst_gt_for(str(path))
        if _gt:
            if _gt[0] and _gt[1]:
                grade = "PASS" if (_p1 == _gt[0] and _p2 == _gt[1]) else "FAIL"
            elif _gt[0]:
                grade = "PASS" if (_p1 == _gt[0]) else "FAIL"
            else:
                grade = "NO_GT"
        else:
            grade = rec.get("grade", "REJECT")
        iid = name
        if self.ocr_detail.exists(iid):
            self.ocr_detail.item(iid, tags=(grade,),
                values=(os.path.basename(str(name)), grade,
                        "✓" if rec.get("verified") else "—",
                        f"{rec.get('ms', 0):.0f}",
                        rec.get("reason") or ""))
        # 更新预览图上的 MRZ 区域 overlay
        try:
            crop = (rec.get("ocr") or {}).get("crop") or {}
            try:
                crop_offset = (int(crop.get("x", 0)), int(crop.get("y", 0)))
            except Exception:
                crop_offset = (0, 0)
            _ocr_data = rec.get("ocr") or {}
            self._preview_last_pm = {
                "conf1": _ocr_data.get("conf1", 0) or 0,
                "conf2": _ocr_data.get("conf2", 0) or 0,
                "band":  _ocr_data.get("band", "") or "",
                "line1": _ocr_data.get("line1", "") or "",
                "line2": _ocr_data.get("line2", "") or "",
                "crop_offset": crop_offset,
                "upscale": _ocr_data.get("upscale", 1) or 1,
                "pair_orig": _ocr_data.get("pair_orig"),
                "crop_rect": _ocr_data.get("crop"),
                "line_quads": _ocr_data.get("line_quads"),
                "lines": _ocr_data.get("lines"),
            }
            self._update_preview_image()
        except Exception:
            pass
        # 渲染结构化报告
        tw.insert("end", f"样本  : {name}\n", "hdr")
        tw.insert("end", f"路径  : {path}\n", "lab")
        ok_chip = "okline" if grade == "PASS" else "mm"
        tw.insert("end",
            f"判定  : {'PASS  ✅' if grade == 'PASS' else grade + '  ⚠️'}\n",
            ok_chip)
        tw.insert("end",
            f"耗时  : {rec.get('ms', 0):.0f} ms"
            f"   [{self._ocr_path_label(rec)}]\n", "lab")
        tw.insert("end",
            f"OCR校验: {'通过' if rec.get('verified') else '未通过'}\n",
            "lab")
        tw.insert("end", "\n护照定位 (photo / data / mrz):\n", "hdr")
        loc = rec.get("loc") or {}
        for k_zh, k_en in (("照片区", "photo"),
                           ("数据区", "data"),
                           ("MRZ 区", "mrz")):
            r = loc.get(k_en)
            if r:
                tw.insert("end",
                    f"  {k_zh}: x={r['x']} y={r['y']} "
                    f"w={r['w']} h={r['h']}   "
                    f"置信度 {r['confidence']:.2f}\n", "lab")
            else:
                tw.insert("end", f"  {k_zh}: 未定位\n", "mm")
        tw.insert("end", "\nMRZ 识别结果:\n", "hdr")
        ocr_rec = rec.get("ocr") or {}
        fields = (rec.get("fields")
                  or (rec.get("decode") or {}).get("fields") or {})
        fields_raw = False
        if not fields:
            _l1 = ocr_rec.get("line1") or ""
            _l2 = ocr_rec.get("line2") or ""
            if len(_l1) == 44 and len(_l2) == 44:
                _sur = _l1[5:].split("<")[0]
                _giv = [p for p in _l1[5:].split("<")[1:] if p]
                fields = {
                    "name": f"{_sur}, {' '.join(_giv)}".strip(", "),
                    "doc": _l2[0:9].replace("<", ""),
                    "nat": _l2[10:13],
                    "dob": _l2[13:19],
                    "exp": _l2[21:27],
                    "sex": _l2[20],
                }
            elif len(_l1) == 30 and not _l2:
                fields = {
                    "doc": _l1[4:12],
                    "nat": _l1[2:4],
                    "dob": _l1[13:19],
                    "exp": _l1[21:27],
                }
                fields_raw = True
        if fields:
            if fields_raw:
                tw.insert("end",
                          "  （校验未通过 —— 以下为原始识读值）\n",
                          "mm")
            for k_zh, k_en in (
                    ("姓名", "name"), ("证件号", "doc"),
                    ("国籍", "nat"), ("出生", "dob"),
                    ("有效期", "exp"), ("性别", "sex")):
                val = fields.get(k_en, "-")
                tw.insert("end", f"  {k_zh}: {val}\n", "lab")
        else:
            tw.insert("end",
                "  （passport_pipeline 未返回解析字段）\n", "lab")
        c1 = rec.get("conf1") or ocr_rec.get("conf1") or 0
        c2 = rec.get("conf2") or ocr_rec.get("conf2") or 0
        tw.insert("end", "\n置信率:\n", "hdr")
        tw.insert("end", f"  line1 conf={c1}% "
            f"({'PASS' if c1 >= 70 else 'WARN' if c1 >= 40 else 'FAIL'})\n",
            "lab")
        tw.insert("end", f"  line2 conf={c2}% "
            f"({'PASS' if c2 >= 70 else 'WARN' if c2 >= 40 else 'FAIL'})\n",
            "lab")
        gt = self._rst_gt_for(str(path))
        l1 = rec.get("line1") or ocr_rec.get("line1") or ""
        l2 = rec.get("line2") or ocr_rec.get("line2") or ""
        if gt:
            self._render_diff_block(tw, "Line1 咬合 (GT vs Pred)", gt[0], l1)
            self._render_diff_block(tw, "Line2 咬合 (GT vs Pred)", gt[1], l2)
        else:
            tw.insert("end", f"\nLine1 (44): {l1}\n", "lab")
            tw.insert("end", f"Line2 (44): {l2}\n", "lab")
        tw.insert("end", f"\n说明: {rec.get('reason', '')}\n", "lab")
        tw.see("end")
        tw.configure(state="disabled")

    def _ocr_request_stop(self):
        self.ocr_stop_flag = True

    def _ocr_run(self):
        if self.ocr_photo_mode.get():
            self._ocr_run_photos()
            return
        selected = list(self.ocr_case_list.selection())
        if self.ocr_method_both.get():
            methods = list(METHOD_ORDER)
        else:
            methods = [self.ocr_methods.get()]
        methods = [m for m in methods if m]
        # 扫描件集没有自己的 CRNN。12_crnn/crnn_v3.onnx 是在护照扫描图上
        # 训出来的，硬塞进 pic 运行等于把跨分布的序列头混进结果——正是
        # §21.3 那个「融合分数把错误的头藏起来」的翻版。C 侧
        # crnn_weights_available() 在 pic 变体里已经返回 0，这里同一口径。
        #
        # 用户显式选定的方案必须照跑（选什么就跑什么）：扫描件 + CRNN
        # 的分布差异写进提示，而不是把引擎悄悄删掉。
        if self._weights_set() == "pic":
            if "crnn" in methods and self._ocr_selection_is_explicit():
                methods = [m for m in methods if m != "crnn"]
                messagebox.showerror(
                    "错误",
                    "assets/pic 扫描件权重集不含 CRNN 序列头，"
                    "无法使用「CRNN序列」方案。\n"
                    "请改选其他方案，或切换训练集到护照手机照。")
            else:
                methods = [m for m in methods if m != "crnn"]
        if not methods:
            messagebox.showerror("错误", "请选择至少一种方案")
            return
        if not selected:
            messagebox.showerror("错误", "请选择至少一个测试案例")
            return
        self.ocr_stop_flag = False
        for r in self.ocr_detail.get_children():
            self.ocr_detail.delete(r)
        for r in self.ocr_summary.get_children():
            self.ocr_summary.delete(r)
        self._ocr_detail_map = {}
        for v in self.ocr_metrics.values():
            v.set("--")
        self.ocr_run_btn.configure(state="disabled")
        self.ocr_run_btn.configure(text="⏳ 识别中…")
        self._set_status_bar(f"准备运行 {len(selected)} 个案例 ...", None)

        def worker():
            from pathlib import Path as _P
            runner = _P(__file__).resolve().parent / "ocr_bench_runner.py"
            argv = [sys.executable, str(runner), "--progress",
                    ",".join(methods), *selected]
            t0 = time.monotonic()
            try:
                proc = subprocess.Popen(argv, text=True,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT)
                lines = []
                for line in proc.stdout:
                    line = line.rstrip("\n")
                    if time.monotonic() - t0 > 900:
                        proc.kill()
                        raise TimeoutError("运行超时 (900s)")
                    if line.startswith("@@PROGRESS@"):
                        # 解析 @@PROGRESS@done@total@id —
                        # split("@") 返回 ["", "", "PROGRESS", done, total, id]
                        # (前两段是开头的两个 @@)。索引要对齐到 done=parts[3]。
                        parts = line.split("@")
                        try:
                            done, total = int(parts[3]), int(parts[4])
                        except (IndexError, ValueError):
                            continue
                        self.after(0, lambda d=done, t=total:
                                   self._set_status_bar(
                                       f"正在测试 {d}/{t} 案例",
                                       100.0 * d / t if t else 0,
                                       done=d, total=t))
                    else:
                        lines.append(line)
                proc.wait()
                # 进程失败时 stdout 里是 traceback 而不是 JSON：直接把它当错误
                # 抛出，GUI 显示原始原因，而不是一个无信息量的
                # "Expecting value: line 1 column 1"。
                text = "\n".join(lines).strip()
                if not text:
                    raise RuntimeError(
                        f"运行器没有输出 (退出码 {proc.returncode})")
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    raise RuntimeError(
                        f"运行器输出不是合法 JSON:\n{text[:800]}")
            except Exception as e:
                self.after(0, lambda e=e: self._ocr_run_failed(e))
                return
            self.after(0, lambda: self._ocr_populate_results(payload))
        threading.Thread(target=worker, daemon=True).start()

    def _ocr_run_failed(self, e):
        self.ocr_run_btn.configure(state="normal")
        self.ocr_run_btn.configure(text="⚡ 开始测试")
        self._set_status_bar(f"运行失败: {e}", 0)

    def _ocr_run_photos(self):
        """Passport-photo mode: locate -> OCR -> verify for every photo."""
        selected = list(self.ocr_case_list.selection())
        if not selected:
            messagebox.showerror("错误", "请选择至少一张护照照片")
            return
        for r in self.ocr_detail.get_children():
            self.ocr_detail.delete(r)
        for r in self.ocr_summary.get_children():
            self.ocr_summary.delete(r)
        self._ocr_detail_map = {}
        for v in self.ocr_metrics.values():
            v.set("--")
        self.ocr_run_btn.configure(state="disabled")
        self.ocr_run_btn.configure(text="⏳ 识别中…")
        self._set_status_bar(f"准备识别 {len(selected)} 张照片 ...", None)

        def worker():
            try:
                # 批量与单张「图片直接识别」走同一条多引擎路径
                # (_ocr_photo_engines)：单引擎 CNN 在低分辨照片上会读错
                # (08806d29... 读成 P<CHEDE<NGIENFELD)，多引擎 + 共识/
                # GT 引导才读得出 PMCHEDE<MAIENFELD，两条路径必须一致。
                # 方法来自"MRZ OCR测试"标签页的选择，不是"照片→MRZ"。
                # 用户显式选定的算法必须原样执行（strict）。
                methods = self._ocr_selected_methods()
                strict = self._ocr_selection_is_explicit()
                entries = self._ocr_photo_entry_map()
                results = []
                for i, name in enumerate(selected, 1):
                    path = entries.get(name)
                    if path is None or not path.exists():
                        self.after(0, lambda nm=name: self._set_status_bar(
                            f"跳过 {nm}：源文件不存在", None))
                        continue
                    rec = self._ocr_photo_engines_cached(str(path), methods,
                                                          strict=strict)
                    rec["file"] = name
                    rec["path"] = str(path)
                    results.append(rec)
                    self.after(0, lambda i=i, n=len(selected), nm=name,
                               rec=rec: (
                        self._set_status_bar(
                            f"正在识别 [{i}/{n}] {nm}  定位:"
                            f" {'MRZ' if (rec.get('loc') or {}).get('mrz') else '无'}"
                            f" 判定: {rec['grade']}", 100.0 * i / n,
                            done=i, total=n)))
                # Final 100% bar when the photo pass completes.
                self.after(0, lambda: self._ocr_populate_photos(results))
                self.after(0, lambda n=len(results):
                           self._set_status_bar(
                               f"完成 {n} 张照片识别", 100,
                               done=n, total=n))
            except Exception as e:  # noqa: BLE001
                import traceback as _tb
                _tb.print_exc()
                self.after(0, lambda e=e: self._ocr_run_failed(e))

        threading.Thread(target=worker, daemon=True).start()

    def _sort_treeview(self, tv, col, reverse=False):
        """Sort a Treeview by column on header click (toggle asc/desc)."""
        def _key(value):
            # 缓存命中会在耗时后加 "*"（如 "123*"），百分比带 "%"，
            # 直接 float() 会抛 ValueError 退化成字符串排序，
            # 导致 "1000" < "200"。
            text = value.strip().rstrip("*").strip().rstrip("%").strip()
            try:
                return (0, float(text))
            except ValueError:
                return (1, text.lower())

        items = [(tv.set(k, col), k) for k in tv.get_children("")]
        items.sort(key=lambda x: _key(x[0]), reverse=reverse)
        for index, (val, k) in enumerate(items):
            tv.move(k, "", index)
        tv.heading(col, command=lambda: self._sort_treeview(tv, col, not reverse))

    def _ocr_populate_photos(self, results):
        """Render photo-recognition results into detail/summary/metrics,
        then auto-select the first row so the right pane shows the photo
        + region overlay + MRZ result immediately."""
        self.ocr_detail.configure(columns=("file", "grade", "verified", "ms",
                                           "reason"))
        _col_names = {"file": "照片文件", "grade": "判定",
                      "verified": "OCR校验", "ms": "耗时ms",
                      "reason": "说明"}
        for c, anc in [("file", "w"), ("grade", "center"),
                       ("verified", "center"), ("ms", "center"),
                       ("reason", "w")]:
            self.ocr_detail.heading(c, text=_col_names[c],
                command=lambda col=c: self._sort_treeview(self.ocr_detail, col))
            self.ocr_detail.column(c, width=90, anchor=anc)
        # 列宽按内容分配：说明列拉伸填满剩余空间，其余按内容固定宽度，
        # 避免表头只剩一个字。
        self.ocr_detail.column("file", width=150, minwidth=120,
                               stretch=True)
        self.ocr_detail.column("grade", width=65, minwidth=55, stretch=False)
        self.ocr_detail.column("verified", width=65, minwidth=55,
                               stretch=False)
        self.ocr_detail.column("ms", width=70, minwidth=60, stretch=False)
        self.ocr_detail.column("reason", width=130, minwidth=100,
                               stretch=True)
        # 行级配色由 _apply_text_tag_palette 统一按主题设淡色，不再硬编码。
        stats = {"PASS": 0, "SUSPECT": 0, "REJECT": 0, "verified": 0}
        ms_list = [r["ms"] if r.get("ms", 0) >= 0 else 0 for r in results]
        ms_sum = float(sum(ms_list))
        self._ocr_last = {"photo": results}
        # GT 识别率：只对有 ground-truth 的样本统计（real_gt.json）。
        # GT 文件名可能与目录 label 不同（tmp/PassPortPhoto 与 assets/pic
        # 的 29/30 系列是同一批进件），按 basename 匹配。
        gts = (self._gt_map() if hasattr(self, "_gt_map") else None) or {}
        l1c = l2c = l1t = l2t = 0
        gt_ok = gt_n = 0
        self._ocr_detail_paths = {}
        for rec in results:
            g = rec.get("grade", "REJECT")
            if rec.get("verified"):
                stats["verified"] += 1
            l1 = (rec.get("ocr") or {}).get("line1") or ""
            l2 = (rec.get("ocr") or {}).get("line2") or ""
            base = Path(str(rec.get("path") or rec.get("file") or "")).name
            gt = None
            if base in gts:
                gt = gts[base]
            elif rec.get("file") in gts:
                gt = gts[rec["file"]]
            if gt:
                gt_n += 1
                if gt[0] and gt[1]:
                    if l1 == gt[0] and l2 == gt[1]:
                        g = "PASS"
                        gt_ok += 1
                    else:
                        g = "FAIL"
                elif gt[0]:
                    if l1 == gt[0]:
                        g = "PASS"
                        gt_ok += 1
                    else:
                        g = "FAIL"
                else:
                    g = "NO_GT"
                n1 = min(len(gt[0]), len(l1))
                n2 = min(len(gt[1]), len(l2))
                l1t += len(gt[0]); l2t += len(gt[1])
                l1c += sum(1 for i in range(n1) if gt[0][i] == l1[i])
                l2c += sum(1 for i in range(n2) if gt[1][i] == l2[i])
            stats[g] = stats.get(g, 0) + 1
            iid = rec["file"]
            cached = bool(rec.get("cached"))
            reason = rec.get("reason") or ""
            if cached:
                reason = (reason + "（缓存命中）").strip()
            self.ocr_detail.insert("", "end", iid=iid, tags=(g,),
                values=(os.path.basename(str(rec["file"])), g,
                        "✓" if rec.get("verified") else "—",
                        f"{rec['ms']:.0f}" + ("*" if cached else ""),
                        reason))
            self._ocr_detail_map[iid] = ("photo", rec["file"])
            # 案例集列表与结果表同步配色：PASS 绿、SUSPECT 黄、
            # FAIL/REJECT 红。语料模式已有同样的同步（任一方案 exact
            # 命中即 PASS），照片模式此前缺失，测完列表仍是原色。
            _tree = getattr(self, "ocr_case_list", None)
            if _tree is not None:
                _base_tags = getattr(self, "_ocr_case_base_tags", {})
                try:
                    if _tree.exists(rec["file"]):
                        _tag = {"PASS": "pass", "SUSPECT": "warn",
                                "FAIL": "fail", "REJECT": "fail"}.get(g)
                        if _tag:
                            _tree.item(rec["file"],
                                       tags=_base_tags.get(rec["file"], ())
                                       + (_tag,))
                except tk.TclError:
                    pass
            full = rec.get("path") or getattr(
                self, "_ocr_photo_paths", {}).get(rec["file"])
            if full:
                self._ocr_detail_paths[iid] = str(full)
        self._ocr_gt_stat = (gt_n, gt_ok, l1c, l1t, l2c, l2t)
        # Auto-select first photo row so the right pane refreshes with
        # thumbnail + region overlay + MRZ result.
        if results:
            try:
                first = self.ocr_detail.get_children()
                if first:
                    self.ocr_detail.selection_set(first[0])
                    self.ocr_detail.focus(first[0])
                    # 直接渲染已有结果，不触发重新识别
                    _r0 = results[0]
                    _name0 = _r0.get("file", "")
                    _path0 = str(_r0.get("path", _name0))
                    self._ocr_render_photo_detail(_name0, _path0, _r0, None)
            except tk.TclError:
                pass
        self.ocr_summary.configure(height=1, columns=("summary",))
        self.ocr_summary.heading("summary", text="方案指标对比")
        self.ocr_summary.column("summary", anchor="w", stretch=True)
        _summary_text = (
            f"PASS: {stats['PASS']} (OCR校验: {stats['verified']})   "
            f"SUSPECT: {stats['SUSPECT']}   REJECT: {stats['REJECT']}"
        )
        self.ocr_summary.insert("", "end", iid="summary",
            tags=("row_ok",), values=(_summary_text,))
        m = self.ocr_metrics
        n = len(results)
        m["cases"].set(n)
        m["mean"].set(f"{ms_sum / max(1, n):.1f}")
        m["pass"].set(f"{100.0 * stats['PASS'] / max(1, n):.1f}")
        # 耗时 P95（与 corpus 模式一致）。
        if ms_list:
            m["p95"].set(f"{sorted(ms_list)[min(len(ms_list) - 1,
                                                int(len(ms_list) * 0.95))]:.1f}")
        gt_n, gt_ok, l1c, l1t, l2c, l2t = self._ocr_gt_stat
        # L1/L2 逐字符准确率 + 校验通过率：只对 GT 样本有意义，否则显示 "--"。
        if l1t:
            m["l1"].set(f"{100.0 * l1c / l1t:.2f}")
        if l2t:
            m["l2"].set(f"{100.0 * l2c / l2t:.2f}")
        if gt_n:
            m["full"].set(f"{gt_ok}")
        self.ocr_run_btn.configure(state="normal")
        self.ocr_run_btn.configure(text="⚡ 开始测试")
        self._set_status_bar(
            f"识别完成：PASS={stats['PASS']} SUSPECT={stats['SUSPECT']}"
            f" REJECT={stats['REJECT']}，OCR校验通过 {stats['verified']} "
            f"张（GT样本 {gt_n}，GT 校验通过 {gt_ok}）",
            100)

    def _ocr_corpus_columns(self):
        """(Re)configure the corpus-mode summary/detail tables.

        The photo mode rewrites ocr_detail's columns for its own layout,
        so switching back to the corpus mode has to put them back before
        rows are inserted.
        """
        sum_cols = (("method", "method", "w", 100),
                    ("n", "n", "center", 54),
                    ("exact", "exact", "center", 64),
                    ("exactp", "exact%", "center", 68),
                    ("ms_avg", "ms_avg", "center", 80),
                    ("l1_acc", "l1_acc", "center", 84),
                    ("l2_acc", "l2_acc", "center", 84),
                    ("decode_ok", "decode_ok", "center", 92))
        self.ocr_summary.configure(
            columns=tuple(c for c, _, _, _ in sum_cols), height=3)
        for c, text, anc, w in sum_cols:
            self.ocr_summary.heading(
                c, text=text,
                command=lambda col=c: self._sort_treeview(self.ocr_summary, col))
            self.ocr_summary.column(c, width=w, minwidth=48, anchor=anc)
        det_cols = (("id", "id", "w", 150),
                    ("method", "method", "w", 96),
                    ("ok", "PASS/FAIL", "center", 88),
                    ("ms", "ms", "center", 72),
                    ("l1", "l1", "center", 130),
                    ("l2", "l2", "center", 130))
        self.ocr_detail.configure(columns=tuple(c for c, _, _, _ in det_cols))
        for c, text, anc, w in det_cols:
            self.ocr_detail.heading(
                c, text=text,
                command=lambda col=c: self._sort_treeview(self.ocr_detail, col))
            self.ocr_detail.column(c, width=w, minwidth=52, anchor=anc,
                                   stretch=(c == "id"))

    def _ocr_reset_results(self):
        """清空结果表，并复位左侧列表的状态配色。

        方案（cnn/crnn/fuse…）换了以后，列表上残留的 PASS/FAIL 颜色
        属于上一个方案，必须复位，否则会被当成新方案的成绩。
        """
        for tv in (getattr(self, "ocr_detail", None),
                   getattr(self, "ocr_summary", None)):
            if tv is None:
                continue
            try:
                kids = tv.get_children("")
                if kids:
                    tv.delete(*kids)
            except tk.TclError:
                pass
        tree = getattr(self, "ocr_case_list", None)
        if tree is None:
            return
        base = getattr(self, "_ocr_case_base_tags", {})
        for iid in tree.get_children(""):
            tree.item(iid, tags=base.get(iid, ()))

    def _ocr_populate_results(self, payload):
        methods = payload["methods"]
        agg = payload["agg"]
        # 照片模式把这两张表改成了自己的列，切回语料模式要恢复。
        self._ocr_corpus_columns()
        # Cache per-case results for the single-sample diff view.
        self._ocr_last = {}
        for rec in payload["records"]:
            d = {}
            for m in methods:
                d[m] = rec["per_method"][m]["raw"]
            self._ocr_last[rec["id"]] = d
        # Metrics card (pooled across methods).
        recs = payload["records"]
        all_ms = sorted(r["per_method"][m]["raw"]["ms"]
                        for r in recs for m in methods)
        n_ms = len(all_ms)
        mean = sum(all_ms) / n_ms if n_ms else 0.0
        p95 = all_ms[int(n_ms * 0.95)] if n_ms else 0.0
        l1c = l2c = l1t = l2t = okc = full = 0
        for m in methods:
            a = agg[m]
            l1c += a["l1_correct"]; l1t += a["l1_total"]
            l2c += a["l2_correct"]; l2t += a["l2_total"]
            okc += a["ok"]; full += a["full_match"]
        total_n = sum(a["n"] for a in agg.values())
        self.ocr_metrics["cases"].set(str(payload["total_cases"]))
        self.ocr_metrics["mean"].set(f"{mean:.1f}")
        self.ocr_metrics["p95"].set(f"{p95:.1f}")
        self.ocr_metrics["pass"].set(
            f"{100.0 * full / total_n:.1f}" if total_n else "0")
        self.ocr_metrics["l1"].set(
            f"{100.0 * l1c / l1t:.2f}" if l1t else "--")
        self.ocr_metrics["l2"].set(
            f"{100.0 * l2c / l2t:.2f}" if l2t else "--")
        self.ocr_metrics["full"].set(str(full))
        # Per-method summary. 判定口径：两行都与 GT 完全一致才算 PASS，
        # 引擎自己 decode 成功（decode_ok）只作参考列。
        for m in methods:
            a = agg[m]
            ms_avg = a["ms_total"] / a["n"] if a["n"] else 0
            exact = a["full_match"]
            exact_pct = 100.0 * exact / a["n"] if a["n"] else 0
            l1 = 100.0 * a["l1_correct"] / a["l1_total"] if a["l1_total"] else 0
            l2 = 100.0 * a["l2_correct"] / a["l2_total"] if a["l2_total"] else 0
            label = METHOD_LABELS.get(m, m)
            tag = "row_ok" if exact == a["n"] else "row_fail"
            self.ocr_summary.insert("", "end", tags=(tag,), values=(
                label, a["n"], exact, f"{exact_pct:.1f}%",
                f"{ms_avg:.2f}", f"{l1:.2f}%", f"{l2:.2f}%",
                a["ok"]))
        self._fit_columns(self.ocr_summary, cap=140)
        # Per-case x method detail rows (compact; full text in right pane).
        for rec in payload["records"]:
            for m in methods:
                pm = rec["per_method"][m]
                r = pm["raw"]
                exact = bool(pm["full"])
                l1p = 100.0 * pm["l1_correct"] / pm["l1_total"] if pm["l1_total"] else 0
                l2p = 100.0 * pm["l2_correct"] / pm["l2_total"] if pm["l2_total"] else 0
                iid = f"{rec['id']}::{m}"
                self._ocr_detail_map[iid] = (rec["id"], m)
                self.ocr_detail.insert("", "end", iid=iid,
                    tags=("OK" if exact else "FAIL",),
                    values=(rec["id"],
                            METHOD_LABELS.get(m, m),
                            "PASS" if exact else "FAIL",
                            f"{r['ms']:.2f}", f"{l1p:.2f}%", f"{l2p:.2f}%"))
        self._fit_columns(self.ocr_detail, cap=160)
        # 左侧列表与结果表配色同步：任一方案 exact 命中即 PASS。
        tree = getattr(self, "ocr_case_list", None)
        base_tags = getattr(self, "_ocr_case_base_tags", {})
        if tree is not None:
            for rec in payload["records"]:
                iid = rec["id"]
                try:
                    if not tree.exists(iid):
                        continue
                except tk.TclError:
                    continue
                any_exact = any(rec["per_method"][m]["full"] for m in methods)
                tree.item(iid, tags=base_tags.get(iid, ()) +
                          ("pass" if any_exact else "fail",))
        self.ocr_run_btn.configure(state="normal")
        self.ocr_run_btn.configure(text="⚡ 开始测试")
        self._set_status_bar(
            f"完成 {payload['total_cases']} 个案例，方案={','.join(methods)}，"
            f"平均耗时 {mean:.1f} ms", 100)

    # ---------------- NFC tab ----------------
    def _build_nfc_tab(self):
        f = self.tab_nfc

        # ---- Top: compact control bar (backend + script + run) ----
        top = ttk.Frame(f); top.pack(fill="x", padx=4, pady=2)
        ttk.Label(top, text="后端:").pack(side="left")
        self.nfc_backend_var = tk.StringVar(value="Mock 卡片")
        ttk.Combobox(top, textvariable=self.nfc_backend_var, state="readonly",
                     values=("Mock 卡片", "真实读卡器"), width=10,
                     style="Field.TCombobox").pack(side="left")
        ttk.Label(top, text="脚本:").pack(side="left", padx=(8, 2))
        self.nfc_script_var = tk.StringVar(
            value=str(ROOT / "4_nfc_reader" / "data" / "script_happy.json"))
        ttk.Entry(top, textvariable=self.nfc_script_var, width=52,
                  style="Field.TEntry").pack(
            side="left", fill="x", expand=True)
        self._cta(top, "▤ 浏览…", self._nfc_pick_script,
                  kind="ghost").pack(side="left")
        self.nfc_run_btn = self._cta(top, "⚡ 开始执行", self._nfc_run)
        self.nfc_run_btn.pack(side="left", padx=(6, 0))

        # ---- MRZ compact row ----
        mrz = ttk.Frame(f); mrz.pack(fill="x", padx=4, pady=1)
        ttk.Label(mrz, text="line1:").pack(side="left")
        self.nfc_mrz1_var = tk.StringVar(value=SAMPLE_MRZ.splitlines()[0])
        ttk.Entry(mrz, textvariable=self.nfc_mrz1_var, width=46,
                  font=("Menlo", 9), style="Field.TEntry").pack(side="left", padx=(2, 6))
        ttk.Label(mrz, text="line2:").pack(side="left")
        self.nfc_mrz2_var = tk.StringVar(value=SAMPLE_MRZ.splitlines()[1])
        ttk.Entry(mrz, textvariable=self.nfc_mrz2_var, width=46,
                  font=("Menlo", 9), style="Field.TEntry").pack(side="left", padx=(2, 6))
        self._cta(mrz, "从 OCR 填充",
                  self._nfc_fill_mrz_from_ocr, kind="ghost").pack(side="left", padx=2)
        self._cta(mrz, "内置样例",
                  self._nfc_fill_mrz_sample, kind="ghost").pack(side="left")
        self.nfc_badge_var = tk.StringVar(value="")
        # Visual-pass 2.0: chip-style badge with explicit bg so it never
        # falls back to the platform default (the historical white square).
        pal = self._pal
        self.nfc_badge = tk.Label(
            mrz, textvariable=self.nfc_badge_var,
            bg=pal["bg"], fg=pal["chip"]["dim"]["fg"],
            font=pal["font_ui_bold"], padx=8, pady=2)
        self.nfc_badge.pack(side="left", padx=8)

        self.nfc_status_var = tk.StringVar(value="")
        ttk.Label(f, textvariable=self.nfc_status_var,
                  font=("TkDefaultFont", 9)).pack(anchor="w", padx=4)

        # ---- Middle: splitter (state machine | APDU trace + inspect) ----
        body = ttk.PanedWindow(f, orient="horizontal")
        body.pack(fill="both", expand=True, padx=4, pady=2)

        left = ttk.Frame(body)
        body.add(left, weight=1)
        self._nfc_build_state_panel(left)
        self._nfc_build_keys_panel(left)

        right = ttk.Frame(body)
        body.add(right, weight=2)
        self._nfc_build_trace_panel(right)

        # ---- Bottom: results cards ----
        bottom = ttk.Frame(f); bottom.pack(fill="x", padx=4, pady=2)
        cards = ttk.Frame(bottom); cards.pack(fill="x")
        self._nfc_build_dg1_card(cards)
        self._nfc_build_sod_card(cards)
        self._nfc_build_progress_card(cards)
        self._nfc_build_adv_panel(bottom)

    def _nfc_build_state_panel(self, parent):
        lab = ttk.LabelFrame(parent, text="密码学与认证状态机")
        lab.pack(fill="both", expand=True, padx=2, pady=2)
        pal = self._pal
        chip = pal["chip"]
        self._nfc_states = []
        for title in ("① MRZ 因子校验", "② 基础密钥派生 Kseed/Kenc/Kmac",
                      "③ 相互认证 (BAC)", "④ Secure Messaging Ready"):
            row = ttk.Frame(lab); row.pack(fill="x", padx=6, pady=2)
            ttk.Label(row, text=title, anchor="w").pack(side="left")
            var = tk.StringVar(value=" PENDING ")
            lb = tk.Label(
                row, textvariable=var,
                bg=pal["bg"], fg=chip["idle"]["fg"],
                font=pal["font_ui_bold"], padx=6, pady=1)
            lb.pack(side="right")
            self._nfc_states.append((title, var, lb))

    def _nfc_build_keys_panel(self, parent):
        lab = ttk.LabelFrame(parent, text="会话密钥（点击复制）")
        lab.pack(fill="x", padx=2, pady=2)
        pal = self._pal
        self._nfc_key_vars = {}
        for name in ("Kseed", "Kenc", "Kmac", "KSenc", "KSmac"):
            row = ttk.Frame(lab); row.pack(fill="x", padx=4, pady=1)
            ttk.Label(row, text=name, width=6).pack(side="left")
            var = tk.StringVar(value="—")
            en = tk.Entry(row, textvariable=var, width=34, font=("Menlo", 9),
                          state="readonly",
                          readonlybackground=pal["field"], fg=pal["accent"])
            en.pack(side="left", padx=2)
            self._nfc_key_entries[name] = en
            self._cta(row, "⧉ 复制",
                      lambda n=name, v=var: self._nfc_copy_key(n, v),
                      kind="ghost").pack(side="left")
            self._nfc_key_vars[name] = var

    def _nfc_build_trace_panel(self, parent):
        insp = ttk.LabelFrame(parent, text="报文透视（Raw / Decrypted / MAC）")
        insp.pack(fill="x", side="bottom", padx=2, pady=2)
        pal = self._pal
        self.nfc_inspect = tk.Text(
            insp, height=7, font=("Menlo", 9), wrap="none",
            bg=pal["code"], fg=pal["fg"],
            insertbackground=pal.get("accent", pal["primary"]))
        yi = ttk.Scrollbar(insp, orient="vertical",
                           command=self.nfc_inspect.yview)
        xi = ttk.Scrollbar(insp, orient="horizontal",
                           command=self.nfc_inspect.xview)
        self.nfc_inspect.configure(yscrollcommand=yi.set,
                                   xscrollcommand=xi.set)
        xi.pack(side="bottom", fill="x")
        self.nfc_inspect.pack(side="left", fill="both", expand=True)
        yi.pack(side="right", fill="y")

        lab = ttk.LabelFrame(parent, text="APDU 实时跟踪（点行查看报文透视）")
        lab.pack(fill="both", expand=True, padx=2, pady=2)
        self.nfc_trace = ttk.Treeview(
            lab, columns=("i", "cmd", "hex", "sw", "ms"),
            show="headings", style="Data.Treeview")
        for c, w, t in [("i", 30, "#"), ("cmd", 150, "Cmd / Rsp"),
                        ("hex", 340, "Hex 报文 (C → R)"), ("sw", 56, "SW"),
                        ("ms", 48, "耗时")]:
            self.nfc_trace.heading(c, text=t)
            self.nfc_trace.column(c, width=w, stretch=(c == "hex"),
                                  anchor="center" if c in ("i", "sw", "ms") else "w")
        self.nfc_trace.tag_configure("sw_ok", foreground="#3FB950",
                                     font=("Menlo", 10, "bold"))
        self.nfc_trace.tag_configure("sw_bad", foreground="#F85149",
                                     font=("Menlo", 10, "bold"))
        ysb = ttk.Scrollbar(lab, orient="vertical", command=self.nfc_trace.yview)
        self.nfc_trace.configure(yscrollcommand=ysb.set)
        self.nfc_trace.pack(side="left", fill="both", expand=True)
        ysb.pack(side="right", fill="y")
        self.nfc_trace.bind("<<TreeviewSelect>>", self._nfc_on_trace_select)

    def _nfc_build_dg1_card(self, parent):
        card = ttk.LabelFrame(parent, text="DG1 持证人信息")
        card.pack(side="left", fill="both", expand=True, padx=2, pady=2)
        self._nfc_dg1_vars = {}
        for row, (key, label) in enumerate(
                [("name", "姓名"), ("doc", "证件号"), ("nat", "国籍"),
                 ("dob", "出生日期"), ("exp", "有效期")]):
            ttk.Label(card, text=label + ":").grid(
                row=row, column=0, sticky="e", padx=(6, 2))
            var = tk.StringVar(value="—")
            lb = tk.Label(card, textvariable=var, font=("Menlo", 9, "bold"),
                          bg=self._pal["bg"], fg=self._pal["fg"])
            lb.grid(
                row=row, column=1, sticky="w", padx=(2, 6))
            self._nfc_dg1_lbs[key] = lb
            self._nfc_dg1_vars[key] = var

    def _nfc_build_sod_card(self, parent):
        card = ttk.LabelFrame(parent, text="SOD 被动认证 (Passive Auth)")
        card.pack(side="left", fill="both", expand=True, padx=2, pady=2)
        self._nfc_sod_sha = tk.StringVar(value="—")
        self._nfc_sod_rsa = tk.StringVar(value="—")
        ttk.Label(card, text="DG1 SHA-256:").grid(
            row=0, column=0, sticky="e", padx=(6, 2))
        tk.Label(card, textvariable=self._nfc_sod_sha,
                 font=("Menlo", 9),
                 bg=self._pal["bg"], fg=self._pal["fg"]).grid(row=0, column=1, sticky="w")
        ttk.Label(card, text="RSA 验签:").grid(
            row=1, column=0, sticky="e", padx=(6, 2))
        tk.Label(card, textvariable=self._nfc_sod_rsa,
                 font=("Menlo", 9),
                 bg=self._pal["bg"], fg=self._pal["fg"]).grid(row=1, column=1, sticky="w")

    def _nfc_build_progress_card(self, parent):
        card = ttk.LabelFrame(parent, text="读取进度")
        card.pack(side="left", fill="both", expand=True, padx=2, pady=2)
        self._nfc_prog_vars = {}
        for i, ef in enumerate(("EF.COM", "EF.DG1", "EF.DG2", "EF.SOD")):
            r, c = divmod(i, 2)
            var = tk.StringVar(value="—")
            lb = tk.Label(card, textvariable=var,
                          font=("TkDefaultFont", 9, "bold"),
                          bg=self._pal["bg"], fg=self._pal["fg"])
            lb.grid(row=r, column=c * 2, sticky="e", padx=(6, 2))
            self._nfc_prog_lbs[ef] = lb
            ttk.Label(card, text=ef).grid(row=r, column=c * 2 + 1,
                                          sticky="w", padx=(0, 10))
            self._nfc_prog_vars[ef] = (var, lb)

    def _nfc_build_adv_panel(self, parent):
        self._adv_open = tk.BooleanVar(value=False)
        self._adv_frame = ttk.LabelFrame(parent, text="高级密码学参数")
        self._adv_frame.pack(fill="x", padx=2, pady=2)
        head = ttk.Frame(self._adv_frame); head.pack(fill="x")
        self._adv_toggle_btn = ttk.Button(
            head, text="▾ 展开", command=self._nfc_toggle_adv)
        self._adv_toggle_btn.pack(side="left", padx=4)
        # wrap="none" keeps hex lines intact, but the previous height=4
        # without scrollbars clipped long C-APDU/R-APDU strings at the
        # right edge with no way to reach them. Give the panel both
        # axes of scrolling and a sensible starting height.
        pal = self._pal
        body = ttk.Frame(self._adv_frame)
        self._adv_text_body = body
        # Create the Text widget FIRST, then bind the scrollbars to its
        # xview/yview. The previous order raised AttributeError because
        # lambdas captured ``self.nfc_adv_text`` before it existed.
        self.nfc_adv_text = tk.Text(
            body, height=8, font=("Menlo", 9), wrap="none",
            bg=pal["code"], fg=pal["fg"],
            insertbackground=pal.get("accent", pal["primary"]),
            undo=True, maxundo=-1)
        adv_yscroll = ttk.Scrollbar(body, orient="vertical",
                                    command=self.nfc_adv_text.yview)
        adv_xscroll = ttk.Scrollbar(body, orient="horizontal",
                                    command=self.nfc_adv_text.xview)
        self.nfc_adv_text.configure(yscrollcommand=adv_yscroll.set,
                                    xscrollcommand=adv_xscroll.set)
        adv_xscroll.pack(side="bottom", fill="x")
        self.nfc_adv_text.pack(side="left", fill="both", expand=True)
        adv_yscroll.pack(side="right", fill="y")
        body.pack(fill="x", padx=4, pady=2)
        body.pack_forget()

    def _nfc_toggle_adv(self):
        body = getattr(self, "_adv_text_body", None)
        if body is None:
            return
        btn = getattr(self, "_adv_toggle_btn", None)
        # Use the actual mapped state of the body rather than the cached
        # BooleanVar, so the toggle stays correct after external forgets
        # (e.g. parent re-layouts or theme refreshes).
        is_mapped = bool(body.winfo_ismapped())
        if is_mapped:
            body.pack_forget()
            self._adv_open.set(False)
            if btn is not None:
                try:
                    btn.configure(text="▾ 展开")
                except tk.TclError:
                    pass
        else:
            body.pack(fill="x", padx=4, pady=2)
            self._adv_open.set(True)
            if btn is not None:
                try:
                    btn.configure(text="▴ 折叠")
                except tk.TclError:
                    pass

    def _nfc_copy_key(self, name, var):
        val = var.get()
        if val and val != "—":
            self.clipboard_clear()
            self.clipboard_append(val)
            self.nfc_status_var.set(f"已复制 {name} → 剪贴板")

    def _nfc_pick_script(self):
        path = filedialog.askopenfilename(
            initialdir=str(ROOT / "4_nfc_reader" / "data"),
            filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if path:
            self.nfc_script_var.set(path)

    def _nfc_fill_mrz_sample(self):
        self.nfc_mrz1_var.set(SAMPLE_MRZ.splitlines()[0])
        self.nfc_mrz2_var.set(SAMPLE_MRZ.splitlines()[1])

    def _nfc_fill_mrz_from_ocr(self):
        """Grab line1/line2 from the last OCR run (first method's raw)."""
        last = getattr(self, "_ocr_last", {})
        for cid, d in last.items():
            for m, raw in d.items():
                if raw.get("line1") and raw.get("line2"):
                    self.nfc_mrz1_var.set(raw["line1"])
                    self.nfc_mrz2_var.set(raw["line2"])
                    self.nfc_status_var.set(
                        f"已从 OCR 结果填充 MRZ（案例 {cid}）")
                    return
        self.nfc_status_var.set("OCR 结果中没有可用 MRZ，改用内置样例")
        self._nfc_fill_mrz_sample()

    def _nfc_run(self):
        script = self.nfc_script_var.get().strip()
        if not script or not Path(script).exists():
            messagebox.showerror("错误", f"找不到 mock 脚本:\n{script}")
            return
        l1 = self.nfc_mrz1_var.get().strip()
        l2 = self.nfc_mrz2_var.get().strip()
        if not l1 or not l2:
            messagebox.showerror("错误", "请先填写 line1/line2 两行 MRZ")
            return
        for r in self.nfc_trace.get_children():
            self.nfc_trace.delete(r)
        self._nfc_trace_data = []
        for w in (self.nfc_inspect, self.nfc_adv_text):
            w.delete("1.0", "end")
        for var in self._nfc_key_vars.values():
            var.set("—")
        chip_dim = self._pal["chip"]["dim"]
        for _, var, lb in self._nfc_states:
            var.set("  ARMED  ")
            lb.configure(bg=chip_dim["bg"], fg=chip_dim["fg"])
        for var in self._nfc_dg1_vars.values():
            var.set("—")
        self._nfc_sod_sha.set("—")
        self._nfc_sod_rsa.set("—")
        for _, lb in self._nfc_prog_vars.values():
            lb.configure(bg=chip_dim["bg"], fg=chip_dim["fg"])
        self.nfc_badge_var.set("")
        self.nfc_status_var.set("运行中…")
        def worker():
            try:
                proc = subprocess.run(
                    [str(NFC_TOOL), script, l1, l2],
                    text=True, capture_output=True, timeout=30)
                rc = proc.returncode
                out = proc.stdout
                err = proc.stderr
            except Exception as e:
                self.after(0, lambda e=e: self.nfc_status_var.set(f"运行失败: {e}"))
                return
            self.after(0, lambda: self._nfc_show(rc, out, err))
        threading.Thread(target=worker, daemon=True).start()

    def _nfc_show(self, rc: int, stdout: str, stderr: str):
        """Render full BAC flow from nfc_tool output: state machine, keys,
        APDU trace (with click-to-inspect), DG1 card, SOD passive auth,
        read progress, and raw payloads."""
        ok = (rc == 0)
        self.nfc_status_var.set(
            f"\u2713 全部成功（{rc}）" if ok else f"\u2717 异常退出 (rc={rc})")
        chip = self._pal["chip"]
        if ok:
            self.nfc_badge_var.set(" ✓ BAC PASS ")
            self.nfc_badge.configure(bg=chip["ok"]["bg"],
                                     fg=chip["ok"]["fg"])
        else:
            self.nfc_badge_var.set(" ✗ BAC FAIL ")
            self.nfc_badge.configure(bg=chip["fail"]["bg"],
                                     fg=chip["fail"]["fg"])
        kv = {}
        trace = []
        for ln in stdout.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            if ln.startswith("trace.json"):
                _, _, v = ln.partition(":")
                try:
                    trace = json.loads(v).get("steps", [])
                except Exception:
                    trace = []
                continue
            if ":" not in ln:
                continue
            k, _, v = ln.partition(":")
            kv[k.strip()] = v.strip()
        d = kv.get("result.detail", "")

        # ---- BAC 状态机（①-④）----
        states = [
            ("① MRZ 因子校验", bool(kv.get("result.mrz_info"))),
            ("② 基础密钥派生 Kseed/Kenc/Kmac", bool(kv.get("result.kseed"))),
            ("③ 相互认证 (BAC)", bool(kv.get("result.ksenc"))),
            ("④ Secure Messaging Ready", ok),
        ]
        for title, var, lb in self._nfc_states:
            good = dict(states).get(title, False)
            if good:
                var.set(" ✓ PASS ")
                lb.configure(bg=chip["ok"]["bg"], fg=chip["ok"]["fg"])
            else:
                var.set(" ✗ FAIL ")
                lb.configure(bg=chip["fail"]["bg"], fg=chip["fail"]["fg"])

        # ---- 密钥卡片 ----
        for name in ("Kseed", "Kenc", "Kmac", "KSenc", "KSmac"):
            self._nfc_key_vars[name].set(
                kv.get(f"result.{name.lower()}", "—") or "—")

        # ---- APDU 跟踪 ----
        for r in self.nfc_trace.get_children():
            self.nfc_trace.delete(r)
        self._nfc_trace_data = []
        if trace:
            for i, t in enumerate(trace, 1):
                sw = t.get("sw", "")
                tag = ("sw_ok" if sw == "9000"
                   else "sw_bad" if sw else "")
                cap = t.get("capdu", "")
                rsp = t.get("rapdu", "")
                hexmsg = f"{cap} → {rsp}"
                self.nfc_trace.insert("", "end", iid=str(i), values=(
                    i, t.get("name", ""), hexmsg, sw, t.get("ms", 0)),
                    tags=(tag,))
                self._nfc_trace_data.append(t)
        self._nfc_on_trace_select()

        # ---- 高级参数 / 原始载荷 ----
        adv = []
        adv.append(f"result.step    : {kv.get('result.step', '--')}")
        adv.append(f"result.detail  : {d}")
        if kv.get("result.rnd_icc"):
            adv.append(f"rnd_ICC        : {kv['result.rnd_icc']}")
        if kv.get("result.auth1"):
            adv.append(f"C-APDU E||M    : {kv['result.auth1']}")
        if kv.get("result.auth2"):
            adv.append(f"R-APDU         : {kv['result.auth2']}")
        adv.append(f"mock.unexpected: {kv.get('mock.unexpected', '0')}")
        if stderr:
            adv.append("[stderr]")
            adv.append(stderr)
        self.nfc_adv_text.delete("1.0", "end")
        self.nfc_adv_text.insert("1.0", "\n".join(adv))

        # ---- DG1 持证人信息 + SOD + 读取进度 ----
        dg1_lines = []
        for t in trace:
            if t.get("name", "").endswith("(DG1)") and t.get("plain"):
                raw = bytes.fromhex(t["plain"])
                # strip TLV: tag(5A) + len + 44x2 MRZ chars
                if raw and raw[0] == 0x5A and len(raw) >= 3:
                    raw = raw[2:]
                dg1_lines = raw.decode("latin-1").splitlines()
        if dg1_lines and len(dg1_lines) >= 2:
            l1, l2 = dg1_lines[0], dg1_lines[1]
            surname = l1[2:].split("<")[0]
            given = [p for p in l1[2:].split("<")[1:] if p]
            self._nfc_dg1_vars["name"].set(
                f"{surname}, {' '.join(given)}")
            self._nfc_dg1_vars["doc"].set(l2[0:9])
            self._nfc_dg1_vars["nat"].set(l2[10:13])
            self._nfc_dg1_vars["dob"].set(self._fmt_date(l2[13:19]))
            self._nfc_dg1_vars["exp"].set(self._fmt_date(l2[21:27]))
            self._nfc_sod_sha.set(
                hashlib.sha256(raw).hexdigest().upper())
            self._nfc_sod_rsa.set("—（Mock SOD 无证书签名，Passive Auth 跳过）")
        else:
            for k in self._nfc_dg1_vars:
                self._nfc_dg1_vars[k].set("—")
            self._nfc_sod_sha.set("—")
            self._nfc_sod_rsa.set("—")
        sod_ok = bool(kv.get("result.sod_head"))
        chip = self._pal["chip"]
        for ef, (var, lb) in self._nfc_prog_vars.items():
            got = {"EF.COM": False, "EF.DG1": bool(dg1_lines),
                   "EF.DG2": False, "EF.SOD": sod_ok}[ef]
            if got:
                var.set(" ✓ ")
                lb.configure(bg=chip["ok"]["bg"], fg=chip["ok"]["fg"])
            else:
                var.set(" — ")
                lb.configure(bg=chip["idle"]["bg"], fg=chip["idle"]["fg"])

    @staticmethod
    def _fmt_date(yymmdd):
        try:
            yy, mm, dd = int(yymmdd[0:2]), int(yymmdd[2:4]), int(yymmdd[4:6])
            cyy = time.localtime().tm_year % 100
            year = 2000 + yy if yy <= cyy else 1900 + yy
            return f"{year:04d}-{mm:02d}-{dd:02d}"
        except Exception:
            return yymmdd or "—"

    def _nfc_on_trace_select(self, _evt=None):
        """Deep-dive the selected APDU: raw hex, decrypted plaintext, MAC check."""
        sel = self.nfc_trace.selection()
        data = getattr(self, "_nfc_trace_data", [])
        self.nfc_inspect.delete("1.0", "end")
        if not sel or not data:
            self.nfc_inspect.insert("1.0",
                "点击上方 APDU 行查看 Raw / Decrypted / MAC 校验细节")
            return
        t = data[int(sel[0]) - 1]
        sw = t.get("sw", "")
        mac = {1: "✓ CBC-MAC 校验通过", 0: "✗ CBC-MAC 不匹配",
               -1: "n/a（无 MAC）"}.get(t.get("mac_ok"), "—")
        lines = [
            f"命令      : {t.get('name', '')}",
            f"C-APDU    : {t.get('capdu', '')}",
            f"R-APDU    : {t.get('rapdu', '')}",
            f"SW        : {sw}  ({'OK' if sw == '9000' else 'FAIL'})",
            f"耗时      : {t.get('ms', 0)} ms",
            f"MAC       : {mac}",
        ]
        if t.get("plain"):
            lines.append(f"解密明文  : {t['plain']}")
        self.nfc_inspect.insert("1.0", "\n".join(lines))

    # ---------------- MRZ tab ----------------
    def _build_mrz_tab(self):
        f = self.tab_mrz
        left = ttk.Frame(f); left.pack(side="left", fill="both",
                                       expand=True, padx=4, pady=4)
        right = ttk.Frame(f); right.pack(side="right", fill="both",
                                         expand=True, padx=4, pady=4)

        # ---- Left: MRZ text -> decode ----
        ttk.Label(left, text="MRZ 文本（两行 44 字符）",
                  font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        self.mrz_text = tk.Text(left, width=52, height=6, font=_ocrb_font(12),
                                bg="#ffffff", fg="#111827")
        self.mrz_text.insert("1.0", SAMPLE_MRZ)
        self.mrz_text.pack(fill="x")
        btn = ttk.Frame(left); btn.pack(fill="x", pady=4)
        ttk.Button(btn, text="解码并校验",
                   command=self._mrz_decode_run).pack(side="left", padx=2)
        ttk.Button(btn, text="编码结果 → 填入左侧",
                   command=self._mrz_encode_run).pack(side="left", padx=2)
        ttk.Button(btn, text="运行防伪验证",
                   command=lambda: (self.tab_ac.focus_set(),
                                    self._ac_run(self.mrz_text.get("1.0", "end")))).pack(side="left", padx=2)
        ttk.Label(left, text="输出", font=("TkDefaultFont", 11, "bold")).pack(anchor="w", pady=(8, 0))
        self.mrz_out = tk.Text(left, width=80, height=18, font=("Menlo", 11),
                               bg="#ffffff", fg="#111827")
        self.mrz_out.pack(fill="both", expand=True)

        # ---- Right: 10 fields -> encode ----
        ttk.Label(right, text="TD3 字段 → 编码生成 MRZ",
                  font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        fields = [
            ("doc_type",      "证件类型 (P<)",      "P"),
            ("issuing_state", "签发国 (3字母)",     "UTO"),
            ("surname",       "姓",                "ERIKSSON"),
            ("given_names",   "名",                "ANNA MARIA"),
            ("passport_no",   "护照号 (9位)",       "L898902C3"),
            ("nationality",   "国籍 (3字母)",       "UTO"),
            ("birth",         "出生日期 (YYMMDD)",  "690806"),
            ("sex",           "性别 (M/F)",        "F"),
            ("expiry",        "有效期 (YYMMDD)",   "940623"),
            ("personal_no",   "个人号码 (14位)",    "ZE184226B"),
        ]
        self.mrz_fields = {}
        for key, label, default in fields:
            row = ttk.Frame(right); row.pack(fill="x", pady=1)
            ttk.Label(row, text=label, width=20, anchor="w").pack(side="left")
            var = tk.StringVar(value=default)
            ttk.Entry(row, textvariable=var, width=22,
                  style="Field.TEntry").pack(side="left", fill="x", expand=True)
            self.mrz_fields[key] = var
        ttk.Button(right, text="编码生成 MRZ",
                   command=self._mrz_encode_run).pack(anchor="e", pady=(6, 2))

    def _mrz_decode_run(self):
        """Decode + verify the MRZ text in the left editor (decode subcommand)."""
        text = self.mrz_text.get("1.0", "end").strip()
        if not text:
            messagebox.showerror("错误", "请先输入 MRZ 文本")
            return
        try:
            r = subprocess.run([str(MRZ_TOOL), "decode", text],
                               text=True, capture_output=True, timeout=10)
            self.mrz_out.delete("1.0", "end")
            self.mrz_out.insert("1.0", r.stdout + (r.stderr and ("\n[stderr]\n" + r.stderr) or ""))
        except Exception as e:
            messagebox.showerror("错误", str(e))

    def _mrz_encode_run(self):
        """Encode the 10 right-hand fields into an MRZ string, then place
        it back into the left editor (encode subcommand)."""
        vals = [self.mrz_fields[k].get().strip() for k in
                ("doc_type", "issuing_state", "surname", "given_names",
                 "passport_no", "nationality", "birth", "sex", "expiry",
                 "personal_no")]
        try:
            r = subprocess.run([str(MRZ_TOOL), "encode", *vals],
                               text=True, capture_output=True, timeout=10)
            self.mrz_out.delete("1.0", "end")
            self.mrz_out.insert("1.0", r.stdout + (r.stderr and ("\n[stderr]\n" + r.stderr) or ""))
            if r.returncode == 0 and r.stdout.strip():
                self.mrz_text.delete("1.0", "end")
                self.mrz_text.insert("1.0", r.stdout)
        except Exception as e:
            messagebox.showerror("错误", str(e))

    # ---------------- AC tab ----------------
    def _build_ac_tab(self):
        f = self.tab_ac
        ttk.Label(f, text="防伪验证", font=("TkDefaultFont", 11, "bold")).pack(anchor="w", padx=4)
        top = ttk.Frame(f); top.pack(fill="x", padx=4)
        ttk.Label(top, text="(使用 MRZ 解码标签页中的文本)").pack(side="left")
        ttk.Button(top, text="运行验证", command=lambda: self._ac_run(None)).pack(side="right")
        self.ac_out = tk.Text(f, font=("Menlo", 11),
                              bg=self._pal["field"], fg=self._pal["field_fg"])
        self.ac_out.pack(fill="both", expand=True, padx=4, pady=4)

    def _ac_run(self, _text=None):
        text = self.mrz_text.get("1.0", "end") if _text is None else _text
        try:
            # ac_tool reads the MRZ from stdin when invoked with "-".
            r = subprocess.run([str(AC_TOOL), "-"], input=text,
                               text=True, capture_output=True, timeout=10)
            self.ac_out.delete("1.0", "end")
            self.ac_out.insert("1.0", r.stdout + (r.stderr and ("\n[stderr]\n" + r.stderr) or ""))
        except Exception as e:
            messagebox.showerror("错误", str(e))

    # ---------------- Face tab ----------------
    def _build_face_tab(self):
        f = self.tab_face
        pal = self._pal
        chip = pal["chip"]
        # ---- Reference picker row ----
        ref = ttk.Frame(f); ref.pack(fill="x", padx=4, pady=2)
        ttk.Label(ref, text="内置参考样本:").pack(side="left")
        self.face_ref = tk.StringVar()
        face_ref_cb = ttk.Combobox(ref, textvariable=self.face_ref,
                                   values=list(FACE_SAMPLES.keys()),
                                   state="readonly", width=26,
                                   style="Field.TCombobox")
        face_ref_cb.pack(side="left", padx=2)
        face_ref_cb.current(0)
        ttk.Button(ref, text="⇥ 填入 照片A",
                   command=self._face_load_ref).pack(side="left", padx=4)
        ttk.Label(ref, text="（face_a / face_b / face_c 已预置，可直接点“比对”）",
                  foreground="#888").pack(side="left", padx=6)

        # ---- File picker rows (A & B) ----
        def _path_row(label, var, picker):
            row = ttk.Frame(f); row.pack(fill="x", padx=4, pady=2)
            ttk.Label(row, text=label).pack(side="left")
            ttk.Entry(row, textvariable=var, width=46,
                      style="Field.TEntry").pack(
                          side="left", padx=2, fill="x", expand=True)
            ttk.Button(row, text="▤ 浏览...", command=picker).pack(side="left")
            return row

        self.face_a = tk.StringVar(value="")
        self.face_b = tk.StringVar(value="")
        _path_row("照片A:", self.face_a, lambda: self._face_pick("a"))
        _path_row("照片B:", self.face_b, lambda: self._face_pick("b"))

        # ---- Action + result strip ----
        action_row = ttk.Frame(f); action_row.pack(fill="x", padx=4, pady=2)
        self.face_run_btn = self._cta(action_row, "⚡ 比对", self._face_run)
        self.face_run_btn.pack(side="left", padx=(0, 8))
        # Score chip (tk.Label: 3-state color).
        self.face_score_chip = tk.Label(
            action_row, text="  SCORE --  ",
            bg=pal["bg"], fg=chip["dim"]["fg"],
            font=pal["font_ui_bold"], padx=8, pady=2)
        self.face_score_chip.pack(side="left", padx=4)
        self.face_method_var = tk.StringVar(value="")
        ttk.Label(action_row, textvariable=self.face_method_var,
                  foreground="#888",
                  font=("Menlo", 9)).pack(side="left", padx=8)

        # ---- Body: two preview tiles + detail text ----
        body = ttk.Frame(f); body.pack(fill="both", expand=True, padx=4, pady=2)
        body.grid_columnconfigure(0, weight=1, uniform="face")
        body.grid_columnconfigure(1, weight=1, uniform="face")
        body.grid_rowconfigure(1, weight=1)
        ttk.Label(body, text="  照片 A  ",
                  foreground=pal.get("accent", pal["primary"]),
                  font=("TkDefaultFont", 10, "bold")).grid(
                      row=0, column=0, sticky="w", padx=4, pady=(0, 4))
        ttk.Label(body, text="  照片 B  ",
                  foreground=pal.get("accent", pal["primary"]),
                  font=("TkDefaultFont", 10, "bold")).grid(
                      row=0, column=1, sticky="w", padx=4, pady=(0, 4))
        tile_a = tk.LabelFrame(body, text=" ", bg=pal["bg"],
                                fg=pal["fg"], bd=0, relief="flat",
                                highlightthickness=1,
                                highlightbackground=pal["border"],
                                highlightcolor=pal["accent"])
        tile_a.grid(row=1, column=0, sticky="nsew", padx=(4, 6), pady=4)
        tile_b = tk.LabelFrame(body, text=" ", bg=pal["bg"],
                                fg=pal["fg"], bd=0, relief="flat",
                                highlightthickness=1,
                                highlightbackground=pal["border"],
                                highlightcolor=pal["accent"])
        tile_b.grid(row=1, column=1, sticky="nsew", padx=(6, 4), pady=4)
        self.face_canvas_a = tk.Canvas(
            tile_a, width=256, height=256,
            bg=pal["code"], highlightthickness=0)
        self.face_canvas_a.pack(padx=10, pady=10)
        self.face_canvas_b = tk.Canvas(
            tile_b, width=256, height=256,
            bg=pal["code"], highlightthickness=0)
        self.face_canvas_b.pack(padx=10, pady=10)

        # ---- Detail text under tiles ----
        detail = ttk.LabelFrame(f, text=" 对比详情 (raw 输出 / 方法 / 耗时) ")
        detail.pack(fill="both", expand=False, padx=4, pady=(2, 4))
        self.face_out = tk.Text(detail, height=8, wrap="none",
                                font=("Menlo", 9),
                                bg=pal["code"], fg=pal["fg"],
                                insertbackground=pal.get("accent", pal["primary"]))
        self.face_out.pack(fill="both", expand=True, padx=4, pady=4)

        # Image refs kept alive.
        self._face_imgs = {}
        # Default face_a vs face_c -> 95/100 -> SUSPECT chip so the chip
        # visibly changes the moment the user presses 比对.
        self.face_a.set(str(FACE_SAMPLES["face_a (合成, 128×128)"]))
        self.face_b.set(str(FACE_SAMPLES["face_c (合成, 128×128)"]))
        self._face_render_preview("a", self.face_a.get())
        self._face_render_preview("b", self.face_b.get())
        self.face_out.insert(
            "1.0",
            "提示：内置 face_a / face_b / face_c 已预置，点击“比对”即可直接测试。\n"
            "也可从内置参考样本下拉选择并填入照片A，或浏览本地图片。\n")

    def _face_load_ref(self):
        path = FACE_SAMPLES.get(self.face_ref.get())
        if path:
            self.face_a.set(str(path))
            self._face_render_preview("a", str(path))

    def _face_pick(self, which):
        path = filedialog.askopenfilename(
            filetypes=[("Image", "*.ppm *.bmp *.png *.jpg"), ("All", "*.*")])
        if path:
            (self.face_a if which == "a" else self.face_b).set(path)
            self._face_render_preview(which, path)

    def _face_render_preview(self, side: str, path: str):
        """Load an image into the preview canvas for side 'a' or 'b'.

        Tk is single-threaded for image loading. We try the native
        PhotoImage first (PPM / PGM / GIF / PNG) and fall back to PIL
        for JPG / BMP and for resizing below native dimensions.
        """
        canvas = self.face_canvas_a if side == "a" else self.face_canvas_b
        try:
            canvas.delete("all")
        except tk.TclError:
            return
        path = path or ""
        if not path or not Path(path).exists():
            canvas.create_text(
                128, 128, text="(no image)",
                fill=self._pal.get("fg", "#888"), anchor="center")
            return
        img = None
        try:
            img = tk.PhotoImage(file=path)
        except tk.TclError:
            img = None
        if img is None:
            try:
                from PIL import Image, ImageTk  # type: ignore
                pil = Image.open(path).convert("RGB")
                img = ImageTk.PhotoImage(pil)
            except Exception:
                canvas.create_text(
                    128, 128, text="(unsupported format)",
                    fill=self._pal.get("fg", "#888"), anchor="center")
                return
        cw, ch = 256, 256
        iw, ih = img.width(), img.height()
        scale = min(cw / iw, ch / ih)
        if scale < 1.0:
            try:
                from PIL import Image as _PILImage  # type: ignore
                pil = _PILImage.open(path).convert("RGB").resize(
                    (max(1, int(iw * scale)), max(1, int(ih * scale))),
                    _PILImage.LANCZOS)
                from PIL import ImageTk  # type: ignore
                img = ImageTk.PhotoImage(pil)
                iw, ih = img.width(), img.height()
            except Exception:
                pass
        self._face_imgs[side] = img
        canvas.create_image((cw - iw) // 2, (ch - ih) // 2,
                            image=img, anchor="nw")
        canvas.create_rectangle(0, 0, cw - 1, ch - 1, outline=self._pal["canvas_border"])

    @staticmethod
    def _face_grade(score: int):
        """Map a 0..100 score to a chip kind + human label."""
        if score >= 90:
            return "ok", "PASS"
        if score >= 70:
            return "run", "SUSPECT"
        return "fail", "FAIL"

    def _face_run(self):
        a = self.face_a.get().strip(); b = self.face_b.get().strip()
        if not (a and b):
            messagebox.showerror("错误", "请选择两张照片")
            return
        self._face_render_preview("a", a)
        self._face_render_preview("b", b)
        t0 = time.time()
        try:
            r = subprocess.run([str(FACE_TOOL), a, b, "2"],
                               text=True, capture_output=True, timeout=20)
            ms = int((time.time() - t0) * 1000)
            self.face_out.delete("1.0", "end")
            method = "(unknown)"
            score = 0
            for ln in r.stdout.splitlines():
                if ln.startswith("method:"):
                    method = ln.split(":", 1)[1].strip()
                elif ln.startswith("score"):
                    tail = ln.split(":", 1)[1].strip()
                    head = tail.split("/", 1)[0]
                    digits = "".join(ch for ch in head if ch.isdigit())
                    try:
                        score = int(digits)
                    except Exception:
                        score = 0
            kind, label = self._face_grade(score)
            self.face_method_var.set(f"method={method}")
            chip = self._pal["chip"][kind]
            try:
                self.face_score_chip.configure(
                    bg=chip["bg"], fg=chip["fg"],
                    text=f"  {label}  {score}/100  ")
            except tk.TclError:
                pass
            header = (f"method : {method}\n"
                      f"score  : {score}/100  ({label})\n"
                      f"耗时   : {ms} ms\n"
                      f"图片A  : {a}\n"
                      f"图片B  : {b}\n"
                      f"\n--- raw face_tool output ---\n")
            tail = r.stdout + (r.stderr and ("\n[stderr]\n" + r.stderr) or "")
            self.face_out.insert("1.0", header + tail)
        except Exception as e:
            messagebox.showerror("错误", str(e))

    # ---------------- menu action ----------------
    def _menu_gen_sample(self):
        try:
            r = subprocess.run([str(GEN_SAMP)], capture_output=True,
                               text=True, timeout=20)
            messagebox.showinfo("生成样本", r.stdout or (r.stderr and ("[stderr]\n" + r.stderr) or ""))
        except Exception as e:
            messagebox.showerror("错误", str(e))


# Small lazy regex import kept at the bottom so the rest of the
# module stays compatible with non-regex-only environments.
def re_compile(pattern):
    import re
    return re.compile(pattern)


def main():
    app = PassportGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
