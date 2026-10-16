"""passport_photo_restore — 护照机拍照图：固定矫正结构先验 + 去倒影算法（独立自包含）。

仅依赖：OpenCV + NumPy。不依赖仓库 tests/ 或 6_mrz_ocr。

核心函数：
  load(rgb)           降采样到 max_side 工作图（默认 1250px 宽）
  detect_structure(g) 结构先验检测：
                       - solid 黑带（护照机固定取景的黑色横条，y~725-770）
                       - MRZ 两行（solid 下沿之后的连续 text 段）
                       - 倒影分界（gap>15px 判定进入倒影区）
  crop_mrz(rgb, st)   从定位结果裁出 MRZ 两行（天然去除底部倒影）
  rectify(work, st)   四点单应矫正：MRZ 两行四边形 -> 标准 44格x2行 矩形
  restore(f)          端到端：load -> detect -> crop -> (可选)rectify
"""
from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np

STD_W, STD_H = 1250, 875
GLYPH_W, GLYPH_H = 14, 48
TGT_W, TGT_H = 44 * GLYPH_W, 2 * GLYPH_H + 8


def load(rgb, max_side=1250):
    """保持纵横比缩放到 max_side 内（宽优先）。返回 (work, scale)。"""
    h, w = rgb.shape[:2]
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        return cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR), scale
    return rgb, scale


def _gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _run_groups(mask, min_len=5):
    """布尔数组 -> 连续 True 区间 [(start,end), ...]"""
    out, s = [], None
    n = len(mask)
    for i in range(n):
        if mask[i] and s is None:
            s = i
        elif not mask[i] and s is not None:
            if i - s >= min_len:
                out.append((s, i - 1))
            s = None
    if s is not None and n - s >= min_len:
        out.append((s, n - 1))
    return out


def detect_structure(g, gap_thr=15):
    """固定矫正结构先验检测。

    - solid 带：row_ink>0.9*W 的深色横带（护照机固定取景的黑色横条）。
    - MRZ 带：solid 下沿之后连续 text 段（gap<=gap_thr），倒影在 gap 更大的
      下一段 —— 由此天然切除倒影。
    返回 {'solid': [...], 'text': [...], 'mrz': (y0,y1) 或 None}
    """
    H, W = g.shape
    row_ink = (g < 180).sum(axis=1)
    gx = np.abs(np.diff(g.astype(np.int16), axis=1))
    gx = np.pad(gx, ((0, 0), (1, 0)))
    E = gx.sum(axis=1)
    solid = row_ink > 0.9 * W
    text = (row_ink > 0.2 * W) & (row_ink < 0.9 * W) & (E > 1800)
    solid_b = _run_groups(solid, min_len=5)
    text_b = _run_groups(text, min_len=4)

    mrz = None
    cand = [b for b in solid_b
            if int(H * 0.3) <= b[0] <= int(H * 0.9) and b[1] - b[0] >= 10]
    if cand:
        s_top = cand[-1][1] + 1
        below = [t for t in text_b if t[0] > s_top]
        if below:
            start = below[0][0]
            end = below[0][1]
            for t in below[1:]:
                if t[0] - end - 1 > gap_thr:
                    break  # 进入倒影区
                end = t[1]
            if end - start + 1 >= 10:
                mrz = (start, end)
    return {"solid": solid_b, "text": text_b, "mrz": mrz}


def _mrzh_quad(g, y0, y1, min_ink_rows=2):
    """MRZ 带四端点（左右边界取列投影中 >=min_ink_rows 行的列）。"""
    band = (g[y0:y1 + 1] < 180).astype(np.uint8)
    col = band.sum(axis=0)
    xs = np.where(col >= min_ink_rows)[0]
    if len(xs) == 0:
        return None
    xl = max(0, int(xs[0]) - 2)
    xr = min(g.shape[1] - 1, int(xs[-1]) + 2)
    return np.array([[xl, y0], [xr, y0], [xr, y1], [xl, y1]], np.float32)


def crop_mrz(work_rgb, st, pad=0.06):
    """裁出 MRZ 两行（不含倒影）。返回 numpy RGB crop；无 MRZ 返回 None。"""
    mrz = st.get("mrz")
    if not mrz:
        return None
    y0, y1 = mrz
    pad_px = max(3, int((y1 - y0) * pad))
    cy0 = max(0, y0 - pad_px)
    cy1 = min(work_rgb.shape[0] - 1, y1 + pad_px)
    return work_rgb[cy0:cy1 + 1]


def rectify(work_rgb, st, tgt_w=TGT_W, tgt_h=TGT_H):
    """单应矫正：MRZ 带四边形 -> 标准 44格x2行 矩形。返回 RGB 图或 None。"""
    mrz = st.get("mrz")
    if not mrz:
        return None
    g = _gray(work_rgb)
    y0, y1 = mrz
    quad = _mrzh_quad(g, y0, y1)
    if quad is None:
        return None
    dst = np.array([[0, 0], [tgt_w - 1, 0], [tgt_w - 1, tgt_h - 1],
                    [0, tgt_h - 1]], np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(work_rgb, M, (tgt_w, tgt_h),
                               flags=cv2.INTER_LANCZOS4)


def restore(img_path, max_side=1250, do_rectify=True):
    """端到端：读图 -> 降采样 -> 结构检测 -> crop/矫正。

    返回 dict：{ok, mrz, crop, rect, solid, ms}；crop/rect 为 numpy 图。
    """
    import time
    rgb = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    t0 = time.perf_counter()
    work, scale = load(rgb, max_side=max_side)
    g = _gray(work)
    st = detect_structure(g)
    ms = (time.perf_counter() - t0) * 1000
    crop = crop_mrz(work, st) if st.get("mrz") else None
    rect = rectify(work, st) if (do_rectify and st.get("mrz")) else None
    return {
        "ok": st.get("mrz") is not None,
        "mrz": st.get("mrz"),
        "solid": st.get("solid"),
        "crop": crop,
        "rect": rect,
        "scale": scale,
        "ms": ms,
    }
