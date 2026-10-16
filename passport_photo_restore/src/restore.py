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
    """固定矫正结构先验检测 + 反射识别.

    反射识别 (更激进):
      - line2 下方: 从 line2.y1+1 到图底, 任何连续 ink>180 行 (含反射过渡区) 都视为反射
      - line1 上方: 同样规则
      - 反射段总高包括 line2 字符衰减区 (line2.y1 ~ 反射起点)
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
            prev_h = end - start + 1
            for t in below[1:]:
                gap = t[0] - end - 1
                if gap > max(gap_thr, prev_h * 1.5):
                    break
                end = t[1]
                prev_h = t[1] - t[0] + 1
            if end - start + 1 >= 10:
                mrz = (start, end)

    pair = find_mrz_pair(g)
    reflection = []
    if pair:
        (l1y0, l1y1), (l2y0, l2y1) = pair
        # line2 下方反射段: 从 line2.y1+1 起, 连续 ink>180 行 (含反射过渡)
        refl_below_start = None
        for y in range(l2y1 + 1, H):
            if row_ink[y] > 180:
                refl_below_start = y
                break
        if refl_below_start is not None:
            # 找到连续 ink>180 行的终点
            end = refl_below_start
            for y in range(refl_below_start + 1, H):
                if row_ink[y] > 180:
                    end = y
                else:
                    # 允许最多 5 行空白间断 (反射字符之间的间隙)
                    if y + 5 < H and all(row_ink[y2] <= 180 for y2 in range(y, min(y + 5, H))):
                        break
            reflection.append((refl_below_start, end))
        # line1 上方反射段 (作为保留裁切的边界): line1.y0 之前连续 ink>180 行
        refl_above_end = None  # 最深的反射终点
        for y in range(l1y0 - 1, -1, -1):
            if row_ink[y] > 180:
                refl_above_end = y
                break
        if refl_above_end is not None:
            # 反射段起点: 找最早的反射起点 (从 refl_above_end 往上)
            start = refl_above_end
            for y in range(refl_above_end - 1, -1, -1):
                if row_ink[y] > 180:
                    start = y
                else:
                    if y - 5 >= 0 and all(row_ink[y2] <= 180 for y2 in range(max(0, y - 5), y)):
                        break
            reflection.append((start, refl_above_end))
    return {
        "solid": solid_b,
        "text": text_b,
        "mrz": mrz,
        "mrz_pair": pair,
        "reflection": reflection,
    }



def find_mrz_pair(g):
    """严格 MRZ 双行定位：找行高近似、间距合理的相邻 text-run pair。

    反射倒影特征：连续 text_run 长度通常 > 2.5x line1 行高，且与 line2 之间
    有可识别 gap。本函数天然剔除之（只接受相邻 + 行高比 0.4~1.6 的对子）。
    返回 (line1, line2) 两个 (y0, y1)；找不到返回 None。
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
    cand = [b for b in solid_b
            if int(H * 0.3) <= b[0] <= int(H * 0.9) and b[1] - b[0] >= 10]
    if not cand:
        return None
    s_top = cand[-1][1] + 1
    below = [t for t in text_b if t[0] > s_top]
    if not below:
        return None
    best = None
    for i, t1 in enumerate(below[:-1]):
        h1 = t1[1] - t1[0] + 1
        if h1 < 4:
            continue
        for t2 in below[i + 1:]:
            h2 = t2[1] - t2[0] + 1
            gap = t2[0] - t1[1] - 1
            if (0.4 <= h2 / h1 <= 1.6) and (0 <= gap <= max(8, int(3.0 * h1))):
                score = abs(h1 - h2) + gap
                if best is None or score < best[0]:
                    best = (score, t1, t2)
                break
    if best is None:
        return None
    return best[1], best[2]


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


def drop_reflection(work_rgb, st, mode="crop", force_min_drop=150):
    """按你的指示: 反射段总高 = MRZ + 反射延伸, 应接近 force_min_drop=150.

    策略: 用 detect_structure 已识别的 reflection 字段, 区分 line1 上方 vs line2 下方.
    保留 MRZ, 裁掉两段反射.
    """
    reflection = st.get("reflection", []) or []
    pair = st.get("mrz_pair")

    if mode == "fill":
        keep = work_rgb.copy()
        for (y0, y1) in reflection:
            band = keep[y0:y1 + 1]
            if band.size > 0:
                flat = band.reshape(-1, 3)
                med = np.median(flat, axis=0).astype(np.uint8)
                keep[y0:y1 + 1] = med
        st2 = dict(st); st2["reflection"] = []
        return keep, st2, {"intended_drop": force_min_drop, "actual_drop": 0,
                            "crop_top": keep.shape[0], "mode": "fill"}

    H = work_rgb.shape[0]
    if not pair:
        if not reflection:
            return work_rgb, st, {"intended_drop": force_min_drop, "actual_drop": 0,
                                    "crop_top": H, "mode": "noop"}
        crop_top = min(t[0] for t in reflection)
        keep = work_rgb[:max(0, crop_top)]
        st2 = dict(st); st2["reflection"] = []
        return keep, st2, {"intended_drop": force_min_drop,
                            "actual_drop": H - keep.shape[0],
                            "crop_top": crop_top, "mode": "crop"}

    (l1y0, l1y1), (l2y0, l2y1) = pair

    # 用 reflection 字段:
    # above = y1 < l1y0 的反射段 (line1 上方)
    # below = y0 > l2y1 的反射段 (line2 下方)
    above = [t for t in reflection if t[1] < l1y0]
    below = [t for t in reflection if t[0] > l2y1]

    # 反射总高
    above_h = sum(t[1] - t[0] + 1 for t in above)
    below_h = sum(t[1] - t[0] + 1 for t in below)
    refl_total = above_h + below_h

    # 若反射总高 < force_min_drop, 把 above 起点往上挪 (裁更多)
    if refl_total < force_min_drop:
        extra = force_min_drop - refl_total
        # above 最早起点往上挪 extra
        if above:
            new_above_start = min(t[0] for t in above) - extra
            # 但要确保不裁掉 line1.y0 之前的非反射区太多
            # 安全下限 = l1y0 - 30 (line1 上方最多 30px 反射延伸)
            new_above_start = max(new_above_start, l1y0 - 30)
        else:
            new_above_start = max(0, l1y0 - extra)
    else:
        new_above_start = min(t[0] for t in above) if above else l1y0

    # 裁掉 [new_above_start, l1y0) (line1 上方反射) + (l2y1, H) (line2 下方反射)
    # 但 (l2y1, H) 不一定全是反射, 我们只裁 below 范围内的反射段
    # 简化: 裁掉 below 段每个 + above 起点到 l1y0
    parts = []
    # part 1: 保留 [0, new_above_start)
    if new_above_start > 0:
        parts.append(work_rgb[:new_above_start])
    # part 2: 保留 [l1y0, l2y1+1] (MRZ)
    parts.append(work_rgb[l1y0:l2y1 + 1])
    # part 3: 保留 [below_end+1, H) (裁掉 below 反射段)
    if below:
        max_below_end = max(t[1] for t in below)
        if max_below_end + 1 < H:
            parts.append(work_rgb[max_below_end + 1:])
    keep = np.vstack(parts) if len(parts) > 1 else parts[0]

    # 重新校准 mrz_pair
    new_l1y0 = (work_rgb[:new_above_start].shape[0] if new_above_start > 0 else 0)
    new_l1y1 = new_l1y0 + (l1y1 - l1y0)
    new_l2y0 = new_l1y1 + 1
    new_l2y1 = new_l2y0 + (l2y1 - l2y0)

    st2 = dict(st)
    st2["mrz_pair"] = ((new_l1y0, new_l1y1), (new_l2y0, new_l2y1))
    st2["reflection"] = []
    actual_drop = H - keep.shape[0]
    return keep, st2, {"intended_drop": force_min_drop,
                        "actual_drop": actual_drop,
                        "crop_top": new_above_start, "mode": "crop"}



def clean_band_reflection(band_rgb, line1, line2, clear_rows=2,
                            unsharp=True, sharp_amount=1.5, blur_sigma=2.0):
    """清除 band 内反射残留.

    三层策略:
    1) clear_rows: 清 line1.y1+1 起的 N 行 (line1 镜面反射最强区, 反射随距离衰减)
    2) unsharp=True: USM 锐化 (抑制反射模糊层, 增强 line1/line2 锐利字符)
       - sharp_amount 越大, 字符边缘越强 (反射模糊被抑制)
       - 反射是"虚像", 边缘比真实字符模糊, USM 选择性增强锐利字符
    3) clear_rows 也清掉 line2 下方反射残留的最后 N 行 (line2 镜面)
    """
    import cv2 as _cv2
    import numpy as _np
    H = band_rgb.shape[0]
    clean = band_rgb.copy()
    # 1) 清 line1 紧邻的反射 (clear_rows 行)
    l1y0, l1y1 = line1
    for y in range(l1y1 + 1, min(l1y1 + clear_rows, H - 1) + 1):
        if 0 <= y < H:
            clean[y] = 255
    # 2) 清 line2 下方反射的最后 N 行
    l2y0, l2y1 = line2
    for y in range(max(l2y1 + 1, l2y1 - clear_rows + 1), l2y1 + 1):
        if 0 <= y < H:
            clean[y] = 255
    # 3) USM 锐化 (抑制反射模糊层)
    if unsharp:
        blur = _cv2.GaussianBlur(clean, (0, 0), blur_sigma)
        clean = _cv2.addWeighted(clean, sharp_amount, blur,
                                 -(sharp_amount - 1.0), 0)
    return clean


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


def crop_pair(work_rgb, st, pad=0.06):
    """严格双行 crop: 基于 mrz_pair (line1, line2) 框出最小矩形，不拉伸。

    与 crop_mrz 的区别: crop_mrz 用旧 mrz 字段(含倒影), 本函数用 mrz_pair(剔除倒影)。
    返回 numpy RGB crop; 无 pair 返回 None。
    """
    pair = st.get("mrz_pair")
    if not pair:
        return None
    (l1y0, _), (l2y0, l2y1) = pair
    y0, y1 = l1y0, l2y1
    pad_px = max(3, int((y1 - y0) * pad))
    cy0 = max(0, y0 - pad_px)
    cy1 = min(work_rgb.shape[0] - 1, y1 + pad_px)
    return work_rgb[cy0:cy1 + 1]


def crop_pair_clean(work_rgb, st, pad=0.06, clear_rows=2, unsharp=True):
    """crop_pair + clean_band_reflection(USM): 抑制 band 内反射模糊层.

    clear_rows: 涂白 line1 后/line2 后的多少行 (默认 2, 反射最强区)
    unsharp: 是否应用 USM 锐化抑制反射 (默认 True)
    """
    pair = st.get("mrz_pair")
    if not pair:
        return None, None
    (l1y0, l1y1), (l2y0, l2y1) = pair
    y0, y1 = l1y0, l2y1
    pad_px = max(3, int((y1 - y0) * pad))
    cy0 = max(0, y0 - pad_px)
    cy1 = min(work_rgb.shape[0] - 1, y1 + pad_px)
    band = work_rgb[cy0:cy1 + 1]
    l1_in_band = (l1y0 - cy0, l1y1 - cy0)
    l2_in_band = (l2y0 - cy0, l2y1 - cy0)
    clean = clean_band_reflection(band, l1_in_band, l2_in_band,
                                  clear_rows=clear_rows, unsharp=unsharp)
    return clean, {"band_in_work": (cy0, cy1), "l1_in_band": l1_in_band,
                   "l2_in_band": l2_in_band}


def rectify_pair(work_rgb, st, tgt_w=TGT_W, tgt_h=TGT_H):
    """严格双行单应矫正：line1/line2 各自四点 -> 矩形上下两半，分别 warp 后拼接。

    反射倒影下方的连续 text 段不再被纳入 MRZ 带。每行字符严格匹配训练分布
    (GLYPH_W x GLYPH_H)。返回 (rgb, info) 或 (None, None)。
    """
    pair = st.get("mrz_pair")
    if not pair:
        return None, None
    g = _gray(work_rgb)
    (l1y0, l1y1), (l2y0, l2y1) = pair
    q1 = _mrzh_quad(g, l1y0, l1y1)
    q2 = _mrzh_quad(g, l2y0, l2y1)
    if q1 is None or q2 is None:
        return None, None
    # 目标两半
    dst1 = np.array([[0, 0], [tgt_w - 1, 0],
                     [tgt_w - 1, GLYPH_H - 1], [0, GLYPH_H - 1]], np.float32)
    dst2 = np.array([[0, 0], [tgt_w - 1, 0],
                     [tgt_w - 1, GLYPH_H - 1], [0, GLYPH_H - 1]], np.float32)
    M1 = cv2.getPerspectiveTransform(q1, dst1)
    M2 = cv2.getPerspectiveTransform(q2, dst2)
    line1 = cv2.warpPerspective(work_rgb, M1, (tgt_w, GLYPH_H),
                                flags=cv2.INTER_LANCZOS4)
    line2 = cv2.warpPerspective(work_rgb, M2, (tgt_w, GLYPH_H),
                                flags=cv2.INTER_LANCZOS4)
    # 紧贴输出 (与 CNN 工具期望 band 连续两行的分布对齐)
    rgb = np.zeros((tgt_h, tgt_w, 3), dtype=np.uint8)
    rgb[0:GLYPH_H] = line1
    rgb[GLYPH_H:GLYPH_H * 2] = line2
    info = {
        "line1_y_in_dst": (0, GLYPH_H - 1),
        "line2_y_in_dst": (GLYPH_H, tgt_h - 1),
        "src_quad_l1": q1.tolist(),
        "src_quad_l2": q2.tolist(),
    }
    return rgb, info


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
