"""passport_photo_restore — 护照机拍照图：固定矫正结构先验 + 去倒影算法（独立自包含）。

仅依赖：OpenCV + NumPy。不依赖仓库 tests/ 或 6_mrz_ocr。

核心函数：
  load(rgb)           降采样到 max_side 工作图（默认 1250px 宽）
  detect_structure(g) 结构先验检测：
                       - solid 黑带（护照机固定取景的黑色横条，y~725-770）
                       - MRZ 两行（最后一条 solid 黑带上方的连续 text 段）
                       - 倒影分界（gap>15px 判定进入倒影区）
  crop_pair(rgb, st)  从定位结果裁出严格 MRZ 双行（天然去除底部倒影）
  rectify_pair(work, st)
                       两行分别单应矫正 -> 标准 44格x2行矩形
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

    反射识别:
      - 仅 line2 下方: 从 line2.y1+1 到图底, 连续 ink>180 行 (含反射过渡区) 视为倒影
      - line1 上方是护照页底部(照片/数据区), 属真实内容, 不视为反射
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
    pair = find_mrz_pair(g)
    mrz = (pair[0][0], pair[1][1]) if pair else None
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
    return {
        "solid": solid_b,
        "text": text_b,
        "mrz": mrz,
        "mrz_pair": pair,
        "reflection": reflection,
    }



def find_mrz_pair(g):
    """真实 MRZ 双行定位：从最后一条取景黑带上方的 text 段中找相邻 pair。

    布局: [0, l1y0) = 黑色空白 + 完整梯形护照(照片/数据区),
    [l1y0, l2y1] = 真实 MRZ 两行, 取景黑带 + 镜片倒影在其下方.
    旧实现从黑带下方找, 会把 line1 的镜像倒影当成 line2. 因此优先在黑带
    上方找行高近似、间距合理的相邻 text-run pair; 找不到再回退旧逻辑.
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

    def line_shape(t):
        """Reject page edges and borders that look like dense text runs."""
        y0, y1 = t
        h = y1 - y0 + 1
        if h < 5 or h > 48:
            return None
        band = g[y0:y1 + 1] < 180
        col_ink = band.sum(axis=0)
        strong = col_ink >= 4
        groups = _run_groups(strong, min_len=1)
        max_group = max((b - a + 1 for a, b in groups), default=0)
        density = float(strong.mean())
        # A real MRZ row has many separated glyph strokes.  The lower
        # passport edge is almost continuous and has very high density.
        if density > 0.90 or max_group > max(160, int(W * 0.45)):
            return None
        return density, len(groups), max_group

    def pick(runs):
        candidates = []
        for i, t1 in enumerate(runs[:-1]):
            h1 = t1[1] - t1[0] + 1
            if h1 < 4:
                continue
            if line_shape(t1) is None:
                continue
            t2 = runs[i + 1]
            h2 = t2[1] - t2[0] + 1
            gap = t2[0] - t1[1] - 1
            if line_shape(t2) is None:
                continue
            if (0.4 <= h2 / h1 <= 1.6) and (0 <= gap <= max(8, int(3.0 * h1))):
                # Prefer the pair with the most stable, largest glyphs.
                # The second term breaks the small-photo edge case where
                # the lower page border forms a short pseudo-row.
                rank = (-min(h1, h2), abs(h1 - h2) + gap, gap)
                candidates.append((rank, t1, t2))
        if not candidates:
            return None
        _, t1, t2 = min(candidates, key=lambda item: item[0])
        return t1, t2

    # 黑带上方: 枚举相邻 text 段，排除护照底边/边框伪文字。
    above = [t for t in text_b if t[1] < s_top - 1]
    if len(above) >= 2:
        res = pick(above)
        if res:
            return res
    # 回退: 旧逻辑 (黑带下方, 仅用于无真实 MRZ 的布局)
    below = [t for t in text_b if t[0] > s_top]
    if below:
        res = pick(below)
        if res:
            return res
    return None


def _mrzh_quad(g, y0, y1, min_ink_rows=2):
    """MRZ 行四端点，优先使用护照页面边界而不是黑色画面边缘.

    MRZ 所在的页面是亮色梯形，照片区域外的固定取景框则是黑色。
    直接对“墨点列投影”取首尾会把黑色取景框当成 MRZ 左右边界，
    造成矫正后字符被压缩并倾斜。这里对行附近的亮度做横向闭运算，
    得到每个 y 的页面亮区边界，再用上下边界组成四边形。
    """
    H, W = g.shape

    def page_span(y):
        lo = max(0, y - 4)
        hi = min(H, y + 5)
        col = np.median(g[lo:hi], axis=0)
        threshold = max(70.0, float(np.percentile(col, 20)))
        mask = (col > threshold).astype(np.uint8)
        kernel = np.ones(31, np.uint8)
        mask = cv2.morphologyEx(mask[None, :], cv2.MORPH_CLOSE, kernel)[0]
        runs = _run_groups(mask.astype(bool), min_len=max(20, W // 50))
        if not runs:
            return None
        center = W // 2
        return max(runs, key=lambda r: (
            r[0] <= center <= r[1],
            r[1] - r[0] + 1,
        ))

    top = page_span(y0)
    bottom = page_span(y1)
    if top and bottom:
        xl0, xr0 = top
        xl1, xr1 = bottom
        if min(xr0 - xl0, xr1 - xl1) >= max(80, W // 4):
            return np.array(
                [[xl0, y0], [xr0, y0], [xr1, y1], [xl1, y1]],
                np.float32,
            )

    # 兼容亮度不足或反色输入：保留原来的墨点投影作为退化路径。
    band = (g[y0:y1 + 1] < 180).astype(np.uint8)
    col = band.sum(axis=0)
    xs = np.where(col >= min_ink_rows)[0]
    if len(xs) == 0:
        return None
    xl = max(0, int(xs[0]) - 2)
    xr = min(W - 1, int(xs[-1]) + 2)
    return np.array([[xl, y0], [xr, y0], [xr, y1], [xl, y1]], np.float32)


def drop_reflection(work_rgb, st, mode="crop", force_min_drop=150):
    """去镜片倒影: 保留 [0, line2.y1+1] 连续区域 (黑色空白 + 完整梯形护照 + MRZ),
    只裁掉 line2 下方整段倒影; line1 上方是护照页真实内容, 不裁.
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

    # 真实布局: [0, l1y0) = 黑色空白 + 完整梯形护照(照片/数据区),
    # [l1y0, l2y1] = MRZ 两行, (l2y1, H) = 镜片倒影.
    # line1 上方并非反射 (那只是护照页底部黑带区), 不裁.
    # 只裁掉 line2 下方的倒影段, 保留 [0, l2y1+1] 连续区域.
    # MRZ 行 2 之后的内容全部属于页面下方的边界/镜片倒影。
    # 只保留到真实 MRZ 的最后一行，确保模糊 MRZ 和数据倒影不再进入后续算法。
    crop_bottom = min(H, l2y1 + 1)
    keep = work_rgb[:crop_bottom]

    # 坐标不变 (top 未裁), 只需清除 reflection 记录
    new_l1y0, new_l1y1 = l1y0, l1y1
    new_l2y0, new_l2y1 = l2y0, l2y1

    st2 = dict(st)
    st2["mrz_pair"] = ((new_l1y0, new_l1y1), (new_l2y0, new_l2y1))
    st2["reflection"] = []
    actual_drop = H - keep.shape[0]
    return keep, st2, {"intended_drop": force_min_drop,
                        "actual_drop": actual_drop, "crop_bottom": crop_bottom,
                        "crop_top": 0, "mode": "crop"}



def clean_band_reflection(band_rgb, line1, line2, clear_rows=2,
                            unsharp=True, sharp_amount=1.5, blur_sigma=2.0):
    """清除 band 内残留的镜片反射过渡行.

    1) 清 line1 紧邻的 N 行 (line1 下方反射过渡最强区, 反射随距离衰减)
    2) 清 line2 紧邻的 N 行 (line2 下方倒影过渡)
    3) unsharp USM 锐化: 抑制反射模糊层, 增强锐利的字符边缘
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
    # 2) 清 line2 下方反射的最强 N 行
    l2y0, l2y1 = line2
    for y in range(l2y1 + 1, min(l2y1 + clear_rows + 1, H)):
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
    rgb = np.full((tgt_h, tgt_w, 3), 255, dtype=np.uint8)
    rgb[0:GLYPH_H] = line1
    rgb[GLYPH_H:GLYPH_H * 2] = line2
    info = {
        "line1_y_in_dst": (0, GLYPH_H - 1),
        "line2_y_in_dst": (GLYPH_H, GLYPH_H * 2 - 1),
        "separator_y_in_dst": (GLYPH_H * 2, tgt_h - 1),
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
    crop = crop_pair(work, st) if st.get("mrz_pair") else None
    rect = rectify_pair(work, st)[0] if (do_rectify and st.get("mrz_pair")) else None
    return {
        "ok": st.get("mrz") is not None,
        "mrz": st.get("mrz"),
        "mrz_pair": st.get("mrz_pair"),
        "solid": st.get("solid"),
        "crop": crop,
        "rect": rect,
        "scale": scale,
        "ms": ms,
    }
