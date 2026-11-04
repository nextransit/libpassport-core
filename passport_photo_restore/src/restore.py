"""passport_photo_restore — 护照机拍照图：固定矫正结构先验 + 去倒影算法（独立自包含）。

仅依赖：OpenCV + NumPy。不依赖仓库 tests/ 或 6_mrz_ocr。

核心函数：
  load(rgb)           降采样到 max_side 工作图（默认 1250px 宽）
  detect_structure(g) 结构先验检测：
                       - solid 黑带（护照机固定取景的黑色横条，y~725-770）
                       - MRZ 两行（最后一条 solid 黑带上方的连续 text 段）
                       - 倒影行段（相对行墨量判定，仅 line2 下方）
  crop_pair(rgb, st)  从定位结果裁出严格 MRZ 双行（天然去除底部倒影）
  crop_pair_clean(work, st)
                       上面那条的加强版：擦掉 line1/line2 紧邻的反射行。
                       **这是识别真正吃的那条带**（OCR 输入）。
  rectify_pair(work, st)
                       两行分别单应矫正 -> 标准 44格x2行矩形。
                       展示/调研用，不是识别路径——见它的 docstring。
  restore(f)          端到端：load -> detect -> crop -> (可选)rectify
"""
from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np

STD_W, STD_H = 1250, 875
GLYPH_W, GLYPH_H = 14, 48
TGT_W, TGT_H = 44 * GLYPH_W, 2 * GLYPH_H + 8


def remove_guilloche(img):
    """Remove guilloche security-printing background from CS/CD MRZ band.

    Uses morphological black-hat transform (closing - image) with an
    elliptical kernel sized between text strokes (1-2px) and character
    width (8px). The black-hat extracts dark text while suppressing
    the larger-scale guilloche pattern.

    Accepts grayscale (H,W) or RGB (H,W,3) uint8. Returns same format.
    """
    if img.ndim == 3:
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        g = img
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 3))
    blackhat = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, kernel)
    blackhat = np.clip(blackhat.astype(np.float32) * 1.8, 0, 255).astype(np.uint8)
    if img.ndim == 3:
        return cv2.cvtColor(blackhat, cv2.COLOR_GRAY2RGB)
    return blackhat


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


def _reflection_segments(row_ink, width, pair):
    """line2 下方镜片倒影行段（相对墨量判定）。

    实拍图中倒影经镜片模糊且随距离衰减，墨量远低于清晰 MRZ（photo3-6 实测
    约 100-130px，而 line2 墨量中位数约 560px）；旧固定阈值（row_ink>180）
    只能抓到紧邻过渡行。这里以 line2 自身墨量中位数的比例作阈值，随照片的
    实际对比度自适应。
    """
    if not pair:
        return []
    (_, _), (l2y0, l2y1) = pair
    l2_ink = row_ink[l2y0:l2y1 + 1].astype(np.float64)
    thr = max(0.12 * float(np.median(l2_ink)), 0.02 * width)
    mask = row_ink[l2y1 + 1:] > thr
    return [(l2y1 + 1 + s, l2y1 + 1 + e)
            for s, e in _run_groups(mask, min_len=1)]


def detect_structure(g, gap_thr=15):
    """固定矫正结构先验检测 + 反射识别.

    反射识别:
      - 仅 line2 下方: 用相对墨量阈值 (line2 墨量中位数比例) 检出倒影行段
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
    single = find_mrz_single(g) if not pair else None
    reflection = _reflection_segments(row_ink, W, pair)
    return {
        "solid": solid_b,
        "text": text_b,
        "mrz": mrz,
        "mrz_pair": pair,
        "mrz_single": single,
        "reflection": reflection,
        "row_ink": row_ink,
    }



def _mrz_pair_pass(g, thr, row_ink180, E):
    """单阈值 MRZ 双行定位 pass（solid 判定恒用 180 阈值）。

    text 行掩码与 line_shape 用 thr 二值化; 返回 (line1, line2) 或 None。
    """
    H, W = g.shape
    row_ink_s = (g < thr).sum(axis=1)
    solid = row_ink180 > 0.9 * W
    # 墨量上下限 0.16W/0.98W（原 0.2W/0.9W，后放宽 0.12W）：
    # 实拍护照 MRZ 行墨量常在 0.2W~0.9W 边缘抖动，旧窗口会把真实
    # MRZ 行首尾截断甚至整行并入“页面边框”候选，导致 pick() 从
    # 数据区文字里选错 pair（GUI 照片→MRZ 手动测试 2026-9-30-AD/AF/G
    # 均因此定位错、OCR unreadable）。
    # 但 0.12W 太低：护照页数据区的浅底纹/小字墨量（0.12W~0.16W）处处
    # 越过下界，把真实 line1 桥接进上方的大 text-run，line_groups 因
    # h>48 直接判 0，真实 line1 永远成不了候选；pick() 只能拿真实
    # line2（独立 run）配上其下方反射/黑带前缘，整个 pair 错位
    # （Photo/3-6 实测：真实 line1 在 work y613-633、line2 y650-672，
    #  旧逻辑返回 work ((652,671),(704,719)) → 原图把 line2 当 line1、
    #  全宽黑带当 line2，OCR 全错）。
    # 0.16W 恰好把数据区底纹（work 实测墨量 150-430）挡在 MRZ 行
    # （峰值 480+/540+）之下，真实双行各自独立成 run 且字符段数达标。
    text = (row_ink_s > 0.16 * W) & (row_ink_s < 0.98 * W) & (E > 1800)
    solid_b = _run_groups(solid, min_len=5)
    text_b = _run_groups(text, min_len=4)
    cand = [b for b in solid_b
            if int(H * 0.3) <= b[0] <= int(H * 0.9) and b[1] - b[0] >= 10]
    s_top = cand[-1][1] + 1 if cand else None

    def line_groups(t):
        """Return number of separated glyph columns; 0 for non-MRZ rows.

        A real MRZ row has ~44 glyphs -> >=40 short runs across the width.
        Page borders / dense underlines merge into one long run and score
        few groups; 反射/装饰行 also score few groups. 旧实现按 density/
        max_group 阈值拒边框, 但实拍 MRZ 行墨量接近 0.9W 时会被当成
        “页面边框”整行排除, 导致定位落入数据区文字。这里改为字符段
        数判据, 天然区分 44 字符 MRZ 行与普通文字/边框/反射行。
        """
        y0, y1 = t
        h = y1 - y0 + 1
        if not (5 <= h <= 48):
            return 0
        band = g[y0:y1 + 1] < thr
        col_ink = band.sum(axis=0)
        strong = col_ink >= 4
        groups = _run_groups(strong, min_len=1)
        # 一行的墨迹横跨过长(>0.45W 连续)是边框/下缘线, 不是 MRZ 字符行
        max_group = max((b - a + 1 for a, b in groups), default=0)
        if max_group > max(160, int(W * 0.45)):
            return 0
        return len(groups)

    def pick(runs):
        candidates = []
        for i, t1 in enumerate(runs[:-1]):
            g1 = line_groups(t1)
            if g1 < 40:
                continue
            h1 = t1[1] - t1[0] + 1
            t2 = runs[i + 1]
            h2 = t2[1] - t2[0] + 1
            gap = t2[0] - t1[1] - 1
            g2 = line_groups(t2)
            if g2 < 40:
                continue
            # 真实 MRZ 双行: 行高接近(0.4~1.6), 行距约为行高量级。
            # gap/h 比值区间排除两类假对: 倒影大间距(>2.0)与数据区
            # 紧贴小字(<0.25), 字符段数又排除普通文本/边框行。
            ratio = gap / max(1, min(h1, h2))
            if (0.4 <= h2 / h1 <= 1.6) and (0.25 <= ratio <= 2.0):
                rank = (-min(h1, h2), -(g1 + g2), abs(h1 - h2) + gap, gap)
                candidates.append((rank, t1, t2))
        if not candidates:
            return None
        _, t1, t2 = min(candidates, key=lambda item: item[0])
        return t1, t2

    if s_top is not None:
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
    # 无取景黑带(整页满幅照)或黑带锚定均失败: 全量 text 段搜索。
    if len(text_b) >= 2:
        res = pick(text_b)
        if res:
            return res
    return None


def find_mrz_pair(g):
    """真实 MRZ 双行定位：优先从最后一条取景黑带上方找相邻 text-run pair。

    布局: [0, l1y0) = 黑色空白 + 完整梯形护照(照片/数据区),
    [l1y0, l2y1] = 真实 MRZ 两行, 取景黑带 + 镜片倒影在其下方.
    旧实现从黑带下方找, 会把 line1 的镜像倒影当成 line2. 因此优先在黑带
    上方找行高近似、间距合理的相邻 text-run pair; 找不到再回退旧逻辑.
    整页满幅照(无取景黑带)没有黑带可锚定, 直接在全部 text 段中按同样
    形状规则搜索 (rank 偏好最大字号, MRZ 行通常是全图字号最大的文本行).
    证件带全息/波浪浅色底纹时, 180 阈值会把底纹与 MRZ 桥接成过高/过密的
    文本段; 此时按 170/160/150 若依次收紧阈值重试, 并取 line1 位置最低
    (y0 最大) 的配对 —— MRZ 恒在证件底部, 可排除高阈值下页面顶部的伪配对.
    返回 (line1, line2) 两个 (y0, y1)；找不到返回 None。
    """
    row_ink = (g < 180).sum(axis=1)
    gx = np.abs(np.diff(g.astype(np.int16), axis=1))
    gx = np.pad(gx, ((0, 0), (1, 0)))
    E = gx.sum(axis=1)
    # 无条件收集 180/170/160/150 四个阈值的全部候选，取 line1 y0 最大
    # （页面最底部）的 pair。早期"单一 180 阈值优先、命中即短路"会把
    # 页面中上部的伪 pair（如 0f9780b8 的页面正文 h16/h7 两行、gap 7px）
    # 直接短路返回，真实 MRZ 被桥接进超长 text-run 永远成不了候选；
    # 全量 115 张实测：多阈值取底部 PASS 105 vs 180 短路 102。
    cands = []
    for thr in (180, 170, 160, 150):
        res = _mrz_pair_pass(g, thr, row_ink, E)
        if res:
            cands.append(res)
    if not cands:
        return None
    return max(cands, key=lambda pair: pair[0][0])


def find_mrz_single(g):
    """定位单行 MRZ（CS/CD 港澳台通行证格式）。

    CS/CD 通行证只有一行 MRZ，位于卡片底部。寻找特征：
    - 位于图像底部 30% 区域
    - 高水平梯度能量（字符边缘密集）
    - 行墨量在 0.15W~0.95W 之间（文本行，非纯黑/纯白）
    - 行高 >= 6 像素
    返回 (y0, y1) 或 None。
    """
    H, W = g.shape
    row_ink = (g < 180).sum(axis=1)
    gx = np.abs(np.diff(g.astype(np.int16), axis=1))
    gx = np.pad(gx, ((0, 0), (1, 0)))
    E = gx.sum(axis=1)
    # Adaptive gradient threshold: use percentile of bottom 35% rows
    # to handle guilloche backgrounds where text has lower contrast.
    bottom_start = int(H * 0.65)
    e_bottom = E[bottom_start:]
    if len(e_bottom) > 0:
        e_thr_init = max(500, int(np.percentile(e_bottom, 60)))
    else:
        e_thr_init = 800
    text = (row_ink > 0.15 * W) & (row_ink < 0.95 * W) & (E > e_thr_init)
    text_b = _run_groups(text, min_len=4)
    candidates = [b for b in text_b
                  if b[0] >= bottom_start and b[1] - b[0] >= 6]
    if not candidates:
        return None
    y0, y1 = max(candidates, key=lambda b: b[0])
    e_thr = max(1200, int(np.percentile(e_bottom, 80)))
    max_expand = max(6, (y1 - y0))
    expanded = 0
    while y0 > bottom_start and E[y0 - 1] > e_thr \
            and 0.02 * W < row_ink[y0 - 1] < 0.95 * W \
            and expanded < max_expand:
        y0 -= 1
        expanded += 1
    expanded = 0
    while y1 < H - 1 and E[y1 + 1] > e_thr \
            and 0.02 * W < row_ink[y1 + 1] < 0.95 * W \
            and expanded < max_expand:
        y1 += 1
        expanded += 1
    for _ in range(10):
        if y0 > bottom_start and row_ink[y0 - 1] < 0.90 * W:
            y0 -= 1
        if y1 < H - 1 and row_ink[y1 + 1] < 0.90 * W:
            y1 += 1
    return (y0, y1)


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


def _line2_bottom_tail(row_ink, l2y1, W, thr_ratio=0.16, cap=6, min_ink=3):
    """MRZ line2 底缘字符淡墨的延伸行数（仅当下方墨迹严格递减到 0 时吸收）。

    背景: text 掩码 (row_ink > 0.16W) 会把 line2 底部 3-4px 的真实字符
    淡墨行裁掉（如 tmp/PassPortPhoto/2026-9-30-G 实测 644:236, 645:154,
    646:83, 647:34, 648:13, 649:0），使去影图 / MRZ 截图缺底缘、字不全。

    判据（只吸收真实字符尾段，杜绝误扩）:
      从 l2y1+1 向下扫。首行可不受 0.16W 限制（可能是检测框漏掉的
      line2 主体底缘行），但必须严格小于 line2 底墨量 row_ink[l2y1]；
      其后每行墨量 v 必须 0 < v < 0.16W 且严格递减 (v < 上一行)。
      连续递减行计数为 tail; 遇到 v<=0 (墨尾) 返回累计 tail, 扫满 cap
      仍未归零、或某行 v 非递减 / 后续行 v>=0.16W, 整段不吸收返回 0。
    其余图（assets 0f9780b8/6b911174/f6eeb483 等）line2 紧邻行墨量
    ≥0.16W 且向下回落前出现波动（第四五行回升），tail 恒为 0，
    行为与基准完全一致。

    cap: 扩展行数上限，防止把下一段文字/反射吸入。
    返回扩展行数（0 表示无需扩展）。
    """
    if row_ink is None or l2y1 < 0 or l2y1 >= len(row_ink):
        return 0
    thr = max(3, int(W * thr_ratio))
    base = int(row_ink[l2y1])
    if base <= 0:
        return 0
    prev = base
    tail = 0
    for y in range(l2y1 + 1, min(len(row_ink), l2y1 + 1 + cap)):
        v = int(row_ink[y])
        if v <= 0:
            break
        if v >= prev:
            return 0
        if tail > 0 and v >= thr:
            return 0
        prev = v
        tail += 1
    if tail == 0:
        return 0
    return tail


def drop_reflection(work_rgb, st):
    """去镜片倒影: 保留 [0, line2.y1+1] 连续区域 (黑色空白 + 完整梯形护照 + MRZ),
    只裁掉 line2 下方整段倒影; line1 上方是护照页真实内容, 不裁.

    line2 底缘 3-4px 真实字符淡墨 (text 掩码 0.16W 会裁掉的行) 会在裁剪时
    一并保留 (见 _line2_bottom_tail), 避免 MRZ 字被切掉一截.

    info 如实报告: reflection_detected/reflection_px 来自相对墨量检出,
    未检出倒影时仅裁取景黑带, 不虚报为倒影.
    """
    H = work_rgb.shape[0]
    pair = st.get("mrz_pair")
    reflection = list(st.get("reflection", []) or [])

    if not pair:
        return work_rgb, st, {
            "mode": "noop", "actual_drop": 0,
            "crop_top": H, "crop_bottom": H,
            "reflection": [], "reflection_px": 0,
            "reflection_detected": False}

    # 真实布局: [0, l1y0) = 黑色空白 + 完整梯形护照(照片/数据区),
    # [l1y0, l2y1] = MRZ 两行, (l2y1, H) = 镜片倒影 + 取景黑带.
    # line1 上方并非反射 (那只是护照页底部黑带区), 不裁.
    # 只保留到真实 MRZ 的最后一行, 确保模糊倒影不再进入后续算法.
    # 但 text 掩码会裁掉 line2 底缘 3-4px 真实字符淡墨 (严格递减到 0 的
    # 墨尾), 检测框对此类尾段一律不吸收; 真字符尾段用 _line2_bottom_tail
    # 补回, 避免 MRZ 字被切掉一截 (2026-9-30-G 实测缺 645-648).
    l2y1 = pair[1][1]
    row_ink = st.get("row_ink")
    tail = _line2_bottom_tail(row_ink, l2y1, work_rgb.shape[1])
    new_l2y1 = min(H - 1, l2y1 + tail)
    crop_bottom = min(H, l2y1 + 1 + tail)
    keep = work_rgb[:crop_bottom]
    st2 = dict(st)
    st2["reflection"] = []
    if tail:
        # 只在这条 drop 路径里把检测框底缘补到真实墨尾, 不改 find_mrz_pair 输出.
        st2["mrz_pair"] = (pair[0], (pair[1][0], new_l2y1))
        st2["mrz"] = (pair[0][0], new_l2y1)
        # 原反射段若落在扩展后的 line2 内 (墨尾被误判为反射) 需要移除,
        # 避免 info 虚报反射、clean 把真字符尾段当反射过渡涂白.
        cut = l2y1 + 1 + tail
        reflection = [
            (max(a, cut), b) if a < cut and b >= cut else (a, b)
            for a, b in reflection if b >= cut
        ]
        st2["reflection_cropped"] = reflection
    else:
        st2["reflection_cropped"] = reflection
    return keep, st2, {
        "mode": "crop", "actual_drop": H - keep.shape[0],
        "crop_top": 0, "crop_bottom": crop_bottom,
        "reflection": reflection,
        "reflection_px": sum(b - a + 1 for a, b in reflection),
        "reflection_detected": bool(reflection)}



def clean_band_reflection(band_rgb, line1, line2, clear_rows=2,
                            unsharp=True, sharp_amount=1.5, blur_sigma=2.0,
                            reflection=False):
    """清除 band 内残留的镜片反射过渡行 (仅在检出倒影时).

    1) 清 line1 紧邻的 N 行 (line1 下方反射过渡最强区, 反射随距离衰减)
    2) 清 line2 紧邻的 N 行 (line2 下方倒影过渡)
    3) unsharp USM 锐化: 抑制反射模糊层, 增强锐利的字符边缘

    reflection=False (未检出倒影) 时原样返回, 不做白化/锐化, 避免涂掉
    line1 下方可能存在的真实笔迹.
    """
    if not reflection:
        return band_rgb
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
    仅在 st 检出倒影 (reflection/reflection_cropped) 时清理, 否则只裁剪.
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
    reflection = bool(st.get("reflection") or st.get("reflection_cropped"))
    clean = clean_band_reflection(band, l1_in_band, l2_in_band,
                                  clear_rows=clear_rows, unsharp=unsharp,
                                  reflection=reflection)
    return clean, {"band_in_work": (cy0, cy1), "l1_in_band": l1_in_band,
                   "l2_in_band": l2_in_band, "reflection": reflection}


def rectify_pair(work_rgb, st, tgt_w=TGT_W, tgt_h=TGT_H):
    """严格双行单应矫正：line1/line2 各自四点 -> 矩形上下两半，分别 warp 后拼接。

    反射倒影下方的连续 text 段不再被纳入 MRZ 带。每行字符严格匹配训练分布
    (GLYPH_W x GLYPH_H)。返回 (rgb, info) 或 (None, None)。

    注意：这不是生产识别路径
    ------------------------
    识别走的是 6_mrz_ocr 的 native 流水线，它自己做 deskew（墨点旋转投影）
    + 44 格 pitch-grid + 逐格面积重采样，比这里的单应更准。这个函数保留下来
    是因为它对「页面的四个角在哪」这件事给了一个人眼可看的答案，在调研
    (10_ocr_research/probe_backends.py) 和回归测试里仍被用到。

    但它**不应**被当作识别前的预处理：
      * 它对「MRZ 在哪」的估计靠 _mrzh_quad 的页面亮区估计，在整体偏亮的
        照片上会把整幅图当成页面，矫正后整张照片被压成 44 格 x 2 行；
      * 实测：把矫正图喂给同一个 mrz_ocr_cnn_tool，assets/pic 上 ICAO
        通过 2/72，而自然 crop（crop_pair_clean）是 3/72。
    要给识别用的图，用 crop_pair_clean()。
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
    """单应矫正：MRZ 带四边形 -> 标准 44格x2行 矩形。返回 RGB 图或 None。

    同 rectify_pair()：这是展示/调研用的几何变换，不是识别路径。
    """
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
