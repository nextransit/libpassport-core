"""passport_locator -- synthetic passport-page sample + region locator.

Provides:
  * make_sample_passport() -- render a standards-like TD3 passport page
    (photo zone top-left, personal-data zone, two OCR-B MRZ lines at the
    bottom) with PIL, no external assets required.
  * locate_regions()      -- locate the photo / data / mrz regions of a
    passport-page image purely from pixel statistics (row projection,
    skin-colour mask), returning bbox + confidence + human-readable
    evidence for each region.

Regions:
  photo : passport photograph zone (top-left)
  data  : personal-data text zone
  mrz   : machine-readable zone (the two black OCR-B lines at the bottom)
"""
from __future__ import annotations

import os
from collections import deque

from PIL import Image, ImageDraw, ImageFont

W, H = 900, 600

SAMPLE_LINE1 = "P<EIKSSON<ANNA<MARIA<<<<<<<<<<<<<<<<<<<<"
SAMPLE_LINE2 = "L898902C36UTO6908061F9406236ZE184226B<<<<<18"

_SKIN_LO = (90, 40, 20)
_SKIN_HI = (255, 255, 255)

# OCR-B glyph bitmaps (8x12, bit7=0x80 is the leftmost column).  Copied from
# 6_mrz_ocr/src/template.c so the synthetic MRZ lines are rendered with the
# same glyph shapes the mrz_ocr_tool templates expect.
_OCR_B = {
    '0': (0x3C,0x66,0x66,0x6E,0x76,0x66,0x66,0x66,0x66,0x6E,0x66,0x3C),
    '1': (0x18,0x38,0x78,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x7E),
    '2': (0x3C,0x66,0x66,0x06,0x06,0x0C,0x18,0x30,0x60,0x60,0x60,0x7E),
    '3': (0x3C,0x66,0x66,0x06,0x06,0x1C,0x06,0x06,0x06,0x66,0x66,0x3C),
    '4': (0x0C,0x1C,0x3C,0x6C,0x6C,0x66,0x66,0x7E,0x06,0x06,0x06,0x06),
    '5': (0x7E,0x60,0x60,0x60,0x7C,0x66,0x06,0x06,0x06,0x66,0x66,0x3C),
    '6': (0x3C,0x66,0x60,0x60,0x7C,0x66,0x66,0x66,0x66,0x66,0x66,0x3C),
    '7': (0x7E,0x66,0x06,0x06,0x0C,0x18,0x18,0x18,0x18,0x30,0x30,0x30),
    '8': (0x3C,0x66,0x66,0x66,0x66,0x3C,0x66,0x66,0x66,0x66,0x66,0x3C),
    '9': (0x3C,0x66,0x66,0x66,0x66,0x66,0x3E,0x06,0x06,0x06,0x66,0x3C),
    'A': (0x18,0x18,0x3C,0x3C,0x66,0x66,0x7E,0x7E,0x66,0x66,0x66,0x66),
    'B': (0x7C,0x66,0x66,0x66,0x66,0x7C,0x66,0x66,0x66,0x66,0x66,0x7C),
    'C': (0x3C,0x66,0x66,0x60,0x60,0x60,0x60,0x60,0x60,0x66,0x66,0x3C),
    'D': (0x78,0x6C,0x66,0x66,0x66,0x66,0x66,0x66,0x66,0x66,0x6C,0x78),
    'E': (0x7E,0x60,0x60,0x60,0x60,0x7C,0x60,0x60,0x60,0x60,0x60,0x7E),
    'F': (0x7E,0x60,0x60,0x60,0x60,0x7C,0x60,0x60,0x60,0x60,0x60,0x60),
    'G': (0x3C,0x66,0x66,0x60,0x60,0x60,0x6E,0x66,0x66,0x66,0x66,0x3C),
    'H': (0x66,0x66,0x66,0x66,0x66,0x7E,0x66,0x66,0x66,0x66,0x66,0x66),
    'I': (0x3C,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x3C),
    'J': (0x3E,0x0C,0x0C,0x0C,0x0C,0x0C,0x0C,0x0C,0x0C,0x6C,0x6C,0x38),
    'K': (0x66,0x6C,0x78,0x70,0x78,0x6C,0x66,0x66,0x66,0x66,0x6C,0x66),
    'L': (0x60,0x60,0x60,0x60,0x60,0x60,0x60,0x60,0x60,0x60,0x60,0x7E),
    'M': (0x66,0x7E,0x7E,0x7E,0x66,0x66,0x66,0x66,0x66,0x66,0x66,0x66),
    'N': (0x66,0x66,0x76,0x7E,0x7E,0x6E,0x66,0x66,0x66,0x66,0x66,0x66),
    'O': (0x3C,0x66,0x66,0x66,0x66,0x66,0x66,0x66,0x66,0x66,0x66,0x3C),
    'P': (0x7C,0x66,0x66,0x66,0x66,0x7C,0x60,0x60,0x60,0x60,0x60,0x60),
    'Q': (0x3C,0x66,0x66,0x66,0x66,0x66,0x66,0x66,0x6E,0x6C,0x36,0x1E),
    'R': (0x7C,0x66,0x66,0x66,0x66,0x7C,0x6C,0x66,0x66,0x66,0x66,0x66),
    'S': (0x3E,0x60,0x60,0x60,0x3C,0x06,0x06,0x06,0x06,0x06,0x66,0x3C),
    'T': (0x7E,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18),
    'U': (0x66,0x66,0x66,0x66,0x66,0x66,0x66,0x66,0x66,0x66,0x66,0x3C),
    'V': (0x66,0x66,0x66,0x66,0x66,0x66,0x66,0x66,0x3C,0x3C,0x18,0x18),
    'W': (0x66,0x66,0x66,0x66,0x66,0x66,0x66,0x7E,0x7E,0x7E,0x66,0x66),
    'X': (0x66,0x66,0x66,0x3C,0x3C,0x18,0x3C,0x3C,0x66,0x66,0x66,0x66),
    'Y': (0x66,0x66,0x66,0x66,0x3C,0x18,0x18,0x18,0x18,0x18,0x18,0x18),
    'Z': (0x7E,0x06,0x06,0x06,0x0C,0x18,0x30,0x60,0x60,0x60,0x60,0x7E),
    '<': (0x00,0x00,0x06,0x0C,0x18,0x30,0x60,0x30,0x18,0x0C,0x06,0x00),
}


def _face_font(size: int):
    for name in ("Menlo", "DejaVu Sans Mono", "Courier New", "monospace"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_mrz_line(draw, x0, y0, text, scale=2):
    """Render one MRZ line with the real OCR-B bitmap glyphs (8x12, scaled).

    Uses the exact glyph shapes from 6_mrz_ocr/src/template.c so the
    synthetic page passes the mrz_ocr_tool template matcher.  scale=2 gives
    16x24px glyphs -> ~790px wide for a 44-char line (fits W=900 page).
    Returns the bar height.
    """
    gw, gh = 8 * scale, 12 * scale
    pitch = gw + 2          # glyph advance: glyph width + 2px gap
    w_text = len(text) * pitch - 2
    bar_h = gh + 8
    pad = 4
    draw.rectangle([x0, y0, x0 + w_text + 2 * pad, y0 + bar_h],
                   fill=(255, 255, 255))
    for i, ch in enumerate(text):
        glyph = _OCR_B.get(ch)
        if glyph is None:
            continue
        px = x0 + pad + i * pitch
        py = y0 + pad
        for r, byte in enumerate(glyph):
            for c in range(8):
                if byte & (0x80 >> c):
                    draw.rectangle(
                        [px + c * scale, py + r * scale,
                         px + (c + 1) * scale - 1, py + (r + 1) * scale - 1],
                        fill=(0, 0, 0))
    return bar_h


def make_sample_passport(out: str | None = None, face_path: str | None = None,
                         W: int = 1600, H: int = 1000) -> Image.Image:
    """Render a synthetic TD3 passport page.  Saves to `out` if given."""
    img = Image.new("RGB", (W, H), (242, 242, 248))
    d = ImageDraw.Draw(img)

    # Page header band.
    d.rectangle([0, 0, W, 46], fill=(20, 60, 120))
    d.text((24, 12), "REPUBLIC OF EIKSSONIA", fill=(255, 255, 255),
           font=_face_font(20))
    d.text((W - 260, 14), "PASSPORT  \\  TD3", fill=(220, 232, 255),
           font=_face_font(15))

    # Photo zone: white card + border at top-left.
    px, py, pw, ph = 40, 66, 170, 210
    d.rectangle([px, py, px + pw, py + ph], fill=(255, 255, 255),
                outline=(70, 70, 80), width=2)
    d.text((px + 6, py + 4), "PHOTO", fill=(150, 150, 160), font=_face_font(11))

    # Composite a synthetic face inside the photo zone.
    face = None
    if face_path and os.path.exists(face_path):
        try:
            face = Image.open(face_path).convert("RGB")
        except Exception:
            face = None
    if face is None:
        face = _draw_face(W=140, H=180)
    face = face.resize((pw - 12, ph - 28))
    img.paste(face, (px + 6, py + 22))

    # Personal-data zone right of the photo.
    dx = px + pw + 34
    fields = [
        ("Surname", "EIKSSON"),
        ("Given names", "ANNA MARIA"),
        ("Passport No.", "L898902C3"),
        ("Nationality", "UTO"),
        ("Date of birth", "69 08 06"),
        ("Date of expiry", "94 06 23"),
        ("Authority", "EIKSSONIA"),
    ]
    fy = 84
    lfont = _face_font(13)
    vfont = _face_font(17)
    for label, value in fields:
        d.text((dx, fy), label + ":  ", fill=(90, 90, 100), font=lfont)
        d.text((dx + 118, fy - 2), value, fill=(20, 20, 25), font=vfont)
        fy += 30

    # MRZ zone: two OCR-B bars at the bottom (bitmap glyphs, scale=4, same
    # source bitmap as the C templates -> dense top rows for locate_band).
    mz_y = H - 150
    _sc = 4
    # Cover the whole MRZ strip with white first so the gap between the two
    # bars is pure white (255), keeping the crop strictly binary {0,255}.
    _bar = _draw_mrz_line(d, 40, mz_y, SAMPLE_LINE1, scale=_sc)
    _w = len(SAMPLE_LINE1) * (8 * _sc + 2) - 2
    d.rectangle([40, mz_y, 40 + _w + 8, mz_y + 2 * _bar + 8],
                fill=(255, 255, 255))
    h1 = _draw_mrz_line(d, 40, mz_y, SAMPLE_LINE1, scale=_sc)
    # Bar2 sits right under bar1 (white bars overlap) so the two glyph rows
    # merge into one dense run for mrz_ocr_locate_band (gap <= max_gap).
    _draw_mrz_line(d, 40, mz_y + h1 - 6, SAMPLE_LINE2, scale=_sc)

    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        img.save(out)
    return img


def _draw_face(W: int, H: int) -> Image.Image:
    """Small deterministic cartoon face (skin disc + features), no assets."""
    img = Image.new("RGB", (W, H), (235, 238, 248))
    d = ImageDraw.Draw(img)
    cx, cy, r = W // 2, H // 2 - 6, min(W, H) // 3
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(232, 205, 178))
    d.ellipse([cx - 22, cy - 18, cx - 8, cy - 4], fill=(40, 40, 45))
    d.ellipse([cx + 8, cy - 18, cx + 22, cy - 4], fill=(40, 40, 45))
    d.arc([cx - 26, cy + 4, cx + 26, cy + 44], 20, 160,
          fill=(120, 50, 50), width=3)
    d.arc([cx - r - 4, cy - r - 8, cx + r + 4, cy + r + 8], 180, 360,
          fill=(95, 70, 55), width=6)
    return img


# --------------------------------------------------------------------------
# Region locating (pixel statistics only, no ML).
# --------------------------------------------------------------------------

def _to_gray_np(img: Image.Image):
    import numpy as np
    a = np.asarray(img.convert("RGB"))
    r = a[:, :, 0].astype(np.float64)
    g = a[:, :, 1].astype(np.float64)
    b = a[:, :, 2].astype(np.float64)
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)


def _skin_mask(img: Image.Image):
    import numpy as np
    a = np.asarray(img.convert("RGB"))
    r, g, b = a[:, :, 0].astype(int), a[:, :, 1].astype(int), a[:, :, 2].astype(int)
    lo = _SKIN_LO
    mask = (
        (r >= lo[0]) & (r <= _SKIN_HI[0])
        & (g >= lo[1]) & (g <= _SKIN_HI[1])
        & (b >= lo[2]) & (b <= _SKIN_HI[2])
        & (r > g) & (g > b) & ((r - g) > 15) & ((r - b) > 30)
    )
    return mask


def _largest_skin_bbox(mask):
    """Return (x0,y0,x1,y1) of the largest 8-connected skin component."""
    import numpy as np
    m = mask.astype(np.uint8)
    H, Wp = m.shape
    visited = np.zeros_like(m, dtype=bool)
    best = None
    best_area = 0
    for y in range(H):
        for x in range(Wp):
            if not m[y, x] or visited[y, x]:
                continue
            area = 0
            x0 = y0 = x1 = y1 = -1
            dq = deque([(y, x)])
            visited[y, x] = True
            while dq:
                cy, cx = dq.popleft()
                area += 1
                if x0 < 0:
                    x0 = x1 = cx
                    y0 = y1 = cy
                else:
                    x0, x1 = min(x0, cx), max(x1, cx)
                    y0, y1 = min(y0, cy), max(y1, cy)
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H and 0 <= nx < Wp and m[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        dq.append((ny, nx))
            if area > best_area:
                best_area = area
                best = (x0, y0, x1, y1)
    return best, best_area


def _row_density(gray, dark_thr=128):
    import numpy as np
    return (gray < dark_thr).sum(axis=1)


def _col_density(gray, dark_thr=128):
    import numpy as np
    return (gray < dark_thr).sum(axis=0)


def _expand_to_card(img, x0, y0, x1, y1, mrz):
    """Grow a skin bbox outwards to the enclosing light photo card.

    The card boundary is detected as the first dark edge / coloured frame
    surrounding the bright region, capped at the MRZ zone top.
    """
    import numpy as np
    g = _to_gray_np(img)
    H, Wp = g.shape
    top_limit = mrz[1] if mrz else H - 1
    bright = g > 200                       # card fill incl. skin (213)
    fw = int((x1 - x0) * 0.60)
    fh = int((y1 - y0) * 0.60)

    cx0, cx1 = x0, x1
    for k in range(1, fw):
        if x0 - k < 0:
            cx0 = 0
            break
        col = bright[y0:y1 + 1, x0 - k]
        if col.mean() < 0.9:
            cx0 = x0 - k + 1
            break
    for k in range(1, fw):
        if x1 + k >= Wp:
            cx1 = Wp - 1
            break
        col = bright[y0:y1 + 1, x1 + k]
        if col.mean() < 0.9:
            cx1 = x1 + k - 1
            break
    cy0, cy1 = y0, y1
    for k in range(1, fh):
        if y0 - k < 0:
            cy0 = 0
            break
        row = bright[y0 - k, cx0:cx1 + 1]
        if row.mean() < 0.9:
            cy0 = y0 - k + 1
            break
    for k in range(1, fh):
        if y1 + k >= top_limit:
            cy1 = min(top_limit, H - 1)
            break
        row = bright[y1 + k, cx0:cx1 + 1]
        if row.mean() < 0.9:
            cy1 = y1 + k - 1
            break
    return (cx0, cy0, cx1, cy1)


def _locate_mrz(gray):
    """Locate the bottom MRZ band. Returns (x0, y0, x1, y1) with a small
    margin so the returned box always covers the full glyph height
    (character ascenders / descenders extend a few pixels above and
    below the dense rows)."""
    import numpy as np
    H, Wp = gray.shape
    rd = _row_density(gray)
    # Lower the density threshold from 15% to 8% so we keep the very
    # top and bottom of the two MRZ lines (descenders extend a few
    # pixels beyond the dense core) without spilling into the photo
    # or data areas above the band.
    dense = rd > Wp * 0.08
    rows = [i for i, v in enumerate(dense) if v]
    if not rows:
        return None
    # Group consecutive dense rows.
    groups = []
    start = prev = rows[0]
    for i in rows[1:]:
        if i == prev + 1:
            prev = i
        else:
            groups.append([start, prev])
            start = prev = i
    groups.append([start, prev])
    # Merge groups whose gap is small (both MRZ bars + tiny gaps are one band).
    merged = []
    for g in groups:
        if merged and g[0] - merged[-1][1] <= 40:
            merged[-1][1] = g[1]
        else:
            merged.append(g)
    merged = [g for g in merged if g[1] - g[0] >= 20]
    if not merged:
        return None
    # Pick the band that most likely contains the two MRZ lines:
    #   1) Inside each merged group, look for two dense sub-segments
    #      separated by a single sparse row (the MRZ double-line
    #      signature). If we find that, use the last two sub-segments
    #      as the MRZ box.
    #   2) Otherwise pick the lowest dense group only if it is short
    #      enough to plausibly be MRZ (<= 20% of the page height).
    chosen = None
    # The MRZ "fingerprint" is two dense sub-rows separated by exactly
    # one sparse row of small gap (<= 8 px). Search every merged group
    # for that pattern starting from the *bottom* of the image; the
    # first hit wins because MRZ is always in the lower portion.
    for g in reversed(merged):
        g0, g1 = g
        local = dense[g0:g1 + 1]
        sub_dense_rows = [i + g0 for i, v in enumerate(local) if v]
        if len(sub_dense_rows) < 4:
            continue
        sub_groups = []
        s = p = sub_dense_rows[0]
        for r in sub_dense_rows[1:]:
            if r - p <= 4:
                p = r
            else:
                sub_groups.append([s, p])
                s = p = r
        sub_groups.append([s, p])
        candidates = [sg for sg in sub_groups
                      if 10 <= sg[1] - sg[0] <= 60]
        if len(candidates) >= 2:
            # Two-line signature: pick the lowest pair whose gap is
            # plausible for the two MRZ lines.
            for i in range(len(candidates) - 1, 0, -1):
                line1, line2 = candidates[i - 1], candidates[i]
                if line2[0] - line1[1] <= 25:
                    chosen = (line1[0], line2[1])
                    break
            if chosen is not None:
                break
        if chosen is None and candidates:
            # Single-line fallback: the bottom-most candidate is the
            # MRZ itself. This happens when one of the two MRZ lines
            # is partially clipped or the image is rotated.
            chosen = (candidates[-1][0], candidates[-1][1])
            break
    if chosen is None:
        # Single-line fallback: the bottom-most dense group whose
        # height is <= 60 px and whose bottom edge is in the lower
        # 70% of the image.
        for g in reversed(merged):
            h_band = g[1] - g[0]
            if h_band <= 60 and g[1] >= int(H * 0.30):
                chosen = (g[0], g[1])
                break
    if chosen is None:
        return None
    y0, y1 = chosen
    # x range: from column density inside the band.
    band = gray[y0:y1 + 1]
    cd = _col_density(band)
    cols = np.where(cd > 0)[0]
    if len(cols) == 0:
        return None
    x0, x1 = int(cols[0]), int(cols[-1])
    # Pad the box so the returned region contains every glyph pixel,
    # not just the dense row projection. Characters extend a few pixels
    # above and below the dense band; vertical margin = 4 px,
    # horizontal margin = 8 px (matches the margin used by
    # passport_pipeline.ocr_crop so OCR sees the same crop).
    pad_top, pad_bot, pad_l, pad_r = 4, 4, 8, 8
    y0 = max(0, y0 - pad_top)
    y1 = min(H - 1, y1 + pad_bot)
    x0 = max(0, x0 - pad_l)
    x1 = min(Wp - 1, x1 + pad_r)
    return (x0, y0, x1, y1)


def _locate_data(gray, photo, mrz):
    import numpy as np
    H, Wp = gray.shape
    if isinstance(photo, dict):
        photo = (photo["x"], photo["y"], photo["x"] + photo["w"], photo["y"] + photo["h"])
    if isinstance(mrz, dict):
        mrz = (mrz["x"], mrz["y"], mrz["x"] + mrz["w"], mrz["y"] + mrz["h"])
    # Text rows live to the right of the photo card.
    x0 = (photo[2] + 20) if photo else 20
    if x0 >= Wp:
        return None
    sub = gray[:, x0:]
    rd = (sub < 128).sum(axis=1)
    top = photo[1] if photo else 30
    bottom = mrz[1] if mrz else H - 20
    rows = [i for i, v in enumerate(rd) if v > 3 and top <= i < bottom]
    if not rows:
        return None
    # merge rows within 4px gaps
    groups = []
    start = prev = rows[0]
    for i in rows[1:]:
        if i - prev <= 4:
            prev = i
        else:
            groups.append((start, prev))
            start = prev = i
    groups.append((start, prev))
    y0, y1 = groups[0][0], groups[-1][1]
    if y1 - y0 < 10:
        return None
    # full horizontal extent of dark text within the found band
    band = gray[y0:y1 + 1, :]
    cd = _col_density(band)
    cols = np.where(cd > 0)[0]
    bx0, bx1 = int(cols[0]), int(cols[-1])
    return (bx0, y0, bx1, y1)


def locate_regions(img: Image.Image):
    """Locate photo / data / mrz regions.

    Returns dict of {name: {'x','y','w','h','confidence','evidence'}}.
    A region may be None when it cannot be located.
    """
    import numpy as np
    gray = _to_gray_np(img)
    res = {"photo": None, "data": None, "mrz": None}

    mrz = _locate_mrz(gray)
    if mrz:
        x0, y0, x1, y1 = mrz
        res["mrz"] = {
            "x": x0, "y": y0, "w": x1 - x0 + 1, "h": y1 - y0 + 1,
            "confidence": 0.97,
            "evidence": "bottom band: %d dense dark rows (row projection > 15%% width)"
                        % (y1 - y0 + 1),
        }

    # Photo: largest skin-colour component in the upper half, expanded to
    # cover the surrounding light photo card.
    skin = _skin_mask(img)
    if mrz:
        upper = skin[:mrz[1]]
    else:
        upper = skin
    bbox, area = _largest_skin_bbox(upper)
    if bbox and area > 800:
        x0, y0, x1, y1 = bbox
        # Expand to the enclosing light card (edge = large bright run).
        card = _expand_to_card(img, x0, y0, x1, y1, mrz)
        cx0, cy0, cx1, cy1 = card
        conf = min(0.99, 0.55 + area / 40000.0)
        res["photo"] = {
            "x": cx0, "y": cy0, "w": cx1 - cx0 + 1, "h": cy1 - cy0 + 1,
            "confidence": round(conf, 3),
            "evidence": "skin component (%d px) inside light photo card"
                        % area,
        }

    data = _locate_data(gray, res["photo"], res["mrz"])
    if data:
        x0, y0, x1, y1 = data
        res["data"] = {
            "x": x0, "y": y0, "w": x1 - x0 + 1, "h": y1 - y0 + 1,
            "confidence": 0.92,
            "evidence": "text-line density between photo and MRZ (%d rows)"
                        % (y1 - y0 + 1),
        }
    return res


_REGION_COLOR = {"photo": (220, 40, 40), "data": (30, 90, 220), "mrz": (30, 160, 60)}


def annotate(img: Image.Image, loc, scale=1.0):
    """Return a copy of `img` with region boxes drawn."""
    out = img.copy()
    d = ImageDraw.Draw(out)
    for name, r in loc.items():
        if not r:
            continue
        color = _REGION_COLOR[name]
        lw = max(2, int(round(3 * scale)))
        d.rectangle([r["x"], r["y"], r["x"] + r["w"], r["y"] + r["h"]],
                    outline=color, width=lw)
        d.text((r["x"] + 4, r["y"] + 2), name.upper(), fill=color,
               font=_face_font(12))
    return out


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    sample = os.path.join(here, "assets", "passport_sample.png")
    img = make_sample_passport(sample)
    loc = locate_regions(img)
    for name, r in loc.items():
        if r:
            print("%-6s x=%4d y=%4d w=%4d h=%4d  conf=%.2f  %s"
                  % (name, r["x"], r["y"], r["w"], r["h"],
                     r["confidence"], r["evidence"]))
        else:
            print("%-6s not found" % name)