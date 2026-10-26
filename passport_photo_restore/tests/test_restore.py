from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "passport_photo_restore" / "src"))

from restore import (  # noqa: E402
    GLYPH_H,
    _gray,
    detect_structure,
    drop_reflection,
    load,
    rectify_pair,
)


def test_photo_pages_keep_real_mrz_and_drop_bottom_reflection():
    expected = {
        "IMG_1792133000_003.jpg": ((616, 633), (652, 671)),
        "IMG_1792133002_004.jpg": ((616, 633), (652, 671)),
        "IMG_1792133232_005.jpg": ((616, 633), (652, 671)),
        "IMG_1792133581_006.jpg": ((616, 633), (652, 671)),
        "IMG_1792133649_007.jpg": ((316, 326), (335, 347)),
        "IMG_1792133682_008.jpg": ((615, 635), (653, 676)),
    }
    for name, pair in expected.items():
        image = next(ROOT.glob(f"Photo/*/{name}"))
        rgb = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2RGB)
        work, _ = load(rgb)

        st = detect_structure(_gray(work))
        assert st["mrz_pair"] == pair
        assert st["mrz"] == (pair[0][0], pair[1][1])
        assert st["reflection"], f"{name}: 应检出 line2 下方倒影段"

        cleaned, cleaned_st, info = drop_reflection(work, st)
        assert cleaned.shape[0] == pair[1][1] + 1
        assert info["actual_drop"] == work.shape[0] - cleaned.shape[0]
        assert info["reflection_detected"] is True
        assert info["reflection_px"] > 0
        assert cleaned_st["mrz_pair"] == pair
        assert cleaned_st["reflection"] == []
        assert cleaned_st["reflection_cropped"]
        assert cleaned[-1].tolist() == work[pair[1][1]].tolist()

        rect, info = rectify_pair(cleaned, cleaned_st)
        assert rect is not None
        assert rect.shape[0] >= GLYPH_H * 2
        gray_rect = _gray(rect)
        assert int((gray_rect[:GLYPH_H] < 180).sum()) > 100
        assert int((gray_rect[GLYPH_H:GLYPH_H * 2] < 180).sum()) > 100
        assert info["line1_y_in_dst"] == (0, GLYPH_H - 1)
        assert info["line2_y_in_dst"] == (GLYPH_H, GLYPH_H * 2 - 1)
        if name == "IMG_1792133000_003.jpg":
            # The black camera frame is outside the passport page. It must
            # not become the source quad for the corrected MRZ.
            assert info["src_quad_l1"][0][0] > 0
            assert info["src_quad_l1"][1][0] < work.shape[1] - 1
