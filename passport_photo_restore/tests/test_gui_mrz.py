from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))

from gui import (  # noqa: E402
    MRZ_TOOL,
    _mrz_line2_candidates,
    _rst_adaptive_xpad,
    _rst_cnn_low_res_reason,
    _rst_cnn_source_heights,
    _rst_decode_candidates,
    _rst_tesseract_candidates,
)


EXPECTED_BY_FIRST_DIGIT = {
    "5": (
        "POCHNJIA<<LIMIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
        "5EZ0426536CHN7410024M2808303LMNGMBKCMDPBA970",
    ),
    "E": (
        "POCHNJIA<<LIMIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
        "EE20426536CHN7410024M2808303LMNGMBKCMDPBA970",
    ),
}


def test_photo_7_cnn_low_resolution_is_detected_before_engine_call():
    pair = ((316, 326), (335, 347))
    assert _rst_cnn_source_heights(pair, 1.0) == (11, 13)
    assert "11/13px" in _rst_cnn_low_res_reason(pair, 1.0)
    assert _rst_cnn_low_res_reason(((616, 633), (652, 671)), 0.482) == ""


@pytest.mark.skipif(
    shutil.which("tesseract") is None or not MRZ_TOOL.exists(),
    reason="requires tesseract OCR-B and mrz_tool",
)
def test_photo_mrz_auto_ocr_and_decode():
    pairs = {
        "3": ((616, 633), (652, 671)),
        "4": ((616, 633), (652, 671)),
        "5": ((616, 633), (652, 671)),
        "6": ((616, 633), (652, 671)),
        "7": ((316, 326), (335, 347)),
        "8": ((615, 635), (653, 676)),
    }
    for number, pair in pairs.items():
        image = next((ROOT / "Photo" / number).glob("*.jpg"))
        rgb = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2RGB)
        scale = 1250 / max(rgb.shape[:2]) if max(rgb.shape[:2]) > 1250 else 1.0
        line1s, line2s, _raw, error = _rst_tesseract_candidates(
            rgb, pair, scale)
        assert not error
        line1, line2, _repair, _raw_line2, decoded = \
            _rst_decode_candidates(line1s, line2s)
        key = "5" if line2[0] == "5" else "E"
        assert (line1, line2) == EXPECTED_BY_FIRST_DIGIT[key]
        assert decoded.returncode == 0


def test_line2_candidate_repairs_digit_in_country_code():
    fixed = "ZF40004998POL8101028M3203031" + "<" * 14 + "00"
    assert len(fixed) == 44
    raw = fixed.replace("POL", "P0L")
    candidates = _mrz_line2_candidates(raw)
    assert fixed in [item[3] for item in candidates]


def test_adaptive_xpad_keeps_frame_sides_only():
    frame = np.full((60, 200, 3), 255, dtype=np.uint8)
    frame[:, :8] = 0
    frame[:, -8:] = 0
    assert _rst_adaptive_xpad(frame, ((10, 20), (30, 40)), 1.0, 20) == (20, 20)
    plain = np.full((60, 200, 3), 250, dtype=np.uint8)
    assert _rst_adaptive_xpad(plain, ((10, 20), (30, 40)), 1.0, 20) == (0, 0)
