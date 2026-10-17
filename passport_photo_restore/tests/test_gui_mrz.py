from pathlib import Path
import shutil
import sys

import cv2
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))

from gui import (  # noqa: E402
    MRZ_TOOL,
    _rst_decode_candidates,
    _rst_tesseract_candidates,
)


EXPECTED = (
    "POCHNJIA<<LIMIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
    "EE20426536CHN7410024M2808303LMNGMBKCMDPBA970",
)


@pytest.mark.skipif(
    shutil.which("tesseract") is None or not MRZ_TOOL.exists(),
    reason="requires tesseract OCR-B and mrz_tool",
)
def test_photo_mrz_auto_ocr_and_decode():
    pairs = {
        "3": ((618, 632), (653, 670)),
        "4": ((618, 632), (653, 670)),
        "5": ((618, 632), (653, 670)),
        "6": ((618, 632), (653, 670)),
        "7": ((316, 325), (335, 346)),
        "8": ((616, 633), (654, 674)),
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
        assert (line1, line2) == EXPECTED
        assert decoded.returncode == 0
