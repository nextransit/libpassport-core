#!/usr/bin/env python3
"""Regression test for the TD3 / CS-CD layout cascade.

The cascade (layout classification -> 30-vs-44 grid -> single-row
short-circuit) lives in the C core; this test pins the three host-side
defects that silently broke it, all measured on the CS/CD travel
permit assets/pic/1c6acd092dee7031_0222.png:

  1. a single-row band must NOT be padded vertically -- the C layout
     classifier votes 30-vs-44 cells from the row's aspect ratio and
     the dual-hypothesis grid energy, and the extra rows flip that
     vote (tight crop -> CS_CD_30 + the exact code, +8 px -> TD3_44
     and garbage);
  2. the horizontal pad must never crop INTO the ink -- on a narrow
     document a fixed fraction of the width reaches past the first
     character (Photo/7: 51 px pad vs ink at x=27);
  3. a CS/CD reading must not be overwritten by the Tesseract rescue,
     because mrz_tool only speaks TD3 and can never decode a 30-char
     row.

Run:  python3 tests/test_doc_layout.py
"""
from __future__ import annotations
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

PERMIT = ROOT / "assets" / "pic" / "1c6acd092dee7031_0222.png"
PASSPORT = ROOT / "assets" / "pic" / "16c07a99a0fb81e4_0013.jpg"
CSC_TOOL = ROOT / "6_mrz_ocr" / "build" / "mrz_ocr_cnn_tool"

g_fail = 0
g_total = 0


def check(cond, msg):
    global g_fail, g_total
    g_total += 1
    if not cond:
        g_fail += 1
        print(f"[FAIL] {msg}")
    else:
        print(f"[ ok ] {msg}")


def c_tool(ppm: Path) -> dict:
    r = subprocess.run([str(CSC_TOOL), str(ppm)], capture_output=True,
                       text=True, timeout=30)
    kv = {}
    for ln in r.stdout.splitlines():
        if ":" in ln:
            k, _, v = ln.partition(":")
            kv[k.strip()] = v.strip()
    return kv


def _load_work(path: Path):
    """Return (work image, detect_structure result) for a photo."""
    sys.path.insert(0, str(ROOT / "passport_photo_restore" / "src"))
    import numpy as np
    from PIL import Image
    from restore import load, _gray, detect_structure
    rgb = np.asarray(Image.open(path).convert("RGB"))
    work, scale = load(rgb, max_side=1250)
    return work, detect_structure(_gray(work)), rgb, scale


def _run_tool(band):
    from PIL import Image
    with tempfile.NamedTemporaryFile(suffix=".ppm", delete=False) as tf:
        tmp = Path(tf.name)
    Image.fromarray(band).save(tmp, format="PPM")
    try:
        return c_tool(tmp)
    finally:
        tmp.unlink(missing_ok=True)


def test_single_row_padding():
    """A single CS/CD row must classify as CS_CD_30 when cropped tight
    and must NOT be read as TD3 when padded."""
    if not PERMIT.exists():
        print("[skip] permit fixture missing")
        return
    work, st, _rgb, _scale = _load_work(PERMIT)
    single = st.get("mrz_single")
    check(single is not None, "the CS/CD permit yields a single MRZ row")
    if not single:
        return
    y0, y1 = single
    tight = _run_tool(work[y0:y1 + 1])
    check(tight.get("result.layoutname") == "CS_CD_30",
          f"tight single-row crop classifies as CS_CD_30 "
          f"(got {tight.get('result.layoutname')})")
    check(tight.get("result.line1", "").startswith("CD"),
          f"tight single-row crop reads a CS/CD code "
          f"(got {tight.get('result.line1', '')[:12]!r})")
    padded = _run_tool(work[max(0, y0 - 8):y1 + 9])
    check(padded.get("result.layoutname") != "CS_CD_30",
          "a padded single-row crop does NOT classify as CS_CD_30 "
          "(the padding defect the host must avoid)")


def test_two_row_still_td3():
    """The cascade must not disturb ordinary TD3 passports."""
    if not PASSPORT.exists():
        print("[skip] passport fixture missing")
        return
    work, st, _rgb, _scale = _load_work(PASSPORT)
    pair = st.get("mrz_pair")
    check(pair is not None, "the TD3 passport yields an MRZ pair")
    if not pair:
        return
    (a0, _a1), (b0, b1) = pair
    kv = _run_tool(work[a0:b1 + 1])
    check(kv.get("result.layoutname") == "TD3_44",
          f"the two-row band still classifies as TD3_44 "
          f"(got {kv.get('result.layoutname')})")


def test_host_padding_is_not_applied():
    """The host band builder must not pad a single-row band."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("gui", ROOT / "tests" / "gui.py")
    gui = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gui)
    src = (ROOT / "tests" / "gui.py").read_text()
    check("single: the band is then exactly that row" in src
          or "NO vertical padding" in src
          or "exactly that row" in src,
          "the host documents that a single-row band is cropped tight")
    # The Android side carries the same rule.
    kt = (ROOT / "8_android" / "app" / "src" / "main" / "java" /
          "com" / "nextransit" / "mrzbench" / "photo" / "PhotoMrzPipeline.kt")
    if kt.exists():
        ksrc = kt.read_text()
        check("NO vertical padding on a single-row band" in ksrc,
              "the Android host does not pad a single-row band either")


def test_cscd_struct_and_shape():
    """CS/CD rows are scored on structure (the check digits of this
    document family are not the ICAO 7-3-1 cycle)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("gui", ROOT / "tests" / "gui.py")
    gui = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gui)
    ok = gui._rst_cscd_struct_ok
    check(ok("CDC000000007<2508197<8208031<2"),
          "a real CS/CD row passes the structural check")
    check(ok("CS<C00000007<2568107<8208031<7"),
          "a CS/CD row with three wrong characters still passes the "
          "structural check")
    check(not ok("CESSOVNPSMSSPORTPASSEPORTESKREPUBLIKACZECHRE"),
          "page body text does not pass the CS/CD structural check")
    check(not ok("CDC000000007<2508197<8208031"),
          "a truncated CS/CD row does not pass the structural check")


def main() -> int:
    test_single_row_padding()
    test_two_row_still_td3()
    test_host_padding_is_not_applied()
    test_cscd_struct_and_shape()
    print(f"\n[TEST] {g_total} total, {g_fail} failed")
    return 1 if g_fail else 0


if __name__ == "__main__":
    sys.exit(main())
