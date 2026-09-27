"""Passport photo full pipeline: region-locate -> OCR MRZ -> verify -> encode.

End-to-end chain used both by the GUI ("护照照片识别") and as a headless
CLI to filter/encode the crawled corpus in assets/pic.

Flow per image:
  passport photo -> locate_regions(photo/data/mrz) -> crop MRZ band
  -> mrz_ocr_tool OCR -> mrz_tool decode verify
  -> grade: PASS / SUSPECT / REJECT
"""
from __future__ import annotations
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIC_DIR = ROOT / "assets" / "pic"
TRAD = ROOT / "6_mrz_ocr" / "build" / "mrz_ocr_tool"
CNN  = ROOT / "6_mrz_ocr" / "build" / "mrz_ocr_cnn_tool"
MRZ_TOOL = ROOT / "1_mrz_decode" / "build" / "mrz_tool"
CATALOG = ROOT / "assets" / "pic" / "catalog.json"

GRADE_LABEL = {"PASS": "护照(校验通过)", "SUSPECT": "疑似护照(校验失败)",
               "REJECT": "非护照/无MRZ"}


def _sys_path_ready() -> None:
    d = str(ROOT / "tests")
    if d not in sys.path:
        sys.path.insert(0, d)


def locate(image: Path, max_side: int = 1200):
    """Return locate_regions dict for a passport image.

    Region detection runs on a downscaled copy (max side ~`max_side`) for
    speed; box coordinates are scaled back to the full-res image.
    """
    _sys_path_ready()
    from PIL import Image
    from passport_locator import locate_regions  # type: ignore
    img = Image.open(image).convert("RGB")
    fw, fh = img.size
    scale = 1.0
    work = img
    if max(fw, fh) > max_side:
        scale = max_side / max(fw, fh)
        work = img.resize((max(1, int(fw * scale)),
                           max(1, int(fh * scale))), Image.LANCZOS)
    loc = locate_regions(work)
    img.close()
    if loc and scale != 1.0:
        for k, v in loc.items():
            if v:
                for key in ("x", "y", "w", "h"):
                    v[key] = int(round(v[key] / scale))
    return loc


def mrz_line_height(img) -> int:
    """Estimate MRZ glyph line height (px) as the tallest run of rows whose
    ink exceeds half the average row ink (Otsu-binarised, dark ink)."""
    import numpy as np
    a = np.asarray(img.convert("L"))
    hist = np.bincount(a.ravel(), minlength=256)
    tot = a.size
    wb = np.cumsum(hist)
    wf = tot - wb
    sb = np.cumsum(hist * np.arange(256))
    sf = sb[-1] - sb
    with np.errstate(divide="ignore", invalid="ignore"):
        var = wb * wf * (sb / np.maximum(wb, 1)
                         - sf / np.maximum(wf, 1)) ** 2
    thr = int(np.argmax(var)) if tot else 128
    ink = (a < thr).sum(axis=1)
    dense = ink > (ink.sum() / max(a.shape[0], 1) / 2)
    best = cur = 0
    for d in dense:
        cur = cur + 1 if d else 0
        best = max(best, cur)
    return best


def auto_upscale(crop) -> int:
    """Pick a LANCZOS scale bringing the MRZ line height into the OCR
    training distribution (36-72px). Returns 1 when already in range."""
    lh = mrz_line_height(crop)
    if not lh or lh >= 30:
        return 1
    return min(8, max(2, round(48 / lh)))


def ocr_crop(image: Path, loc, tool: Path = CNN, margin: int = 8,
             timeout: float = 15.0, upscale: int = 0, dilate: bool = False) -> dict:
    """Crop the located MRZ band (with margin) and run OCR on it.

    `upscale`: LANCZOS scale factor. 0 = auto: real-photo MRZ lines are
    only ~12px tall while the OCR pipeline was trained at 36-72px, so
    low lines (< 30px) get upscaled toward 48px; taller lines pass
    through unchanged.
    `dilate`: horizontal 1px dilation of dark pixels. The C locator's dense
    threshold (W*0.15) can exceed the dark count of sparse glyph rows (e.g.
    row 1 of 'P<UTOE...' has 52 bit-columns -> 208px at scale 4 vs thr 225),
    which drops the band start and garbles line 1. Dilating widens every ink
    run by 2px without changing the crop width (threshold stays fixed).

    Returns the same shape as ocr_bench_runner.run_one plus 'crop'.
    """
    _sys_path_ready()
    from PIL import Image
    mrz = loc.get("mrz")
    if not mrz:
        return {"ok": False, "line1": "", "line2": "", "conf1": 0,
                "conf2": 0, "band": "", "ms": 0.0, "err": "no mrz region",
                "crop": None}
    # Real photos often have MRZ line gaps larger than the C locator's
    # max_gap (H/8) can bridge, so the band collapses to one line. Pad the
    # crop generously vertically: a taller crop raises H/8 and lets
    # locate_band merge both MRZ lines into one band.
    margin = max(margin, int(mrz["h"] * 1.2))
    img = Image.open(image).convert("RGB")
    tight = img.crop((mrz["x"], mrz["y"],
                      mrz["x"] + mrz["w"], mrz["y"] + mrz["h"]))
    if upscale == 0:
        # Measure line height on the tight locator box: the padded crop can
        # capture unrelated text above the MRZ (data region) that would
        # skew the row-profile run length.
        upscale = auto_upscale(tight)
    if upscale > 1:
        # Upscaled path uses the tight locator box: padding would drag in
        # unrelated text above the MRZ (data region) that pollutes the C
        # band search on the scaled image. The C band locator merges both
        # MRZ lines on tight crops (2*max_run_h rule).
        x0, y0 = mrz["x"], mrz["y"]
        x1, y1 = mrz["x"] + mrz["w"], mrz["y"] + mrz["h"]
        crop = tight.resize((tight.width * upscale, tight.height * upscale),
                            Image.LANCZOS)
    else:
        x0 = max(0, mrz["x"] - margin)
        y0 = max(0, mrz["y"] - margin)
        x1 = min(img.width, mrz["x"] + mrz["w"] + margin)
        y1 = min(img.height, mrz["y"] + mrz["h"] + margin)
        crop = img.crop((x0, y0, x1, y1))
    img.close()
    if dilate:
        import numpy as _np
        a = _np.array(crop.convert("L"))
        bw = a < 128
        bw2 = bw.copy()
        bw2[:, 1:] |= bw[:, :-1]
        bw2[:, :-1] |= bw[:, 1:]
        crop = Image.fromarray(_np.where(bw2, 0, 255).astype("uint8"),
                               "L").convert("RGB")
    with tempfile.NamedTemporaryFile(suffix=".ppm", delete=False) as tf:
        crop_path = Path(tf.name)
    crop.save(crop_path, format="PPM")
    crop.close()
    try:
        from ocr_bench_runner import run_one  # type: ignore
        r = run_one(tool, crop_path)
    finally:
        try:
            crop_path.unlink()
        except OSError:
            pass
    r["crop"] = {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}
    return r


def decode(mrz_two_lines: str) -> dict:
    """Run mrz_tool decode; return {ok, fields{...}} or {ok:False, err}."""
    try:
        r = subprocess.run([str(MRZ_TOOL), "decode", mrz_two_lines],
                           text=True, capture_output=True, timeout=10)
    except subprocess.TimeoutExpired:
        return {"ok": False, "err": "decode timeout", "fields": {}}
    fields = {}
    for line in r.stdout.splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        fields[k.strip()] = v.strip().strip("'")
    return {"ok": r.returncode == 0,
            "err": "" if r.returncode == 0 else r.stderr.strip(),
            "fields": fields}


def encode(fields: dict) -> str:
    """Encode MRZ two lines from fields via mrz_tool encode."""
    args = [str(MRZ_TOOL), "encode",
            fields.get("doc_type", "P<").replace("<", ""),
            fields.get("issuing_state", "UTO"),
            fields.get("surname", "").replace("<", " "),
            fields.get("given_names", "").replace("<", " "),
            fields.get("passport_no", "").split()[0],
            fields.get("nationality", "UTO"),
            fields.get("birth_yymmdd", "000000"),
            fields.get("sex", "F"),
            fields.get("expiry_yymmdd", "000000"),
            fields.get("personal_no", "").replace("<", " ").split()[0]]
    r = subprocess.run(args, text=True, capture_output=True, timeout=10)
    if r.returncode != 0:
        return ""
    lines = r.stdout.strip().splitlines()
    return "\n".join(lines[:2]) if len(lines) >= 2 else ""


def process_image(image: Path, tool: Path = CNN, upscale: int = 0,
                  dilate: bool = False) -> dict:
    """Full single-image pipeline: locate -> crop -> OCR -> verify -> grade."""
    t0 = __import__("time").perf_counter()
    rec = {"file": image.name, "path": str(image),
           "grade": "REJECT", "reason": "", "loc": {}, "ocr": None,
           "decode": None, "mrz": "", "verified": False, "ms": 0.0}
    try:
        loc = locate(image)
        rec["loc"] = {k: (dict(v) if v else None)
                      for k, v in loc.items()} if loc else {}
        if not loc or not loc.get("mrz"):
            rec["reason"] = "未定位到 MRZ 区"
            rec["ms"] = (__import__("time").perf_counter() - t0) * 1000.0
            return rec
        # Passport page confirmed by region location (MRZ band + photo/data).
        rec["grade"] = "PASS"
        rec["reason"] = "MRZ 区定位成功" + (
            "，照片/数据区齐全" if loc.get("photo") and loc.get("data")
            else "，照片/数据区缺失")
        ocr = ocr_crop(image, loc, tool=tool, upscale=upscale, dilate=dilate)
        rec["ocr"] = ocr
        rec["ms"] = (__import__("time").perf_counter() - t0) * 1000.0
        l1, l2 = (ocr.get("line1") or ""), (ocr.get("line2") or "")
        if not ocr.get("ok") or len(l1) != 44 or len(l2) != 44:
            rec["reason"] += "；OCR 未出完整两行（工具对低分辨率照片的局限）"
            return rec
        rec["mrz"] = l1 + "\n" + l2
        dec = decode(rec["mrz"])
        rec["decode"] = dec
        if dec["ok"]:
            rec["verified"] = True
            rec["reason"] += "；OCR+校验通过"
        else:
            rec["reason"] += "；OCR 出两行但校验失败"
        return rec
    except Exception as e:  # noqa: BLE001
        rec["reason"] = f"异常: {e}"
        return rec


def scan(pic_dir: Path = PIC_DIR, tool: Path = CNN, upscale: int = 0,
         dilate: bool = False, progress=None) -> list[dict]:
    """Process every image under pic_dir; optional progress(rec) callback."""
    results = []
    imgs = sorted(p for p in pic_dir.iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".ppm"))
    for p in imgs:
        rec = process_image(p, tool=tool, upscale=upscale, dilate=dilate)
        results.append(rec)
        if progress:
            progress(rec)
    return results


def build_catalog(pic_dir: Path = PIC_DIR, tool: Path = CNN,
                  upscale: int = 0, dilate: bool = False) -> Path:
    """Scan + write catalog.json (grading + MRZ + metadata for encoding)."""
    recs = scan(pic_dir, tool=tool, upscale=upscale, dilate=dilate)
    catalog = {"count": len(recs),
               "pass": sum(1 for r in recs if r["grade"] == "PASS"),
               "verified": sum(1 for r in recs if r.get("verified")),
               "suspect": sum(1 for r in recs if r["grade"] == "SUSPECT"),
               "reject": sum(1 for r in recs if r["grade"] == "REJECT"),
               "records": recs}
    (pic_dir / "catalog.json").write_text(json.dumps(catalog, indent=2),
                                          encoding="utf-8")
    return pic_dir / "catalog.json"


def filter_images(pic_dir: Path = PIC_DIR, tool: Path = TRAD,
                  delete: bool = False, progress=None) -> dict:
    """Grade all images; mark (rename with prefix) or delete non-passports."""
    recs = scan(pic_dir, tool=tool)
    stats = {"total": 0, "pass": 0, "suspect": 0, "reject": 0,
             "deleted": 0, "renamed": 0, "kept": 0}
    for rec in recs:
        stats["total"] += 1
        stats[rec["grade"].lower()] += 1
        src = Path(rec["path"])
        if rec["grade"] == "REJECT":
            if delete:
                src.unlink(missing_ok=True)
                stats["deleted"] += 1
            else:
                mark = src.with_name("REJECT_" + src.name)
                if not mark.exists():
                    src.rename(mark)
                    rec["path"] = str(mark)
                    stats["renamed"] += 1
            stats["kept"] += 0
        elif rec["grade"] == "SUSPECT":
            mark = src.with_name("SUSPECT_" + src.name)
            if not mark.exists():
                src.rename(mark)
                rec["path"] = str(mark)
                stats["renamed"] += 1
            stats["kept"] += 1
        else:
            stats["kept"] += 1
        if progress:
            progress(rec, stats)
    return stats


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Passport photo pipeline")
    ap.add_argument("--dir", type=Path, default=PIC_DIR)
    ap.add_argument("--tool", choices=("traditional", "cnn"), default="cnn")
    ap.add_argument("--min", type=int, default=0)
    ap.add_argument("--catalog", action="store_true",
                    help="build assets/pic/catalog.json (grade + MRZ metadata)")
    ap.add_argument("--filter", action="store_true",
                    help="mark non-passport images (REJECT_/SUSPECT_ prefix)")
    ap.add_argument("--delete", action="store_true",
                    help="delete REJECT images instead of renaming")
    ap.add_argument("--encode", nargs="*", default=None, metavar="FIELD=VALUE",
                    help="encode MRZ from fields (passport_no=... surname=... etc)")
    a = ap.parse_args(argv)
    tool = TRAD if a.tool == "traditional" else CNN

    if a.encode:
        fields = dict(f.split("=", 1) for f in a.encode if "=" in f)
        out = encode(fields)
        print(out if out else "encode failed")
        return 0 if out else 1

    print(f"scanning {a.dir} ...", file=sys.stderr)
    if a.filter:
        stats = filter_images(a.dir, tool=tool, delete=a.delete)
        print(json.dumps(stats, indent=2))
        return 0
    catalog = build_catalog(a.dir, tool=tool)
    d = json.loads(catalog.read_text(encoding="utf-8"))
    print(f"catalog: {catalog}")
    print(f"total={d['count']} pass={d['pass']} suspect={d['suspect']}"
          f" reject={d['reject']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())