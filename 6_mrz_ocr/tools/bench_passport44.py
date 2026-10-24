#!/usr/bin/env python3
"""End-to-end evaluation of the current CNN build on tmp/PassPortPhoto (44 photos).

Why this script exists
----------------------
`bench_real.py` reports against the *board's* fixmrz weak label, which is a
PaddleOCR reading -- using it as ground truth makes the comparison circular
(the board's CRNN and our recogniser both start from the same crop).  This
script separates the two questions that keep getting mixed together:

  1. **Does the recogniser read the band right?**  Measured against
     `data/gt_passport_manual.json` (hand-read, 18 photos).  Only line 2 can
     be scored objectively-ish, and even there the check digits only localise
     an error to a value class, so a `line2_verified=0` record may itself be
     wrong.  Reported separately, never averaged together.
  2. **Is the reading self-consistent?**  The five ICAO check digits plus the
     two veto flags.  Available for all 44 with no ground truth at all -- but
     §14 of the report showed this is a *rejector*, not an acceptor: 19 of 28
     checksum-valid readings in the assets corpus had a fictional nationality.

The band is cropped from the board's TextBox rows, i.e. this measures the
recogniser **given a working locator**.  It deliberately does NOT measure the
in-C band locator on a full phone photo: that locator returns the whole page
(band height ~880 of 960) and is not the path the app takes -- Android runs
its own Kotlin locator and calls the recogniser on the crop.

Usage:
    python3 tools/bench_passport44.py [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PHOTOS = ROOT.parent / "tmp" / "PassPortPhoto"
OMTEST = ROOT.parent / "tmp" / "omtest"
TOOL = ROOT / "build" / "mrz_ocr_cnn_tool"
GT = ROOT / "data" / "gt_passport_manual.json"

W = (7, 3, 1)


def val(c: str) -> int:
    if c == "<":
        return 0
    if "0" <= c <= "9":
        return ord(c) - 48
    return ord(c) - 55


def cd(s: str) -> int:
    return sum(val(c) * W[i % 3] for i, c in enumerate(s)) % 10


def line2_cksum(l2: str) -> bool:
    if len(l2) != 44:
        return False
    try:
        return (cd(l2[0:9]) == int(l2[9]) and cd(l2[13:19]) == int(l2[19])
                and cd(l2[21:27]) == int(l2[27]) and cd(l2[28:42]) == int(l2[42])
                and cd(l2[0:10] + l2[13:20] + l2[21:43]) == int(l2[43]))
    except ValueError:
        return False


def char_acc(a: str, b: str) -> float:
    n = max(len(a), len(b))
    if not n:
        return 0.0
    return sum((a[i] if i < len(a) else "_") == (b[i] if i < len(b) else "_")
               for i in range(n)) / n


def find_txt(stem: str) -> Path | None:
    for d in sorted(OMTEST.glob("pic_scan*")):
        p = d / f"{stem}.jpg-result.txt"
        if p.exists():
            return p
    return None


def band_rows(txt: Path) -> tuple[int, int] | None:
    """Board TextBox rows -> absolute y range in the original photo.

    The board logs `onlymrz: crop bottom 1/3 minus 50px rows: 1440x960 ->
    1440x270`, and its TextBox y values are relative to that crop.  The
    mapping that actually lands on the glyphs is +590 for a 960-row photo,
    verified independently by row projection: on 2026-9-29-O the two text
    rows sit at y 678..698 and 713..736, while TextBox spans 80..155 and
    80+590=670 .. 155+590=745 -- symmetric ~8 px padding around the text.
    """
    s = txt.read_text(errors="replace")
    pts = [(int(a), int(b)) for a, b in re.findall(r"\[x: (\d+), y: (\d+)\]", s)]
    boxes = [pts[i:i + 4] for i in range(0, len(pts), 4)]
    if len(boxes) < 2:
        return None
    return min(p[1] for p in boxes[0]), max(p[1] for p in boxes[1])


def run_tool(ppm: Path, crnn: bool = False) -> tuple[dict, float, str]:
    # MRZ_OCR_LINE_MODELS=1 selects the real-photo line1/line2 weight pair.
    # WITHOUT IT the tool runs the synthetic "template" weights and every
    # reading on a real photo is noise (conf ~61, 0/41 checksum).  Android
    # sets exactly this flag for backend 1 (CNN) and 2 (CRNN) in
    # mrz_jni.cpp:94, so a host measurement that forgets it is measuring a
    # configuration the app never runs.
    env = dict(os.environ)
    env["MRZ_OCR_LINE_MODELS"] = "1"
    env["MRZ_OCR_PAR"] = "0"
    env["MRZ_OCR_CRNN"] = "1" if crnn else "0"
    env["MRZ_OCR_TIMING"] = "1"
    t0 = time.perf_counter()
    r = subprocess.run([str(TOOL), str(ppm)], capture_output=True, text=True,
                       env=env, timeout=120)
    wall = (time.perf_counter() - t0) * 1000
    kv = {}
    for ln in r.stdout.splitlines():
        if ":" in ln:
            k, _, v = ln.partition(":")
            kv[k.strip()] = v.strip()
    stage = ""
    m = re.search(r"TIMING gray=([\d.]+)ms band\+deskew=([\d.]+)ms "
                  r"resample=([\d.]+)ms decode=([\d.]+)ms", r.stderr)
    if m:
        stage = m.group(0)
        kv["_stage_ms"] = round(sum(float(x) for x in m.groups()), 2)
    return kv, wall, stage


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--crnn", action="store_true", help="use the CRNN head instead of CNN")
    ap.add_argument("--repeat", type=int, default=1,
                    help="run each crop N times, keep the fastest (removes scheduler noise)")
    a = ap.parse_args()

    gt = {r["id"]: r for r in json.loads(GT.read_text())["records"]}
    tmp_ppm = ROOT / "data" / "_bench44.ppm"
    rows = []
    skipped = []

    for jpg in sorted(PHOTOS.glob("*.jpg")):
        stem = jpg.stem
        txt = find_txt(stem)
        if txt is None:
            skipped.append((stem, "no board result.txt"))
            continue
        yr = band_rows(txt)
        if yr is None:
            # The board read fewer than two rows.  On this corpus that is
            # always a TD1 card (3x30), not a broken photo: the MRZ visible
            # in frame is a single 30-character line, so there is no second
            # row to locate and the image is out of scope for a TD3 test.
            tl = re.findall(r"textLine\[\d\]\(([^)]*)\)",
                            txt.read_text(errors="replace"))
            kind = ("single-line MRZ (TD1/TD2 card), not a TD3 passport"
                    if tl else "board found no MRZ row")
            skipped.append((stem, kind))
            continue
        rgb = cv2.imread(str(jpg))
        H, Wd = rgb.shape[:2]
        off = H - (H // 3) - 50
        y1 = yr[0] + off - 4
        y2 = yr[1] + off + 5
        band = rgb[max(0, y1):min(H, y2), 0:Wd]
        if band.shape[0] < 20:
            skipped.append((stem, f"band too short ({band.shape[0]}px)"))
            continue
        Image.fromarray(cv2.cvtColor(band, cv2.COLOR_BGR2RGB)).save(
            str(tmp_ppm), format="PPM")

        best = None
        for _ in range(max(1, a.repeat)):
            kv, wall, stage = run_tool(tmp_ppm, a.crnn)
            if best is None or wall < best[1]:
                best = (kv, wall, stage)
        kv, wall, stage = best

        l1 = kv.get("result.line1", "")
        l2 = kv.get("result.line2", "")
        struct = kv.get("result.struct2", "") == "1"
        nat = kv.get("result.natsuspect", "") == "1"
        issuer = kv.get("result.issuersusp", "") == "1"
        rec = {
            "id": stem,
            "ok": kv.get("result.ok", "?"),
            "line1": l1, "line2": l2,
            "conf1": kv.get("result.conf1", ""), "conf2": kv.get("result.conf2", ""),
            "ambiguous": kv.get("result.ambiguous2", "") == "1",
            "struct": struct, "nat": nat, "issuer": issuer,
            "cksum": line2_cksum(l2),
            "wall_ms": round(wall, 2),
            "stage_ms": kv.get("_stage_ms"),
            "band_h": int(band.shape[0]),
            "stage": stage,
        }
        rec["trustworthy"] = rec["cksum"] and struct and not nat
        # line 1 has no check digits, so the only arithmetic-free test on it
        # is the cross-line one: the issuing state (line1 cols 2..4) is
        # normally the same code as the nationality (line2 cols 10..12).
        # Advisory, not a veto -- a refugee travel document legitimately
        # differs -- but on this corpus it is the ONLY signal that catches a
        # single-character error in line 1 (confidence does not: a line 1
        # with N->Y still scores 98).
        rec["l1_country_ok"] = (len(l1) == 44 and len(l2) == 44
                                and l1[2:5] == l2[10:13])
        g = gt.get(stem)
        if g:
            rec["gt"] = True
            rec["l1_acc"] = round(char_acc(g["line1"], l1), 4)
            rec["l2_acc"] = round(char_acc(g["line2"], l2), 4)
            rec["l2_exact"] = (g["line2"] == l2)
            rec["gt_verified"] = g.get("line2_verified", 0)
            rec["gt_len2"] = len(g["line2"])
        else:
            rec["gt"] = False
        rows.append(rec)
        print(f"{stem:18s} ok={rec['ok']:32s} ck={int(rec['cksum'])} "
              f"st={int(struct)} nat={int(nat)} "
              f"l2={rec.get('l2_acc', '-')} {rec['wall_ms']:6.1f}ms")

    n = len(rows)
    if not n:
        print("no rows", file=sys.stderr)
        return 1

    def pct(x, d):
        return f"{x}/{d} ({100.0 * x / d:.1f}%)" if d else "0/0"

    print(f"\n=== n={n}  skipped={len(skipped)} ===")
    for s in skipped:
        print("  skip", s)
    ck = sum(r["cksum"] for r in rows)
    tr = sum(r["trustworthy"] for r in rows)
    ok = sum(1 for r in rows if r["ok"] == "OK")
    amb = sum(r["ambiguous"] for r in rows)
    print(f"tool OK          : {pct(ok, n)}")
    print(f"line2 checksum   : {pct(ck, n)}")
    print(f"trustworthy      : {pct(tr, n)}")
    print(f"  struct veto    : {sum(not r['struct'] for r in rows)} rejected")
    print(f"  nat suspect    : {sum(r['nat'] for r in rows)} flagged")
    print(f"  issuer suspect : {sum(r['issuer'] for r in rows)} flagged")
    print(f"  ambiguous      : {amb}")
    print(f"line1 country == line2 nat : {pct(sum(r['l1_country_ok'] for r in rows), n)}")

    walls = sorted(r["wall_ms"] for r in rows)
    stages = sorted(r["stage_ms"] for r in rows if r["stage_ms"])
    def q(v, p):
        if not v:
            return 0.0
        i = min(len(v) - 1, int(round(p * (len(v) - 1))))
        return v[i]
    print(f"\nwall ms (process incl. spawn): mean {sum(walls)/len(walls):.2f} "
          f"p50 {q(walls,0.5):.2f} p95 {q(walls,0.95):.2f} min {walls[0]:.2f}")
    if stages:
        print(f"stage sum ms (in-process)   : mean {sum(stages)/len(stages):.2f} "
              f"p50 {q(stages,0.5):.2f} p95 {q(stages,0.95):.2f} min {stages[0]:.2f}")

    gtr = [r for r in rows if r["gt"]]
    if gtr:
        ng = len(gtr)
        print(f"\n=== GT subset n={ng} (hand-read) ===")
        print(f"  line1 char acc : {sum(r['l1_acc'] for r in gtr)/ng:.4f}")
        print(f"  line2 char acc : {sum(r['l2_acc'] for r in gtr)/ng:.4f}")
        print(f"  line2 exact    : {pct(sum(r['l2_exact'] for r in gtr), ng)}")
        print(f"  line2 checksum : {pct(sum(r['cksum'] for r in gtr), ng)}")
        print(f"  trustworthy    : {pct(sum(r['trustworthy'] for r in gtr), ng)}")
        print("  per-record (only the ones that are not exact):")
        for r in gtr:
            if not r["l2_exact"]:
                g = gt[r["id"]]
                diff = [i for i in range(44)
                        if (r["line2"][i] if i < len(r["line2"]) else "_")
                        != g["line2"][i]]
                print(f"    {r['id']:18s} v={r['gt_verified']} len={len(r['line2'])} "
                      f"diff@{diff[:6]} got={r['line2'][:44]}")

    if a.json:
        Path(a.json).write_text(json.dumps(
            {"n": n, "rows": rows, "skipped": skipped}, indent=1))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
