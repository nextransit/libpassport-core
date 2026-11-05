#!/usr/bin/env python3
"""Summarise the on-device APK algorithm test (2026-11-05).

Parses the three logcat dumps (cnn / crnn / fuse runs of
PhotoMrzDeviceTest on 192.168.26.131), attributes every RESULT line to
its photo set by file name, scores the recognised lines against the
per-set ground truth and prints accuracy + latency per set x engine.

Sets
  passport  tmp/PassPortPhoto 44 photos  GT tests/gt_photo_passport_official.json
  camshots  Photo/ 6 camera shots of ONE document  GT gt_review_loop.xlsx
  pic       assets/pic 58 photos            GT tests/gt_assets_pic_official.json
"""

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

W = [7, 3, 1]


def cd(seg):
    t = 0
    for i, c in enumerate(seg):
        v = (ord(c) - 48) if c.isdigit() else ((ord(c) - 55) if 'A' <= c <= 'Z' else 0)
        t += v * W[i % 3]
    return t % 10


def line2_ok(l2):
    if len(l2) != 44:
        return False
    try:
        return (cd(l2[0:9]) == int(l2[9]) and cd(l2[13:19]) == int(l2[19])
                and cd(l2[21:27]) == int(l2[27]) and cd(l2[28:42]) == int(l2[42])
                and cd(l2[0:10] + l2[13:20] + l2[21:43]) == int(l2[43]))
    except (ValueError, IndexError):
        return False


def load_gt():
    out = {}
    d = json.loads((REPO / "tests/gt_photo_passport_official.json").read_text())
    for k, v in d["gt"].items():
        if isinstance(v, dict) and v.get("l1"):
            out[k] = (v["l1"], v.get("l2", ""))
    # camshots: the xlsx GT (one document, six shots)
    cam = ("POCHNJIA<<LIMIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
           "EE20426536CHN7410024M2808303LMNGMBKCMDPBA970")
    d2 = json.loads((REPO / "tests/gt_assets_pic_official.json").read_text())
    g2 = d2["gt"] if isinstance(d2, dict) and "gt" in d2 else d2
    for k, v in g2.items():
        if isinstance(v, dict) and v.get("l1"):
            out[k] = (v["l1"], v.get("l2", ""))
    return out, cam


def set_of(raw_name):
    # 外部照片的 RESULT 名带 external/ 前缀;内置样张(PhotoScan、assets
    # 的 Photo/)不是本次测试对象,返回 None 被排除。
    if not raw_name.startswith("external/"):
        return None
    name = raw_name.split("/")[-1]
    if name.startswith("2026-9-"):
        return "passport"
    if name.startswith("IMG_"):
        return "camshots"
    return "pic"


LINE = re.compile(
    r"RESULT (\S+) \| (.+?) \| decode=(\w+) \| wall=(\d+)ms \| total=(\d+)ms"
    r" \| backend=(\S+) \| ocr=(\d+)ms \| structure=(\d+)ms.*?verdict=(\w+)"
    r" \| L1=([^|]*) \| L2=([^|]*)")


def parse(log):
    rows = []
    for ln in Path(log).read_text(errors="replace").splitlines():
        if "RESULT " not in ln:
            continue
        m = LINE.search(ln)
        if not m:
            continue
        name, method, decode, wall, total, backend, ocr, structure, verdict, l1, l2 = m.groups()
        rows.append({
            "name": name.split("/")[-1], "raw": name, "method": method,
            "decode": decode == "true", "wall": int(wall), "total": int(total),
            "ocr": int(ocr), "structure": int(structure), "backend": backend,
            "verdict": verdict, "l1": l1.strip(), "l2": l2.strip(),
        })
    return rows


def pct(n, d):
    return f"{n}/{d}" if d else "-", (100.0 * n / d if d else 0.0)


def char_acc(pred, gt):
    if len(pred) != len(gt) or not gt:
        return 0, 0
    ok = sum(1 for a, b in zip(pred, gt) if a == b)
    return ok, len(gt)


def lat(vals):
    if not vals:
        return "-"
    v = sorted(vals)
    n = len(v)
    return (f"p50={v[n//2]:.0f} p90={v[int(n*0.9)]:.0f} "
            f"p95={v[min(n-1, int(n*0.95+0.5))]:.0f} max={v[-1]:.0f}")


def main():
    gt, cam_gt = load_gt()
    logs = {"cnn": "/tmp/log_cnn.txt", "crnn": "/tmp/log_crnn.txt",
            "fuse": "/tmp/log_fuse.txt", "auto(产品路径)": "/tmp/log_auto.txt"}
    sets = ("passport", "camshots", "pic")
    print("=" * 100)
    print("Android APK 算法测试 — 192.168.26.131, 产品路径(PhotoMrzPipeline, PAR=1)")
    print("照片: passport=44 (官方 GT), camshots=Photo/ 6 张同一证件 (xlsx GT), pic=assets/pic (官方 GT)")
    print("权重代际: HEAD 40951c5(gated);APK libmrz_jni.so 含 beam/seg/Sauvola 全部优化")
    print("=" * 100)
    for engine, log in logs.items():
        rows = parse(log)
        print(f"\n### 引擎 {engine.upper()}  ({len(rows)} RESULT 行)")
        for s in sets:
            sel = [r for r in rows if set_of(r["raw"]) == s]
            if not sel:
                continue
            n = len(sel)
            dec = sum(1 for r in sel if r["decode"])
            l1x = l2x = both = ck = 0
            l1c_ok = l1c_t = l2c_ok = l2c_t = 0
            vpass = 0
            totals = [r["total"] for r in sel]
            ocrs = [r["ocr"] for r in sel]
            for r in sel:
                g = cam_gt if s == "camshots" else gt.get(r["name"])
                if g is None:
                    continue
                g1, g2 = g
                if r["l1"] == g1:
                    l1x += 1
                if g2 and r["l2"] == g2:
                    l2x += 1
                if g2 and r["l1"] == g1 and r["l2"] == g2:
                    both += 1
                a, b = char_acc(r["l1"], g1)
                l1c_ok += a; l1c_t += b
                if g2:
                    a, b = char_acc(r["l2"], g2)
                    l2c_ok += a; l2c_t += b
                if line2_ok(r["l2"]):
                    ck += 1
                if r["verdict"] == "PASS":
                    vpass += 1
            _, l1p = pct(l1x, n)
            _, l2p = pct(l2x, n)
            _, bp = pct(both, n)
            _, ckp = pct(ck, n)
            print(f"  [{s}] n={n}")
            print(f"    line1 exact {l1x}/{n} ({l1p:.1f}%)  char {l1c_ok}/{l1c_t}={l1c_ok/max(l1c_t,1)*100:.2f}%")
            print(f"    line2 exact {l2x}/{n} ({l2p:.1f}%)  char {l2c_ok}/{l2c_t}={l2c_ok/max(l2c_t,1)*100:.2f}%")
            print(f"    BOTH  exact {both}/{n} ({bp:.1f}%)   cksum {ck}/{n} ({ckp:.1f}%)")
            print(f"    decode {dec}/{n}   verdict PASS {vpass}/{n}")
            print(f"    total ms: {lat(totals)}   ocr ms: {lat(ocrs)}")
    print("\n" + "=" * 100)


if __name__ == "__main__":
    sys.exit(main())
