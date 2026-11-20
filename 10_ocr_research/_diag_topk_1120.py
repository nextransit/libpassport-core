#!/usr/bin/env python3
"""对难照 band 分别取 CNN / CRNN 头 top-K 候选，对照 GT 定位真值排名。

运行:  python3 _diag_topk_1120.py [photo-stem ...]
"""
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "6_mrz_ocr/build/mrz_ocr_cnn_tool"
GT = {
    "2026-9-29-U": (
        "P<CHNYUE<<YUESHU<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
        "E158193223CHN7807091M2505233356G5111BO7S6134"),
    "2026-9-30-AG": (
        "P<CHNCONG<<YINGXIAO<<<<<<<<<<<<<<<<<<<<<<<<<",
        "E066908744CHN9303298F3411263TWW9ZQLBES7OS062"),
    "2026-9-30-H": (
        "AKCHNYANG<<ZHEN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
        "KE90003518CHN9707014F2912268A911<121101<<<72"),
    "2026-9-30-I": ("CSC733210936<2711209<8612261<8", ""),
    "2026-9-30-K": ("CSC495643597<2610265<8706216<0", ""),
    "2026-9-30-J": ("CSC482624408<2610106<7410024<0", ""),
    "P3": ("POCHNJIA<<LIMIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
           "EE20426536CHN7410024M2808303LMNGMBKCMDPBA970"),
    "P8": ("POCHNJIA<<LIMIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
           "EE20426536CHN7410024M2808303LMNGMBKCMDPBA970"),
    "2026-9-29-A": (
        "P<CHNYUN<<YAOYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
        "E0132B7855CHN0110141M3105179GB682O5O108G0B88"),
}
LINE_RE = re.compile(r"TOPK L(\d) c(\d+): (.*)")
CAND_RE = re.compile(r"([0-9A-Z<])\(([\d.]+)\)")


def run(band, crnn):
    env = dict(os.environ)
    env["MRZ_OCR_LINE_MODELS"] = "1"
    env["MRZ_OCR_CRNN"] = "1" if crnn else "0"
    env.pop("MRZ_OCR_FUSE", None)
    env["MRZ_OCR_TOPK"] = "1"
    env["MRZ_OCR_CAND_K"] = "8"
    p = subprocess.run([str(TOOL), str(band)], capture_output=True,
                       text=True, timeout=15, env=env)
    table = {}
    for m in LINE_RE.finditer(p.stderr):
        li, col, body = int(m.group(1)), int(m.group(2)), m.group(3)
        table.setdefault(li, {})[col] = CAND_RE.findall(body)
    lines = {}
    for line in p.stdout.splitlines():
        if line.startswith("result.line"):
            k, _, v = line.partition(":")
            lines[k.strip()] = v.strip()
    return table, lines


def main():
    stems = sys.argv[1:] or list(GT)
    for stem in stems:
        band = ROOT / f"10_ocr_research/_diag1120_{stem}.ppm"
        if not band.exists():
            print(f"== {stem}: band 缺失，跳过"); continue
        gt1, gt2 = GT[stem]
        print("=" * 96)
        print(f"== {stem}")
        for li, gt in ((0, gt1), (1, gt2)):
            if not gt:
                continue
            tc, lc = run(band, crnn=False)
            tr, lr = run(band, crnn=True)
            gotc = lc.get("result.line1" if li == 0 else "result.line2", "")
            gotr = lr.get("result.line1" if li == 0 else "result.line2", "")
            print(f"  L{li} GT  : {gt}")
            print(f"  L{li} CNN : {gotc}")
            print(f"  L{li} CRNN: {gotr}")
            tabc = tc.get(li, {})
            tabr = tr.get(li, {})
            for c in range(len(gt)):
                g = gt[c]
                c0 = tabc.get(c, [])
                r0 = tabr.get(c, [])
                if c0 and c0[0][0] == g and r0 and r0[0][0] == g:
                    continue  # 双头 top1 都对
                rc = next((k for k, v in enumerate(c0) if v[0] == g), None)
                rr = next((k for k, v in enumerate(r0) if v[0] == g), None)
                cs = " ".join(f"{ch}:{p}" for ch, p in c0[:4])
                rs = " ".join(f"{ch}:{p}" for ch, p in r0[:4])
                print(f"    c{c:02d} GT={g} | cnnGT@{rc} [{cs}] "
                      f"| crnnGT@{rr} [{rs}]")


if __name__ == "__main__":
    main()
