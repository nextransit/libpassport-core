#!/usr/bin/env python3
"""Export CNN misclassified character slices for trainer feedback.

Runs the C pipeline (MRZ_OCR_DUMP=1) over the evaluation corpus,
compares each recognised line against the ground truth, and writes
the 16x12 glyph patches where the CNN was WRONG (or low-confidence)
alongside their ground-truth label to an .npz.  This is the
"real edge slices -> trainer" loop: the next train_cnn_real run can
upsample these exact patches instead of adding blind synthetic noise.

Usage:
  python3 tools/export_hard_slices.py \
      --corpus data/corpus_eval \
      --tool build/mrz_ocr_cnn_tool \
      --out data/hard_slices.npz
"""
from __future__ import annotations
import argparse, glob, json, os, shutil, subprocess, sys
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image

GLYPHS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<")
GI = {c: i for i, c in enumerate(GLYPHS)}

def load_dump(path: str) -> np.ndarray:
    a = np.asarray(Image.open(path).convert('L'), dtype=np.float32) / 255.0
    return a  # ink high (dump writes glyph*255)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus_eval")
    ap.add_argument("--tool", default="build/mrz_ocr_cnn_tool")
    ap.add_argument("--out", default="data/hard_slices.npz")
    ap.add_argument("--low-conf-thresh", type=int, default=80,
                    help="also export patches whose avg conf < this")
    a = ap.parse_args()
    ROOT = Path(__file__).resolve().parents[1]
    corpus_dir = ROOT / a.corpus
    corpus = json.loads((corpus_dir / "corpus.json").read_text())["records"]

    env = dict(os.environ); env['MRZ_OCR_DUMP']='1'
    X, Y, meta = [], [], []
    pairs = Counter(); by_col = Counter(); low_conf = 0
    n_imgs = 0
    for rec in corpus:
        img = corpus_dir / rec["image"]
        # clean the shared dump dir before EVERY run (stale-dump guard)
        for f in glob.glob('/tmp/mrz_dump_img_*.ppm'):
            os.remove(f)
        r = subprocess.run([str(ROOT/a.tool), str(img)], capture_output=True,
                           text=True, timeout=20, env=env)
        if r.returncode != 0:
            continue
        got = {}
        for ln in r.stdout.splitlines():
            if ln.startswith("result.line1"): got["line1"] = ln.split(":",1)[1].strip()
            elif ln.startswith("result.line2"): got["line2"] = ln.split(":",1)[1].strip()
            elif ln.startswith("result.conf1"): got["conf1"] = int(ln.split(":",1)[1].strip())
            elif ln.startswith("result.conf2"): got["conf2"] = int(ln.split(":",1)[1].strip())
        n_imgs += 1
        for li, line in (("line1","line1"), ("line2","line2")):
            gt = rec[line]
            ocr = got.get(line, "")
            if not ocr:
                continue
            for c, g in enumerate(gt):
                if c >= len(ocr):
                    break
                h = ocr[c]
                wrong = g != h
                low = (got.get("conf1" if li=="line1" else "conf2", 0) < a.low_conf_thresh)
                if not wrong and not low:
                    continue
                if low and not wrong:
                    low_conf += 1
                dumpf = glob.glob(f'/tmp/mrz_dump_img_l{0 if li=="line1" else 1}_c{c:02d}_r*_w*.ppm')
                if not dumpf:
                    continue
                patch = load_dump(dumpf[0])
                X.append(patch)
                Y.append(GI[g])
                pairs[f"{g}->{h}"] += 1
                by_col[f"{li[4:]}:{c}"] += 1
                meta.append({"id": rec["id"], "col": c, "gt": g, "got": h,
                             "profile": rec.get("profile","?"), "rot": rec.get("rot",0)})
    if not X:
        print("no hard slices collected")
        return 1
    Xa = np.array(X, dtype=np.float32).reshape(-1, 12, 16)
    Ya = np.array(Y, dtype=np.int64)
    out_path = ROOT / a.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, X=Xa, Y=Ya,
             meta=np.array([json.dumps(m) for m in meta]))
    print(f"== exported {len(Xa)} hard slices -> {out_path}")
    print(f"images processed: {n_imgs}; low-conf-only exports: {low_conf}")
    print("\n-- top confusion pairs (gt->got) --")
    for p, n in pairs.most_common(15):
        print(f"  {p:8s} {n}")
    print("\n-- by column (top 12) --")
    for c, n in by_col.most_common(12):
        print(f"  {c:8s} {n}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
