"""CRNN MRZ line reader (ONNX) with the shared stdout KV contract.

Input: a two-row MRZ band image (as produced by tests/gui.py's
_rst_native_mrz_band or the 6_mrz_ocr corpus images).
Output: result.backend/ok/line1/line2/conf1/conf2/band.../ms

The weights are the self-trained CRNN from this session
(train_crnn_v3.py, exported to ONNX).  Note: the training set contains the
11 scan images, so these numbers are optimistic on those images.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "12_crnn", "crnn_v3.onnx")
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<"
H = 32
_SESSION = None


def session():
    global _SESSION
    if _SESSION is None:
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.intra_op_num_threads = int(os.environ.get("MRZ_THREADS", "1"))
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _SESSION = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
    return _SESSION


def to_ink(gray):
    if float(gray.mean()) < 128:
        return gray.astype(np.float32) / 255.0
    return (255.0 - gray.astype(np.float32)) / 255.0


def trim(x, thr=0.25):
    rows = np.where((x > thr).sum(axis=1) > 0)[0]
    cols = np.where((x > thr).sum(axis=0) > 0)[0]
    if len(rows) == 0 or len(cols) == 0:
        return x
    y0, y1 = max(0, rows[0] - 2), min(x.shape[0], rows[-1] + 3)
    x0, x1 = max(0, cols[0] - 2), min(x.shape[1], cols[-1] + 3)
    return x[y0:y1, x0:x1]


def resize_h(x, h=H):
    hh, ww = x.shape[:2]
    w = max(4, int(round(h * ww / float(hh))))
    return cv2.resize(x, (w, h), interpolation=cv2.INTER_AREA if h < hh else cv2.INTER_LINEAR)


def split_rows(gray):
    """Split a two-row band at the deepest mid gutter."""
    Hh = gray.shape[0]
    if Hh < 8:
        return gray, gray
    rd = (gray < 128).sum(axis=1)
    mid_lo, mid_hi = int(Hh * 0.25), int(Hh * 0.75)
    split = Hh // 2 if mid_hi <= mid_lo else int(mid_lo + int(np.argmin(rd[mid_lo:mid_hi])))
    return gray[:split], gray[split:]


def ctc_decode(logits):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    idx = logits.argmax(-1)
    out = []
    prev = -1
    best = []
    for k, p in zip(idx, probs.max(-1)):
        if k != prev and k != 0:
            out.append(ALPHABET[k - 1])
            best.append(float(p))
        prev = k
    score = (sum(best) / len(best)) if best else 0.0
    return "".join(out), score


def read_row(gray):
    x = resize_h(trim(to_ink(gray)))
    inp = x[None, None].astype(np.float32)
    out = session().run(None, {"x": inp})[0][0]
    return ctc_decode(out)


def read_band(img_path):
    t0 = time.perf_counter()
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(str(img_path))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    row1, row2 = split_rows(gray)
    l1, s1 = read_row(row1)
    l2, s2 = read_row(row2)
    ms = (time.perf_counter() - t0) * 1000.0
    l1 = "".join(c for c in l1.upper() if c in ALPHABET or c == "<")
    l2 = "".join(c for c in l2.upper() if c in ALPHABET or c == "<")
    ok = len(l1) > 0 and len(l2) > 0
    return {"ok": ok, "line1": l1, "line2": l2, "conf1": s1 * 100, "conf2": s2 * 100,
            "band": "", "ms": ms, "image": str(img_path)}


def format_kv(result):
    out = [
        "result.backend  : crnn",
        f"result.ok       : {'OK' if result['ok'] else 'unreadable'}",
        f"result.line1    : {result['line1']}",
        f"result.line2    : {result['line2']}",
        f"result.conf1    : {int(round(result['conf1']))}",
        f"result.conf2    : {int(round(result['conf2']))}",
        f"band.x band.y band.w band.h : {result['band']}",
        f"result.image    : {result.get('image', '')}",
        f"ms: {result['ms']:.1f}",
    ]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?")
    ap.add_argument("--batch")
    args = ap.parse_args(argv)
    if args.batch:
        with open(args.batch) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    print(format_kv(read_band(line)))
                except Exception as exc:  # noqa: BLE001
                    print(f"result.ok       : error\nresult.note     : {exc}")
                print()
        return 0
    if not args.image:
        ap.print_help()
        return 2
    print(format_kv(read_band(args.image)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
