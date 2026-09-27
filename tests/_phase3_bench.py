#!/usr/bin/env python3
"""Phase 3 bench: project-split + pure-rec backends vs paddle_local baseline.

Each backend returns two 44-char lines; scored against corpus.json GT
character-by-character (fixed-width ICAO lines, positional alignment).
"""
import sys, json, time, numpy as np
from pathlib import Path
from PIL import Image

ROOT = Path('/Users/zhouyong/Desktop/work/Decard/gitlab/passport')
DATA = ROOT / '6_mrz_ocr/data/corpus_eval'
sys.path.insert(0, str(ROOT / '6_1_PaddleOCR'))
from paddle_local import _find_mrz_band, _norm_two_lines, LINE_LEN

N = 10

def load_records():
    d = json.load(open(DATA / 'corpus.json'))
    return d['records'][:N]

def split_rows(gray):
    """Project-split the band into L1 row / L2 row via the deepest gutter."""
    y0, y1 = _find_mrz_band(gray)
    band = gray[y0:y1]
    H = band.shape[0]
    rd = (band < 128).sum(axis=1)
    mid_lo, mid_hi = int(H*0.25), int(H*0.75)
    if mid_hi <= mid_lo:
        split = H // 2
    else:
        split = int(mid_lo + np.argmin(rd[mid_lo:mid_hi]))
    return band[:split], band[split:]

def score(pred_l1, pred_l2, gt_l1, gt_l2):
    ok = (len(pred_l1) == LINE_LEN and len(pred_l2) == LINE_LEN
          and pred_l1[:1] in "PIVACDR" and pred_l1[1:2] == "<"
          and (pred_l2[:1].isdigit() or pred_l2[:1] == "L"))
    c1 = sum(a == b for a, b in zip(pred_l1, gt_l1))
    c2 = sum(a == b for a, b in zip(pred_l2, gt_l2))
    return ok, c1, c2

# ---------- Backend 1: paddle_local baseline (subprocess, tesseract) ----------
def backend_paddle_local(gray, path):
    import subprocess
    r = subprocess.run([sys.executable, str(ROOT/'6_1_PaddleOCR/paddle_local.py'), str(path)],
                       capture_output=True, text=True)
    l1 = l2 = ""
    for ln in r.stdout.splitlines():
        if ln.startswith("result.line1    : "): l1 = ln.split(": ", 1)[1]
        if ln.startswith("result.line2    : "): l2 = ln.split(": ", 1)[1]
    return l1, l2

# ---------- Backend 2/3: paddlex pure rec (server / mobile en) ----------
class PaddlexRec:
    def __init__(self, model_name, model_dir, engine='onnxruntime'):
        import paddlex
        self.rec = paddlex.create_model(model_name=model_name, model_dir=model_dir,
                                        device='cpu', engine=engine)
    def rows(self, gray, path=None):
        r1, r2 = split_rows(gray)
        out = []
        for row in (r1, r2):
            im = Image.fromarray(row).resize((row.shape[1]*3, row.shape[0]*3), Image.LANCZOS)
            o = list(self.rec.predict(np.asarray(im.convert('RGB'))))[0]
            txt = (o.get('rec_text') or '').strip().upper()
            txt = ''.join(c if c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<" else '' for c in txt)
            out.append(txt[:LINE_LEN].ljust(LINE_LEN, '<'))
        return out[0], out[1]

# ---------- Backend 4: EasyOCR whole-3x + ymid classification ----------
class EasyBackend:
    def __init__(self):
        import easyocr
        self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    def rows(self, gray, path=None):
        img = Image.fromarray(gray).resize((gray.shape[1]*3, gray.shape[0]*3), Image.LANCZOS)
        res = self.reader.readtext(np.asarray(img.convert('RGB')), detail=1, paragraph=False,
                                   allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<')
        cands = []
        for bbox, txt, conf in res:
            ys = [p[1] for p in bbox]
            ymid = (min(ys) + max(ys)) // 2
            cands.append((ymid, txt.strip().upper(), float(conf)))
        cands.sort(key=lambda c: -len(c[1]))
        l1s = [c for c in cands if c[1][:1] in "PIVACDR"]
        l2s = [c for c in cands if (c[1][:1].isdigit() or c[1][:1] == "L")]
        def norm(t):
            t = ''.join(x if x in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<" else '' for x in t)
            return t[:LINE_LEN].ljust(LINE_LEN, '<')
        l1 = norm(l1s[0][1]) if l1s else "<"*LINE_LEN
        l2 = norm(l2s[0][1]) if l2s else "<"*LINE_LEN
        return l1, l2

def run_backend(name, fn, recs):
    accs = []
    t0 = time.perf_counter()
    for rec in recs:
        img = Image.open(DATA / rec['image']).convert('L')
        gray = np.asarray(img)
        l1, l2 = fn(gray, DATA / rec['image'])
        ok, c1, c2 = score(l1, l2, rec['line1'], rec['line2'])
        accs.append((ok, c1, c2))
    dt = time.perf_counter() - t0
    n_ok = sum(a for a, _, _ in accs)
    a1 = sum(c for _, c, _ in accs) / (N*LINE_LEN) * 100
    a2 = sum(c for _, _, c in accs) / (N*LINE_LEN) * 100
    print(f"{name:28s} OK={n_ok}/{N}  L1={a1:5.1f}%  L2={a2:5.1f}%  {dt:6.1f}s")
    return accs

if __name__ == '__main__':
    recs = load_records()
    # Verify GT alignment is correct on record 0
    print("GT sample:", recs[0]['line1'][:20], "|", recs[0]['line2'][:20])
    print("-"*70)
    run_backend("paddle_local(tesseract ocrb)", backend_paddle_local, recs)
    print("Loading PP-OCRv5 server rec...", file=sys.stderr)
    sr = PaddlexRec('PP-OCRv5_server_rec',
                    '/Users/zhouyong/.paddlex/official_models/PP-OCRv5_server_rec_onnx')
    run_backend("project-split + PP-OCRv5 server rec", sr.rows, recs)
    print("Loading EasyOCR...", file=sys.stderr)
    eb = EasyBackend()
    run_backend("EasyOCR whole-3x ymid", eb.rows, recs)
