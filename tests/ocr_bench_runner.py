"""Headless driver invoked by the GUI's OCR tab.

Runs `mrz_ocr_tool` (traditional) and/or `mrz_ocr_cnn_tool` on a
list of corpus images, returns per-image and aggregate statistics.
"""
from __future__ import annotations
import json, subprocess, time, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAD = ROOT / "6_mrz_ocr" / "build" / "mrz_ocr_tool"
CNN  = ROOT / "6_mrz_ocr" / "build" / "mrz_ocr_cnn_tool"
DATA = ROOT / "6_mrz_ocr" / "data" / "corpus"
CORPUS_JSON = DATA / "corpus.json"

METHODS = {
    "traditional": ("传统模板", TRAD),
    "cnn":         ("轻量CNN",   CNN),
}

def run_one(tool: Path, image: Path) -> dict:
    """Invoke the OCR tool and parse its text output."""
    t0 = time.perf_counter()
    try:
        r = subprocess.run([str(tool), str(image)], text=True,
                           capture_output=True, timeout=15)
        ms = (time.perf_counter() - t0) * 1000.0
    except subprocess.TimeoutExpired:
        return {"ok": False, "line1": "", "line2": "", "conf1": 0,
                "conf2": 0, "band": "", "ms": 9999.0, "err": "timeout"}
    out = {}
    for line in r.stdout.splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        out[k.strip()] = v.strip()
    return {"ok": r.returncode == 0,
            "line1": out.get("result.line1", ""),
            "line2": out.get("result.line2", ""),
            "conf1": int(out.get("result.conf1", "0") or 0),
            "conf2": int(out.get("result.conf2", "0") or 0),
            "band":  out.get("band.x band.y band.w band.h", ""),
            "ms":    ms,
            "err":   r.stderr.strip() if r.returncode != 0 else ""}

def compare(gt: str, got: str) -> tuple[int, int, str]:
    """Return (correct_chars, total_chars, per_char_status_string).

    per_char_status_string is `+` for a correct position and `-`
    otherwise, padded to GT length so it can be displayed verbatim.
    """
    n = max(len(gt), len(got))
    status = ""
    correct = 0
    for i in range(n):
        g = gt[i] if i < len(gt) else "_"
        h = got[i] if i < len(got) else "_"
        if g == h:
            status += "+"
            correct += 1
        else:
            status += "-"
    return correct, n, status

def run(case_ids: list[str] | None, methods: list[str],
        on_record=None) -> dict:
    """Run the requested cases (or all if case_ids is None) with the
    given methods. Returns an aggregate result.

    If on_record(done, total, case_id) is given it is invoked after each
    case completes, letting a caller drive a live progress bar."""
    corpus = json.loads(CORPUS_JSON.read_text())
    records = corpus["records"]
    if case_ids is not None:
        wanted = set(case_ids)
        records = [r for r in records if r["id"] in wanted]
    results = []
    agg = {m: {"n": 0, "ok": 0, "ms_total": 0.0,
                "l1_correct": 0, "l1_total": 0,
                "l2_correct": 0, "l2_total": 0,
                "full_match": 0,
                "low_conf": 0}
            for m in methods}
    for rec in records:
        image_path = DATA / rec["image"]
        per_method = {}
        for m in methods:
            r = run_one(METHODS[m][1], image_path)
            c1, t1, s1 = compare(rec["line1"], r["line1"])
            c2, t2, s2 = compare(rec["line2"], r["line2"])
            per_method[m] = {"raw": r,
                             "l1_correct": c1, "l1_total": t1, "l1_status": s1,
                             "l2_correct": c2, "l2_total": t2, "l2_status": s2,
                             "full": c1 == t1 and c2 == t2}
            a = agg[m]
            a["n"] += 1
            if r["ok"]: a["ok"] += 1
            a["ms_total"] += r["ms"]
            a["l1_correct"] += c1; a["l1_total"] += t1
            a["l2_correct"] += c2; a["l2_total"] += t2
            if c1 == t1 and c2 == t2: a["full_match"] += 1
            if r["ok"] and (r["conf1"] + r["conf2"]) < 100:
                a["low_conf"] += 1
        results.append({"id": rec["id"],
                        "image": rec["image"],
                        "gt1": rec["line1"], "gt2": rec["line2"],
                        "scale": rec["scale"], "noise": rec["noise"],
                        "skew": rec["skew"],
                        "per_method": per_method})
        if on_record:
            on_record(len(results), len(records), rec["id"])
    return {"records": results, "agg": agg, "methods": methods,
            "total_cases": len(records)}

if __name__ == "__main__":
    # CLI use: methods on argv (comma-separated), -a = all.
    # --progress emits a @@PROGRESS@done@total@id line after each case.
    args = sys.argv[1:]
    progress = "--progress" in args
    if progress:
        args.remove("--progress")
    if "-a" in args:
        args.remove("-a")
        case_ids = None
    else:
        case_ids = args[1:] if len(args) > 1 else None
    methods = args[0].split(",") if args else ["traditional", "cnn"]
    def _cb(done, total, cid):
        sys.stdout.write(f"@@PROGRESS@{done}@{total}@{cid}\n")
        sys.stdout.flush()
    print(json.dumps(run(case_ids, methods, _cb if progress else None),
                     indent=2))
