#!/usr/bin/env python3
"""Side-by-side benchmark of the traditional template-matching pipeline
vs the CNN-based pipeline on the OCR corpus.

For each image we measure:
  * total wall-clock (ms)
  * recognised-character accuracy
  * per-line confidence
  * pipeline success rate (i.e. did we get two lines out at all)

Results are aggregated and written to:
  bench_results.json
  bench_results.md  (human-readable table)
  bench_results.txt (plain-text table)
"""
from __future__ import annotations
import json, subprocess, time, sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
TRAD  = ROOT / "build" / "mrz_ocr_tool"
CNN   = ROOT / "build" / "mrz_ocr_cnn_tool"
DATA  = ROOT / "data" / "corpus"
CORPUS = DATA / "corpus.json"

def run(tool: Path, image: Path) -> dict:
    t0 = time.perf_counter()
    try:
        r = subprocess.run([str(tool), str(image)], text=True,
                           capture_output=True, timeout=10)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    except subprocess.TimeoutExpired:
        return {"ok": False, "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
                "line1": "", "line2": "", "conf1": 0, "conf2": 0,
                "band": None, "line1_correct": 0, "line1_total": 0,
                "line2_correct": 0, "line2_total": 0}
    d = {"ok": r.returncode == 0, "elapsed_ms": elapsed_ms,
         "line1": "", "line2": "", "conf1": 0, "conf2": 0, "band": None}
    for line in r.stdout.splitlines():
        if line.startswith("result.line1"):
            d["line1"] = line.split(":", 1)[1].strip()
        elif line.startswith("result.line2"):
            d["line2"] = line.split(":", 1)[1].strip()
        elif line.startswith("result.conf1"):
            d["conf1"] = int(line.split(":", 1)[1].strip())
        elif line.startswith("result.conf2"):
            d["conf2"] = int(line.split(":", 1)[1].strip())
        elif line.startswith("band.x"):
            d["band"] = [int(x) for x in line.split(":", 1)[1].split()]
    return d

def compare(gt: str, got: str) -> tuple[int, int]:
    n = min(len(gt), len(got))
    correct = sum(1 for i in range(n) if gt[i] == got[i])
    return correct, len(gt)

def main():
    corpus = json.loads(CORPUS.read_text())
    n = len(corpus["records"])
    out = {"traditional": [], "cnn": []}
    for rec in corpus["records"]:
        p = DATA / rec["image"]
        gt1, gt2 = rec["line1"], rec["line2"]
        # Traditional
        r_trad = run(TRAD, p)
        c1, t1 = compare(gt1, r_trad["line1"])
        c2, t2 = compare(gt2, r_trad["line2"])
        r_trad["line1_correct"] = c1
        r_trad["line1_total"] = t1
        r_trad["line2_correct"] = c2
        r_trad["line2_total"] = t2
        # CNN
        r_cnn = run(CNN, p)
        c1, t1 = compare(gt1, r_cnn["line1"])
        c2, t2 = compare(gt2, r_cnn["line2"])
        r_cnn["line1_correct"] = c1
        r_cnn["line1_total"] = t1
        r_cnn["line2_correct"] = c2
        r_cnn["line2_total"] = t2
        out["traditional"].append(r_trad)
        out["cnn"].append(r_cnn)
        if len(out["traditional"]) % 20 == 0:
            print(f"processed {len(out['traditional'])}/{n}", file=sys.stderr)

    def agg(name, rs):
        n = len(rs)
        ok = sum(1 for r in rs if r["ok"])
        ms = [r["elapsed_ms"] for r in rs]
        ms.sort()
        c1 = sum(r["line1_correct"] for r in rs)
        t1 = sum(r["line1_total"] for r in rs)
        c2 = sum(r["line2_correct"] for r in rs)
        t2 = sum(r["line2_total"] for r in rs)
        return {
            "n": n,
            "ok": ok,
            "ok_rate": ok / n if n else 0,
            "ms_mean": sum(ms)/n if n else 0,
            "ms_p50": ms[n//2] if n else 0,
            "ms_p95": ms[int(n*0.95)] if n else 0,
            "line1_acc": c1 / t1 if t1 else 0,
            "line2_acc": c2 / t2 if t2 else 0,
            "line1_correct": c1, "line1_total": t1,
            "line2_correct": c2, "line2_total": t2,
        }
    summary = {
        "traditional": agg("traditional", out["traditional"]),
        "cnn": agg("cnn", out["cnn"]),
    }

    (ROOT / "bench_results.json").write_text(
        json.dumps({"summary": summary, "per_image": out}, indent=2))
    # Split results by noise / clean subsets.
    def agg_subset(rs_idx):
        rs_trad = [out["traditional"][i] for i in rs_idx]
        rs_cnn  = [out["cnn"][i] for i in rs_idx]
        return agg("traditional", rs_trad), agg("cnn", rs_cnn)
    clean_idx = [i for i, r in enumerate(corpus["records"]) if r["noise"] == 0]
    noisy_idx = [i for i, r in enumerate(corpus["records"]) if r["noise"] > 0]
    sub_clean = {"traditional": agg_subset(clean_idx)[0], "cnn": agg_subset(clean_idx)[1]}
    sub_noisy = {"traditional": agg_subset(noisy_idx)[0], "cnn": agg_subset(noisy_idx)[1]}
    summary["clean_subset"]   = sub_clean
    summary["noisy_subset"]   = sub_noisy
    summary["clean_count"]    = len(clean_idx)
    summary["noisy_count"]    = len(noisy_idx)

    # Markdown
    md = "# MRZ OCR benchmark\n\n"
    md += f"Corpus: {n} synthetic MRZ images (see data/corpus.json).\n\n"
    md += "## Overall\n\n"
    md += "| method | pipeline OK | ms mean | ms p50 | ms p95 | line1 acc | line2 acc |\n"
    md += "|--------|-------------|---------|--------|--------|-----------|-----------|\n"
    for m in ("traditional", "cnn"):
        s = summary[m]
        md += (f"| {m} | {s['ok']}/{s['n']} "
               f"({100*s['ok_rate']:.1f}%) | "
               f"{s['ms_mean']:.1f} | {s['ms_p50']:.1f} | {s['ms_p95']:.1f} | "
               f"{100*s['line1_acc']:.1f}% | {100*s['line2_acc']:.1f}% |\n")
    md += f"\n## Clean subset (noise=0, n={len(clean_idx)})\n\n"
    md += "| method | line1 acc | line2 acc |\n|--------|-----------|-----------|\n"
    for m in ("traditional", "cnn"):
        s = sub_clean[m]
        md += f"| {m} | {100*s['line1_acc']:.1f}% | {100*s['line2_acc']:.1f}% |\n"
    md += f"\n## Noisy subset (noise>0, n={len(noisy_idx)})\n\n"
    md += "| method | line1 acc | line2 acc |\n|--------|-----------|-----------|\n"
    for m in ("traditional", "cnn"):
        s = sub_noisy[m]
        md += f"| {m} | {100*s['line1_acc']:.1f}% | {100*s['line2_acc']:.1f}% |\n"
    (ROOT / "bench_results.md").write_text(md)
    # Plain text
    txt = md
    (ROOT / "bench_results.txt").write_text(txt)
    print(md)
    return 0

if __name__ == "__main__":
    sys.exit(main())
