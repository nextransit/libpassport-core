"""Generate a Markdown + JSON benchmark report from the corpus."""
from __future__ import annotations
import json, statistics, subprocess, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAD = ROOT / "build" / "mrz_ocr_tool"
CNN  = ROOT / "build" / "mrz_ocr_cnn_tool"
DATA = ROOT / "data" / "corpus"
CORPUS_JSON = DATA / "corpus.json"


def run_all():
    corpus = json.loads(CORPUS_JSON.read_text())["records"]
    out = {"traditional": [], "cnn": []}
    for rec in corpus:
        img = DATA / rec["image"]
        for m, tool in (("traditional", TRAD), ("cnn", CNN)):
            try:
                r = subprocess.run([str(tool), str(img)], capture_output=True,
                                   text=True, timeout=20)
                parsed = {}
                for ln in r.stdout.splitlines():
                    if ":" not in ln: continue
                    k, _, v = ln.partition(":")
                    parsed[k.strip()] = v.strip()
                out[m].append({
                    "id": rec["id"],
                    "noise": rec["noise"],
                    "scale": rec["scale"],
                    "skew": rec["skew"],
                    "line1": parsed.get("result.line1", ""),
                    "line2": parsed.get("result.line2", ""),
                    "conf1": int(parsed.get("result.conf1", "0") or 0),
                    "conf2": int(parsed.get("result.conf2", "0") or 0),
                    "ms":    float(parsed.get("timing.ms", "0") or 0),
                    "ok":    r.returncode == 0,
                    "gt1": rec["line1"],
                    "gt2": rec["line2"],
                })
            except Exception as e:
                out[m].append({"id": rec["id"], "noise": rec["noise"],
                               "scale": rec["scale"], "skew": rec["skew"],
                               "line1": "", "line2": "", "conf1": 0, "conf2": 0,
                               "ms": 0.0, "ok": False, "gt1": rec["line1"],
                               "gt2": rec["line2"]})
    return corpus, out


def compare(gt, got):
    n = max(len(gt), len(got))
    correct = 0
    for i in range(n):
        g = gt[i] if i < len(gt) else "_"
        h = got[i] if i < len(got) else "_"
        if g == h: correct += 1
    return correct, n


def aggregate(corpus, results):
    methods = list(results.keys())
    agg = {}
    for m, recs in results.items():
        n = len(recs); ok = 0; l1c = l1t = l2c = l2t = 0
        ms_all = []; full = 0
        for r, c in zip(recs, corpus):
            if r["ok"]: ok += 1
            c1, t1 = compare(r["gt1"], r["line1"]); l1c += c1; l1t += t1
            c2, t2 = compare(r["gt2"], r["line2"]); l2c += c2; l2t += t2
            if c1 == t1 and c2 == t2 and t1 and t2: full += 1
            ms_all.append(r["ms"])
        agg[m] = {"n": n, "ok": ok,
                  "l1_correct": l1c, "l1_total": l1t,
                  "l2_correct": l2c, "l2_total": l2t,
                  "ms_mean": statistics.mean(ms_all) if ms_all else 0,
                  "ms_p50":  statistics.median(ms_all) if ms_all else 0,
                  "ms_p95":  sorted(ms_all)[int(0.95*len(ms_all))] if ms_all else 0,
                  "full_match": full}
    return methods, agg


def render_markdown(corpus, methods, agg, results):
    n_all = len(corpus)
    n_clean = sum(1 for c in corpus if c["noise"] == 0)
    n_noisy = n_all - n_clean
    lines = ["# MRZ OCR benchmark\n",
             f"Corpus: {n_all} synthetic MRZ images (see data/corpus.json).\n",
             "## Overall\n"]
    header = "| method | pipeline OK | ms mean | ms p50 | ms p95 | line1 acc | line2 acc |"
    sep = "|--------|-------------|---------|--------|--------|-----------|-----------|"
    lines += [header, sep]
    for m in methods:
        a = agg[m]
        l1 = 100*a["l1_correct"]/a["l1_total"] if a["l1_total"] else 0
        l2 = 100*a["l2_correct"]/a["l2_total"] if a["l2_total"] else 0
        lines.append(f"| {m} | {a['ok']}/{a['n']} ({100*a['ok']/a['n']:.1f}%) | "
                     f"{a['ms_mean']:.1f} | {a['ms_p50']:.1f} | {a['ms_p95']:.1f} | "
                     f"{l1:.1f}% | {l2:.1f}% |")
    lines += ["", f"## Clean subset (noise=0, n={n_clean})", header, sep]
    for m in methods:
        a = agg[m]
        sub = [r for r in results[m] if r["noise"] == 0]
        if sub:
            l1c = sum(compare(r["gt1"], r["line1"])[0] for r in sub)
            l1t = sum(compare(r["gt1"], r["line1"])[1] for r in sub)
            l2c = sum(compare(r["gt2"], r["line2"])[0] for r in sub)
            l2t = sum(compare(r["gt2"], r["line2"])[1] for r in sub)
            lines.append(f"| {m} | {100*l1c/l1t:.1f}% | {100*l2c/l2t:.1f}% |")
        else:
            lines.append(f"| {m} | n/a | n/a |")
    lines += ["", f"## Noisy subset (noise>0, n={n_noisy})", header, sep]
    for m in methods:
        a = agg[m]
        sub = [r for r in results[m] if r["noise"] > 0]
        if sub:
            l1c = sum(compare(r["gt1"], r["line1"])[0] for r in sub)
            l1t = sum(compare(r["gt1"], r["line1"])[1] for r in sub)
            l2c = sum(compare(r["gt2"], r["line2"])[0] for r in sub)
            l2t = sum(compare(r["gt2"], r["line2"])[1] for r in sub)
            lines.append(f"| {m} | {100*l1c/l1t:.1f}% | {100*l2c/l2t:.1f}% |")
        else:
            lines.append(f"| {m} | n/a | n/a |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    corpus, recs = run_all()
    methods, agg = aggregate(corpus, recs)
    results = recs
    md = render_markdown(corpus, methods, agg, results)
    out_path = ROOT / "bench_results.md"
    out_path.write_text(md)
    json_path = ROOT / "bench_results.json"
    json_path.write_text(json.dumps({"agg": agg, "methods": methods, "total": len(corpus)}, indent=2))
    print(md)
