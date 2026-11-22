"""Full 1380-sample evaluation for cnn/crnn/fuse with per-record detail."""
import json, subprocess, time, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "corpus_eval"
TOOL = ROOT / "build" / "mrz_ocr_cnn_tool"

def run_engine(img, method):
    env = dict(os.environ)
    env["MRZ_OCR_LINE_MODELS"] = "1"
    env["MRZ_OCR_CRNN"] = "1" if method == "crnn" else "0"
    env.pop("MRZ_OCR_FUSE", None)
    if method == "fuse":
        env["MRZ_OCR_FUSE"] = "1"
    t0 = time.perf_counter()
    r = subprocess.run([str(TOOL), str(img)], capture_output=True, text=True, timeout=10, env=env)
    ms = (time.perf_counter() - t0) * 1000
    parsed = {}
    for ln in r.stdout.splitlines():
        if ":" in ln:
            k, _, v = ln.partition(":")
            parsed[k.strip()] = v.strip()
    return (parsed.get("result.line1", ""), parsed.get("result.line2", ""),
            int(parsed.get("result.conf1", "0") or 0),
            int(parsed.get("result.conf2", "0") or 0), ms)

def char_acc(gt, got):
    n = max(len(gt), len(got))
    c = sum(1 for i in range(n) if i < len(gt) and i < len(got) and gt[i] == got[i])
    return c, n

corpus = json.loads((CORPUS_DIR / "corpus.json").read_text())["records"]
methods = ("cnn", "crnn", "fuse")
stats = {m: {"l1_exact": 0, "l2_exact": 0, "both": 0, "l1c": 0, "l1t": 0,
             "l2c": 0, "l2t": 0, "ms": [], "fails": []} for m in methods}

t_start = time.perf_counter()
for i, rec in enumerate(corpus, 1):
    img = CORPUS_DIR / rec["image"]
    for m in methods:
        l1, l2, c1, c2, ms = run_engine(img, m)
        s = stats[m]
        s["ms"].append(ms)
        c1c, c1t = char_acc(rec["line1"], l1)
        c2c, c2t = char_acc(rec["line2"], l2)
        s["l1c"] += c1c; s["l1t"] += c1t
        s["l2c"] += c2c; s["l2t"] += c2t
        l1_ok = (l1 == rec["line1"])
        l2_ok = (l2 == rec["line2"])
        if l1_ok: s["l1_exact"] += 1
        if l2_ok: s["l2_exact"] += 1
        if l1_ok and l2_ok: s["both"] += 1
        if not (l1_ok and l2_ok):
            s["fails"].append({
                "id": rec["id"], "profile": rec.get("profile", ""),
                "gt1": rec["line1"], "got1": l1,
                "gt2": rec["line2"], "got2": l2,
                "conf1": c1, "conf2": c2,
            })
    if i % 100 == 0:
        elapsed = time.perf_counter() - t_start
        print(f"  [{i}/{len(corpus)}] {elapsed:.1f}s", flush=True)

n = len(corpus)
print(f"\n=== 1380-sample full eval ({n} samples, {time.perf_counter()-t_start:.1f}s) ===\n")
for m in methods:
    s = stats[m]
    import statistics
    print(f"{m:6s}: l1_exact={s['l1_exact']}/{n} ({100*s['l1_exact']/n:.1f}%)  "
          f"l2_exact={s['l2_exact']}/{n} ({100*s['l2_exact']/n:.1f}%)  "
          f"both={s['both']}/{n} ({100*s['both']/n:.1f}%)  "
          f"l1_char={100*s['l1c']/s['l1t']:.2f}%  "
          f"l2_char={100*s['l2c']/s['l2t']:.2f}%  "
          f"ms_p50={statistics.median(s['ms']):.1f}  "
          f"ms_mean={statistics.mean(s['ms']):.1f}")

# Save failures for fuse
out = {
    "n": n,
    "stats": {m: {k: v for k, v in s.items() if k != "fails"} | {"n_fails": len(s["fails"])}
              for m, s in stats.items()},
    "fuse_fails": stats["fuse"]["fails"][:50],
}
Path(ROOT / "tests" / "eval_1380_result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(f"\nFuse failures (first 50) saved to tests/eval_1380_result.json")
print(f"Fuse total fails: {len(stats['fuse']['fails'])}/{n}")
