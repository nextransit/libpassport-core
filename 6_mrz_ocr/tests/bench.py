"""Benchmark + regression gate for the MRZ OCR module.

Runs both backends over a frozen corpus directory (default
data/corpus_eval -- NEVER the training corpus) and reports:

  * pipeline OK rate, per-line char accuracy, full-line match rate
  * ICAO checksum-valid rate on line 2 (the product-level metric)
  * per-column error counts and top confusion pairs
  * failures.json export (the fuel for the fix loop)

Gate mode (--gate baseline.json): exits 1 if any key metric regressed
by more than 0.3pp against the baseline, or below absolute floors.
With --update-baseline the baseline is written/refreshed after the run.
"""
from __future__ import annotations
import argparse, json, statistics, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAD = ROOT / "build" / "mrz_ocr_tool"
CNN  = ROOT / "build" / "mrz_ocr_cnn_tool"

W = [7, 3, 1]

def check_digit(seg: str) -> int:
    total = 0
    for i, c in enumerate(seg):
        v = (ord(c) - 48) if '0' <= c <= '9' else (
            (ord(c) - 55) if 'A' <= c <= 'Z' else 0)
        total += v * W[i % 3]
    return total % 10

def line2_checksum_ok(l2: str) -> bool:
    if len(l2) < 44: return False
    try:
        return (check_digit(l2[0:9]) == int(l2[9]) and
                check_digit(l2[13:19]) == int(l2[19]) and
                check_digit(l2[21:27]) == int(l2[27]) and
                check_digit(l2[28:42]) == int(l2[42]) and
                check_digit(l2[0:10] + l2[13:20] + l2[21:43]) == int(l2[43]))
    except ValueError:
        return False

def compare(gt: str, got: str):
    """Char-level diff; returns (correct, total, [mismatch indexes])."""
    n = max(len(gt), len(got))
    errs = []
    for i in range(n):
        g = gt[i] if i < len(gt) else "_"
        h = got[i] if i < len(got) else "_"
        if g != h: errs.append(i)
    return n - len(errs), n, errs


def run_tool(tool: Path, img: Path):
    t0 = time.perf_counter()
    try:
        r = subprocess.run([str(tool), str(img)], capture_output=True,
                           text=True, timeout=20)
        ms = (time.perf_counter() - t0) * 1000.0
    except Exception:
        return "", "", 0, 0, False, 0.0
    parsed = {}
    for ln in r.stdout.splitlines():
        if ":" not in ln: continue
        k, _, v = ln.partition(":")
        parsed[k.strip()] = v.strip()
    conf1 = int(parsed.get("result.conf1", "0") or 0)
    conf2 = int(parsed.get("result.conf2", "0") or 0)
    return (parsed.get("result.line1", ""), parsed.get("result.line2", ""),
            conf1, conf2, r.returncode == 0, ms)


def run_all(corpus_dir: Path):
    corpus = json.loads((corpus_dir / "corpus.json").read_text())["records"]
    out = {"traditional": [], "cnn": []}
    for rec in corpus:
        img = corpus_dir / rec["image"]
        for m, tool in (("traditional", TRAD), ("cnn", CNN)):
            l1, l2, c1, c2, ok, ms = run_tool(tool, img)
            out[m].append({
                "id": rec["id"], "profile": rec.get("profile", "clean"),
                "scale": rec["scale"], "skew": rec["skew"],
                "noise": rec["noise"], "rot": rec.get("rot", 0.0),
                "line1": l1, "line2": l2, "conf1": c1, "conf2": c2,
                "ms": ms, "ok": ok,
                "gt1": rec["line1"], "gt2": rec["line2"],
            })
    return corpus, out


def metrics_for(recs):
    n = len(recs)
    ok = sum(1 for r in recs if r["ok"])
    l1c = l1t = l2c = l2t = 0
    full = cksum = 0
    col_err = [[0] * 44, [0] * 44]
    confusions = {}
    fails = []
    for r in recs:
        c1, t1, e1 = compare(r["gt1"], r["line1"])
        c2, t2, e2 = compare(r["gt2"], r["line2"])
        l1c += c1; l1t += t1; l2c += c2; l2t += t2
        if c1 == t1 and c2 == t2 and t1 and t2: full += 1
        if line2_checksum_ok(r["line2"]): cksum += 1
        for i in e1:
            col_err[0][i] += 1
            g = r["gt1"][i] if i < len(r["gt1"]) else "_"
            h = r["line1"][i] if i < len(r["line1"]) else "_"
            key = f"{g}->{h}"
            confusions[key] = confusions.get(key, 0) + 1
        for i in e2:
            col_err[1][i] += 1
            g = r["gt2"][i] if i < len(r["gt2"]) else "_"
            h = r["line2"][i] if i < len(r["line2"]) else "_"
            key = f"{g}->{h}"
            confusions[key] = confusions.get(key, 0) + 1
        if e1 or e2 or not r["ok"]:
            fails.append({k: r[k] for k in
                          ("id", "profile", "scale", "skew", "noise", "rot",
                           "gt1", "gt2", "line1", "line2",
                           "conf1", "conf2", "ok", "ms")}
                         | {"l1_err": e1, "l2_err": e2})
    ms_all = [r["ms"] for r in recs]
    return {
        "n": n, "ok": ok, "ok_rate": ok / n if n else 0,
        "l1_acc": l1c / l1t if l1t else 0, "l2_acc": l2c / l2t if l2t else 0,
        "full_match_rate": full / n if n else 0,
        "cksum_rate": cksum / n if n else 0,
        "ms_mean": statistics.mean(ms_all) if ms_all else 0,
        "ms_p50": statistics.median(ms_all) if ms_all else 0,
        "ms_p95": sorted(ms_all)[int(0.95 * len(ms_all))] if ms_all else 0,
        "_col_err": col_err, "_confusions": confusions, "_fails": fails,
    }


def render_markdown(methods, agg, results, corpus):
    lines = ["# MRZ OCR benchmark\n",
             f"Corpus: {len(corpus)} images "
             f"({sum(1 for c in corpus if c.get('profile')=='realistic')} realistic). "
             "Metrics: char accuracy over all lines; full = both lines exact; "
             "cksum = line2 passes all 5 ICAO check digits.\n",
             "## Overall\n"]
    header = ("| method | pipeline OK | ms p50 | line1 acc | line2 acc |"
              " full match | cksum valid |")
    sep = "|--------|-------------|--------|-----------|-----------|------------|-------------|"
    lines += [header, sep]
    for m in methods:
        a = agg[m]
        lines.append(
            f"| {m} | {a['ok']}/{a['n']} ({100*a['ok_rate']:.1f}%) | "
            f"{a['ms_p50']:.1f} | {100*a['l1_acc']:.2f}% | {100*a['l2_acc']:.2f}% | "
            f"{100*a['full_match_rate']:.2f}% | {100*a['cksum_rate']:.2f}% |")

    lines += ["", "## Breakdown by profile", header, sep]
    profiles = sorted({r["profile"] for r in results[methods[0]]})
    for p in profiles:
        for m in methods:
            sub = [r for r in results[m] if r["profile"] == p]
            if not sub: continue
            a = metrics_for(sub)
            lines.append(
                f"| {p} | {100*a['ok_rate']:.1f}% | {a['ms_p50']:.1f} | "
                f"{100*a['l1_acc']:.2f}% | {100*a['l2_acc']:.2f}% | "
                f"{100*a['full_match_rate']:.2f}% | {100*a['cksum_rate']:.2f}% |")

    for m in methods:
        a = agg[m]
        conf = a["_confusions"]
        if not conf: continue
        lines += ["", f"## Top confusions ({m})",
                  "| gt->got | count |", "|---------|-------|"]
        for k, v in sorted(conf.items(), key=lambda kv: -kv[1])[:15]:
            lines.append(f"| {k} | {v} |")

    for m in methods:
        a = agg[m]
        col_err = a["_col_err"]
        worst = sorted(range(44),
                       key=lambda i: -col_err[0][i] - col_err[1][i])[:10]
        if all(col_err[0][i] + col_err[1][i] == 0 for i in worst): continue
        lines += ["", f"## Worst columns ({m})",
                  "| line1 col | errors | line2 col | errors |",
                  "|-----------|--------|-----------|--------|"]
        w1 = sorted(range(44), key=lambda i: -col_err[0][i])[:5]
        w2 = sorted(range(44), key=lambda i: -col_err[1][i])[:5]
        for i in range(5):
            lines.append(f"| {w1[i]} | {col_err[0][w1[i]]} | "
                         f"{w2[i]} | {col_err[1][w2[i]]} |")
        del worst
    return "\n".join(lines) + "\n"


GATE_KEYS = ["ok_rate", "l1_acc", "l2_acc", "full_match_rate", "cksum_rate"]
# Absolute floors apply to the product path (cnn) only; the traditional
# backend is a legacy A/B comparison that is tracked for regressions
# but not held to product-level floors.
FLOORED_METHOD = "cnn"
FLOORS = {"ok_rate": 0.98, "l1_acc": 0.95, "l2_acc": 0.95,
          "full_match_rate": 0.85, "cksum_rate": 0.90}
REGRESSION_PP = 0.003   # 0.3pp


def gate(agg, baseline_path: Path, update: bool) -> int:
    current = {m: {k: agg[m][k] for k in GATE_KEYS} for m in ("cnn", "traditional")}
    if not baseline_path.exists():
        baseline_path.write_text(json.dumps(current, indent=2))
        print(f"[gate] no baseline existed -- wrote {baseline_path}")
        return 0
    base = json.loads(baseline_path.read_text())
    rc = 0
    for m in current:
        if m not in base:
            print(f"[gate] method {m} missing in baseline -- skipped")
            continue
        for k in GATE_KEYS:
            if m == FLOORED_METHOD:
                floor = FLOORS.get(k, 0.0)
                if current[m][k] < floor:
                    print(f"[gate] FAIL {m}.{k} = {100*current[m][k]:.2f}% "
                          f"below floor {100*floor:.2f}%")
                    rc = 1
            drop = base[m][k] - current[m][k]
            if drop > REGRESSION_PP:
                print(f"[gate] FAIL {m}.{k} regressed "
                      f"{100*drop:.2f}pp ({100*base[m][k]:.2f}% -> "
                      f"{100*current[m][k]:.2f}%)")
                rc = 1
    if rc == 0:
        print("[gate] PASS")
        if update:
            baseline_path.write_text(json.dumps(current, indent=2))
            print(f"[gate] baseline refreshed -> {baseline_path}")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus_eval")
    ap.add_argument("--gate", metavar="BASELINE_JSON",
                    help="regression-gate against a baseline file")
    ap.add_argument("--update-baseline", action="store_true",
                    help="with --gate: refresh the baseline on PASS")
    ap.add_argument("--no-report", action="store_true",
                    help="skip rewriting bench_results.*")
    a = ap.parse_args()

    corpus_dir = ROOT / a.corpus
    corpus, recs = run_all(corpus_dir)
    methods = ["traditional", "cnn"]
    agg = {m: metrics_for(recs[m]) for m in methods}

    md = render_markdown(methods, agg, recs, corpus)
    print(md)
    if not a.no_report:
        (ROOT / "bench_results.md").write_text(md)
        (ROOT / "bench_results.txt").write_text(md)
        json_path = ROOT / "bench_results.json"
        json_path.write_text(json.dumps(
            {m: {k: v for k, v in agg[m].items() if not k.startswith("_")}
             for m in methods}, indent=2))

    # failures.json per method (loop fuel for mine_failures.py)
    for m in methods:
        (ROOT / f"failures_{m}.json").write_text(
            json.dumps(agg[m]["_fails"], indent=2))

    # metrics history append
    hist = ROOT / "metrics_history.md"
    if not hist.exists():
        hist.write_text("# Metrics history\n\n"
                        "| date | corpus | method | line1 acc | line2 acc |"
                        " full | cksum | ms p50 |\n"
                        "|------|--------|--------|-----------|-----------|"
                        "------|-------|--------|\n")
    with hist.open("a") as f:
        for m in methods:
            a_ = agg[m]
            f.write(f"| {time.strftime('%Y-%m-%d %H:%M')} | {a.corpus} | {m} | "
                    f"{100*a_['l1_acc']:.2f}% | {100*a_['l2_acc']:.2f}% | "
                    f"{100*a_['full_match_rate']:.2f}% | "
                    f"{100*a_['cksum_rate']:.2f}% | {a_['ms_p50']:.1f} |\n")

    if a.gate:
        return gate(agg, ROOT / a.gate, a.update_baseline)
    return 0


if __name__ == "__main__":
    sys.exit(main())
