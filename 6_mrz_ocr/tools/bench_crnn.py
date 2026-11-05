#!/usr/bin/env python3
"""CNN vs CRNN comparison on a frozen corpus.

Both backends run the SAME C geometry front-end; only the classifier
differs (per-character FC vs GRU over the row). Metrics are identical to
tests/bench.py so the numbers are directly comparable to bench_results.json.

The CRNN backend is selected with MRZ_OCR_CRNN=1, which the C recogniser
honours only when generated weights are compiled in.

Prior A/B: the production pipeline applies structural priors (issuer fix,
strict-ICAO field masks, checksum beam) that are correct for real
documents but collide with synthetic corpora whose fields violate ICAO
structure.  The CNN backend is therefore measured in three modes --
  pipeline   production env (priors ON)
  nofix      MRZ_OCR_NOISSUERFIX=1 (issuer rewrite off)
  pureclf    NOISSUERFIX + NOBEAM + STRICT_ICAO=0 (classifier only)
so "prior-conflict errors" and "recognition errors" are reported
separately instead of being blended into one misleading score.
"""
from __future__ import annotations
import argparse, json, os, statistics, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from bench import compare, line2_checksum_ok  # noqa: E402
import corpus_manifest  # noqa: E402  (tools/ is on sys.path)

ROOT = Path(__file__).resolve().parents[1]
CNN = ROOT / "build" / "mrz_ocr_cnn_tool"


def run_tool(tool: Path, img: Path, env: dict):
    t0 = time.perf_counter()
    try:
        r = subprocess.run([str(tool), str(img)], capture_output=True,
                           text=True, timeout=30, env=env)
        ms = (time.perf_counter() - t0) * 1000.0
    except Exception:
        return "", "", 0, 0, False, 0.0
    parsed = {}
    for ln in r.stdout.splitlines():
        if ":" not in ln:
            continue
        k, _, v = ln.partition(":")
        parsed[k.strip()] = v.strip()
    return (parsed.get("result.line1", ""), parsed.get("result.line2", ""),
            int(parsed.get("result.conf1", "0") or 0),
            int(parsed.get("result.conf2", "0") or 0),
            r.returncode == 0, ms)


def metrics(recs):
    n = len(recs)
    l1c = l1t = l2c = l2t = 0
    full = cksum = ok = 0
    for r in recs:
        c1, t1, _ = compare(r["gt1"], r["line1"])
        c2, t2, _ = compare(r["gt2"], r["line2"])
        l1c += c1; l1t += t1; l2c += c2; l2t += t2
        if c1 == t1 and c2 == t2 and t1 and t2:
            full += 1
        if line2_checksum_ok(r["line2"]):
            cksum += 1
        ok += 1 if r["ok"] else 0
    ms = sorted(r["ms"] for r in recs)
    return {
        "n": n, "ok": ok, "ok_rate": ok / n if n else 0.0,
        "l1_acc": l1c / l1t if l1t else 0.0, "l2_acc": l2c / l2t if l2t else 0.0,
        "full_match_rate": full / n if n else 0.0,
        "cksum_rate": cksum / n if n else 0.0,
        "ms_p50": statistics.median(ms) if ms else 0.0,
        "ms_p95": ms[int(len(ms) * 0.95)] if ms else 0.0,
        "ms_mean": statistics.fmean(ms) if ms else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus_eval")
    ap.add_argument("--out", default="")
    ap.add_argument("--dump", default="",
                    help="write per-record results (predictions + GT + "
                         "params) to this JSON for failure analysis")
    ap.add_argument("--no-prior-ab", action="store_true",
                    help="skip the pipeline/nofix/pureclf A/B (CNN only)")
    ap.add_argument("--line-models", type=int, default=1)
    ap.add_argument("--parallel", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    corpus_dir = ROOT / args.corpus if not args.corpus.startswith("/") else Path(args.corpus)
    # Distribution lock: refuse to score against a corpus that drifted
    # from its signed manifest (regenerable + gitignored = one accidental
    # re-run would silently change the benchmark).  See tools/corpus_manifest.py.
    corpus_fp = corpus_manifest.verify_corpus_dir(corpus_dir, fatal=not args.limit)
    corpus = json.loads((corpus_dir / "corpus.json").read_text())["records"]
    if args.limit:
        corpus = corpus[: args.limit]

    base = dict(os.environ)
    base["MRZ_OCR_LINE_MODELS"] = "1" if args.line_models else "0"
    # Both backends must be measured under the SAME threading mode: the
    # parallel worker and the sequential path allocate the scratchpad
    # differently, which changes the extracted features.
    base["MRZ_OCR_PAR"] = "1" if args.parallel else "0"
    env_cnn = dict(base); env_cnn["MRZ_OCR_CRNN"] = "0"
    env_crnn = dict(base); env_crnn["MRZ_OCR_CRNN"] = "1"
    env_fuse = dict(base); env_fuse["MRZ_OCR_CRNN"] = "0"
    env_fuse["MRZ_OCR_FUSE"] = "1"
    env_nofix = dict(env_cnn); env_nofix["MRZ_OCR_NOISSUERFIX"] = "1"
    env_pure = dict(env_nofix)
    env_pure["MRZ_OCR_NOBEAM"] = "1"
    env_pure["MRZ_OCR_STRICT_ICAO"] = "0"

    out = {"cnn": [], "crnn": [], "fuse": []}
    out_ab = {"nofix": [], "pureclf": []}
    for rec in corpus:
        img = corpus_dir / rec["image"]
        for name, env in (("cnn", env_cnn), ("crnn", env_crnn), ("fuse", env_fuse)):
            l1, l2, c1, c2, ok, ms = run_tool(CNN, img, env)
            out[name].append({
                "id": rec["id"], "profile": rec.get("profile", "clean"),
                "line1": l1, "line2": l2, "conf1": c1, "conf2": c2,
                "ms": ms, "ok": ok, "gt1": rec["line1"], "gt2": rec["line2"],
            })
        if not args.no_prior_ab:
            for name, env in (("nofix", env_nofix), ("pureclf", env_pure)):
                l1, l2, c1, c2, ok, ms = run_tool(CNN, img, env)
                out_ab[name].append({
                    "id": rec["id"], "profile": rec.get("profile", "clean"),
                    "line1": l1, "line2": l2, "conf1": c1, "conf2": c2,
                    "ms": ms, "ok": ok, "gt1": rec["line1"], "gt2": rec["line2"],
                })

    m_cnn = metrics(out["cnn"])
    m_crnn = metrics(out["crnn"])
    m_fuse = metrics(out["fuse"])

    print(f"corpus={args.corpus} n={len(corpus)} line_models={args.line_models}")
    print(f"{'method':<8} {'ok':>7} {'l1_acc':>8} {'l2_acc':>8} {'full':>8} {'cksum':>8} "
          f"{'p50':>7} {'p95':>7}")
    for name, m in (("cnn", m_cnn), ("crnn", m_crnn), ("fuse", m_fuse)):
        print(f"{name:<8} {m['ok_rate']*100:6.2f}% {m['l1_acc']*100:7.2f}% "
              f"{m['l2_acc']*100:7.2f}% {m['full_match_rate']*100:7.2f}% "
              f"{m['cksum_rate']*100:7.2f}% {m['ms_p50']:6.2f} {m['ms_p95']:6.2f}")
    d = {k: m_crnn[k] - m_cnn[k] for k in
         ("l1_acc", "l2_acc", "full_match_rate", "cksum_rate", "ms_p50")}
    print("delta(crnn-cnn): " + "  ".join(
        f"{k}={v*100:+.2f}pp" if k != "ms_p50" else f"{k}={v:+.2f}ms" for k, v in d.items()))

    if not args.no_prior_ab:
        print("\n-- prior A/B (CNN backend) --")
        print(f"{'mode':<9} {'l1_acc':>8} {'l2_acc':>8} {'full':>8} {'cksum':>8}")
        for name, recs in (("pipeline", out["cnn"]), ("nofix", out_ab["nofix"]),
                           ("pureclf", out_ab["pureclf"])):
            m = metrics(recs)
            print(f"{name:<9} {m['l1_acc']*100:7.2f}% {m['l2_acc']*100:7.2f}% "
                  f"{m['full_match_rate']*100:7.2f}% {m['cksum_rate']*100:7.2f}%")

    # per-profile breakdown
    for prof in sorted({r["profile"] for r in out["cnn"]}):
        print(f"-- profile={prof}")
        for name in ("cnn", "crnn", "fuse"):
            m = metrics([r for r in out[name] if r["profile"] == prof])
            print(f"   {name:<6} l1={m['l1_acc']*100:6.2f}% l2={m['l2_acc']*100:6.2f}% "
                  f"full={m['full_match_rate']*100:6.2f}% cksum={m['cksum_rate']*100:6.2f}% "
                  f"p50={m['ms_p50']:.2f}ms")

    if not args.no_prior_ab:
        # degradation-axis breakdown of the CNN pipeline numbers: which
        # rendering axis actually costs accuracy (needs corpus.json params).
        axis_of = {r["id"]: r for r in corpus}
        def bucket(rec):
            return {"blur": rec.get("blur", 0), "illum": 40 if (rec.get("illum") or 0) >= 40 else (rec.get("illum") or 0),
                    "rot": "0" if (rec.get("rot") or 0) == 0 else "n0",
                    "scale": rec.get("scale", 0), "invert": rec.get("invert", 0)}
        for axis in ("blur", "illum", "rot", "scale", "invert"):
            groups = {}
            for r in out["cnn"]:
                b = bucket(axis_of[r["id"]])[axis]
                groups.setdefault(b, []).append(r)
            parts = "  ".join(
                f"{k}:full={metrics(v)['full_match_rate']*100:.1f}%/l2={metrics(v)['l2_acc']*100:.1f}%"
                for k, v in sorted(groups.items(), key=lambda kv: str(kv[0])))
            print(f"-- axis {axis}: {parts}")

    if args.dump:
        p = Path(args.dump)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {"cnn": out["cnn"], "crnn": out["crnn"], "fuse": out["fuse"]}
        if not args.no_prior_ab:
            payload["nofix"] = out_ab["nofix"]
            payload["pureclf"] = out_ab["pureclf"]
        p.write_text(json.dumps(payload))
        print("wrote", p)
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"cnn": m_cnn, "crnn": m_crnn, "delta": d},
                                indent=1))
        print("wrote", p)
    # Ledger: full-corpus runs (no --limit) are recorded in
    # metrics_history.md with the corpus fingerprint + weights md5 so the
    # row stays comparable forever (see tools/metrics_log.py).
    if not args.limit:
        import metrics_log
        from weight_fingerprint import tool_md5
        metrics_log.append_rows(
            [{"corpus": args.corpus, "method": name,
              "l1": m["l1_acc"], "l2": m["l2_acc"],
              "full": m["full_match_rate"], "cksum": m["cksum_rate"],
              "p50": m["ms_p50"]}
             for name, m in (("cnn", m_cnn), ("crnn", m_crnn), ("fuse", m_fuse))],
            corpus_fingerprint=corpus_fp,
            weights_md5=tool_md5(ROOT / "build" / "mrz_ocr_cnn_tool"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
