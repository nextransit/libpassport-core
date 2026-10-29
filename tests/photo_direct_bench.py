#!/usr/bin/env python3
"""Benchmark every engine on the photo direct-recognition path.

For each photo this runs:
  1. the shipped path `_ocr_photo_engines` (Tesseract fallback ON),
  2. each engine alone on the same native-gap band (fallback OFF),
  3. Indra on the full photo, to exercise its own detector.

Run:
    python3 tests/photo_direct_bench.py
    python3 tests/photo_direct_bench.py --limit 10
    python3 tests/photo_direct_bench.py --dirs assets/pic
    python3 tests/photo_direct_bench.py --methods traditional,cnn,crnn
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))


def _load_gui():
    spec = importlib.util.spec_from_file_location(
        "gui", ROOT / "tests" / "gui.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pct(values, p):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((p / 100.0) * (len(ordered) - 1)))
    return ordered[max(0, min(len(ordered) - 1, idx))]


def _char_acc(pred, gt):
    pred = pred or ""
    gt = gt or ""
    n = max(len(pred), len(gt))
    if n == 0:
        return 0.0
    hit = sum(1 for i in range(n)
              if (pred[i] if i < len(pred) else "_") ==
                 (gt[i] if i < len(gt) else "_"))
    return hit / n


def _slim(res):
    return {
        "status": "skip" if res.get("engine_status") == "SKIP" else "run",
        "decode_ok": bool(res.get("decode_ok")),
        "engine_ok": bool(res.get("engine_ok")),
        "engine_status": res.get("engine_status", ""),
        "line1": res.get("line1") or "",
        "line2": res.get("line2") or "",
        "conf1": res.get("conf1", 0),
        "conf2": res.get("conf2", 0),
        "backend": res.get("backend") or "",
        "error": res.get("error") or "",
        "note": res.get("note") or "",
        "ms": float(res.get("ms", 0.0)),
    }


def _aggregate(records, keys, gui):
    total = len(records)
    located = sum(1 for r in records if r.get("pair"))
    out = {}
    for key in keys:
        st = {
            "total": total,
            "located": total if key.endswith("_fullpage") else located,
            "ran": 0,
            "decode": 0, "shaped": 0, "fail": 0,
            "skip": 0, "no_pair": 0, "fallback": 0,
            "ms": [], "gt_total": 0, "gt_exact": 0, "gt_bad": 0,
            "acc1": [], "acc2": [],
        }
        for r in records:
            item = (r.get("raw") or {}).get(key)
            gt = r.get("gt")
            if gt:
                if len(gt.get("line1", "")) == 44 and len(gt.get("line2", "")) == 44:
                    st["gt_total"] += 1
                    lines = item or {}
                    st["acc1"].append(_char_acc(lines.get("line1"), gt["line1"]))
                    st["acc2"].append(_char_acc(lines.get("line2"), gt["line2"]))
                    if (lines.get("decode_ok") and
                            lines.get("line1") == gt["line1"] and
                            lines.get("line2") == gt["line2"]):
                        st["gt_exact"] += 1
                else:
                    st["gt_bad"] += 1
            if item is None or item.get("status") == "no_pair":
                st["no_pair"] += 1
                continue
            if item.get("status") == "skip":
                st["skip"] += 1
                continue
            st["ran"] += 1
            if item.get("decode_ok"):
                st["decode"] += 1
            elif gui._rst_mrz_shaped(item.get("line1"), item.get("line2")):
                st["shaped"] += 1
            else:
                st["fail"] += 1
            st["ms"].append(item.get("ms", 0.0))
            if "tesseract-fallback" in (item.get("backend") or ""):
                st["fallback"] += 1
        out[key] = st
    return out


def _official(records, methods):
    grades = Counter()
    ms = []
    verified = 0
    engines = {
        m: {"runs": 0, "decode": 0, "fallback": 0, "ms": []}
        for m in methods
    }
    for r in records:
        o = r.get("official") or {}
        grades[o.get("grade", "ERROR")] += 1
        verified += int(bool(o.get("verified")))
        ms.append(float(o.get("ms", 0.0)))
        for e in o.get("engines") or []:
            m = e.get("method") or "?"
            st = engines.setdefault(
                m, {"runs": 0, "decode": 0, "fallback": 0, "ms": []})
            st["runs"] += 1
            st["decode"] += int(bool(e.get("decode_ok")))
            st["fallback"] += int(
                "tesseract-fallback" in (e.get("backend") or ""))
            st["ms"].append(float(e.get("ms", 0.0)))
    return {
        "photos": len(records),
        "grades": dict(grades),
        "verified": verified,
        "ms_mean": sum(ms) / len(ms) if ms else 0.0,
        "ms_median": _pct(ms, 50),
        "ms_p95": _pct(ms, 95),
        "engines": engines,
    }


def _fmt_stat(st):
    total = st["total"] or 1
    located = st["located"] or 1
    ran = st["ran"] or 1
    mean = sum(st["ms"]) / len(st["ms"]) if st["ms"] else 0.0
    gt_total = st["gt_total"] or 0
    acc1 = (sum(st["acc1"]) / len(st["acc1"]) * 100.0) if st["acc1"] else 0.0
    acc2 = (sum(st["acc2"]) / len(st["acc2"]) * 100.0) if st["acc2"] else 0.0
    exact = f"{st['gt_exact']}/{gt_total}" if gt_total else "-"
    bad = f" bad{st['gt_bad']}" if st.get("gt_bad") else ""
    return (
        f"{st['decode']}/{total}={st['decode']/total*100:5.1f}% "
        f"loc={st['decode']/located*100:5.1f}% "
        f"SUS={st['shaped']:2d} FAIL={st['fail']:2d} "
        f"SKIP={st['skip']:2d} NOPAIR={st['no_pair']:2d} "
        f"ms={mean:6.1f}/{_pct(st['ms'],95):6.1f} "
        f"GT={exact}{bad} acc={acc1:5.1f}/{acc2:5.1f}"
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dirs", default="")
    ap.add_argument("--methods", default="")
    ap.add_argument("--no-indra-fullpage", action="store_true")
    ap.add_argument("--resident", action="store_true",
                    help="预加载模型并常驻引擎；测的是单次完整识别时间，"
                         "不含进程/模型启动开销")
    ap.add_argument("--no-preload", action="store_true",
                    help="与 --resident 相反：每张照片重新起进程（历史口径）")
    ap.add_argument("--json", type=Path,
                    default=ROOT / "10_ocr_research" / "photo_direct_bench.json")
    args = ap.parse_args(argv)

    gui = _load_gui()
    # The GUI pre-warms the resident engine pool on startup; the bench
    # warms it itself when --resident is given, so disable the GUI's
    # background warm-up to keep timings comparable.
    gui.PassportGUI.AUTO_WARM_ENGINES = False
    app = gui.PassportGUI()
    app.update()
    app.withdraw()
    resident = bool(args.resident) and not args.no_preload
    preload_ms = 0.0
    if resident:
        # Load every model BEFORE the first photo so the reported times
        # are recognition time, not process / model start-up time.
        sys.path.insert(0, str(ROOT / "tests"))
        import mrz_engine_pool
        preload_ms = mrz_engine_pool.warmup_all(gui_module=gui)
        print(f"preloaded engines in {preload_ms:.0f} ms")
    try:
        app.rst_method_both.set(True)
        methods = list(gui.METHOD_ORDER)
        if args.methods:
            wanted = [m.strip() for m in args.methods.split(",") if m.strip()]
            methods = [m for m in methods if m in wanted]
        app._rst_selected_methods = lambda: list(methods)
        keys = list(methods)
        if "indra" in methods and not args.no_indra_fullpage:
            keys.append("indra_fullpage")

        dirs = [d.strip() for d in args.dirs.split(",") if d.strip()]
        photos = []
        for label, path in gui.iter_passport_photos():
            if dirs and not any(label == d or label.startswith(d + "/")
                                for d in dirs):
                continue
            photos.append((label, path))
        if args.limit:
            photos = photos[:args.limit]

        gt_map = app._gt_map()
        # gui.py defaults to the resident engine pool (interactive speed);
        # the bench must pin the flag so --no-preload keeps the historical
        # per-photo sub-process behaviour and numbers stay comparable.
        app._rst_resident = resident
        records = []
        started = time.perf_counter()
        for idx, (label, path) in enumerate(photos, 1):
            t0 = time.perf_counter()
            try:
                rec = app._ocr_photo_engines(str(path))
            except Exception as exc:
                rec = {"grade": "ERROR", "verified": False, "ms": 0.0,
                       "reason": f"{type(exc).__name__}: {exc}",
                       "engines": [], "pair": None, "work2": None,
                       "scale": 1.0, "band_rgb": None,
                       "ocr": {"line1": "", "line2": ""},
                       "loc": {}, "decode": {}}
            wall_ms = (time.perf_counter() - t0) * 1000.0
            ocr = rec.get("ocr") or {}
            l1 = ocr.get("line1") or ""
            l2 = ocr.get("line2") or ""
            chosen = None
            for e in rec.get("engines") or []:
                if (e.get("decode_ok") and e.get("line1") == l1 and
                        e.get("line2") == l2):
                    chosen = e.get("method")
                    break
            gt = gt_map.get(path.name)
            record = {
                "label": label,
                "path": str(path),
                "pair": bool(rec.get("pair")),
                "official": {
                    "grade": rec.get("grade"),
                    "verified": bool(rec.get("verified")),
                    "ms": float(rec.get("ms", wall_ms)),
                    "wall_ms": wall_ms,
                    "reason": rec.get("reason") or "",
                    "chosen": chosen,
                    "line1": l1,
                    "line2": l2,
                    "engines": [{
                        "method": e.get("method"),
                        "decode_ok": bool(e.get("decode_ok")),
                        "engine_ok": bool(e.get("engine_ok")),
                        "engine_status": e.get("engine_status", ""),
                        "conf1": e.get("conf1", 0),
                        "conf2": e.get("conf2", 0),
                        "backend": e.get("backend") or "",
                        "error": e.get("error") or "",
                        "ms": float(e.get("ms", 0.0)),
                    } for e in rec.get("engines") or []],
                },
                "gt": ({"line1": gt[0], "line2": gt[1]} if gt else None),
                "raw": {},
            }
            pair = rec.get("pair")
            work2 = rec.get("work2")
            scale = float(rec.get("scale") or 1.0)
            rgb = None
            bgr = cv2.imread(str(path))
            if bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            band_path = None
            tmp_path = None
            if pair and work2 is not None and rgb is not None and scale > 0:
                try:
                    native_band = app._rst_native_mrz_band(
                        rgb, pair, scale,
                        source_bottom=int(round(work2.shape[0] / scale)))
                    tmp = tempfile.NamedTemporaryFile(
                        suffix=".ppm", prefix="_photo_bench_", delete=False)
                    tmp.close()
                    tmp_path = Path(tmp.name)
                    band_path = tmp_path
                    Image.fromarray(native_band).save(
                        str(band_path), format="PPM")
                except Exception:
                    native_band = None
                    band_path = None
            else:
                native_band = None

            if rgb is not None:
                for m in methods:
                    if pair and band_path is not None:
                        try:
                            results, _ = app._rst_run_ocr_methods(
                                rgb, pair, scale, band_path, [m],
                                native_band=native_band,
                                source_path=str(path),
                                allow_tesseract_fallback=False,
                                resident=resident)
                            record["raw"][m] = _slim(results[0])
                        except Exception as exc:
                            record["raw"][m] = {
                                "status": "run", "decode_ok": False,
                                "engine_ok": False, "line1": "", "line2": "",
                                "conf1": 0, "conf2": 0, "ms": 0.0,
                                "backend": "", "note": "",
                                "error": f"{type(exc).__name__}: {exc}"}
                    else:
                        record["raw"][m] = {"status": "no_pair"}
                if pair and "indra" in methods and not args.no_indra_fullpage:
                    try:
                        results, _ = app._rst_run_ocr_methods(
                            rgb, pair, scale, None, ["indra"],
                            native_band=None, source_path=str(path),
                            allow_tesseract_fallback=False)
                        record["raw"]["indra_fullpage"] = _slim(results[0])
                    except Exception as exc:
                        record["raw"]["indra_fullpage"] = {
                            "status": "run", "decode_ok": False,
                            "engine_ok": False, "line1": "", "line2": "",
                            "conf1": 0, "conf2": 0, "ms": 0.0,
                            "backend": "", "note": "",
                            "error": f"{type(exc).__name__}: {exc}"}
                elif (not pair) and "indra" in methods and \
                        not args.no_indra_fullpage:
                    try:
                        results, _ = app._rst_run_ocr_methods(
                            rgb, pair, scale, None, ["indra"],
                            native_band=None, source_path=str(path),
                            allow_tesseract_fallback=False)
                        record["raw"]["indra_fullpage"] = _slim(results[0])
                    except Exception as exc:
                        record["raw"]["indra_fullpage"] = {
                            "status": "run", "decode_ok": False,
                            "engine_ok": False, "line1": "", "line2": "",
                            "conf1": 0, "conf2": 0, "ms": 0.0,
                            "backend": "", "note": "",
                            "error": f"{type(exc).__name__}: {exc}"}

            if tmp_path is not None:
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            records.append(record)
            print(f"[{idx}/{len(photos)}] {label[:48]:48s} "
                  f"{record['official']['grade']:8s} "
                  f"pair={record['pair']!s:5s} "
                  f"{record['official']['ms']:7.0f} ms", flush=True)

        raw_stats = _aggregate(records, keys, gui)
        official = _official(records, methods)
        elapsed = time.perf_counter() - started

        print("\n=== official direct-recognition path (fallback ON) ===")
        print(f"photos={official['photos']} "
              f"PASS={official['grades'].get('PASS', 0)} "
              f"SUSPECT={official['grades'].get('SUSPECT', 0)} "
              f"REJECT={official['grades'].get('REJECT', 0)} "
              f"verified={official['verified']} "
              f"ms mean/median/p95="
              f"{official['ms_mean']:.1f}/"
              f"{official['ms_median']:.1f}/"
              f"{official['ms_p95']:.1f}")
        print(f"{'engine':16s} {'runs':>5s} {'decode':>7s} "
              f"{'fallback':>8s} {'ms mean/p95':>14s}")
        for m, st in official["engines"].items():
            mean = sum(st["ms"]) / len(st["ms"]) if st["ms"] else 0.0
            print(f"{m:16s} {st['runs']:5d} {st['decode']:7d} "
                  f"{st['fallback']:8d} "
                  f"{mean:7.1f}/{_pct(st['ms'], 95):6.1f}")

        print("\n=== each engine alone on the same band (fallback OFF) ===")
        print(f"photos={len(records)} "
              f"located={raw_stats[keys[0]]['located'] if keys else 0} "
              f"elapsed={elapsed:.1f}s")
        for m in keys:
            print(f"{m:16s} {_fmt_stat(raw_stats[m])}")

        payload = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_s": round(elapsed, 3),
            "resident": resident,
            "preload_ms": round(preload_ms, 3),
            "tesseract_fallback": False,
            "methods": methods,
            "official": official,
            "raw": raw_stats,
            "records": records,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1),
            encoding="utf-8")
        print(f"\njson: {args.json}")
        return 0
    finally:
        app.destroy()
        if resident:
            # Release the resident C tools explicitly instead of relying
            # on interpreter shutdown.
            try:
                import mrz_engine_pool
                mrz_engine_pool.close_all()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
