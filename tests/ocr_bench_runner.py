"""Headless driver invoked by the GUI's OCR tab.

Runs `mrz_ocr_tool` (traditional) and/or `mrz_ocr_cnn_tool` on a
list of corpus images, returns per-image and aggregate statistics.
"""
from __future__ import annotations
import json, os, subprocess, time, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAD = ROOT / "6_mrz_ocr" / "build" / "mrz_ocr_tool"
CNN  = ROOT / "6_mrz_ocr" / "build" / "mrz_ocr_cnn_tool"
TESS = ROOT / "6_2_Tesseract" / "tesseract_tool.py"
# PaddleOCR via paddle_v6.py — Phase 3 PP-OCRv6 pure-rec decoder
# (project-split rows, rec only, no det). Uses PP-OCRv6_medium_rec via
# ONNX Runtime when available, else falls back to paddle_local
# (tesseract ocrb baseline). Same stdout contract on both paths.
PADDLE = ROOT / "6_1_PaddleOCR" / "paddle_v6.py"
# Indra OCR (datalab-to/indra-ocr) — tiny ONNX det+rec pair that runs its
# own detection over the whole page. Same stdout contract via mrz_indra.py.
INDRA = ROOT / "11_indra" / "mrz_indra.py"
CRNN = ROOT / "12_crnn" / "mrz_crnn_tool.py"

# Prefer the interpreter that has the ONNX Runtime dependency. The current
# Python is checked first so `python gui.py` and its Paddle subprocess use
# the same environment.
def _probe_paddle_python() -> str:
    import os as _os
    candidates = [
        sys.executable,
        "/tmp/paddle_venv/bin/python",
        str(ROOT / ".venv" / "bin" / "python"),
        "/tmp/paddle_venv/bin/python3",
    ]
    for cand in candidates:
        if not _os.path.exists(cand):
            continue
        try:
            probe = subprocess.run(
                [cand, "-c", "import onnxruntime"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if probe.returncode == 0:
            return cand
    return sys.executable


PADDLE_PY = _probe_paddle_python()
# The GUI's OCR batch test runs the SAME frozen evaluation corpus as
# bench.py (data/corpus_eval, ICAO-valid TD3, clean + realistic), NOT
# the historical 500-image free-form corpus (data/corpus). The old
# corpus mixed digits into alpha-only fields, which collides with the
# default strict ICAO syntax mask and reports misleadingly low line-1
# accuracy (e.g. 76% while the eval corpus scores 95.6%).
DATA = ROOT / "6_mrz_ocr" / "data" / "corpus_eval"
CORPUS_JSON = DATA / "corpus.json"

METHODS = {
    "traditional": ("传统模板", TRAD),
    "cnn":         ("轻量CNN",   CNN),
    "tesseract":   ("Tesseract", TESS),
    "paddle":      ("PaddleOCR", PADDLE),
    "indra":       ("IndraOCR",  INDRA),
    "crnn":        ("CRNN",      CRNN),
    # fuse 与 cnn 共用 host CNN tool，靠环境变量切换到 CNN+CRNN 融合头
    # （见 6_mrz_ocr/src/mrz_ocr_cnn.c）。GUI 的 METHOD_ORDER 一直包含
    # "fuse"，这里缺失会让 GUI 跑全部方案时 KeyError，traceback 混进
    # stdout 后 GUI 的 json.loads 报 "Expecting value: line 1 column 1"。
    "fuse":        ("融合CNN+CRNN", CNN),
}

# 每个方法调用工具时附加的环境变量。host CNN tool 用这些开关选择
# 识别头（MRZ_OCR_LINE_MODELS=1 用按行训练的权重，MRZ_OCR_FUSE=1 打开
# 融合头，MRZ_OCR_CRNN=1 换成序列头）。
# 与 tests/gui.py 的照片链路保持同一套组合。
METHOD_ENV = {
    "cnn":  {"MRZ_OCR_LINE_MODELS": "1", "MRZ_OCR_CRNN": "0"},
    "fuse": {"MRZ_OCR_LINE_MODELS": "1", "MRZ_OCR_CRNN": "0",
             "MRZ_OCR_FUSE": "1"},
}

def run_one(tool: Path, image: Path, env_extra: dict | None = None) -> dict:
    """Invoke the OCR tool and parse its text output.

    Compiled C tools are invoked directly; Python tools
    (*.py, e.g. tesseract_tool.py) are wrapped with sys.executable so
    the active interpreter (which has the extra OCR deps installed)
    is the one that runs Wrappers.

    env_extra: environment overrides for the tool (e.g. the fusion head
    switches of the host CNN tool)."""
    t0 = time.perf_counter()
    try:
        if str(tool).endswith(".py"):
            args = [sys.executable, str(tool), str(image)]
        else:
            args = [str(tool), str(image)]
        env = None
        if env_extra:
            env = dict(os.environ)
            env.update(env_extra)
        r = subprocess.run(args, text=True,
                           capture_output=True, timeout=15, env=env)
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
            "lines": out.get("result.line1y line1h line2y line2h", ""),
            "ms":    ms,
            "err":   r.stderr.strip() if r.returncode != 0 else ""}

def run_batch_paddle(images: list[tuple[str, Path]]) -> dict:
    """Batch-drive paddle_v6.py: one process decodes every image.

    Reusing one ONNX Runtime session across all images is ~100x cheaper than
    per-image subprocess spawns (each spawn would reload the ONNX model).

    Images is a list of (case_id, image_path). The listfile keeps the
    same order, and paddle_v6 emits one stdout contract block per image
    (6 result lines + ms + blank). Returns {case_id: result_dict}.
    """
    listfile = ROOT / "tests" / ".paddle_local_tmp" / f"_batch_{os.getpid()}.txt"
    listfile.parent.mkdir(parents=True, exist_ok=True)
    listfile.write_text("".join(f"{cid}\t{ip}\n" for cid, ip in images))
    proc = subprocess.Popen(
        [PADDLE_PY, str(PADDLE), "--batch", str(listfile)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    results: dict = {}
    idx = 0
    cur: dict | None = None
    for line in proc.stdout:
        line = line.rstrip("\n")
        if not line or line.startswith("@@PROGRESS@"):
            if cur is not None:
                results[images[idx][0]] = cur
                idx += 1
                cur = None
            continue
        k, _, v = line.partition(":")
        key = k.strip()
        if key == "ms":
            if cur is not None:
                cur["ms"] = float(v.strip())
        elif key == "result.ok":
            cur = {"ok": v.strip() == "OK", "err": "",
                   "line1": "", "line2": "",
                   "conf1": 0, "conf2": 0, "band": "", "ms": 0.0}
        elif key == "result.line1y line1h line2y line2h":
            if cur is not None:
                cur["lines"] = v.strip()
        elif key.startswith("result."):
            if cur is not None:
                cur[key[len("result."):]] = v.strip()
                if key in ("result.conf1", "result.conf2"):
                    cur[key[len("result."):]] = int(v.strip() or "0")
        elif key == "band.x band.y band.w band.h":
            if cur is not None:
                cur["band"] = v.strip()
    proc.wait()
    return results


def run_batch_indra(images: list[tuple[str, Path]]) -> dict:
    """Batch-drive mrz_indra.py: one process reads every image.

    Indra keeps its ONNX sessions alive across images, so one process is
    far cheaper than a per-image spawn (each spawn reloads the models).
    Returns {case_id: result_dict}.
    """
    listfile = (ROOT / "tests" / ".paddle_local_tmp" /
                f"_indra_batch_{os.getpid()}.txt")
    listfile.parent.mkdir(parents=True, exist_ok=True)
    listfile.write_text("".join(f"{ip}\n" for _cid, ip in images))
    proc = subprocess.Popen(
        [sys.executable, str(INDRA), "--batch", str(listfile)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    blocks: list[dict] = []
    cur: dict | None = None
    for line in proc.stdout:
        line = line.rstrip("\n")
        if not line:
            if cur is not None:
                blocks.append(cur)
                cur = None
            continue
        if line.startswith("#"):
            continue
        k, _, v = line.partition(":")
        key = k.strip()
        if key == "ms":
            if cur is not None:
                cur["ms"] = float(v.strip())
        elif key == "result.backend":
            cur = {"ok": False, "err": "", "line1": "", "line2": "",
                   "conf1": 0, "conf2": 0, "band": "", "ms": 0.0,
                   "backend": v.strip()}
        elif key == "result.ok":
            if cur is None:
                cur = {"ok": False, "err": "", "line1": "", "line2": "",
                       "conf1": 0, "conf2": 0, "band": "", "ms": 0.0}
            cur["ok"] = v.strip() == "OK"
        elif key == "result.line1y line1h line2y line2h":
            if cur is not None:
                cur["lines"] = v.strip()
        elif key.startswith("result."):
            if cur is not None:
                cur[key[len("result."):]] = v.strip()
                if key in ("result.conf1", "result.conf2"):
                    cur[key[len("result."):]] = int(v.strip() or "0")
        elif key == "band.x band.y band.w band.h":
            if cur is not None:
                cur["band"] = v.strip()
    if cur is not None:
        blocks.append(cur)
    proc.wait()
    results: dict = {}
    for (cid, _ip), blk in zip(images, blocks):
        results[cid] = blk
    return results


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
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise KeyError(
            f"unknown method(s): {', '.join(unknown)}; "
            f"known: {', '.join(sorted(METHODS))}")

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

    # Paddle uses one long-lived process (ONNX session loaded once).
    # Batch runs first; progress callbacks are emitted per-image here so
    # the GUI bar moves during the (slow) paddle pass.
    batch_paddle: dict = {}
    paddle_fired: set = set()
    if "paddle" in methods:
        batch_paddle = run_batch_paddle(
            [(rec["id"], DATA / rec["image"]) for rec in records])
        if on_record:
            done_n = 0
            for cid, _ in [(rec["id"], rec["image"]) for rec in records]:
                if cid in batch_paddle:
                    done_n += 1
                    paddle_fired.add(cid)
                    on_record(done_n, len(records), cid)

    batch_indra: dict = {}
    indra_fired: set = set()
    if "indra" in methods:
        batch_indra = run_batch_indra(
            [(rec["id"], DATA / rec["image"]) for rec in records])
        if on_record:
            done_n = 0
            for cid, _ in [(rec["id"], rec["image"]) for rec in records]:
                if cid in batch_indra:
                    done_n += 1
                    indra_fired.add(cid)
                    on_record(done_n, len(records), cid)

    for rec in records:
        image_path = DATA / rec["image"]
        per_method = {}
        for m in methods:
            if m == "paddle" and rec["id"] in batch_paddle:
                r = batch_paddle[rec["id"]]
            elif m == "indra" and rec["id"] in batch_indra:
                r = batch_indra[rec["id"]]
            else:
                r = run_one(METHODS[m][1], image_path, METHOD_ENV.get(m))
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
        if on_record and not ((methods == ["paddle"]
                               and rec["id"] in paddle_fired)
                              or (methods == ["indra"]
                                  and rec["id"] in indra_fired)):
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
    methods = args[0].split(",") if args else list(METHODS.keys())
    def _cb(done, total, cid):
        sys.stdout.write(f"@@PROGRESS@{done}@{total}@{cid}\n")
        sys.stdout.flush()
    print(json.dumps(run(case_ids, methods, _cb if progress else None),
                     indent=2))
