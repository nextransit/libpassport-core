#!/usr/bin/env python3
"""Evaluate the production photo path, not just the CNN band reader.

The old end-to-end script measures a single CNN pass over a band.  The app
runs `processFast`: fastPair -> nativeBand -> CNN engine, then a FUSE retry,
then Kotlin postProcess (line-1 enforcement, evidence-only line-2 repair),
decodesOk and MrzVerdict.  This script mirrors that path on the PC so the
number we report is the number the user sees.

Metrics are PASS / SUSPECT / FAIL from MrzVerdict, plus the legacy
line1-exact / line2-in-truth-set numbers for comparison, and exact matches
against the human-corrected truth in data/gt_frozen_passport_corrected.json.
A PASS that is not exact is a FALSE pass and is reported separately: those
are the dangerous ones.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_omtest_e2e as EE  # noqa: E402
import export_omtest_slices as E  # noqa: E402

DEFAULT_OMTEST = EE.DEFAULT_OMTEST
DEFAULT_TOOL = EE.DEFAULT_TOOL
FAST_NATIVE_MAX_WIDTH = EE.FAST_NATIVE_MAX_WIDTH
GT_CORRECTED = HERE.parent / "data" / "gt_frozen_passport_corrected.json"
W = [7, 3, 1]
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"


def load_corrected_truth():
    try:
        data = json.loads(GT_CORRECTED.read_text())
    except Exception:
        return {}
    out = {}
    for rec in data.get("records", []):
        name = Path(rec.get("id", "")).name
        line2 = rec.get("line2", "")
        if name and len(line2) == 44:
            out[name] = line2
            out[Path(name).stem] = line2
    return out

# Same ordered list as PhotoMrzPipeline.confusionPairs.
# BEGIN GENERATED CONFUSION PAIRS (tools/gen_confusion_tables.py -- do not edit by hand)
_PAIRS = [
    # curated base (same as PhotoMrzPipeline.confusionPairs)
    ("0", "O"), ("0", "Q"), ("0", "U"), ("0", "D"),
    ("0", "V"), ("0", "G"), ("1", "I"), ("1", "L"),
    ("1", "T"), ("1", "7"), ("5", "S"), ("2", "Z"),
    ("8", "B"), ("8", "R"), ("8", "X"), ("8", "3"),
    ("6", "G"), ("4", "W"), ("4", "A"), ("W", "H"),
    ("W", "A"), ("W", "U"), ("W", "M"), ("H", "N"),
    ("H", "M"), ("M", "N"), ("R", "B"), ("L", "C"),
    ("K", "B"), ("K", "X"), ("D", "U"), ("D", "O"),
    ("U", "V"), ("U", "O"), ("V", "Y"), ("Q", "O"),
    ("T", "I"), ("T", "7"), ("X", "K"),
    # measured on real photos (tools/collect_confusions.py)
    ("0", "8"),  # n=18
    ("C", "E"),  # n=7
    ("0", "1"),  # n=7
    ("8", "M"),  # n=6
    ("G", "O"),  # n=6
    ("H", "J"),  # n=5
    ("0", "9"),  # n=5
]
# END GENERATED CONFUSION PAIRS


def _build_confusions():
    raw = {}
    for a, b in _PAIRS:
        cost = 1 if (a in "0152689" or b in "0152689") else 2
        raw.setdefault(a, []).append((b, cost))
        raw.setdefault(b, []).append((a, cost))
    out = {}
    for k, vals in raw.items():
        seen = []
        for ch, cost in vals:
            if ch not in {x[0] for x in seen}:
                seen.append((ch, cost))
        out[k] = sorted(seen, key=lambda x: x[1])
    return out


CONFUSIONS = _build_confusions()
COUNTRY_FIX = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"}
DIGIT_POS = {9, 19, 27, 42, 43} | set(range(13, 19)) | set(range(21, 27))


def digit_for(seg: str) -> int:
    total = 0
    for i, c in enumerate(seg):
        if "0" <= c <= "9":
            v = ord(c) - 48
        elif "A" <= c <= "Z":
            v = ord(c) - 55
        else:
            v = 0
        total += v * W[i % 3]
    return total % 10


def _check_ok(seg: str, ch: str) -> bool:
    return "0" <= ch <= "9" and digit_for(seg) == ord(ch) - 48


def line2_ok(l2: str) -> bool:
    if len(l2) < 44:
        return False
    try:
        return (
            _check_ok(l2[0:9], l2[9])
            and _check_ok(l2[13:19], l2[19])
            and _check_ok(l2[21:27], l2[27])
            and _check_ok(l2[28:42], l2[42])
            and _check_ok(l2[0:10] + l2[13:20] + l2[21:43], l2[43])
        )
    except Exception:
        return False


def normalize_raw(raw: str) -> str:
    return "".join(c for c in raw.upper() if c == "<" or "A" <= c <= "Z" or "0" <= c <= "9")


def enforce_line1(raw: str) -> str:
    out = list(normalize_raw(raw)[:44].ljust(44, "<"))
    out[0] = "P"
    out[1] = "<"
    for i, c in enumerate(out):
        if i > 1 and c.isdigit():
            cand = next((x for x, _ in CONFUSIONS.get(c, []) if x.isalpha()), None)
            if cand:
                out[i] = cand
    return "".join(out)


def enforce_line2_slots(line: str) -> str:
    out = list(line)
    for i, c in enumerate(out):
        if i in DIGIT_POS and c.isalpha():
            cand = next((x for x, _ in CONFUSIONS.get(c, []) if x.isdigit()), None)
            if cand:
                out[i] = cand
        elif 10 <= i <= 12 and c.isdigit():
            if c in COUNTRY_FIX:
                out[i] = COUNTRY_FIX[c]
    return "".join(out)


def align_near44(raw: str):
    if len(raw) == 44:
        return raw
    if len(raw) == 43:
        for i in range(44):
            for ch in CHARS:
                trial = raw[:i] + ch + raw[i:]
                if line2_ok(enforce_line2_slots(trial)):
                    return trial
    elif len(raw) == 45:
        for i in range(45):
            trial = raw[:i] + raw[i + 1:]
            if line2_ok(enforce_line2_slots(trial)):
                return trial
    return None


def repair_line2(raw: str):
    filtered = normalize_raw(raw)
    aligned = align_near44(filtered)
    base = (aligned if aligned is not None else filtered)[:44].ljust(44, "<")
    normalized = enforce_line2_slots(base)
    if line2_ok(normalized):
        return normalized, []
    chars = list(normalized)
    reports = []

    def set_check(pos, val):
        if chars[pos] != val:
            reports.append((pos, val))
            chars[pos] = val

    set_check(9, str(digit_for(normalized[0:9])))
    set_check(19, str(digit_for(normalized[13:19])))
    set_check(27, str(digit_for(normalized[21:27])))
    set_check(42, str(digit_for(normalized[28:42])))
    set_check(43, str(digit_for(normalized[0:10] + normalized[13:20] + normalized[21:43])))
    trial = "".join(chars)
    if not line2_ok(trial):
        return normalized, []
    notes = [f"pos {p} {normalized[p]}->{v}" for p, v in reports]
    return trial, notes


def is_mrz_shape(l1: str, l2: str) -> bool:
    if len(l1) != 44 or len(l2) != 44:
        return False
    if any(not (c == "<" or "A" <= c <= "Z") for c in l1):
        return False
    if any(not (c == "<" or "A" <= c <= "Z" or "0" <= c <= "9") for c in l2):
        return False
    if any(not ("0" <= l2[i] <= "9") for i in (9, 19, 27, 42, 43)):
        return False
    if any(not ("A" <= l2[i] <= "Z") for i in range(10, 13)):
        return False
    if any(not ("0" <= l2[i] <= "9") for i in list(range(13, 19)) + list(range(21, 27))):
        return False
    if l2[20] not in "MF<":
        return False
    return True


def is_strict_td3(l1: str, l2: str) -> bool:
    if not is_mrz_shape(l1, l2):
        return False
    if l1[0] != "P" or l1[1] != "<":
        return False
    if any(not ("A" <= c <= "Z") for c in l1[2:5]):
        return False
    if "<<" not in l1[5:]:
        return False
    if any(not (c == "<" or "A" <= c <= "Z") for c in l1[5:]):
        return False
    return line2_ok(l2)


def rec_from_kv(kv):
    ok = kv.get("result.ok", "")
    try:
        conf1 = int(kv.get("result.conf1", "0"))
        conf2 = int(kv.get("result.conf2", "0"))
    except ValueError:
        conf1 = conf2 = 0
    return dict(
        raw1=kv.get("result.line1", ""),
        raw2=kv.get("result.line2", ""),
        conf1=conf1,
        conf2=conf2,
        struct=int(kv.get("result.struct2", "0") or 0) == 1,
        nat=int(kv.get("result.natsuspect", "0") or 0) == 1,
        country=int(kv.get("result.countrymm", "1") or 1) == 1,
        issuer=int(kv.get("result.issuersusp", "0") or 0) == 1,
        ok=ok,
        note="" if ok == "OK" else ok,
    )


def post_process(rec):
    raw1 = normalize_raw(rec["raw1"])
    raw2 = normalize_raw(rec["raw2"])
    line1 = enforce_line1(raw1)
    if len(raw2) >= 40 and "timeout" not in rec["note"] and "deadline" not in rec["note"]:
        line2, repairs = repair_line2(raw2)
    else:
        line2, repairs = raw2, []
    line1_repairs = 0
    if len(raw1) == len(line1):
        line1_repairs = sum(1 for a, b in zip(raw1, line1) if a != b)
    out = dict(rec)
    out.update(
        raw1=raw1,
        raw2=raw2,
        line1=line1,
        line2=line2,
        repairs=repairs,
        repairCount=line1_repairs + len(repairs),
    )
    return out


def decodes_ok(rec) -> bool:
    if not is_strict_td3(rec["line1"], rec["line2"]):
        return False
    if is_strict_td3(rec["raw1"], rec["raw2"]):
        return True
    return (
        len(rec["raw1"]) == 44
        and len(rec["raw2"]) == 44
        and 1 <= rec["repairCount"] <= 2
        and rec["conf1"] >= 65
        and rec["conf2"] >= 65
        and is_mrz_shape(rec["raw1"], rec["raw2"])
    )


def verdict_of(decode_ok: bool, struct_ok: bool, nat: bool, issuer: bool) -> str:
    if not decode_ok:
        return "FAIL"
    if not struct_ok or nat or issuer:
        return "SUSPECT"
    return "PASS"


def run_tool(tool: Path, band: Path, extra_env, diag_path=None):
    env = dict(os.environ)
    env["MRZ_OCR_LINE_MODELS"] = "1"
    env.update(extra_env)
    if diag_path is not None:
        env["MRZ_OCR_DIAG"] = str(diag_path)
    r = subprocess.run([str(tool), str(band)], capture_output=True, text=True,
                       env=env, timeout=60)
    kv = {}
    for ln in r.stdout.splitlines():
        if ":" in ln:
            k, _, v = ln.partition(":")
            kv[k.strip()] = v.strip()
    diag = None
    if diag_path is not None:
        try:
            diag = json.loads(Path(diag_path).read_text())
        except Exception:
            diag = None
    return kv, diag


def diag_summary(d):
    if not d:
        return None
    lines = d.get("lines") or []
    return {
        "status": d.get("status_name"),
        "skew": d.get("skew_deg"),
        "grid_ok": [ln.get("grid_ok") for ln in lines],
        "seg": [ln.get("seg_score") for ln in lines],
        "nchars": [ln.get("nchars") for ln in lines],
        "conf": [ln.get("conf_avg") for ln in lines],
        "ambiguous": [ln.get("ambiguous") for ln in lines],
    }


def pick_second(first, first_ok, second, second_ok) -> bool:
    if first_ok and not second_ok:
        return False
    if not first_ok and second_ok:
        return True
    if first["raw1"] == second["raw1"] and first["raw2"] == second["raw2"]:
        return False
    return second["conf1"] + second["conf2"] > first["conf1"] + first["conf2"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--omtest", default=str(DEFAULT_OMTEST))
    ap.add_argument("--tool", default=str(DEFAULT_TOOL))
    ap.add_argument("--json", default="")
    ap.add_argument("--diag-dir", default="")
    args = ap.parse_args()
    om = Path(args.omtest)
    tool = Path(args.tool)
    diag_dir = Path(args.diag_dir) if args.diag_dir else None
    if diag_dir:
        diag_dir.mkdir(parents=True, exist_ok=True)
    if not tool.exists():
        print(f"tool not found: {tool}", file=sys.stderr)
        return 2
    tmp = Path("/tmp/mrz_eval_prod")
    tmp.mkdir(parents=True, exist_ok=True)
    gt_corr = load_corrected_truth()

    rows = []
    for d in sorted(p for p in om.iterdir() if p.is_dir()):
        for txt in sorted((om / d.name).glob("*.jpg-result.txt")):
            img = txt.with_name(txt.name.replace("-result.txt", ""))
            if not img.exists():
                continue
            gt2 = gt_corr.get(img.name) or gt_corr.get(img.stem)
            l1, l2 = E.parse_board(txt)
            if not l1 or not l2:
                continue
            truth1 = E.fix_line1(l1)
            cands = E.all_valid_variants(l2, max_edits=2)
            if not cands:
                continue
            im = Image.open(img).convert("RGB")
            a = np.asarray(im, dtype=np.uint32)
            gray = ((a[:, :, 0] * 299 + a[:, :, 1] * 587 + a[:, :, 2] * 114 + 500)
                    // 1000).astype(np.uint8)
            pair = EE.fast_pair(gray)
            row = dict(file=f"{d.name}/{img.name}", n_cand=len(cands),
                       hard=(len(cands) == 1))
            if pair is None:
                row.update(located=False, verdict="FAIL")
                rows.append(row)
                continue
            xl, xr = EE.adaptive_xpad(gray, pair)
            top, bot = EE.native_band(gray, pair[0], pair[1], xl, xr)
            band = im.crop((xl, top, im.width - xr, bot))
            if band.width > FAST_NATIVE_MAX_WIDTH:
                nh = max(1, int(round(band.height * FAST_NATIVE_MAX_WIDTH / band.width)))
                band = band.resize((FAST_NATIVE_MAX_WIDTH, nh), Image.BILINEAR)
            p = tmp / (img.stem + "_band.ppm")
            band.save(p)

            dp_fuse = diag_dir / f"{d.name}_{img.stem}_fuse.json" if diag_dir else None
            dp_cnn = diag_dir / f"{d.name}_{img.stem}_cnn.json" if diag_dir else None
            kv_cnn, dj_cnn = run_tool(
                tool, p, {"MRZ_OCR_PAR": "1"}, dp_cnn)
            cnn = post_process(rec_from_kv(kv_cnn))
            cnn_ok = decodes_ok(cnn)
            need_second = bool(cnn["note"]) or not cnn_ok
            kv_fuse, dj_fuse = run_tool(
                tool, p, {"MRZ_OCR_FUSE": "1", "MRZ_OCR_PAR": "1"}, dp_fuse)
            fuse = post_process(rec_from_kv(kv_fuse))
            fuse_ok = decodes_ok(fuse)
            if need_second and pick_second(cnn, cnn_ok, fuse, fuse_ok):
                chosen, engine = fuse, "fuse"
            else:
                chosen, engine = cnn, "cnn"

            decode_ok = decodes_ok(chosen)
            raw_ok = is_strict_td3(chosen["raw1"], chosen["raw2"])
            v = verdict_of(decode_ok, chosen["struct"], chosen["nat"], chosen["issuer"])
            l2_in = chosen["line2"] in cands
            row.update(
                located=True,
                engine=engine,
                fuse_ok=fuse_ok,
                cnn_ok=cnn_ok,
                fuse_line2=fuse["line2"],
                cnn_line2=cnn["line2"],
                line1=chosen["line1"],
                line2=chosen["line2"],
                raw_line2=chosen["raw2"],
                conf1=chosen["conf1"],
                conf2=chosen["conf2"],
                repaired=chosen["repairCount"] > 0,
                decode_ok=decode_ok,
                raw_ok=raw_ok,
                struct_ok=chosen["struct"],
                nat=chosen["nat"],
                issuer=chosen["issuer"],
                verdict=v,
                line1_exact=(chosen["line1"] == truth1) if truth1 else None,
                l2_in=l2_in,
                truth2=gt2,
                truth2_ck=line2_ok(gt2) if gt2 else None,
                l2_exact=(chosen["line2"] == gt2) if gt2 else None,
                l2_exact_fuse=(fuse["line2"] == gt2) if gt2 else None,
                l2_exact_cnn=(cnn["line2"] == gt2) if gt2 else None,
                cksum=line2_ok(chosen["line2"]),
                truth1=truth1,
                diag_fuse=diag_summary(dj_fuse),
                diag_cnn=diag_summary(dj_cnn),
            )
            rows.append(row)

    located = [r for r in rows if r.get("located")]
    l1_eval = [r for r in located if r.get("truth1")]
    passed = [r for r in located if r.get("verdict") == "PASS"]
    suspect = [r for r in located if r.get("verdict") == "SUSPECT"]
    fail = [r for r in located if r.get("verdict") == "FAIL"]
    true_pass = sum(1 for r in passed if r.get("l2_in"))
    false_pass = sum(1 for r in passed if not r.get("l2_in"))
    corr = [r for r in located if r.get("truth2")]
    true_pass_corr = sum(1 for r in passed if r.get("l2_exact"))
    false_pass_corr = sum(1 for r in passed if r.get("truth2") and not r.get("l2_exact"))
    fuse_n = sum(1 for r in located if r.get("engine") == "fuse")
    cnn_n = sum(1 for r in located if r.get("engine") == "cnn")

    print(f"samples={len(rows)}  located={len(located)}  "
          f"unambiguous(line2 truth set size 1)={sum(1 for r in located if r.get('hard'))}")
    print(f"  verdict PASS    = {len(passed)}/{len(located)}"
          f"  (fixmrz TRUE={true_pass}  FALSE={false_pass})")
    print(f"    corrected truth: TRUE={true_pass_corr}  FALSE={false_pass_corr}")
    print(f"  verdict SUSPECT = {len(suspect)}/{len(located)}")
    print(f"  verdict FAIL    = {len(fail)}/{len(located)}")
    print(f"  engines: fuse chosen={fuse_n}  cnn chosen={cnn_n}")
    print(f"  legacy line1 exact = {sum(1 for r in l1_eval if r.get('line1_exact'))}/{len(l1_eval)}"
          f"  line2 in set = {sum(1 for r in located if r.get('l2_in'))}/{len(located)}"
          f"  cksum = {sum(1 for r in located if r.get('cksum'))}/{len(located)}")
    print(f"  corrected exact = chosen {sum(1 for r in corr if r.get('l2_exact'))}/{len(corr)}"
          f"  fuse {sum(1 for r in corr if r.get('l2_exact_fuse'))}/{len(corr)}"
          f"  cnn {sum(1 for r in corr if r.get('l2_exact_cnn'))}/{len(corr)}")
    corr_ck = [r for r in corr if r.get("truth2_ck")]
    corr_bad = [r for r in corr if not r.get("truth2_ck")]
    if corr_bad:
        print(f"  corrected truth cksum-valid = {len(corr_ck)}/{len(corr)}"
              f"  (invalid truth rows: {', '.join(Path(r['file']).stem for r in corr_bad)})")
        print(f"    exact on cksum-valid truth only:"
              f" chosen {sum(1 for r in corr_ck if r.get('l2_exact'))}/{len(corr_ck)}"
              f"  fuse {sum(1 for r in corr_ck if r.get('l2_exact_fuse'))}/{len(corr_ck)}"
              f"  cnn {sum(1 for r in corr_ck if r.get('l2_exact_cnn'))}/{len(corr_ck)}")
    print()
    print("non-PASS / false-PASS:")
    for r in rows:
        truth_ok = r.get("l2_exact") if r.get("truth2") else r.get("l2_in")
        if r.get("verdict") == "PASS" and truth_ok:
            continue
        why = "not located" if not r.get("located") else f"{r.get('verdict')} " \
            f"(decode={r.get('decode_ok')} exact={r.get('l2_exact')} " \
            f"in-set={r.get('l2_in')} " \
            f"struct={r.get('struct_ok')} nat={r.get('nat')} issuer={r.get('issuer')})"
        if r.get("truth2") and not r.get("truth2_ck"):
            why += "  [truth cksum-invalid]"
        print(f"  {r['file']:32s} {why}")
        if r.get("located"):
            print(f"      line2 = {r.get('line2')}")
            if r.get("truth2") and r.get("line2") != r.get("truth2"):
                print(f"      truth = {r.get('truth2')}")
    if diag_dir:
        print(f"  diag files = {len(list(diag_dir.glob('*.json')))} in {diag_dir}")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1, default=str))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
