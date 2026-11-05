#!/usr/bin/env python3
"""Corpus manifest lock: sha256 manifest of every file in a corpus dir.

Why
---
The synthetic corpora live under gitignored, regenerable directories
(data/corpus_eval, data/corpus_eval_v2, ...).  One accidental
``gen_corpus.py`` re-run silently replaces the evaluation baseline, and
every historical number in metrics_history.md becomes incomparable --
this is the eval-side recurrence of the 10-24 training/eval band
distribution lock incident, and on 11-04 it actually happened (the old
bitmap corpus was regenerated in place and is now unrecoverable).

Usage
-----
  generate   corpus_dir [--out manifest.json]   write manifest (default:
              data/manifests/<dirname>.manifest.json, OUTSIDE the ignored
              corpus dir so it can be committed)
  verify     corpus_dir [--manifest m.json]     recompute + compare;
              exit 1 on drift (list first N mismatches)
  fingerprint corpus_dir                        short hash for log rows

Signing a manifest freezes "this directory content == that benchmark
number".  bench/eval scripts call verify_corpus_dir() at startup: a
missing manifest is allowed (first run), a MISMATCH is fatal unless
MRZ_ALLOW_CORPUS_DRIFT=1 (e.g. you just regenerated on purpose and are
about to re-sign).
"""

import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFEST_DIR = REPO / "6_mrz_ocr" / "data" / "manifests"
MAX_REPORT = 8


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def default_manifest_path(corpus_dir: Path) -> Path:
    return MANIFEST_DIR / f"{corpus_dir.name}.manifest.json"


def build_manifest(corpus_dir: Path) -> dict:
    corpus_dir = corpus_dir.resolve()
    entries = {}
    for root, _dirs, files in os.walk(corpus_dir):
        for name in sorted(files):
            p = Path(root) / name
            rel = str(p.relative_to(corpus_dir))
            entries[rel] = {"sha256": _sha256(p), "size": p.stat().st_size}
    return {
        "dir": str(corpus_dir),
        "files": len(entries),
        "entries": entries,
    }


def generate(corpus_dir: Path, out: Path | None) -> Path:
    corpus_dir = Path(corpus_dir)
    if out is None:
        out = default_manifest_path(corpus_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(corpus_dir)
    out.write_text(json.dumps(manifest, indent=0, sort_keys=True) + "\n")
    print(f"manifest: {out}")
    print(f"  dir   : {manifest['dir']}")
    print(f"  files : {manifest['files']}")
    print(f"  fp    : {fingerprint_of(manifest)}")
    return out


def fingerprint_of(manifest: dict) -> str:
    blob = json.dumps(manifest["entries"], sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def load_manifest(corpus_dir: Path, manifest: Path | None) -> tuple[dict, Path]:
    corpus_dir = Path(corpus_dir).resolve()
    if manifest is None:
        manifest = default_manifest_path(corpus_dir)
    if not manifest.exists():
        raise FileNotFoundError(
            f"no manifest for {corpus_dir.name}: {manifest}\n"
            f"  sign it once:  python3 {__file__} generate {corpus_dir}")
    return json.loads(manifest.read_text()), manifest


def verify(corpus_dir: Path, manifest_path: Path | None) -> list[str]:
    manifest, _mp = load_manifest(Path(corpus_dir), manifest_path)
    current = build_manifest(Path(corpus_dir))
    problems = []
    old, new = manifest["entries"], current["entries"]
    for rel in sorted(set(old) - set(new)):
        problems.append(f"missing  {rel}")
    for rel in sorted(set(new) - set(old)):
        problems.append(f"added    {rel}")
    for rel in sorted(set(old) & set(new)):
        if old[rel]["sha256"] != new[rel]["sha256"]:
            problems.append(f"drift    {rel}")
    return problems


def verify_corpus_dir(corpus_dir, fatal: bool = True, quiet: bool = False) -> str | None:
    """Startup hook for bench/eval scripts.

    Returns the 12-char corpus fingerprint when the directory matches its
    manifest (or has none yet), so callers can stamp it into result rows.
    Exits non-zero on drift unless fatal=False.
    """
    corpus_dir = Path(corpus_dir)
    mpath = default_manifest_path(corpus_dir)
    if not mpath.exists():
        if not quiet:
            print(f"[corpus-manifest] {corpus_dir.name}: unsigned (no manifest) -- "
                  f"sign with tools/corpus_manifest.py generate", file=sys.stderr)
        return None
    problems = verify(corpus_dir, mpath)
    if problems:
        for line in problems[:MAX_REPORT]:
            print(f"[corpus-manifest] DRIFT {line}", file=sys.stderr)
        if len(problems) > MAX_REPORT:
            print(f"[corpus-manifest] ... and {len(problems) - MAX_REPORT} more",
                  file=sys.stderr)
        if fatal and os.environ.get("MRZ_ALLOW_CORPUS_DRIFT") != "1":
            print(f"[corpus-manifest] {corpus_dir.name} does not match its manifest; "
                  f"numbers would not be comparable to previously recorded runs.\n"
                  f"  re-sign on purpose:  python3 tools/corpus_manifest.py generate {corpus_dir}\n"
                  f"  override:            MRZ_ALLOW_CORPUS_DRIFT=1", file=sys.stderr)
            sys.exit(2)
        return None
    fp = fingerprint_of(json.loads(mpath.read_text()))
    if not quiet:
        print(f"[corpus-manifest] {corpus_dir.name} ok, fingerprint {fp}", file=sys.stderr)
    return fp


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else ""
    if cmd == "generate":
        generate(Path(argv[2]), Path(argv[4]) if len(argv) > 4 and argv[3] == "--out" else None)
        return 0
    if cmd == "verify":
        problems = verify(Path(argv[2]), Path(argv[4]) if len(argv) > 4 and argv[3] == "--manifest" else None)
        if problems:
            for line in problems[:MAX_REPORT]:
                print("DRIFT", line)
            print(f"{len(problems)} file(s) differ from the manifest")
            return 1
        print("ok:", len(json.loads(default_manifest_path(Path(argv[2])).read_text())["entries"]),
              "files match")
        return 0
    if cmd == "fingerprint":
        manifest, _ = load_manifest(Path(argv[2]), None)
        print(fingerprint_of(manifest))
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
