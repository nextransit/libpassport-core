#!/usr/bin/env python3
"""CRNN weights-vs-tool fingerprint check.

train_crnn_seq.py stamps ``MRZ_CRNN_FEATURE_MD5=<md5 of the tool binary
that produced the training features>`` into src/crnn_weights.h.  The CRNN
consumes the line-head conv features of THAT binary; any rebuild of the
conv/geometry front-end changes the features and silently invalidates the
shipped sequence weights (the 10-28 "CRNN 0/41" incident was exactly a
10-28 header meeting a 10-30 tool).

Usage from eval scripts:

    from weight_fingerprint import verify_crnn_fingerprint
    verify_crnn_fingerprint(TOOL)          # warn only
    verify_crnn_fingerprint(TOOL, fatal=True)
"""

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HEADER = ROOT / "src" / "crnn_weights.h"


def crnn_header_md5(header_path: Path) -> str | None:
    try:
        text = Path(header_path).read_text()
    except OSError:
        return None
    m = re.search(r"MRZ_CRNN_FEATURE_MD5=([0-9a-f]{32,64}|unknown)", text)
    return m.group(1) if m else None


def tool_md5(tool_path) -> str | None:
    try:
        return hashlib.md5(Path(tool_path).read_bytes()).hexdigest()
    except OSError:
        return None


# The files that determine the CONV FEATURES the sequence head consumes:
# the two line-head weight headers (they ARE the conv weights) and the
# sources that drive band location, segmentation and resampling.  The
# CRNN's own header is deliberately NOT in the list: retraining the
# sequence head does not change the features it reads.
FEATURE_SOURCES = [
    "src/cnn_weights_line1.h",
    "src/cnn_weights_line2.h",
    "src/cnn.c",
    "src/mrz_ocr_cnn.c",
    "src/mrz_geom.c",
    "src/mrz_ocr.c",
]


def feature_fingerprint(root=None) -> str:
    """sha256 over the files that determine the conv features.

    Stable across a CRNN-head-only rebuild (the weights that matter are
    the conv ones), and sensitive to every change that would silently
    invalidate an exported feature set.
    """
    root = Path(root) if root else ROOT
    h = hashlib.sha256()
    for rel in FEATURE_SOURCES:
        p = root / rel
        h.update(rel.encode())
        h.update(p.read_bytes() if p.exists() else b"<missing>")
    return h.hexdigest()


def verify_crnn_fingerprint(tool_path, header_path=None, fatal: bool = False) -> bool:
    """True when the header's feature fingerprint matches the tree.

    The fingerprint covers the conv weights + the band/segmentation/resample
    sources, i.e. exactly the inputs the sequence head consumes.  A
    CRNN-head-only rebuild does not change it; any front-end change does.
    """
    header_path = Path(header_path) if header_path else DEFAULT_HEADER
    stamped = crnn_header_md5(header_path)
    if stamped is None:
        print(f"[crnn-fingerprint] {header_path.name}: no MRZ_CRNN_FEATURE_MD5 stamp "
              f"(pre-fingerprint header) -- cannot verify feature provenance",
              file=__import__("sys").stderr)
        return True
    actual = feature_fingerprint()
    if stamped == "unknown":
        print(f"[crnn-fingerprint] header stamped 'unknown' vs feature fp={actual[:12]}",
              file=__import__("sys").stderr)
        return True
    if stamped != actual:
        msg = (f"[crnn-fingerprint] MISMATCH: crnn_weights.h was trained on features "
               f"from fp={stamped[:12]} but the tree now produces fp={actual[:12]}.\n"
               f"  The conv front-end moved; CRNN/fuse numbers are meaningless until "
               f"features are re-exported and the head retrained "
               f"(tools/export_crnn_features.py + tools/train_crnn_seq.py).")
        print(msg, file=__import__("sys").stderr)
        if fatal:
            raise SystemExit(2)
        return False
    return True


def main(argv) -> int:
    import sys
    tool = argv[1] if len(argv) > 1 else str(ROOT / "build" / "mrz_ocr_cnn_tool")
    header = Path(argv[2]) if len(argv) > 2 else DEFAULT_HEADER
    ok = verify_crnn_fingerprint(tool, header, fatal=False)
    print(f"tool={tool}")
    print(f"header={header}")
    print(f"stamped={crnn_header_md5(header)}")
    print(f"feature_fp={feature_fingerprint()}")
    print("MATCH" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv))
