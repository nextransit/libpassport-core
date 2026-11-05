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
    m = re.search(r"MRZ_CRNN_FEATURE_MD5=([0-9a-f]{32}|unknown)", text)
    return m.group(1) if m else None


def tool_md5(tool_path) -> str | None:
    try:
        return hashlib.md5(Path(tool_path).read_bytes()).hexdigest()
    except OSError:
        return None


def verify_crnn_fingerprint(tool_path, header_path=None, fatal: bool = False) -> bool:
    """True when the header fingerprint matches the tool (or is unknown)."""
    header_path = Path(header_path) if header_path else DEFAULT_HEADER
    stamped = crnn_header_md5(header_path)
    if stamped is None:
        print(f"[crnn-fingerprint] {header_path.name}: no MRZ_CRNN_FEATURE_MD5 stamp "
              f"(pre-fingerprint header) -- cannot verify feature provenance",
              file=__import__("sys").stderr)
        return True
    actual = tool_md5(tool_path)
    if actual is None:
        print(f"[crnn-fingerprint] tool {tool_path} not found -- cannot verify",
              file=__import__("sys").stderr)
        return True
    if stamped == "unknown":
        print(f"[crnn-fingerprint] header stamped 'unknown' vs tool md5={actual[:12]}",
              file=__import__("sys").stderr)
        return True
    if stamped != actual:
        msg = (f"[crnn-fingerprint] MISMATCH: crnn_weights.h was trained on features "
               f"from md5={stamped[:12]} but the tool under test is md5={actual[:12]}.\n"
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
    print(f"stamped={crnn_header_md5(header)} actual={tool_md5(tool)}")
    print("MATCH" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv))
