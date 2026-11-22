#!/usr/bin/env python3
"""Build a hybrid line head: shipped conv layers + retrained fc layers.

Why this exists
---------------
The CRNN sequence head consumes the CNN's post-conv feature vector
(`cnn_row_features_batch()` -> `crnn_forward_line()`), so the CONV
layers of a line head are shared between the per-character CNN decoder
and the CRNN decoder.  Retraining a line head end to end (tools/
train_cnn_omtest.py rewrites conv1/conv2/conv3 as well as fc1/fc2)
therefore invalidates the CRNN head that was trained on the OLD
features: measured on the 50-photo product corpus, a v3 line-2 head
took the CRNN engine from 38/50 exact to 0/50 (it read
"Q655562647A<<6664464446467441<2" instead of
"EE20426536CHN7410024M2808303LMBGMBEGBBKSIM78").

The fc layers, on the other hand, sit AFTER the shared feature and are
used only by the per-character decoder.  Swapping just those keeps the
CRNN features bit-identical while letting the character head benefit
from the retrain.

Usage:
  python3 tools/merge_line_head.py \
      --base src/cnn_weights_line2.h \
      --fc   /tmp/cnn_weights_line2_v3.h \
      --out  /tmp/cnn_weights_line2_hybrid.h
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

FC_BLOCKS = ("FC1_W", "FC1_B", "FC2_W", "FC2_B")


def block_re(name: str) -> re.Pattern:
    return re.compile(
        r"static const float " + name + r"\[[^\]]*\](?: = \{[^;]*?\}|\s*=\s*\{[^;]*?\};)",
        re.S,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True,
                    help="shipped header providing the conv layers")
    ap.add_argument("--fc", required=True,
                    help="retrained header providing the fc layers")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="LINE",
                    help="array-name prefix, LINE for line2, LINE1 for line1")
    args = ap.parse_args()

    base = Path(args.base).read_text()
    fc = Path(args.fc).read_text()

    out = base
    for suffix in FC_BLOCKS:
        name = args.prefix + "_" + suffix
        pat = block_re(name)
        if not pat.search(out):
            raise SystemExit(f"{name} not found in {args.base}")
        m = pat.search(fc)
        if not m:
            raise SystemExit(f"{name} not found in {args.fc}")
        out = pat.sub(lambda _m: m.group(0), out, count=1)

    header = (
        "/* Hybrid line head: conv layers from the shipped head (they feed\n"
        " * the CRNN sequence features -- changing them invalidates the CRNN\n"
        " * head), fc layers from the retrain.\n"
        " * Built by 6_mrz_ocr/tools/merge_line_head.py; do not edit. */\n"
    )
    body = out[out.index("#ifndef"):]
    Path(args.out).write_text(header + body)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
