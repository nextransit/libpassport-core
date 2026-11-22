#!/usr/bin/env python3
"""Check that the exported ONNX CNN reproduces the C reference exactly.

Reads a dump produced by tools/cnn_parity_dump.c and replays the same
glyph inputs through onnxruntime.  Gate: the argmax (predicted class) must
agree on every sample and the probability vectors must agree to within
`--tol` (float32 arithmetic order differs between the C loops and the ONNX
runtime, so the tolerance is on the probability values, not on bit patterns).

Exit code 0 = gate passed, 1 = gate failed.
"""

import argparse
import struct
import sys

import numpy as np
import onnxruntime as ort


def read_dump(path):
    with open(path, "rb") as fh:
        data = fh.read()
    magic, count, in_dim, out_dim = struct.unpack_from("<IIII", data, 0)
    if magic != 0x435a524D:
        raise SystemExit("bad magic in %s" % path)
    stride = 4 * (in_dim + out_dim)
    body = np.frombuffer(data, dtype="<f4", offset=16).reshape(count, in_dim + out_dim)
    return body[:, :in_dim].copy(), body[:, in_dim:].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="max allowed |p_onnx - p_c| (default 1e-4)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    x, p_c = read_dump(args.dump)
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    p_o = sess.run(None, {in_name: x.reshape(-1, 1, 12, 16)})[0].reshape(x.shape[0], -1)

    absdiff = np.abs(p_o.astype(np.float64) - p_c.astype(np.float64))
    maxdiff = float(absdiff.max()) if absdiff.size else 0.0
    meandiff = float(absdiff.mean()) if absdiff.size else 0.0

    cls_c = p_c.argmax(axis=1)
    cls_o = p_o.argmax(axis=1)
    mism = np.nonzero(cls_c != cls_o)[0]

    # Margin between the C top-1 and top-2 classes: a small margin means a
    # tiny numeric difference can flip the decision, so report the worst
    # margin seen among the mismatches.
    srt = np.sort(p_c, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    worst_margin = float(margin.min())

    print("samples      : %d" % x.shape[0])
    print("max |dp|     : %.3e   mean |dp|: %.3e   (tol %.1e)" % (maxdiff, meandiff, args.tol))
    print("argmax match : %d/%d  (mismatches %d)"
          % (x.shape[0] - mism.size, x.shape[0], mism.size))
    print("min top1-top2 margin (C): %.3e" % worst_margin)
    if mism.size:
        idx = mism[: min(10, mism.size)]
        for i in idx:
            print("  mismatch #%d  C=%d p=%.6f  onnx=%d p=%.6f  margin=%.3e"
                  % (i, cls_c[i], p_c[i, cls_c[i]], cls_o[i], p_o[i, cls_o[i]], margin[i]))
    ok = mism.size == 0 and maxdiff <= args.tol
    print("RESULT: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
