#!/usr/bin/env python3
"""Export the tiny MRZ glyph CNN (6_mrz_ocr/src/cnn.c) to ONNX.

Why: the C reference implementation is the production path today, but it is
pure scalar/NEON C -- on a LoongArch board it costs ~3 ms per 88-glyph row
pair.  Every NPU runtime we could target (CANN/AscendCL, MindSpore Lite `om`,
OpenHarmony NNRt) ingests ONNX, so exporting the *exact* production weights
to ONNX is the mandatory precondition for any NPU port.  The geometry front
end (band locate / deskew / cell split / resample) stays in C: it is pure
pixel work with no tensor semantics and no NPU benefit.

Graph (mirrors cnn.c exactly):

    input (N,1,12,16) float32, raw glyph intensities
      -> (x - mean) / std            per-pixel normalisation, done in-graph
      -> Conv1 3x3 pad1  C1=8  + Relu   -> (N,8,12,16)
      -> MaxPool 2x1 (height only)     -> (N,8, 6,16)
      -> Conv2 3x3 pad1  C2=16 + Relu   -> (N,16,6,16)
      -> MaxPool 2x2                   -> (N,16,3, 8)
      -> Conv3 1x1       C3=4          -> (N, 4,3, 8)
      -> Reshape                        -> (N, 96)     [C,H,W order]
      -> Gemm fc1 96->64   + Relu
      -> Gemm fc2 64->37
      -> Softmax(axis=1)                -> (N, 37)

Weight layout translation (the C headers use trainer-specific orders):

    conv1_w  (kh,kw,co)                -> ONNX (co,ci=1,kh,kw)
    conv2_w  (kh,kw,ci,co)             -> ONNX (co,ci,kh,kw)
    conv3_w  (ci,co)                   -> ONNX (co,ci,1,1)
    fc1_w    (hidden, feat)            -> ONNX (hidden, feat)   [Gemm transB=1]
    fc2_w    (out, hidden)             -> ONNX (out, hidden)    [Gemm transB=1]

Usage:
    python3 tools/export_cnn_onnx.py --out-dir 6_mrz_ocr/models
    python3 tools/export_cnn_onnx.py --out-dir /tmp/onnx --heads line1 line2

The script never touches the weight headers; it only reads them.
"""

import argparse
import hashlib
import os
import re
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "src"))

# Architecture constants (must match 6_mrz_ocr/include/cnn.h).
IN_H, IN_W = 12, 16
C1, C2, C3 = 8, 16, 4
POOL1_H, POOL1_W = 6, 16
POOL2_H, POOL2_W = 3, 8
FLAT = C3 * POOL2_H * POOL2_W
HIDDEN, OUT = 64, 37

# head -> (weight header file, name prefix)
HEADS = {
    "default": ("cnn_weights.h", "DEFAULT_"),
    "line1": ("cnn_weights_line1.h", "LINE1_"),
    "line2": ("cnn_weights_line2.h", "LINE2_"),
    "line1_pic": ("cnn_weights_line1_pic.h", "LINE1_"),
    "line2_pic": ("cnn_weights_line2_pic.h", "LINE2_"),
}

_ARRAY_RE = re.compile(
    r"static\s+const\s+float\s+([A-Za-z0-9_]+)\s*\[[^\]]*\]\s*=\s*\{(.*?)\}\s*;",
    re.S,
)
_NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def parse_arrays(path):
    """Return {name: np.float32 array} for every `static const float X[] = {...}`."""
    with open(path, "r") as fh:
        text = fh.read()
    out = {}
    for name, body in _ARRAY_RE.findall(text):
        vals = [float(v) for v in _NUM_RE.findall(body)]
        out[name] = np.asarray(vals, dtype=np.float32)
    return out


def load_head(head):
    fname, prefix = HEADS[head]
    path = os.path.join(SRC, fname)
    arrays = parse_arrays(path)
    need = [
        "IN_MEAN", "IN_STD", "CONV1_W", "CONV1_B", "CONV2_W", "CONV2_B",
        "CONV3_W", "CONV3_B", "FC1_W", "FC1_B", "FC2_W", "FC2_B",
    ]
    got = {}
    for suffix in need:
        key = prefix + suffix
        if key not in arrays:
            raise SystemExit("missing %s in %s" % (key, path))
        got[suffix] = arrays[key]

    expect = {
        "IN_MEAN": IN_H * IN_W, "IN_STD": IN_H * IN_W,
        "CONV1_W": 3 * 3 * 1 * C1, "CONV1_B": C1,
        "CONV2_W": 3 * 3 * C1 * C2, "CONV2_B": C2,
        "CONV3_W": C2 * C3, "CONV3_B": C3,
        "FC1_W": FLAT * HIDDEN, "FC1_B": HIDDEN,
        "FC2_W": HIDDEN * OUT, "FC2_B": OUT,
    }
    for suffix, n in expect.items():
        if got[suffix].size != n:
            raise SystemExit(
                "%s in %s has %d values, expected %d"
                % (prefix + suffix, fname, got[suffix].size, n))

    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()[:16]
    return got, digest


def build_model(head, w, source_digest):
    mean = w["IN_MEAN"].reshape(1, 1, IN_H, IN_W)
    std = w["IN_STD"].reshape(1, 1, IN_H, IN_W)

    # (kh,kw,co) -> (co,ci,kh,kw)
    conv1 = w["CONV1_W"].reshape(3, 3, C1).transpose(2, 0, 1)[:, None, :, :]
    # (kh,kw,ci,co) -> (co,ci,kh,kw)
    conv2 = w["CONV2_W"].reshape(3, 3, C1, C2).transpose(3, 2, 0, 1)
    # (ci,co) -> (co,ci,1,1)
    conv3 = w["CONV3_W"].reshape(C2, C3).transpose(1, 0)[:, :, None, None]

    inits = [
        numpy_helper.from_array(mean.astype(np.float32), "in_mean"),
        numpy_helper.from_array(std.astype(np.float32), "in_std"),
        numpy_helper.from_array(conv1.astype(np.float32), "conv1_w"),
        numpy_helper.from_array(w["CONV1_B"].astype(np.float32), "conv1_b"),
        numpy_helper.from_array(conv2.astype(np.float32), "conv2_w"),
        numpy_helper.from_array(w["CONV2_B"].astype(np.float32), "conv2_b"),
        numpy_helper.from_array(conv3.astype(np.float32), "conv3_w"),
        numpy_helper.from_array(w["CONV3_B"].astype(np.float32), "conv3_b"),
        numpy_helper.from_array(w["FC1_W"].reshape(HIDDEN, FLAT).astype(np.float32), "fc1_w"),
        numpy_helper.from_array(w["FC1_B"].astype(np.float32), "fc1_b"),
        numpy_helper.from_array(w["FC2_W"].reshape(OUT, HIDDEN).astype(np.float32), "fc2_w"),
        numpy_helper.from_array(w["FC2_B"].astype(np.float32), "fc2_b"),
    ]

    nodes = [
        helper.make_node("Sub", ["input", "in_mean"], ["x_sub"], name="norm_sub"),
        helper.make_node("Div", ["x_sub", "in_std"], ["x_norm"], name="norm_div"),
        helper.make_node(
            "Conv", ["x_norm", "conv1_w", "conv1_b"], ["c1"],
            kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1], name="conv1"),
        helper.make_node("Relu", ["c1"], ["c1r"], name="relu1"),
        helper.make_node(
            "MaxPool", ["c1r"], ["p1"],
            kernel_shape=[2, 1], strides=[2, 1], name="pool1"),
        helper.make_node(
            "Conv", ["p1", "conv2_w", "conv2_b"], ["c2"],
            kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1], name="conv2"),
        helper.make_node("Relu", ["c2"], ["c2r"], name="relu2"),
        helper.make_node(
            "MaxPool", ["c2r"], ["p2"],
            kernel_shape=[2, 2], strides=[2, 2], name="pool2"),
        helper.make_node(
            "Conv", ["p2", "conv3_w", "conv3_b"], ["c3"],
            kernel_shape=[1, 1], name="conv3"),
        helper.make_node(
            "Reshape", ["c3", "shape_flat"], ["flat"],
            name="flatten"),
        helper.make_node(
            "Gemm", ["flat", "fc1_w", "fc1_b"], ["h1"],
            transB=1, name="fc1"),
        helper.make_node("Relu", ["h1"], ["h1r"], name="relu_fc1"),
        helper.make_node(
            "Gemm", ["h1r", "fc2_w", "fc2_b"], ["logits"],
            transB=1, name="fc2"),
        helper.make_node("Softmax", ["logits"], ["probs"], axis=1, name="softmax"),
    ]
    inits.append(numpy_helper.from_array(
        np.asarray([-1, FLAT], dtype=np.int64), "shape_flat"))

    graph = helper.make_graph(
        nodes, "mrz_cnn_%s" % head,
        [helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, ["N", 1, IN_H, IN_W])],
        [helper.make_tensor_value_info(
            "probs", TensorProto.FLOAT, ["N", OUT])],
        initializer=inits)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)],
        producer_name="export_cnn_onnx.py")
    model.doc_string = (
        "MRZ glyph CNN exported from 6_mrz_ocr/src/cnn.c; weights from %s "
        "(sha256[:16]=%s); float32, no preprocessing beyond the in-graph "
        "per-pixel (x-mean)/std." % (HEADS[head][0], source_digest))
    onnx.checker.check_model(model)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(HERE, "..", "models"))
    ap.add_argument("--heads", nargs="+", default=sorted(HEADS.keys()))
    ap.add_argument("--prefix", default="mrz_cnn_")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for head in args.heads:
        if head not in HEADS:
            raise SystemExit("unknown head %r" % head)
        w, digest = load_head(head)
        model = build_model(head, w, digest)
        path = os.path.join(args.out_dir, "%s%s.onnx" % (args.prefix, head))
        onnx.save(model, path)
        print("%-28s %8.1f KB  <- %s (sha256:%s)"
              % (os.path.relpath(path), os.path.getsize(path) / 1024.0,
                 HEADS[head][0], digest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
