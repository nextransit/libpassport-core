#!/usr/bin/env bash
# Build a throw-away recogniser binary from arbitrary line-head headers.
#
# Why: the weights are `static const` arrays compiled into the binary, so
# every experiment that wants to try a head that is not the shipped one
# (holdout runs, per-line mixes, A/B comparisons) has to swap
# src/cnn_weights_line*.h, rebuild, and put them back.  Doing that by hand
# left the tree in the swapped state twice, which silently poisons the
# next build -- the "zero regression" number would then be measured
# against the wrong head.
#
# This script keeps the swap atomic: it copies the current headers to a
# temp dir, installs the requested pair, builds into a SEPARATE build
# tree (so build/ keeps the shipped binaries), copies the result out,
# and restores the headers via an EXIT trap -- including on failure.
#
# Usage:
#   tools/build_weights_variant.sh <line1.h> <line2.h> <out-binary> [pic]
#
#   pic   build the MRZ_WEIGHTS_SET_PIC variant (default: passport)
#
# Example -- evaluate a holdout pair on the assets/pic corpus:
#   tools/build_weights_variant.sh /tmp/w2/pic_hold_line1.h \
#       /tmp/w2/pic_hold_line2.h /tmp/tool_pic_hold pic
set -euo pipefail

if [ $# -lt 3 ]; then
    sed -n '2,20p' "$0"
    exit 2
fi

L1=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
L2=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
OUT=$3
VARIANT=${4:-passport}

SRC=$(cd "$(dirname "$0")/.." && pwd)
BUILD=${VARIANT_BUILD_DIR:-/tmp/wvariant_build}

[ -f "$L1" ] || { echo "missing: $L1" >&2; exit 1; }
[ -f "$L2" ] || { echo "missing: $L2" >&2; exit 1; }

if [ "$VARIANT" = "pic" ]; then
    H1=$SRC/src/cnn_weights_line1_pic.h
    H2=$SRC/src/cnn_weights_line2_pic.h
    TARGET=mrz_ocr_cnn_tool_pic
else
    H1=$SRC/src/cnn_weights_line1.h
    H2=$SRC/src/cnn_weights_line2.h
    TARGET=mrz_ocr_cnn_tool
fi

BACKUP=$(mktemp -d /tmp/wbak.XXXXXX)
cp "$H1" "$BACKUP/l1.h"
cp "$H2" "$BACKUP/l2.h"
restore() {
    cp "$BACKUP/l1.h" "$H1"
    cp "$BACKUP/l2.h" "$H2"
    rm -rf "$BACKUP"
    echo "headers restored"
}
trap restore EXIT

cp "$L1" "$H1"
cp "$L2" "$H2"

cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release > /tmp/wvariant_cfg.log 2>&1
cmake --build "$BUILD" --target "$TARGET" -j4 > /tmp/wvariant_build.log 2>&1

cp "$BUILD/$TARGET" "$OUT"
echo "built $OUT from $(basename "$L1") + $(basename "$L2") ($VARIANT)"
