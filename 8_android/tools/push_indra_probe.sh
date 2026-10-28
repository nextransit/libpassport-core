#!/usr/bin/env bash
# push_indra_probe.sh — Push the Indra ONNX probe tensors and models to the
# device so IndraOnnxProbeTest can compare the on-device ORT output against
# the host (x86_64) numbers.
#
# Usage: 8_android/tools/push_indra_probe.sh [device-serial]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SERIAL="${1:-192.168.26.131:5555}"
PKG="com.nextransit.mrzbench"
BASE="/sdcard/Android/data/${PKG}/files"

echo ">> serial=$SERIAL"
adb -s "$SERIAL" shell "mkdir -p $BASE/indra_probe $BASE/models/indra"

echo ">> pushing probe tensors"
for f in "$HERE"/indra_probe/*.bin; do
    adb -s "$SERIAL" push "$f" "$BASE/indra_probe/" >/dev/null
done

echo ">> pushing indra models"
for f in "$HERE"/../app/src/main/assets/models/indra/*; do
    [ -f "$f" ] || continue
    adb -s "$SERIAL" push "$f" "$BASE/models/indra/" >/dev/null
done

adb -s "$SERIAL" shell "ls -la $BASE/indra_probe $BASE/models/indra"
echo ">> done"
