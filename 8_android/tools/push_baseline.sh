#!/usr/bin/env bash
# push_baseline.sh — Copy the host benchmark JSON so the on-device
# Compare tab can diff its own metrics against the host reference.
#
# Usage:
#   8_android/tools/push_baseline.sh [host-baseline-json] [device-serial]
#
# Default baseline = 6_mrz_ocr/bench_results.json (the frozen
# bench_results.json that backs 6_mrz_ocr/bench_results.md).
set -euo pipefail
HOST="${1:-$(cd "$(dirname "$0")/../.."; pwd)/6_mrz_ocr/bench_results.json}"
SERIAL="${2:-192.168.26.123:5555}"
PKG="com.nextransit.mrzbench"

if [ ! -f "$HOST" ]; then
    echo "ERROR: baseline not found: $HOST" >&2
    exit 2
fi

DEST="/sdcard/Android/data/${PKG}/files/results/host_bench_results.json"
adb -s "$SERIAL" shell "mkdir -p $(dirname "$DEST")"
adb -s "$SERIAL" push "$HOST" "$DEST"
echo ">> pushed $(basename "$HOST") -> $DEST"
echo ">> open Compare tab in the app, tap 'Import baseline'"
