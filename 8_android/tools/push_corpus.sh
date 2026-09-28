#!/usr/bin/env bash
# push_corpus.sh — Copy the host MRZ evaluation corpus to the device's
# app-private external dir so the on-device bench can read it.
#
# Usage:
#   8_android/tools/push_corpus.sh [host-corpus-dir] [device-serial]
#
# Defaults:
#   host-corpus-dir  = ../6_mrz_ocr/data/corpus_eval   (the frozen eval set)
#   device-serial    = 192.168.26.123:5555            (the project's target)
set -euo pipefail
HOST_DIR="${1:-$(cd "$(dirname "$0")/../.."; pwd)/6_mrz_ocr/data/corpus_eval}"
SERIAL="${2:-192.168.26.123:5555}"

if [ ! -d "$HOST_DIR" ]; then
    echo "ERROR: host corpus dir not found: $HOST_DIR" >&2
    exit 2
fi
if [ ! -f "$HOST_DIR/corpus.json" ]; then
    echo "ERROR: $HOST_DIR/corpus.json missing — generate with:" >&2
    echo "    6_mrz_ocr/tests/gen_corpus.py --eval --out 6_mrz_ocr/data/corpus_eval --seed 2002" >&2
    exit 3
fi

PKG="com.nextransit.mrzbench"
DEST="/sdcard/Android/data/${PKG}/files/corpus"
echo ">> adb -s $SERIAL shell mkdir -p $DEST"
adb -s "$SERIAL" shell "mkdir -p $DEST"

# PPMs are large; use adb push in chunks via rsync-on-device workaround.
# We just iterate to keep the progress visible.
total=$(ls "$HOST_DIR"/img_*.ppm 2>/dev/null | wc -l | tr -d ' ')
echo ">> pushing $total PPM + 1 corpus.json to $SERIAL:$DEST"
i=0
for f in "$HOST_DIR"/img_*.ppm; do
    i=$((i+1))
    if ! adb -s "$SERIAL" push "$f" "$DEST/$(basename "$f")" >/dev/null; then
        echo "    ! failed at $i/$total on $(basename "$f")"
        exit 4
    fi
    if [ $((i % 50)) -eq 0 ]; then
        printf "    ... %d/%d\n" "$i" "$total" >&2
    fi
done
adb -s "$SERIAL" push "$HOST_DIR/corpus.json" "$DEST/corpus.json" >/dev/null
printf "    ... %d/%d\n" "$total" "$total" >&2
echo ">> done"
echo
echo "On device, the app will read: $DEST/corpus.json + $DEST/img_*.ppm"
