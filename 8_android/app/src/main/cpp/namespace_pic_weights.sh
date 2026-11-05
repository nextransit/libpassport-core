#!/usr/bin/env bash
# Rename every DEFINED global symbol of the pic weight-set archive with a
# `pic_` prefix so it can be linked alongside the passport set in one .so.
# Undefined refs (face_*, log, libc) are not in the list and stay intact.
# Usage: namespace_pic_weights.sh <nm> <objcopy> <libpic.a> <out.a>
set -euo pipefail
NM="$1"; OBJCOPY="$2"; IN="$3"; OUT="$4"
LIST="$(mktemp)"
trap 'rm -f "$LIST"' EXIT
"$NM" --defined-only "$IN" | awk '/ [TDRB] /{print $3}' | sort -u | sed "s/.*/& pic_&/" > "$LIST"
n=$(wc -l < "$LIST" | tr -d ' ')
"$OBJCOPY" --redefine-syms="$LIST" "$IN" "$OUT"
echo "namespace_pic_weights: renamed $n symbols -> $(basename "$OUT")"
