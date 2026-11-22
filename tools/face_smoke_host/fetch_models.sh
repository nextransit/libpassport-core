#!/usr/bin/env bash
# Refresh the SeetaFace2 model weights under assets/seetaface2_models/.
#
# The vendored 3_1_seetaface2/ source tree does NOT ship the .dat files.
# They are distributed as Dropbox links in the upstream README. We do
# not vendor them into git; the whole assets/ tree is git-ignored.
#
# This script is idempotent: if a file already exists with the expected
# size, it is skipped. Resume of partial downloads is handled by
# `wget -c`.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="${SEETA_FACE_MODELS_DIR:-$ROOT/assets/seetaface2_models}"
mkdir -p "$DEST"

# (filename, url, min_bytes). The min bytes guard against the Dropbox
# HTML stub (~17 bytes) that some links return before following the
# redirect.
ENTRIES=(
    "fd_2_00.dat|https://www.dropbox.com/s/cemt9fl48t5igfh/fd_2_00.dat?dl=1|1000000"
    "pd_2_00_pts5.dat|https://www.dropbox.com/s/noy8tien1gmw165/pd_2_00_pts5.dat?dl=1|100000"
    "pd_2_00_pts81.dat|https://www.dropbox.com/s/v41lmclaxpwow1d/pd_2_00_pts81.dat?dl=1|1000000"
    "fr_2_10.dat|https://www.dropbox.com/s/6aslqcokpljha5j/fr_2_10.dat?dl=1|100000000"
)

for entry in "${ENTRIES[@]}"; do
    name=${entry%%|*}
    rest=${entry#*|}
    url=${rest%|*}
    min=${rest##*|}

    out="$DEST/$name"
    if [ -s "$out" ]; then
        cur=$(wc -c <"$out" | tr -d ' ')
        if [ "$cur" -ge "$min" ]; then
            printf '[fetch] %-22s already present (%s bytes), skipping\n' "$name" "$cur"
            continue
        fi
        printf '[fetch] %-22s present but only %s bytes (< %s), re-downloading\n' \
            "$name" "$cur" "$min"
    fi
    printf '[fetch] %-22s <- %s\n' "$name" "$url"
    wget --no-check-certificate -c -q -O "$out.part" "$url"
    mv "$out.part" "$out"
    final=$(wc -c <"$out" | tr -d ' ')
    printf '[fetch] %-22s downloaded %s bytes\n' "$name" "$final"
done

printf '[fetch] done. Set SEETA_FACE_MODELS_DIR=%s and run face_real_load\n' "$DEST"
