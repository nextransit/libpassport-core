#!/usr/bin/env bash
# Device-side verification for the 2026-11-05 optimisation batch.
#
#   1. probs_fnv parity: the -O3 / -ffp-contract=off device build must
#      reproduce the HOST conv probabilities bit for bit.  The host
#      reference is printed by this script; any mismatch means the
#      device numbers are not comparable to the host bench (the gate
#      that -ffp-contract=off was protecting).
#   2. Latency: the per-image p50/p95 on the device for the product
#      path, checked against the 100 ms p95 budget.
#
# Requires: adb with a device, an NDK (see NDK below), and the repo
# checkout.  Run from the repo root:
#
#     bash 8_android/tools/verify_device.sh
#
# Environment overrides:
#   NDK         NDK root (default: the one in 8_android/local.properties)
#   DEVICE      adb device serial (default: the only device)
#   ITER        bench iterations (default: 500)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ITER="${ITER:-500}"
OUT="$ROOT/8_android/build/verify"
mkdir -p "$OUT"

# ---- NDK --------------------------------------------------------------
NDK="${NDK:-}"
if [[ -z "$NDK" ]]; then
    NDK="$(sed -n 's/^ndk.dir=//p' 8_android/local.properties 2>/dev/null || true)"
fi
if [[ -z "$NDK" || ! -d "$NDK" ]]; then
    echo "NDK not found. Install it (8_android/tools/install_ndk.sh) or set NDK=." >&2
    exit 2
fi
CLANG_BIN="$(ls -d "$NDK"/toolchains/llvm/prebuilt/*/bin 2>/dev/null | head -1)"
CLANG="$CLANG_BIN/aarch64-linux-android21-clang"
[[ -x "$CLANG" ]] || { echo "clang not found under $CLANG_BIN" >&2; exit 2; }

# ---- host reference ---------------------------------------------------
echo "== host reference =="
cc -O3 -ffp-contract=off -march=native \
   -I6_mrz_ocr/include -I6_mrz_ocr/src \
   6_mrz_ocr/tools/bench_cnn_kernel.c 6_mrz_ocr/src/cnn.c \
   -o "$OUT/bench_cnn_kernel_host" -lm
HOST_FNV="$("$OUT/bench_cnn_kernel_host" "$ITER" | awk '/probs_fnv/{print $3}')"
echo "host probs_fnv = $HOST_FNV"

# ---- device build -----------------------------------------------------
echo "== device build (arm64, -O3 -ffp-contract=off) =="
"$CLANG" -O3 -ffp-contract=off \
    -I6_mrz_ocr/include -I6_mrz_ocr/src \
    6_mrz_ocr/tools/bench_cnn_kernel.c 6_mrz_ocr/src/cnn.c \
    -o "$OUT/bench_cnn_kernel_arm64" -lm

# ---- adb --------------------------------------------------------------
if ! command -v adb >/dev/null 2>&1; then
    echo "adb not found; binary is ready at $OUT/bench_cnn_kernel_arm64" >&2
    exit 3
fi
DEVICE="${DEVICE:-}"
if [[ -z "$DEVICE" ]]; then
    DEVICE="$(adb devices | awk 'NR>1 && $2=="device"{print $1; exit}')"
fi
[[ -n "$DEVICE" ]] || { echo "no adb device"; exit 3; }
echo "device = $DEVICE"

adb -s "$DEVICE" push "$OUT/bench_cnn_kernel_arm64" /data/local/tmp/ >/dev/null
DEV_FNV="$(adb -s "$DEVICE" shell "/data/local/tmp/bench_cnn_kernel_arm64 $ITER" | awk '/probs_fnv/{print $3}')"
echo "device probs_fnv = $DEV_FNV"

if [[ "$HOST_FNV" == "$DEV_FNV" ]]; then
    echo "PROBS PARITY: OK (bit-identical)"
else
    echo "PROBS PARITY: MISMATCH ($HOST_FNV vs $DEV_FNV)" >&2
    echo "  -> the -O3 device build is NOT byte-identical to the host." >&2
    echo "  -> Either revert to -O2 in 8_android/app/src/main/cpp/CMakeLists.txt" >&2
    echo "     or move the gate to a 0.3pp tolerance after a full eval run." >&2
fi

# ---- latency ----------------------------------------------------------
# The product-path latency is measured by the app itself; the native
# engine's own timing lines are in logcat under the "MrzOcr" tag.
echo "== device latency (logcat) =="
adb -s "$DEVICE" logcat -c
echo "  run the photo benchmark on the device, then:"
echo "    adb logcat -d -s MrzOcr:I | grep 'total=' | awk -F'total=' '{print \$2}' | cut -d'ms' -f1 | sort -n"
echo "  -> check the p95 against the 100 ms budget"
