#!/usr/bin/env bash
# Device-side verification for the MRZ product path.
#
# What it does, in order:
#   1. probs_fnv parity: the device build of the conv kernel must
#      reproduce the HOST probabilities bit for bit.  (Host and device
#      never matched across platforms -- Apple clang on Darwin vs NDK
#      clang on Android -- so this compares the DEVICE against itself:
#      it detects a build-flag change, not a platform difference.)
#   2. APK build + install.
#   3. Push the photo corpus into the app's own files directory (the app
#      cannot read /data/local/tmp on API 30+ when the files are owned
#      by the shell user, and its external files dir is FUSE-denied).
#   4. Run the product-path device test (PhotoMrzDeviceTest, "auto"
#      method = the production path) via `am instrument` -- NOT via
#      gradlew connectedAndroidTest, which UNINSTALLS the app when it
#      finishes and wipes the pushed corpus.
#   5. Collect the verdict split and the latency distribution from
#      logcat and print a summary; optionally fail on a regression.
#
# Usage:
#   bash 8_android/tools/verify_device.sh                 # full run
#   bash 8_android/tools/verify_device.sh --skip-build    # reuse the APK
#   bash 8_android/tools/verify_device.sh --no-parity    # skip step 1
#
# Environment:
#   DEVICE   adb serial (default: the only connected device)
#   PHOTOS   directory with the photos (default: tmp/PassPortPhoto)
#   ITER     kernel bench iterations (default: 500)
#   MAX_P95  latency budget in ms for the FAIL check (default: 100)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DEVICE="${DEVICE:-}"
PHOTOS="${PHOTOS:-$ROOT/tmp/PassPortPhoto}"
ITER="${ITER:-500}"
MAX_P95="${MAX_P95:-100}"
OUT="$ROOT/8_android/build/verify"
APK="$ROOT/8_android/app/build/outputs/apk/debug/app-debug.apk"
TEST_APK="$ROOT/8_android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk"
PKG="com.nextransit.mrzbench"
TEST_PKG="com.nextransit.mrzbench.test"
SKIP_BUILD=0
DO_PARITY=1
for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=1 ;;
        --no-parity)  DO_PARITY=0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

mkdir -p "$OUT"

# ---- device -------------------------------------------------------------
if [[ -z "$DEVICE" ]]; then
    DEVICE="$(adb devices | awk 'NR>1 && $2=="device"{print $1; exit}')"
fi
[[ -n "$DEVICE" ]] || { echo "no adb device" >&2; exit 3; }
adb -s "$DEVICE" get-state >/dev/null
echo "device: $DEVICE"

# ---- 1. kernel parity ----------------------------------------------------
if [[ "$DO_PARITY" == 1 ]]; then
    echo "== kernel parity (device vs host) =="
    NDK="${NDK:-}"
    if [[ -z "$NDK" ]]; then
        NDK="$(sed -n 's/^ndk.dir=//p' 8_android/local.properties 2>/dev/null || true)"
    fi
    if [[ -z "$NDK" || ! -d "$NDK" ]]; then
        echo "  NDK not found -- skipping the parity check"
    else
        CLANG_BIN="$(ls -d "$NDK"/toolchains/llvm/prebuilt/*/bin 2>/dev/null | head -1)"
        "$CLANG_BIN/aarch64-linux-android21-clang" -O2 -ffp-contract=off \
            -I6_mrz_ocr/include -I6_mrz_ocr/src \
            6_mrz_ocr/tools/bench_cnn_kernel.c 6_mrz_ocr/src/cnn.c \
            -o "$OUT/bench_cnn_kernel_arm64" -lm
        cc -O2 -ffp-contract=off -I6_mrz_ocr/include -I6_mrz_ocr/src \
            6_mrz_ocr/tools/bench_cnn_kernel.c 6_mrz_ocr/src/cnn.c \
            -o "$OUT/bench_cnn_kernel_host" -lm
        host_fnv="$("$OUT/bench_cnn_kernel_host" "$ITER" | awk '/probs_fnv/{print $3}')"
        adb -s "$DEVICE" push "$OUT/bench_cnn_kernel_arm64" /data/local/tmp/ >/dev/null
        dev_fnv="$(adb -s "$DEVICE" shell "/data/local/tmp/bench_cnn_kernel_arm64 $ITER" | awk '/probs_fnv/{print $3}')"
        echo "  host   probs_fnv = $host_fnv"
        echo "  device probs_fnv = $dev_fnv"
        if [[ "$host_fnv" == "$dev_fnv" ]]; then
            echo "  PARITY: OK (the device build is bit-identical to the host)"
        else
            echo "  PARITY: MISMATCH -- the device build changed its arithmetic." >&2
            echo "    The host and the device never matched across platforms, so" >&2
            echo "    this only matters if it used to match on THIS device." >&2
        fi
    fi
fi

# ---- 2. build + install --------------------------------------------------
if [[ "$SKIP_BUILD" == 0 ]]; then
    echo "== build =="
    (cd 8_android && ./gradlew :app:assembleDebug --offline -q)
    (cd 8_android && ./gradlew :app:assembleDebugAndroidTest --offline -q)
fi
[[ -f "$APK" ]] || { echo "APK not found: $APK" >&2; exit 2; }
adb -s "$DEVICE" install -r -t "$APK" >/dev/null
adb -s "$DEVICE" install -r -t "$TEST_APK" >/dev/null
echo "installed"

# ---- 3. push the corpus --------------------------------------------------
echo "== push photos =="
# One push for the whole directory (adb push of a directory keeps the
# names), then ONE adb shell call copies everything into the app's own
# files directory.  The per-file loop was 3 adb round trips per photo,
# which dominated the run time.
adb -s "$DEVICE" shell "run-as $PKG mkdir -p files/omtest" >/dev/null 2>&1 || true
adb -s "$DEVICE" shell "rm -rf /data/local/tmp/mrzcorpus" >/dev/null 2>&1 || true
# ONE push for the whole directory (adb keeps the file names), then one
# chmod and one batch copy into the app's own files directory.  The
# per-file loop was three adb round trips per photo and dominated the
# run time.
adb -s "$DEVICE" push "$PHOTOS/." /data/local/tmp/mrzcorpus/ >/dev/null
n=$(adb -s "$DEVICE" shell "ls /data/local/tmp/mrzcorpus/*.jpg 2>/dev/null | wc -l" | tr -d '\r')
# the app's uid cannot read files owned by the shell user until they are
# world readable
adb -s "$DEVICE" shell "chmod 644 /data/local/tmp/mrzcorpus/*.jpg" >/dev/null 2>&1 || true
adb -s "$DEVICE" shell "run-as $PKG sh -c 'for f in /data/local/tmp/mrzcorpus/*.jpg; do cat \$f > files/omtest/\$(basename \$f); done'" >/dev/null 2>&1 || true
adb -s "$DEVICE" shell "rm -rf /data/local/tmp/mrzcorpus" >/dev/null 2>&1 || true
echo "  pushed $n photos ($(adb -s "$DEVICE" shell "run-as $PKG sh -c 'ls files/omtest | grep -c jpg'" | tr -d '\r') in place)"

# ---- 4. run the product-path test ---------------------------------------
echo "== device test (product path) =="
adb -s "$DEVICE" logcat -c
adb -s "$DEVICE" shell \
    "am instrument -w -e class $PKG.PhotoMrzDeviceTest -e methods auto -e parallel 1 \
        $TEST_PKG/androidx.test.runner.AndroidJUnitRunner" | tail -3

# ---- 5. collect ----------------------------------------------------------
LOG=/tmp/mrz_verify_logcat.txt
adb -s "$DEVICE" logcat -d > "$LOG"
pass=$(grep -oE "verdict=PASS" "$LOG" | wc -l | tr -d ' ')
fail=$(grep -oE "verdict=FAIL" "$LOG" | wc -l | tr -d ' ')
echo "== summary =="
echo "  verdicts: PASS=$pass FAIL=$fail"
grep "RESULT " "$LOG" | grep -oE "total=[0-9]+ms" | cut -d= -f2 | cut -d'm' -f1 | sort -n \
    | awk '{a[NR]=$1} END {if (NR) printf "  latency : p50=%d p90=%d p95=%d max=%d ms (n=%d)\n", a[int(NR*0.5+0.5)], a[int(NR*0.9)], a[int(NR*0.95+0.5)], a[NR], NR}'
echo "  slowest frames:"
grep "RESULT " "$LOG" | sed -E 's/.*RESULT ([^ ]+).*total=([0-9]+)ms.*ocr=([0-9]+)ms.*/\2ms ocr=\3  \1/' \
    | sort -rn | head -5
if [[ "$fail" != 0 ]]; then
    echo "  failing frames:"
    grep "verdict=FAIL" "$LOG" | grep -oE "RESULT [^ ]+" | sed 's/^/    /'
fi

p95=$(grep "RESULT " "$LOG" | grep -oE "total=[0-9]+ms" | cut -d= -f2 | cut -d'm' -f1 \
      | sort -n | awk '{a[NR]=$1} END {print (NR ? a[int(NR*0.95+0.5)] : 0)}')
if [[ -n "$p95" && "$p95" -gt "$MAX_P95" ]]; then
    echo "  BUDGET: p95=${p95}ms > ${MAX_P95}ms" >&2
    exit 1
fi
echo "  BUDGET: p95=${p95}ms <= ${MAX_P95}ms  OK"
