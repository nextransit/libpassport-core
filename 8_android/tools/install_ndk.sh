#!/usr/bin/env bash
# install_ndk.sh — Download Android NDK r26b into ~/.ndk/ if missing.
# Used because the canonical /Volumes/.../Android/ndk directory on this
# host had an incomplete copy (no source.properties) and NDK r30-beta1's
# ld.lld hangs on macOS 26.5 / Apple Silicon during the build's
# try-compile phase.
set -euo pipefail
VERSION="${1:-26.1.10909125}"
DEST="${HOME}/.ndk/${VERSION}"
URL="https://dl.google.com/android/repository/android-ndk-r26b-darwin.zip"

if [ -f "${DEST}/source.properties" ]; then
    echo "NDK already installed at ${DEST}"
    exit 0
fi

TMP=$(mktemp -d -t ndk-r26b)
trap "rm -rf '${TMP}'" EXIT
echo ">> downloading ${URL}"
curl -L --connect-timeout 30 --max-time 1800 -o "${TMP}/ndk.zip" "${URL}"
echo ">> unzipping into ${DEST}"
mkdir -p "${DEST}"
python3 -c "
import zipfile, sys
zipfile.ZipFile('${TMP}/ndk.zip').extractall('${DEST}')
"
mv "${DEST}/android-ndk-r26b"/* "${DEST}/" 2>/dev/null || true
[ -d "${DEST}/android-ndk-r26b" ] && mv "${DEST}/android-ndk-r26b" "${DEST}-bak"
echo ">> done: ${DEST}"
cat "${DEST}/source.properties"
