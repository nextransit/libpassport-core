#!/usr/bin/env bash
# Push the four SeetaFace2 .dat files to the iOS device's app sandbox.
# Mirrors 9_face_app_android/tools/push_models.sh for the iOS bundle
# id. Requires `ideviceinstaller` from libimobiledevice.
#
# Usage: ./tools/push_models.sh <udid>
# (or use Xcode > Window > Devices and Simulators to find the UDID)
set -euo pipefail
UDID="${1:-}"

if [ -z "$UDID" ]; then
    echo "Usage: $0 <udid>" >&2
    exit 1
fi

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$ROOT/assets/seetaface2_models"
PKG="${IOS_BUNDLE_ID:-com.decard.faceapp}"

if [ ! -d "$SRC" ]; then
    echo "ERROR: $SRC not found. Run tools/face_smoke_host/fetch_models.sh first." >&2
    exit 1
fi

command -v ideviceinstaller >/dev/null 2>&1 || {
    echo "ERROR: ideviceinstaller not installed (brew install libimobiledevice)" >&2
    exit 1
}

echo "==> Looking up $PKG sandbox path on $UDID"
APP_PATH=$(ideviceinstaller -u "$UDID" -l -o list_appid 2>/dev/null | grep "$PKG" | awk '{print $1}')
if [ -z "$APP_PATH" ]; then
    echo "App $PKG is not installed yet. Install it first." >&2
    exit 2
fi

# Use `idevicebackup2` would be the right tool but most setups instead
# push via Xcode's Window > Devices. We print instructions instead.
cat <<INSTRUCTIONS
The model files must live inside the app sandbox at:
    $APP_PATH/Documents/models/

The sandbox is not directly writable from the host. Either:

  1) Open Xcode > Window > Devices and Simulators > $UDID
     > FaceApp > Download Container, edit the container, and
     drag $SRC/*.dat into Documents/models/. Then re-upload.

  2) Use `idevicebackup2` to back up, then edit the backup tree.

  3) Add a debug-only model-fetch path that pulls them from a local
     HTTP server on first launch (recommended for development):
       FaceApp/NetworkModelLoader.swift
INSTRUCTIONS
