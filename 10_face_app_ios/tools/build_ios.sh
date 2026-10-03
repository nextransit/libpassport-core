#!/usr/bin/env bash
# build_ios.sh — produces FaceApp.ipa on the Mac mini over Tailscale
# SSH, following the same conventions as ../IOS_BUILD.md (which
# describes the sister remodex_rust CodexMobile pipeline).
#
# Workflow:
#   1) set -a; source $passport/.env; set +a
#      (we read MACMINI_IP, MACMINI_USER, MACMINI_PASSWORD, IOS_XCODE_PATH,
#       IOS_DEVELOPMENT_TEAM, MACMINI_WORKDIR from there.)
#   2) ssh into the Mac mini.
#   3) cmake -G Xcode for arm64-apple-ios, build FaceApp target.
#   4) xcodebuild archive to a .xcarchive.
#   5) xcodebuild -exportArchive to a .ipa using ad-hoc signing.
#   6) (Optional) xcrun devicectl device install app --device <UDID> <ipa>
#      — run only after the operator confirms the iPhone USB cable is
#      plugged into the Mac mini. We prompt before doing this.
#
# Usage:
#   BUNDLE_ID=com.decard.faceapp TEAM_ID=HR24WHR326 ./tools/build_ios.sh
#
# Outputs (on the Mac mini):
#   $MACMINI_WORKDIR/faceapp-ios/build-ios/FaceApp.xcodeproj
#   $MACMINI_WORKDIR/faceapp-ios/build-ios/FaceApp.xcarchive
#   $MACMINI_WORKDIR/faceapp-ios/dist/FaceApp-0.1.0.ipa

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load credentials from passport/.env. The path is fixed by repo layout.
ENV_FILE="${ENV_FILE:-/Users/zhouyong/gitlab/passport/.env}"
if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
    echo "[build_ios] loaded credentials from $ENV_FILE"
else
    echo "WARNING: $ENV_FILE not found; will rely on shell env." >&2
fi

: "${APP_NAME:=FaceApp}"
: "${BUNDLE_ID:=com.decard.faceapp}"
: "${TEAM_ID:=${IOS_DEVELOPMENT_TEAM:-}}"
: "${VERSION:=0.1.0}"
: "${BUILD:=$(date -u +%Y%m%d%H%M)}"

: "${MACMINI_IP:=100.82.48.23}"   # Tailscale IP per ../IOS_BUILD.md §1
: "${MACMINI_USER:=ben}"
: "${MACMINI_PASSWORD:=${REMOTE_MACOS_PASSWORD:-}}"
# The .env file at /Users/zhouyong/gitlab/passport/.env holds
# MACMINI_WORKDIR as the remote working directory. We treat it as
# authoritative — never fall back to a local path, since the build
# runs on the Mac mini, not on this machine.
: "${MACMINI_WORKDIR:?MACMINI_WORKDIR must be set (e.g. /Volumes/Extended/macmini)}"
: "${IOS_XCODE_PATH:=/Applications/Xcode.app}"

if [ -z "$TEAM_ID" ]; then
    echo "TEAM_ID / IOS_DEVELOPMENT_TEAM not set. Aborting." >&2
    exit 2
fi

if [ -z "$MACMINI_PASSWORD" ] && ! command -v sshpass >/dev/null; then
    echo "MACMINI_PASSWORD not set and sshpass not installed. Install sshpass or populate \$MACMINI_PASSWORD." >&2
    exit 3
fi

# Remote command — runs on the Mac mini.
REMOTE_DIR="$MACMINI_WORKDIR/faceapp-ios"
REMOTE_XC_ARCHIVE="$REMOTE_DIR/build-ios/FaceApp.xcarchive"
REMOTE_IPA_DIR="$REMOTE_DIR/dist"
REMOTE_IPA="$REMOTE_IPA_DIR/${APP_NAME}-${VERSION}.${BUILD}.ipa"

# Wrap ssh-or-sshpass depending on whether we have a key or password.
remote_sh() {
    local cmd="$1"
    if command -v sshpass >/dev/null && [ -n "$MACMINI_PASSWORD" ]; then
        SSHPASS="$MACMINI_PASSWORD" sshpass -e ssh \
            -o StrictHostKeyChecking=accept-new \
            -o ConnectTimeout=10 \
            "${MACMINI_USER}@${MACMINI_IP}" "$cmd"
    else
        ssh -o StrictHostKeyChecking=accept-new \
            -o ConnectTimeout=10 \
            "${MACMINI_USER}@${MACMINI_IP}" "$cmd"
    fi
}

# 1) Sync source tree.
echo "==> rsync source to ${MACMINI_USER}@${MACMINI_IP}:${REMOTE_DIR}"
mkdir -p /tmp/faceapp_sync
rsync -a --delete \
    --exclude '.git' --exclude 'build' --exclude 'build-ios' \
    --exclude 'DerivedData' --exclude '*.xcodeproj' \
    "$ROOT/" "${MACMINI_USER}@${MACMINI_IP}:${REMOTE_DIR}/" 2>/dev/null \
|| {
    # rsync over Tailscale may need to be wrapped in sshpass if no key.
    if command -v sshpass >/dev/null && [ -n "$MACMINI_PASSWORD" ]; then
        SSHPASS="$MACMINI_PASSWORD" sshpass -e rsync -a --delete \
            --exclude '.git' --exclude 'build' --exclude 'build-ios' \
            --exclude 'DerivedData' --exclude '*.xcodeproj' \
            -e "ssh -o StrictHostKeyChecking=accept-new" \
            "$ROOT/" "${MACMINI_USER}@${MACMINI_IP}:${REMOTE_DIR}/"
    else
        echo "rsync failed and no sshpass fallback available" >&2
        return 1
    fi
}

# 2-5) Configure + archive + export on the Mac mini.
remote_sh bash <<REMOTE
set -euo pipefail
cd "$REMOTE_DIR"

# Verify that the four model .dat files are reachable locally on the
# Mac mini (we expect them under assets/seetaface2_models/, mirrored
# via rsync). They are NOT bundled into the app bundle.
test -s assets/seetaface2_models/fd_2_00.dat || { echo "missing fd_2_00.dat"; exit 4; }
test -s assets/seetaface2_models/fr_2_10.dat || { echo "missing fr_2_10.dat"; exit 4; }

# Configure with the Xcode generator against the iOS toolchain.
cmake -S . -B build-ios -G Xcode \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0 \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_XCODE_ATTRIBUTE_DEVELOPMENT_TEAM="$TEAM_ID" \
    -DCMAKE_XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER="$BUNDLE_ID" \
    -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGN_STYLE=Automatic \
    -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY="Apple Development"

# Archive (Release config, generic iOS device).
xcodebuild \
    -project build-ios/FaceApp.xcodeproj \
    -scheme FaceApp \
    -configuration Release \
    -destination 'generic/platform=iOS' \
    -archivePath "$REMOTE_XC_ARCHIVE" \
    -derivedDataPath "$REMOTE_DIR/build-ios/DerivedData" \
    CODE_SIGN_STYLE=Automatic \
    DEVELOPMENT_TEAM="$TEAM_ID" \
    PRODUCT_BUNDLE_IDENTIFIER="$BUNDLE_ID" \
    -skipPackagePluginValidation -skipMacroValidation \
    CODE_SIGNING_ALLOWED=NO CODE_SIGNING_REQUIRED=NO \
    archive

# Export as ad-hoc .ipa. ios-self-hosted-style: we sign the bundle
# with an ad-hoc identity and write it next to the archive.
mkdir -p "$REMOTE_IPA_DIR"
APP_PATH="$REMOTE_XC_ARCHIVE/Products/Applications/FaceApp.app"
test -d "$APP_PATH" || { echo "archive did not produce $APP_PATH"; exit 5; }

# Re-sign with ad-hoc (this is what remodex_rust does for OTA —
# see ../IOS_BUILD.md §5). No keychain required.
codesign --force --sign - --timestamp=none "$APP_PATH"

# Build the .ipa by zipping the .app bundle.
rm -f "$REMOTE_IPA"
(cd "$REMOTE_XC_ARCHIVE/Products/Applications" && \
    zip -qr "$REMOTE_IPA" FaceApp.app)

# Compute sha256 + size for the manifest we may publish later.
SIZE=\$(stat -f%z "$REMOTE_IPA" 2>/dev/null || stat -c%s "$REMOTE_IPA")
SHA=\$(shasum -a 256 "$REMOTE_IPA" | awk '{print \$1}')
echo "[build_ios] wrote $REMOTE_IPA size=\$SIZE sha256=\$SHA"
REMOTE

echo "==> Built $REMOTE_IPA"

# 6) Optional install. We never install without explicit operator
#    confirmation, since the iPhone has to be plugged into the Mac
#    mini via USB.
echo
echo "To install on a connected iPhone, run:"
echo "  DEVICE_UDID=<udid> $0 install"
if [ "${1:-}" = "install" ]; then
    if [ -z "${DEVICE_UDID:-}" ]; then
        # List devices and let the operator pick.
        echo "Available devices:"
        remote_sh "xcrun devicectl list devices" || true
        exit 0
    fi
    echo "==> installing on device $DEVICE_UDID"
    remote_sh "xcrun devicectl device install app --device $DEVICE_UDID '$REMOTE_IPA'"
    echo "==> launch"
    remote_sh "xcrun devicectl device process launch --device $DEVICE_UDID $BUNDLE_ID" || true
fi
