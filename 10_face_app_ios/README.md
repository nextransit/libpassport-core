# 10_face_app_ios — FaceApp iOS

iOS 16+ SwiftUI app that consumes the vendored SeetaFace2 components
in `../3_1_seetaface2/` (`FaceDetector` / `FaceLandmarker` /
`FaceRecognizer` + `FaceDatabase` for register / query) through a
Swift / Objective-C++ glue layer (`Native/FaceBridge.mm`).

## Build status (as of 2026-09-29)

| Step | Status | Notes |
|------|:-----:|-------|
| Swift / Obj-C++ source | ✅ | hand-written |
| CMakeLists.txt for vendored SeetaFace2 static archives | ✅ | builds via `xcodebuild` |
| **Prebuilt `.a` archives produced** | ✅ | `libSeetaFace{D,Landmarker,Recognizer}.a` + `libSeetaNet.a`, ~1.7 MB total, arm64 iphoneos |
| App target (SwiftUI + Obj-C++) | ❌ | CMake Xcode generator cannot reliably build SwiftUI iOS apps on Xcode 15+ (issues: Swift runtime auto-link, simulator SDK selection, FRAMEWORK_SEARCH_PATHS, SwiftINFO declarative deps). A hand-written `FaceApp.xcodeproj` is the recommended path, but requires a stable pbxproj template (xcodegen, pbxproj python lib, or Xcode UI). |
| App icon (1024×1024 HUD style) | ✅ | generated programmatically |
| Push models | n/a | iOS sandbox — Xcode Devices or in-app download |
| Install via USB | n/a | needs `[需要你插入iPhone USB到Macmini]` |

## Source layout

```
FaceApp/                 @main, FaceModel singleton, PersonEntry struct,
                         and FaceApp-Bridging-Header.h
Views/                   ContentView + LiveDetectView + RegisterView + DatabaseView
Camera/CameraSession     AVFoundation NV21 capture
Data/PersonStore         JSON metadata store
Native/FaceBridge        Obj-C++ facade over C++ Pipeline
Native/face_pipeline     C++ pipeline (mirrors Android impl)
Resources/Assets.xcassets      AppIcon (HUD style, generated programmatically)
Resources/Info.plist.in        Configured at cmake time
tools/build_ios.sh       Runs the prereq build + an Xcode archive
tools/push_models.sh     Pushes .dat files into the app sandbox
```

## What works today (verified on Mac mini `100.82.48.23` over Tailscale)

```sh
# 1) Push source to the Mac mini.
cd tools
sshpass -p "$MACMINI_PASSWORD" rsync -a --delete \
    --exclude '.git' --exclude 'build*' \
    --exclude 'DerivedData' --exclude '*.xcodeproj' \
    ../ "$MACMINI_USER@$MACMINI_IP:~/faceapp-ios/"

# 2) Build the vendored SeetaFace2 static archives on the Mac mini.
sshpass -p "$MACMINI_PASSWORD" ssh "$MACMINI_USER@$MACMINI_IP" bash <<'REMOTE'
set -euo pipefail
cd ~/faceapp-ios
mkdir -p build && cd build
cmake -S .. -B . -G Xcode \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=16.0 \
    -DCMAKE_OSX_ARCHITECTURES=arm64
xcodebuild -project faceapp_ios_prereqs.xcodeproj \
    -scheme ALL_BUILD \
    -configuration Release \
    -sdk iphoneos \
    -derivedDataPath ./DD \
    CODE_SIGNING_ALLOWED=NO CODE_SIGNING_REQUIRED=NO \
    build
ls -la lib/Release/  # libSeetaFace{Detector,Landmarker,Recognizer,Net}.a
REMOTE
```

This produces `lib/Release/libSeetaFaceDetector.a` (107 KB),
`libSeetaFaceLandmarker.a` (84 KB), `libSeetaFaceRecognizer.a` (189 KB),
`libSeetaNet.a` (1.3 MB) — all arm64-iphoneos static archives. Verified
on `2026-09-29` via Tailscale to Mac mini `100.82.48.23`.

## What's blocked

The remaining gap is the **iOS app target** itself: the
SwiftUI+Obj-C++ glue that links those `.a` archives into a `.app`
bundle. CMake's Xcode generator cannot reliably build a SwiftUI iOS
app on Xcode 15+ — we encountered:

* Swift runtime symbols (`_swift_willThrow`, `swift_willThrowTypeMetadata`)
  not auto-linked when the app target uses vendored C++ sources that
  throw C++ exceptions across the Swift/Obj-C boundary.
* SDKROOT defaulting to `iphoneos` even when `-sdk iphonesimulator`
  is passed via the command line.
* FRAMEWORK_SEARCH_PATHS not resolving to the simulator SDK's
  `System/Library/Frameworks` when system frameworks are referenced
  by name.
* Auto-linked frameworks (`CoreAudioTypes`, `UIUtilities`) silently
  missing because the simulator SDK's `libswift*.tbd` is built for
  `iOS`, not `iOS-simulator`.

A correct iOS app target requires either a hand-written
`FaceApp.xcodeproj` (we have the source code and the `.a` archives
ready) or a stable pbxproj-generating tool (`xcodegen` is the common
choice; `brew install xcodegen` failed on the Mac mini during this
session, presumably due to network restrictions).

## Recommended next step

The cleanest way to ship an iOS build from this scaffold is to open
`/Users/zhouyong/faceapp-ios/` in Xcode 15+, do "File > New > Project >
iOS > App", and add our files:

1. Drop `FaceApp/`, `Views/`, `Camera/`, `Data/`, `Native/`,
   `Resources/` into the new project.
2. Set Build Settings:
   * `IPHONEOS_DEPLOYMENT_TARGET = 16.0`
   * `SWIFT_OBJC_BRIDGING_HEADER = FaceApp/FaceApp-Bridging-Header.h`
   * `HEADER_SEARCH_PATHS = Native ../3_1_seetaface2/include/...`
   * `LIBRARY_SEARCH_PATHS = $(BUILT_PRODUCTS_DIR)/Release-iphoneos`
3. Link the four `.a` archives (drag from `lib/Release/`).
4. Set `DEVELOPMENT_TEAM = D4LM897A5P` (or your Team ID).
5. Build and run.

That is a 5-minute Xcode UI flow versus the half-day I just spent
fighting CMake's iOS app generator.

## Why CMake-driven (rather than checking in a hand-written pbxproj)

Hand-rolled `project.pbxproj` is brittle across Xcode versions
(14/15/16 differ in expected keys). The `pbxproj` Python library 4.3.3
only supports *modifying* existing projects — it does not have an
`add_target` API. CMake's Xcode generator produces a valid project
every time, but its iOS app-target support lags behind on Xcode 15+
(issues listed above). The current CMakeLists builds the vendored
SeetaFace2 static libraries, which is the part that Xcode's app
generator handles reliably.
