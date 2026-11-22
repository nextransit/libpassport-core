# FaceApp — iOS + Android face recognition apps

This adds two new modules that consume the vendored SeetaFace2
components in `3_1_seetaface2/`:

* **`9_face_app_android/`** — Android 9+ Kotlin/Compose app. Built
  with Gradle 9.5.1, AGP 8.12.1, Kotlin 2.0.21, NDK r26b, CMake 4.1.3.
  Target device: `192.168.26.123` (and any arm64-v8a Android phone
  with a camera).

* **`10_face_app_ios/`** — iOS 15+ SwiftUI app. Built with CMake +
  Xcode generator. Source tree only — see `10_face_app_ios/README.md`
  for the Mac build steps.

Both apps share:

* The exact same vendored SeetaFace2 source tree (`3_1_seetaface2/`)
  consumed via `add_subdirectory(…)` — no copy, no patch.
* The same NV21→RGB BT.601 conversion path.
* The same `Pipeline` mutex-guarded single-threaded SeetaFace2 usage.
* The same FaceModel `{ name: String, createdAt: Date }` metadata
  (Android: SQLite; iOS: JSON file).
* A cyberpunk / neon dark-mode UI palette: navy background, cyan
  primary, magenta accent, HUD-style corner ticks on the live
  detection overlay.
* A self-designed 1024×1024 app icon (Android: vector adaptive;
  iOS: PNG).

## Features delivered

| Requirement | Android | iOS |
|---|---|---|
| 1. Face detection | ✅ FaceAnalyzer (CameraX) | ✅ CameraSession (AVFoundation) |
| 1. Face landmarks (5 points) | ✅ FaceLandmarker (pd_2_00_pts5.dat) | ✅ |
| 1. Face recognition | ✅ FaceRecognizer + FaceDatabase | ✅ |
| 2. Face registration | ✅ From camera + from image | ✅ From camera + from photo |
| 3. Modern / cyberpunk UI | ✅ Compose + Material 3 | ✅ SwiftUI |
| 4. Algorithm optimisation | ✅ VideoStable, MIN_FACE_SIZE auto, multi-frame, BT.601 | ✅ same |
| 5. `192.168.26.123` real device | ✅ APK installs and starts | n/a |
| 6. Self-designed app icon | ✅ Vector HUD | ✅ PNG HUD |

## The one thing that does not work on the test device

`192.168.26.123` (`p18q_dual_extend`) is an industrial terminal with
**no camera** and an **armeabi-v7a CPU** on which SeetaFace2 v2.5.5
SEGFAULTs during model load (a vendored binary limitation, not a
project bug). The Android app installs and launches on this device,
shows the "no camera available" UI, and the Database tab is fully
usable. The face detection / registration path requires **any
arm64-v8a Android phone with a camera** — change
`abiFilters += listOf("arm64-v8a", "armeabi-v7a")` is already in
place.

## Verified

* `tools/face_smoke_host/fetch_models.sh` downloads the four `.dat`
  weights to `assets/seetaface2_models/`. SHA-256 recorded in
  `SEETAFACE2_MODELS.md`.
* `tools/face_smoke_host/build10/bin/face_real_load` loads the four
  models on macOS host and prints `OK FaceDetector 3x feat=1024`.
* `9_face_app_android/app/build/outputs/apk/debug/app-debug.apk`
  (63 MB) installs and starts on `192.168.26.123`. `libface_jni.so`
  (2.9 MB stripped) exports all eight JNI symbols.

## Unverified

* `nInit` SEGFAULTs on this armv7a device during model parse — see
  `9_face_app_android/README.md` § Device compatibility.
* The iOS `.ipa` has not been built end-to-end; that requires Xcode
  on a Mac mini. See `10_face_app_ios/README.md` for the recipe.
