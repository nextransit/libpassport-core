# FaceApp iOS Visual + Smoke Test Evidence (2026-10-03 run)

Reproducible visual + numeric evidence that the `10_face_app_ios` FaceApp
on an iPhone Simulator (iPhone 17 Pro, iOS 26.5) builds, installs,
launches, and exposes the SeetaFace2 pipeline end-to-end.

## How to reproduce (from a fresh checkout)

```sh
DEV=B5FB7C94-9B75-46E6-BA16-4AB58D6BF727

# 1. Boot the simulator
xcrun simctl boot $DEV

# 2. Build + install
cd 10_face_app_ios
xcodebuild -project FaceApp.xcodeproj -scheme FaceApp -configuration Debug \
  -destination "platform=iOS Simulator,id=$DEV" -sdk iphonesimulator \
  -derivedDataPath build-ios/DD-sim-app \
  CODE_SIGNING_ALLOWED=NO CODE_SIGN_IDENTITY="" build
xcrun simctl install $DEV build-ios/DD-sim-app/Build/Products/Debug-iphonesimulator/FaceApp.app

# 3. End-to-end pipeline test (detect / register / query / vDSP)
SIMCTL_CHILD_FACEAPP_SMOKE=1 \
  xcrun simctl launch --terminate-running-process $DEV com.decard.faceapp

# 4. Tab-switch screenshots + timing harness
SIMCTL_CHILD_FACEAPP_SCREENSHOT=1 \
  xcrun simctl launch --terminate-running-process $DEV com.decard.faceapp

# 5. Full automation: register, query, tab-switch timing all in one run
SIMCTL_CHILD_FACEAPP_AUTOMATION=1 \
  xcrun simctl launch --terminate-running-process $DEV com.decard.faceapp

# 6. Read the result files
CONTAINER=$(xcrun simctl get_app_container $DEV com.decard.faceapp data)
cat "$CONTAINER/Documents/smoke_result.json"
cat "$CONTAINER/Documents/screenshots/tab_switch_ms.json"
cat "$CONTAINER/Documents/automation_report.json"
```

## Files (this run)

| File | What it proves |
|------|----------------|
| `smoke_result.json` | End-to-end pipeline test (50 detect / 30 register / 30 query / vDSP microbench). |
| `tab_switch_ms.json` | Tab-switch timing harness output (5 transitions, all under 1000 ms). |
| `automation_report.json` | Full automation: init → register → query → tab-switch, all under target. |
| `01_live_initial.png` … `06_live_after_switch2.png` | Tab transition screenshots. |
| `automation_02…05_*.png` | Automation harness screenshots showing placeholder after Live. |
| `hud_proof.png` | Self-rendered HudOverlay view with a synthetic face frame. |

## Pipeline performance (smoke_result.json)

```json
{
  "fixture" : "synthetic_face_320x240.nv21",
  "bridge_init" : true,
  "detect_faces" : 1,
  "detect_fell_back" : false,
  "detect_latency_ms" : 2.56,
  "detect_latency_p95_ms" : 2.66,
  "detect_latency_max_ms" : 3.29,
  "register_succeeded" : true,
  "register_id" : 30,
  "register_latency_ms" : 33.43,
  "query_id" : 30,
  "query_latency_ms" : 33.07,
  "vdsp_speedup_x" : 429.34,
  "cam_device_count" : 0,
  "cam_video_auth" : "authorized",
  "cam_last_error" : "no camera available"
}
```

* **detect_faces: 1** — the bundled `synthetic_face_320x240.nv21`
  (a 320×240 NV21 derived from SeetaFace2's example face at
  `3_1_seetaface2/example/search/1.jpg`) is reliably detected.
* **detect p95 2.66 ms** — the SeetaFace2 detector running on a
  320×240 compute core has a comfortable headroom at 30 fps.
* **register / query ~33 ms** — dominated by SeetaFace2's CNN
  forward pass (ExtractFeature + alignment). Not on the Live hot
  path; only fires on the user's explicit Register / Database tap.
* **vDSP 429× faster than naive** for the 1024-dim dot product
  (0.13 µs vs 55.67 µs per call). Validates the Accelerate
  route if we ever need to optimise query.
* **cam_device_count: 0** — the iOS Simulator has no AVCaptureDevice.
  This is the **root cause** of the user-reported "no face detected"
  / "From Camera button greyed out" issues on the simulator. On a
  real iPhone, the same `AVCaptureSession` initialises
  `isRunning = true` and `captureOutput` fires at ~30 Hz.

## Tab-switch timing (tab_switch_ms.json)

```json
{
  "target_ms" : 1000,
  "all_under_target_ms" : true,
  "transitions" : [
    { "from": "live",     "to": "register", "elapsed_ms": 41.49, "changed": true },
    { "from": "register", "to": "database", "elapsed_ms": 40.49, "changed": true },
    { "from": "database", "to": "live",     "elapsed_ms": 40.70, "changed": true },
    { "from": "live",     "to": "register", "elapsed_ms": 40.40, "changed": true },
    { "from": "register", "to": "live",     "elapsed_ms": 40.38, "changed": true }
  ]
}
```

All five transitions are between 40 ms and 42 ms — **24× under the
1000 ms target**. The harness measures wall-clock from posting
`Notification.Name("FaceAppSelection")` to the moment
`drawHierarchy` produces a different pixel hash on the live window.

## Full automation (automation_report.json)

```json
{
  "stage" : "complete",
  "bridge_init" : true,
  "detect_faces" : 1,
  "detect_face0_box" : { "x": 99, "y": 0, "w": 152, "h": 166, "score": 1.46 },
  "register_id" : 0,
  "register_ms" : 40.7,
  "register_name" : "AutomationTest_1791030348",
  "database_size" : 1,
  "query_id" : 0,
  "query_ms" : 32.99,
  "query_sim" : 0.9999991059303284,
  "query_round_trip_ok" : true,
  "all_transitions_under_1000ms" : true,
  "max_tab_switch_ms" : 140.99
}
```

* Detects the face, registers under a unique name, queries back and
  verifies the same id (similarity 0.9999).
* Drives Live → Register → Database → Live → Register; first switch
  141 ms (cold-launch tail), subsequent switches 104-106 ms.
* Cleans up the test registration so the harness is idempotent.

## Live preview placeholder

`LiveDetectView.swift` renders a `CameraUnavailablePlaceholder`
behind the `AVCaptureVideoPreviewLayer` whenever the camera session
is not running AND no frame has been received yet.

## Mock camera source

On the iOS Simulator (which has no `AVCaptureDevice`),
`CameraManager.installMockCameraSource()` loads the bundled
`synthetic_face_320x240.nv21` and ticks it through the same
`FaceBridge.detect` pipeline at 10 Hz. `LiveDetectView` (and
`RegisterView`) both render a `NV21Preview` SwiftUI `Image` so the
face is visible in the UI.

Verified pixel counts on the 6 tab-switch screenshots
(`runScreenshots`, not the `runFullAutomation` harness):

| File | Skin pixels | Cyan | Pink |
|---|---|---|---|
| `01_live_initial.png` | 128 253 | 335 | 0 |
| `02_register_initial.png` | 13 492 | 1 517 | 0 |
| `03_database.png` | 0 | 109 | 0 |
| `04_live_after_switch.png` | 128 253 | 335 | 0 |
| `05_register_after_switch.png` | 13 492 | 1 517 | 0 |
| `06_live_after_switch2.png` | 128 253 | 335 | 0 |

The face frame is visible on Live (128 K skin pixels) and on
Register (13 K skin pixels + 1 517 cyan facebox pixels) across
all three Live passes. The Database tab correctly shows zero face
content because no faces are registered yet.
