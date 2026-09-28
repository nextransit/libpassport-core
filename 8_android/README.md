# 8_android — MrzBench (on-device batch MRZ OCR)

Android 9 (API 28) app that drives the in-tree **3_face_recognition + 6_mrz_ocr**
C pipelines from a Jetpack UI and compares on-device results against the host
benchmark (`6_mrz_ocr/bench_results.json`).

## What it does

* Loads a corpus (PPM images + `corpus.json`) either bundled (small demo set)
  or pushed to the app-private external dir.
* Runs both **Template** and **CNN** backends over the corpus on-device
  (NDK 26.1 / armeabi-v7a).
* Aggregates per-line accuracy, full-line match, ICAO checksum-valid rate,
  ms p50 / p95 — exactly the same five metrics as `6_mrz_ocr/bench_results.json`.
* Writes per-image JSONL + a Markdown report to
  `/sdcard/Android/data/com.nextransit.mrzbench/files/results/`.
* The **Compare** tab diffs on-device metrics against the host baseline you
  push via `tools/push_baseline.sh`.

## Module boundary

The NDK module (`app/src/main/cpp/`) is a *thin* JNI bridge. It links
`6_mrz_ocr/src/{template,mrz_ocr,mrz_ocr_cnn,cnn,mrz_geom}.c` and the PPM/gray
stack from `3_face_recognition/src/{image,process}.c` *verbatim*. This
guarantees per-image results stay byte-identical with the host
`mrz_ocr_tool / mrz_ocr_cnn_tool` so the on-device benchmark is a faithful
port of the host numbers.

## Quick start

```sh
# 1. Build the debug APK. Requires:
#    - Gradle 9.5 (already cached at ~/.gradle/wrapper/dists/gradle-9.5.1-bin)
#    - AGP 8.12.1 (cached at ~/.gradle/caches/.../com.android.tools.build/gradle)
#    - Android SDK at /Volumes/Extended/macpro/work/Android
#    - Android NDK r26b at $HOME/.ndk/26.1.10909125  (downloaded by this script;
#      see tools/install_ndk.sh)
#    - cmake 4.1.3 in PATH (homebrew). Pointed to via local.properties.
./gradlew assembleDebug

# 2. Push the host benchmark corpus to the device.
./tools/push_corpus.sh ../6_mrz_ocr/data/corpus_eval 192.168.26.123:5555

# 3. (Optional) Push the host baseline for the Compare tab.
./tools/push_baseline.sh ../6_mrz_ocr/bench_results.json 192.168.26.123:5555

# 4. Install.
adb -s 192.168.26.123:5555 install -r app/build/outputs/apk/debug/app-debug.apk
```

In the app, the **Run** tab picks the backend, the **Stream** tab shows live
per-image log lines, and the **Compare** tab reports the on-device vs host
gate (any metric > 0.3pp off → GATE: FAIL).

## Acceptance gates

1. `assembleDebug` produces a debug APK whose `lib/armeabi-v7a/libmrz_jni.so`
   links the host C source unchanged.
2. `push_corpus.sh` pushes all 1380 PPMs + corpus.json under
   `/sdcard/Android/data/com.nextransit.mrzbench/files/corpus/`.
3. On-device benchmark writes
   `<backend>_metrics_<ts>.json`, `<backend>_records_<ts>.jsonl`,
   `<backend>_report_<ts>.md`.
4. Compare tab diff matches the host `bench_results.json` within the
   0.3pp regression gate defined in `6_mrz_ocr/data/baseline.json`.

## Layout

```
8_android/
├── README.md
├── build.gradle.kts
├── settings.gradle.kts
├── gradle.properties
├── gradle/wrapper/
├── gradlew, gradlew.bat
├── local.properties                    (not committed)
├── app/
│   ├── build.gradle.kts
│   ├── proguard-rules.pro
│   └── src/main/
│       ├── AndroidManifest.xml
│       ├── assets/corpus/              (slim demo: corpus.json only)
│       ├── res/                        (themes, layouts, mipmap)
│       └── java/com/nextransit/mrzbench/
│           ├── MainActivity.kt
│           ├── bench/                  (Benchmark, Metrics, IcaoChecksum, StreamBus)
│           ├── data/                   (Corpus, CorpusEntry, Cyber palette)
│           ├── jni/                    (MrzNative — JNI shim)
│           ├── ui/                     (RunFragment, StreamFragment, CompareFragment)
│           └── util/                   (TinyJson)
├── app/src/main/cpp/
│   ├── CMakeLists.txt                  (links host C sources verbatim)
│   ├── mrz_jni.cpp                     (JNI bridge)
│   └── mrz_run.cpp                     (placeholder for future native helpers)
└── tools/
    ├── push_corpus.sh
    ├── push_baseline.sh
    └── install_ndk.sh
```
