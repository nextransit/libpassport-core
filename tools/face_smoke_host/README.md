# Face smoke host (tooling)

This directory is a **standalone** host-side smoke test for the
vendored SeetaFace2 C++ classes in `../../3_1_seetaface2/`:

- `FaceDetector`     — detect faces in an image
- `FaceLandmarker`   — extract 5 / 81 facial landmarks
- `FaceRecognizer`   — crop + embed a face into a 1024-d feature

We **do not** copy the vendored source. The `CMakeLists.txt` here
simply `add_subdirectory()` the upstream tree and links the produced
static archives into one executable.

## Layout

```
tools/face_smoke_host/
├── CMakeLists.txt        # consumes ../../3_1_seetaface2 via add_subdirectory
├── cmake_uninstall.cmake.in  # copied from upstream so the vendor tree can configure
├── fetch_models.sh       # idempotent downloader for the four .dat weights
├── src/
│   ├── main.cpp          # `face_smoke_host`: load-only check, exits 2 if .dat missing
│   └── real_load.cpp     # `face_real_load`: actually construct FaceDetector / FaceLandmarker / FaceRecognizer
└── build10/              # local out-of-tree build (git-ignored)
    ├── bin/face_smoke_host
    ├── bin/face_real_load
    └── lib/libSeetaNet.a + libSeetaFace{D,Landmarker,Recognizer}.a
```

## Building

```sh
cd tools/face_smoke_host
mkdir -p build && cd build

# The vendored SeetaNet enables -msse2/-mfma whenever PLATFORM is x86/x64/x86_64.
# Those flags are invalid on Apple Silicon, so PLATFORM is pinned to a non-x86
# sentinel inside CMakeLists.txt; nothing to override here.
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_CXX_COMPILER="$(xcrun --find clang++)" \
  -DCMAKE_C_COMPILER="$(xcrun --find clang)" \
  -DCMAKE_OSX_SYSROOT="$(xcrun --show-sdk-path)"
cmake --build . --parallel 4
```

`DEVELOPER_DIR` must point to a real Xcode (not just CommandLineTools) for
`cstdio` / `cstddef` to be visible. The CMake invocation above does that
per-call.

## Downloading the .dat models

The vendored tree does **not** bundle the `.dat` weights. They live on
Dropbox (see `../../3_1_seetaface2/README.md`). The downloader is
idempotent and resumes partial downloads via `wget -c`:

```sh
./fetch_models.sh
# -> places fd_2_00.dat / pd_2_00_pts5.dat / pd_2_00_pts81.dat / fr_2_10.dat
#    under assets/seetaface2_models/ (git-ignored)
```

The exact byte sizes and SHA-256s of the four files are recorded in
`../../SEETAFACE2_MODELS.md`.

## Running the smoke

```sh
SEETA_FACE_MODELS_DIR=../../assets/seetaface2_models \
    ./bin/face_real_load
# Expected:
#   [INFO] FaceDetector: Core size: 640x480
#   [real] OK FaceDetector loaded ../../assets/seetaface2_models/fd_2_00.dat
#   [INFO] FaceLandmarker: Number: 5
#   [real] OK FaceLandmarker loaded (5 pts)
#   [INFO] FaceRecognizer: Feature size: 1024
#   [real] OK FaceRecognizer loaded (crop=256x256 ch=3 feat=1024)
```

If any model is missing, `face_smoke_host` exits **2** with a clear
"missing model" message — it never silently passes.

## Why this exists

The mobile apps (`9_face_app_android/`, `10_face_app_ios/`) will reuse
the exact same vendored tree via `add_subdirectory()`. This host smoke
is the fastest way to catch ABI / flag / link mismatches *before* we
spend a 5-minute Android build or a Mac-mini SSH round trip.
