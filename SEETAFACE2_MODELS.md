# SeetaFace2 Model Files

The vendored `3_1_seetaface2` source tree in this repository does **not**
bundle its model weights — they are distributed separately. We download
them on first checkout and store them under `assets/seetaface2_models/`.

The whole `assets/` directory is git-ignored (see `.gitignore` line
`assets/`), so these files never enter version control.

## Files

| File | Bytes | SHA-256 | Source |
|------|------:|---------|--------|
| `fd_2_00.dat` | 1738167 | `9141d60684317a3b5e55d459e63730fa966a4db9413d354f3bec995aaf34ea43` | Dropbox: https://www.dropbox.com/s/cemt9fl48t5igfh/fd_2_00.dat?dl=1 (vendored README) |
| `fr_2_10.dat` | 122288046 | `d66f0a34a1ab568c5a29dc24d06444f8db5c453996ba921734be08683c3c3596` | Dropbox: https://www.dropbox.com/s/6aslqcokpljha5j/fr_2_10.dat?dl=1 (vendored README) |
| `pd_2_00_pts5.dat` | 415572 | `724fa151e488e10417498a3cd5e6bbed5d6f31da9ecfb8de8f15e9c252b3d3bc` | Dropbox: https://www.dropbox.com/s/noy8tien1gmw165/pd_2_00_pts5.dat?dl=1 (vendored README) |
| `pd_2_00_pts81.dat` | 2984980 | `ec30d8312c354c5f4e99b7f28ce2e682766f441049983c43d80c89d57d241b31` | Dropbox: https://www.dropbox.com/s/v41lmclaxpwow1d/pd_2_00_pts81.dat?dl=1 (vendored README) |

## Verification

We probe each file with the vendored C++ classes:

```
$ SEETA_FACE_MODELS_DIR=assets/seetaface2_models \\
    tools/face_smoke_host/build10/bin/face_real_load
[INFO] FaceDetector: Core size: 640x480
[real] OK FaceDetector loaded assets/seetaface2_models/fd_2_00.dat
[INFO] FaceLandmarker: Number: 5
[real] OK FaceLandmarker loaded (5 pts)
[INFO] FaceRecognizer: Feature size: 1024
[real] OK FaceRecognizer loaded (crop=256x256 ch=3 feat=1024)
exit=0
```

## Refreshing

If a file is missing, run `tools/face_smoke_host/fetch_models.sh`
(see that script for the URLs).
