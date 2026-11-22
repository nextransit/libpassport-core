// Standalone host smoke driver for the 3_1_seetaface2 components:
//   - FaceDetector
//   - FaceLandmarker
//   - FaceRecognizer (used as a feature extractor; no DB writes)
//
// We deliberately do NOT depend on FaceDatabase here: FaceDatabase is a
// single-threaded wrapper and lives in the FaceRecognizer library, but it
// requires the same .dat model files. The Register/Query API we will need
// for the mobile apps is exercised end-to-end on-device in Phase 2/3.
//
// This program loads each model, prints its fingerprint, and exits 0.
// If any model is missing, it exits 2 with a clear error so the consumer
// is never silently broken.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <seeta/FaceDetector.h>
#include <seeta/FaceLandmarker.h>
#include <seeta/FaceRecognizer.h>
#include <seeta/CStruct.h>

namespace {

struct ModelFile {
    const char* filename;
    const char* label;
};

// Fail loudly if any required .dat is absent. We never silently fall back
// to "skip this test" because that masks problems on real devices.
int require_models(const char* models_dir) {
    if (!models_dir || !*models_dir) {
        std::fprintf(stderr,
            "[smoke] SEETA_FACE_MODELS_DIR is empty. Set it to the directory "
            "that contains fd_2_00.dat, pd_2_00_pts5.dat and fr_2_10.dat.\n");
        return 2;
    }
    const ModelFile required[] = {
        {"fd_2_00.dat",       "FaceDetector"},
        {"pd_2_00_pts5.dat",  "FaceLandmarker (5-pt)"},
        {"fr_2_10.dat",       "FaceRecognizer"},
    };
    int missing = 0;
    for (const auto& m : required) {
        std::string p = std::string(models_dir) + "/" + m.filename;
        FILE* fp = std::fopen(p.c_str(), "rb");
        if (!fp) {
            std::fprintf(stderr, "[smoke] MISSING model: %s (%s)\n",
                         p.c_str(), m.label);
            ++missing;
            continue;
        }
        std::fclose(fp);
    }
    if (missing) {
        std::fprintf(stderr,
            "[smoke] %d model file(s) missing under %s. See "
            "3_1_seetaface2/README.md for the original Dropbox links.\n",
            missing, models_dir);
        return 2;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const char* models_dir = std::getenv("SEETA_FACE_MODELS_DIR");
    if (argc >= 2) models_dir = argv[1];
    if (int rc = require_models(models_dir); rc != 0) return rc;

    seeta::ModelSetting::Device device = seeta::ModelSetting::CPU;
    int id = 0;
    const std::string fd_path  = std::string(models_dir) + "/fd_2_00.dat";
    const std::string pd_path  = std::string(models_dir) + "/pd_2_00_pts5.dat";
    const std::string fr_path  = std::string(models_dir) + "/fr_2_10.dat";

    seeta::ModelSetting FD_cfg(fd_path.c_str(), device, id);
    seeta::ModelSetting FL_cfg(pd_path.c_str(), device, id);
    seeta::ModelSetting FR_cfg(fr_path.c_str(), device, id);

    seeta::FaceDetector     FD(FD_cfg);
    seeta::FaceLandmarker   FL(FL_cfg);
    seeta::FaceRecognizer   FR(FR_cfg);

    FD.set(seeta::FaceDetector::PROPERTY_VIDEO_STABLE, 1);
    FD.set(seeta::FaceDetector::PROPERTY_MIN_FACE_SIZE, 80);

    std::printf("[smoke] OK FaceDetector loaded (%s)\n", fd_path.c_str());
    std::printf("[smoke] OK FaceLandmarker loaded (%d points)\n", FL.number());
    std::printf("[smoke] OK FaceRecognizer loaded (crop=%dx%d ch=%d feat=%d)\n",
                FR.GetCropFaceWidth(), FR.GetCropFaceHeight(),
                FR.GetCropFaceChannels(), FR.GetExtractFeatureSize());
    return 0;
}
