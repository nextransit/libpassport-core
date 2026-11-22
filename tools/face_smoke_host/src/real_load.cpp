// Real model-load test: construct the actual vendored classes with the
// downloaded .dat files and inspect them. If any class throws or aborts,
// the consumer is never silent; the exit code is non-zero.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>

#include <seeta/FaceDetector.h>
#include <seeta/FaceLandmarker.h>
#include <seeta/FaceRecognizer.h>
#include <seeta/CStruct.h>

static int try_load(const char* model_dir) {
    const std::string fd = std::string(model_dir) + "/fd_2_00.dat";
    const std::string fl = std::string(model_dir) + "/pd_2_00_pts5.dat";
    const std::string fr = std::string(model_dir) + "/fr_2_10.dat";

    seeta::ModelSetting FD_cfg(fd.c_str(), seeta::ModelSetting::CPU, 0);
    seeta::ModelSetting FL_cfg(fl.c_str(), seeta::ModelSetting::CPU, 0);
    seeta::ModelSetting FR_cfg(fr.c_str(), seeta::ModelSetting::CPU, 0);

    int failures = 0;
    try {
        seeta::FaceDetector FD(FD_cfg);
        FD.set(seeta::FaceDetector::PROPERTY_VIDEO_STABLE, 1);
        FD.set(seeta::FaceDetector::PROPERTY_MIN_FACE_SIZE, 80);
        std::printf("[real] OK FaceDetector loaded %s\n", fd.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[real] FAIL FaceDetector: %s\n", e.what());
        ++failures;
    } catch (...) {
        std::fprintf(stderr, "[real] FAIL FaceDetector (unknown)\n");
        ++failures;
    }

    try {
        seeta::FaceLandmarker FL(FL_cfg);
        std::printf("[real] OK FaceLandmarker loaded (%d pts)\n", FL.number());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[real] FAIL FaceLandmarker: %s\n", e.what());
        ++failures;
    } catch (...) {
        std::fprintf(stderr, "[real] FAIL FaceLandmarker (unknown)\n");
        ++failures;
    }

    try {
        seeta::FaceRecognizer FR(FR_cfg);
        std::printf("[real] OK FaceRecognizer loaded (crop=%dx%d ch=%d feat=%d)\n",
                    FR.GetCropFaceWidth(), FR.GetCropFaceHeight(),
                    FR.GetCropFaceChannels(), FR.GetExtractFeatureSize());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[real] FAIL FaceRecognizer: %s\n", e.what());
        ++failures;
    } catch (...) {
        std::fprintf(stderr, "[real] FAIL FaceRecognizer (unknown)\n");
        ++failures;
    }

    return failures;
}

int main(int argc, char** argv) {
    const char* dir = std::getenv("SEETA_FACE_MODELS_DIR");
    if (argc >= 2) dir = argv[1];
    if (!dir || !*dir) {
        std::fprintf(stderr, "[real] SEETA_FACE_MODELS_DIR not set\n");
        return 2;
    }
    return try_load(dir);
}
