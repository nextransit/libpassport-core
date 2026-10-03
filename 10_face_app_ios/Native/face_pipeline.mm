// Implementation of the iOS face pipeline. Mirrors the Android version
// exactly so behaviour is identical across platforms.
//
// We do NOT depend on OpenCV; NV21 -> RGB conversion is pure C++ using
// BT.601 full-range coefficients.

#include "face_pipeline.h"
#include <seeta/FaceDetector.h>
#include <seeta/FaceLandmarker.h>
#include <seeta/FaceRecognizer.h>
#include <seeta/FaceDatabase.h>
#include <seeta/CStruct.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace faceapp_ios {

namespace {

void NV21ToRGB(const uint8_t* nv21, int w, int h, uint8_t* out_rgb) {
    const int ySize = w * h;
    const uint8_t* Yp = nv21;
    const uint8_t* VUp = nv21 + ySize;
    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            const uint8_t Y = Yp[j * w + i];
            const int vui = (j >> 1) * w + (i & ~1);
            int8_t V = static_cast<int8_t>(VUp[vui + 0]) - 128;
            int8_t U = static_cast<int8_t>(VUp[vui + 1]) - 128;
            int16_t r = Y + ((1471 * V) >> 10);
            int16_t g = Y - ((352 * U + 731 * V) >> 10);
            int16_t b = Y + ((1815 * U) >> 10);
            if (r < 0) r = 0; else if (r > 255) r = 255;
            if (g < 0) g = 0; else if (g > 255) g = 255;
            if (b < 0) b = 0; else if (b > 255) b = 255;
            uint8_t* px = out_rgb + (j * w + i) * 3;
            px[0] = (uint8_t)r;
            px[1] = (uint8_t)g;
            px[2] = (uint8_t)b;
        }
    }
}

std::vector<uint8_t> nv21ToRgbOwned(const uint8_t* nv21, int w, int h,
                                     SeetaImageData* sd) {
    std::vector<uint8_t> rgb((size_t)w * h * 3);
    NV21ToRGB(nv21, w, h, rgb.data());
    sd->data = rgb.data();
    sd->width = w;
    sd->height = h;
    sd->channels = 3;
    return rgb;
}

}  // namespace

class Pipeline::Impl {
public:
    explicit Impl(const std::string& model_dir) {
        std::string fd_path = model_dir + "/fd_2_00.dat";
        std::string fl5_path = model_dir + "/pd_2_00_pts5.dat";
        std::string fr_path = model_dir + "/fr_2_10.dat";

        seeta::ModelSetting::Device device = seeta::ModelSetting::CPU;
        seeta::ModelSetting FD_cfg(fd_path.c_str(), device, 0);
        seeta::ModelSetting FL_cfg(fl5_path.c_str(), device, 0);
        seeta::ModelSetting FR_cfg(fr_path.c_str(), device, 0);

        // Construct the detector with an explicit 320×240 compute
        // core instead of the default 640×480. The detector
        // internally rescales the input into this smaller core,
        // which roughly quarters the per-frame CPU and roughly
        // doubles the effective framerate. The returned face box
        // is auto-rescaled back to the input image coordinates.
        detector_.reset(new seeta::FaceDetector(FD_cfg, 320, 240));
        landmarker_.reset(new seeta::FaceLandmarker(FL_cfg));
        recognizer_.reset(new seeta::FaceRecognizer(FR_cfg));
        database_.reset(new seeta::FaceDatabase(FR_cfg));

        detector_->set(seeta::FaceDetector::PROPERTY_VIDEO_STABLE, 1);
        // Note: FaceLandmarker (pd_2_00_pts5.dat) does not expose a
        // configurable score threshold in SeetaFace2 v2 — it either
        // returns its five points or none. The 2026-10-01 register
        // fix relies on min_face=40 plus the synthetic-points
        // fallback in RegisterView for frames where the landmarker
        // declines to fire.
    }

    int DetectAndMark(const uint8_t* nv21, int32_t w, int32_t h,
                      std::vector<DetectionOut>* faces,
                      std::vector<std::array<float, 10>>* points) {
        std::lock_guard<std::mutex> lk(mu_);
        faces->clear(); points->clear();
        SeetaImageData sd{};
        auto rgb = nv21ToRgbOwned(nv21, w, h, &sd);

        // The Android sister pipeline uses
        //   min_face = max(40, width/8)
        // so detection accepts anything that's plausibly a face at
        // the phone's typical hold distance. The previous iOS
        // implementation hard-coded 80 px, which silently rejected
        // every face at >40 cm from a 640×480 front-camera frame.
        const int32_t min_face = std::max<int32_t>(40, w / 8);
        detector_->set(seeta::FaceDetector::PROPERTY_MIN_FACE_SIZE, min_face);

        auto raw = detector_->detect(sd);
        const int n = raw.size;
        faces->reserve(n);
        points->reserve(n);
        for (int i = 0; i < n; ++i) {
            const auto& f = raw.data[i];
            DetectionOut d;
            d.x = f.pos.x; d.y = f.pos.y;
            d.w = f.pos.width; d.h = f.pos.height;
            d.score = f.score;
            faces->push_back(d);

            auto pts = landmarker_->mark(sd, f.pos);
            std::array<float, 10> arr{};
            for (int j = 0; j < 5 && j < static_cast<int>(pts.size()); ++j) {
                arr[j * 2 + 0] = pts[j].x;
                arr[j * 2 + 1] = pts[j].y;
            }
            points->push_back(arr);
        }
        return n;
    }

    int64_t RegisterFromNv21(const uint8_t* nv21, int32_t w, int32_t h,
                             const std::array<float, 10>& pts) {
        std::lock_guard<std::mutex> lk(mu_);
        SeetaImageData sd{};
        auto rgb = nv21ToRgbOwned(nv21, w, h, &sd);
        SeetaPointF points[5];
        for (int i = 0; i < 5; ++i) {
            points[i].x = pts[i * 2 + 0];
            points[i].y = pts[i * 2 + 1];
        }
        // SeetaFaceDatabase::Register returns the new face's index
        // (>= 0) on success and -1 on failure. We match the Android
        // behaviour exactly.
        return database_->Register(sd, points);
    }

    int64_t QueryFromNv21(const uint8_t* nv21, int32_t w, int32_t h,
                          const std::array<float, 10>& pts, float* sim) {
        std::lock_guard<std::mutex> lk(mu_);
        SeetaImageData sd{};
        auto rgb = nv21ToRgbOwned(nv21, w, h, &sd);
        SeetaPointF points[5];
        for (int i = 0; i < 5; ++i) {
            points[i].x = pts[i * 2 + 0];
            points[i].y = pts[i * 2 + 1];
        }
        float s = 0.0f;
        int64_t id = database_->Query(sd, points, &s);
        if (sim) *sim = s;
        return id;
    }

    size_t DatabaseSize() {
        std::lock_guard<std::mutex> lk(mu_);
        return database_->Count();
    }

    int DeleteByIndex(int64_t index) {
        std::lock_guard<std::mutex> lk(mu_);
        return database_->Delete(index);
    }

    int SaveDatabase(const std::string& path) {
        std::lock_guard<std::mutex> lk(mu_);
        return database_->Save(path.c_str()) ? 0 : -1;
    }

    int LoadDatabase(const std::string& path) {
        std::lock_guard<std::mutex> lk(mu_);
        return database_->Load(path.c_str()) ? 0 : -1;
    }

private:
    std::unique_ptr<seeta::FaceDetector>   detector_;
    std::unique_ptr<seeta::FaceLandmarker> landmarker_;
    std::unique_ptr<seeta::FaceRecognizer> recognizer_;
    std::unique_ptr<seeta::FaceDatabase>   database_;
    std::mutex mu_;
};

Pipeline::Pipeline(const std::string& model_dir) : impl_(new Impl(model_dir)) {}
Pipeline::~Pipeline() = default;
int Pipeline::DetectAndMark(const uint8_t* nv21, int32_t w, int32_t h,
                             std::vector<DetectionOut>* faces,
                             std::vector<std::array<float, 10>>* points) {
    return impl_->DetectAndMark(nv21, w, h, faces, points);
}
int64_t Pipeline::RegisterFromNv21(const uint8_t* nv21, int32_t w, int32_t h,
                                    const std::array<float, 10>& pts) {
    return impl_->RegisterFromNv21(nv21, w, h, pts);
}
int64_t Pipeline::QueryFromNv21(const uint8_t* nv21, int32_t w, int32_t h,
                                 const std::array<float, 10>& pts, float* sim) {
    return impl_->QueryFromNv21(nv21, w, h, pts, sim);
}
size_t Pipeline::DatabaseSize() { return impl_->DatabaseSize(); }
int Pipeline::DeleteByIndex(int64_t index) { return impl_->DeleteByIndex(index); }
int Pipeline::SaveDatabase(const std::string& path) { return impl_->SaveDatabase(path); }
int Pipeline::LoadDatabase(const std::string& path) { return impl_->LoadDatabase(path); }

}  // namespace faceapp_ios
