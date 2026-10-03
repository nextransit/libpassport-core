// Same logic as the Android face_pipeline.cpp, in a separate file so the
// Objective-C++ wrapper (.mm) can include it.
#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace faceapp_ios {
struct DetectionOut { int32_t x, y, w, h; float score; };

class Pipeline {
public:
    explicit Pipeline(const std::string& model_dir);
    ~Pipeline();

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    int DetectAndMark(const uint8_t* nv21, int32_t w, int32_t h,
                      std::vector<DetectionOut>* faces,
                      std::vector<std::array<float, 10>>* points);
    int64_t RegisterFromNv21(const uint8_t* nv21, int32_t w, int32_t h,
                             const std::array<float, 10>& pts);
    int64_t QueryFromNv21(const uint8_t* nv21, int32_t w, int32_t h,
                          const std::array<float, 10>& pts, float* sim);
    size_t DatabaseSize();
    int DeleteByIndex(int64_t index);
    int SaveDatabase(const std::string& path);
    int LoadDatabase(const std::string& path);
private:
    struct Impl; Impl* impl_;
};
} // namespace faceapp_ios
