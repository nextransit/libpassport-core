// JNI entry points for com.nextransit.mrzbench.jni.MrzNative.
//
// Three responsibilities:
//   1. loadPpmString()  — turn a UTF-8 Java path into face_image_t.
//   2. recogniseString — run the chosen backend (0=trad, 1=cnn) and
//      return line1/line2/avg-confidence/duration-ms as a JSON string.
//   3. versionString()  — expose the in-tree CNN/TEMPLATE_VERSION so the
//      app can show what is actually linked (no silent fallbacks).
//
// All error paths return a JSON object with {"ok":false,"err":"..."} so
// the Kotlin side never has to deal with thrown JNI exceptions across
// thousands of images.
#include <jni.h>
#include <android/log.h>
#include <string>
#include <cstdio>
#include <cstring>
#include <chrono>

// Include the in-tree header; the mrz_ocr_recognise_cnn declaration
// at the bottom of mrz_ocr.h is NOT inside the extern "C" block, so
// the C++ compiler mangles any reference inside this TU. Redefine
// the symbol via a macro shim that hides the C++-linkage declaration
// and exposes the same name with C linkage instead. This keeps the
// host header unchanged.
#define mrz_ocr_recognise_cnn mrz_ocr_recognise_cnn_header_decl
#include "mrz_ocr.h"
#undef mrz_ocr_recognise_cnn
extern "C" {
mrz_ocr_status_t mrz_ocr_recognise_cnn(const face_image_t *img,
                                       mrz_ocr_result_t *out);
}

#define LOG_TAG "MrzNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

extern "C" {

JNIEXPORT jstring JNICALL
Java_com_nextransit_mrzbench_jni_MrzNative_versionString(JNIEnv *env, jobject) {
    // Static, debuggable version string. Keep in sync with
    // 6_mrz_ocr bench_results.json generation script if changed.
    const char *v = "mrzbench/1.0.0 backend=dual (template+cnn) cnn=trained-real";
    return env->NewStringUTF(v);
}

static std::string json_escape(const char *s) {
    std::string out;
    out.reserve(strlen(s) + 8);
    for (const char *p = s; *p; ++p) {
        switch (*p) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if ((unsigned char)*p < 0x20) {
                    char buf[8]; std::snprintf(buf, sizeof(buf), "\\u%04x", *p);
                    out += buf;
                } else {
                    out += *p;
                }
        }
    }
    return out;
}

JNIEXPORT jstring JNICALL
Java_com_nextransit_mrzbench_jni_MrzNative_recogniseString(
        JNIEnv *env, jobject, jstring jpath, jint backend) {
    const char *path = env->GetStringUTFChars(jpath, nullptr);
    if (!path) return env->NewStringUTF("{\"ok\":false,\"err\":\"path-null\"}");

    face_image_t *img = face_image_load_ppm(path);
    env->ReleaseStringUTFChars(jpath, path);
    if (!img) return env->NewStringUTF("{\"ok\":false,\"err\":\"load-ppm\"}");

    mrz_ocr_init();
    mrz_ocr_result_t r;
    std::memset(&r, 0, sizeof(r));
    auto t0 = std::chrono::steady_clock::now();
    mrz_ocr_status_t st = (backend == 1)
            ? mrz_ocr_recognise_cnn(img, &r)
            : mrz_ocr_recognise(img, &r);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    char buf[1024];
    if (st != MRZ_OCR_OK) {
        std::snprintf(buf, sizeof(buf),
                      "{\"ok\":false,\"err\":\"%s\",\"backend\":%d,\"ms\":%.3f}",
                      mrz_ocr_strerror(st), backend, ms);
    } else {
        std::snprintf(buf, sizeof(buf),
            "{\"ok\":true,\"line1\":\"%s\",\"line2\":\"%s\","
            "\"conf1\":%d,\"conf2\":%d,\"ms\":%.3f,"
            "\"band\":[%d,%d,%d,%d],\"backend\":%d,\"err\":0}",
            json_escape(r.line1).c_str(),
            json_escape(r.line2).c_str(),
            r.line1_avg_conf, r.line2_avg_conf, ms,
            r.band_x, r.band_y, r.band_w, r.band_h, backend);
    }
    face_image_free(img);
    return env->NewStringUTF(buf);
}

JNIEXPORT jstring JNICALL
Java_com_nextransit_mrzbench_jni_MrzNative_loadPpmInfo(
        JNIEnv *env, jobject, jstring jpath) {
    const char *path = env->GetStringUTFChars(jpath, nullptr);
    if (!path) return env->NewStringUTF("{\"ok\":false}");
    face_image_t *img = face_image_load_ppm(path);
    env->ReleaseStringUTFChars(jpath, path);
    if (!img) return env->NewStringUTF("{\"ok\":false}");
    char buf[128];
    std::snprintf(buf, sizeof(buf), "{\"ok\":true,\"w\":%d,\"h\":%d}", img->width, img->height);
    face_image_free(img);
    return env->NewStringUTF(buf);
}

} // extern "C"
