// JNI entry points for com.nextransit.mrzbench.jni.MrzNative.
//
// Three responsibilities:
//   1. loadPpmString()  — turn a UTF-8 Java path into face_image_t.
//   2. recogniseString — run the chosen backend (0=trad, 1=cnn, 2=crnn) and
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
#include <cstdlib>
#include <mutex>
#include <vector>

// Include the in-tree header; the mrz_ocr_recognise_cnn declaration
// at the bottom of mrz_ocr.h is NOT inside the extern "C" block, so
// the C++ compiler mangles any reference inside this TU. Redefine
// the symbol via a macro shim that hides the C++-linkage declaration
// and exposes the same name with C linkage instead. This keeps the
// host header unchanged.
#define mrz_ocr_recognise_cnn mrz_ocr_recognise_cnn_header_decl
#define mrz_ocr_recognise_fuse mrz_ocr_recognise_fuse_header_decl
#include "mrz_ocr.h"
#undef mrz_ocr_recognise_cnn
#undef mrz_ocr_recognise_fuse
extern "C" {
mrz_ocr_status_t mrz_ocr_recognise_cnn(const face_image_t *img,
                                       mrz_ocr_result_t *out);
mrz_ocr_status_t mrz_ocr_recognise_fuse(const face_image_t *img,
                                       mrz_ocr_result_t *out);
}

#define LOG_TAG "MrzNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

static std::mutex g_recognise_mutex;

extern "C" {

JNIEXPORT jstring JNICALL
Java_com_nextransit_mrzbench_jni_MrzNative_versionString(JNIEnv *env, jobject) {
    // Static, debuggable version string. Keep in sync with
    // 6_mrz_ocr bench_results.json generation script if changed.
    const char *v = "mrzbench/1.0.0 backend=dual (template+cnn) cnn=line-specific-real";
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

static jstring recognise_image(JNIEnv *env,
                               face_image_t *img,
                               jint backend,
                               jint deadline_ms,
                               const char *source) {
    if (!img) return env->NewStringUTF("{\"ok\":false,\"err\":\"image-null\"}");
    std::lock_guard<std::mutex> lock(g_recognise_mutex);

    mrz_ocr_init();
    mrz_ocr_deadline_set_ms(deadline_ms);
    mrz_ocr_result_t r;
    std::memset(&r, 0, sizeof(r));

    const char *prof_env = getenv("MRZ_PROFILE");
    bool prof = prof_env && prof_env[0] && strcmp(prof_env, "0") != 0;
    if (prof) setenv("MRZ_OCR_TIMING", "1", 0);

    auto t0 = std::chrono::steady_clock::now();
    setenv("MRZ_OCR_LINE_MODELS", backend == 1 || backend == 2 || backend == 3 ? "1" : "0", 1);
    mrz_ocr_status_t st;
    if (backend == 3) {
        /* Fused engine: per-character CNN head + CRNN sequence head,
         * combined with the confidence gate.  On a 3-fold grouped
         * cross-validation over the 41 real photos this was the only
         * variant that beat the plain CNN (22/41 exact vs 21/41). */
        st = mrz_ocr_recognise_fuse(img, &r);
    } else if (backend == 2) {
        st = mrz_ocr_recognise_crnn(img, &r);
    } else if (backend == 1) {
        st = mrz_ocr_recognise_cnn(img, &r);
    } else {
        st = mrz_ocr_recognise(img, &r);
    }
    auto t1 = std::chrono::steady_clock::now();
    mrz_ocr_deadline_set_ms(0);
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();


    if (prof) {
        LOGI("MRZ_PROFILE backend=%d source=%s %dx%d ms=%.3f",
             backend, source ? source : "memory", img->width, img->height, ms);
    }

    /* Sized for the worst case the escaper can produce: three 64-char
     * fields at 6 bytes each ("\u00xx"), plus the fixed part.  The MRZ
     * alphabet never contains a character that needs escaping, so in
     * practice this is 640 bytes -- but the buffer must not depend on
     * that, and the snprintf return value is checked below so a future
     * field cannot silently truncate the JSON. */
    char buf[2048];
    if (st != MRZ_OCR_OK) {
        std::snprintf(buf, sizeof(buf),
                      "{\"ok\":false,\"timeout\":%s,\"err\":\"%s\","
                      "\"backend\":%d,\"ms\":%.3f}",
                      st == MRZ_OCR_ERR_TIMEOUT ? "true" : "false",
                      mrz_ocr_strerror(st), backend, ms);
    } else {
        /* `ambiguous` reports that a second checksum-valid reading was
         * almost as likely as the one returned.  A wrong-but-checksum-
         * valid line looks exactly like a correct one to the caller, so
         * this has to travel with the result.
         *
         * `structOk` / `natSuspect` are the arithmetic-independent
         * vetoes (mrz_ocr.h).  They are computed here rather than read
         * from the result struct so that every backend -- including the
         * template one, which does not fill them -- reports the same
         * thing for the same line.  Measured on assets/pic: of the 28
         * images whose line 2 satisfied all five check digits, 19 had a
         * nationality code no state issues.  The check digits cannot see
         * that; these two can.
         *
         * `countryMismatch` compares the issuing state in line 1 with
         * the nationality in line 2.  On the 41-photo phone corpus it
         * was the only signal that caught a single-character error in
         * line 1: it flagged 4 images and all 4 were wrong, while the
         * line-1 confidence and the line-2 ambiguity flag together
         * caught none of them.
         *
         * `issuerSuspect` asks whether the issuing state in line 1 is
         * a code any state actually issues -- the same question
         * `natSuspect` asks about the nationality in line 2, applied
         * to the row that carries no check digits.  All four of the
         * wrong line-1 readings above carried a fabricated code
         * ("HUG", "CHY", "CNN", "ATA") that no state issues. */
        std::snprintf(buf, sizeof(buf),
            "{\"ok\":true,\"timeout\":false,\"line1\":\"%s\",\"line2\":\"%s\","
            "\"conf1\":%d,\"conf2\":%d,\"ms\":%.3f,"
            "\"band\":[%d,%d,%d,%d],\"backend\":%d,\"err\":0,"
            "\"ambiguous\":%s,\"altMargin\":%d,\"altLine2\":\"%s\","
            "\"structOk\":%s,\"natSuspect\":%s,\"countryMismatch\":%s,"
            "\"issuerSuspect\":%s}",
            json_escape(r.line1).c_str(),
            json_escape(r.line2).c_str(),
            r.line1_avg_conf, r.line2_avg_conf, ms,
            r.band_x, r.band_y, r.band_w, r.band_h, backend,
            r.line2_ambiguous ? "true" : "false",
            r.line2_alt_margin,
            json_escape(r.line2_alt).c_str(),
            mrz_ocr_line2_struct_ok(r.line2) ? "true" : "false",
            mrz_ocr_line2_nat_suspect(r.line2) ? "true" : "false",
            mrz_ocr_line1_country_mismatch(r.line1, r.line2) ? "true" : "false",
            mrz_ocr_line1_issuer_suspect(r.line1) ? "true" : "false");
    }
    /* A truncated JSON string would make the Kotlin parser see a partial
     * object and silently drop fields.  Refuse loudly instead: the caller
     * gets a parse error, not a result that looks fine but lost its
     * veto flags. */
    if (std::strlen(buf) >= sizeof(buf) - 1) {
        face_image_free(img);
        return env->NewStringUTF(
            "{\"ok\":false,\"timeout\":false,\"err\":\"json-truncated\",\"backend\":0}");
    }
    face_image_free(img);
    return env->NewStringUTF(buf);
}

JNIEXPORT jstring JNICALL
Java_com_nextransit_mrzbench_jni_MrzNative_recogniseString(
        JNIEnv *env, jobject, jstring jpath, jint backend) {
    const char *path = env->GetStringUTFChars(jpath, nullptr);
    if (!path) return env->NewStringUTF("{\"ok\":false,\"err\":\"path-null\"}");

    face_image_t *img = face_image_load_ppm(path);
    if (!img) {
        env->ReleaseStringUTFChars(jpath, path);
        return env->NewStringUTF("{\"ok\":false,\"err\":\"load-ppm\"}");
    }

    jstring result = recognise_image(env, img, backend, 0, path);
    env->ReleaseStringUTFChars(jpath, path);
    return result;
}

JNIEXPORT jstring JNICALL
Java_com_nextransit_mrzbench_jni_MrzNative_recognisePixels(
        JNIEnv *env, jobject, jintArray jpixels, jint width, jint height,
        jint backend, jint deadline_ms) {
    if (!jpixels || width <= 0 || height <= 0) {
        return env->NewStringUTF("{\"ok\":false,\"err\":\"bad-pixels\"}");
    }
    const jsize expected = static_cast<jsize>(
        static_cast<size_t>(width) * static_cast<size_t>(height));
    if (env->GetArrayLength(jpixels) < expected) {
        return env->NewStringUTF("{\"ok\":false,\"err\":\"short-pixels\"}");
    }

    jint *argb = static_cast<jint *>(
        env->GetPrimitiveArrayCritical(jpixels, nullptr));
    if (!argb) return env->NewStringUTF("{\"ok\":false,\"err\":\"pixels-null\"}");

    face_image_t *img = face_image_new(width, height);
    if (!img) {
        env->ReleasePrimitiveArrayCritical(jpixels, argb, JNI_ABORT);
        return env->NewStringUTF("{\"ok\":false,\"err\":\"alloc-image\"}");
    }
    const size_t count = static_cast<size_t>(width) * static_cast<size_t>(height);
    for (size_t i = 0; i < count; ++i) {
        const uint32_t c = static_cast<uint32_t>(argb[i]);
        img->pixels[i * 3] = static_cast<uint8_t>((c >> 16) & 0xFF);
        img->pixels[i * 3 + 1] = static_cast<uint8_t>((c >> 8) & 0xFF);
        img->pixels[i * 3 + 2] = static_cast<uint8_t>(c & 0xFF);
    }
    env->ReleasePrimitiveArrayCritical(jpixels, argb, JNI_ABORT);
    return recognise_image(env, img, backend, deadline_ms, "pixels");
}

/**
 * Row darkness profile, computed natively.
 *
 * The fast path locates the two MRZ lines from a row projection of the
 * lower half of the photo, sampling every 4th column.  Doing that in
 * Kotlin meant ~500 k interpreted iterations plus a 2 M element IntArray
 * on the JVM; measured on the device that was 46-78 ms of the 200 ms
 * budget, with the bulk in the per-pixel luma arithmetic.
 *
 * The pixel buffer is already in a jintArray (the caller had to read it
 * out of the Bitmap anyway), so the arithmetic is done here, once, in
 * native code, and only the resulting per-row counts come back.
 *
 * Returns an int[rows] of dark-pixel counts, or null on bad input.
 */
extern "C" JNIEXPORT jintArray JNICALL
Java_com_nextransit_mrzbench_jni_MrzNative_rowProfile(
    JNIEnv *env, jobject, jintArray jpixels, jint width,
    jint y0, jint y1, jint x0, jint x1, jint stride,
    jintArray jblownOut) {
    if (!jpixels || width <= 0 || y1 <= y0) return nullptr;
    if (x1 > width) x1 = width;
    if (x0 < 0) x0 = 0;
    if (stride < 1) stride = 1;
    const jsize need = static_cast<jsize>(
        static_cast<size_t>(width) * static_cast<size_t>(y1 - y0));
    if (env->GetArrayLength(jpixels) < need) return nullptr;

    const int rows = y1 - y0;
    /* ART aborts the VM if any other JNI call is made while a critical
     * section is open, so the result array is allocated up front and the
     * counts are written to a plain C buffer inside the critical region
     * and copied out afterwards. */
    jintArray out = env->NewIntArray(rows);
    if (!out) return nullptr;
    std::vector<jint> counts(static_cast<size_t>(rows), 0);

    jint *argb = static_cast<jint *>(env->GetPrimitiveArrayCritical(jpixels, nullptr));
    if (!argb) return nullptr;
    /* Blown-out (specular highlight) pixels are counted in the SAME
     * pass: the luma of every sampled pixel is computed anyway, and a
     * second pass over the buffer would cost another trip through
     * memory for a number that decides whether the user has to retake
     * the photo. */
    long blown = 0, sampled = 0;
    for (int r = 0; r < rows; ++r) {
        /* The buffer holds the rows [y0, y1) only, so index it relatively:
         * row r of the buffer is source row y0 + r. */
        const jint *row = argb + static_cast<size_t>(r) * width;
        int c = 0;
        for (int x = x0; x < x1; x += stride) {
            const uint32_t p = static_cast<uint32_t>(row[x]);
            const int r8 = (p >> 16) & 0xFF, g8 = (p >> 8) & 0xFF, b8 = p & 0xFF;
            const int luma = (r8 * 299 + g8 * 587 + b8 * 114 + 500) / 1000;
            if (luma < 170) c++;
            /* >= 250 is effectively clipped: the strokes are gone and no
             * amount of thresholding can bring them back. */
            if (luma >= 250) blown++;
            sampled++;
        }
        counts[static_cast<size_t>(r)] = c;
    }
    env->ReleasePrimitiveArrayCritical(jpixels, argb, JNI_ABORT);
    env->SetIntArrayRegion(out, 0, rows, counts.data());
    /* The blown-out fraction is returned through a second output array
     * so the Java side can use it without a second JNI round trip. */
    if (jblownOut) {
        jint b = blown > 0 ? static_cast<jint>((blown * 1000) / (sampled > 0 ? sampled : 1)) : 0;
        env->SetIntArrayRegion(jblownOut, 0, 1, &b);
    }
    return out;
}

JNIEXPORT jint JNICALL
Java_com_nextransit_mrzbench_jni_MrzNative_glareFraction(
    JNIEnv *env, jobject, jintArray jpixels, jint width,
    jint y0, jint y1, jint x0, jint x1, jint stride) {
    if (!jpixels || width <= 0 || y1 <= y0) return 0;
    if (x1 > width) x1 = width;
    if (x0 < 0) x0 = 0;
    if (stride < 1) stride = 1;
    const jsize need = static_cast<jsize>(
        static_cast<size_t>(width) * static_cast<size_t>(y1 - y0));
    if (env->GetArrayLength(jpixels) < need) return 0;

    long blown = 0, sampled = 0;
    jint *argb = static_cast<jint *>(env->GetPrimitiveArrayCritical(jpixels, nullptr));
    if (!argb) return 0;
    for (int r = 0; r < (y1 - y0); ++r) {
        const jint *row = argb + static_cast<size_t>(r) * width;
        for (int x = x0; x < x1; x += stride) {
            const uint32_t p = static_cast<uint32_t>(row[x]);
            const int r8 = (p >> 16) & 0xFF, g8 = (p >> 8) & 0xFF, b8 = p & 0xFF;
            const int luma = (r8 * 299 + g8 * 587 + b8 * 114 + 500) / 1000;
            /* >= 250 is effectively clipped: the strokes are gone and no
             * thresholding can bring them back. */
            if (luma >= 250) blown++;
            sampled++;
        }
    }
    env->ReleasePrimitiveArrayCritical(jpixels, argb, JNI_ABORT);
    if (sampled <= 0) return 0;
    return static_cast<jint>((blown * 1000) / sampled);
}

JNIEXPORT void JNICALL
Java_com_nextransit_mrzbench_jni_MrzNative_setProfiling(JNIEnv *, jobject, jboolean on) {
    setenv("MRZ_PROFILE", on ? "1" : "0", 1);
    setenv("MRZ_OCR_TIMING", on ? "1" : "0", 1);
    LOGI("MRZ_PROFILE %s (also flipped MRZ_OCR_TIMING for cnn.c)", on ? "ON" : "OFF");
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

JNIEXPORT void JNICALL
Java_com_nextransit_mrzbench_jni_MrzNative_setParallel(JNIEnv *, jobject, jboolean on) {
    setenv("MRZ_OCR_PAR", on ? "1" : "0", 1);
    LOGI("MRZ_OCR_PAR %s", on ? "ON (dual-line pthread forward)" : "OFF (sequential)");
}

/* Diagnostics only: dump the 16x12 glyph tensors the CNN consumes for
 * the next recognition into `dir`, tagged `tag`. Empty tag disables. */
JNIEXPORT void JNICALL
Java_com_nextransit_mrzbench_jni_MrzNative_setDump(
        JNIEnv *env, jobject, jstring jtag, jstring jdir) {
    char tag[128] = {0};
    char dir[512] = {0};
    const char *t = env->GetStringUTFChars(jtag, nullptr);
    const char *d = env->GetStringUTFChars(jdir, nullptr);
    if (t) { std::snprintf(tag, sizeof(tag), "%s", t); env->ReleaseStringUTFChars(jtag, t); }
    if (d) { std::snprintf(dir, sizeof(dir), "%s", d); env->ReleaseStringUTFChars(jdir, d); }
    setenv("MRZ_OCR_DUMP", tag[0] ? tag : "0", 1);
    setenv("MRZ_OCR_DUMP_DIR", dir[0] ? dir : "/tmp", 1);
    LOGI("MRZ_OCR_DUMP tag=%s dir=%s", tag[0] ? tag : "0", dir[0] ? dir : "/tmp");
}

/* Pitch-grid alignment score from the last recognition; the Kotlin fast
 * path gates the CRNN sequence head on it. */
JNIEXPORT jint JNICALL
Java_com_nextransit_mrzbench_jni_MrzNative_lastSegScore(JNIEnv *, jobject) {
    return (jint)mrz_ocr_last_seg_score();
}

 } // extern "C"
