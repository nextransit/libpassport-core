package com.nextransit.mrzbench.jni

/**
 * JNI wrapper for the in-tree C pipeline (3_face_recognition +
 * 6_mrz_ocr). The native side lives in
 * app/src/main/cpp/mrz_jni.cpp and links libmrz_ocr.a + libface.a.
 *
 * Conventions:
 *  - All calls return a JSON string. The Kotlin layer parses it; the
 *    native layer never throws across JNI (host-side exceptions on
 *    thousands of images would be unrecoverable).
 *  - backend 0 = OCR-B template bank; backend 1 = trained CNN.
 *    These names line up with the host mrz_ocr_tool / mrz_ocr_cnn_tool
 *    so the per-image comparisons stay meaningful.
 */
object MrzNative {
    init {
        System.loadLibrary("mrz_jni")
    }

    @JvmStatic external fun versionString(): String

    /** Recognise a PPM (P6) image at `path`. Returns JSON. */
    @JvmStatic external fun recogniseString(path: String, backend: Int): String

    /** Load PPM header only (cheap, used to filter non-image files). */
    @JvmStatic external fun loadPpmInfo(path: String): String

    /**
     * Toggle per-image stage logging through __android_log_print under
     * the "MrzNative" tag. Also flips the in-tree `MRZ_OCR_TIMING` env
     * var so `mrz_ocr_cnn.c` records its own per-stage breakdown.
     */
    external fun setProfiling(on: Boolean)

    /**
     * Toggle dual-line CNN forward parallelism in the recogniser
     * (Step 2 of the A53 perf plan). When ON, the recogniser
     * serialises the resample step on the main thread and then
     * forks a worker that runs line 1's cnn forward while the
     * main thread runs line 0. Disjoint scratchpad halves keep
     * the bump allocator race-free. Default is OFF (sequential).
     */
    @JvmStatic external fun setParallel(on: Boolean)
}
