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

    /**
     * Recognise an Android ARGB_8888 pixel buffer without filesystem I/O.
     *
     * `deadlineMs <= 0` disables the native deadline. A positive deadline
     * is measured inside the native recogniser and returns `timeout` when
     * the recogniser cannot finish in time.
     */
    @JvmStatic
    external fun recognisePixels(
        pixels: IntArray,
        width: Int,
        height: Int,
        backend: Int,
        deadlineMs: Int,
    ): String

    /**
     * Row darkness profile over the sampled columns of the rows
     * [y0, y1) of an ARGB pixel buffer.
     *
     * Returns one count per row: how many of the sampled pixels are
     * darker than the MRZ ink threshold.  Doing this natively avoids
     * ~500 k interpreted luma computations per photo.
     */
    @JvmStatic
    external fun rowProfile(
        pixels: IntArray,
        width: Int,
        y0: Int,
        y1: Int,
        x0: Int,
        x1: Int,
        stride: Int,
        blownOut: IntArray?,
    ): IntArray?

    /**
     * Blown-out (specular highlight) fraction of a region, in
     * thousandths.
     *
     * A passport page under a plastic laminate throws a specular
     * highlight that erases the strokes; when that happens the OCR
     * result is unreliable and the user should be told to tilt the
     * phone rather than be shown a plausible-looking but wrong line.
     * The fraction comes from the SAME pixel pass as the row profile
     * (the luma of every sampled pixel is computed anyway), so it
     * costs nothing extra.
     */
    @JvmStatic
    external fun glareFraction(
        pixels: IntArray,
        width: Int,
        y0: Int,
        y1: Int,
        x0: Int,
        x1: Int,
        stride: Int,
    ): Int


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

    /**
     * Diagnostics only (bench harness): write the exact 16x12 glyph
     * tensors the CNN consumes for the next recognition into `dir`,
     * filenames tagged with `tag`. Empty tag disables dumping.
     */
    @JvmStatic external fun setDump(tag: String, dir: String)

    /**
     * Pitch-grid alignment score (0..100) from the most recent recognition.
     * 100 means every ink run sat at the centre of its 44-cell slot; low
     * values mean the monospaced-cell assumption is broken, which is what
     * makes the CRNN sequence head propagate a single misaligned cell
     * across the whole row. Returns -1 before the first recognition.
     *
     * The fast path uses this to decide whether the sequence head is safe
     * or whether the per-character CNN should run instead.
     */
    @JvmStatic external fun lastSegScore(): Int
}
