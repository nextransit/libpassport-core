package com.nextransit.mrzbench.photo

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.core.app.ApplicationProvider
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * 改进点 F1（多阈值回退）/ F2（相对倒影）/ F3（边条框检测自适应 xpad）的可观测行为。
 * 通过 processStructure 跑完 5 步流水线前 4 步（去影/矫正/crop），OCR 留 T8 注入 fake。
 * 期望：6 张实拍图全部找到 MRZ pair + 矫正到 616x104 + 各步骤在 budget 内。
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [28])
class PhotoMrzPipelineTest {

    private val cases = listOf(
        Case(3, "IMG_1792133000_003.jpg", 618, 632, 653, 670),
        Case(4, "IMG_1792133002_004.jpg", 618, 632, 653, 670),
        Case(5, "IMG_1792133232_005.jpg", 618, 632, 653, 670),
        Case(6, "IMG_1792133581_006.jpg", 618, 632, 653, 670),
        Case(7, "IMG_1792133649_007.jpg", 316, 325, 335, 346),
        Case(8, "IMG_1792133682_008.jpg", 616, 633, 654, 674),
    )

    @Test
    fun allPhotos_locateAndRectifyWithinBudget() {
        val pipeline = PhotoMrzPipeline
        val report = StringBuilder()
        var failures = 0

        for (c in cases) {
            val bmp = loadJpeg(c)
            if (bmp == null) {
                report.append("Photo/${c.dir}: LOAD FAILED (missing fixture)\n")
                failures++
                continue
            }
            val t0 = System.nanoTime()
            val s = pipeline.processStructure(bmp)
            val total = (System.nanoTime() - t0) / 1_000_000
            val pair = s.pair
            val rectified = s.rectified
            val rectifyMs = s.timings["rectify"] ?: -1L
            val totalMs = s.timings["total"] ?: total

            val pairOk = pair != null &&
                kotlin.math.abs(pair.first.first - c.l1y0) <= 2 &&
                kotlin.math.abs(pair.first.last - c.l1y1) <= 2 &&
                kotlin.math.abs(pair.second.first - c.l2y0) <= 2 &&
                kotlin.math.abs(pair.second.last - c.l2y1) <= 2
            val rectOk = rectified != null &&
                rectified.width == 616 && rectified.height == 104
            val noReflOk = s.noReflection != null

            report.append("Photo/${c.dir}: pair=${pair?.let { "(${it.first.first},${it.first.last})(${it.second.first},${it.second.last})" } ?: "NULL"} ")
                .append("noRefl=${noReflOk} rect=${rectified?.let { "${it.width}x${it.height}" } ?: "NULL"} ")
                .append("rectifyMs=$rectifyMs totalMs=$totalMs err='${s.error}'\n")

            if (!pairOk) {
                failures++
                val diff = pair?.let {
                    "diff=(${it.first.first - c.l1y0},${it.first.last - c.l1y1},${it.second.first - c.l2y0},${it.second.last - c.l2y1})"
                } ?: "diff=N/A"
                report.append("  -> pair mismatch (expected (${c.l1y0},${c.l1y1})(${c.l2y0},${c.l2y1})) $diff\n")
            }
            if (!rectOk) failures++
            if (rectifyMs > 2000) { failures++; report.append("  -> rectify > 2000ms\n") }
            if (totalMs > 8000) { failures++; report.append("  -> total > 8000ms\n") }
        }

        println("===== PhotoMrzPipelineTest report =====\n$report")
        assertEquals("F1/F2/F3 regressions (see report). failures=$failures\n$report", 0, failures)
    }

    private fun loadJpeg(c: Case): Bitmap? {
        val classLoader = PhotoMrzPipelineTest::class.java.classLoader
        val resource = classLoader?.getResourceAsStream("Photo/${c.dir}/${c.file}")
            ?: return null
        return resource.use { BitmapFactory.decodeStream(it) }
    }

    private data class Case(
        val dir: Int,
        val file: String,
        val l1y0: Int,
        val l1y1: Int,
        val l2y0: Int,
        val l2y1: Int,
    )
}
