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
        // 满幅照：拉脱维亚护照，无黑色取景框；多阈值回退路径必须命中
        Case(0, "2c793efe12276f9b_0258.jpg", 522, 540, 569, 590, dirName = "lv"),
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
            val wall = (System.nanoTime() - t0) / 1_000_000
            val pair = s.pair
            val rectified = s.rectified
            val structureMs = s.timings["structure"] ?: -1L
            val reflectMs = s.timings["remove reflection"] ?: -1L
            val rectifyMs = s.timings["rectify"] ?: -1L
            val cropMs = s.timings["MRZ crop"] ?: -1L
            val totalMs = s.timings["total"] ?: wall
            val sumStages = structureMs + reflectMs + rectifyMs + cropMs

            val pairOk = pair != null &&
                kotlin.math.abs(pair.first.first - c.l1y0) <= 2 &&
                kotlin.math.abs(pair.first.last - c.l1y1) <= 2 &&
                kotlin.math.abs(pair.second.first - c.l2y0) <= 2 &&
                kotlin.math.abs(pair.second.last - c.l2y1) <= 2
            val rectOk = rectified != null &&
                rectified.width == 616 && rectified.height == 104
            val noReflOk = s.noReflection != null
            val reflection = s.timings.entries.firstOrNull { it.key == "reflection" }?.value ?: -1L
            val xpad = s.xpadAdaptive
            // 机位照源图 1250 → 期望 (100, 100)；Photo/7 源 640 → 期望 (51, 51)
            val expectedXpad = when (c.dir) {
                7 -> 51
                0 -> 0  // 满幅照无边条
                else -> 100
            }
            val xpadOk = xpad.first == expectedXpad && xpad.second == expectedXpad

            report.append(
                "Photo/${c.dirName} src=${bmp.width}x${bmp.height} | " +
                "pair=(${pair?.first?.first},${pair?.first?.last})(${pair?.second?.first},${pair?.second?.last}) " +
                "noRefl=$noReflOk rect=${rectified?.let { "${it.width}x${it.height}" } ?: "NULL"} " +
                "xpad=(${xpad.first},${xpad.second}) " +
                "struct=${structureMs}ms refDrop=${reflectMs}ms rectify=${rectifyMs}ms crop=${cropMs}ms " +
                "sum=${sumStages}ms total=${totalMs}ms err='${s.error}'\n"
            )

            if (!pairOk) {
                failures++
                val diff = pair?.let {
                    "diff=(${it.first.first - c.l1y0},${it.first.last - c.l1y1},${it.second.first - c.l2y0},${it.second.last - c.l2y1})"
                } ?: "diff=N/A"
                report.append("  -> pair mismatch (expected (${c.l1y0},${c.l1y1})(${c.l2y0},${c.l2y1})) $diff\n")
            }
            if (!rectOk) failures++
            if (!xpadOk) { failures++; report.append("  -> xpad mismatch (got (${xpad.first},${xpad.second}), expected ($expectedXpad,$expectedXpad))\n") }
            if (rectifyMs > 2000) { failures++; report.append("  -> rectify > 2000ms\n") }
            if (totalMs > 8000) { failures++; report.append("  -> total > 8000ms\n") }
        }

        println("===== PhotoMrzPipelineTest report =====\n$report")
        assertEquals("F1/F2/F3 regressions (see report). failures=$failures\n$report", 0, failures)
    }

    private fun loadJpeg(c: Case): Bitmap? {
        val classLoader = PhotoMrzPipelineTest::class.java.classLoader
        val resource = classLoader?.getResourceAsStream("Photo/${c.dirName}/${c.file}")
            ?: return null
        return resource.use { BitmapFactory.decodeStream(it) }
    }

    /**
     * F3 合成图验证：纯白 Bitmap（无边条框）→ adaptiveXpad 返回 (0, 0)；
     * 边缘 8 列全黑（机位照取景框模型）→ 返回 (nominal, nominal)。
     */
    @Test
    fun adaptiveXpad_zeroOnBorderless() {
        val pipeline = PhotoMrzPipeline
        // 纯白 200x60 Bitmap，无取景框 → (0, 0)
        val borderless = Bitmap.createBitmap(200, 60, Bitmap.Config.ARGB_8888)
        borderless.eraseColor(0xFFFFFFFF.toInt())
        val pair = PairRows(10..20, 30..40, emptyList())
        val (l0, r0) = pipeline.adaptiveXpad(borderless, borderless, pair)
        assertEquals("borderless left", 0, l0)
        assertEquals("borderless right", 0, r0)

        // 边缘 8 列全黑（机位照）→ nominal=16, 16
        val framed = Bitmap.createBitmap(200, 60, Bitmap.Config.ARGB_8888)
        framed.eraseColor(0xFFFFFFFF.toInt())
        paintFrame(framed, 8)
        val (l1, r1) = pipeline.adaptiveXpad(framed, framed, pair)
        // 200*0.08=16, capped [10, 100] → 16
        assertEquals("framed left", 16, l1)
        assertEquals("framed right", 16, r1)
    }

    private fun paintFrame(bmp: Bitmap, strip: Int) {
        val w = bmp.width
        val h = bmp.height
        val black = 0xFF000000.toInt()
        val pixels = IntArray(w * h)
        bmp.getPixels(pixels, 0, w, 0, 0, w, h)
        for (y in 0 until h) {
            for (x in 0 until strip) pixels[y * w + x] = black
            for (x in (w - strip) until w) pixels[y * w + x] = black
        }
        bmp.setPixels(pixels, 0, w, 0, 0, w, h)
    }

    /**
     * F4 国家码 O/0 归一：line2 [10..12] 是国家码。OCR 误读成 P0L → 必须归一到 POL。
     * 通过 postProcess 私有入口（re-invoked via reflection）不可行；改成在测试里直接构造
     * 完整 44 字符的 line2（P0L 国家码 + 正确 checksum），跑 process() 整条链，
     * 观察 postProcess 输出。
     * 实际：postProcess 私有，无暴露面。改：测 enforceLine2Slots 间接路径——通过构造一个
     * 错误国家码但其他位置都对的 line2 走 repairLine2，期望它被修复成 POL。
     * 妥协：用 process() 全流程不可（需要 Bitmap + onnx）。改为反射调用 repairLine2：
     */
    @Test
    fun countryCodeO0Normalize_viaRepair() {
        val pipeline = PhotoMrzPipeline
        // 用 P0L 占位（应为 POL），其余字段和 checksum 均正确
        val line2Raw = "ZF40004998P0L8101028M3203031<<<<<<<<<<<<<<00"
        // 反射调 repairLine2（私有方法）：构造完整 path
        val cls = pipeline::class.java
        val method = cls.declaredMethods.first { it.name == "repairLine2" }
        method.isAccessible = true
        val result = method.invoke(pipeline, line2Raw) as Pair<*, *>
        val fixed = result.first as String
        assertEquals(
            "country 0->O via repair",
            "ZF40004998POL8101028M3203031<<<<<<<<<<<<<<00",
            fixed,
        )
    }

    private data class Case(
        val dir: Int,
        val file: String,
        val l1y0: Int,
        val l1y1: Int,
        val l2y0: Int,
        val l2y1: Int,
        val dirName: String = dir.toString(),
    )
}
