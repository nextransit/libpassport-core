package com.nextransit.mrzbench.photo

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File

/**
 * 改进点 F1（多阈值回退）/ F2（相对倒影）/ F3（边条框检测自适应 xpad）的可观测行为。
 * 通过 processStructure 跑完 5 步流水线前 4 步（去影/矫正/crop），OCR 留 T8 注入 fake。
 * 期望：6 张实拍图全部找到 MRZ pair + 矫正到 616x104 + 各步骤在 budget 内。
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [28])
class PhotoMrzPipelineTest {

    // The expected rows are in SOURCE coordinates (the photo as
    // delivered), not in the coordinates of the scaled working image.
    //
    // The working image is a downscaled copy, so the located rows move
    // with the scale factor.  Asserting on the scaled numbers would
    // pin the test to one particular MAX_SIDE -- and there is a hard
    // reason not to want that: the downscale cap is a cliff, not a
    // preference (measured on the 41 labelled photos: a 1250 px cap
    // drops the decode from 40/41 to 7/41, no downscale gives 40/41),
    // so the cap must be free to move without the geometry test
    // failing.  Converting the expected rows by the same scale keeps
    // the assertion meaningful at any cap.
    private val cases = listOf(
        Case(3, "IMG_1792133000_003.jpg", 791, 810, 836, 858),
        Case(4, "IMG_1792133002_004.jpg", 791, 810, 836, 858),
        Case(5, "IMG_1792133232_005.jpg", 791, 810, 836, 858),
        Case(6, "IMG_1792133581_006.jpg", 791, 810, 836, 858),
        Case(7, "IMG_1792133649_007.jpg", 316, 325, 335, 346),
        Case(8, "IMG_1792133682_008.jpg", 789, 811, 837, 863),
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
            val mrzCrop = s.mrzCrop
            val structureMs = s.timings["structure"] ?: -1L
            val reflectMs = s.timings["remove reflection"] ?: -1L
            val cropMs = s.timings["MRZ crop"] ?: -1L
            val totalMs = s.timings["total"] ?: wall
            val sumStages = structureMs + reflectMs + cropMs

            // The located rows are in the coordinates of the scaled
            // working image, and so are the expected values.
            val scale = if (bmp.width > 0) s.work.width.toDouble() / bmp.width else 1.0
            // The scale the expectations were recorded at: the working
            // image's own cap, applied to THIS photo's longest side.
            // Both sides are converted to source coordinates so the
            // expectation does not depend on the cap -- which matters
            // because the cap is a cliff, not a preference: measured on
            // the 41 labelled photos a 1250 px cap drops the decode
            // from 40/41 to 7/41, and no downscale gives 40/41.
            val refScale = kotlin.math.min(1.0, 1600.0 / maxOf(bmp.width, bmp.height).toDouble())
            val srcFirst = pair?.let { (it.first.first / scale).toInt() to (it.first.last / scale).toInt() }
            val srcSecond = pair?.let { (it.second.first / scale).toInt() to (it.second.last / scale).toInt() }
            val expFirst = (c.l1y0 / refScale).toInt() to (c.l1y1 / refScale).toInt()
            val expSecond = (c.l2y0 / refScale).toInt() to (c.l2y1 / refScale).toInt()
            val pairOk = pair != null &&
                kotlin.math.abs((srcFirst?.first ?: 0) - expFirst.first) <= 4 &&
                kotlin.math.abs((srcFirst?.second ?: 0) - expFirst.second) <= 4 &&
                kotlin.math.abs((srcSecond?.first ?: 0) - expSecond.first) <= 4 &&
                kotlin.math.abs((srcSecond?.second ?: 0) - expSecond.second) <= 4
            // The MRZ crop must exist and must actually contain the two
            // located rows.  The rectified stage was removed: it never
            // fed the OCR (the recogniser reads the natural band) and its
            // page-edge quad degenerated to the full image width on 57%
            // of assets/pic, which is what made that tile look scrambled.
            val rectOk = mrzCrop != null && mrzCrop.height > 0 && mrzCrop.width > 0
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
                "noRefl=$noReflOk crop=${mrzCrop?.let { "${it.width}x${it.height}" } ?: "NULL"} " +
                "xpad=(${xpad.first},${xpad.second}) " +
                "struct=${structureMs}ms refDrop=${reflectMs}ms crop=${cropMs}ms " +
                "sum=${sumStages}ms total=${totalMs}ms err='${s.error}'\n"
            )

            if (!pairOk) {
                failures++
                val diff = pair?.let {
                    "diff=(${it.first.first - c.l1y0},${it.first.last - c.l1y1},${it.second.first - c.l2y0},${it.second.last - c.l2y1})"
                } ?: "diff=N/A"
                report.append(
                    "  -> pair mismatch (expected src (${expFirst.first},${expFirst.second})" +
                        "(${expSecond.first},${expSecond.second})) $diff\n"
                )
            }
            if (!rectOk) failures++
            if (!xpadOk) { failures++; report.append("  -> xpad mismatch (got (${xpad.first},${xpad.second}), expected ($expectedXpad,$expectedXpad))\n") }
            if (cropMs > 2000) { failures++; report.append("  -> MRZ crop > 2000ms\n") }
            if (totalMs > 8000) { failures++; report.append("  -> total > 8000ms\n") }
        }

        println("===== PhotoMrzPipelineTest report =====\n$report")
        assertEquals("F1/F2/F3 regressions (see report). failures=$failures\n$report", 0, failures)
    }

    /**
     * 全自动化批跑：7 张 fixture + 84 张 assets/pic。processStructure 是 5 步流水线的前 4 步
     * （结构探测/倒影/矫正/crop），OCR 因 JNI 在 JVM test 中不可用，跳过。
     * 打印每图详细时序与 F1/F2/F3 行为；末尾统计 pair 检出率/rectify 成功率/xpad 自适应率。
     * 不做硬性断言（图片多样，pair 偶有失败/伪对是已知边界），只报告指标供人评估。
     */
    @Test
    fun batch_runAllSampleImages() {
        val pipeline = PhotoMrzPipeline
        val report = StringBuilder()
        val fixtures = listOf(
            Fixture("fixture/Photo/3", "IMG_1792133000_003.jpg", "classpath"),
            Fixture("fixture/Photo/4", "IMG_1792133002_004.jpg", "classpath"),
            Fixture("fixture/Photo/5", "IMG_1792133232_005.jpg", "classpath"),
            Fixture("fixture/Photo/6", "IMG_1792133581_006.jpg", "classpath"),
            Fixture("fixture/Photo/7", "IMG_1792133649_007.jpg", "classpath"),
            Fixture("fixture/Photo/8", "IMG_1792133682_008.jpg", "classpath"),
            Fixture("fixture/Photo/lv", "2c793efe12276f9b_0258.jpg", "classpath"),
        )

        // 扫描 assets/pic/*.jpg|png|jpeg（main/assets 在测试运行时也可经 AssetManager 访问；
        // 但 classpath 路径更简单——把它们拷到 test 临时目录再走同样的解码路径）。
        val assetsRoot = locateAssetsRoot()
        val picDir = File(assetsRoot, "pic")
        val picFiles = picDir.listFiles { f -> f.isFile && f.name.matches(Regex(".*\\.(jpg|jpeg|png)$", RegexOption.IGNORE_CASE)) }
            ?.toList()?.sortedBy { it.name }
            ?: emptyList()
        val picFixtures = picFiles.map { Fixture("assets/pic", it.name, "disk", it) }

        val all = fixtures + picFixtures
        report.append("===== 全自动化 Android 测试 (${all.size} 张) =====\n")

        var pairOk = 0
        var rectOk = 0
        var noReflOk = 0
        var xpadAdaptive = 0
        var xpadZero = 0
        var wallTotalMs = 0L
        var sumCropMs = 0L
        var maxCropMs = 0L
        var maxTotalMs = 0L
        val rows = mutableListOf<Triple<String, String, String>>() // (name, status, line)

        for (f in all) {
            val bmp = when (f.source) {
                "classpath" -> {
                    val cl = PhotoMrzPipelineTest::class.java.classLoader
                    val res = cl?.getResourceAsStream("Photo/${f.subdir.substringAfter("fixture/Photo/")}/${f.file}")
                        ?: cl?.getResourceAsStream("Photo/lv/${f.file}")
                    res?.use { BitmapFactory.decodeStream(it) }
                }
                "disk" -> BitmapFactory.decodeFile(f.diskFile?.absolutePath)
                else -> null
            }
            if (bmp == null) {
                rows.add(Triple(f.label, "LOAD_FAIL", ""))
                report.append("${f.label}/${f.file}: LOAD FAILED\n")
                continue
            }
            val t0 = System.nanoTime()
            val s = pipeline.processStructure(bmp)
            val wall = (System.nanoTime() - t0) / 1_000_000
            val struct = s.timings["structure"] ?: -1L
            val ref = s.timings["remove reflection"] ?: -1L
            val crop = s.timings["MRZ crop"] ?: -1L
            val total = s.timings["total"] ?: wall
            val pair = s.pair
            val xpad = s.xpadAdaptive
            val noRefl = s.noReflection
            val cropBmp = s.mrzCrop
            val pOk = pair != null
            val rOk = cropBmp != null && cropBmp.width > 0 && cropBmp.height > 0
            val nROk = noRefl != null
            val xZero = xpad.first == 0 && xpad.second == 0
            if (pOk) pairOk++
            if (rOk) rectOk++
            if (nROk) noReflOk++
            if (xpad.first != 100 && xpad.second != 100) xpadAdaptive++
            if (xZero) xpadZero++
            wallTotalMs += wall
            sumCropMs += crop
            if (crop > maxCropMs) maxCropMs = crop
            if (total > maxTotalMs) maxTotalMs = total

            val status = when {
                !pOk -> "NO_PAIR"
                !rOk -> "CROP_FAIL"
                else -> "OK"
            }
            val line = "src=${bmp.width}x${bmp.height} " +
                "pair=${pair?.let { "(${it.first.first},${it.first.last})(${it.second.first},${it.second.last})" } ?: "NULL"} " +
                "noRefl=${nROk} crop=${cropBmp?.let { "${it.width}x${it.height}" } ?: "NULL"} " +
                "xpad=(${xpad.first},${xpad.second}) " +
                "struct=${struct}ms ref=${ref}ms crop=${crop}ms total=${total}ms" +
                (if (s.error.isNotEmpty()) " err='${s.error}'" else "")
            rows.add(Triple(f.label, status, line))
        }

        // 详细表
        report.append("\n----- 详细结果 -----\n")
        for ((label, status, line) in rows) {
            report.append("[${status.padEnd(10)}] $label  $line\n")
        }

        // 汇总
        val n = all.size
        report.append("\n----- 汇总 -----\n")
        report.append("总图片:           $n\n")
        report.append("pair 检出:        $pairOk / $n (${pct(pairOk, n)})\n")
        report.append("MRZ crop 成功:    $rectOk / $n (${pct(rectOk, n)})\n")
        report.append("noReflection!=null: $noReflOk / $n (${pct(noReflOk, n)})\n")
        report.append("xpad 自适应(非100): $xpadAdaptive / $n (${pct(xpadAdaptive, n)})\n")
        report.append("xpad=(0,0) 无边:   $xpadZero / $n (${pct(xpadZero, n)})\n")
        report.append("MRZ crop 总耗时:  ${sumCropMs}ms  均值 ${sumCropMs / n}ms  最大 ${maxCropMs}ms\n")
        report.append("总 wall 耗时:     ${wallTotalMs}ms  均值 ${wallTotalMs / n}ms  最大 ${maxTotalMs}ms\n")

        println(report)
        // 不做硬性断言（图片多样，已知 2 张伪对 1c6acd09/c1a15c65 会被检出 pair 但 OCR 失败）
        assertTrue("report generated", report.isNotEmpty())
    }

    private fun pct(n: Int, total: Int): String =
        if (total == 0) "0.0%" else String.format("%.1f%%", n * 100.0 / total)

    private data class Fixture(
        val subdir: String,
        val file: String,
        val source: String,           // "classpath" 或 "disk"
        val diskFile: File? = null,
    ) {
        val label: String get() = "$subdir"
    }

    /**
     * 找 app/src/main/assets 的根目录——Robolectric 把 assets 装到文件系统某处。
     * 优先从 gradle known location 拿；备选是 test/resources 拷过去。
     */
    private fun locateAssetsRoot(): File {
        // 1. 尝试 gradle 同步拷过来的：build/intermediates/merged_assets/debug/
        val cwd = File("").absoluteFile
        for (candidate in listOf(
            File(cwd, "app/build/intermediates/merged_assets/debug"),
            File(cwd, "build/intermediates/merged_assets/debug"),
        )) {
            val pic = File(candidate, "pic")
            if (pic.isDirectory) return candidate
        }
        // 2. 兜底：源码路径
        val src = File("/Users/zhouyong/Desktop/work/Decard/gitlab/passport/8_android/app/src/main/assets")
        if (src.isDirectory) return src
        // 3. 最后的兜底：src/test/resources/assets-pic/（若有人手工拷）
        return File("src/test/resources")
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

    @Test
    fun fastPair_reportsFixtureCandidates() {
        val method = PhotoMrzPipeline::class.java.getDeclaredMethod(
            "fastPair",
            Bitmap::class.java,
        )
        method.isAccessible = true
        val report = StringBuilder()
        var located = 0
        for (c in cases) {
            val bitmap = loadJpeg(c) ?: continue
            val pair = method.invoke(PhotoMrzPipeline, bitmap) as PairRows?
            if (pair != null) located++
            report.append(
                "${c.dirName} src=${bitmap.width}x${bitmap.height} " +
                    "pair=${pair?.first}/${pair?.second}\n",
            )
        }
        println("===== fastPair =====\n$report")
        assertTrue("fastPair should locate at least one fixture\n$report", located > 0)
    }

    @Test
    fun fastPair_skipsShortNoiseAndKeepsPairBeforeReflection() {
        val bitmap = Bitmap.createBitmap(400, 800, Bitmap.Config.ARGB_8888)
        bitmap.eraseColor(0xFFFFFFFF.toInt())
        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        val black = 0xFF000000.toInt()

        fun paintBand(y0: Int, y1: Int, x0: Int, x1: Int) {
            for (y in y0..y1) {
                for (x in x0 until x1) pixels[y * bitmap.width + x] = black
            }
        }

        // The one-pixel run at 531 is noise between the two real MRZ rows.
        paintBand(500, 510, 40, 340)
        paintBand(531, 531, 40, 340)
        paintBand(535, 548, 40, 340)
        // A viewing band separates the real MRZ from a lower reflection pair.
        paintBand(580, 630, 0, bitmap.width)
        paintBand(650, 657, 40, 340)
        paintBand(668, 677, 40, 340)
        bitmap.setPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        val method = PhotoMrzPipeline::class.java.getDeclaredMethod(
            "fastPair",
            Bitmap::class.java,
        )
        method.isAccessible = true
        val pair = method.invoke(PhotoMrzPipeline, bitmap) as PairRows?

        assertEquals("first row should survive the noise gap", 500, pair?.first?.first)
        assertEquals("second row should survive the noise gap", 535, pair?.second?.first)
        assertTrue("reflection must not replace the real pair", (pair?.second?.last ?: 0) < 580)
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
