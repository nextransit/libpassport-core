package com.nextransit.mrzbench

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.nextransit.mrzbench.jni.MrzNative
import com.nextransit.mrzbench.photo.PaddleOcrAndroid
import com.nextransit.mrzbench.photo.PhotoMrzPipeline
import com.nextransit.mrzbench.photo.verdict
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import kotlin.math.roundToInt

/**
 * 免界面设备侧自动化：对 assets 的 Photo 目录样例与 androidTest 内置扫描图
 * （assets/PhotoScan）逐算法跑 PhotoMrzPipeline，结果用 logcat 输出
 * （adb logcat -s MrzOcr；imageFilter 参数可切片）。
 */
@RunWith(AndroidJUnit4::class)
class PhotoMrzDeviceTest {

    private val context get() = InstrumentationRegistry.getInstrumentation().targetContext

    @Test
    fun allMethodsOnAllImages() {
        val filter = InstrumentationRegistry.getArguments().getString("imageFilter").orEmpty()
        val filters = filter.split(',').filter { it.isNotEmpty() }
        // Default to the PRODUCT path ("auto").  The other methods are
        // comparison channels: they run the detailed pipeline, which
        // downscales the photo and costs 300+ ms, so they must be
        // selected explicitly.  Without this default a plain
        // `connectedDebugAndroidTest` would spend most of its time on
        // the comparison modes and never exercise the path the product
        // actually uses.
        val methodFilter = InstrumentationRegistry.getArguments().getString("methods").orEmpty()
            .ifEmpty { "auto" }
        val requestedMethods = methodFilter.split(',').filter { it.isNotEmpty() }.toSet()
        // Default to the omtest corpus that is pushed to the app's
        // external files directory.  Without it the batch only covers
        // the 18 bundled samples and the 44 real photos are never
        // exercised.
        //
        // Several roots may be given, comma-separated.  The app's
        // *external* files directory (/sdcard/Android/data/<pkg>/files)
        // is unreadable from the app process on API 30+ when the
        // directory was created by `adb push` as the `shell` user: the
        // FUSE layer denies it to the app's own uid.  The app's
        // *internal* files directory is always readable, so the
        // corpus is pushed there instead.  Both roots are tried and
        // the first that exists wins, so a caller does not have to
        // know which of the two the current device allows.
        val externalRoots = InstrumentationRegistry.getArguments().getString("externalDir").orEmpty()
            .ifEmpty { "/sdcard/Android/data/com.nextransit.mrzbench/files/omtest" }
            .split(',')
            .map { it.trim() }
            .filter { it.isNotEmpty() }
        val externalDir = externalRoots.firstOrNull { File(it).isDirectory }
            ?: File(context.filesDir, "omtest").absolutePath
        if (!File(externalDir).isDirectory) {
            Log.i(TAG, "TEST externalDir unavailable roots='$externalRoots'")
        } else {
            Log.i(TAG, "TEST externalDir='$externalDir'")
        }
        val parallel = InstrumentationRegistry.getArguments().getString("parallel").orEmpty()
        if (parallel.isNotEmpty()) {
            MrzNative.setParallel(parallel == "1")
            Log.i(TAG, "TEST parallel=${parallel == "1"}")
        }
        // Weight-set switch: "passport" (default) or "pic" (the assets/pic
        // set, namespaced into the same .so).  Mirrors the host split into
        // mrz_ocr_cnn_tool vs mrz_ocr_cnn_tool_pic.
        val weightSet = InstrumentationRegistry.getArguments().getString("weightSet").orEmpty()
        if (weightSet.isNotEmpty()) {
            MrzNative.setWeightSet(weightSet)
            Log.i(TAG, "TEST weightSet=$weightSet")
        }
        val paddleMode = InstrumentationRegistry.getArguments().getString("paddleMode").orEmpty()
        if (paddleMode.isNotEmpty()) {
            PaddleOcrAndroid.mode = when (paddleMode) {
                "medium" -> PaddleOcrAndroid.Mode.MEDIUM
                "mobile" -> PaddleOcrAndroid.Mode.MOBILE
                else -> PaddleOcrAndroid.Mode.HYBRID
            }
            Log.i(TAG, "TEST paddleMode=$paddleMode")
        }
        val dumpGlyphs = InstrumentationRegistry.getArguments().getString("dump").orEmpty() == "1"
        val dumpDir = File(context.filesDir, "glyphdump").apply { mkdirs() }
        if (dumpGlyphs) Log.i(TAG, "TEST dumpGlyphs dir=$dumpDir")
        val all = assetImages() + scanImages() + externalImages(externalDir)
        val images = if (filters.isEmpty()) all else all.filter { p -> filters.any { p.first.contains(it) } }
        Log.i(TAG, "TEST start images=${images.size} filter='$filter' externalDir='$externalDir'")
        var pass = 0
        var fail = 0
        val selectedMethods = methods().filter { requestedMethods.isEmpty() || it.first in requestedMethods }
        Log.i(TAG, "TEST methods=${selectedMethods.map { it.first }}")
        images.forEach { (label, load) ->
            val bmp = load()
            if (bmp == null) {
                Log.i(TAG, "RESULT $label | decode failed")
                return@forEach
            }
            try {
                if (dumpGlyphs) {
                    MrzNative.setDump(File(label).nameWithoutExtension, dumpDir.absolutePath)
                }
                selectedMethods.forEach { (key, display) ->
                    if (run(label, display, key, bmp)) pass++ else fail++
                }
            } finally {
                if (dumpGlyphs) MrzNative.setDump("", dumpDir.absolutePath)
                bmp.recycle()
            }
        }
        Log.i(TAG, "TEST summary pass=$pass fail=$fail")
    }

    private fun run(label: String, display: String, key: String, bmp: Bitmap): Boolean {
        val started = System.currentTimeMillis()
        return try {
            val out = PhotoMrzPipeline.process(context, bmp, key)
            Log.i(
                TAG,
                "RESULT $label | $display | decode=${out.decodeOk} | wall=${System.currentTimeMillis() - started}ms" +
                    " | total=${out.timings["total"]}ms | backend=${out.backend}" +
                    " | ocr=${out.timings["OCR + decode"]}ms | structure=${out.timings["structure"]}ms" +
                    " | raw=${out.rawDecodeOk} | repaired=${out.repaired}" +
                    " | struct=${out.structOk} | nat=${out.natSuspect}" +
                    " | issuer=${out.issuerSuspect} | country=${out.countryMismatch}" +
                    " | verdict=${out.verdict}" +
                    " | L1=${out.line1} | L2=${out.line2} | err=${out.error}",
            )
            out.decodeOk
        } catch (t: Throwable) {
            Log.i(TAG, "RESULT $label | $display | CRASH ${t.javaClass.simpleName}: ${t.message}")
            false
        }
    }

    /** 探测：PC 生成的 band.ppm（与 PC tool 完全同路径）在设备上的识别结果。 */
    @Test
    fun probeBandPpm() {
        val path = InstrumentationRegistry.getArguments().getString("ppmPath").orEmpty()
        // backend 可选：默认 1（纯 CNN），传 3 走融合引擎，用于与 PC 端
        // MRZ_OCR_FUSE=1 的输出做逐字符比对。
        val backend = InstrumentationRegistry.getArguments().getString("backend")
            ?.toIntOrNull() ?: 1
        val rec = MrzNative.recogniseString(path, backend)
        Log.i(TAG, "PROBE recogn_string=$rec")
    }

    /** 探测：同一张图在设备上经 processDetailed 的 band 与 PC band.ppm 是否逐像素一致。 */
    @Test
    fun probeBandPixels() {
        val jpg = InstrumentationRegistry.getArguments().getString("jpgPath").orEmpty()
        val ppm = InstrumentationRegistry.getArguments().getString("ppmPath").orEmpty()
        val bmp = BitmapFactory.decodeFile(jpg) ?: run {
            Log.i(TAG, "PROBE decode-fail $jpg"); return
        }
        val pairL = InstrumentationRegistry.getArguments().getString("pairL").orEmpty().split("..").map { it.toInt() }
        val pairR = InstrumentationRegistry.getArguments().getString("pairR").orEmpty().split("..").map { it.toInt() }
        val xp = InstrumentationRegistry.getArguments().getString("xp").orEmpty().split(",").map { it.toInt() }
        val first = pairL[0]..pairL[1]
        val second = pairR[0]..pairR[1]
        val firstSourceY1 = minOf(bmp.height, first.last + 1)
        val rows = listOf(first, second).map { run ->
            var y0 = run.first
            val y1 = run.last + 1
            if (run == second) y0 = maxOf(y0, firstSourceY1)
            val h = y1 - y0
            val pad = maxOf(8, minOf(25, (h * 0.70).roundToInt()))
            val w = bmp.width - xp[0] - xp[1]
            Bitmap.createBitmap(bmp, xp[0], maxOf(0, y0 - pad), w, minOf(bmp.height, y1 + pad) - maxOf(0, y0 - pad))
        }
        val firstTop = first.first
        val secondBottom = second.last + 1
        val sharedHeight = maxOf(rows[0].height, rows[1].height)
        val sharedPad = maxOf(8, minOf(25, (sharedHeight * 0.70).roundToInt()))
        val cropTop = maxOf(0, firstTop - sharedPad)
        val cropBottom = minOf(bmp.height, secondBottom + sharedPad)
        val width = minOf(rows[0].width, rows[1].width)
        val band = Bitmap.createBitmap(bmp, xp[0], cropTop, width, cropBottom - cropTop)
        Log.i(TAG, "PROBE band=${band.width}x${band.height} cropTop=$cropTop cropBottom=$cropBottom sharedPad=$sharedPad")
        val pixels = IntArray(band.width * band.height)
        band.getPixels(pixels, 0, band.width, 0, 0, band.width, band.height)
        var s1 = 0L
        var s2 = 0L
        for (i in pixels.indices) {
            s1 += pixels[i] and 0xFF
            s2 += (pixels[i] ushr 16) and 0xFF
        }
        Log.i(TAG, "PROBE band_hash sum_r=$s1 sum_b=$s2")
        val recPix = MrzNative.recognisePixels(pixels, band.width, band.height, 1, 0)
        Log.i(TAG, "PROBE recogn_pixels=$recPix")
        val ppmPixels = IntArray(band.width * band.height)
        val ok = MrzNative.loadPpmInfo(ppm)
        Log.i(TAG, "PROBE loadPpmInfo=$ok")
        val outPpm = File(context.filesDir, "device_band.ppm")
        outPpm.outputStream().use { os ->
            os.write("P6\n${band.width} ${band.height}\n255\n".toByteArray(Charsets.US_ASCII))
            val rgb = ByteArray(band.width * band.height * 3)
            for (i in pixels.indices) {
                val c = pixels[i]
                rgb[i * 3] = ((c ushr 16) and 0xFF).toByte()
                rgb[i * 3 + 1] = ((c ushr 8) and 0xFF).toByte()
                rgb[i * 3 + 2] = (c and 0xFF).toByte()
            }
            os.write(rgb)
        }
        val recStr = MrzNative.recogniseString(outPpm.absolutePath, 1)
        Log.i(TAG, "PROBE recogn_string_device_ppm=$recStr")
    }

    private fun methods(): List<Pair<String, String>> =
        listOf("auto" to "自动") + PhotoMrzPipeline.METHODS

    private fun assetImages(): List<Pair<String, () -> Bitmap?>> {
        val assets = context.assets
        val out = mutableListOf<Pair<String, () -> Bitmap?>>()
        for (dir in (assets.list("Photo") ?: emptyArray()).sorted()) {
            for (f in (assets.list("Photo/$dir") ?: emptyArray()).sorted()) {
                if (!imageName(f)) continue
                val path = "Photo/$dir/$f"
                out += path to { assets.open(path).use { BitmapFactory.decodeStream(it) } }
            }
        }
        return out
    }

    /** 扫描样张随 androidTest APK 打包（assets/PhotoScan），免设备推送。 */
    private fun scanImages(): List<Pair<String, () -> Bitmap?>> {
        val assets = InstrumentationRegistry.getInstrumentation().context.assets
        return (assets.list("PhotoScan") ?: emptyArray())
            .sorted()
            .filter { imageName(it) }
            .map { f -> f to { assets.open("PhotoScan/$f").use { BitmapFactory.decodeStream(it) } } }
    }

    private fun externalImages(rootPath: String): List<Pair<String, () -> Bitmap?>> {
        if (rootPath.isBlank()) return emptyList()
        val root = File(rootPath)
        if (!root.isDirectory) {
            Log.i(TAG, "TEST externalDir unavailable path='$rootPath'")
            return emptyList()
        }
        return root.walkTopDown()
            .filter { it.isFile && imageName(it.name) && !it.name.endsWith("-result.jpg", true) }
            .sortedBy { it.absolutePath }
            .map { file ->
                val label = "external/" + file.relativeTo(root).path
                label to { BitmapFactory.decodeFile(file.absolutePath) }
            }
            .toList()
    }

    private fun imageName(name: String): Boolean =
        name.endsWith(".jpg", true) || name.endsWith(".jpeg", true) || name.endsWith(".png", true)

    private companion object {
        const val TAG = "MrzOcr"
    }
}
