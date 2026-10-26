package com.nextransit.mrzbench.photo.ocr.tesseract

import android.content.Context
import android.graphics.Bitmap
import com.nextransit.mrzbench.photo.ocr.OcrEngine
import com.nextransit.mrzbench.photo.ocr.OcrResult
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.roundToInt
import kotlin.math.sin

class TesseractEngine private constructor(private val assetDir: File) : OcrEngine {
    override val name = "tesseract"
    private val lock = ReentrantLock()
    private val cond: java.util.concurrent.locks.Condition = lock.newCondition()
    private val pool = ArrayDeque<NativeTess>()
    private var created = 0
    private val maxHandles = 2

    private fun acquire(): NativeTess? {
        lock.withLock {
            while (true) {
                pool.removeFirstOrNull()?.let { return it }
                if (created < maxHandles) {
                    created++
                    val t = NativeTess.create(assetDir.parentFile?.absolutePath ?: assetDir.absolutePath)
                    if (t == null) { created--; cond.signalAll(); return null }
                    return t
                }
                cond.await()
            }
        }
    }

    private fun release(t: NativeTess) {
        lock.withLock { pool.addLast(t); cond.signalAll() }
    }

    /** Pre-open a handle so the first photo does not pay engine init. */
    fun warm() {
        acquire()?.let { release(it) }
    }

    override fun recognize(line1: Bitmap, line2: Bitmap, psm: Int): OcrResult {
        val start = System.nanoTime()
        val t = acquire() ?: return OcrResult(name, "", "", 0, 0, 0, "init failed")
        return try {
            t.setPsm(psm)
            // 桌面：line2 恒 2x；line1 仅 row_h<20（≈bitmap 高<46）时 2x；灰度域 LANCZOS 等价桌面。
            val g1 = prepare(line1, line1.height < 46)
            val g2 = prepare(line2, true)
            val l1 = t.recognize(g1.bytes, g1.width, g1.height, 1, g1.width).trim()
            val l2 = t.recognize(g2.bytes, g2.width, g2.height, 1, g2.width).trim()
            android.util.Log.i("MrzOcr", "tess raw l1='$l1' l2='$l2'")
            OcrResult(name, l1, l2, 0, 0, (System.nanoTime() - start) / 1_000_000L)
        } finally {
            release(t)
        }
    }

    private data class Gray(val bytes: ByteArray, val width: Int, val height: Int)

    /** 桌面路径等价：先转灰度（线性），再在灰度域 LANCZOS-3 放大 2x。 */
    private fun prepare(b: Bitmap, upscale: Boolean): Gray {
        val base = bitmapToLuminance(b)
        if (!upscale) return Gray(base, b.width, b.height)
        return Gray(lanczos2x(base, b.width, b.height), b.width * 2, b.height * 2)
    }

    /** 可分离 Lanczos-3 2x 放大（灰度），边缘 clamp；等价桌面 PIL LANCZOS。 */
    private fun lanczos2x(src: ByteArray, w: Int, h: Int): ByteArray {
        val ow = w * 2
        val oh = h * 2
        val tmp = FloatArray(ow * h)
        for (y in 0 until h) {
            val rb = y * w
            val rt = y * ow
            for (x in 0 until ow) {
                val sx = (x + 0.5f) * 0.5f - 0.5f
                val j0 = ceil(sx - 3f).toInt()
                val j1 = floor(sx + 3f).toInt()
                var acc = 0f
                var sum = 0f
                for (j in j0..j1) {
                    val k = lanczosKernel(sx - j)
                    acc += k * (src[rb + j.coerceIn(0, w - 1)].toInt() and 0xff)
                    sum += k
                }
                tmp[rt + x] = acc / sum
            }
        }
        val out = ByteArray(ow * oh)
        for (y in 0 until oh) {
            val sy = (y + 0.5f) * 0.5f - 0.5f
            val j0 = ceil(sy - 3f).toInt()
            val j1 = floor(sy + 3f).toInt()
            val rt = y * ow
            for (x in 0 until ow) {
                var acc = 0f
                var sum = 0f
                for (j in j0..j1) {
                    val k = lanczosKernel(sy - j)
                    acc += k * tmp[j.coerceIn(0, h - 1) * ow + x]
                    sum += k
                }
                out[rt + x] = (acc / sum).roundToInt().coerceIn(0, 255).toByte()
            }
        }
        return out
    }

    /** sinc(x) * sinc(x/3)，|x| >= 3 时为 0。 */
    private fun lanczosKernel(x: Float): Float {
        if (x == 0f) return 1f
        if (abs(x) >= 3f) return 0f
        val px = (Math.PI * x).toDouble()
        val px3 = px / 3.0
        return (sin(px) / px * (sin(px3) / px3)).toFloat()
    }

    private fun bitmapToLuminance(b: Bitmap): ByteArray {
        val w = b.width; val h = b.height
        val pixels = IntArray(w * h)
        b.getPixels(pixels, 0, w, 0, 0, w, h)
        val out = ByteArray(w * h)
        for (i in pixels.indices) {
            val c = pixels[i]
            val r = (c shr 16) and 0xff
            val g = (c shr 8) and 0xff
            val bb = c and 0xff
            out[i] = ((r * 299 + g * 587 + bb * 114) / 1000).toByte()
        }
        return out
    }

    companion object {
        fun fromAssets(context: Context): OcrEngine {
            val dir = File(context.filesDir, "tessdata")
            dir.mkdirs()
            val traineddata = File(dir, "ocrb.traineddata")
            if (!traineddata.isFile || traineddata.length() < 10_000_000L) {
                try {
                    context.assets.open("tessdata/ocrb.traineddata").use { input ->
                        FileOutputStream(traineddata).use { input.copyTo(it) }
                    }
                } catch (_: Throwable) {
                    return Stub
                }
            }
            return TesseractEngine(dir)
        }
        object Stub : OcrEngine {
            override val name = "stub"
            override fun recognize(line1: Bitmap, line2: Bitmap, psm: Int) =
                OcrResult("stub", "", "", 0, 0, 0, "tesseract traineddata missing")
        }
    }
}
