package com.nextransit.mrzbench.photo.ocr.tesseract

import android.content.Context
import android.graphics.Bitmap
import com.nextransit.mrzbench.photo.ocr.OcrEngine
import com.nextransit.mrzbench.photo.ocr.OcrResult
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

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
                    val t = NativeTess.create()
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

    override fun recognize(line1: Bitmap, line2: Bitmap, psm: Int): OcrResult {
        val start = System.nanoTime()
        val t = acquire() ?: return OcrResult(name, "", "", 0, 0, 0, "init failed")
        return try {
            t.setPsm(psm)
            val bytes1 = bitmapToLuminance(line1)
            val bytes2 = bitmapToLuminance(line2)
            val l1 = t.recognize(bytes1, line1.width, line1.height, 1, line1.width).trim()
            val l2 = t.recognize(bytes2, line2.width, line2.height, 1, line2.width).trim()
            OcrResult(name, l1, l2, 0, 0, (System.nanoTime() - start) / 1_000_000L)
        } finally {
            release(t)
        }
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
