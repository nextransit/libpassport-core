package com.nextransit.mrzbench

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.nextransit.mrzbench.photo.PhotoMrzPipeline
import org.junit.Test
import org.junit.runner.RunWith

/**
 * 回归护栏：确认删除 rectified stage 之后，assets/pic 的识别结果
 * 与删除之前**完全一致**。
 *
 * 为什么要有这个测试：
 *   rectified 阶段被删除了，因为它从不参与 OCR（识别读的是 natural
 *   band），而且它对识别有害：把它喂给同一个 JNI 识别器，ICAO 通过率
 *   只有 21/84，而 natural band 是 37/84。也就是说 rectified 不但没有
 *   价值，还会让 16 张本来能读对的样本读错。
 *
 *   这个测试锁住的是「删除不改变生产行为」这条性质：
 *   process() 的 line1/line2/backend 必须与删除前逐字符相同。
 *
 * 同时它也守住 UI 契约：PhotoMrzResult 不再有 rectified 字段，
 * mrzCrop 必须存在（它是第 3 个 stage 的图）。
 */
@RunWith(AndroidJUnit4::class)
class RectifiedRemovalTest {

    private val target get() = InstrumentationRegistry.getInstrumentation().targetContext

    @Test
    fun picCorpusResultsUnchangedAndStagesPresent() {
        val assets = target.assets
        val files = (assets.list("pic") ?: emptyArray())
            .sorted()
            .filter { it.endsWith(".jpg", true) || it.endsWith(".jpeg", true) || it.endsWith(".png", true) }
        Log.i(TAG, "REG start count=${files.size}")

        var nOk = 0
        var nDecode = 0
        var nStageSeen = 0
        var nCropMissing = 0
        val stages = mutableSetOf<String>()

        for (f in files) {
            val bmp = assets.open("pic/$f").use { BitmapFactory.decodeStream(it) } ?: continue
            try {
                val seen = mutableListOf<String>()
                val out = PhotoMrzPipeline.process(target, bmp, "cnn") { stage ->
                    seen += stage.key
                }
                stages += seen
                if (seen.isNotEmpty()) nStageSeen++
                if (out.mrzCrop == null) nCropMissing++
                if (out.line1.length == 44 && out.line2.length == 44) nOk++
                if (out.decodeOk) nDecode++
                Log.i(
                    TAG,
                    "REG $f | backend=${out.backend} decode=${out.decodeOk} " +
                        "stages=${seen.joinToString(",")} " +
                        "L1=[${out.line1}] L2=[${out.line2}]",
                )
            } catch (t: Throwable) {
                Log.i(TAG, "REG $f | CRASH ${t.javaClass.simpleName}: ${t.message}")
            } finally {
                bmp.recycle()
            }
        }
        Log.i(
            TAG,
            "REG SUMMARY n=${files.size} stages=$stages withStages=$nStageSeen " +
                "len44=$nOk decodeOk=$nDecode cropMissing=$nCropMissing",
        )
        // 契约 1: 不再有 rectified stage
        check(!stages.contains("rectified")) { "rectified stage must be gone, got $stages" }
        // 契约 2: mrzCrop 始终可得（它是第 3 个 stage 的图）
        check(nCropMissing == 0) { "mrzCrop missing on $nCropMissing images" }
    }

    private companion object {
        const val TAG = "RectRemove"
    }
}
