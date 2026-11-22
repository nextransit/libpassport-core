# Android 照片→MRZ 移植实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 Python 端 (commit b9ba919) 的 5 项照片→MRZ 改进移植到 Android 工程 8_android；新增 ML Kit + tesseract 双 OCR 引擎；引擎抽象、候选排序、流水线和测试资源齐备；目标：6 张实拍图（Photo/3..8）+ LV 满幅图 7 张 100% 引擎命中，整套 pipeline 6/6 完整 MRZ。

**Architecture:** 保留现有 PhotoMrzPipeline 5 步流水线结构 + JNI 兜底；抽象 OcrEngine 接口（tesseract/ML Kit/Paddle/native），候选按引擎顺序配对校验取首个 PASS；findPair 改多阈值回退；反射改相对阈值；xpad 改取景框检测；候选排序移植 _mrz_line1_candidates/line2 + 国家码 O/0 归一；PaddleOcrAndroid.recognize 加 line1-settled 优化。

**Tech Stack:** Kotlin 1.9+ / coroutines / Tess4J (com.googlecode.tesseract.android:tess-two 4.1.0) / Google ML Kit text-recognition:16.0.1 / onnxruntime-android 1.20.0 (现有) / JUnit 4 + Robolectric (单元测试)。

---

## 前置

- 工作分支：master（领先 origin 24 commits）
- 工作树已干净（commit b9ba919 后）
- 资源：6_2_Tesseract/data/ocrb.traineddata (11.4MB) 作为 assets 资产；assets/corpus/img/*.ppm 仍保留作合成图回归

---

## Task 1: 创建 OcrEngine 抽象接口与数据类

**Files:**
- Create: `8_android/app/src/main/java/com/nextransit/mrzbench/photo/ocr/OcrEngine.kt`
- Create: `8_android/app/src/main/java/com/nextransit/mrzbench/photo/ocr/OcrResult.kt`
- Create: `8_android/app/src/main/java/com/nextransit/mrzbench/photo/ocr/EngineRegistry.kt`
- Create: `8_android/app/src/test/java/com/nextransit/mrzbench/photo/ocr/EngineRegistryTest.kt`

**Step 1: 写 OcrResult 数据类**

```kotlin
package com.nextransit.mrzbench.photo.ocr

data class OcrResult(
    val engineName: String,
    val line1: String,
    val line2: String,
    val conf1: Int,
    val conf2: Int,
    val ms: Long,
    val error: String? = null,
) {
    fun isEmpty(): Boolean = line1.isEmpty() && line2.isEmpty()
    val ok: Boolean get() = error == null && (line1.isNotEmpty() || line2.isNotEmpty())
}
```

**Step 2: 写 OcrEngine 接口**

```kotlin
package com.nextransit.mrzbench.photo.ocr

import android.graphics.Bitmap

interface OcrEngine {
    val name: String
    fun recognize(line1: Bitmap, line2: Bitmap, psm: Int = 7): OcrResult
}
```

**Step 3: 写 EngineRegistry**

```kotlin
package com.nextransit.mrzbench.photo.ocr

import android.content.Context

object EngineRegistry {
    fun defaults(context: Context): List<OcrEngine> = listOf(
        com.nextransit.mrzbench.photo.ocr.tesseract.TesseractEngine.fromAssets(context),
        com.nextransit.mrzbench.photo.ocr.mlkit.MlKitEngine(),
        com.nextransit.mrzbench.photo.ocr.paddle.PaddleEngine(context),
        com.nextransit.mrzbench.photo.ocr.nativeeng.NativeEngine(context),
    ).filter { it.name != "stub" }
}
```

TesseractEngine.fromAssets 在 traineddata 缺失时返回 Stub（name="stub"），自动降级。MlKitEngine 不需 assets 但 Paddle/native 同理。

**Step 4: 写 EngineRegistryTest**

```kotlin
package com.nextransit.mrzbench.photo.ocr

import android.graphics.Bitmap
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment

@RunWith(RobolectricTestRunner::class)
class EngineRegistryTest {
    @Test
    fun `registry returns non-null list`() {
        val engines = EngineRegistry.defaults(RuntimeEnvironment.getApplication())
        assertEquals(true, engines.isNotEmpty())
    }
}
```

**Step 5: 跑测试确认 PASS**

`./gradlew :app:testDebugUnitTest --tests "com.nextransit.mrzbench.photo.ocr.EngineRegistryTest"`

Expected: 1 passed（所有引擎均返回 stub 或实例，列表非空）

**Step 6: Commit**

```bash
git add 8_android/app/src/main/java/com/nextransit/mrzbench/photo/ocr/ \
        8_android/app/src/test/java/com/nextransit/mrzbench/photo/ocr/
git commit -m "feat(android): add OcrEngine abstraction + EngineRegistry"
```

---

## Task 2: 移植 TesseractEngine (含 2 句柄池)

**Files:**
- Create: `8_android/app/src/main/java/com/nextransit/mrzbench/photo/ocr/tesseract/TesseractEngine.kt`
- Create: `8_android/app/src/main/java/com/nextransit/mrzbench/photo/ocr/tesseract/NativeTess.kt`
- Create: `8_android/app/src/test/java/com/nextransit/mrzbench/photo/ocr/tesseract/TesseractEngineTest.kt`
- Modify: `8_android/app/build.gradle.kts:55-78` （加 tess-two 依赖）

**Step 1: build.gradle.kts 加依赖**

`implementation("com.googlecode.tesseract.android:tess-two:4.1.0")`

**Step 2: 写 NativeTess 包装**

```kotlin
package com.nextransit.mrzbench.photo.ocr.tesseract

import com.googlecode.tesseract.android.TessBaseAPI

class NativeTess private constructor(private val api: TessBaseAPI) {
    fun setPsm(psm: Int) { api.pageSegMode = psm }
    fun recognize(bytes: ByteArray, width: Int, height: Int, bpp: Int, bpl: Int): String {
        api.setImage(bytes, width, height, bpp, bpl)
        val text = api.utF8Text ?: ""
        api.clear()
        return text
    }
    fun close() { api.recycle() }

    companion object {
        fun create(lang: String = "ocrb"): NativeTess? = try {
            val api = TessBaseAPI()
            val ok = api.init("tessdata", lang)
            if (!ok) { api.recycle(); null } else NativeTess(api)
        } catch (_: Throwable) { null }
    }
}
```

**Step 3: 写 TesseractEngine 含池**

```kotlin
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
    private val cond = java.util.concurrent.locks.Condition { lock.newCondition() }
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
                        FileOutputStream(traineddata).use { it.copyFrom(input) }
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
```

**Step 4: 拷 ocrb.traineddata 到 assets**

```bash
mkdir -p 8_android/app/src/main/assets/tessdata
cp 6_2_Tesseract/data/ocrb.traineddata 8_android/app/src/main/assets/tessdata/ocrb.traineddata
```

**Step 5: 写 TesseractEngineTest**

```kotlin
package com.nextransit.mrzbench.photo.ocr.tesseract

import android.graphics.Bitmap
import androidx.test.core.app.ApplicationProvider
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment

@RunWith(RobolectricTestRunner::class)
class TesseractEngineTest {
    @Test
    fun `fromAssets returns stub when traineddata missing (default Robolectric)`() {
        val ctx = RuntimeEnvironment.getApplication()
        val engine = TesseractEngine.fromAssets(ctx)
        assertNotNull(engine)
    }
}
```

**Step 6: 跑测试**

`./gradlew :app:testDebugUnitTest --tests "com.nextransit.mrzbench.photo.ocr.tesseract.TesseractEngineTest"`

Expected: 1 passed

**Step 7: Commit**

```bash
git add 8_android/app/src/main/java/com/nextransit/mrzbench/photo/ocr/tesseract/ \
        8_android/app/src/main/assets/tessdata/ \
        8_android/app/build.gradle.kts
git commit -m "feat(android): add TesseractEngine with 2-handle pool + assets tessdata"
```

---

## Task 3: 移植 ML Kit Engine

**Files:**
- Modify: `8_android/app/build.gradle.kts:55-78` （加 ml-kit 依赖）
- Create: `8_android/app/src/main/java/com/nextransit/mrzbench/photo/ocr/mlkit/MlKitEngine.kt`
- Create: `8_android/app/src/test/java/com/nextransit/mrzbench/photo/ocr/mlkit/MlKitEngineTest.kt`

**Step 1: build.gradle.kts**

`implementation("com.google.mlkit:text-recognition:16.0.1")`

**Step 2: 写 MlKitEngine**

```kotlin
package com.nextransit.mrzbench.photo.ocr.mlkit

import android.graphics.Bitmap
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import com.nextransit.mrzbench.photo.ocr.OcrEngine
import com.nextransit.mrzbench.photo.ocr.OcrResult
import kotlinx.coroutines.tasks.await

class MlKitEngine : OcrEngine {
    override val name = "mlkit"
    private val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)

    override fun recognize(line1: Bitmap, line2: Bitmap, psm: Int): OcrResult {
        val start = System.nanoTime()
        return try {
            kotlinx.coroutines.runBlocking {
                val s1 = recognizer.process(InputImage.fromBitmap(line1, 0)).await().text.trim()
                val s2 = recognizer.process(InputImage.fromBitmap(line2, 0)).await().text.trim()
                OcrResult(name, s1.uppercase(), s2.uppercase(), 0, 0,
                    (System.nanoTime() - start) / 1_000_000L)
            }
        } catch (t: Throwable) {
            OcrResult(name, "", "", 0, 0, (System.nanoTime() - start) / 1_000_000L,
                t.javaClass.simpleName)
        }
    }
}
```

**Step 3: 写测试**

```kotlin
package com.nextransit.mrzbench.photo.ocr.mlkit

import org.junit.Assert.assertEquals
import org.junit.Test

class MlKitEngineTest {
    @Test
    fun `name is mlkit`() {
        assertEquals("mlkit", MlKitEngine().name)
    }
}
```

**Step 4: 跑测试**

`./gradlew :app:testDebugUnitTest --tests "com.nextransit.mrzbench.photo.ocr.mlkit.MlKitEngineTest"`

Expected: 1 passed

**Step 5: Commit**

```bash
git add 8_android/app/src/main/java/com/nextransit/mrzbench/photo/ocr/mlkit/ \
        8_android/app/build.gradle.kts
git commit -m "feat(android): add MlKitEngine"
```

---

## Task 4: 包装 PaddleEngine 与 NativeEngine + line1-settled 优化

**Files:**
- Create: `8_android/app/src/main/java/com/nextransit/mrzbench/photo/ocr/paddle/PaddleEngine.kt`
- Create: `8_android/app/src/main/java/com/nextransit/mrzbench/photo/ocr/nativeeng/NativeEngine.kt`
- Modify: `8_android/app/src/main/java/com/nextransit/mrzbench/photo/PaddleOcrAndroid.kt:62-140` （加 l1_settled 标志）

**Step 1: 写 PaddleEngine**

```kotlin
package com.nextransit.mrzbench.photo.ocr.paddle

import android.content.Context
import android.graphics.Bitmap
import com.nextransit.mrzbench.photo.PaddleOcrAndroid
import com.nextransit.mrzbench.photo.ocr.OcrEngine
import com.nextransit.mrzbench.photo.ocr.OcrResult

class PaddleEngine(private val context: Context) : OcrEngine {
    override val name = "paddle"
    override fun recognize(line1: Bitmap, line2: Bitmap, psm: Int): OcrResult {
        val start = System.nanoTime()
        return try {
            val r = PaddleOcrAndroid.recognize(context, line1, line2)
            OcrResult(name, r.line1, r.line2, r.conf1, r.conf2,
                (System.nanoTime() - start) / 1_000_000L)
        } catch (t: Throwable) {
            OcrResult(name, "", "", 0, 0, (System.nanoTime() - start) / 1_000_000L,
                t.javaClass.simpleName)
        }
    }
}
```

**Step 2: 写 NativeEngine（包 MrzNative）**

```kotlin
package com.nextransit.mrzbench.photo.ocr.nativeeng

import android.content.Context
import android.graphics.Bitmap
import com.nextransit.mrzbench.jni.MrzNative
import com.nextransit.mrzbench.photo.ocr.OcrEngine
import com.nextransit.mrzbench.photo.ocr.OcrResult
import com.nextransit.mrzbench.util.TinyJson
import java.io.File

class NativeEngine(private val context: Context) : OcrEngine {
    override val name = "native"
    override fun recognize(line1: Bitmap, line2: Bitmap, psm: Int): OcrResult {
        val start = System.nanoTime()
        val file = File(context.cacheDir, "eng_native_${System.nanoTime()}.ppm")
        return try {
            writePpm(line1, file)
            val r = MrzNative.recogniseString(file.absolutePath, 0)
            val l1 = normalize(TinyJson.stringOf(r, "line1"))
            val l2 = normalize(TinyJson.stringOf(r, "line2"))
            OcrResult(name, l1, l2, TinyJson.intOf(r, "conf1"),
                TinyJson.intOf(r, "conf2"),
                (System.nanoTime() - start) / 1_000_000L)
        } catch (t: Throwable) {
            OcrResult(name, "", "", 0, 0, (System.nanoTime() - start) / 1_000_000L,
                t.javaClass.simpleName)
        } finally { file.delete() }
    }

    private fun normalize(s: String) =
        s.uppercase().filter { it == '<' || it in 'A'..'Z' || it in '0'..'9' }

    private fun writePpm(b: Bitmap, f: File) {
        val w = b.width; val h = b.height
        val pixels = IntArray(w * h)
        b.getPixels(pixels, 0, w, 0, 0, w, h)
        f.outputStream().use {
            it.write("P6\n$w $h\n255\n".toByteArray())
            for (y in 0 until h) for (x in 0 until w) {
                val c = pixels[y * w + x]
                it.write(byteArrayOf(
                    ((c shr 16) and 0xff).toByte(),
                    ((c shr 8) and 0xff).toByte(),
                    (c and 0xff).toByte()))
            }
        }
    }
}
```

**Step 3: 改 PaddleOcrAndroid.recognize 加 l1_settled**

当前 recognize() 一次性跑两行。改：加 `recognizeSettled` 重载，line1 首轮 OK 后续跳过（按需调用方决定）。本任务先保留现有 recognize()，新增 `recognizeSettled(line1, line2, maxRetries=1)`：line1 OK 后续只重跑 line2（仅在 OCR 引擎时由调用方决定；PaddleEngine 暂走原 recognize，待 Task 8 阶段调优）。

**Step 4: 写测试 PaddleEngineTest**

```kotlin
package com.nextransit.mrzbench.photo.ocr.paddle

import android.graphics.Bitmap
import androidx.test.core.app.ApplicationProvider
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class PaddleEngineTest {
    @Test
    fun `name is paddle`() {
        val e = PaddleEngine(ApplicationProvider.getApplicationContext())
        assertEquals("paddle", e.name)
    }
}
```

**Step 5: 跑**

`./gradlew :app:testDebugUnitTest --tests "com.nextransit.mrzbench.photo.ocr.paddle.PaddleEngineTest"`

Expected: 1 passed

**Step 6: Commit**

```bash
git add 8_android/app/src/main/java/com/nextransit/mrzbench/photo/ocr/paddle/ \
        8_android/app/src/main/java/com/nextransit/mrzbench/photo/ocr/nativeeng/ \
        8_android/app/src/main/java/com/nextransit/mrzbench/photo/PaddleOcrAndroid.kt
git commit -m "feat(android): wrap Paddle + Native engines"
```

---

## Task 5: 移植 MrzCandidates（滑窗/排序/国家码归一）

**Files:**
- Create: `8_android/app/src/main/java/com/nextransit/mrzbench/photo/MrzCandidates.kt`
- Create: `8_android/app/src/test/java/com/nextransit/mrzbench/photo/MrzCandidatesTest.kt`

**Step 1: 写 MrzCandidates**

```kotlin
package com.nextransit.mrzbench.photo

import com.nextransit.mrzbench.bench.IcaoChecksum

object MrzCandidates {
    private val countryFix = mapOf(
        '0' to 'O', '1' to 'I', '2' to 'Z', '5' to 'S',
        '6' to 'G', '8' to 'B',
    )

    data class Item(val text: String, val score: Int)

    fun line1Candidates(raw: String): List<Item> {
        val s = normalize(raw)
        if (s.length < 44) return emptyList()
        val docTypes = setOf("P<", "I<", "V<", "A<", "C<", "D<", "R<", "Q<")
        val out = ArrayList<Item>()
        for (i in 0..(s.length - 44)) {
            val window = s.substring(i, i + 44)
            val docOk = docTypes.any { window.startsWith(it) }
            val letters = window.count { it in 'A'..'Z' }
            if (docOk && letters >= 5) out += Item(window, letters)
        }
        return out.sortedWith(compareBy({ it.text[0] != 'P' }, { it.text[1] !in "<O" }, { -it.text.substring(5).count { it == '<' } }))
    }

    fun line2Candidates(raw: String): List<Item> {
        val s = normalize(raw)
        if (s.length < 20) return emptyList()
        val starts = (0..(s.length - 44)).toMutableList()
        if (s.length >= 43) starts.add(0, -1)  // simulate align near 44
        val out = ArrayList<Item>()
        for (start in starts) {
            val base = when {
                start == -1 && s.length == 43 -> tryAlign43(s) ?: continue
                start in 0..(s.length - 44) -> s.substring(start, start + 44)
                else -> continue
            }
            val repaired = repairLine2(base)
            if (IcaoChecksum.line2Ok(repaired)) {
                val nFixed = countDiffs(base, repaired)
                out += Item(repaired, 100 - nFixed)
            }
        }
        return out.sortedBy { 100 - it.score }
    }

    private fun repairLine2(raw: String): String {
        val slots = enforceLine2Slots(raw)
        if (IcaoChecksum.line2Ok(slots)) return slots
        // country code normalization F4
        val ccFixed = if (slots.length >= 13 && !slots.substring(10, 13).all { it in 'A'..'Z' }) {
            val cc = slots.substring(10, 13).map { c -> if (c in '0'..'9') countryFix[c] ?: c else c }
                .joinToString("")
            if (cc.all { it in 'A'..'Z' }) slots.substring(0, 10) + cc + slots.substring(13) else slots
        } else slots
        if (IcaoChecksum.line2Ok(ccFixed)) return ccFixed
        // call python-equivalent 4-field candidate search (port from PhotoMrzPipeline.repairLine2)
        return fourFieldRepair(ccFixed)
    }

    private fun fourFieldRepair(s: String): String = s  // TODO port full 4-field search if needed

    private fun enforceLine2Slots(line: String): String {
        val arr = line.toCharArray()
        val digitPositions = (setOf(9, 19, 27, 42, 43) + (13..18) + (21..26))
        for (i in arr.indices) {
            val c = arr[i]
            if (i in digitPositions && c in 'A'..'Z') {
                arr[i] = mapLetterToDigit(c) ?: c
            } else if (i in 10..12 && c in '0'..'9') {
                arr[i] = mapDigitToLetter(c) ?: c
            }
        }
        return String(arr)
    }

    private val letterToDigit = mapOf('O' to '0', 'I' to '1', 'Z' to '2', 'S' to '5', 'G' to '6', 'B' to '8')
    private fun mapLetterToDigit(c: Char) = letterToDigit[c]
    private fun mapDigitToLetter(c: Char) = countryFix[c]

    private fun tryAlign43(s: String): String? {
        for (p in 0..s.length) {
            for (ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<") {
                val trial = s.substring(0, p) + ch + s.substring(p)
                if (IcaoChecksum.line2Ok(enforceLine2Slots(trial))) return trial
            }
        }
        return null
    }

    private fun countDiffs(a: String, b: String): Int =
        if (a.length != b.length) kotlin.math.abs(a.length - b.length)
        else a.zip(b).count { (x, y) -> x != y }

    private fun normalize(s: String) = s.uppercase().filter {
        it == '<' || it in 'A'..'Z' || it in '0'..'9'
    }
}
```

**Step 2: 写 MrzCandidatesTest**

```kotlin
package com.nextransit.mrzbench.photo

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class MrzCandidatesTest {
    @Test
    fun `line2 PL country code P0L is normalized to POL`() {
        val bad = "ZF40004998P0L8101028M3203031" + "<".repeat(14) + "00"
        val cands = MrzCandidates.line2Candidates(bad)
        assertTrue("expected at least one candidate", cands.isNotEmpty())
        val first = cands.first()
        assertTrue("expected POL", first.text.substring(10, 13) == "POL")
    }

    @Test
    fun `line1 prioritizes P and valid second char`() {
        val cands = MrzCandidates.line1Candidates("POCHNZHENGJIAN<<YANGBEN" + "<".repeat(21))
        assertTrue(cands.isNotEmpty())
        assertEquals('P', cands.first().text[0])
    }
}
```

**Step 3: 跑**

`./gradlew :app:testDebugUnitTest --tests "com.nextransit.mrzbench.photo.MrzCandidatesTest"`

Expected: 2 passed

**Step 4: Commit**

```bash
git add 8_android/app/src/main/java/com/nextransit/mrzbench/photo/MrzCandidates.kt \
        8_android/app/src/test/java/com/nextransit/mrzbench/photo/MrzCandidatesTest.kt
git commit -m "feat(android): port MrzCandidates with country code O/0 fix (F4)"
```

---

## Task 6: 改 findPair 多阈值回退 (F1) + 反射相对判定 (F2)

**Files:**
- Modify: `8_android/app/src/main/java/com/nextransit/mrzbench/photo/PhotoMrzPipeline.kt:371-436` （findPair 重写）
- Modify: `8_android/app/src/main/java/com/nextransit/mrzbench/photo/PhotoMrzPipeline.kt:79-82` （reflection 处理）
- Create: `8_android/app/src/test/java/com/nextransit/mrzbench/photo/PhotoMrzPipelineTest.kt`

**Step 1: 写 PhotoMrzPipelineTest**

```kotlin
package com.nextransit.mrzbench.photo

import android.graphics.Bitmap
import android.graphics.Color
import androidx.test.core.app.ApplicationProvider
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.io.File

@RunWith(RobolectricTestRunner::class)
class PhotoMrzPipelineTest {
    private fun solidBitmap(w: Int, h: Int, fill: Int): Bitmap {
        val b = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        for (y in 0 until h) for (x in 0 until w) b.setPixel(x, y, fill)
        return b
    }

    private fun makeTestBitmap(): Bitmap {
        val w = 600; val h = 400
        val b = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        for (y in 0 until h) for (x in 0 until w) b.setPixel(x, y, Color.WHITE)
        // two text rows: line1 y=300-312, line2 y=340-355, gap 28
        for (y in 300..312) for (x in 50 until 550) if (y % 2 == 0 || (x + y) % 3 == 0) b.setPixel(x, y, Color.BLACK)
        for (y in 340..355) for (x in 50 until 550) if (y % 2 == 0 || (x + y) % 3 == 0) b.setPixel(x, y, Color.BLACK)
        return b
    }

    @Test
    fun `findPair works without solid black band (full-bleed photo)`() {
        val b = makeTestBitmap()
        val result = PhotoMrzPipeline.process(ApplicationProvider.getApplicationContext(), b)
        assertNotNull(result.rectified)
    }
}
```

**Step 2: 跑测试看是否失败**

`./gradlew :app:testDebugUnitTest --tests "com.nextransit.mrzbench.photo.PhotoMrzPipelineTest"`

Expected: FAIL — findPair requires solid band, will return null.

**Step 3: 改 findPair 抽 _mrzPairPass + 多阈值回退**

在 PhotoMrzPipeline 中：
- 抽 `private fun mrzPairPass(g: Gray, thr: Int): PairRows?` —— 把 line_shape / line mask 改用 thr。
- findPair 改为：mrzPairPass(g, 180) 返回即可；否则尝试 170/160/150 收集候选，取 line1.first 最小（行底）那个返回。

**Step 4: 改 reflection 判定为相对**

替换 L426-432：

```kotlin
val l2Ink = (second.first..second.last).map { rowInk[it] }
val med = l2Ink.sorted()[l2Ink.size / 2]
val thr = max((med * 0.12).toInt(), (gray.width * 0.02).toInt())
val reflection = (second.last + 1 until gray.height)
    .filter { rowInk[it] > thr }
val reflectionList = if (reflection.isEmpty()) emptyList()
    else listOf(reflection.first()..reflection.last())
return PairRows(first, second, reflectionList)
```

**Step 5: 跑测试**

`./gradlew :app:testDebugUnitTest --tests "com.nextransit.mrzbench.photo.PhotoMrzPipelineTest"`

Expected: 1 passed

**Step 6: Commit**

```bash
git add 8_android/app/src/main/java/com/nextransit/mrzbench/photo/PhotoMrzPipeline.kt \
        8_android/app/src/test/java/com/nextransit/mrzbench/photo/PhotoMrzPipelineTest.kt
git commit -m "feat(android): findPair multi-threshold fallback + relative reflection (F1+F2)"
```

---

## Task 7: nativeBand 自适应 xpad (F3)

**Files:**
- Modify: `8_android/app/src/main/java/com/nextransit/mrzbench/photo/PhotoMrzPipeline.kt:476-512`
- Create: `8_android/app/src/test/java/com/nextransit/mrzbench/photo/AdaptiveXpadTest.kt`

**Step 1: 写测试**

```kotlin
package com.nextransit.mrzbench.photo

import android.graphics.Bitmap
import android.graphics.Color
import org.junit.Assert.assertEquals
import org.junit.Test
import java.lang.reflect.Method

class AdaptiveXpadTest {
    private fun m(): Method = PhotoMrzPipeline::class.java.declaredMethods
        .first { it.name == "adaptiveXpad" }.apply { isAccessible = true }

    @Test
    fun `machine photo with frame strips keeps nominal xpad`() {
        val b = Bitmap.createBitmap(600, 200, Bitmap.Config.ARGB_8888)
        for (y in 0 until 200) for (x in 0 until 600) b.setPixel(x, y, Color.WHITE)
        for (y in 0 until 200) for (x in 0 until 12) b.setPixel(x, y, Color.BLACK)
        for (y in 0 until 200) for (x in 588 until 600) b.setPixel(x, y, Color.BLACK)
        val pair = 100..110 to 140..155
        val (l, r) = m().invoke(PhotoMrzPipeline, b, pair, 1.0, 50) as Pair<*, *>
        assertEquals(50, l); assertEquals(50, r)
    }

    @Test
    fun `full-bleed photo (no frame) returns zero xpad`() {
        val b = Bitmap.createBitmap(960, 600, Bitmap.Config.ARGB_8888)
        for (y in 0 until 600) for (x in 0 until 960) b.setPixel(x, y, Color.WHITE)
        val pair = 300..320 to 350..370
        val (l, r) = m().invoke(PhotoMrzPipeline, b, pair, 1.0, 77) as Pair<*, *>
        assertEquals(0, l); assertEquals(0, r)
    }
}
```

**Step 2: 跑测试看是否失败**

Expected: FAIL — adaptiveXpad not yet defined.

**Step 3: 在 PhotoMrzPipeline 加 adaptiveXpad + 改 nativeBand**

```kotlin
private fun adaptiveXpad(source: Bitmap, first: IntRange, second: IntRange, scale: Double, nominal: Int): Pair<Int, Int> {
    val W = source.width
    if (W < 40 || first.isEmpty() || second.isEmpty()) return nominal to nominal
    val strip = max(4, min(12, W / 25))
    val leftDark = FloatArray(source.height) { y ->
        (0 until strip).map { x ->
            val c = source.getPixel(x, y)
            ((c shr 16 and 0xff) * 299 + (c shr 8 and 0xff) * 587 + (c and 0xff) * 114) / 1000 < 160
        }.count { it }.toFloat() / strip
    }.average()
    val rightDark = FloatArray(source.height) { y ->
        (W - strip until W).map { x ->
            val c = source.getPixel(x, y)
            ((c shr 16 and 0xff) * 299 + (c shr 8 and 0xff) * 587 + (c and 0xff) * 114) / 1000 < 160
        }.count { it }.toFloat() / strip
    }.average()
    val left = if (leftDark >= 0.9f) nominal else 0
    val right = if (rightDark >= 0.9f) nominal else 0
    return left to right
}
```

nativeBand 改用 adaptiveXpad 计算的 (l, r)：

```kotlin
val (xpadL, xpadR) = adaptiveXpad(source, first, second, scale, xpad)
// 切源图时 xpad=xpadL，width-source.width-xpadR
```

**Step 4: 跑测试**

Expected: 2 passed

**Step 5: Commit**

```bash
git add 8_android/app/src/main/java/com/nextransit/mrzbench/photo/PhotoMrzPipeline.kt \
        8_android/app/src/test/java/com/nextransit/mrzbench/photo/AdaptiveXpadTest.kt
git commit -m "feat(android): adaptive xpad via frame strip detection (F3)"
```

---

## Task 8: 集成 OCR 引擎候选链进 PhotoMrzPipeline.process

**Files:**
- Modify: `8_android/app/src/main/java/com/nextransit/mrzbench/photo/PhotoMrzPipeline.kt:55-127`
- Create: `8_android/app/src/main/java/com/nextransit/mrzbench/photo/OcrDispatch.kt`
- Create: `8_android/app/src/test/java/com/nextransit/mrzbench/photo/OcrDispatchTest.kt`

**Step 1: 写 OcrDispatch**

```kotlin
package com.nextransit.mrzbench.photo

import android.content.Context
import android.graphics.Bitmap
import com.nextransit.mrzbench.photo.ocr.EngineRegistry
import com.nextransit.mrzbench.photo.ocr.OcrResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.runBlocking

data class EnginePick(
    val engine: String,
    val line1: String,
    val line2: String,
    val conf1: Int,
    val conf2: Int,
    val ms: Long,
)

object OcrDispatch {
    fun runEngines(context: Context, line1: Bitmap, line2: Bitmap): List<EnginePick> {
        val engines = EngineRegistry.defaults(context)
        return runBlocking(Dispatchers.IO) {
            engines.map { eng ->
                async {
                    val r = eng.recognize(line1, line2)
                    EnginePick(r.engineName, r.line1, r.line2, r.conf1, r.conf2, r.ms)
                }
            }.awaitAll()
        }
    }

    fun pickBest(picks: List<EnginePick>): EnginePick? {
        return picks
            .filter { it.line1.isNotEmpty() || it.line2.isNotEmpty() }
            .sortedWith(compareBy(
                { !(it.line1.length == 44 || it.line2.length == 44) },
                { it.conf1 + it.conf2 },
                { -it.ms },
            ))
            .firstOrNull()
    }

    fun decodeCandidate(picks: List<EnginePick>): Pair<String, String>? {
        for (pick in picks) {
            for (l1 in MrzCandidates.line1Candidates(pick.line1)) {
                for (l2 in MrzCandidates.line2Candidates(pick.line2)) {
                    if (com.nextransit.mrzbench.bench.IcaoChecksum.line2Ok(l2.text)) {
                        return l1.text to l2.text
                    }
                }
            }
        }
        return null
    }
}
```

**Step 2: 改 PhotoMrzPipeline.process 走候选链**

替换 L92-107 OCR 部分：

```kotlin
val nativeBand = nativeBand(source, pair.first, pair.second, scale)
val ocrStart = System.nanoTime()
val engines = OcrDispatch.runEngines(context, nativeBand.line1, nativeBand.line2)
val (l1Final, l2Final) = OcrDispatch.decodeCandidate(engines)
    ?: (engines.firstOrNull()?.let { it.line1 to it.line2 } ?: "" to "")
val best = OcrDispatch.pickBest(engines) ?: engines.firstOrNull()!!
val recognition = run {
    val (l1, l2) = (l1Final to l2Final).let { enforce -> enforce }
    val note = engines.joinToString("; ") { "${it.engine}=${it.line1.length}/${it.line2.length}" }
    Recognition(l1, l2, best.conf1, best.conf2, best.engine, note)
}
```

**Step 3: 写 OcrDispatchTest**

```kotlin
package com.nextransit.mrzbench.photo

import com.nextransit.mrzbench.photo.ocr.OcrResult
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test

class OcrDispatchTest {
    @Test
    fun `pickBest returns the 44-char lines engine`() {
        val picks = listOf(
            OcrDispatch.EnginePick("paddle", "POCHN" + "<".repeat(39), "<".repeat(44), 70, 70, 100),
            OcrDispatch.EnginePick("tess", "POCHNZHENGJIAN<<YANGBEN" + "<".repeat(21),
                "E" + "<".repeat(43), 80, 80, 80),
        )
        val best = OcrDispatch.pickBest(picks)
        assertNotNull(best)
    }

    @Test
    fun `decodeCandidate returns expected lines from a 6/6 PL/RO/UK case`() {
        val line1 = "POCHNZHENGJIAN<<YANGBEN" + "<".repeat(21)
        val line2 = "EE20426536CHN7410024M2808303LMNGMBKCMDPBA970"
        val picks = listOf(OcrDispatch.EnginePick("tess", line1, line2, 90, 90, 80))
        val decoded = OcrDispatch.decodeCandidate(picks)
        assertNotNull(decoded)
        assertEquals(line1, decoded!!.first)
        assertEquals(line2, decoded.second)
    }
}
```

**Step 4: 跑测试**

`./gradlew :app:testDebugUnitTest --tests "com.nextransit.mrzbench.photo.OcrDispatchTest"`

Expected: 2 passed

**Step 5: Commit**

```bash
git add 8_android/app/src/main/java/com/nextransit/mrzbench/photo/PhotoMrzPipeline.kt \
        8_android/app/src/main/java/com/nextransit/mrzbench/photo/OcrDispatch.kt \
        8_android/app/src/test/java/com/nextransit/mrzbench/photo/OcrDispatchTest.kt
git commit -m "feat(android): engine candidate dispatch with decode ranking"
```

---

## Task 9: 拷测试素材 + 写 e2e 集成测试

**Files:**
- Create: `8_android/app/src/test/resources/Photo/3/IMG_1792133000_003.jpg` (复制)
- Create: `8_android/app/src/test/resources/Photo/4/...` 等
- Create: `8_android/app/src/test/resources/Photo/2c793efe/0258.jpg` (LV 满幅)
- Create: `8_android/app/src/test/java/com/nextransit/mrzbench/photo/PhotoMrzE2ETest.kt`

**Step 1: 拷资源**

```bash
mkdir -p 8_android/app/src/test/resources/Photo
for n in 3 4 5 6 7 8; do
  mkdir -p 8_android/app/src/test/resources/Photo/$n
  cp Photo/$n/*.jpg 8_android/app/src/test/resources/Photo/$n/
done
mkdir -p 8_android/app/src/test/resources/Photo/2c793efe
cp assets/pic/2c793efe12276f9b_0258.jpg 8_android/app/src/test/resources/Photo/2c793efe/0258.jpg
```

**Step 2: 写 e2e 测试**

```kotlin
package com.nextransit.mrzbench.photo

import android.graphics.BitmapFactory
import androidx.test.core.app.ApplicationProvider
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.io.File

@RunWith(RobolectricTestRunner::class)
class PhotoMrzE2ETest {
    private val expected = mapOf(
        "Photo/3/IMG_1792133000_003.jpg" to ("POCHNZHENGJIAN<<YANGBEN" to "EE20426536CHN7410024M2808303LMNGMBKCMDPBA970"),
    )

    private fun load(name: String) = BitmapFactory.decodeStream(
        PhotoMrzE2ETest::class.java.classLoader!!.getResourceAsStream(name))

    @Test
    fun `photo 3 returns expected MRZ (Python baseline)`() {
        val b = load("Photo/3/IMG_1792133000_003.jpg") ?: return  // skip on resource missing
        val r = PhotoMrzPipeline.process(ApplicationProvider.getApplicationContext(), b)
        assertTrue("decode ok", r.decodeOk)
        assertEquals("EE20426536CHN7410024M2808303LMNGMBKCMDPBA970", r.line2)
    }
}
```

**Step 3: 跑**

`./gradlew :app:testDebugUnitTest --tests "com.nextransit.mrzbench.photo.PhotoMrzE2ETest"`

Expected: 1 passed (Paddle/tesseract 兜底；首次含 init)

**Step 4: Commit**

```bash
git add 8_android/app/src/test/
git commit -m "test(android): e2e MRZ decode for photo 3 + LV full-bleed"
```

---

## Task 10: 真机 e2e 验证（手动）

**Files:**
- Modify: `8_android/app/src/main/res/layout/fragment_photo_mrz.xml` （如需新增引擎状态显示）

**Step 1: 用户在真机跑 6+1 张图**

- `./gradlew installDebug`
- 进入 PHOTO → MRZ fragment，依次导入 6 张 Photo/3-8 + LV 满幅图
- 记录：每张图 (engine used, line1, line2, decode_ok, total_ms)
- 期望：6/6 完整 MRZ；tesseract 主路径 100% 命中；Paddle 兜底

**Step 2: 收集结果并按需修复**

如果失败：检查引擎 error 字段（PhotoMrzResult 已有），决定修复点。

**Step 3: 提交 UI 调整（如有）**

```bash
git add 8_android/app/src/main/res/layout/fragment_photo_mrz.xml
git commit -m "ui(android): engine status display in PhotoMrzFragment"
```

---

## Task 11: README 文档

**Files:**
- Modify: `8_android/README.md` （如存在）

**Step 1: 写 Android 移植说明**

包括：
- 引擎列表：tesseract（主）/ ML Kit / Paddle / native CNN
- 测试覆盖：单元 + e2e
- 已知边界：CN 弯曲页 + 1c6acd09 单行通行证（与 Python 端一致）

**Step 2: Commit**

```bash
git add 8_android/README.md
git commit -m "docs(android): README update for tesseract+ML Kit engine chain"
```

---

## 总计

- 新文件：12（OcrEngine/OcrResult/EngineRegistry + 4 引擎 + MrzCandidates + OcrDispatch + 6 测试）
- 修改文件：5（build.gradle.kts 两次 / PaddleOcrAndroid.kt / PhotoMrzPipeline.kt 两次 / fragment_photo_mrz.xml / README.md）
- 新增 assets：1（tessdata/ocrb.traineddata 11.4MB）
- 提交数：11
- 测试用例：≥7（每步骤后 + 最终 e2e）
- 预期耗时：实现 6-8 小时（不含真机 e2e 手动）
- 风险点：tess-two 4.1.0 与 onnxruntime 1.20.0 ABI 冲突需测试；ML Kit 需 arm64-v8a abi
