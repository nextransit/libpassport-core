package com.nextransit.mrzbench

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer
import kotlin.math.abs

/**
 * Indra OCR 五个模型在真机（arm64-v8a）上的数值 + 耗时探针。
 *
 * 输入张量由host 端Python 用 indra_ocr 的真实预处理生成
 * （tools/indra_probe 目录下的 bin 文件），推送到设备后原样喂给 ORT。
 * 这样比对的只有「ORT 在 ARM NEON 上的算子结果」，不掺入 cv2 预处理的差异。
 *
 * 期望值见 tools/indra_probe/probe_expected.json（host 端 ORT 1.30 / x86_64）。
 */
@RunWith(AndroidJUnit4::class)
class IndraOnnxProbeTest {

    private val context get() = InstrumentationRegistry.getInstrumentation().targetContext
    private val tag get() = "IndraProbe"

    private fun modelPath(name: String): String {
        // 1) 外部推送目录（设备 FUSE 层允许 App 自己的 uid 读）
        val ext = File("/sdcard/Android/data/com.nextransit.mrzbench/files/models/indra/$name")
        if (ext.isFile && ext.length() > 1000) return ext.absolutePath
        // 2) App 内部目录（从 assets 复制）
        val target = File(context.filesDir, "models/indra/$name")
        if (target.isFile && target.length() > 1000) return target.absolutePath
        target.parentFile?.mkdirs()
        context.assets.open("models/indra/$name").use { input ->
            target.outputStream().use { output -> input.copyTo(output) }
        }
        return target.absolutePath
    }

    private fun probeDir(): File {
        val roots = listOf(
            File(context.filesDir, "indra_probe"),
            File("/sdcard/Android/data/com.nextransit.mrzbench/files/indra_probe"),
        )
        for (r in roots) if (r.isDirectory) return r
        return roots.first().also { it.mkdirs() }
    }

    private fun floats(name: String): FloatArray {
        val f = File(probeDir(), "$name.bin")
        if (!f.isFile) { Log.i(tag, "FAIL missing $name (${f.absolutePath})"); return FloatArray(0) }
        val bb = ByteBuffer.wrap(f.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
        val out = FloatArray(f.length().toInt() / 4)
        bb.asFloatBuffer().get(out)
        return out
    }

    private fun longs(name: String): LongArray {
        val f = File(probeDir(), "$name.bin")
        if (!f.isFile) { Log.i(tag, "FAIL missing $name"); return LongArray(0) }
        val bb = ByteBuffer.wrap(f.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
        val out = LongArray(f.length().toInt() / 8)
        bb.asLongBuffer().get(out)
        return out
    }

    private var pass = 0
    private var fail = 0

    private fun check(name: String, expected: Double, actual: Double, tol: Double) {
        val d = abs(expected - actual)
        val rel = if (expected != 0.0) d / abs(expected) else d
        val ok = d <= tol || rel <= tol
        if (ok) pass++ else fail++
        Log.i(tag, "%s %s expected=%.8f actual=%.8f diff=%.3e rel=%.3e".format(
            if (ok) "PASS" else "FAIL", name, expected, actual, d, rel))
    }

    private fun checkVec(name: String, expected: DoubleArray, actual: DoubleArray, tol: Double) {
        if (expected.size != actual.size) {
            fail++
            Log.i(tag, "FAIL $name size exp=${expected.size} act=${actual.size}")
            return
        }
        var worstIdx = -1
        var worst = 0.0
        for (i in expected.indices) {
            val d = abs(expected[i] - actual[i])
            val rel = if (expected[i] != 0.0) d / abs(expected[i]) else d
            val m = minOf(d, rel)
            if (m > worst) { worst = m; worstIdx = i }
        }
        if (worst <= tol) {
            pass++
            Log.i(tag, "PASS $name n=${expected.size} worst=%.3e@%d".format(worst, worstIdx))
        } else {
            fail++
            Log.i(tag, "FAIL $name worst=%.3e@%d exp=%.8f act=%.8f".format(
                worst, worstIdx, expected[worstIdx], actual[worstIdx]))
        }
    }

    /** ORT 输出的一维平坦 float 缓冲（按 shape 自行索引）。 */
    private fun flat(r: ai.onnxruntime.OrtSession.Result, idx: Int): FloatArray {
        val t = r[idx] as OnnxTensor
        val buf = t.floatBuffer
        val out = FloatArray(buf.remaining())
        buf.duplicate().get(out)
        return out
    }

    /** ORT 输出的一维平坦 long 缓冲。 */
    private fun flatLong(r: ai.onnxruntime.OrtSession.Result, idx: Int): LongArray {
        val t = r[idx] as OnnxTensor
        val buf = t.longBuffer
        val out = LongArray(buf.remaining())
        buf.duplicate().get(out)
        return out
    }

    @Test
    fun probeIndraModelsOnDevice() {
        val env = OrtEnvironment.getEnvironment()
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(2)
            setInterOpNumThreads(1)
            setCPUArenaAllocator(false)
        }

        // ---------- DET ----------
        run {
            val x = floats("probe_det_x")
            if (x.isEmpty()) return@run
            val s = env.createSession(modelPath("det.onnx"), opts)
            var t = 0L; var cold = 0L
            var sum = 0.0; var mx = 0.0
            val first8 = DoubleArray(8)
            repeat(3) { iter ->
                val t0 = System.nanoTime()
                OnnxTensor.createTensor(
                    env, FloatBuffer.wrap(x), longArrayOf(1L, 1L, 768L, 1024L)
                ).use { inp ->
                    s.run(mapOf("x" to inp)).use { r ->
                        // det 输出 [1, 3, 384, 512]：通道 0 = core logit，
                        // 通道 1/2 = 上下距离场。只统计通道 0。
                        val all = flat(r, 0)
                        val plane = 384 * 512
                        sum = 0.0; mx = 0.0
                        for (i in 0 until plane) {
                            val v = all[i].toDouble()
                            sum += v; if (v > mx) mx = v
                        }
                        for (i in 0 until 8) first8[i] = all[i].toDouble()
                    }
                }
                val el = (System.nanoTime() - t0) / 1_000_000
                if (iter == 0) cold = el else if (iter == 1) t = el
            }
            Log.i(tag, "DET cold=${cold}ms warm=${t}ms")
            check("det.ch0.sum", -2187334.75, sum, 1e-4)
            check("det.ch0.max", 6.7140703201293945, mx, 1e-4)
            checkVec("det.ch0.first8",
                doubleArrayOf(-10.259490966796875, -10.50318717956543, -11.67933177947998,
                    -12.07832145690918, -12.77878189086914, -13.155778884887695,
                    -13.332367897033691, -13.478971481323242),
                first8, 1e-4)
        }

        // ---------- DET on MRZ band（[1,1,96,1024]，面积只有整页的 1/8）----------
        run {
            val x = floats("probe_band_det_x")
            if (x.isEmpty()) return@run
            val s = env.createSession(modelPath("det.onnx"), opts)
            var t = 0L; var cold = 0L
            var sum = 0.0; var mx = 0.0
            val first8 = DoubleArray(8)
            repeat(5) { iter ->
                val t0 = System.nanoTime()
                OnnxTensor.createTensor(
                    env, FloatBuffer.wrap(x), longArrayOf(1L, 1L, 96L, 1024L)
                ).use { inp ->
                    s.run(mapOf("x" to inp)).use { r ->
                        val all = flat(r, 0)
                        val plane = 48 * 512
                        sum = 0.0; mx = 0.0
                        for (i in 0 until plane) {
                            val v = all[i].toDouble()
                            sum += v; if (v > mx) mx = v
                        }
                        for (i in 0 until 8) first8[i] = all[i].toDouble()
                    }
                }
                val el = (System.nanoTime() - t0) / 1_000_000
                if (iter == 0) cold = el else if (iter == 1) t = el
            }
            Log.i(tag, "BAND-DET cold=${cold}ms warm=${t}ms")
            check("band.det.ch0.sum", -170348.1875, sum, 1e-4)
            check("band.det.ch0.max", 7.581881046295166, mx, 1e-4)
            checkVec("band.det.ch0.first8",
                doubleArrayOf(-9.59749984741211, -9.659344673156738, -9.752752304077148,
                    -9.835411071777344, -10.13741683959961, -10.172911643981934,
                    -10.269953727722168, -10.226320266723633),
                first8, 1e-4)
        }

        // ---------- REC ----------
        run {
            val x = floats("probe_rec_x")
            val w = longs("probe_rec_w")
            if (x.isEmpty()) return@run
            val s = env.createSession(modelPath("rec.onnx"), opts)
            var t = 0L; var cold = 0L
            var ids = ""
            var conf = 0.0
            repeat(3) { iter ->
                val t0 = System.nanoTime()
                OnnxTensor.createTensor(
                    env, FloatBuffer.wrap(x), longArrayOf(1L, 1L, 96L, 360L)
                ).use { inp ->
                    OnnxTensor.createTensor(env, LongBuffer.wrap(w), longArrayOf(1L)).use { wid ->
                        s.run(mapOf("x" to inp, "widths" to wid)).use { r ->
                            val idArr = flatLong(r, 0)
                            val cfArr = flat(r, 1)
                            val lnArr = flatLong(r, 2)
                            val n = lnArr[0].toInt()
                            val sb = StringBuilder(); var cs = 0.0; var cc = 0
                            var prev = -1L
                            for (i in 0 until n) {
                                val id = idArr[i]
                                if (id != 0L && id != prev) {
                                    sb.append(id).append(","); cs += cfArr[i].toDouble(); cc++
                                }
                                prev = id
                            }
                            ids = sb.toString(); conf = if (cc > 0) cs / cc else 0.0
                        }
                    }
                }
                val el = (System.nanoTime() - t0) / 1_000_000
                if (iter == 0) cold = el else if (iter == 1) t = el
            }
            Log.i(tag, "REC cold=${cold}ms warm=${t}ms ids=$ids conf=$conf")
            val first = ids.split(",").filter { it.isNotEmpty() }
            if (first.isNotEmpty()) check("rec.ids[0]", 81.0, first[0].toDouble(), 1e-9)
            check("rec.conf", 0.5250513553619385, conf, 1e-4)
        }

        // ---------- SCRIPT ----------
        run {
            val x = floats("probe_script_x")
            val w = longs("probe_script_w")
            if (x.isEmpty()) return@run
            val s = env.createSession(modelPath("script.onnx"), opts)
            var t = 0L; var cold = 0L
            var p = DoubleArray(0)
            repeat(3) { iter ->
                val t0 = System.nanoTime()
                OnnxTensor.createTensor(
                    env, FloatBuffer.wrap(x), longArrayOf(1L, 1L, 32L, 360L)
                ).use { inp ->
                    OnnxTensor.createTensor(env, LongBuffer.wrap(w), longArrayOf(1L)).use { wid ->
                        s.run(mapOf("x" to inp, "widths" to wid)).use { r ->
                            val arr = flat(r, 0)
                            p = DoubleArray(arr.size) { arr[it].toDouble() }
                        }
                    }
                }
                val el = (System.nanoTime() - t0) / 1_000_000
                if (iter == 0) cold = el else if (iter == 1) t = el
            }
            Log.i(tag, "SCRIPT cold=${cold}ms warm=${t}ms p=${p.joinToString(",")}")
            checkVec("script.p",
                doubleArrayOf(0.21501460671424866, 0.3558780550956726,
                    0.21101701259613037, 0.21809028089046478), p, 1e-4)
        }

        // ---------- ORIENT ----------
        run {
            val s = env.createSession(modelPath("orient.onnx"), opts)
            val zeros = FloatArray(768 * 768)
            var t = 0L; var cold = 0L
            var p = DoubleArray(0)
            repeat(3) { iter ->
                val t0 = System.nanoTime()
                OnnxTensor.createTensor(
                    env, FloatBuffer.wrap(zeros), longArrayOf(1L, 1L, 768L, 768L)
                ).use { inp ->
                    s.run(mapOf("x" to inp)).use { r ->
                        val arr = flat(r, 0)
                        p = DoubleArray(arr.size) { arr[it].toDouble() }
                    }
                }
                val el = (System.nanoTime() - t0) / 1_000_000
                if (iter == 0) cold = el else if (iter == 1) t = el
            }
            Log.i(tag, "ORIENT cold=${cold}ms warm=${t}ms p=${p.joinToString(",")}")
            checkVec("orient.p",
                doubleArrayOf(0.27725735306739807, 0.25656944513320923, 0.2064194679260254,
                    0.24302607774734497, 0.016727659851312637), p, 1e-4)
        }

        // ---------- ORDER ----------
        run {
            val f = floats("probe_order_f")
            if (f.isEmpty()) return@run
            val s = env.createSession(modelPath("order.onnx"), opts)
            var t = 0L; var cold = 0L
            var cls = DoubleArray(0); var aSum = 0.0; var sameSum = 0.0; var tabSum = 0.0
            repeat(3) { iter ->
                val t0 = System.nanoTime()
                OnnxTensor.createTensor(
                    env, FloatBuffer.wrap(f), longArrayOf(1L, 12L, 16L)
                ).use { inp ->
                    // BOOL 张量：ByteBuffer + OnnxJavaType.BOOL（1 字节一个 true）
                    val mbuf = ByteBuffer.allocate(12).apply { put(ByteArray(12) { 1 }) }
                        .flip() as ByteBuffer
                    OnnxTensor.createTensor(
                        env, mbuf, longArrayOf(1L, 12L), ai.onnxruntime.OnnxJavaType.BOOL
                    ).use { m ->
                        s.run(mapOf("feats" to inp, "mask" to m)).use { r ->
                            // cls 输出 [1, 12, 10]：第 0 行第 0 个样本的 10 个 block 类
                            val cAll = flat(r, 1)
                            val clsArr = FloatArray(10)
                            for (i in 0 until 10) clsArr[i] = cAll[i]
                            cls = DoubleArray(clsArr.size) { clsArr[it].toDouble() }
                            aSum = flat(r, 0).sumOf { it.toDouble() }
                            sameSum = flat(r, 2).sumOf { it.toDouble() }
                            tabSum = flat(r, 3).sumOf { it.toDouble() }
                        }
                    }
                }
                val el = (System.nanoTime() - t0) / 1_000_000
                if (iter == 0) cold = el else if (iter == 1) t = el
            }
            Log.i(tag, "ORDER cold=${cold}ms warm=${t}ms")
            checkVec("order.cls",
                doubleArrayOf(3.650989294052124, 1.7662113904953003, -0.7021074891090393,
                    -2.3150711059570312, -2.331355094909668, -1.1576764583587646,
                    -3.564815044403076, -7.264808654785156, 0.8088746666908264,
                    -2.4026598930358887), cls, 1e-4)
            check("order.a_sum", 2.86102294921875e-06, aSum, 1e-4)
            check("order.same_sum", -174.64108276367188, sameSum, 1e-4)
            check("order.tab_sum", -790.7263793945312, tabSum, 1e-4)
        }

        Log.i(tag, "PROBE SUMMARY pass=$pass fail=$fail")
    }
}
