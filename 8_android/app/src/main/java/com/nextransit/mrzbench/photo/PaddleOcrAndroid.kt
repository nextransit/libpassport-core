package com.nextransit.mrzbench.photo

import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.ceil

data class PaddleRecognition(
    val line1: String,
    val line2: String,
    val conf1: Int,
    val conf2: Int,
)

/**
 * Recognition-only PP-OCR adapter. The photo pipeline supplies two
 * already-separated rows, so no detector is needed on Android.
 *
 * 模型组合（离线 24 行样本 + 设备矩阵选定）：
 *  - MEDIUM：两行都用 PP-OCRv6_medium_rec（73MB，最准，约 11s/张）
 *  - HYBRID：行1 v6 + 行2 PP-OCRv4_mobile_rec（10MB）；离线真实 L2 行
 *    与 v6 逐字一致，约 5-6s/张（默认）
 *  - MOBILE：两行都用 v4 mobile（约 3-4s/张，行1 少量退化，极限速度选项）
 */
object PaddleOcrAndroid {
    enum class Mode { MEDIUM, HYBRID, MOBILE }

    @Volatile
    var mode: Mode = Mode.MOBILE

    /** v4m 行1 平均置信度低于该值时用 v6 medium 重跑行1（实测误差行显著低于正常行）。 */
    private const val L1_RESCUE_SCORE = 0.90f

    private const val ASSET_V6 = "models/ppocrv6_medium_rec.onnx"
    private const val ASSET_V4 = "models/ppocrv4_mobile_rec.onnx"
    private const val ASSET_DICT_V4 = "models/ppocrv4_mobile_rec.dict.txt"
    private const val INPUT_HEIGHT = 48

    /** 识别输入宽度再压缩系数（离线 22 行样本：对应真实行输出逐字不变的下限）。 */
    private const val WIDTH_FACTOR_ROW1_V6 = 0.8f
    private const val WIDTH_FACTOR_ROW1_V4 = 0.9f
    private const val WIDTH_FACTOR_ROW2_V4 = 0.75f
    private const val WIDTH_FACTOR_ROW2_V6 = 0.85f

    /** PP-OCRv6_medium_rec 字典前 94 项（符号+数字+大写+小写），索引 1 对应下标 0。 */
    private const val REC_CHARS =
        "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    private val lock = Any()
    private var env: OrtEnvironment? = null
    private var sessionV6: OrtSession? = null
    private var sessionV4: OrtSession? = null
    private val vocabV6: List<String> = REC_CHARS.map { it.toString() }
    private var vocabV4: List<String>? = null

    /** 后台预创建两种模型的会话（避免首张图付模型拷贝/会话冷启动）。 */
    fun preload(context: Context) {
        synchronized(lock) {
            if (sessionV6 != null && sessionV4 != null) return
        }
        Thread({
            try {
                val app = context.applicationContext
                session(app, v4 = false)
                session(app, v4 = true)
            } catch (_: Throwable) { /* retry on first recognize() */ }
        }, "paddle-preload").start()
    }

    fun recognize(context: Context, line1: Bitmap, line2: Bitmap): PaddleRecognition {
        val useV4Row1 = mode == Mode.MOBILE
        val useV4Row2 = mode != Mode.MEDIUM
        val (runtime, model1) = session(context, useV4Row1)
        val (_, model2) = session(context, useV4Row2)
        val row1Wf = if (useV4Row1) WIDTH_FACTOR_ROW1_V4 else WIDTH_FACTOR_ROW1_V6
        val row2Wf = if (useV4Row2) WIDTH_FACTOR_ROW2_V4 else WIDTH_FACTOR_ROW2_V6
        // ORT Session 线程安全：两行并行推理（混合模式两行用不同模型）。
        var secondRow: Pair<String, Float>? = null
        val worker = Thread({
            secondRow = recognizeRow(runtime, model2, line2, vocab(context, useV4Row2), row2Wf)
        }, "paddle-row2")
        worker.start()
        val first = recognizeRow(runtime, model1, line1, vocab(context, useV4Row1), row1Wf)
        worker.join()
        val second = secondRow ?: ("" to 0f)
        // 快速模式（双 v4m）下行1 置信度低（斜拍/浅淡扫描行）→ v6 medium 兜底重跑行1。
        val line1Final = if (mode == Mode.MOBILE && first.second < L1_RESCUE_SCORE) {
            android.util.Log.i("MrzOcr", "paddle l1 rescue score=%.3f".format(first.second))
            val (runtime6, model6) = session(context, v4 = false)
            recognizeRow(runtime6, model6, line1, vocabV6, WIDTH_FACTOR_ROW1_V6)
        } else {
            first
        }
        return PaddleRecognition(
            normalize(line1Final.first),
            normalize(second.first),
            (line1Final.second * 100.0f).toInt().coerceIn(0, 100),
            (second.second * 100.0f).toInt().coerceIn(0, 100),
        )
    }

    private fun session(context: Context, v4: Boolean): Pair<OrtEnvironment, OrtSession> {
        synchronized(lock) {
            val currentEnv = env ?: OrtEnvironment.getEnvironment().also { env = it }
            val cached = if (v4) sessionV4 else sessionV6
            val current = cached ?: run {
                val asset = if (v4) ASSET_V4 else ASSET_V6
                val name = if (v4) "v4m" else "v6m"
                val modelFile = copyModel(context, asset)
                val cores = Runtime.getRuntime().availableProcessors()
                val intraOp = cores.coerceAtMost(4)
                val options = OrtSession.SessionOptions()
                options.setIntraOpNumThreads(intraOp)
                val started = System.currentTimeMillis()
                val created = try {
                    currentEnv.createSession(modelFile.absolutePath, options)
                } catch (t: Throwable) {
                    android.util.Log.i(
                        "MrzOcr",
                        "paddle createSession fail $name: ${t.javaClass.simpleName}: ${t.message}",
                    )
                    throw t
                }
                android.util.Log.i(
                    "MrzOcr",
                    "paddle init $name ${System.currentTimeMillis() - started}ms " +
                        "model=${modelFile.length() / 1048576}MB cores=$cores intraOp=$intraOp",
                )
                android.util.Log.i(
                    "MrzOcr",
                    "paddle wf r1v6=$WIDTH_FACTOR_ROW1_V6 r1v4=$WIDTH_FACTOR_ROW1_V4 " +
                        "r2v4=$WIDTH_FACTOR_ROW2_V4 r2v6=$WIDTH_FACTOR_ROW2_V6",
                )
                if (v4) sessionV4 = created else sessionV6 = created
                warmup(currentEnv, created)
                created
            }
            return currentEnv to current
        }
    }

    private fun vocab(context: Context, v4: Boolean): List<String> {
        if (!v4) return vocabV6
        synchronized(lock) {
            vocabV4?.let { return it }
            val loaded = runCatching {
                context.assets.open(ASSET_DICT_V4).bufferedReader().readLines()
            }.getOrNull()?.filter { it.isNotEmpty() } ?: return emptyList()
            if (loaded.isEmpty()) return emptyList()
            vocabV4 = loaded
            return loaded
        }
    }

    /** 首次 run 含 ORT 内存 arena 分配，在会话创建后立即热身。 */
    private fun warmup(runtime: OrtEnvironment, model: OrtSession) {
        runCatching {
            val width = 32
            val data = FloatArray(3 * INPUT_HEIGHT * width)
            val inputName = model.inputNames.iterator().next()
            OnnxTensor.createTensor(
                runtime,
                FloatBuffer.wrap(data),
                longArrayOf(1, 3, INPUT_HEIGHT.toLong(), width.toLong()),
            ).use { input ->
                model.run(mapOf(inputName to input)).use { }
            }
            android.util.Log.i("MrzOcr", "paddle warmup done")
        }.onFailure {
            android.util.Log.i("MrzOcr", "paddle warmup fail: ${it.javaClass.simpleName}: ${it.message}")
        }
    }

    private fun copyModel(context: Context, asset: String): File {
        val target = File(context.filesDir, asset)
        if (target.isFile && target.length() > 10_000_000L) return target
        target.parentFile?.mkdirs()
        context.assets.open(asset).use { input ->
            target.outputStream().use { output -> input.copyTo(output) }
        }
        return target
    }

    fun recognizeRow(
        runtime: OrtEnvironment,
        model: OrtSession,
        source: Bitmap,
        vocab: List<String>,
        widthFactor: Float,
    ): Pair<String, Float> {
        val started = System.currentTimeMillis()
        val base = INPUT_HEIGHT * source.width.toDouble() / source.height
        val width = maxOf(1, ceil(base * widthFactor).toInt())
        val resized = Bitmap.createScaledBitmap(source, width, INPUT_HEIGHT, true)
        val pixels = IntArray(width * INPUT_HEIGHT)
        resized.getPixels(pixels, 0, width, 0, 0, width, INPUT_HEIGHT)
        val data = FloatArray(3 * INPUT_HEIGHT * width)
        val plane = INPUT_HEIGHT * width
        for (y in 0 until INPUT_HEIGHT) {
            for (x in 0 until width) {
                val color = pixels[y * width + x]
                val baseIndex = y * width + x
                // 与桌面 _decode_one 一致走灰度（彩色输入会丢字符）：luma 复制到三通道。
                val luma = 0.299f * (color shr 16 and 0xff) +
                    0.587f * (color shr 8 and 0xff) + 0.114f * (color and 0xff)
                val value = (luma / 255.0f - 0.5f) / 0.5f
                data[baseIndex] = value
                data[plane + baseIndex] = value
                data[plane * 2 + baseIndex] = value
            }
        }
        val inputName = model.inputNames.iterator().next()
        OnnxTensor.createTensor(
            runtime,
            FloatBuffer.wrap(data),
            longArrayOf(1, 3, INPUT_HEIGHT.toLong(), width.toLong()),
        ).use { input ->
            model.run(mapOf(inputName to input)).use { result ->
                val output = result[0].value as Array<*>
                val timeSteps = output[0] as Array<*>
                val indices = IntArray(timeSteps.size)
                val scores = FloatArray(timeSteps.size)
                for (i in timeSteps.indices) {
                    val vector = timeSteps[i] as FloatArray
                    var best = 0
                    var score = vector[0]
                    for (j in 1 until vector.size) {
                        if (vector[j] > score) {
                            best = j
                            score = vector[j]
                        }
                    }
                    indices[i] = best
                    scores[i] = score
                }
                val text = StringBuilder()
                var previous = -1
                var scoreTotal = 0.0f
                var scoreCount = 0
                for (i in indices.indices) {
                    val index = indices[i]
                    if (index == previous || index == 0) {
                        previous = index
                        continue
                    }
                    if (index in 1..vocab.size) {
                        text.append(vocab[index - 1])
                        scoreTotal += scores[i]
                        scoreCount++
                    }
                    previous = index
                }
                android.util.Log.i(
                    "MrzOcr",
                    "paddle row ${source.width}x${source.height}->${width}x$INPUT_HEIGHT " +
                        "t=${System.currentTimeMillis() - started}ms chars=${text.length} " +
                        "wf=$widthFactor score=%.3f".format(
                            if (scoreCount == 0) 0.0f else scoreTotal / scoreCount,
                        ),
                )
                return text.toString() to
                    if (scoreCount == 0) 0.0f else scoreTotal / scoreCount
            }
        }
    }

    private fun normalize(raw: String): String {
        return raw.uppercase().filter {
            it == '<' || it in 'A'..'Z' || it in '0'..'9'
        }
    }
}
