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
 * Recognition-only PP-OCRv6 adapter. The photo pipeline supplies two
 * already-separated rows, so no detector is needed on Android.
 */
object PaddleOcrAndroid {
    private const val ASSET_MODEL = "models/ppocrv6_medium_rec.onnx"
    private const val INPUT_HEIGHT = 48
    private const val ASCII_FIRST = 33
    private const val ASCII_LAST = 126

    private val lock = Any()
    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null

    /**
     * Eagerly initialise the ORT environment, copy the recognition model
     * to internal storage and create the inference session on a worker
     * thread. Safe to call multiple times — subsequent calls short-circuit
     * if the session already exists. Call from Application.onCreate (or
     * MainActivity.onCreate) so the first photo does not pay 2-3s of
     * cold start.
     */
    fun preload(context: Context) {
        synchronized(lock) {
            if (session != null) return
        }
        Thread({
            try {
                getSession(context.applicationContext)
            } catch (_: Throwable) { /* will retry on first recognize() */ }
        }, "paddle-preload").start()
    }

    fun recognize(context: Context, line1: Bitmap, line2: Bitmap): PaddleRecognition {
        val (runtime, model) = getSession(context)
        val first = recognizeRow(runtime, model, line1)
        val second = recognizeRow(runtime, model, line2)
        return PaddleRecognition(
            normalize(first.first),
            normalize(second.first),
            (first.second * 100.0f).toInt().coerceIn(0, 100),
            (second.second * 100.0f).toInt().coerceIn(0, 100),
        )
    }

    private fun getSession(context: Context): Pair<OrtEnvironment, OrtSession> {
        synchronized(lock) {
            val currentEnv = env ?: OrtEnvironment.getEnvironment().also { env = it }
            val currentSession = session ?: run {
                val modelFile = copyModel(context)
                val options = OrtSession.SessionOptions()
                options.setIntraOpNumThreads(2)
                currentEnv.createSession(modelFile.absolutePath, options).also {
                    session = it
                }
            }
            return currentEnv to currentSession
        }
    }

    private fun copyModel(context: Context): File {
        val target = File(context.filesDir, "models/ppocrv6_medium_rec.onnx")
        if (target.isFile && target.length() > 10_000_000L) return target
        target.parentFile?.mkdirs()
        context.assets.open(ASSET_MODEL).use { input ->
            target.outputStream().use { output -> input.copyTo(output) }
        }
        return target
    }

    private fun recognizeRow(
        runtime: OrtEnvironment,
        model: OrtSession,
        source: Bitmap,
    ): Pair<String, Float> {
        val width = maxOf(1, ceil(INPUT_HEIGHT * source.width / source.height.toDouble()).toInt())
        val resized = Bitmap.createScaledBitmap(source, width, INPUT_HEIGHT, true)
        val pixels = IntArray(width * INPUT_HEIGHT)
        resized.getPixels(pixels, 0, width, 0, 0, width, INPUT_HEIGHT)
        val data = FloatArray(3 * INPUT_HEIGHT * width)
        val plane = INPUT_HEIGHT * width
        for (y in 0 until INPUT_HEIGHT) {
            for (x in 0 until width) {
                val color = pixels[y * width + x]
                val base = y * width + x
                data[base] = ((color shr 16 and 0xff) / 255.0f - 0.5f) / 0.5f
                data[plane + base] = ((color shr 8 and 0xff) / 255.0f - 0.5f) / 0.5f
                data[plane * 2 + base] = ((color and 0xff) / 255.0f - 0.5f) / 0.5f
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
                    if (index in 1..(ASCII_LAST - ASCII_FIRST + 1)) {
                        text.append((ASCII_FIRST + index - 1).toChar())
                        scoreTotal += scores[i]
                        scoreCount++
                    }
                    previous = index
                }
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
