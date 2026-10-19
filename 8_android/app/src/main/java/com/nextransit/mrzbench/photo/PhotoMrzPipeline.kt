package com.nextransit.mrzbench.photo

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import com.nextransit.mrzbench.bench.IcaoChecksum
import com.nextransit.mrzbench.jni.MrzNative
import com.nextransit.mrzbench.util.TinyJson
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

data class PhotoMrzResult(
    val original: Bitmap,
    val noReflection: Bitmap?,
    val rectified: Bitmap?,
    val mrzCrop: Bitmap?,
    val line1: String,
    val line2: String,
    val conf1: Int,
    val conf2: Int,
    val backend: String,
    val decodeOk: Boolean,
    val fields: Map<String, String>,
    val timings: LinkedHashMap<String, Long>,
    val error: String = "",
)

data class PairRows(
    val first: IntRange,
    val second: IntRange,
    val reflection: List<IntRange>,
)

data class StructureResult(
    val source: Bitmap,
    val work: Bitmap,
    val pair: PairRows?,
    val noReflection: Bitmap?,
    val rectified: Bitmap?,
    val mrzCrop: Bitmap?,
    val timings: LinkedHashMap<String, Long>,
    val error: String = "",
)

object PhotoMrzPipeline {
    private const val MAX_SIDE = 1250
    private const val GLYPH_W = 14
    private const val GLYPH_H = 48
    private const val TARGET_W = 44 * GLYPH_W
    private const val TARGET_H = GLYPH_H * 2 + 8

    private data class Gray(val width: Int, val height: Int, val pixels: IntArray)
    private data class Band(
        val bitmap: Bitmap,
        val line1: Bitmap,
        val line2: Bitmap,
    )

    fun process(context: Context, source: Bitmap): PhotoMrzResult {
        val allStart = System.nanoTime()
        val structure = processStructure(source)
        val timings = structure.timings
        if (structure.error.isNotEmpty() || structure.pair == null) {
            timings["total"] = elapsedMs(allStart)
            return PhotoMrzResult(
                structure.source, structure.noReflection, structure.rectified,
                structure.mrzCrop, "", "", 0, 0, "none", false, emptyMap(),
                timings, structure.error.ifEmpty { "MRZ 双行未定位" },
            )
        }

        val nativeBand = nativeBand(source, structure.pair.first, structure.pair.second,
            scaleFor(structure.source))
        val ocrStart = System.nanoTime()
        val recognition = try {
            val paddle = PaddleOcrAndroid.recognize(context, nativeBand.line1, nativeBand.line2)
            postProcess(
                paddle.line1, paddle.line2, paddle.conf1, paddle.conf2,
                "paddle-onnxruntime",
            )
        } catch (paddleError: Throwable) {
            val fallback = nativeFallback(context, nativeBand.bitmap)
            postProcess(
                fallback.line1, fallback.line2, fallback.conf1, fallback.conf2,
                backend = "native-${fallback.backend}",
                note = "Paddle ONNX unavailable: ${paddleError.javaClass.simpleName}; ${fallback.note}",
            )
        }
        timings["OCR + decode"] = elapsedMs(ocrStart)
        val fields = decodeFields(recognition.line1, recognition.line2)
        val decodeOk = IcaoChecksum.line2Ok(recognition.line2)
        timings["total"] = elapsedMs(allStart)
        return PhotoMrzResult(
            structure.source, structure.noReflection, structure.rectified,
            structure.mrzCrop,
            recognition.line1, recognition.line2, recognition.conf1, recognition.conf2,
            recognition.backend, decodeOk, fields, timings, recognition.note,
        )
    }

    fun processStructure(source: Bitmap): StructureResult {
        val timings = linkedMapOf<String, Long>()
        val workStart = System.nanoTime()
        val scale = scaleFor(source)
        val work = if (scale < 1.0) {
            Bitmap.createScaledBitmap(
                source,
                max(1, (source.width * scale).roundToInt()),
                max(1, (source.height * scale).roundToInt()),
                true,
            )
        } else source
        val gray = grayscale(work)
        val pair = findPair(gray)
        timings["structure"] = elapsedMs(workStart)
        if (pair == null) {
            return StructureResult(source, work, null, work, null, null, timings, "MRZ 双行未定位")
        }

        val cropStart = System.nanoTime()
        val cropBottom = pair.second.last + 1
        val noReflection = Bitmap.createBitmap(work, 0, 0, work.width, cropBottom)
        timings["remove reflection"] = elapsedMs(cropStart)

        val rectStart = System.nanoTime()
        val rectified = rectify(noReflection, gray, pair.first, pair.second)
        timings["rectify"] = elapsedMs(rectStart)

        val mrzCropStart = System.nanoTime()
        val mrzCrop = cropPairClean(noReflection, pair.first, pair.second)
        timings["MRZ crop"] = elapsedMs(mrzCropStart)

        return StructureResult(source, work, pair, noReflection, rectified, mrzCrop, timings)
    }

    private fun scaleFor(source: Bitmap): Double =
        min(1.0, MAX_SIDE.toDouble() / max(source.width, source.height))

    private data class Recognition(
        val line1: String,
        val line2: String,
        val conf1: Int,
        val conf2: Int,
        val backend: String,
        val note: String = "",
    )

    private fun nativeFallback(context: Context, bitmap: Bitmap): Recognition {
        val file = File(context.cacheDir, "photo_mrz_${System.nanoTime()}.ppm")
        return try {
            writePpm(bitmap, file)
            val cnn = parseNative(MrzNative.recogniseString(file.absolutePath, 1))
            if (cnn.line1.isNotEmpty() || cnn.line2.isNotEmpty()) {
                cnn
            } else {
                parseNative(MrzNative.recogniseString(file.absolutePath, 0))
            }
        } finally {
            file.delete()
        }
    }

    private fun parseNative(raw: String): Recognition {
        val ok = TinyJson.boolOf(raw, "ok", false)
        return Recognition(
            normalize(TinyJson.stringOf(raw, "line1")),
            normalize(TinyJson.stringOf(raw, "line2")),
            TinyJson.intOf(raw, "conf1"),
            TinyJson.intOf(raw, "conf2"),
            if (TinyJson.intOf(raw, "backend") == 1) "cnn" else "template",
            if (ok) "" else TinyJson.stringOf(raw, "err", "native OCR failed"),
        )
    }

    private fun postProcess(
        rawLine1: String,
        rawLine2: String,
        conf1: Int,
        conf2: Int,
        backend: String,
        note: String = "",
    ): Recognition {
        val line1 = enforceLine1(rawLine1)
        val (line2, repairs) = repairLine2(rawLine2)
        val repairNote = repairs.joinToString("; ")
        return Recognition(
            line1,
            line2,
            conf1,
            conf2,
            backend,
            listOf(note, repairNote).filter { it.isNotEmpty() }.joinToString("; "),
        )
    }

    private data class Candidate(
        val edits: List<Pair<Int, Char>>,
        val cost: Int,
    )

    private val confusionPairs = listOf(
        "0" to "O", "0" to "Q", "0" to "U", "0" to "D",
        "0" to "V", "0" to "G", "1" to "I", "1" to "L",
        "1" to "T", "1" to "7", "5" to "S", "2" to "Z",
        "8" to "B", "8" to "R", "8" to "X", "8" to "3",
        "6" to "G", "4" to "W", "4" to "A", "W" to "H",
        "W" to "A", "W" to "U", "W" to "M", "H" to "N",
        "H" to "M", "M" to "N", "R" to "B", "L" to "C",
        "K" to "B", "K" to "X", "D" to "U", "D" to "O",
        "U" to "V", "U" to "O", "V" to "Y", "Q" to "O",
        "T" to "I", "T" to "7", "X" to "K",
    ).flatMap { (a, b) ->
        val cost = if (a in "0152689" || b in "0152689") 1 else 2
        listOf(a[0] to (b[0] to cost), b[0] to (a[0] to cost))
    }.groupBy({ it.first }, { it.second })
        .mapValues { (_, values) -> values.distinctBy { it.first }.sortedBy { it.second } }

    private fun enforceLine1(raw: String): String {
        val out = normalizeRaw(raw).padEnd(44, '<').take(44).toCharArray()
        for (i in out.indices) {
            if (i != 1 && out[i].isDigit()) {
                val candidate = confusionPairs[out[i]]
                    ?.firstOrNull { it.first.isLetter() }
                    ?.first
                if (candidate != null) out[i] = candidate
            }
        }
        return out.concatToString()
    }

    private fun repairLine2(raw: String): Pair<String, List<String>> {
        val filtered = normalizeRaw(raw)
        val aligned = alignNear44(filtered)
        val base = (aligned ?: filtered).padEnd(44, '<').take(44)
        val normalized = enforceLine2Slots(base)
        if (IcaoChecksum.line2Ok(normalized)) return normalized to emptyList()

        val fields = listOf(
            Triple("passport_no", 0, 9) to 9,
            Triple("date_of_birth", 13, 19) to 19,
            Triple("date_of_expiry", 21, 27) to 27,
            Triple("personal_no", 28, 42) to 42,
        )
        val options = fields.map { (field, checkPos) ->
            localCandidates(normalized, field.second, field.third, checkPos)
        }
        var best: Pair<String, Candidate>? = null
        for (a in options[0]) for (b in options[1]) for (c in options[2]) for (d in options[3]) {
            val selected = listOf(a, b, c, d)
            val cost = selected.sumOf { it.cost }
            if (cost > 4) continue
            val chars = normalized.toCharArray()
            val reports = ArrayList<Pair<Int, Char>>()
            selected.forEach { candidate ->
                candidate.edits.forEach { (position, replacement) ->
                    chars[position] = replacement
                    reports += position to replacement
                }
            }
            var trial = chars.concatToString()
            var totalCost = cost
            val composite = IcaoChecksum.digitFor(
                trial.substring(0, 10) +
                    trial.substring(13, 20) +
                    trial.substring(21, 28) +
                    trial.substring(28, 43),
            ).toString()[0]
            if (trial[43] != composite) {
                trial = trial.substring(0, 43) + composite
                totalCost++
                reports += 43 to composite
            }
            if (!IcaoChecksum.line2Ok(trial)) continue
            val candidate = Candidate(reports, totalCost)
            if (best == null || candidate.cost < best!!.second.cost ||
                candidate.cost == best!!.second.cost &&
                candidate.edits.size < best!!.second.edits.size) {
                best = trial to candidate
            }
        }
        if (best == null) return normalized to emptyList()

        val original = normalized
        val notes = best!!.second.edits.map { (position, replacement) ->
            "pos $position ${original[position]}->$replacement"
        }
        return best!!.first to notes
    }

    private fun localCandidates(
        line: String,
        start: Int,
        end: Int,
        checkPos: Int,
    ): List<Candidate> {
        val field = line.substring(start, end)
        val check = line[checkPos]
        if (IcaoChecksum.digitFor(field) == checkDigit(check)) {
            return listOf(Candidate(emptyList(), 0))
        }
        val result = ArrayList<Candidate>()
        for (position in start until end) {
            for ((replacement, cost) in confusionPairs[line[position]].orEmpty()) {
                val trial = line.toCharArray()
                trial[position] = replacement
                if (IcaoChecksum.digitFor(trial.concatToString().substring(start, end)) ==
                    checkDigit(check)
                ) {
                    result += Candidate(listOf(position to replacement), cost)
                }
            }
        }
        val expected = IcaoChecksum.digitFor(field).toString()[0]
        if (expected != check) {
            result += Candidate(listOf(checkPos to expected), 1)
        }
        return (result + Candidate(emptyList(), 5))
            .sortedBy { it.cost }
            .distinctBy { it.edits }
            .take(8)
    }

    private fun enforceLine2Slots(line: String): String {
        val out = line.toCharArray()
        val digitPositions = setOf(9, 19, 27, 42, 43) +
            (13..18) + (21..26)
        for (i in out.indices) {
            if (i in digitPositions && out[i].isLetter()) {
                confusionPairs[out[i]]
                    ?.firstOrNull { it.first.isDigit() }
                    ?.first
                    ?.let { out[i] = it }
            } else if (i in 10..12 && out[i].isDigit()) {
                confusionPairs[out[i]]
                    ?.firstOrNull { it.first.isLetter() }
                    ?.first
                    ?.let { out[i] = it }
            }
        }
        return out.concatToString()
    }

    private fun normalizeRaw(raw: String): String = raw.uppercase().filter {
        it == '<' || it in 'A'..'Z' || it in '0'..'9'
    }

    private fun alignNear44(raw: String): String? {
        if (raw.length == 44) return raw
        if (raw.length == 43) {
            for (position in 0..raw.length) {
                for (candidate in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<") {
                    val trial = raw.substring(0, position) + candidate + raw.substring(position)
                    if (IcaoChecksum.line2Ok(enforceLine2Slots(trial))) return trial
                }
            }
        } else if (raw.length == 45) {
            for (position in raw.indices) {
                val trial = raw.removeRange(position..position)
                if (IcaoChecksum.line2Ok(enforceLine2Slots(trial))) return trial
            }
        }
        return null
    }

    private fun checkDigit(value: Char): Int =
        if (value in '0'..'9') value.code - '0'.code else -1

    private fun grayscale(bitmap: Bitmap): Gray {
        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        val gray = IntArray(pixels.size)
        for (i in pixels.indices) {
            val c = pixels[i]
            gray[i] = minOf(255, ((c shr 16 and 0xff) * 299 +
                (c shr 8 and 0xff) * 587 +
                (c and 0xff) * 114 + 500) / 1000)
        }
        return Gray(bitmap.width, bitmap.height, gray)
    }

    private fun findPair(gray: Gray): PairRows? {
        val rowInk = IntArray(gray.height)
        val edge = IntArray(gray.height)
        for (y in 0 until gray.height) {
            var ink = 0
            var energy = 0
            for (x in 0 until gray.width) {
                val value = gray.pixels[y * gray.width + x]
                if (value < 180) ink++
                if (x > 0) energy += abs(value - gray.pixels[y * gray.width + x - 1])
            }
            rowInk[y] = ink
            edge[y] = energy
        }
        val pass180 = mrzPairPass(gray, 180, rowInk, edge)
        if (pass180 != null) return attachReflection(pass180, rowInk, gray.width)
        val cands = ArrayList<PickResult>()
        for (thr in intArrayOf(170, 160, 150)) {
            val r = mrzPairPass(gray, thr, rowInk, edge)
            if (r != null) cands += r
        }
        if (cands.isEmpty()) return null
        val best = cands.maxBy { it.first.first }
        return attachReflection(best, rowInk, gray.width)
    }

    private data class PickResult(val first: IntRange, val second: IntRange)

    private fun mrzPairPass(
        gray: Gray,
        thr: Int,
        rowInk180: IntArray,
        edge: IntArray,
    ): PickResult? {
        val H = gray.height
        val W = gray.width
        val rowInkS = IntArray(H)
        for (y in 0 until H) {
            var ink = 0
            for (x in 0 until W) if (gray.pixels[y * W + x] < thr) ink++
            rowInkS[y] = ink
        }
        val solid = BooleanArray(H) { rowInk180[it] > W * 0.90 }
        val text = BooleanArray(H) {
            rowInkS[it] > W * 0.20 && rowInkS[it] < W * 0.90 && edge[it] > 1800
        }
        val solidB = groups(solid, 5)
        val textB = groups(text, 4)
        val cand = solidB.filter {
            it.first >= H * 0.30 && it.first <= H * 0.90 &&
                it.last - it.first + 1 >= 10
        }
        val sTop = if (cand.isNotEmpty()) cand.last().last + 1 else null

        fun lineShape(run: IntRange): Boolean {
            val h = run.last - run.first + 1
            if (h < 5 || h > 48) return false
            val strong = BooleanArray(W)
            for (x in 0 until W) {
                var ink = 0
                for (y in run) if (gray.pixels[y * W + x] < thr) ink++
                strong[x] = ink >= 4
            }
            val density = strong.count { it }.toDouble() / W
            val maxGroup = groups(strong, 1).maxOfOrNull { it.last - it.first + 1 } ?: 0
            return density <= 0.90 && maxGroup <= max(160, (W * 0.45).toInt())
        }

        fun pick(runs: List<IntRange>): PickResult? {
            var best: Triple<IntArray, IntRange, IntRange>? = null
            for (i in 0 until runs.size - 1) {
                val t1 = runs[i]
                val h1 = t1.last - t1.first + 1
                if (h1 < 4 || !lineShape(t1)) continue
                val t2 = runs[i + 1]
                val h2 = t2.last - t2.first + 1
                val gap = t2.first - t1.last - 1
                if (!lineShape(t2)) continue
                if (h2.toDouble() / h1 !in 0.4..1.6) continue
                val gapMax = max(8, (h1 * 3.0).toInt())
                if (gap < 0 || gap > gapMax) continue
                val rank = intArrayOf(-minOf(h1, h2), kotlin.math.abs(h1 - h2) + gap, gap)
                if (best == null || compareInts(rank, best.first) < 0) {
                    best = Triple(rank, t1, t2)
                }
            }
            return best?.let { PickResult(it.second, it.third) }
        }

        if (sTop != null) {
            val above = textB.filter { it.last < sTop - 1 }
            if (above.size >= 2) pick(above)?.let { return it }
            val below = textB.filter { it.first > sTop }
            if (below.isNotEmpty()) pick(below)?.let { return it }
        }
        if (textB.size >= 2) return pick(textB)
        return null
    }

    private fun compareInts(a: IntArray, b: IntArray): Int {
        for (i in a.indices) {
            if (a[i] != b[i]) return a[i] - b[i]
        }
        return 0
    }

    private fun attachReflection(
        pair: PickResult,
        rowInk: IntArray,
        width: Int,
    ): PairRows {
        val (l1, l2) = pair
        val l2Ink = IntArray(l2.last - l2.first + 1) { l2.first + it }
            .map { rowInk[it] }
            .sorted()
        val med = l2Ink[l2Ink.size / 2]
        val thr = maxOf((0.12 * med).toInt(), (0.02 * width).toInt())
        val segs = ArrayList<IntRange>()
        var start = -1
        for (y in l2.last + 1 until rowInk.size) {
            if (rowInk[y] > thr) {
                if (start < 0) start = y
            } else if (start >= 0) {
                segs += start..(y - 1)
                start = -1
            }
        }
        if (start >= 0) segs += start..(rowInk.size - 1)
        return PairRows(l1, l2, segs)
    }

    private fun groups(mask: BooleanArray, minLength: Int): List<IntRange> {
        val result = ArrayList<IntRange>()
        var start = -1
        for (i in mask.indices) {
            if (mask[i] && start < 0) start = i
            if ((!mask[i] || i == mask.lastIndex) && start >= 0) {
                val end = if (mask[i] && i == mask.lastIndex) i else i - 1
                if (end - start + 1 >= minLength) result += start..end
                start = -1
            }
        }
        return result
    }

    private fun cropPairClean(bitmap: Bitmap, first: IntRange, second: IntRange): Bitmap {
        val y0 = first.first
        val y1 = second.last
        val pad = max(3, ((y1 - y0 + 1) * 0.06).toInt())
        val top = max(0, y0 - pad)
        val bottom = min(bitmap.height, y1 + pad + 1)
        val out = Bitmap.createBitmap(bitmap, 0, top, bitmap.width, bottom - top).copy(
            Bitmap.Config.ARGB_8888,
            true,
        )
        val canvas = Canvas(out)
        val paint = Paint().apply { color = Color.WHITE }
        val clear = 2
        canvas.drawRect(
            0f, (first.last - top + 1).toFloat(),
            out.width.toFloat(), (first.last - top + 1 + clear).toFloat(), paint,
        )
        canvas.drawRect(
            0f, (second.last - top + 1).toFloat(),
            out.width.toFloat(), (second.last - top + 1 + clear + 1).toFloat(), paint,
        )
        return out
    }

    private fun nativeBand(
        source: Bitmap,
        first: IntRange,
        second: IntRange,
        scale: Double,
    ): Band {
        val xpad = min(100, max(10, (source.width * 0.08).roundToInt()))
        val rows = listOf(first, second).map { run ->
            val sourceY0 = max(0, (run.first / scale).roundToInt())
            val sourceY1 = min(source.height, ((run.last + 1) / scale).roundToInt())
            val rowHeight = max(1, sourceY1 - sourceY0)
            val pad = max(8, min(25, (rowHeight * 0.70).roundToInt()))
            Bitmap.createBitmap(
                source,
                xpad,
                max(0, sourceY0 - pad),
                max(1, source.width - 2 * xpad),
                min(source.height, sourceY1 + pad) - max(0, sourceY0 - pad),
            )
        }
        val width = min(rows[0].width, rows[1].width)
        val gutter = max(12, (min(rows[0].height, rows[1].height) * 0.25).roundToInt())
        val output = Bitmap.createBitmap(
            width,
            rows[0].height + gutter + rows[1].height,
            Bitmap.Config.ARGB_8888,
        )
        output.eraseColor(Color.WHITE)
        val canvas = Canvas(output)
        canvas.drawBitmap(rows[0], 0f, 0f, null)
        canvas.drawBitmap(rows[1], 0f, (rows[0].height + gutter).toFloat(), null)
        return Band(
            output,
            Bitmap.createBitmap(rows[0], 0, 0, width, rows[0].height),
            Bitmap.createBitmap(rows[1], 0, 0, width, rows[1].height),
        )
    }

    private fun rectify(
        bitmap: Bitmap,
        gray: Gray,
        first: IntRange,
        second: IntRange,
    ): Bitmap? {
        val q1 = quad(gray, first) ?: return null
        val q2 = quad(gray, second) ?: return null
        val line1 = warp(bitmap, q1, TARGET_W, GLYPH_H)
        val line2 = warp(bitmap, q2, TARGET_W, GLYPH_H)
        val out = Bitmap.createBitmap(TARGET_W, TARGET_H, Bitmap.Config.ARGB_8888)
        out.eraseColor(Color.WHITE)
        val canvas = Canvas(out)
        canvas.drawBitmap(line1, 0f, 0f, null)
        canvas.drawBitmap(line2, 0f, GLYPH_H.toFloat() + 8f, null)
        return out
    }

    private fun quad(gray: Gray, run: IntRange): FloatArray? {
        fun pageSpan(y: Int): IntRange? {
            val values = IntArray(gray.width)
            for (x in 0 until gray.width) {
                val window = (max(0, y - 4)..min(gray.height - 1, y + 4))
                    .map { gray.pixels[it * gray.width + x] }
                    .sorted()
                values[x] = window[window.size / 2]
            }
            val sorted = values.sorted()
            val threshold = max(70, sorted[(sorted.size - 1) * 20 / 100])
            val mask = BooleanArray(gray.width) { values[it] > threshold }
            val closed = close(mask, 15)
            val runs = groups(closed, max(20, gray.width / 50))
            if (runs.isEmpty()) return null
            val center = gray.width / 2
            return runs.maxWithOrNull(
                compareBy<IntRange>({ if (center in it) 1 else 0 })
                    .thenBy { it.last - it.first + 1 },
            )
        }
        val top = pageSpan(run.first) ?: return null
        val bottom = pageSpan(run.last) ?: return null
        if (min(top.last - top.first, bottom.last - bottom.first) < max(80, gray.width / 4)) {
            return null
        }
        return floatArrayOf(
            top.first.toFloat(), run.first.toFloat(),
            top.last.toFloat(), run.first.toFloat(),
            bottom.last.toFloat(), run.last.toFloat(),
            bottom.first.toFloat(), run.last.toFloat(),
        )
    }

    private fun close(mask: BooleanArray, radius: Int): BooleanArray {
        val dilated = BooleanArray(mask.size)
        for (i in mask.indices) {
            for (j in max(0, i - radius)..min(mask.lastIndex, i + radius)) {
                if (mask[j]) {
                    dilated[i] = true
                    break
                }
            }
        }
        val eroded = BooleanArray(mask.size)
        for (i in mask.indices) {
            var all = true
            for (j in max(0, i - radius)..min(mask.lastIndex, i + radius)) {
                if (!dilated[j]) {
                    all = false
                    break
                }
            }
            eroded[i] = all
        }
        return eroded
    }

    private fun warp(source: Bitmap, quad: FloatArray, width: Int, height: Int): Bitmap {
        val from = floatArrayOf(
            0f, 0f, (width - 1).toFloat(), 0f,
            (width - 1).toFloat(), (height - 1).toFloat(), 0f, (height - 1).toFloat(),
        )
        val h = homography(from, quad)
        val sourcePixels = IntArray(source.width * source.height)
        source.getPixels(sourcePixels, 0, source.width, 0, 0, source.width, source.height)
        val outputPixels = IntArray(width * height)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val d = h[6] * x + h[7] * y + 1.0
                val sx = ((h[0] * x + h[1] * y + h[2]) / d).roundToInt()
                val sy = ((h[3] * x + h[4] * y + h[5]) / d).roundToInt()
                outputPixels[y * width + x] = if (
                    sx in 0 until source.width && sy in 0 until source.height
                ) {
                    sourcePixels[sy * source.width + sx]
                } else Color.WHITE
            }
        }
        return Bitmap.createBitmap(outputPixels, width, height, Bitmap.Config.ARGB_8888)
    }

    private fun homography(from: FloatArray, to: FloatArray): DoubleArray {
        val a = Array(8) { DoubleArray(9) }
        for (i in 0 until 4) {
            val x = from[i * 2].toDouble()
            val y = from[i * 2 + 1].toDouble()
            val u = to[i * 2].toDouble()
            val v = to[i * 2 + 1].toDouble()
            val r = i * 2
            a[r][0] = x; a[r][1] = y; a[r][2] = 1.0
            a[r][6] = -u * x; a[r][7] = -u * y; a[r][8] = u
            a[r + 1][3] = x; a[r + 1][4] = y; a[r + 1][5] = 1.0
            a[r + 1][6] = -v * x; a[r + 1][7] = -v * y; a[r + 1][8] = v
        }
        for (col in 0 until 8) {
            var pivot = col
            for (row in col + 1 until 8) {
                if (abs(a[row][col]) > abs(a[pivot][col])) pivot = row
            }
            val swap = a[col]; a[col] = a[pivot]; a[pivot] = swap
            val divisor = a[col][col]
            if (abs(divisor) < 1e-9) return DoubleArray(8)
            for (j in col until 9) a[col][j] /= divisor
            for (row in 0 until 8) {
                if (row == col) continue
                val factor = a[row][col]
                for (j in col until 9) a[row][j] -= factor * a[col][j]
            }
        }
        return DoubleArray(8) { a[it][8] }
    }

    private fun decodeFields(line1: String, line2: String): Map<String, String> {
        if (line1.length < 44 || line2.length < 44) return emptyMap()
        return linkedMapOf(
            "doc_type" to line1.substring(0, 2),
            "issuing_state" to line1.substring(2, 5),
            "surname" to line1.substring(5).substringBefore("<<").replace('<', ' ').trim(),
            "given_names" to line1.substringAfter("<<").replace('<', ' ').trim(),
            "passport_no" to line2.substring(0, 9),
            "nationality" to line2.substring(10, 13),
            "birth_yymmdd" to line2.substring(13, 19),
            "sex" to line2.substring(20, 21),
            "expiry_yymmdd" to line2.substring(21, 27),
            "personal_no" to line2.substring(28, 42).replace('<', ' ').trim(),
            "composite_ck" to line2.substring(43, 44),
        )
    }

    private fun normalize(raw: String): String {
        val clean = raw.uppercase().filter {
            it == '<' || it in 'A'..'Z' || it in '0'..'9'
        }
        return clean.padEnd(44, '<').take(44)
    }

    private fun writePpm(bitmap: Bitmap, file: File) {
        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        BufferedOutputStream(FileOutputStream(file)).use { output ->
            output.write("P6\n${bitmap.width} ${bitmap.height}\n255\n".toByteArray())
            val row = ByteArray(bitmap.width * 3)
            for (y in 0 until bitmap.height) {
                for (x in 0 until bitmap.width) {
                    val color = pixels[y * bitmap.width + x]
                    val i = x * 3
                    row[i] = (color shr 16).toByte()
                    row[i + 1] = (color shr 8).toByte()
                    row[i + 2] = color.toByte()
                }
                output.write(row)
            }
        }
    }

    private fun elapsedMs(start: Long): Long =
        ((System.nanoTime() - start) / 1_000_000L).coerceAtLeast(0L)
}
