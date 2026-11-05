package com.nextransit.mrzbench.photo

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.Log
import com.nextransit.mrzbench.bench.IcaoChecksum
import com.nextransit.mrzbench.jni.MrzNative
import com.nextransit.mrzbench.photo.ocr.NativeEngine
import com.nextransit.mrzbench.photo.ocr.OcrDispatch
import com.nextransit.mrzbench.photo.ocr.OcrEngine
import com.nextransit.mrzbench.photo.ocr.OcrResult
import com.nextransit.mrzbench.photo.ocr.PaddleEngine
import com.nextransit.mrzbench.photo.ocr.mlkit.MlKitEngine
import com.nextransit.mrzbench.photo.ocr.tesseract.TesseractEngine
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

data class PhotoMrzResult(
    val original: Bitmap,
    val noReflection: Bitmap?,
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
    val rawDecodeOk: Boolean = false,
    val repaired: Boolean = false,
    /** True when the decoder found a second checksum-valid reading that
     * was almost as likely as the one returned.  The line passes the
     * checksum, but the checksum could not distinguish it from an
     * alternative reading -- which means the result may be wrong even
     * though it looks successful. */
    val ambiguous: Boolean = false,
    /** Log-probability gap to the runner-up reading, in thousandths. */
    val altMargin: Int = 0,
    /** The runner-up reading, empty when there is none. */
    val altLine2: String = "",
    /** Fraction of the MRZ region whose pixels are blown out (luma >=
     * 250), in thousandths.  A laminate highlight erases the strokes,
     * so the result is unreliable when this is high. */
    val glarePermille: Int = 0,
    /**
     * line 2 的字段在结构上是否可能（`mrz_ocr_line2_struct_ok`）。
     *
     * 与 [decodeOk] 的校验位检查互相独立：校验位只证明 44 个字符自洽，
     * 不能证明它们是文字。assets 语料上 28 张校验通过的里 19 张国籍码
     * 不存在、1 张日期不存在，全部通过了 [decodeOk]。
     *
     * false 表示日期不可能存在——这种结果不要展示、不要落库。
     */
    val structOk: Boolean = true,
    /**
     * 国籍码可疑（`mrz_ocr_line2_nat_suspect`）：三字母、不含 '<'、且不是
     * ISO-3166-1 alpha-3 或 ICAO 保留码。只作告警（德国印 `D<<` 不告警，
     * 科索沃印 `RKS` 会误报），由调用方决定要不要丢。
     */
    val natSuspect: Boolean = false,
    /**
     * 签发国（line1 2..4）与国籍（line2 10..12）不一致。
     *
     * 41 张真机照片上它标出 4 张、4 张全错，是唯一能抓住 line1 单字符错误的
     * 信号——置信度与 ambiguous 合起来一个都没抓到。只作告警。
     */
    val countryMismatch: Boolean = false,
    /**
     * 签发国码（line1 2..4）可疑：不是任何国家会签发的码。
     *
     * 这是 [natSuspect] 在 line1 上的镜像，而补上它才是对的：line2 有五个
     * 校验位，line1 一个都没有——恰恰是没有校验位的那一行最需要内容检查。
     *
     * 41 张真机照片上 line1 读错的 4 张，签发国码全是伪造的（`HUG`/`CHY`/
     * `CNN`/`ATA`）。只作告警。
     */
    val issuerSuspect: Boolean = false,
) {
    /**
     * 可以展示 / 落库：解码通过，且校验位之外的两个否决位没有触发。
     *
     * [decodeOk] 只说明「校验位闭合 + MRZ 形状合法」，一个在噪声上被
     * beam 凑出来的行同样满足它。UI 和下游应该用这个。
     */
    val trustworthy: Boolean get() = decodeOk && structOk && !natSuspect && !issuerSuspect
}

/**
 * One pipeline stage, delivered to the UI as soon as it is ready.
 *
 * The stage images are produced long before the OCR result is known, so
 * publishing them incrementally lets the UI show the progress of a run
 * instead of a single frozen frame at the end.
 */
data class MrzStage(
    val key: String,
    val label: String,
    val bitmap: Bitmap?,
)

data class PairRows(
    val first: IntRange,
    val second: IntRange,
    val reflection: List<IntRange>,
    /** Fraction of the sampled lower half whose pixels are blown out
     * (luma >= 250), in thousandths.  A laminate specular highlight
     * erases the strokes, so this is what tells the UI to ask for a
     * different angle. */
    val glarePermille: Int = 0,
)

data class StructureResult(
    val source: Bitmap,
    val work: Bitmap,
    val pair: PairRows?,
    val noReflection: Bitmap?,
    val mrzCrop: Bitmap?,
    val xpadAdaptive: Pair<Int, Int> = Pair(0, 0),
    val timings: LinkedHashMap<String, Long>,
    val error: String = "",
    /** Fraction of the MRZ region whose pixels are blown out (luma >=
     * 250), in thousandths.  A laminate highlight erases the strokes
     * and no thresholding can bring them back, so the caller can ask
     * the user to tilt the phone. */
    val glarePermille: Int = 0,
    /** CS/CD 单行 MRZ 的行范围，当 pair 为 null 时使用。 */
    val singleRow: IntRange? = null,
)

object PhotoMrzPipeline {
    // Largest source side the DETAILED path works on.  The product hot
    // path (processFast) never scales the photo at all -- it locates the
    // rows and crops the band in the source resolution -- so this only
    // affects the inspect/detailed pipeline.
    //
    // It is NOT a knob that can be tightened freely: downscaling the
    // photo before the MRZ rows are cropped destroys the glyph
    // resolution.  Measured on the 41 labelled photos with the same
    // crop, the same weights and the same code:
    //
    //     MAX_SIDE 1250 ->  7/41 lines inside the truth set
    //     MAX_SIDE 1440 -> 40/41   (no downscale at all)
    //     MAX_SIDE 1600 -> 40/41
    //
    // The checksum beam can only repair a cell the network still ranks
    // in its top three, and a 13% loss of linear resolution moves
    // enough of those margins that the correct reading drops out of the
    // candidate set -- after which no amount of search can bring it
    // back.  1600 is the cap the native hot path already uses
    // (FAST_NATIVE_MAX_WIDTH), so the band is never scaled twice.
    //
    // Raised from 1250 to 1600 so the detailed path's band matches the
    // native hot path (FAST_NATIVE_MAX_WIDTH) and the band is never
    // scaled twice.
    private const val MAX_SIDE = 1600
    private const val FAST_BUDGET_MS = 190

    /**
     * Minimum remaining budget (ms) required before the second engine is
     * allowed to run.  The FUSED fallback costs 50-80 ms on the device, so
     * starting it with less than that left would blow the 200 ms request
     * budget for a result that is no better than the review flag.
     *
     * The gate must cover the ENGINE cost, not just "some progress left":
     * 70 ms of headroom still allowed 190-70+85 = 205 ms worst case.
     * 85 = 70 ms engine cost + 15 ms scheduling slack keeps the worst
     * case inside the 190 ms fast budget: 190 - 85 + 85 = 190.
     */
    private const val FALLBACK_MIN_BUDGET_MS = 85

    /**
     * Band framing alternates for the candidate loop (shared-pad scale
     * factors on the restore pair; 0.70 = the shipped framing, implicit).
     * Only tried when the primary framing fails decode+shape+ICAO; see
     * [processFast] and 10_ocr_research/band_candidates_product_port.md.
     */
    private val FRAME_PAD_ALTERNATES = doubleArrayOf(1.35, 0.45)

    /**
     * Minimum remaining budget (ms) for one framing candidate.  A
     * candidate is a single CNN call (~40-50 ms on the device), so it
     * needs less headroom than the fused fallback (which runs the CNN and
     * the sequence head, ~85 ms).
     */
    private const val FRAMING_MIN_BUDGET_MS = 55

    private const val FAST_NATIVE_MAX_WIDTH = 1600
    private const val LOCATE_DOWNSCALE_THRESHOLD = 1600
    private const val LOCATE_TARGET_SIDE = 1280
    private const val MIN_REPAIR_CONF = 65
    private data class Gray(val width: Int, val height: Int, val pixels: IntArray)
    private data class Band(
        val bitmap: Bitmap,
        val line1: Bitmap,
        val line2: Bitmap,
    )

    private val engineLock = Any()

    @Volatile
    private var engineCache: Pair<Context, List<OcrEngine>>? = null

    private fun enginesFor(context: Context): List<OcrEngine> {
        val app = context.applicationContext
        val cached = engineCache
        if (cached != null && cached.first === app) return cached.second
        synchronized(engineLock) {
            val again = engineCache
            if (again != null && again.first === app) return again.second
            // 对比模式仍保留全部引擎；默认 auto 走独立 native fast path。
            val engines = listOf(
                PaddleEngine(app),
                MlKitEngine(),
                TesseractEngine.fromAssets(app),
                NativeEngine(app),
            )
            engineCache = app to engines
            return engines
        }
    }

    /** gui.py METHOD_ORDER 对齐的算法清单（Android 另含 mlkit），「全部方案」逐个独立运行。 */
    val METHODS: List<Pair<String, String>> = listOf(
        "traditional" to "传统模板",
        "cnn" to "轻量CNN",
        "crnn" to "CRNN",
        "fuse" to "融合CNN+CRNN",
        "tesseract" to "Tesseract",
        "paddle" to "PaddleOCR",
        "indra" to "IndraOCR",
        "mlkit" to "ML Kit",
    )

    /** 按算法筛选引擎；"auto" 为链式（优先级顺序 + 短路）。 */
    private fun enginesForMethod(context: Context, method: String): List<OcrEngine> {
        val app = context.applicationContext
        return when (method) {
            "traditional" -> listOf(NativeEngine(app, NativeEngine.MODE_TEMPLATE))
            "cnn" -> listOf(NativeEngine(app, NativeEngine.MODE_CNN))
            "crnn" -> listOf(NativeEngine(app, NativeEngine.MODE_CRNN))
            "fuse" -> listOf(NativeEngine(app, NativeEngine.MODE_FUSE))
            "tesseract" -> enginesFor(app).filterIsInstance<TesseractEngine>()
            "paddle" -> enginesFor(app).filterIsInstance<PaddleEngine>()
            "indra" -> listOf(com.nextransit.mrzbench.photo.ocr.IndraEngine(app))
            "mlkit" -> enginesFor(app).filterIsInstance<MlKitEngine>()
            else -> enginesFor(app)
        }
    }

    /** Pre-open heavy engines (tesseract handles) off the UI path. */
    fun warm(context: Context) {
        val app = context.applicationContext
        Thread({
            try {
                enginesFor(app).filterIsInstance<TesseractEngine>().forEach { it.warm() }
            } catch (_: Throwable) {
            }
            try {
                warmNative(app)
            } catch (_: Throwable) {
            }
        }, "photo-warm").start()
    }

    /**
     * Warm the native recogniser.
     *
     * The first recognition in a process pays for loading the JNI
     * library, the CNN weights, the 4 MB scratch pad (every page of it a
     * first-touch page fault) and the JIT warm-up of the Kotlin side.
     * Measured on the device that made the first photo cost 205 ms
     * against 145-154 ms for the ones after it, i.e. the very first
     * request was over the 200 ms budget while the steady state was
     * well inside it.
     *
     * Running one tiny synthetic MRZ through the same path in the
     * background removes that one-off cost from the user's first photo.
     */
    private fun warmNative(app: Context) {
        val w = 640
        val h = 96
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(bmp)
        canvas.drawColor(android.graphics.Color.WHITE)
        val paint = android.graphics.Paint()
        paint.color = android.graphics.Color.BLACK
        paint.textSize = 28f
        // Two rows of characters, which is what the recogniser's band
        // locator and its 44-cell grid expect.
        canvas.drawText("P<CHNCHEN<<YANG<<<<<<<<<<<<<<<<<<<<<", 10f, 40f, paint)
        canvas.drawText("E12345678CHN9001011M3001019<<<<<<<<<<<<<<<", 10f, 80f, paint)
        try {
            MrzNative.setParallel(true)
            NativeEngine(app, NativeEngine.MODE_CNN).recognizeBitmap(bmp, 500)
        } catch (_: Throwable) {
        } finally {
            bmp.recycle()
        }
    }

    fun process(
        context: Context,
        source: Bitmap,
        method: String = "auto",
        onStage: ((MrzStage) -> Unit)? = null,
    ): PhotoMrzResult {
        if (method == "auto" || method.isBlank()) {
            return processFast(context, source, onStage)
        }
        return processDetailed(context, source, method, onStage)
    }

    /**
     * 默认照片链路：一次底部 ROI + 一次 native CNN。
     * 展示用的矫正图不进入热路径；mrzCrop 仅保留同一 ROI 供 UI 预览。
     */
    private fun processFast(
        context: Context,
        source: Bitmap,
        onStage: ((MrzStage) -> Unit)? = null,
    ): PhotoMrzResult {
        val allStart = System.nanoTime()
        onStage?.invoke(MrzStage("original", "\u2460 original", source))
        val structure = locateFast(source)
        val timings = structure.timings
        if (structure.error.isNotEmpty() ||
            (structure.pair == null && structure.singleRow == null)
        ) {
            timings["total"] = elapsedMs(allStart)
            return PhotoMrzResult(
                original = source,
                noReflection = null,
                mrzCrop = null,
                line1 = "",
                line2 = "",
                conf1 = 0,
                conf2 = 0,
                backend = "none",
                decodeOk = false,
                fields = emptyMap(),
                timings = timings,
                error = structure.error.ifEmpty { "MRZ 双行未定位" },
            )
        }

        val bandBitmap = if (structure.pair != null) {
            nativeBandFast(
                source,
                structure.pair.first,
                structure.pair.second,
                structure.xpadAdaptive,
            )
        } else {
            nativeBandSingle(source, structure.singleRow!!, structure.xpadAdaptive)
        }
        // The fast path produces the located band only -- no
        // de-reflection and no perspective warp are computed on the hot
        // path -- so stages 2-4 all show that band.  Publishing it as soon
        // as the two lines are located means the user sees the ROI while
        // the OCR is still running, instead of one frozen frame at the end.
        onStage?.invoke(MrzStage("clean", "\u2461 reflection removed", bandBitmap))
        onStage?.invoke(MrzStage("mrz", "\u2462 MRZ crop", bandBitmap))
        val normalizeStart = System.nanoTime()
        // Keep the natural inter-line geometry. Per-line stretching changes
        // the segmentation distribution and makes the CNN collapse on
        // repeated characters; only cap the width for the native hot path.
        val ocrInput = fastNativeInput(bandBitmap)
        timings["fast normalize"] = elapsedMs(normalizeStart)
        val ocrStart = System.nanoTime()
        var remaining = (FAST_BUDGET_MS - elapsedMs(allStart).toInt()).coerceAtLeast(1)
        // Two engines, in a fixed order: the per-character CNN first and
        // the FUSED head only as a fallback.
        //
        // Measured on the 41 labelled photos against the human-corrected
        // truth (host, same band, same weights):
        //
        //   per-character CNN : line2 exact 37/41
        //   fused             : line2 exact 31/41
        //
        // The fused head wins only when the CNN cannot decode.  In an
        // offline both-engines comparison, fusion overwrites the CNN's
        // checksum-beam repairs with the raw top-1 character; on
        // E/G/K/Q/T/AJ that turned six correct CNN lines into wrong ones.
        // The old 22/41-vs-21/41 result came from the board's noisy
        // fixmrz labels and no longer applies.
        val app = context.applicationContext
        // The resident worker runs the two MRZ lines concurrently; without
        // it the native side processes them one after the other and the
        // OCR step costs roughly twice as much.  It must be enabled for
        // BOTH engines: it used to be set only on the non-CRNN branch,
        // which meant the CRNN branch silently ran serially.
        MrzNative.setParallel(true)
        // ---- Band framing candidates ------------------------------------
        // The restore-pair band with the shipped pad (0.70 x row height) is
        // the p50-optimal framing, but the measured assets gap (74.17% char
        // on the product framing vs 97.15% with candidate selection, see
        // 10_ocr_research/band_candidates_product_port.md) shows some
        // photos only decode under a different vertical framing.  The two
        // alternates below re-frame the SAME restore pair wider/tighter.
        // They are only built when the primary framing fails the strict
        // test (decode + shape + five ICAO checksums), so a clean frame --
        // ~98% of photos -- costs exactly what it did before.  Each
        // alternate is budget-gated like the fused fallback.
        fun postProcessOf(res: OcrResult): Recognition = postProcess(
            res.line1, res.line2, res.conf1, res.conf2,
            res.engineName, res.error ?: "",
            res.structOk, res.natSuspect, res.countryMismatch, res.issuerSuspect,
        )
        fun strictAccept(rec: Recognition): Boolean {
            if (!decodesOk(rec)) return false
            // Gate on the RAW engine output, not the repaired line: a
            // checksum-repaired line can be valid while the raw read was
            // not, and the framing question is about the raw read.
            val raw = rec.rawLine2
            if (raw.length != 44) return false
            return isMrzShape(rec.rawLine1, raw) && IcaoChecksum.line2Ok(raw)
        }
        val first: OcrResult
        val firstRec: Recognition
        var framingUsed = 0
        var framingTried = false
        run {
            val primary = try {
                NativeEngine(app, NativeEngine.MODE_CNN).recognizeBitmap(ocrInput, remaining)
            } finally {
                /* bitmaps released after both engines have run */
            }
            var rec = postProcessOf(primary)
            var res = primary
            val frameSeq = System.nanoTime() and 0xFFFFFF
            android.util.Log.i(
                "MrzOcr",
                "segScore=${MrzNative.lastSegScore()} " +
                    "raw2len=${rec.rawLine2.length} raw2='${rec.rawLine2.take(24)}'",
            )
            // Diagnostic: keep the band of a frame that did not decode,
            // so the slow path can be reproduced on the host (the device
            // timing and the host timing differ, and only the device
            // shows the real budget).  Written to the app's own files
            // directory; the file name carries the index so the frames
            // can be lined up with the log.
            if (!rec.rawLine2.isEmpty() || rec.rawLine1.length != 44) {
                try {
                    val dir = java.io.File(context.filesDir, "failedbands").apply { mkdirs() }
                    val out = java.io.File(dir, "band_${System.currentTimeMillis()}_$frameSeq.ppm")
                    out.outputStream().use { fos ->
                        val w = bandBitmap.width
                        val h = bandBitmap.height
                        val px = IntArray(w * h)
                        bandBitmap.getPixels(px, 0, w, 0, 0, w, h)
                        fos.write("P6\n$w $h\n255\n".toByteArray())
                        val buf = ByteArray(w * h * 3)
                        var o = 0
                        for (p2 in px) {
                            buf[o++] = ((p2 shr 16) and 0xFF).toByte()
                            buf[o++] = ((p2 shr 8) and 0xFF).toByte()
                            buf[o++] = (p2 and 0xFF).toByte()
                        }
                        fos.write(buf)
                    }
                } catch (_: Throwable) {
                }
            }
            // Only try the alternates when the band DID contain two
            // readable rows but the reading is wrong.  When the primary
            // framing produced no 44-character row at all ("could not
            // split into two lines"), the band itself is mis-framed and
            // a different pad is unlikely to fix it -- measured on the
            // device (2026-11-05): all three such frames failed every
            // alternate too, and the two extra CNN calls pushed the p95
            // from 108 ms to 137 ms for zero accuracy gain.
            val primaryHasTwoRows = rec.rawLine2.length == 44
            if (structure.pair != null && !strictAccept(rec) && primaryHasTwoRows) {
                framingTried = true
                android.util.Log.i(
                    "MrzOcr",
                    "framing candidates: primary rejected, trying " +
                        "${FRAME_PAD_ALTERNATES.size} alternates",
                )
                for ((idx, padSpec) in FRAME_PAD_ALTERNATES.withIndex()) {
                    remaining = (FAST_BUDGET_MS - elapsedMs(allStart).toInt()).coerceAtLeast(1)
                    // A framing candidate is ONE CNN call (the plain
                    // engine), not the CNN+FUSE pair the fallback needs, so
                    // it gets its own gate.  Reusing the fallback's 85 ms
                    // skipped the second alternate on the device: the
                    // first candidate had already spent the budget and
                    // the tighter framing (0.45) -- the one that actually
                    // fits a band whose two rows have very different
                    // heights -- never ran.
                    if (remaining < FRAMING_MIN_BUDGET_MS) {
                        android.util.Log.i(
                            "MrzOcr",
                            "framing candidate $idx skipped: remaining=${remaining}ms " +
                                "< $FRAMING_MIN_BUDGET_MS",
                        )
                        break
                    }
                    val altBand = nativeBandFast(
                        source,
                        structure.pair.first,
                        structure.pair.second,
                        structure.xpadAdaptive,
                        padSpec,
                    )
                    val altInput = fastNativeInput(altBand)
                    val alt = try {
                        MrzNative.setParallel(true)
                        NativeEngine(app, NativeEngine.MODE_CNN)
                            .recognizeBitmap(altInput, remaining)
                    } catch (_: Throwable) {
                        null
                    }
                    val altRec = alt?.let { postProcessOf(it) }
                    val accept = altRec != null && strictAccept(altRec)
                    try {
                        if (altInput !== altBand) altInput.recycle()
                        altBand.recycle()
                    } catch (_: Throwable) {
                    }
                    android.util.Log.i(
                        "MrzOcr",
                        "framing candidate $idx (pad=$padSpec): " +
                            "accept=$accept line2='${alt?.line2 ?: ""}'",
                    )
                    if (accept && alt != null && altRec != null) {
                        res = alt
                        rec = altRec
                        framingUsed = idx + 1
                        break
                    }
                }
            }
            first = res
            firstRec = rec
        }
        if (framingUsed > 0) {
            timings["framing candidate"] = framingUsed.toLong()
        }
        val firstOk = decodesOk(firstRec)
        val needSecond = when {
            first.error != null && first.error.isNotEmpty() -> true
            !firstOk -> true
            // The framing candidates already ran on this frame and none
            // of them decoded.  The fused fallback would re-run the whole
            // CNN front-end on the SAME band to add the sequence head,
            // and it cannot rescue a structural failure (the band itself
            // is wrong -- the CRNN reads the same mis-framed pixels).
            // Measured on the device: of the frames that failed, none was
            // rescued by the fallback, and running it on top of the
            // candidates pushed the p95 from 108 ms to 137 ms.
            framingTried -> false
            // A structural failure -- the band did not even yield two
            // 44-character rows ("could not split into two lines") --
            // cannot be rescued by the fused head either: it reads the
            // same mis-framed pixels with a sequence model.  Measured on
            // the device: 0 of the 3 structural failures were rescued by
            // the fallback, and each attempt cost 50-80 ms of the frame
            // budget (the worst frame went from 137 ms to 223 ms).
            firstRec.rawLine2.length != 44 -> false
            // The CNN decoded cleanly BUT reports a checksum-valid near-tie
            // (native line2_ambiguous).  The residual CNN errors on the
            // real-photo corpus are exactly this shape -- clean decode,
            // checksum-valid, wrong text (2026-9-30-H) -- and the fused
            // engine (CNN decode + CRNN arbitration) can rescue precisely
            // those: it equals the CNN unless the CRNN clearly favours the
            // runner-up on the differing columns.  The budget gate below
            // still applies.
            first.ambiguous -> true
            else -> false
        }
        val result: OcrResult
        val recognition: Recognition
        if (needSecond) {
            remaining = (FAST_BUDGET_MS - elapsedMs(allStart).toInt()).coerceAtLeast(1)
            // Fallback: the FUSED engine (CNN + CRNN with a confidence
            // gate), i.e. the engine the first pass did NOT use.  Running
            // the same engine twice could never change the outcome and
            // doubled the OCR cost for exactly the samples that were
            // slowest.
            //
            // The fallback is only worth its ~50-80 ms when the budget can
            // absorb it: on a loaded device the first engine alone can
            // already take 70 ms, and a fallback that pushes the total
            // past the 200 ms request budget is worse than a clean
            // "needs review" result.
            val second = if (remaining >= FALLBACK_MIN_BUDGET_MS) {
                try {
                    MrzNative.setParallel(true)
                    // The CNN already ran; the fused fallback adds the
                    // sequence head and is only consulted when the CNN did
                    // not decode.  It costs the same conv front-end plus
                    // the per-row recurrence.
                    NativeEngine(app, NativeEngine.MODE_FUSE).recognizeBitmap(ocrInput, remaining)
                } catch (_: Throwable) {
                    null
                }
            } else {
                null
            }
            val secondRec = second?.let {
                postProcess(
                    it.line1, it.line2, it.conf1, it.conf2, it.engineName, it.error ?: "",
                    it.structOk, it.natSuspect, it.countryMismatch, it.issuerSuspect,
                )
            }
            // Prefer whichever engine actually decoded; if BOTH decode, the
            // one with the higher confidence wins, and the disagreement
            // itself is reported so the UI can ask for review.
            val pickSecond = when {
                secondRec == null -> false
                firstOk && !decodesOk(secondRec) -> false
                !firstOk && decodesOk(secondRec) -> true
                else -> {
                    val same = firstRec.rawLine1 == secondRec.rawLine1 &&
                        firstRec.rawLine2 == secondRec.rawLine2
                    if (same) {
                        // Agreement is strong evidence: keep the first text.
                        false
                    } else {
                        secondRec.conf1 + secondRec.conf2 > firstRec.conf1 + firstRec.conf2
                    }
                }
            }
            if (pickSecond && secondRec != null) {
                result = second!!
                recognition = secondRec
            } else {
                result = first
                recognition = firstRec
            }
        } else {
            result = first
            recognition = firstRec
        }
        try {
            if (ocrInput !== bandBitmap) ocrInput.recycle()
        } catch (_: Throwable) {
        }
        val rawDecodeOk = isStrictTd3(recognition.rawLine1, recognition.rawLine2)
        val decodeOk = decodesOk(recognition)
        timings["OCR + decode"] = elapsedMs(ocrStart)
        timings["total"] = elapsedMs(allStart)
        val note = listOf(
            recognition.note,
            if (!decodeOk && result.error.isNullOrEmpty()) "MRZ 校验未通过" else "",
            // 校验位之外的否决：这些行通过了校验位，但字段不可能是真的。
            if (!recognition.structOk) "字段结构不可能（日期非法）" else "",
            if (recognition.natSuspect) "国籍码可疑" else "",
            if (recognition.countryMismatch) "签发国与国籍不一致" else "",
            if (recognition.issuerSuspect) "签发国码可疑" else "",
        ).filter { it.isNotEmpty() }.joinToString("; ")
        android.util.Log.i(
            "MrzOcr",
            "fast chosen=${recognition.backend} rawDecode=$rawDecodeOk " +
                "decode=$decodeOk conf=${recognition.conf1}/${recognition.conf2} " +
                "ambiguous=${result.ambiguous} altMargin=${result.altMargin} " +
                "structure=${timings["structure"]}ms " +
                "normalize=${timings["fast normalize"]}ms ocr=${timings["OCR + decode"]}ms " +
                "total=${timings["total"]}ms",
        )
        return PhotoMrzResult(
            original = source,
            noReflection = null,
            mrzCrop = bandBitmap,
            line1 = recognition.line1,
            line2 = recognition.line2,
            conf1 = recognition.conf1,
            conf2 = recognition.conf2,
            backend = recognition.backend,
            decodeOk = decodeOk,
            fields = if (decodeOk) decodeFields(recognition.line1, recognition.line2) else emptyMap(),
            timings = timings,
            error = note,
            rawDecodeOk = rawDecodeOk,
            repaired = recognition.repairCount > 0,
            ambiguous = result.ambiguous,
            altMargin = result.altMargin,
            altLine2 = result.altLine2,
            glarePermille = structure.glarePermille,
            structOk = recognition.structOk,
            natSuspect = recognition.natSuspect,
            countryMismatch = recognition.countryMismatch,
            issuerSuspect = recognition.issuerSuspect,
        )
    }

    /**
     * Fast-path band extraction: creates only the combined band bitmap,
     * skipping the per-line bitmaps that [nativeBand] produces for the
     * detailed path.  The sharedPad calculation matches nativeBand exactly
     * (padded row heights → shared pad), so the band is identical.
     *
     * [padSpec] scales the shared vertical pad (default 0.70 = the shipped
     * framing).  The framing alternates used by the band-candidate loop
     * pass 1.35 / 0.45 to re-frame photos whose MRZ only decodes with more
     * or less surrounding context (see FRAME_PAD_ALTERNATES in
     * [processFast] and 10_ocr_research/band_candidates_product_port.md).
     */
    private fun nativeBandFast(
        source: Bitmap,
        first: IntRange,
        second: IntRange,
        xpadAdaptive: Pair<Int, Int>,
        padSpec: Double = 0.70,
    ): Bitmap {
        val xpadL = xpadAdaptive.first
        val xpadR = xpadAdaptive.second
        val firstSourceY1 = min(source.height, first.last + 1)
        val row0Y0 = max(0, first.first)
        val row0Y1 = min(source.height, first.last + 1)
        val row0H = max(1, row0Y1 - row0Y0)
        val pad0 = max(8, min(25, (row0H * 0.70).roundToInt()))
        val row0Height = min(source.height, row0Y1 + pad0) - max(0, row0Y0 - pad0)
        val row1Y0 = max(0, second.first).coerceAtLeast(firstSourceY1)
        val row1Y1 = min(source.height, second.last + 1)
        val row1H = max(1, row1Y1 - row1Y0)
        val pad1 = max(8, min(25, (row1H * 0.70).roundToInt()))
        val row1Height = min(source.height, row1Y1 + pad1) - max(0, row1Y0 - pad1)
        val sharedHeight = max(row0Height, row1Height)
        // padSpec scales the shared pad; the clamp widens proportionally so
        // the alternates produce genuinely different framings (the shipped
        // 25 px ceiling would swallow a 1.35x pad on tall rows).
        val sharedPad = max(
            (8 * padSpec).roundToInt(),
            min((25 * padSpec).roundToInt(), (sharedHeight * padSpec).roundToInt()),
        )
        val firstTop = max(0, first.first)
        val secondBottom = min(source.height, second.last + 1)
        val cropTop = max(0, firstTop - sharedPad)
        val cropBottom = min(source.height, secondBottom + sharedPad)
        val width = max(1, source.width - xpadL - xpadR)
        return Bitmap.createBitmap(source, xpadL, cropTop, width, cropBottom - cropTop)
    }

    /** CS/CD 单行 band 提取：只裁一行加 padding。 */
    private fun nativeBandSingle(
        source: Bitmap,
        row: IntRange,
        xpadAdaptive: Pair<Int, Int>,
    ): Bitmap {
        val xpadL = xpadAdaptive.first
        val xpadR = xpadAdaptive.second
        val rowY0 = max(0, row.first)
        val rowY1 = min(source.height, row.last + 1)
        // NO vertical padding on a single-row band.  The C layout
        // classifier votes 30-vs-44 cells from the row's aspect ratio and
        // the dual-hypothesis grid energy, and extra rows around the
        // glyphs flip that vote: measured on the CS/CD permit
        // (assets/pic/1c6acd092dee7031_0222.png) a tight crop gives
        // CS_CD_30 and the exact code, +8 px padding gives TD3_44 and
        // garbage.  A TD3 band (two rows) still gets its padding.
        val cropTop = rowY0
        val cropBottom = rowY1
        val width = max(1, source.width - xpadL - xpadR)
        return Bitmap.createBitmap(source, xpadL, cropTop, width, cropBottom - cropTop)
    }

    private fun locateFast(source: Bitmap): StructureResult {
        val timings = linkedMapOf<String, Long>()
        val start = System.nanoTime()
        val pair = fastPair(source)
        timings["structure"] = elapsedMs(start)
        if (pair == null) {
            val single = fastSingleRow(source)
            if (single != null) {
                val xpad = adaptiveXpad(
                    source, source, PairRows(single, single, emptyList(), 0),
                )
                return StructureResult(
                    source = source, work = source, pair = null, noReflection = null,
                    mrzCrop = null, xpadAdaptive = xpad, timings = timings,
                    error = "", singleRow = single,
                )
            }
            return StructureResult(
                source = source, work = source, pair = null, noReflection = null,
                mrzCrop = null, xpadAdaptive = Pair(0, 0), timings = timings,
                error = "MRZ 双行未定位",
            )
        }
        val xpad = adaptiveXpad(source, source, pair)
        return StructureResult(
            source = source, work = source, pair = pair, noReflection = null,
            mrzCrop = null, xpadAdaptive = xpad, timings = timings,
            error = "", glarePermille = pair.glarePermille,
        )
    }

    /**
     * Row darkness profile, computed natively when the JNI library is
     * available and in Kotlin otherwise.
     *
     * The native version exists because the Kotlin loop ran ~500 k
     * interpreted luma computations per photo (25-38 ms on the device,
     * measured).  The Kotlin fallback keeps the JVM unit tests -- which
     * run without the native library -- working, and is also the path
     * used on a host JVM.
     */
    private fun rowProfile(
        pixels: IntArray,
        width: Int,
        y0: Int,
        y1: Int,
        x0: Int,
        x1: Int,
        stride: Int,
    ): IntArray {
        val rows = y1 - y0
        val native = try {
            MrzNative.rowProfile(pixels, width, y0, y1, x0, x1, stride, null)
        } catch (_: Throwable) {
            null
        } catch (_: UnsatisfiedLinkError) {
            null
        }
        if (native != null && native.size == rows) return native
        val dark = IntArray(rows)
        for (row in 0 until rows) {
            val offset = row * width
            var count = 0
            var x = x0
            while (x < x1) {
                val c = pixels[offset + x]
                val luma = ((c shr 16 and 0xFF) * 299 +
                    (c shr 8 and 0xFF) * 587 +
                    (c and 0xFF) * 114 + 500) / 1000
                if (luma < 170) count++
                x += stride
            }
            dark[row] = count
        }
        return dark
    }

    /**
     * 轻量行投影：只读取照片下半部，按 4x1 采样统计暗像素行。
     * 这里使用源图坐标，避免 createScaledBitmap 丢失小字符的垂直分辨率；
     * 精确裁 band 后再交给 native 做二次定位和识别。
     */
    /**
     * Row-pair locator with coarse-to-fine downscale.
     *
     * For high-res photos (max side > [LOCATE_MAX_SIDE]) the full-res
     * getPixels + rowProfile is the structure bottleneck (37-57ms on
     * 2592x1944).  Downsampling to 1280 for the row-profile analysis
     * cuts that to ~12ms; the pair coordinates are mapped back to source
     * resolution (the few-pixel imprecision is absorbed by nativeBand's
     * 8-25px padding).  Band extraction still uses the original source,
     * so CNN input quality is unchanged.
     */
    private fun fastPair(source: Bitmap): PairRows? {
        val maxSide = maxOf(source.width, source.height)
        if (maxSide <= LOCATE_DOWNSCALE_THRESHOLD) return fastPairImpl(source)
        val scale = LOCATE_TARGET_SIDE.toDouble() / maxSide
        val sw = max(1, (source.width * scale).toInt())
        val sh = max(1, (source.height * scale).toInt())
        val small = Bitmap.createScaledBitmap(source, sw, sh, true)
        val smallPair = fastPairImpl(small)
        small.recycle()
        if (smallPair == null) return null
        val invScale = 1.0 / scale
        return PairRows(
            (smallPair.first.first * invScale).toInt()..(smallPair.first.last * invScale).toInt(),
            (smallPair.second.first * invScale).toInt()..(smallPair.second.last * invScale).toInt(),
            smallPair.reflection,
            smallPair.glarePermille,
        )
    }

    private fun fastPairImpl(source: Bitmap): PairRows? {
        val width = source.width
        val height = source.height
        if (width < 80 || height < 120) return null
        val y0 = (height * 0.55).toInt()
        val y1 = (height * 0.96).toInt().coerceAtMost(height)
        val sampleX0 = (width * 0.03).toInt()
        val sampleX1 = (width * 0.97).toInt()
        val stride = 4
        val sampleWidth = ((sampleX1 - sampleX0) + stride - 1) / stride
        if (y1 <= y0 || sampleWidth < 40) return null

        // The pixel buffer has to be read out of the Bitmap either way;
        // the row profile itself is computed natively.  Doing the luma
        // arithmetic in Kotlin cost ~500 k interpreted iterations per
        // photo, measured at 25-38 ms on the device.
        val pixels = IntArray((y1 - y0) * width)
        source.getPixels(pixels, 0, width, 0, y0, width, y1 - y0)
        val dark = rowProfile(pixels, width, y0, y1, sampleX0, sampleX1, stride)
        // The same pixel buffer answers "is there a specular highlight
        // on the page": luma >= 250 is clipped, the strokes are gone,
        // and no threshold can bring them back.  The measurement rides
        // along with the row profile instead of adding a second read of
        // the photo.
        val glare = try {
            MrzNative.glareFraction(pixels, width, y0, y1, sampleX0, sampleX1, stride)
        } catch (_: Throwable) {
            0
        } catch (_: UnsatisfiedLinkError) {
            0
        }

        // 行亮度方差剖面：纯暗带里 rowProfile 是一整条满宽实心带，MRZ 两行
        // 无法分开；字符行的地面/笔画亮度交替让方差远高于均匀的带底，用
        // 它做 solid 带内的兜底定位。
        val rowStd = try {
            MrzNative.rowStdProfile(pixels, width, y0, y1, sampleX0, sampleX1, stride)
        } catch (_: Throwable) {
            null
        } catch (_: UnsatisfiedLinkError) {
            null
        }?.takeIf { it.size == y1 - y0 }

        val minInk = max(3, (sampleWidth * 0.18).toInt())
        val maxInk = (sampleWidth * 0.88).toInt()
        val runs = ArrayList<IntRange>()
        var runStart = -1
        var gap = 0
        for (i in dark.indices) {
            val active = dark[i] in minInk..maxInk
            if (active) {
                if (runStart < 0) runStart = i
                gap = 0
            } else if (runStart >= 0) {
                gap++
                if (gap > 2) {
                    runs += runStart..(i - gap)
                    runStart = -1
                    gap = 0
                }
            }
        }
        if (runStart >= 0) runs += runStart..dark.lastIndex

        val solidMin = (sampleWidth * 0.90).toInt()
        val solidRuns = ArrayList<IntRange>()
        var solidStart = -1
        for (i in 0..dark.size) {
            val solid = i < dark.size && dark[i] >= solidMin
            if (solid) {
                if (solidStart < 0) solidStart = i
            } else if (solidStart >= 0) {
                if (i - solidStart >= 8) solidRuns += solidStart..(i - 1)
                solidStart = -1
            }
        }

        fun pickPair(before: Int? = null): Pair<IntRange, IntRange>? {
            var candidate: Pair<IntRange, IntRange>? = null
            fun validPair(a: IntRange, b: IntRange): Boolean {
                if (before != null && b.last >= before - 2) return false
                val h1 = a.last - a.first + 1
                val h2 = b.last - b.first + 1
                val gapRows = b.first - a.last - 1
                if (h1 !in 5..100 || h2 !in 5..100) return false
                if (h2.toDouble() / h1 !in 0.35..2.5) return false
                if (gapRows !in 0..max(12, min(h1, h2) * 2 + 4)) return false
                return true
            }

            for (i in 0 until runs.size - 1) {
                val a = runs[i]
                val h1 = a.last - a.first + 1
                if (h1 !in 5..100) continue
                val last = min(runs.lastIndex, i + 3)
                for (j in (i + 1)..last) {
                    val b = runs[j]
                    val h2 = b.last - b.first + 1
                    val maxNoiseHeight = max(2, min(h1, h2) / 3)
                    val skipped = runs.subList(i + 1, j)
                    if (skipped.any { it.last - it.first + 1 > maxNoiseHeight }) continue
                    if (!validPair(a, b)) continue
                    if (candidate == null || b.last > candidate.second.last) candidate = a to b
                }
            }
            return candidate
        }

        // 主路径：真实 MRZ 位于观察黑带上方；反射 MRZ 通常出现在黑带之后，
        // 因此在候选对中优先选择黑带上方最近的一对。
        var best: Pair<IntRange, IntRange>? = null
        for (solid in solidRuns) {
            val candidate = pickPair(solid.first)
            if (candidate != null &&
                (best == null || candidate.second.last > best!!.second.last)
            ) {
                best = candidate
            }
        }
        if (best == null) best = pickPair()
        Log.i("MrzOcr", "locator dark path: $best")
        // 兜底：只有 dark 剖面完全找不到双行对时才启用行亮度方差峰对。
        // 这样 I/J/K 这类 MRZ 印在暗色面板内的照片（rowProfile 是一条满宽
        // 实心带、dark 两行分不开）能被救回，又不影响 dark 主路径已正常
        // 判定的普通照片。
        if (best == null && rowStd != null) {
            val peaks = variancePeaks(rowStd, sampleWidth)
            val candidate = pairFromVariancePeaks(peaks, dark, maxInk)
            Log.i(
                "MrzOcr",
                "locator var-fallback img-h=$height rowStd=${rowStd.size} " +
                    "sw=$sampleWidth thr=${maxOf(40000.0, sampleWidth * 120.0).toInt()} " +
                    "peaks=${peaks.joinToString { "[${it.first},${it.last}]" }} " +
                    "cand=$candidate",
            )
            if (candidate != null) best = candidate
        }
        return best?.let { (a, b) ->
            Log.i("MrzOcr", "locator chosen dark(y=${a.first}-${b.last}) y0=$y0")
            PairRows(y0 + a.first..y0 + a.last, y0 + b.first..y0 + b.last, emptyList(), glare)
        }
    }

    /** 单行 MRZ 定位（CS/CD 港澳台通行证）：与 fastPair 相同的降采样策略。 */
    private fun fastSingleRow(source: Bitmap): IntRange? {
        val maxSide = maxOf(source.width, source.height)
        if (maxSide <= LOCATE_DOWNSCALE_THRESHOLD) return fastSingleRowImpl(source)
        val scale = LOCATE_TARGET_SIDE.toDouble() / maxSide
        val sw = max(1, (source.width * scale).toInt())
        val sh = max(1, (source.height * scale).toInt())
        val small = Bitmap.createScaledBitmap(source, sw, sh, true)
        val smallRow = fastSingleRowImpl(small)
        small.recycle()
        if (smallRow == null) return null
        val invScale = 1.0 / scale
        return (smallRow.first * invScale).toInt()..(smallRow.last * invScale).toInt()
    }

    /** 单行 MRZ 定位实现：在下半部寻找最长的暗行 run 作为文本行。 */
    private fun fastSingleRowImpl(source: Bitmap): IntRange? {
        val width = source.width
        val height = source.height
        if (width < 80 || height < 120) return null
        val y0 = (height * 0.55).toInt()
        val y1 = (height * 0.96).toInt().coerceAtMost(height)
        val sampleX0 = (width * 0.03).toInt()
        val sampleX1 = (width * 0.97).toInt()
        val stride = 4
        val sampleWidth = ((sampleX1 - sampleX0) + stride - 1) / stride
        if (y1 <= y0 || sampleWidth < 40) return null

        val pixels = IntArray((y1 - y0) * width)
        source.getPixels(pixels, 0, width, 0, y0, width, y1 - y0)
        val dark = rowProfile(pixels, width, y0, y1, sampleX0, sampleX1, stride)

        val minInk = max(3, (sampleWidth * 0.18).toInt())
        val maxInk = (sampleWidth * 0.88).toInt()
        val runs = ArrayList<IntRange>()
        var runStart = -1
        var gap = 0
        for (i in dark.indices) {
            val active = dark[i] in minInk..maxInk
            if (active) {
                if (runStart < 0) runStart = i
                gap = 0
            } else if (runStart >= 0) {
                gap++
                if (gap > 2) {
                    runs += runStart..(i - gap)
                    runStart = -1
                    gap = 0
                }
            }
        }
        if (runStart >= 0) runs += runStart..dark.lastIndex

        var best: IntRange? = null
        for (run in runs) {
            val h = run.last - run.first + 1
            if (h !in 5..100) continue
            if (best == null || h > best!!.last - best!!.first + 1) best = run
        }

        if (best == null) {
            val rowStd = try {
                MrzNative.rowStdProfile(pixels, width, y0, y1, sampleX0, sampleX1, stride)
            } catch (_: Throwable) {
                null
            } catch (_: UnsatisfiedLinkError) {
                null
            }?.takeIf { it.size == y1 - y0 }
            if (rowStd != null) {
                val peaks = variancePeaks(rowStd, sampleWidth)
                if (peaks.isNotEmpty()) best = peaks.last()
            }
        }

        return best?.let {
            Log.i("MrzOcr", "locator single-row chosen y=${y0 + it.first}-${y0 + it.last}")
            (y0 + it.first)..(y0 + it.last)
        }
    }

    /** 行亮度方差(×100)剖面里高于阈值的连续峰段，gap<=2 的峰合并。 */
    private fun variancePeaks(
        rowStd: IntArray,
        sw: Int,
    ): List<IntRange> {
        val n = rowStd.size
        if (n < 16) return emptyList()
        // 绝对阈值（native 已按 x100 缩放）：字符行 ~7 万-60 万，均匀面板
        // ~几千。用采样宽度缩放避免极窄扫描区把大面积字符行误判。
        val thr = maxOf(40000.0, sw * 120.0)

        val lines = ArrayList<IntRange>()
        val from = (n * 15) / 100  // 跳过页首干扰区，MRZ 在照片下部
        var start = -1
        for (i in from until n) {
            if (rowStd[i].toDouble() > thr) {
                if (start < 0) start = i
            } else if (start >= 0) {
                if (i - start >= 2) lines += start..(i - 1)
                start = -1
            }
        }
        if (start >= 0 && n - start >= 2) lines += start..(n - 1)

        // 合并相距 <=2 的峰（单个低方差谷值把一行切成两段时）
        val merged = ArrayList<IntRange>()
        for (rl in lines) {
            if (merged.isNotEmpty() && rl.first - merged.last().last <= 2) {
                val prev = merged.removeAt(merged.size - 1)
                merged += prev.first..maxOf(prev.last, rl.last)
            } else {
                merged += rl
            }
        }
        return merged
    }

    /** 从峰段里取「最靠下」的相邻合法对（MRZ 两行）。
     *  dark/maxInk 用来过滤纯黑带边缘产生的方差峰——这些峰的 dark 值
     *  远超 maxInk（整行几乎全黑），不是真正的文字行。 */
    private fun pairFromVariancePeaks(
        peaks: List<IntRange>,
        dark: IntArray? = null,
        maxInk: Int = Int.MAX_VALUE,
    ): Pair<IntRange, IntRange>? {
        /* The "solid bar" test must NOT use the text row's 88% ink
         * window.  On a dark panel the MRZ rows themselves are nearly
         * full-width in ink, so the 88% test classified them as solid
         * and filtered them out; the fallback then paired the two peaks
         * ABOVE them and cropped the wrong band.
         *
         * Measured on the device (2026-11-05, 192.168.26.131): with the
         * 88% window, 2026-9-30-I/J/K picked cand=(120..161, 184..208)
         * instead of the real rows (301..331, 351..367) and all three
         * ended as FAIL with a garbage CS/CD reading.  The solid limit
         * is now 11% above the text window, which still rejects the
         * 取景框 black bar but keeps the MRZ rows. */
        /* A peak is only a device black bar (取景框) when it is solid AND
         * tall.  A solid-but-short peak is an MRZ row printed on a dark
         * panel: on 2026-9-30-I/J/K the whole row is ink, so ANY ink
         * threshold classifies it as solid and the fallback then paired
         * the peaks above it and cropped the wrong band.
         *
         * MEASURED on the device (2026-11-05, 192.168.26.131): with the
         * 88% window the three photos picked cand=(120..161, 184..208)
         * instead of the real rows (301..331, 351..367) and all three
         * ended as FAIL with a garbage CS/CD reading.  With the height
         * condition the real rows survive (an MRZ row is 5..40 rows tall,
         * a black bar is much taller) and the bottom-most pair -- the
         * one the fallback already prefers -- is chosen. */
        val solidLimit = (maxInk * 1.11).toInt()
        val solidMinHeight = 60
        fun isSolid(peak: IntRange): Boolean {
            if (dark == null) return false
            val len = peak.last - peak.first + 1
            if (len <= 0) return false
            if (len < solidMinHeight) return false
            var sum = 0
            for (i in peak.first..peak.last) sum += dark[i]
            return sum / len > solidLimit
        }
        val filtered = peaks.filterNot { isSolid(it) }
        var best: Pair<IntRange, IntRange>? = null
        for (i in 0 until filtered.size - 1) {
            val a = filtered[i]
            val b = filtered[i + 1]
            val h1 = a.last - a.first + 1
            val h2 = b.last - b.first + 1
            if (h1 < 3 || h2 < 3) continue
            // 行高比放宽到 0.15..3（暗面板里 line2 常被截短，行高比可远小于 1）
            if (h2.toDouble() / h1 !in 0.15..3.0) continue
            val gap = b.first - a.last - 1
            if (gap > max(14, (1.5 * maxOf(h1, h2)).toInt() + 6)) continue
            if (best == null || b.last > best!!.second.last) best = a to b
        }
        return best
    }

    private fun processDetailed(
        context: Context,
        source: Bitmap,
        method: String,
        onStage: ((MrzStage) -> Unit)? = null,
    ): PhotoMrzResult {
        val allStart = System.nanoTime()
        val structure = processStructure(source, onStage)
        val timings = structure.timings
        if (structure.error.isNotEmpty() || structure.pair == null) {
            timings["total"] = elapsedMs(allStart)
            return PhotoMrzResult(
                original = structure.source,
                noReflection = structure.noReflection,
                mrzCrop = structure.mrzCrop,
                line1 = "",
                line2 = "",
                conf1 = 0,
                conf2 = 0,
                backend = "none",
                decodeOk = false,
                fields = emptyMap(),
                timings = timings,
                error = structure.error.ifEmpty { "MRZ 双行未定位" },
            )
        }

        // 定位优先用 fastPair（源坐标，与 PC eval 的 fast_pair 逐位一致）；
        // 仅当 fastPair 未命中（返回 null，一般意味着页面没有标准 MRZ 双行）
        // 时才回退 processStructure.findPair 在缩放 work 坐标上的结果。
        val fast = fastPair(source)
        val nativeBand =
            if (fast != null) {
                nativeBand(source, fast.first, fast.second, 1.0,
                    adaptiveXpad(source, source, fast))
            } else {
                nativeBand(source, structure.pair.first, structure.pair.second,
                    scaleFor(structure.source), structure.xpadAdaptive)
            }
        val ocrStart = System.nanoTime()
        val engines = enginesForMethod(context, method)
        val nativeOnly = engines.size == 1 && engines[0] is NativeEngine
        val attempts = if (nativeOnly) {
            // 与 PC eval（native_band 整带 + band.ppm）一致：喂整带而不是
            // 两行独立位图。stackBand 用 25% 白底 gutter 重堆叠会改变行距，
            // 导致 CNN 输入与训练分布不一致（41 张评测 25/41 → 40/41）。
            val engine = engines[0] as NativeEngine
            listOf(engine.recognizeBitmap(fastNativeInput(nativeBand.bitmap), 0))
        } else {
            OcrDispatch.run(
                engines, nativeBand.line1, nativeBand.line2,
                stopWhen = null,
            )
        }
        val processed = attempts.map { result ->
            postProcess(
                result.line1, result.line2, result.conf1, result.conf2,
                backend = result.engineName,
                note = result.error ?: "",
                structOk = result.structOk,
                natSuspect = result.natSuspect,
                countryMismatch = result.countryMismatch,
                issuerSuspect = result.issuerSuspect,
                ambiguous = result.ambiguous,
                altMargin = result.altMargin,
                altLine2 = result.altLine2,
            )
        }
        val recognition = OcrDispatch.pick(
            processed,
            backend = { it.backend },
            confidence = { it.conf1 + it.conf2 },
            decodes = { decodesOk(it) },
        ) ?: Recognition("", "", 0, 0, "none", "OCR 引擎无候选结果")
        android.util.Log.i(
            "MrzOcr",
            "chosen=${recognition.backend} conf=${recognition.conf1}/${recognition.conf2} " +
                "rawDecode=${isStrictTd3(recognition.rawLine1, recognition.rawLine2)} " +
                "decodes=${decodesOk(recognition)}",
        )
        timings["OCR + decode"] = elapsedMs(ocrStart)
        val rawDecodeOk = isStrictTd3(recognition.rawLine1, recognition.rawLine2)
        val decodeOk = decodesOk(recognition)
        val fields = if (decodeOk) decodeFields(recognition.line1, recognition.line2) else emptyMap()
        timings["total"] = elapsedMs(allStart)
        return PhotoMrzResult(
            original = structure.source,
            noReflection = structure.noReflection,
            mrzCrop = structure.mrzCrop,
            line1 = recognition.line1,
            line2 = recognition.line2,
            conf1 = recognition.conf1,
            conf2 = recognition.conf2,
            backend = recognition.backend,
            decodeOk = decodeOk,
            fields = fields,
            timings = timings,
            error = listOf(
                recognition.note,
                if (!recognition.structOk) "字段结构不可能（日期非法）" else "",
                if (recognition.natSuspect) "国籍码可疑" else "",
                if (recognition.countryMismatch) "签发国与国籍不一致" else "",
                if (recognition.issuerSuspect) "签发国码可疑" else "",
            ).filter { it.isNotEmpty() }.joinToString("; "),
            rawDecodeOk = rawDecodeOk,
            repaired = recognition.repairCount > 0,
            ambiguous = recognition.ambiguous,
            altMargin = recognition.altMargin,
            altLine2 = recognition.altLine2,
            glarePermille = structure.glarePermille,
            structOk = recognition.structOk,
            natSuspect = recognition.natSuspect,
            countryMismatch = recognition.countryMismatch,
            issuerSuspect = recognition.issuerSuspect,
        )
    }

    fun processStructure(
        source: Bitmap,
        onStage: ((MrzStage) -> Unit)? = null,
    ): StructureResult {
        val timings = linkedMapOf<String, Long>()
        onStage?.invoke(MrzStage("original", "\u2460 original", source))
        val workStart = System.nanoTime()
        val scale = scaleFor(source)
        val work = if (scale < 1.0) {
            // 与 restore.load 的 int(w*scale) 截断一致：roundToInt 会把
            // 1944*scale=937.5 抬成 938，多出的 1 行使双行检测整体偏移 1px。
            Bitmap.createScaledBitmap(
                source,
                max(1, (source.width * scale).toInt()),
                max(1, (source.height * scale).toInt()),
                true,
            )
        } else source
        val gray = grayscale(work)
        val pair = findPair(gray)
        timings["structure"] = elapsedMs(workStart)
        if (pair == null) {
            return StructureResult(
                source = source, work = work, pair = null, noReflection = work,
                mrzCrop = null, xpadAdaptive = Pair(0, 0), timings = timings,
                error = "MRZ 双行未定位",
            )
        }

        val xpad = adaptiveXpad(source, work, pair)

        val cropStart = System.nanoTime()
        val cropBottom = pair.second.last + 1
        val noReflection = Bitmap.createBitmap(work, 0, 0, work.width, cropBottom)
        timings["remove reflection"] = elapsedMs(cropStart)
        onStage?.invoke(MrzStage("clean", "\u2461 reflection removed", noReflection))

        val mrzCropStart = System.nanoTime()
        val mrzCrop = cropPairClean(noReflection, pair.first, pair.second)
        timings["MRZ crop"] = elapsedMs(mrzCropStart)
        onStage?.invoke(MrzStage("mrz", "\u2463 MRZ crop", mrzCrop))

        return StructureResult(
            source = source, work = work, pair = pair, noReflection = noReflection,
            mrzCrop = mrzCrop, xpadAdaptive = xpad, timings = timings,
        )
    }

    /**
     * F3 边条框检测自适应 xpad：机位照左右各取 8 列（strip∈[4,12]）
     * 计算整列暗占比；有框（>=0.9）一侧保留名义 xpad，无框一侧返回 0。
     * 对齐 Python _rst_adaptive_xpad v4：机位照 1.00 → 100/51；满幅照 0.00-0.67 → 0。
     * 返回值在源图坐标系（与 nativeBand 的 source.width 对齐）。
     */
    fun adaptiveXpad(source: Bitmap, work: Bitmap, pair: PairRows): Pair<Int, Int> {
        val Wsrc = source.width
        if (Wsrc < 40 || pair.first.isEmpty() || pair.second.isEmpty()) {
            val nominal = min(100, max(10, (Wsrc * 0.08).roundToInt()))
            return Pair(nominal, nominal)
        }
        val strip = max(4, min(12, work.width / 25))
        val leftDark = stripDarkFrac(work, 0, strip)
        val rightDark = stripDarkFrac(work, work.width - strip, work.width)
        val nominal = min(100, max(10, (Wsrc * 0.08).roundToInt()))
        var left = if (leftDark >= 0.9) nominal else 0
        var right = if (rightDark >= 0.9) nominal else 0
        // Never crop INTO the MRZ ink.  The nominal pad is a fraction of
        // the image width, so on a narrow document it can reach past the
        // first or last character: measured on the CS/CD permit
        // (assets/pic/1c6acd092dee7031_0222.png) a 20 px crop already
        // dropped the trailing characters and 40 px cut into the code.
        // Clamp the pad to the ink extent measured inside the rows.
        val ink = inkExtent(source, pair)
        if (ink != null) {
            left = min(left, max(0, ink.first))
            right = min(right, max(0, Wsrc - 1 - ink.second))
        }
        return Pair(left, right)
    }

    /** Leftmost and rightmost column that carry ink inside the MRZ rows. */
    private fun inkExtent(source: Bitmap, pair: PairRows): Pair<Int, Int>? {
        val y0 = max(0, pair.first.first)
        val y1 = min(source.height, pair.first.last + 1)
        val y2 = max(0, pair.second.first)
        val y3 = min(source.height, pair.second.last + 1)
        val top = min(y0, y2)
        val bottom = max(y1, y3)
        if (bottom <= top) return null
        val width = source.width
        val rows = bottom - top
        val pixels = IntArray(width * rows)
        source.getPixels(pixels, 0, width, 0, top, width, rows)
        // A column counts as a CHARACTER column only when its ink does
        // not run through the whole row height: the camera frame's black
        // border (and page edge lines) do exactly that, and treating
        // them as characters would cancel the very pad that removes
        // them.
        var left = -1
        var right = -1
        for (x in 0 until width) {
            var inked = 0
            for (y in 0 until rows) {
                val px = pixels[y * width + x]
                val r = (px shr 16) and 0xFF
                val g = (px shr 8) and 0xFF
                val b = px and 0xFF
                val luma = (299 * r + 587 * g + 114 * b + 500) / 1000
                if (luma < 160) inked++
            }
            if (inked * 100 >= rows * 95) continue   // frame / page edge
            if (inked > 0) {
                if (left < 0) left = x
                right = x
            }
        }
        return if (left < 0) null else Pair(left, right)
    }

    private fun stripDarkFrac(work: Bitmap, x0: Int, x1: Int): Double {
        val W = x1 - x0
        if (W <= 0) return 0.0
        val H = work.height
        val pixels = IntArray(W * H)
        work.getPixels(pixels, 0, W, x0, 0, W, H)
        var dark = 0
        for (px in pixels) {
            val r = (px shr 16) and 0xFF
            val g = (px shr 8) and 0xFF
            val b = px and 0xFF
            // ITU-R BT.601 luma, 0-255 范围
            val luma = (299 * r + 587 * g + 114 * b + 500) / 1000
            if (luma < 160) dark++
        }
        return dark.toDouble() / pixels.size
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
        val rawLine1: String = "",
        val rawLine2: String = "",
        val repairCount: Int = 0,
        /** 校验位之外的否决位，来自引擎（见 OcrResult.structOk）。 */
        val structOk: Boolean = true,
        /** 国籍码可疑（见 OcrResult.natSuspect）。 */
        val natSuspect: Boolean = false,
        /** 签发国与国籍不一致（见 OcrResult.countryMismatch）。 */
        val countryMismatch: Boolean = false,
        /** 签发国码可疑（见 OcrResult.issuerSuspect）。 */
        val issuerSuspect: Boolean = false,
        /** 存在同样校验合法、概率接近的第二读法（见 OcrResult.ambiguous）。 */
        val ambiguous: Boolean = false,
        /** 首选与第二读法的 log-prob 差（千分位，见 OcrResult.altMargin）。 */
        val altMargin: Int = 0,
        /** 第二读法（见 OcrResult.altLine2）。 */
        val altLine2: String = "",
    )

    private fun postProcess(
        rawLine1: String,
        rawLine2: String,
        conf1: Int,
        conf2: Int,
        backend: String,
        note: String = "",
        structOk: Boolean = true,
        natSuspect: Boolean = false,
        countryMismatch: Boolean = false,
        issuerSuspect: Boolean = false,
        ambiguous: Boolean = false,
        altMargin: Int = 0,
        altLine2: String = "",
    ): Recognition {
        val raw1 = normalizeRaw(rawLine1)
        val raw2 = normalizeRaw(rawLine2)
        val line1 = enforceLine1(raw1)
        // 超时/失败结果可能只带少量残留字符；不要把补齐后的 "<" 当成
        // 可修复 OCR 输入，也不要为 deadline 结果生成伪 repair note。
        val (line2, repairs) = if (
            raw2.length >= 40 &&
            !note.contains("timeout", ignoreCase = true) &&
            !note.contains("deadline", ignoreCase = true)
        ) {
            repairLine2(raw2)
        } else {
            raw2 to emptyList()
        }
        val repairNote = repairs.joinToString("; ")
        val line1Repairs = if (raw1.length == line1.length) {
            raw1.indices.count { raw1[it] != line1[it] }
        } else {
            0
        }
        return Recognition(
            line1,
            line2,
            conf1,
            conf2,
            backend,
            listOf(note, repairNote).filter { it.isNotEmpty() }.joinToString("; "),
            raw1,
            raw2,
            line1Repairs + repairs.size,
            structOk,
            natSuspect,
            countryMismatch,
            issuerSuspect,
            ambiguous,
            altMargin,
            altLine2,
        )
    }

    private fun decodesOk(recognition: Recognition): Boolean {
        if (isCsCdShape(recognition.line1, recognition.line2)) return true
        if (!isStrictTd3(recognition.line1, recognition.line2)) return false
        if (isStrictTd3(recognition.rawLine1, recognition.rawLine2)) return true
        // A repaired result is accepted only when the OCR supplied two
        // complete MRZ-shaped rows and made a very small correction.
        return recognition.rawLine1.length == 44 &&
            recognition.rawLine2.length == 44 &&
            recognition.repairCount in 1..2 &&
            recognition.conf1 >= MIN_REPAIR_CONF &&
            recognition.conf2 >= MIN_REPAIR_CONF &&
            isMrzShape(recognition.rawLine1, recognition.rawLine2)
    }

    /** CS/CD 港澳台通行证: 单行 30 字符, line2 为空. */
    private fun isCsCdShape(line1: String, line2: String): Boolean {
        if (line1.length != 30 || line2.isNotEmpty()) return false
        if (line1[0] != 'C') return false
        if (line1[1] != 'S' && line1[1] != 'D') return false
        // Positions 4-11, 13-19, 21-27, 29 must be digits
        for (i in listOf(4,5,6,7,8,9,10,11, 13,14,15,16,17,18,19, 21,22,23,24,25,26,27, 29)) {
            if (i < line1.length && !line1[i].isDigit()) return false
        }
        // Separators at 12, 20, 28
        for (i in listOf(12, 20, 28)) {
            if (i < line1.length && line1[i] != '<') return false
        }
        return true
    }

    private fun isStrictTd3(line1: String, line2: String): Boolean =
        isMrzShape(line1, line2) &&
            line1[0] == 'P' &&
            line1[1] == '<' &&
            line1.substring(2, 5).all { it in 'A'..'Z' } &&
            line1.substring(5).contains("<<") &&
            line1.substring(5).all { it == '<' || it in 'A'..'Z' } &&
            IcaoChecksum.line2Ok(line2)

    private fun isMrzShape(line1: String, line2: String): Boolean {
        if (line1.length != 44 || line2.length != 44) return false
        if (!line1.all { it == '<' || it in 'A'..'Z' }) return false
        if (!line2.all { it == '<' || it in 'A'..'Z' || it in '0'..'9' }) return false
        if (!line2[9].isDigit() || !line2[19].isDigit() ||
            !line2[27].isDigit() || !line2[42].isDigit() || !line2[43].isDigit()
        ) return false
        if (!line2.substring(10, 13).all { it in 'A'..'Z' }) return false
        if (!line2.substring(13, 19).all { it.isDigit() } ||
            !line2.substring(21, 27).all { it.isDigit() }
        ) return false
        return line2[20] == 'M' || line2[20] == 'F' || line2[20] == '<'
    }

    // BEGIN GENERATED CONFUSION PAIRS (tools/gen_confusion_tables.py -- do not edit by hand)
    private val confusionPairs = listOf(
        // curated base
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
        // measured on real photos (tools/collect_confusions.py)
        "0" to "1",  // n=9
        "L" to "P",  // n=6
        "A" to "N",  // n=5

// END GENERATED CONFUSION PAIRS
    ).flatMap { (a, b) ->
        val cost = if (a in "0152689" || b in "0152689") 1 else 2
        listOf(a[0] to (b[0] to cost), b[0] to (a[0] to cost))
    }.groupBy({ it.first }, { it.second })
        .mapValues { (_, values) -> values.distinctBy { it.first }.sortedBy { it.second } }

    /**
     * F4 国家码 O/0 归一：line2 [10..12] 是 3 字母国家码，常见 OCR 误读仅 6 类。
     * 通用 confusionPairs 在该窗口可能给出 0→Q/U/D 等误归一（成本 2），收紧到仅 O/0 等
     * 1-成本映射，与 Python _mrz_line2_candidates country_fix 对齐。
     */
    private val countryFix = mapOf(
        '0' to 'O', '1' to 'I', '2' to 'Z', '5' to 'S', '6' to 'G', '8' to 'B',
    )

    private fun enforceLine1(raw: String): String {
        val out = normalizeRaw(raw).padEnd(44, '<').take(44).toCharArray()
        // CS/CD 港澳台通行证: 不强制 P< 前缀, 保留原始文档类型
        if (out[0] == 'C' && (out[1] == 'S' || out[1] == 'D')) {
            for (i in out.indices) {
                if (i > 1 && out[i].isDigit()) {
                    val candidate = confusionPairs[out[i]]
                        ?.firstOrNull { it.first.isLetter() }
                        ?.first
                    if (candidate != null) out[i] = candidate
                }
            }
            return out.concatToString()
        }
        // This pipeline is passport-only. TD3's first two slots are fixed
        // and should not make an otherwise valid checksum result fail.
        out[0] = 'P'
        out[1] = '<'
        for (i in out.indices) {
            if (i > 1 && out[i].isDigit()) {
                val candidate = confusionPairs[out[i]]
                    ?.firstOrNull { it.first.isLetter() }
                    ?.first
                if (candidate != null) out[i] = candidate
            }
        }
        return out.concatToString()
    }

    /**
     * F5 证据式 line2 修复：只改校验位，绝不改数据位。
     *
     * 历史实现（搜索式修复）会在 4 个数据字段 + 5 个校验位上做组合搜索，
     * 找到第一个能让 5 个校验位全部通过的候选就采纳。实测（41 张真实护照
     * 照片，板端弱标签 + ICAO 校验位派生真值）结果是：
     *   准确率：改善 0 / 恶化 8 / 中性 2
     *   校验位通过率：从 31/41 抬到 41/41
     * 也就是说它从不提升识别准确率，只是把"校验位不合法"的结果强行凑成
     * 合法，从而把 18 张实际识别错误的样本报成 decode=true（假成功）。
     *
     * 现在只做有证据的修复：数据位（证件号/出生日期/有效期/个人号）保持
     * 原样不动，只重算 5 个校验位。校验位是唯一能由数据位"证明"出错的
     * 位置——数据位错了应该由 OCR 层（native 的概率 beam）去纠正，而不是
     * 在这里靠搜索蒙。修复后的行仍必须通过全部 5 个校验位才被认为有效。
     */
    private fun repairLine2(raw: String): Pair<String, List<String>> {
        val filtered = normalizeRaw(raw)
        val aligned = alignNear44(filtered)
        val base = (aligned ?: filtered).padEnd(44, '<').take(44)
        val normalized = enforceLine2Slots(base)
        if (IcaoChecksum.line2Ok(normalized)) return normalized to emptyList()

        // 证据式修复：只重算校验位，数据位保持不变。
        val chars = normalized.toCharArray()
        val reports = ArrayList<Pair<Int, Char>>()
        fun setCheck(position: Int, value: Char) {
            if (chars[position] != value) {
                reports += position to value
                chars[position] = value
            }
        }
        setCheck(9, IcaoChecksum.digitFor(normalized.substring(0, 9)).toString()[0])
        setCheck(19, IcaoChecksum.digitFor(normalized.substring(13, 19)).toString()[0])
        setCheck(27, IcaoChecksum.digitFor(normalized.substring(21, 27)).toString()[0])
        setCheck(42, IcaoChecksum.digitFor(normalized.substring(28, 42)).toString()[0])
        setCheck(
            43,
            IcaoChecksum.digitFor(
                normalized.substring(0, 10) +
                    normalized.substring(13, 20) +
                    normalized.substring(21, 43),
            ).toString()[0],
        )
        val trial = chars.concatToString()
        // 只改了校验位却仍然过不了全部校验位，说明数据位本身有问题，
        // 此时不应再输出一个"看起来合法"的行。
        if (!IcaoChecksum.line2Ok(trial)) return normalized to emptyList()

        val original = normalized
        val notes = reports.map { (position, replacement) ->
            "pos $position ${original[position]}->$replacement"
        }
        return trial to notes
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
                // F4 国家码 10..12：仅用 countryFix 收紧集合（不引 0→Q/U/D 等）
                countryFix[out[i]]?.let { out[i] = it }
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
        // 桌面链固定 170/160/150；无解时走 tallSplitPass 大段底部切开兜底。
        for (thr in intArrayOf(170, 160, 150)) {
            val r = mrzPairPass(gray, thr, rowInk, edge)
            android.util.Log.i("MrzOcr", "findPair thr=$thr -> ${r?.first}")
            if (r != null) cands += r
        }
        val stdBest = cands.maxByOrNull { it.first.first }
        // 倾斜拍摄（AF）会把真 MRZ 行拆成窄段（thr 掩膜下 h~6/12，ratio>1.6 被规则
        // 拒绝），而页中部标签行冒充有效对。包络扩展（行 ink>0.05W 补全行高）后取
        // 位置最低的合法对；仅当其明显低于常规结果时采用（真 MRZ 恒在页面底部）。
        val envCands = ArrayList<PickResult>()
        for (thr in intArrayOf(170, 160, 150)) {
            val r = mrzPairExpandPass(gray, thr, edge)
            android.util.Log.i("MrzOcr", "findPair env thr=$thr -> ${r?.first}+${r?.second}")
            if (r != null) envCands += r
        }
        val envBest = envCands.maxByOrNull { it.first.first }
        val best = when {
            stdBest != null && envBest != null ->
                if (envBest.first.first > stdBest.first.first + 10) envBest else stdBest
            stdBest != null -> stdBest
            envBest != null -> envBest
            else -> null
        }
        if (best != null) return attachReflection(best, rowInk, gray.width)
        // 常规与包络均无解 → 大段底部切开兜底（first-success，降序阈值）
        for (thr in intArrayOf(160, 155, 152, 150, 148, 146, 144, 142, 140, 138)) {
            val r = tallSplitPass(gray, thr, edge)
            android.util.Log.i("MrzOcr", "findPair tallSplit thr=$thr -> ${r?.first}+${r?.second}")
            if (r != null) return attachReflection(r, rowInk, gray.width)
        }
        return null
    }

    private data class PickResult(val first: IntRange, val second: IntRange)

    private fun lineShapeOf(gray: Gray, thr: Int, run: IntRange): Boolean {
        val h = run.last - run.first + 1
        if (h < 5 || h > 48) return false
        val W = gray.width
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

    /**
     * 兜底：全息底纹把 MRZ 行1 桥接进上方大文本段且无阈值可拆时（AF 图桥接行
     * ink 距 0.20W 门限仅差 1），取该大段底部（3 行平滑 ink 谷处切开）作为
     * 行1 候选，再对后续文本段走 pick 规则。
     */
    private fun tallSplitPass(gray: Gray, thr: Int, edge: IntArray): PickResult? {
        val H = gray.height
        val W = gray.width
        val rowInkS = IntArray(H)
        for (y in 0 until H) {
            var ink = 0
            for (x in 0 until W) if (gray.pixels[y * W + x] < thr) ink++
            rowInkS[y] = ink
        }
        val text = BooleanArray(H) {
            rowInkS[it] > W * 0.20 && rowInkS[it] < W * 0.90 && edge[it] > 1800
        }
        val textB = groups(text, 4)
        var best: Triple<IntArray, IntRange, IntRange>? = null
        for (i in 0 until textB.size - 1) {
            val r = textB[i]
            if (r.last - r.first + 1 <= 48) continue
            val lo = maxOf(r.first, r.last - 47)
            val hi = r.last - 8
            if (hi <= lo) continue
            var s = lo
            var bestScore = Int.MAX_VALUE
            for (y in lo..hi) {
                val score = rowInkS[maxOf(0, y - 1)] + rowInkS[y] + rowInkS[minOf(H - 1, y + 1)]
                if (score < bestScore) {
                    bestScore = score
                    s = y
                }
            }
            val tail = (s + 1)..r.last
            val h1 = tail.last - tail.first + 1
            if (h1 < 4 || !lineShapeOf(gray, thr, tail)) continue
            for (j in i + 1 until textB.size) {
                val t2 = textB[j]
                val gap = t2.first - r.last - 1
                if (gap > max(8, 48 * 3)) break
                val h2 = t2.last - t2.first + 1
                if (!lineShapeOf(gray, thr, t2)) continue
                if (h2.toDouble() / h1 !in 0.4..1.6) continue
                val gapMax = max(8, (h1 * 3.0).toInt())
                if (gap < 0 || gap > gapMax) continue
                val rank = intArrayOf(-minOf(h1, h2), abs(h1 - h2) + gap, gap)
                if (best == null || compareInts(rank, best.first) < 0) {
                    best = Triple(rank, tail, t2)
                }
            }
        }
        return best?.let { PickResult(it.second, it.third) }
    }

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
                it.last - it.first >= 10
        }
        val sTop = if (cand.isNotEmpty()) cand.last().last + 1 else null

        fun lineShape(run: IntRange): Boolean = lineShapeOf(gray, thr, run)

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
        if (textB.size >= 2) pick(textB)?.let { return it }
        android.util.Log.i(
            "MrzOcr",
            "pass thr=$thr miss solid=[${solidB.joinToString(",") { "${it.first}-${it.last}" }}] " +
                "text=[${textB.joinToString(",") { "${it.first}-${it.last}" }}] sTop=$sTop",
        )
        return null
    }

    /**
     * 包络扩展对探测：倾斜页面的 MRZ 行在行投影里被拆成窄段，按行 ink>0.05W
     * 向上/下补齐为整行包络后走 pick 规则；返回位置最低（y0 最大）的合法对，
     * 供 findPair 与常规结果比较（真 MRZ 恒在页面底部）。
     */
    private fun mrzPairExpandPass(
        gray: Gray,
        thr: Int,
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
        val text = BooleanArray(H) {
            rowInkS[it] > W * 0.20 && rowInkS[it] < W * 0.90 && edge[it] > 1800
        }
        val runs = groups(text, 4).map { r ->
            var a = r.first
            while (a > 0 && rowInkS[a - 1] > W * 0.05) a--
            var b = r.last
            while (b < H - 1 && rowInkS[b + 1] > W * 0.05) b++
            a..b
        }
        var best: Pair<IntRange, IntRange>? = null
        for (i in 0 until runs.size - 1) {
            val t1 = runs[i]
            val h1 = t1.last - t1.first + 1
            if (h1 < 5 || h1 > 48 || !lineShapeOf(gray, thr, t1)) continue
            val t2 = runs[i + 1]
            val h2 = t2.last - t2.first + 1
            val gap = t2.first - t1.last - 1
            if (h2 < 5 || h2 > 48 || !lineShapeOf(gray, thr, t2)) continue
            if (h2.toDouble() / h1 !in 0.4..1.6) continue
            if (gap < 0 || gap > max(8, (h1 * 3.0).toInt())) continue
            if (best == null || t1.first > best.first.first) best = t1 to t2
        }
        return best?.let { PickResult(it.first, it.second) }
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
        return PairRows(l1, l2, segs, 0)
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
        xpadAdaptive: Pair<Int, Int>,
    ): Band {
        // xpadAdaptive 已经在源图坐标系下（work 系 pair 通过 /scale 转换），直接用
        val xpadL = xpadAdaptive.first
        val xpadR = xpadAdaptive.second
        val firstSourceY1 = min(source.height, ((first.last + 1) / scale).roundToInt())
        val rows = listOf(first, second).map { run ->
            var sourceY0 = max(0, (run.first / scale).roundToInt())
            val sourceY1 = min(source.height, ((run.last + 1) / scale).roundToInt())
            if (run === second) {
                sourceY0 = max(sourceY0, firstSourceY1)
            }
            val rowHeight = max(1, sourceY1 - sourceY0)
            val pad = max(8, min(25, (rowHeight * 0.70).roundToInt()))
            val w = max(1, source.width - xpadL - xpadR)
            Bitmap.createBitmap(
                source,
                xpadL,
                max(0, sourceY0 - pad),
                w,
                min(source.height, sourceY1 + pad) - max(0, sourceY0 - pad),
            )
        }
        val firstTop = max(0, (first.first / scale).roundToInt())
        val secondBottom = min(source.height, ((second.last + 1) / scale).roundToInt())
        val sharedHeight = max(rows[0].height, rows[1].height)
        val sharedPad = max(8, min(25, (sharedHeight * 0.70).roundToInt()))
        val cropTop = max(0, firstTop - sharedPad)
        val cropBottom = min(source.height, secondBottom + sharedPad)
        android.util.Log.i(
            "MrzOcr",
            "band src=${source.width}x${source.height} scale=$scale xp=$xpadL/$xpadR " +
                "pair=(${first.first}..${first.last})(${second.first}..${second.last}) " +
                "rows=${rows[0].width}x${rows[0].height},${rows[1].width}x${rows[1].height}",
        )
        val width = min(rows[0].width, rows[1].width)
        // Fast native path must see the original inter-line geometry. Stacking
        // two independently padded rows changes the perspective and can make
        // the native splitter reject a valid pair.
        val output = Bitmap.createBitmap(
            source,
            xpadL,
            cropTop,
            width,
            cropBottom - cropTop,
        )
        return Band(
            output,
            Bitmap.createBitmap(rows[0], 0, 0, width, rows[0].height),
            Bitmap.createBitmap(rows[1], 0, 0, width, rows[1].height),
        )
    }

    private fun fastNativeInput(band: Bitmap): Bitmap {
        if (band.width <= FAST_NATIVE_MAX_WIDTH) return band
        val height = max(1, (band.height * FAST_NATIVE_MAX_WIDTH.toDouble() / band.width).roundToInt())
        return Bitmap.createScaledBitmap(band, FAST_NATIVE_MAX_WIDTH, height, true)
    }

    private fun decodeFields(line1: String, line2: String): Map<String, String> {
        // CS/CD 港澳台通行证: 单行 30 字符
        if (line1.length == 30 && line2.isEmpty() &&
            line1[0] == 'C' && (line1[1] == 'S' || line1[1] == 'D')) {
            return decodeFieldsCsCd(line1)
        }
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

    /** CS/CD 港澳台通行证字段解析 (30 字符单行).
     *  格式: CS<CN12345678<YYMMDDC<YYMMDDC<C
     *  0-1:  文档类型 (CS=港澳, CD=台湾)
     *  2-3:  国家码
     *  4-11: 证件号码 (8位)
     *  12:   '<'
     *  13-19: 出生日期 YYMMDD + 校验位
     *  20:   '<'
     *  21-27: 有效期 YYMMDD + 校验位
     *  28:   '<'
     *  29:   复合校验位 */
    private fun decodeFieldsCsCd(line1: String): Map<String, String> {
        if (line1.length < 30) return emptyMap()
        return linkedMapOf(
            "doc_type" to line1.substring(0, 2),
            "issuing_state" to line1.substring(2, 4),
            "doc_no" to line1.substring(4, 12),
            "birth_yymmdd" to line1.substring(13, 19),
            "expiry_yymmdd" to line1.substring(21, 27),
            "composite_ck" to line1.substring(29, 30),
        )
    }

    private fun elapsedMs(start: Long): Long =
        ((System.nanoTime() - start) / 1_000_000L).coerceAtLeast(0L)
}
