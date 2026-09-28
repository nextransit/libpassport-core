package com.nextransit.mrzbench.bench

import com.nextransit.mrzbench.jni.MrzNative
import com.nextransit.mrzbench.util.TinyJson
import com.nextransit.mrzbench.data.CorpusEntry
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Drives one backend over the whole corpus, returning per-image records
 * + the aggregated metrics. Designed to be called from a coroutine
 * (withContext IO) — native call is short but synchronous, so we keep
 * the dispatcher honest.
 */
class Benchmark(
    private val backend: Backend,
    private val imagesDirProvider: () -> java.io.File,
) {
    enum class Backend(val code: Int, val displayName: String) {
        TRAD(0, "traditional"),
        CNN (1, "cnn");

        companion object {
            fun of(name: String) = values().firstOrNull { it.displayName == name } ?: CNN
        }
    }

    data class Outcome(
        val records: List<ResultRecord>,
        val metrics: MethodMetrics,
    )

    suspend fun run(
        entries: List<CorpusEntry>,
        progress: (Int, Int) -> Unit = { _, _ -> }
    ): Outcome = withContext(Dispatchers.IO) {
        val recs = ArrayList<ResultRecord>(entries.size)
        val total = entries.size
        for ((i, e) in entries.withIndex()) {
            val img = java.io.File(imagesDirProvider(), e.image)
            val raw = try {
                MrzNative.recogniseString(img.absolutePath, backend.code)
            } catch (t: Throwable) { "{\"ok\":false,\"err\":\"${t.javaClass.simpleName}\"}" }
            val ok = TinyJson.boolOf(raw, "ok", false)
            val r = ResultRecord(
                id      = e.id,
                backend = backend.name,
                profile = e.profile,
                scale   = e.scale,
                skew    = e.skew,
                noise   = e.noise,
                gt1     = e.line1,
                gt2     = e.line2,
                line1   = if (ok) TinyJson.stringOf(raw, "line1") else "",
                line2   = if (ok) TinyJson.stringOf(raw, "line2") else "",
                conf1   = if (ok) TinyJson.intOf(raw, "conf1") else 0,
                conf2   = if (ok) TinyJson.intOf(raw, "conf2") else 0,
                ms      = TinyJson.doubleOf(raw, "ms"),
                ok      = ok,
                errMsg  = if (ok) "" else TinyJson.stringOf(raw, "err"),
            )
            recs.add(r)
            progress(i + 1, total)
        }
        Outcome(recs, Metrics.from(recs))
    }
}
