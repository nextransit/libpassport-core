package com.nextransit.mrzbench.bench

import android.content.Context
import android.util.Log
import com.nextransit.mrzbench.data.Corpus
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Headless driver used by MainActivity's broadcast receiver. Runs the
 * benchmark outside of any Fragment lifecycle so a hidden / detached
 * fragment view never cancels the work.
 */
class HeadlessRunner private constructor(private val ctx: Context) {
    companion object {
        private const val TAG = "MrzBench"

        @JvmStatic
        fun run(
            ctx: Context,
            backends: List<Benchmark.Backend>,
            imagesDir: File,
            corpusJson: File,
            onComplete: (Map<Benchmark.Backend, MethodMetrics>) -> Unit = {}
        ) {
            CoroutineScope(SupervisorJob() + Dispatchers.IO).launch {
                try {
                    val records = Corpus.parse(corpusJson)
                    if (records.isEmpty()) {
                        Log.w(TAG, "HeadlessRunner: empty corpus, aborting")
                        return@launch
                    }
                    val totals = mutableMapOf<Benchmark.Backend, MethodMetrics>()
                    for (b in backends) {
                        val outcome = Benchmark(b) { imagesDir }.run(records) { _, _ -> }
                        totals[b] = outcome.metrics
                        writeResults(ctx, b, outcome.records, outcome.metrics)
                    }
                    CoroutineScope(Dispatchers.Main).launch { onComplete(totals) }
                } catch (t: Throwable) {
                    Log.e(TAG, "HeadlessRunner failed", t)
                }
            }
        }

        private fun writeResults(
            ctx: Context,
            backend: Benchmark.Backend,
            recs: List<ResultRecord>,
            m: MethodMetrics,
        ) {
            val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val ext = ctx.getExternalFilesDir("results") ?: return
            ext.mkdirs()
            val perImage = File(ext, "${backend.displayName}_records_${ts}.jsonl")
            perImage.bufferedWriter().use { w ->
                for (r in recs) {
                    w.write(serialize(r)); w.write("\n")
                }
            }
            File(ext, "${backend.displayName}_metrics_${ts}.json").writeText(m.toJson())
            File(ext, "${backend.displayName}_report_${ts}.md").writeText(buildReport(backend, recs, m))
        }

        private fun buildReport(b: Benchmark.Backend, recs: List<ResultRecord>, m: MethodMetrics): String {
            val sb = StringBuilder()
            sb.appendLine("# MrzBench — on-device report ($b)")
            sb.appendLine()
            sb.appendLine("Records: " + m.n + "    Pipeline OK: " + m.ok + " (" + "%.2f".format(m.okRate * 100) + "%)")
            sb.appendLine("Line 1 acc: " + "%.2f".format(m.l1Acc * 100) + "%    Line 2 acc: " + "%.2f".format(m.l2Acc * 100) + "%")
            sb.appendLine("Full match: " + "%.2f".format(m.fullMatchRate * 100) + "%    Cksum valid: " + "%.2f".format(m.cksumRate * 100) + "%")
            sb.appendLine("ms mean " + "%.3f".format(m.msMean) + "  p50 " + "%.3f".format(m.msP50) + "  p95 " + "%.3f".format(m.msP95))
            sb.appendLine()
            val conf = HashMap<String, Int>()
            for (r in recs) {
                for (i in r.gt1.indices) {
                    val g = r.gt1[i]
                    val h = if (i < r.line1.length) r.line1[i] else '_'
                    if (g != h) { val k = "$g->$h"; conf[k] = (conf[k] ?: 0) + 1 }
                }
                for (i in r.gt2.indices) {
                    val g = r.gt2[i]
                    val h = if (i < r.line2.length) r.line2[i] else '_'
                    if (g != h) { val k = "$g->$h"; conf[k] = (conf[k] ?: 0) + 1 }
                }
            }
            sb.appendLine("## Top confusions (count)")
            conf.entries.sortedByDescending { it.value }.take(15).forEach {
                sb.appendLine("| `$it.key` | ${it.value} |")
            }
            return sb.toString()
        }

        private fun serialize(r: ResultRecord): String {
            fun esc(s: String) = s.replace("\\", "\\\\").replace("\"", "\\\"")
            return "{" +
                "\"id\":\"" + esc(r.id) + "\",\"backend\":\"" + r.backend + "\"," +
                "\"profile\":\"" + r.profile + "\",\"scale\":" + r.scale + ",\"skew\":" + r.skew + "," +
                "\"noise\":" + r.noise + "," +
                "\"gt1\":\"" + esc(r.gt1) + "\",\"gt2\":\"" + esc(r.gt2) + "\"," +
                "\"line1\":\"" + esc(r.line1) + "\",\"line2\":\"" + esc(r.line2) + "\"," +
                "\"conf1\":" + r.conf1 + ",\"conf2\":" + r.conf2 + "," +
                "\"ms\":" + "%.6f".format(r.ms) + ",\"ok\":" + r.ok + "," +
                "\"err\":\"" + esc(r.errMsg) + "\"" +
            "}"
        }
    }
}
