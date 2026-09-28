package com.nextransit.mrzbench.ui

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import com.google.android.material.button.MaterialButtonToggleGroup
import com.google.android.material.progressindicator.LinearProgressIndicator
import com.nextransit.mrzbench.MainActivity
import com.nextransit.mrzbench.R
import com.nextransit.mrzbench.bench.Benchmark
import com.nextransit.mrzbench.bench.MethodMetrics
import com.nextransit.mrzbench.bench.ResultRecord
import com.nextransit.mrzbench.data.Corpus
import com.nextransit.mrzbench.data.Cyber
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** "Run" tab — pick corpus, choose backend, kick a batch, see metrics. */
class RunFragment : Fragment() {

    private lateinit var corpusDirTv: TextView
    private lateinit var corpusCountTv: TextView
    private lateinit var pickBtn: Button
    private lateinit var backendGroup: MaterialButtonToggleGroup
    private lateinit var subsetTv: TextView
    private lateinit var runBtn: Button
    private lateinit var progress: LinearProgressIndicator
    private lateinit var progressText: TextView
    private lateinit var backendTag: TextView

    private val metricCells = mutableMapOf<String, MetricCell>()
    private var chosenDir: File? = null
    private var records: List<com.nextransit.mrzbench.data.CorpusEntry> = emptyList()
    private var running = false

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?
    ): View = inflater.inflate(R.layout.fragment_run, container, false)

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        corpusDirTv = view.findViewById(R.id.corpus_dir)
        corpusCountTv = view.findViewById(R.id.corpus_count)
        pickBtn = view.findViewById(R.id.btn_pick_dir)
        backendGroup = view.findViewById(R.id.backend_group)
        subsetTv = view.findViewById(R.id.subset_text)
        runBtn = view.findViewById(R.id.btn_run)
        progress = view.findViewById(R.id.progress)
        progressText = view.findViewById(R.id.progress_text)
        backendTag = view.findViewById(R.id.metrics_backend_tag)

        // The corpus card lives under the inlined h_corpus TextView —
        // it has no card_header in this layout (the include is a
        // stand-in) so the actual TextView is the next sibling.
        (view.findViewById<View>(R.id.h_corpus) as TextView).text = "CORPUS"
        (view.findViewById<View>(R.id.h_backend) as TextView).text = "BACKEND"
        (view.findViewById<View>(R.id.h_metrics) as TextView).text = "LAST METRICS"

        // Collect metric cells (6 includes, ids: m_ok / m_l1 / m_l2 /
        // m_full / m_ck / m_ms). Each cell has metric_lbl + metric_val.
        listOf("m_ok" to "ok", "m_l1" to "l1", "m_l2" to "l2",
               "m_full" to "full", "m_ck" to "cksum", "m_ms" to "ms")
            .forEach { (id, key) ->
                val cell = view.findViewById<View>(view.resources.getIdentifier(id, "id", requireContext().packageName))
                metricCells[key] = MetricCell(
                    cell.findViewById(R.id.metric_lbl),
                    cell.findViewById(R.id.metric_val),
                    key,
                )
            }
        backendGroup.check(R.id.btn_backend_cnn)
        com.nextransit.mrzbench.bench.StreamBus.publish("selected backend group id=${backendGroup.checkedButtonId} records=${records.size}")

        // 1) Try the canonical app-private external dir FIRST
        //    (where the host push script drops data/corpus_eval/).
        //    Only fall back to the bundled assets demo when nothing
        //    is there, so a real batch run isn't silently capped at
        //    30 records.
        tryExternal()
        if (records.isEmpty()) {
            loadFromAssets()
        }

        pickBtn.setOnClickListener {
            // Trivial "rescan" — the app reads from
            // getExternalFilesDir("corpus") which is the only place the
            // host has write access to on API 28+ without a permission.
            tryExternal()
        }

        runBtn.setOnClickListener { startRun() }
        // (The RUN_BENCH broadcast is owned by MainActivity, which
        //  forwards to HeadlessRunner and only calls
        //  setBackendSelection() on us for UI sync.)
    }

    private fun loadFromAssets() {
        try {
            val json = requireContext().assets.open("corpus/corpus.json")
                .bufferedReader().use { it.readText() }
            val tmp = File(requireContext().cacheDir, "demo_corpus.json")
            tmp.writeText(json)
            records = Corpus.parse(tmp)
            // Materialise PPMs out of assets/ into cacheDir so the
            // native side can fopen() them by absolute path.
            val dst = File(requireContext().cacheDir, "demo_img")
            if (!dst.exists() || dst.listFiles().isNullOrEmpty()) {
                dst.mkdirs()
                requireContext().assets.list("corpus/img")?.forEach { name ->
                    requireContext().assets.open("corpus/img/" + name).use { ins ->
                        File(dst, name).outputStream().use { ins.copyTo(it) }
                    }
                }
            }
            chosenDir = dst
            corpusDirTv.text = "assets:/corpus/corpus.json  (demo, n=" + records.size + ")"
            corpusCountTv.text = records.size.toString()
            subsetTv.text = "demo (assets)"
        } catch (_: Throwable) { /* no bundled demo is fine */ }
    }

    /**
     * Pure UI hook: toggle the backend buttons so the on-screen state
     * reflects what the host driver is about to run. We do NOT call
     * startRun() here — MainActivity's HeadlessRunner kicks the actual
     * batch outside the fragment lifecycle.
     */
    fun setBackendSelection(backend: String) {
        try {
            when (backend.lowercase()) {
                "trad", "template" -> backendGroup.check(R.id.btn_backend_trad)
                "cnn"              -> backendGroup.check(R.id.btn_backend_cnn)
                else               -> backendGroup.check(R.id.btn_backend_both)
            }
            android.util.Log.i("MrzBench", "setBackendSelection backend=" + backend)
        } catch (t: Throwable) {
            android.util.Log.w("MrzBench", "setBackendSelection skipped: " + t.message)
        }
    }

    /** Legacy entry kept for the in-app button. */
    fun handleHostCommand(backend: String) {
        setBackendSelection(backend)
        startRun()
    }

    private fun tryExternal() {
        val dir = requireContext().getExternalFilesDir("corpus") ?: return
        val json = File(dir, "corpus.json")
        if (!json.exists()) {
            corpusDirTv.text = if (records.isNotEmpty()) corpusDirTv.text
                else "${dir.absolutePath}\n  (no corpus.json yet — push from host)"
            return
        }
        chosenDir = File(dir, "img")
        if (!chosenDir!!.exists()) chosenDir = dir
        records = Corpus.parse(json)
        corpusDirTv.text = dir.absolutePath
        corpusCountTv.text = records.size.toString()
        subsetTv.text = "full (${records.size} records)"
    }

    private fun startRun() {
        android.util.Log.i("MrzBench", "startRun() enter records=" + records.size + " running=" + running)
        if (running) return
        if (records.isEmpty()) {
            (activity as? MainActivity)?.setStatus("no corpus loaded")
            android.util.Log.w("MrzBench", "startRun aborted: empty records")
            return
        }
        // Snapshot the records + imgDir locally so the coroutine
        // doesn't go through `records` after the host kicks a
        // second RUN_BENCH (which would mutate chosenDir). View refs
        // captured up-front so hide/show doesn't blow up either.
        val recSnap = records.toList()
        val imgDirSnap = chosenDir ?: File(requireContext().cacheDir, "demo_img")
        val total = recSnap.size
        val updateProgress: (Int) -> Unit = fun(done: Int) {
            val n = done.coerceAtMost(total)
            try {
                if (isAdded) {
                    progress.max = total
                    progress.setProgressCompat(n, true)
                    progressText.text = "$n / $total"
                }
            } catch (t: Throwable) {
                android.util.Log.w("MrzBench", "progress update skipped: " + t.message)
            }
        }
        running = true
        try {
            runBtn.isEnabled = false
            pickBtn.isEnabled = false
            progress.max = total
            progress.setProgressCompat(0, true)
            progressText.text = "0 / $total"
            (activity as? MainActivity)?.setStatus("running…")
        } catch (t: Throwable) {
            android.util.Log.w("MrzBench", "startRun UI init failed", t)
        }

        val backends = when (backendGroup.checkedButtonId) {
            R.id.btn_backend_trad -> listOf(Benchmark.Backend.TRAD)
            R.id.btn_backend_cnn  -> listOf(Benchmark.Backend.CNN)
            else                  -> listOf(Benchmark.Backend.TRAD, Benchmark.Backend.CNN)
        }
        val scope = viewLifecycleOwner.lifecycleScope
        scope.launch {
            try {
                val out = withContext(Dispatchers.IO) {
                    val pairs = mutableListOf<Pair<Benchmark.Backend, List<ResultRecord>>>()
                    for (b in backends) {
                        val o = Benchmark(b) { imgDirSnap }.run(recSnap) { done, total ->
                            scope.launch(Dispatchers.Main) { updateProgress(done) }
                        }
                        pairs.add(b to o.records)
                        (activity as? MainActivity)?.setStatus("writing ${'$'}{b.name}-results.json…")
                        for (r in o.records) {
                            val tag = if (r.ok) "OK " else "ERR"
                            com.nextransit.mrzbench.bench.StreamBus.publish(
                                "$tag ${'$'}{b.name} ${'$'}{r.id} l1=${'$'}{r.line1} l2=${'$'}{r.line2} c=${'$'}{r.conf1}/${'$'}{r.conf2} ms=${"%.2f".format(r.ms)}"
                            )
                        }
                        writeResults(b, o.records, o.metrics)
                    }
                    pairs
                }
                renderLast(backends, out.map { it.first to it.second })
            } catch (t: Throwable) {
                android.util.Log.e("MrzBench", "startRun failed", t)
                (activity as? MainActivity)?.setStatus("error: ${'$'}{t.javaClass.simpleName}")
            } finally {
                running = false
                if (isAdded) {
                    runBtn.isEnabled = true
                    pickBtn.isEnabled = true
                }
                (activity as? MainActivity)?.setStatus("done")
            }
        }
    }

    private fun renderLast(
        backends: List<Benchmark.Backend>,
        results: List<Pair<Benchmark.Backend, List<ResultRecord>>>,
    ) {
        val pair = results.last()
        val m = com.nextransit.mrzbench.bench.Metrics.from(pair.second)
        backendTag.text = "backend=${pair.first.name}  n=${m.n}"
        metricCells["ok"]?.set("Pipeline OK", "${m.ok}/${m.n}", pct(m.okRate))
        metricCells["l1"]?.set("Line 1 acc",  pct1(m.l1Acc), pct(m.l1Acc))
        metricCells["l2"]?.set("Line 2 acc",  pct1(m.l2Acc), pct(m.l2Acc))
        metricCells["full"]?.set("Full match", pct1(m.fullMatchRate), pct(m.fullMatchRate))
        metricCells["ck"]?.set("Cksum valid", pct1(m.cksumRate), pct(m.cksumRate))
        metricCells["ms"]?.set("ms p50 / p95", "%.2f / %.2f".format(m.msP50, m.msP95),
                                if (m.msP50 < 15) Cyber.ok else Cyber.warn)
    }

    private fun writeResults(
        backend: Benchmark.Backend,
        recs: List<ResultRecord>,
        m: MethodMetrics,
    ) {
        val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val ext = requireContext().getExternalFilesDir("results") ?: return
        ext.mkdirs()
        val perImage = File(ext, "${backend.name}_records_${ts}.jsonl")
        perImage.bufferedWriter().use { w ->
            for (r in recs) {
                w.write(serialize(r))
                w.write("\n")
            }
        }
        val metricsJson = File(ext, "${backend.name}_metrics_${ts}.json")
        metricsJson.writeText(m.toJson())
        val report = File(ext, "${backend.name}_report_${ts}.md")
        report.writeText(buildReport(backend, recs, m))
    }

    private fun buildReport(
        backend: Benchmark.Backend,
        recs: List<ResultRecord>,
        m: MethodMetrics,
    ): String {
        val sb = StringBuilder()
        sb.appendLine("# MrzBench — on-device report ($backend)")
        sb.appendLine()
        sb.appendLine("Records: ${m.n}    Pipeline OK: ${m.ok} (${pct1(m.okRate)})")
        sb.appendLine("Line 1 acc: ${pct1(m.l1Acc)}    Line 2 acc: ${pct1(m.l2Acc)}")
        sb.appendLine("Full match: ${pct1(m.fullMatchRate)}    Cksum valid: ${pct1(m.cksumRate)}")
        sb.appendLine("ms mean ${"%.3f".format(m.msMean)}  p50 ${"%.3f".format(m.msP50)}  p95 ${"%.3f".format(m.msP95)}")
        sb.appendLine()
        // Top confusions (gt->got)
        val conf = HashMap<String, Int>()
        for (r in recs) {
            for (i in r.gt1.indices) {
                val g = r.gt1[i]
                val h = if (i < r.line1.length) r.line1[i] else '_'
                if (g != h) {
                    val k = "$g->$h"; conf[k] = (conf[k] ?: 0) + 1
                }
            }
            for (i in r.gt2.indices) {
                val g = r.gt2[i]
                val h = if (i < r.line2.length) r.line2[i] else '_'
                if (g != h) {
                    val k = "$g->$h"; conf[k] = (conf[k] ?: 0) + 1
                }
            }
        }
        sb.appendLine("## Top confusions (count)")
        conf.entries.sortedByDescending { it.value }.take(15).forEach {
            sb.appendLine("| `${it.key}` | ${it.value} |")
        }
        return sb.toString()
    }

    private fun serialize(r: ResultRecord): String {
        fun esc(s: String) = s.replace("\\", "\\\\").replace("\"", "\\\"")
        return "{" +
            "\"id\":\"${esc(r.id)}\",\"backend\":\"${r.backend}\"," +
            "\"profile\":\"${r.profile}\",\"scale\":${r.scale},\"skew\":${r.skew}," +
            "\"noise\":${r.noise}," +
            "\"gt1\":\"${esc(r.gt1)}\",\"gt2\":\"${esc(r.gt2)}\"," +
            "\"line1\":\"${esc(r.line1)}\",\"line2\":\"${esc(r.line2)}\"," +
            "\"conf1\":${r.conf1},\"conf2\":${r.conf2}," +
            "\"ms\":${"%.6f".format(r.ms)},\"ok\":${r.ok}," +
            "\"err\":\"${esc(r.errMsg)}\"" +
        "}"
    }

    private fun pct(x: Double) = when {
        x >= 0.85 -> Cyber.ok
        x >= 0.50 -> Cyber.warn
        else      -> Cyber.err
    }

    private fun pct1(x: Double) = "%.2f%%".format(x * 100)

    private data class MetricCell(val lbl: TextView, val val0: TextView, val key: String) {
        fun set(label: String, value: String, tint: Int) {
            lbl.text = label
            val0.text = value
            val0.setTextColor(tint)
        }
    }
}
