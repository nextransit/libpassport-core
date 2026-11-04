package com.nextransit.mrzbench.ui

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.style.ForegroundColorSpan
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.CheckBox
import android.widget.ImageView
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import com.nextransit.mrzbench.R
import com.nextransit.mrzbench.data.Cyber
import com.nextransit.mrzbench.photo.MrzStage
import com.nextransit.mrzbench.photo.MrzTally
import com.nextransit.mrzbench.photo.MrzVerdict
import com.nextransit.mrzbench.photo.PhotoBatchRunner
import com.nextransit.mrzbench.photo.PhotoMrzPipeline
import com.nextransit.mrzbench.photo.PhotoMrzResult
import com.nextransit.mrzbench.photo.verdict
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class PhotoMrzFragment : Fragment() {
    private companion object {
        const val STAGE_HINT =
            "\u2460 original   \u2461 reflection removed   \u2462 MRZ crop"

        /**
         * Blown-out fraction (per mille) at which the photo is reported
         * as glared.
         *
         * The MRZ area of a passport page is mostly light background, so
         * a few per mille of clipped pixels is not yet a problem: on the
         * 44-photo corpus the highest measured value was 6 per mille.
         * A specular highlight on the laminate covers a patch, so 30
         * per mille (3% of the sampled region) is where a real highlight
         * starts to eat the strokes.  The threshold is deliberately high
         * so a clean photo is never flagged.
         */
        const val GLARE_WARN_PERMILLE = 30
    }

    private lateinit var pick: Button
    private lateinit var sample: Button
    private lateinit var run: Button
    private lateinit var batch: Button
    private lateinit var batchGroup: Button
    private lateinit var methodButton: Button
    private lateinit var allToggle: CheckBox
    private lateinit var scroll: ScrollView
    private lateinit var status: TextView
    private lateinit var result: TextView
    private val images = mutableMapOf<String, ImageView>()
    private lateinit var stageLabels: TextView
    private var source: Bitmap? = null
    private var method = "auto"
    private var batchGroupFilter = "all"

    private val picker = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        if (uri == null) return@registerForActivityResult
        viewLifecycleOwner.lifecycleScope.launch {
            try {
                val bitmap = withContext(Dispatchers.IO) { decode(uri) }
                // Wipe every stage and the previous RESULT before the new
                // photo lands.  Keeping the old frame on screen reads as
                // "this is the result for the photo I just picked", which
                // is exactly the stale-output bug this clears.
                clearOutputs()
                source = bitmap
                images["original"]?.setImageBitmap(bitmap)
                status.text = "photo loaded ${bitmap.width}x${bitmap.height}"
                result.text = "processing…"
                run.isEnabled = true
                // Auto-run: picking a photo is a request to see its MRZ,
                // not just to stage it.
                startPipeline()
            } catch (t: Throwable) {
                status.text = "load failed: ${t.javaClass.simpleName}"
            }
        }
    }

    /** Reset every stage tile and the RESULT pane to their idle state. */
    private fun clearOutputs() {
        listOf("original", "clean", "mrz").forEach { key ->
            images[key]?.setImageDrawable(null)
        }
        if (::stageLabels.isInitialized) stageLabels.text = STAGE_HINT
        if (::result.isInitialized) result.text = "processing…"
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?
    ): View = inflater.inflate(R.layout.fragment_photo_mrz, container, false)

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        pick = view.findViewById(R.id.btn_photo_pick)
        sample = view.findViewById(R.id.btn_photo_sample)
        run = view.findViewById(R.id.btn_photo_run)
        batch = view.findViewById(R.id.btn_photo_batch)
        batchGroup = view.findViewById(R.id.btn_photo_batch_group)
        methodButton = view.findViewById(R.id.btn_photo_method)
        allToggle = view.findViewById(R.id.cb_photo_all)
        scroll = view.findViewById(R.id.photo_scroll)
        status = view.findViewById(R.id.photo_status)
        result = view.findViewById(R.id.photo_result)
        stageLabels = view.findViewById(R.id.photo_stage_labels)
        listOf("original", "clean", "mrz").forEach { key ->
            images[key] = view.findViewById(
                resources.getIdentifier("photo_$key", "id", requireContext().packageName)
            )
        }
        run.isEnabled = false
        pick.setOnClickListener { picker.launch(arrayOf("image/jpeg", "image/png", "image/*")) }
        sample.setOnClickListener { showSamplePicker() }
        run.setOnClickListener { startPipeline() }
        methodButton.setOnClickListener { showMethodPicker() }
        batch.setOnClickListener { startBatchTest() }
        batchGroup.setOnClickListener { showBatchGroupPicker() }
    }

    private fun photoMethodOptions(): List<Pair<String, String>> =
        listOf("auto" to "自动") + PhotoMrzPipeline.METHODS

    private fun showMethodPicker() {
        val options = photoMethodOptions()
        val labels = options.map { it.second }.toTypedArray()
        val checked = options.indexOfFirst { it.first == method }.coerceAtLeast(0)
        AlertDialog.Builder(requireContext())
            .setTitle("识别算法")
            .setSingleChoiceItems(labels, checked) { dialog, which ->
                method = options[which].first
                methodButton.text = "算法: ${options[which].second}"
                dialog.dismiss()
            }
            .setNegativeButton(android.R.string.cancel) { d, _ -> d.dismiss() }
            .show()
    }

    private fun showSamplePicker() {
        viewLifecycleOwner.lifecycleScope.launch {
            val entries = withContext(Dispatchers.IO) { listSampleAssets() }
            if (entries.isEmpty()) {
                status.text = "no bundled samples found in assets/"
                return@launch
            }
            val labels = entries.map { (label, _) -> label }.toTypedArray()
            AlertDialog.Builder(requireContext())
                .setTitle("Bundled samples (${entries.size})")
                .setItems(labels) { dialog, which ->
                    val (_, path) = entries[which]
                    loadFromAssets(path)
                }
                .setNegativeButton(android.R.string.cancel) { d, _ -> d.dismiss() }
                .show()
        }
    }

    private fun listSampleAssets(): List<Pair<String, String>> {
        val ctx = requireContext()
        val results = mutableListOf<Pair<String, String>>()
        val photoDirs = ctx.assets.list("Photo") ?: emptyArray()
        for (dir in photoDirs.sorted()) {
            val files = ctx.assets.list("Photo/$dir") ?: continue
            for (f in files.sorted()) {
                if (!f.endsWith(".jpg", true) && !f.endsWith(".jpeg", true) && !f.endsWith(".png", true)) continue
                results += "Photo/$dir/$f" to "Photo/$dir/$f"
            }
        }
        val picFiles = ctx.assets.list("pic") ?: emptyArray()
        for (f in picFiles.sorted()) {
            if (!f.endsWith(".jpg", true) && !f.endsWith(".jpeg", true) && !f.endsWith(".png", true)) continue
            results += "pic/$f" to "pic/$f"
        }
        val passPortFiles = ctx.assets.list("PassPortPhoto") ?: emptyArray()
        for (f in passPortFiles.sorted()) {
            if (!f.endsWith(".jpg", true) && !f.endsWith(".jpeg", true) && !f.endsWith(".png", true)) continue
            results += "PassPortPhoto/$f" to "PassPortPhoto/$f"
        }
        return results
    }

    private fun loadFromAssets(path: String) {
        viewLifecycleOwner.lifecycleScope.launch {
            try {
                val bitmap = withContext(Dispatchers.IO) {
                    requireContext().assets.open(path).use { BitmapFactory.decodeStream(it) }
                        ?: error("unsupported image")
                }
                clearOutputs()
                source = bitmap
                images["original"]?.setImageBitmap(bitmap)
                status.text = "sample loaded $path  ${bitmap.width}x${bitmap.height}"
                result.text = "processing…"
                run.isEnabled = true
                startPipeline()
            } catch (t: Throwable) {
                status.text = "sample load failed: ${t.javaClass.simpleName}"
            }
        }
    }

    private fun startPipeline() {
        val input = source ?: return
        val all = allToggle.isChecked
        pick.isEnabled = false
        run.isEnabled = false
        status.text = if (all) "processing all methods…" else "processing photo…"
        viewLifecycleOwner.lifecycleScope.launch {
            try {
                val outputs = withContext(Dispatchers.Default) {
                    // Stages arrive on the worker thread; hop back to the
                    // main thread so each tile can be filled the moment
                    // that stage is ready instead of all at the end.
                    val onStage: (MrzStage) -> Unit = { stage ->
                        android.os.Handler(android.os.Looper.getMainLooper()).post {
                            if (source !== input) return@post
                            images[stage.key]?.setImageBitmap(stage.bitmap)
                            stageLabels.text = stage.label
                        }
                    }
                    if (all) {
                        PhotoMrzPipeline.METHODS.map { (key, label) ->
                            label to PhotoMrzPipeline.process(requireContext(), input, key, onStage)
                        }
                    } else {
                        listOf("" to PhotoMrzPipeline.process(requireContext(), input, method, onStage))
                    }
                }
                if (all) renderAll(outputs) else render(outputs.first().second)
            } catch (t: Throwable) {
                status.text = "pipeline failed: ${t.message ?: t.javaClass.simpleName}"
                if (::result.isInitialized) result.text = "pipeline failed: ${t.message ?: t.javaClass.simpleName}"
            } finally {
                pick.isEnabled = true
                run.isEnabled = true
            }
        }
    }

    private fun render(output: PhotoMrzResult) {
        images["original"]?.setImageBitmap(output.original)
        images["clean"]?.setImageBitmap(output.noReflection ?: output.original)
        images["mrz"]?.setImageBitmap(output.mrzCrop ?: output.original)
        stageLabels.text = STAGE_HINT
        status.text = when {
            !output.decodeOk -> "OCR result needs review  ${output.backend}"
            // 校验位之外的两个否决位。它们比 ambiguous 更硬：ambiguous 是
            // 「可能读错」，这两个是「读出来的东西不可能是真的」。
            !output.structOk ->
                "字段不可能（日期非法）  ${output.backend}  ${output.timings["total"] ?: 0} ms"
            output.natSuspect ->
                "国籍码可疑  ${output.backend}  ${output.timings["total"] ?: 0} ms"
            output.issuerSuspect ->
                "签发国码可疑  ${output.backend}  ${output.timings["total"] ?: 0} ms"
            output.ambiguous ->
                "AMBIGUOUS  ${output.backend}  ${output.timings["total"] ?: 0} ms"
            else -> "PASS  ${output.backend}  ${output.timings["total"] ?: 0} ms"
        }
        // A specular highlight on the laminate erases the strokes, and
        // no thresholding can bring them back: the result would look
        // fine and still be wrong.  Tell the user to move the phone.
        if (output.glarePermille >= GLARE_WARN_PERMILLE) {
            status.text = "反光过曝，请微调角度重拍  ${output.backend}"
        }
        val sb = SpannableStringBuilder()
        appendColored(sb, "line1  ", Cyber.textDim)
        appendColored(sb, output.line1 + "\n", Cyber.neonCyan)
        appendColored(sb, "line2  ", Cyber.textDim)
        appendColored(sb, output.line2 + "\n\n", Cyber.neonCyan)
        appendColored(sb, "decode  ", Cyber.textDim)
        appendColored(sb, if (output.decodeOk) "PASS\n" else "FAIL\n",
            if (output.decodeOk) Cyber.ok else Cyber.err)
        // 五个校验位只证明 44 个字符自洽，不证明它们是文字：在 assets 语料
        // 上 28 张校验通过的里 19 张国籍码不存在、1 张日期不存在。这两行
        // 就是那两种情况，必须显示，否则结果看起来和正确的一样。
        if (!output.structOk) {
            appendColored(sb, "struct  ", Cyber.textDim)
            appendColored(sb, "IMPOSSIBLE  (出生/到期日期不存在)\n", Cyber.warn)
        }
        if (output.natSuspect) {
            appendColored(sb, "nat     ", Cyber.textDim)
            appendColored(sb, "SUSPECT  (国籍码不在 ISO-3166 内)\n", Cyber.warn)
        }
        // line1 没有任何校验位，所以它的内容必须另查。41 张真机照片上
        // line1 读错的 4 张，签发国码全是没有国家会签发的伪造码。
        if (output.issuerSuspect) {
            appendColored(sb, "issuer  ", Cyber.textDim)
            appendColored(sb, "SUSPECT  (签发国码不在 ISO-3166 内)\n", Cyber.warn)
        }
        // 签发国与国籍不一致：难民旅行证可以合法地不一致，所以只提示、
        // 不否决。纯告警。
        if (output.countryMismatch) {
            appendColored(sb, "country ", Cyber.textDim)
            appendColored(sb, "MISMATCH  (签发国与国籍不一致)\n", Cyber.warn)
        }
        // A checksum-valid line is not automatically the right one: the
        // decoder can find a second reading that the checksum cannot
        // distinguish.  Say so, because otherwise the result looks
        // exactly as trustworthy as an unambiguous one.
        if (output.ambiguous) {
            appendColored(sb, "ambiguous  ", Cyber.textDim)
            appendColored(sb, "YES  (runner-up within 0.5)\n", Cyber.warn)
            if (output.altLine2.isNotEmpty()) {
                appendColored(sb, "alt line2  ", Cyber.textDim)
                appendColored(sb, output.altLine2 + "\n", Cyber.warn)
            }
        }
        if (output.glarePermille >= GLARE_WARN_PERMILLE) {
            appendColored(sb, "glare  ", Cyber.textDim)
            appendColored(sb, "${output.glarePermille} per mille blown out\n", Cyber.warn)
        }
        appendColored(sb, "backend  ", Cyber.textDim)
        appendColored(sb, output.backend + "\n", Cyber.neonPurple)
        output.fields.forEach { (key, value) ->
            appendColored(sb, "$key  ", Cyber.textDim)
            appendColored(sb, "$value\n", Cyber.text)
        }
        sb.append("\nTIMING\n")
        output.timings.forEach { (key, value) -> sb.append("$key  ${value} ms\n") }
        if (output.error.isNotEmpty()) {
            appendColored(sb, "\nnote  ", Cyber.textDim)
            appendColored(sb, output.error + "\n", Cyber.warn)
        }
        result.text = sb
        scrollToBottom()
    }

    private fun renderAll(outputs: List<Pair<String, PhotoMrzResult>>) {
        val first = outputs.first().second
        images["original"]?.setImageBitmap(first.original)
        images["clean"]?.setImageBitmap(first.noReflection ?: first.original)
        images["mrz"]?.setImageBitmap(first.mrzCrop ?: first.original)
        stageLabels.text = STAGE_HINT
        // 两个数而不是一个。校验位通过（decoded）与内容可能为真（passed）
        // 之间的差，就是「beam 在噪声上凑出了校验位」的那部分——assets
        // 语料上这个差是 19/28。只报一个数会把噪声算成能力。
        val verdicts = outputs.map { it.second.verdict }
        val tally = MrzTally.of(verdicts)
        status.text = "全部方案  ${tally.passed}/${tally.total} 可信" +
            if (tally.suspect > 0) "  ·  ${tally.decoded}/${tally.total} 校验位通过" else ""
        val sb = SpannableStringBuilder()
        outputs.forEachIndexed { i, (label, output) ->
            val verdict = verdicts[i]
            appendColored(sb, "$label  ", Cyber.neonPurple)
            appendColored(sb, "${output.backend}  ${output.timings["total"] ?: 0} ms  ", Cyber.textDim)
            appendColored(
                sb,
                when (verdict) {
                    MrzVerdict.PASS -> "decode PASS\n"
                    MrzVerdict.SUSPECT -> "decode PASS · 内容可疑\n"
                    MrzVerdict.FAIL -> "decode FAIL\n"
                },
                when (verdict) {
                    MrzVerdict.PASS -> Cyber.ok
                    MrzVerdict.SUSPECT -> Cyber.warn
                    MrzVerdict.FAIL -> Cyber.err
                },
            )
            if (output.line1.isNotEmpty() || output.line2.isNotEmpty()) {
                appendColored(sb, "line1  ", Cyber.textDim)
                appendColored(sb, output.line1 + "\n", Cyber.neonCyan)
                appendColored(sb, "line2  ", Cyber.textDim)
                appendColored(sb, output.line2 + "\n", Cyber.neonCyan)
            }
            if (output.error.isNotEmpty()) {
                appendColored(sb, "note  ", Cyber.textDim)
                appendColored(sb, output.error + "\n", Cyber.warn)
            }
            sb.append("\n")
        }
        result.text = sb
        scrollToBottom()
    }

    private fun scrollToBottom() {
        scroll.post { scroll.fullScroll(View.FOCUS_DOWN) }
    }

    private fun appendColored(sb: SpannableStringBuilder, text: String, color: Int) {
        val start = sb.length
        sb.append(text)
        sb.setSpan(
            ForegroundColorSpan(color),
            start,
            sb.length,
            Spanned.SPAN_EXCLUSIVE_EXCLUSIVE,
        )
    }

    private fun showBatchGroupPicker() {
        val options = listOf(
            "all" to "全部",
            "assets/pic" to "assets/pic",
            "Photo" to "Photo",
            "tmp/PassPortPhoto" to "tmp/PassPortPhoto",
        )
        val labels = options.map { it.second }.toTypedArray()
        val checked = options.indexOfFirst { it.first == batchGroupFilter }.coerceAtLeast(0)
        AlertDialog.Builder(requireContext())
            .setTitle("图片源")
            .setSingleChoiceItems(labels, checked) { dialog, which ->
                batchGroupFilter = options[which].first
                batchGroup.text = "图片源: ${options[which].second}"
                dialog.dismiss()
            }
            .setNegativeButton(android.R.string.cancel) { d, _ -> d.dismiss() }
            .show()
    }

    private fun startBatchTest() {
        viewLifecycleOwner.lifecycleScope.launch {
            try {
                val allEntries = withContext(Dispatchers.IO) {
                    PhotoBatchRunner.listAllPhotos(requireContext())
                }
                val entries = when (batchGroupFilter) {
                    "assets/pic" -> allEntries.filter { it.group == "assets/pic" }
                    "Photo" -> allEntries.filter { it.group == "Photo" }
                    "tmp/PassPortPhoto" -> allEntries.filter { it.group == "tmp/PassPortPhoto" }
                    else -> allEntries
                }
                if (entries.isEmpty()) {
                    status.text = "未找到图片"
                    return@launch
                }
                batch.isEnabled = false
                pick.isEnabled = false
                run.isEnabled = false
                status.text = "批量测试 0/${entries.size}  算法=$method"
                val summary = withContext(Dispatchers.Default) {
                    PhotoBatchRunner.run(
                        requireContext(),
                        entries,
                        method,
                    ) { done, total, _ ->
                        android.os.Handler(android.os.Looper.getMainLooper()).post {
                            status.text = "批量测试 $done/$total  算法=$method"
                        }
                    }
                }
                renderBatchSummary(summary)
            } catch (t: Throwable) {
                status.text = "批量测试失败: ${t.message ?: t.javaClass.simpleName}"
            } finally {
                batch.isEnabled = true
                pick.isEnabled = true
                run.isEnabled = true
            }
        }
    }

    private fun renderBatchSummary(summary: PhotoBatchRunner.BatchSummary) {
        val sb = SpannableStringBuilder()
        appendColored(sb, "批量测试结果\n\n", Cyber.neonCyan)
        appendColored(sb, "总数  ", Cyber.textDim)
        sb.append("${summary.total}\n")
        appendColored(sb, "PASS  ", Cyber.textDim)
        appendColored(sb, "${summary.pass}\n", Cyber.ok)
        appendColored(sb, "SUSPECT  ", Cyber.textDim)
        appendColored(sb, "${summary.suspect}\n", Cyber.warn)
        appendColored(sb, "FAIL  ", Cyber.textDim)
        appendColored(sb, "${summary.fail}\n", Cyber.err)
        appendColored(sb, "校验位通过  ", Cyber.textDim)
        sb.append("${summary.decoded}/${summary.total}\n")
        appendColored(sb, "GT 匹配数  ", Cyber.textDim)
        sb.append("${summary.gtCount}\n")
        appendColored(sb, "line1 字符准确率  ", Cyber.textDim)
        appendColored(sb, "${"%.2f".format(summary.l1CharAcc * 100)}%\n", Cyber.neonCyan)
        appendColored(sb, "line2 字符准确率  ", Cyber.textDim)
        appendColored(sb, "${"%.2f".format(summary.l2CharAcc * 100)}%\n", Cyber.neonCyan)
        appendColored(sb, "精确匹配  ", Cyber.textDim)
        appendColored(sb, "${summary.exactMatch}/${summary.gtCount}\n", Cyber.neonCyan)
        appendColored(sb, "p50  ", Cyber.textDim)
        sb.append("${summary.msP50.toInt()} ms\n")
        appendColored(sb, "p95  ", Cyber.textDim)
        sb.append("${summary.msP95.toInt()} ms\n\n")
        appendColored(sb, "明细\n", Cyber.neonPurple)
        summary.results.forEach { r ->
            val color = when (r.verdict) {
                MrzVerdict.PASS -> Cyber.ok
                MrzVerdict.SUSPECT -> Cyber.warn
                MrzVerdict.FAIL -> Cyber.err
            }
            appendColored(sb, "${r.entry.path}  ", Cyber.textDim)
            appendColored(
                sb,
                when (r.verdict) {
                    MrzVerdict.PASS -> "PASS"
                    MrzVerdict.SUSPECT -> "SUSPECT"
                    MrzVerdict.FAIL -> "FAIL"
                } + "  ${r.backend}  ${r.ms}ms\n",
                color,
            )
            if (r.line1.isNotEmpty() || r.line2.isNotEmpty()) {
                appendColored(sb, "  ${r.line1}\n  ${r.line2}\n", Cyber.neonCyan)
            }
            if (r.error.isNotEmpty()) {
                appendColored(sb, "  note: ${r.error}\n", Cyber.warn)
            }
        }
        result.text = sb
        scrollToBottom()
        status.text = "批量测试完成  " +
            "PASS ${summary.pass}/${summary.total}  " +
            "line1 ${"%.1f".format(summary.l1CharAcc * 100)}%  " +
            "line2 ${"%.1f".format(summary.l2CharAcc * 100)}%  " +
            "p50 ${summary.msP50.toInt()}ms"
    }

    private fun decode(uri: Uri): Bitmap {
        val resolver = requireContext().contentResolver
        resolver.openInputStream(uri).use { input ->
            return BitmapFactory.decodeStream(input)
                ?: error("unsupported image")
        }
    }
}
