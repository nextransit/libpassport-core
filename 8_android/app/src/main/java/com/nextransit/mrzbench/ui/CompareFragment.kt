package com.nextransit.mrzbench.ui

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.TextView
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import com.google.android.material.snackbar.Snackbar
import com.nextransit.mrzbench.R
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.Locale

/**
 * "Compare" tab — diff the on-device metrics against the host
 * bench_results.json. The host is expected to push that JSON to
 * <external>/results/host_bench_results.json via tools/push_baseline.sh.
 */
class CompareFragment : Fragment() {

    private lateinit var diffText: TextView
    private lateinit var importBtn: Button

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?
    ): View = inflater.inflate(R.layout.fragment_compare, container, false)

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        diffText = view.findViewById(R.id.diff_text)
        importBtn = view.findViewById(R.id.btn_import_baseline)

        importBtn.setOnClickListener {
            viewLifecycleOwner.lifecycleScope.launch {
                val msg = withContext(Dispatchers.IO) { runDiff() }
                diffText.text = msg
            }
        }
    }

    /** Compute a side-by-side metric diff and return as monospaced text. */
    private fun runDiff(): String {
        val dir = requireContext().getExternalFilesDir("results") ?: return "(no external dir)"
        val hostFile = File(dir, "host_bench_results.json")
        if (!hostFile.exists()) {
            return ("No baseline found at:\n  ${hostFile.absolutePath}\n\n" +
                "Push with: 8_android/tools/push_baseline.sh")
        }
        val devFiles = dir.listFiles { _, n -> n.endsWith("_metrics_*.json") }
            ?.sortedBy { it.lastModified() } ?: return "(no on-device runs)"
        if (devFiles.isEmpty()) return "(no on-device runs yet)"

        val latest = devFiles.last()
        val hostJson = hostFile.readText()
        val devJson  = latest.readText()

        val keys = listOf(
            "ok_rate", "l1_acc", "l2_acc",
            "full_match_rate", "cksum_rate",
            "ms_p50", "ms_p95"
        )
        val sb = StringBuilder()
        sb.appendLine("DEVICE  ${latest.name}")
        sb.appendLine("HOST    ${hostFile.name}")
        sb.appendLine()
        sb.appendLine("metric            device        host          delta")
        for (k in keys) {
            val a = pickNumber(devJson, k)
            val b = pickNumber(hostJson, k)
            val delta = a - b
            sb.appendLine(
                String.format(Locale.US, "%-18s %10.4f   %10.4f   %+10.4f (%+.2fpp)",
                    k, a, b, delta, delta * 100.0)
            )
        }
        sb.appendLine()
        // 0.3pp is the host regression gate; surface any breach.
        val breaches = mutableListOf<String>()
        for (k in keys) {
            val d = (pickNumber(devJson, k) - pickNumber(hostJson, k)).absoluteValue()
            if (d > 0.003) breaches.add(k)
        }
        if (breaches.isEmpty()) {
            sb.appendLine("GATE: PASS (all within 0.3pp)")
        } else {
            sb.appendLine("GATE: FAIL on ${breaches.joinToString(", ")}")
        }
        return sb.toString()
    }

    private fun Double.absoluteValue() = if (this < 0) -this else this

    private fun pickNumber(s: String, key: String): Double {
        val rx = Regex("\"$key\"\\s*:\\s*(-?\\d+(?:\\.\\d+)?)")
        val m = rx.find(s) ?: return 0.0
        return m.groupValues[1].toDoubleOrNull() ?: 0.0
    }
}
