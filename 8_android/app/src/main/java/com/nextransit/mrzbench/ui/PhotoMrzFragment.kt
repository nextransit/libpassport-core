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
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import com.nextransit.mrzbench.R
import com.nextransit.mrzbench.data.Cyber
import com.nextransit.mrzbench.photo.PhotoMrzPipeline
import com.nextransit.mrzbench.photo.PhotoMrzResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class PhotoMrzFragment : Fragment() {
    private lateinit var pick: Button
    private lateinit var run: Button
    private lateinit var status: TextView
    private lateinit var result: TextView
    private val images = mutableMapOf<String, ImageView>()
    private var source: Bitmap? = null

    private val picker = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        if (uri == null) return@registerForActivityResult
        viewLifecycleOwner.lifecycleScope.launch {
            try {
                val bitmap = withContext(Dispatchers.IO) { decode(uri) }
                source = bitmap
                images["original"]?.setImageBitmap(bitmap)
                status.text = "photo loaded ${bitmap.width}x${bitmap.height}"
                result.text = "ready to run"
                run.isEnabled = true
            } catch (t: Throwable) {
                status.text = "load failed: ${t.javaClass.simpleName}"
            }
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?
    ): View = inflater.inflate(R.layout.fragment_photo_mrz, container, false)

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        pick = view.findViewById(R.id.btn_photo_pick)
        run = view.findViewById(R.id.btn_photo_run)
        status = view.findViewById(R.id.photo_status)
        result = view.findViewById(R.id.photo_result)
        listOf("original", "clean", "rectified", "mrz").forEach { key ->
            images[key] = view.findViewById(
                resources.getIdentifier("photo_$key", "id", requireContext().packageName)
            )
        }
        run.isEnabled = false
        pick.setOnClickListener { picker.launch(arrayOf("image/jpeg", "image/png", "image/*")) }
        run.setOnClickListener { startPipeline() }
    }

    private fun startPipeline() {
        val input = source ?: return
        pick.isEnabled = false
        run.isEnabled = false
        status.text = "processing photo…"
        viewLifecycleOwner.lifecycleScope.launch {
            try {
                val output = withContext(Dispatchers.Default) {
                    PhotoMrzPipeline.process(requireContext(), input)
                }
                render(output)
            } catch (t: Throwable) {
                status.text = "pipeline failed: ${t.message ?: t.javaClass.simpleName}"
            } finally {
                pick.isEnabled = true
                run.isEnabled = true
            }
        }
    }

    private fun render(output: PhotoMrzResult) {
        images["original"]?.setImageBitmap(output.original)
        images["clean"]?.setImageBitmap(output.noReflection)
        images["rectified"]?.setImageBitmap(output.rectified)
        images["mrz"]?.setImageBitmap(output.mrzCrop)
        status.text = if (output.decodeOk) {
            "PASS  ${output.backend}  ${output.timings["total"] ?: 0} ms"
        } else {
            "OCR result needs review  ${output.backend}"
        }
        val sb = SpannableStringBuilder()
        appendColored(sb, "line1  ", Cyber.textDim)
        appendColored(sb, output.line1 + "\n", Cyber.neonCyan)
        appendColored(sb, "line2  ", Cyber.textDim)
        appendColored(sb, output.line2 + "\n\n", Cyber.neonCyan)
        appendColored(sb, "decode  ", Cyber.textDim)
        appendColored(sb, if (output.decodeOk) "PASS\n" else "FAIL\n",
            if (output.decodeOk) Cyber.ok else Cyber.err)
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

    private fun decode(uri: Uri): Bitmap {
        val resolver = requireContext().contentResolver
        resolver.openInputStream(uri).use { input ->
            return BitmapFactory.decodeStream(input)
                ?: error("unsupported image")
        }
    }
}
