package com.nextransit.mrzbench.ui

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.fragment.app.Fragment
import com.nextransit.mrzbench.R
import com.nextransit.mrzbench.bench.StreamBus

/**
 * "Stream" tab — append-only monospaced console. The RunFragment posts
 * lines here via StreamBus; we throttle writes to 50 ms bursts to
 * avoid UI jank on Android 9.
 */
class StreamFragment : Fragment() {

    private lateinit var streamText: TextView
    private val pending = StringBuilder()
    private var lastFlush = 0L

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?
    ): View = inflater.inflate(R.layout.fragment_stream, container, false)

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        streamText = view.findViewById(R.id.stream_text)
        StreamBus.subscribe { append(it) }
    }

    private fun append(line: String) {
        synchronized(pending) {
            pending.append(line).append('\n')
        }
        val now = System.currentTimeMillis()
        val view = view ?: return
        if (now - lastFlush > 50) flush()
        else view.postDelayed({ flush() }, 60)
    }

    private fun flush() {
        lastFlush = System.currentTimeMillis()
        val chunk: String = synchronized(pending) {
            if (pending.isEmpty()) return
            val s = pending.toString()
            pending.clear()
            s
        }
        streamText.append(chunk)
        // Bound the TextView text so we don't blow up the heap on
        // Android 9 over thousands of records. text.length is the
        // Editable length, which we use as the proxy.
        val cur = streamText.length()
        if (cur > 80_000) {
            streamText.text = streamText.text.subSequence(cur - 40_000, cur)
        }
    }
}
