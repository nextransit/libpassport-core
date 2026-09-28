package com.nextransit.mrzbench.data

import org.json.JSONObject
import java.io.File

/**
 * Tiny corpus.json parser. Same shape as
 * 6_mrz_ocr/data/corpus_eval/corpus.json (records[].line1/line2/etc).
 *
 * On-device path: /sdcard/Android/data/com.nextransit.mrzbench/files/
 * corpus/corpus.json — the deploy step `tools/push_corpus.sh` copies
 * the host directory verbatim so every gt/line1 maps 1:1 to a PPM
 * shipped the same way.
 */
object Corpus {
    fun parse(jsonPath: File): List<CorpusEntry> {
        val txt = jsonPath.readText(Charsets.UTF_8)
        val root = JSONObject(txt)
        val recs = root.optJSONArray("records") ?: return emptyList()
        val out = ArrayList<CorpusEntry>(recs.length())
        for (i in 0 until recs.length()) {
            val r = recs.getJSONObject(i)
            out.add(
                CorpusEntry(
                    id      = r.optString("id"),
                    image   = r.optString("image"),
                    line1   = r.optString("line1"),
                    line2   = r.optString("line2"),
                    profile = r.optString("profile", "clean"),
                    scale   = r.optInt("scale", 4),
                    skew    = r.optInt("skew", 0),
                    noise   = r.optDouble("noise", 0.0),
                )
            )
        }
        return out
    }
}
