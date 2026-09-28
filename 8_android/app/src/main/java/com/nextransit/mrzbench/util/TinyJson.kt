package com.nextransit.mrzbench.util

/**
 * Minimal pull reader for the JSON the native side returns. We avoid
 * pulling org.json / Gson / kotlinx.serialization — the schema is fixed
 * (10 fields per image) and we ship in <100 KB of classes. Keep this
 * small and stupid.
 */
object TinyJson {
    /** Returns the integer value at "key" or `def` if missing/bad. */
    fun intOf(s: String, key: String, def: Int = 0): Int {
        val rx = Regex("\"$key\"\\s*:\\s*(-?\\d+)")
        val m = rx.find(s) ?: return def
        return m.groupValues[1].toIntOrNull() ?: def
    }

    fun doubleOf(s: String, key: String, def: Double = 0.0): Double {
        val rx = Regex("\"$key\"\\s*:\\s*(-?\\d+(?:\\.\\d+)?)")
        val m = rx.find(s) ?: return def
        return m.groupValues[1].toDoubleOrNull() ?: def
    }

    fun boolOf(s: String, key: String, def: Boolean = false): Boolean {
        val rx = Regex("\"$key\"\\s*:\\s*(true|false)")
        val m = rx.find(s) ?: return def
        return m.groupValues[1] == "true"
    }

    /** Unescapes \", \\, \n, \r, \t, \uXXXX — enough for our PPM GT. */
    fun stringOf(s: String, key: String, def: String = ""): String {
        val rx = Regex("\"$key\"\\s*:\\s*\"((?:[^\"\\\\]|\\\\.)*)\"")
        val m = rx.find(s) ?: return def
        val raw = m.groupValues[1]
        val sb = StringBuilder(raw.length)
        var i = 0
        while (i < raw.length) {
            val c = raw[i]
            if (c == '\\' && i + 1 < raw.length) {
                when (raw[i + 1]) {
                    '"'  -> { sb.append('"');  i += 2 }
                    '\\' -> { sb.append('\\'); i += 2 }
                    'n'  -> { sb.append('\n'); i += 2 }
                    'r'  -> { sb.append('\r'); i += 2 }
                    't'  -> { sb.append('\t'); i += 2 }
                    'u'  -> {
                        if (i + 5 < raw.length) {
                            val code = raw.substring(i + 2, i + 6).toInt(16)
                            sb.append(code.toChar())
                            i += 6
                        } else { sb.append(c); i += 1 }
                    }
                    else -> { sb.append(c); i += 1 }
                }
            } else { sb.append(c); i += 1 }
        }
        return sb.toString()
    }
}
