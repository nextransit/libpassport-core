package com.nextransit.mrzbench.bench

/**
 * ICAO Doc 9303 TD3 line-2 check digits, ported from
 * 6_mrz_ocr/tests/bench.py (check_digit + line2_checksum_ok). Same
 * weight cycle 7,3,1 and same field slicing. Keep this exact — the
 * `cksum_rate` metric in the on-device report must agree to the bit
 * with the host bench_results.json.
 */
object IcaoChecksum {
    private val W = intArrayOf(7, 3, 1)

    fun digitFor(seg: String): Int {
        var total = 0
        for (i in seg.indices) {
            val c = seg[i]
            val v = when {
                c in '0'..'9' -> c.code - 48
                c in 'A'..'Z' -> c.code - 55
                else          -> 0
            }
            total += v * W[i % 3]
        }
        return total % 10
    }

    /** True if all five line-2 check digits validate. */
    fun line2Ok(l2: String): Boolean {
        if (l2.length < 44) return false
        return try {
            digitFor(l2.substring(0, 9)) == l2[9].digitToInt() &&
            digitFor(l2.substring(13, 19)) == l2[19].digitToInt() &&
            digitFor(l2.substring(21, 27)) == l2[27].digitToInt() &&
            digitFor(l2.substring(28, 42)) == l2[42].digitToInt() &&
            digitFor(l2.substring(0, 10) + l2.substring(13, 20) +
                     l2.substring(21, 43)) == l2[43].digitToInt()
        } catch (_: IllegalArgumentException) { false }
    }

    /** Char-level diff: returns Pair(correct, total) over max(len1, len2). */
    fun charAcc(gt: String, got: String): Pair<Int, Int> {
        val n = maxOf(gt.length, got.length)
        var correct = 0
        for (i in 0 until n) {
            val g = if (i < gt.length) gt[i] else '_'
            val h = if (i < got.length) got[i] else '_'
            if (g == h) correct++
        }
        return correct to n
    }
}
