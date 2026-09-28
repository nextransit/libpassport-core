package com.nextransit.mrzbench.bench

/** Aggregate a list of records into MethodMetrics. */
object Metrics {
    fun from(recs: List<ResultRecord>): MethodMetrics {
        val n = recs.size
        if (n == 0) return MethodMetrics(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        val ok = recs.count { it.ok }
        var l1c = 0; var l1t = 0; var l2c = 0; var l2t = 0
        var full = 0; var cksum = 0
        val durations = DoubleArray(n)
        for ((idx, r) in recs.withIndex()) {
            val (c1, t1) = IcaoChecksum.charAcc(r.gt1, r.line1)
            val (c2, t2) = IcaoChecksum.charAcc(r.gt2, r.line2)
            l1c += c1; l1t += t1; l2c += c2; l2t += t2
            if (c1 == t1 && c2 == t2 && t1 > 0 && t2 > 0) full++
            if (IcaoChecksum.line2Ok(r.line2)) cksum++
            durations[idx] = r.ms
        }
        val sorted = durations.sorted()
        fun p(p: Double): Double {
            if (sorted.isEmpty()) return 0.0
            val rank = (p * (sorted.size - 1)).coerceIn(0.0, (sorted.size - 1).toDouble())
            val lo = rank.toInt()
            val hi = (lo + 1).coerceAtMost(sorted.size - 1)
            val frac = rank - lo
            return sorted[lo] * (1.0 - frac) + sorted[hi] * frac
        }
        return MethodMetrics(
            n              = n,
            ok             = ok,
            okRate         = ok.toDouble() / n,
            l1Acc          = if (l1t > 0) l1c.toDouble() / l1t else 0.0,
            l2Acc          = if (l2t > 0) l2c.toDouble() / l2t else 0.0,
            fullMatchRate  = full.toDouble() / n,
            cksumRate      = cksum.toDouble() / n,
            msMean         = durations.average(),
            msP50          = p(0.50),
            msP95          = p(0.95),
        )
    }
}
