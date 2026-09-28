package com.nextransit.mrzbench.bench

/**
 * Aggregated metrics for one (method, corpus) pair. Same key set as
 * bench_results.json so the on-device vs host comparison is one diff:
 *   n, ok, ok_rate, l1_acc, l2_acc, full_match_rate, cksum_rate,
 *   ms_mean, ms_p50, ms_p95
 */
data class MethodMetrics(
    val n: Int,
    val ok: Int,
    val okRate: Double,
    val l1Acc: Double,
    val l2Acc: Double,
    val fullMatchRate: Double,
    val cksumRate: Double,
    val msMean: Double,
    val msP50: Double,
    val msP95: Double
) {
    fun toJson(): String {
        fun f(x: Double) = String.format("%.6f", x)
        return "{" +
            "\"n\":$n,\"ok\":$ok,\"ok_rate\":${f(okRate)}," +
            "\"l1_acc\":${f(l1Acc)},\"l2_acc\":${f(l2Acc)}," +
            "\"full_match_rate\":${f(fullMatchRate)}," +
            "\"cksum_rate\":${f(cksumRate)}," +
            "\"ms_mean\":${f(msMean)},\"ms_p50\":${f(msP50)},\"ms_p95\":${f(msP95)}" +
        "}"
    }
}
