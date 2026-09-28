package com.nextransit.mrzbench.bench

/** One per-image record. Mirrors the JSONL shape used in
 *  6_mrz_ocr/tests/bench.py (subset we keep on-device). */
data class ResultRecord(
    val id: String,
    val backend: String,         // "traditional" | "cnn"
    val profile: String,
    val scale: Int,
    val skew: Int,
    val noise: Double,
    val gt1: String,
    val gt2: String,
    val line1: String,
    val line2: String,
    val conf1: Int,
    val conf2: Int,
    val ms: Double,
    val ok: Boolean,
    val errMsg: String = ""
)
