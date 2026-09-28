package com.nextransit.mrzbench.data

/** One record from corpus.json (subset we care about). */
data class CorpusEntry(
    val id: String,
    val image: String,         // e.g. "img_0001_b0_v0.ppm"
    val line1: String,         // GT line 1
    val line2: String,         // GT line 2
    val profile: String,       // "clean" or "realistic"
    val scale: Int,
    val skew: Int,
    val noise: Double
)
