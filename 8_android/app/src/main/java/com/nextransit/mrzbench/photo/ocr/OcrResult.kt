package com.nextransit.mrzbench.photo.ocr

data class OcrResult(
    val engineName: String,
    val line1: String,
    val line2: String,
    val conf1: Int,
    val conf2: Int,
    val ms: Long,
    val error: String? = null,
) {
    fun isEmpty(): Boolean = line1.isEmpty() && line2.isEmpty()
    val ok: Boolean get() = error == null && (line1.isNotEmpty() || line2.isNotEmpty())
}
