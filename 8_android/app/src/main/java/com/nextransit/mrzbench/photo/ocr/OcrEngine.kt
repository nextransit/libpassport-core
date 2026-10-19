package com.nextransit.mrzbench.photo.ocr

import android.graphics.Bitmap

interface OcrEngine {
    val name: String
    fun recognize(line1: Bitmap, line2: Bitmap, psm: Int = 7): OcrResult
}
