package com.nextransit.mrzbench.photo.ocr.tesseract

import com.googlecode.tesseract.android.TessBaseAPI

class NativeTess private constructor(private val api: TessBaseAPI) {
    fun setPsm(psm: Int) { api.pageSegMode = psm }
    fun recognize(bytes: ByteArray, width: Int, height: Int, bpp: Int, bpl: Int): String {
        api.setImage(bytes, width, height, bpp, bpl)
        val text = api.utF8Text
        api.clear()
        return text
    }
    fun close() { api.end() }

    companion object {
        fun create(lang: String = "ocrb"): NativeTess? = try {
            val api = TessBaseAPI()
            val ok = api.init("tessdata", lang)
            if (!ok) { api.end(); null } else NativeTess(api)
        } catch (_: Throwable) { null }
    }
}
