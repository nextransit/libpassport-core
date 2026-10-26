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
    fun close() { api.recycle() }

    companion object {
        private const val WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"

        fun create(dataParent: String, lang: String = "ocrb"): NativeTess? = try {
            val api = TessBaseAPI()
            val ok = api.init(dataParent, lang)
            android.util.Log.i("MrzOcr", "tess init '$dataParent' ok=$ok")
            if (!ok) {
                api.recycle()
                null
            } else {
                val wl = api.setVariable("tessedit_char_whitelist", WHITELIST)
                android.util.Log.i("MrzOcr", "tess whitelist set=$wl")
                NativeTess(api)
            }
        } catch (t: Throwable) {
            android.util.Log.e("MrzOcr", "tess create failed: ${t.message}", t)
            null
        }
    }
}
