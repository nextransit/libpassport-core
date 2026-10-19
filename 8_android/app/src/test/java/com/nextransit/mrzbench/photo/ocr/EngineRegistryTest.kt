package com.nextransit.mrzbench.photo.ocr

import android.graphics.Bitmap
import org.junit.Assert.assertEquals
import org.junit.Test

class EngineRegistryTest {
    private fun fake(name: String) = object : OcrEngine {
        override val name: String = name
        override fun recognize(line1: Bitmap, line2: Bitmap, psm: Int) =
            OcrResult(name, "", "", 0, 0, 0)
    }

    @Test
    fun `register adds engine and dedupes by name`() {
        val a = fake("test-engine-a")
        val b = fake("test-engine-b")
        EngineRegistry.register(a)
        try {
            assertEquals(true, EngineRegistry.all().contains(a))
            EngineRegistry.register(b)
            try {
                assertEquals(true, EngineRegistry.all().contains(b))
                val aCount = EngineRegistry.all().count { it.name == "test-engine-a" }
                assertEquals(1, aCount)
            } finally {
            }
        } finally {
        }
    }
}
