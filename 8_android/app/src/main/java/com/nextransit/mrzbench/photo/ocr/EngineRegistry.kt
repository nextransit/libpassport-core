package com.nextransit.mrzbench.photo.ocr

object EngineRegistry {
    private val engines = mutableListOf<OcrEngine>()

    fun register(engine: OcrEngine) {
        if (engines.none { it.name == engine.name }) {
            engines.add(engine)
        }
    }

    fun all(): List<OcrEngine> = engines.toList()
}
