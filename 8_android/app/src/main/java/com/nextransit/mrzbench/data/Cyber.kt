package com.nextransit.mrzbench.data

/**
 * Cyber-tech semantic colour palette — mirror of `.impeccable.md` and
 * skills/gui-cyber-style/SKILL.md. Every UI surface must read tokens
 * from here; never hardcode hex at the call site.
 */
object Cyber {
    // Surfaces
    val bg          = 0xFF0A0E14.toInt()   // deep space canvas
    val card        = 0xFF161F30.toInt()   // elevated card
    val field       = 0xFF111722.toInt()   // input / console field
    val border      = 0xFF1F2A3D.toInt()   // 1px hairline
    val borderHi    = 0xFF2E3F5C.toInt()   // hover / focus

    // Text
    val text        = 0xFFE6EDF3.toInt()
    val textDim     = 0xFF8B98A8.toInt()
    val textMute    = 0xFF5A6678.toInt()

    // Neon accents
    val neonCyan    = 0xFF00E5FF.toInt()
    val neonBlue    = 0xFF2E86FF.toInt()
    val neonPurple  = 0xFFB388FF.toInt()

    // Status (chip badges)
    val ok          = 0xFF2EE6A8.toInt()
    val err         = 0xFFFF5C7A.toInt()
    val warn        = 0xFFFFCB6B.toInt()
    val run         = 0xFF00E5FF.toInt()   // alias of cyan for "in-flight"
    val idle        = 0xFF5A6678.toInt()
    val dim         = 0xFF3D4856.toInt()

    // Method palette (each backend keeps its own accent)
    val tradAccent  = neonPurple
    val cnnAccent   = neonCyan
}
