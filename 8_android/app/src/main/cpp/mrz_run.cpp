// Placeholder for future native helpers (e.g. streaming output writer).
// Kept as a separate translation unit so JNI bridge code stays small
// and self-contained. The C pipeline lives in libmrz_ocr.a; this file
// exists purely to satisfy CMakeLists.txt's add_library(mrz_jni ...) target.
#include <stddef.h>

extern "C" {
    void mrz_jni_version_note(void) { /* intentionally empty */ }
}
