# MrzBench JNI — bridges the in-tree 3_face_recognition + 6_mrz_ocr C
# pipelines into the Android app. The goal is byte-identical per-image
# results with the host ./build/mrz_ocr_tool / mrz_ocr_cnn_tool so the
# on-device batch benchmark is directly comparable to
# 6_mrz_ocr/bench_results.json.
cmake_minimum_required(VERSION 3.22.1)
project(mrzbench_jni LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force march=native OFF for arm cross-compile; let NDK pick defaults.
# -O3 with FP contraction disabled.
#
# The CNN regression gate is byte-exact: the host build (-O3
# -march=native) and the device build must produce the SAME floating
# point sums, because the fallback weights are selected on a corpus
# that the host produced.  Clang and GCC fuse a*b+c into FMA by
# default (-ffp-contract=fast), which changes the accumulation order
# and therefore the result.  -ffp-contract=off keeps the sums
# identical while still letting -O3 vectorise the loops that are safe
# to vectorise.
#
# -march=native is deliberately NOT used: the library is cross-compiled
# for arm64-v8a and must not depend on the build machine's CPU.
#
# -O2 stays.  MEASURED on the device (192.168.26.131, 2026-11-05): -O3
# buys nothing there and is slightly SLOWER, and it changes nothing
# numerically:
#   -O2  conv 85899.9 ns/glyph   fc 18399.9   probs_fnv 27df7d5a60aa147a
#   -O3  conv 87507.2 ns/glyph   fc 17728.4   probs_fnv 27df7d5a60aa147a
# The host and the device never matched bit for bit anyway (Apple clang
# on Darwin vs NDK clang on Android, same sources and flags), so the
# byte-exact gate was always host-vs-host, not host-vs-device.
#
# The real device win came from the decoder instead: see
# 6_mrz_ocr/src/mrz_ocr_cnn.c (precomputed beam tables + incremental
# five-segment checksum) -- p95 147.1 ms -> 88.8 ms with the accuracy
# unchanged bit for bit.
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -DNDEBUG -O2 -ffp-contract=off -fvisibility=hidden")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -DNDEBUG -O2 -ffp-contract=off -fvisibility=hidden")

# Re-use the exact source from the host project so the CNN weights and
# geometric heuristics stay in lock-step with bench_results.json.
set(ROOT_PASSPORT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../..)
set(FACE_DIR ${ROOT_PASSPORT}/3_face_recognition)
set(MRZ_DIR  ${ROOT_PASSPORT}/6_mrz_ocr)

# 1) Face image IO + grayscale + resize (subset of what mrz_ocr needs)
add_library(face STATIC
    ${FACE_DIR}/src/image.c
    ${FACE_DIR}/src/process.c)
set_source_files_properties(${FACE_DIR}/src/image.c ${FACE_DIR}/src/process.c PROPERTIES COMPILE_OPTIONS "-fvisibility=default;")
target_include_directories(face PUBLIC
    ${FACE_DIR}/include
    ${FACE_DIR}/src)
# mrz_ocr only uses face_image_t + image_load/save/to_grayscale;
# process.c adds resize_nn etc. which are unused but harmless and tiny.
target_link_libraries(face PUBLIC m)

# 2) MRZ OCR engine — both backends live in one static archive.
# -fvisibility=default so mrz_ocr_recognise{,_cnn}, mrz_ocr_init,
# mrz_ocr_strerror stay linkable from the JNI shim. The host build uses
# the NDK clang default (default visibility) so per-symbol decisions
# match the host `./build/mrz_ocr_cnn_tool` behaviour.
add_library(mrz_ocr STATIC
    ${MRZ_DIR}/src/template.c
    ${MRZ_DIR}/src/mrz_geom.c
    ${MRZ_DIR}/src/mrz_ocr.c
    ${MRZ_DIR}/src/mrz_ocr_cnn.c
    ${MRZ_DIR}/src/mrz_ocr_crnn.c
    ${MRZ_DIR}/src/mrz_classify.c
    ${MRZ_DIR}/src/crnn.c
    ${MRZ_DIR}/src/cnn.c)
set_source_files_properties(${MRZ_DIR}/src/template.c ${MRZ_DIR}/src/mrz_geom.c ${MRZ_DIR}/src/mrz_ocr.c ${MRZ_DIR}/src/mrz_ocr_cnn.c ${MRZ_DIR}/src/mrz_ocr_crnn.c ${MRZ_DIR}/src/mrz_classify.c ${MRZ_DIR}/src/crnn.c ${MRZ_DIR}/src/cnn.c PROPERTIES COMPILE_OPTIONS "-fvisibility=default;")
target_include_directories(mrz_ocr PUBLIC
    ${MRZ_DIR}/include
    ${MRZ_DIR}/src
    ${FACE_DIR}/include)
target_link_libraries(mrz_ocr PUBLIC face m)

# The weights live in headers, and the arrays are `static const`, so the
# compiler cannot see a dependency on them: replacing a weight header
# does NOT invalidate the object files and the APK silently keeps the
# OLD weights.  The host build declares this dependency
# (6_mrz_ocr/CMakeLists.txt, MRZ_WEIGHT_HEADERS); the app needs the same.
set(MRZ_WEIGHT_HEADERS
    ${MRZ_DIR}/src/cnn_weights_line1.h
    ${MRZ_DIR}/src/cnn_weights_line2.h
    ${MRZ_DIR}/src/crnn_weights.h
    ${MRZ_DIR}/src/cnn_weights.h)
set_property(TARGET mrz_ocr APPEND PROPERTY
             OBJECT_DEPENDS ${MRZ_WEIGHT_HEADERS})
# Needed when the recogniser emits per-stage TIMING to logcat
# (mrz_ocr_cnn.c only references __android_log_print under
# #ifdef __ANDROID__, but we always link on the Android target
# to keep the source identical across host/device builds).
target_link_libraries(mrz_ocr PUBLIC log)

# 3) JNI shim that turns Java strings into UTF-8 paths and forwards
#    per-image calls into the same C API the host tools use.
add_library(mrz_jni SHARED
    mrz_jni.cpp
    mrz_run.cpp)
target_link_libraries(mrz_jni PRIVATE mrz_ocr face log)
target_include_directories(mrz_jni PRIVATE
    ${MRZ_DIR}/include
    ${MRZ_DIR}/src
    ${FACE_DIR}/include)
target_compile_options(mrz_jni PRIVATE -fno-exceptions -fno-rtti)
