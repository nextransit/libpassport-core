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
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -DNDEBUG -O2 -fvisibility=hidden")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -DNDEBUG -O2 -fvisibility=hidden")

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
    ${MRZ_DIR}/src/cnn.c)
set_source_files_properties(${MRZ_DIR}/src/template.c ${MRZ_DIR}/src/mrz_geom.c ${MRZ_DIR}/src/mrz_ocr.c ${MRZ_DIR}/src/mrz_ocr_cnn.c ${MRZ_DIR}/src/cnn.c PROPERTIES COMPILE_OPTIONS "-fvisibility=default;")
target_include_directories(mrz_ocr PUBLIC
    ${MRZ_DIR}/include
    ${MRZ_DIR}/src
    ${FACE_DIR}/include)
target_link_libraries(mrz_ocr PUBLIC face m)

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
