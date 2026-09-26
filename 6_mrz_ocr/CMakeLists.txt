cmake_minimum_required(VERSION 3.13)
project(mrz_ocr C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Wpedantic -O3 -DNDEBUG -march=native)

# Re-use the PPM/BMP I/O and grayscale helpers from 3_face_recognition.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../3_face_recognition
                 ${CMAKE_CURRENT_BINARY_DIR}/face_build)

find_library(M_LIB m)
add_library(mrz_ocr STATIC src/template.c src/mrz_ocr.c src/mrz_ocr_cnn.c src/cnn.c)
target_include_directories(mrz_ocr PUBLIC include src)
target_link_libraries(mrz_ocr PUBLIC face ${M_LIB})

add_executable(mrz_ocr_tool src/main.c)
target_link_libraries(mrz_ocr_tool PRIVATE mrz_ocr)

add_executable(mrz_ocr_cnn_tool src/main_cnn.c)
target_link_libraries(mrz_ocr_cnn_tool PRIVATE mrz_ocr)

add_executable(gen_mrz_image tools/gen_mrz_image.c)
target_link_libraries(gen_mrz_image PRIVATE mrz_ocr)

enable_testing()
add_executable(test_mrz_ocr tests/test_mrz_ocr.c)
target_link_libraries(test_mrz_ocr PRIVATE mrz_ocr)
add_test(NAME mrz_ocr_unit COMMAND test_mrz_ocr)

# Post-processing internals (includes mrz_ocr_cnn.c directly to reach
# the static helpers; the archive member is never pulled because all
# its symbols are already defined in this TU).
add_executable(test_postproc tests/test_postproc.c)
target_include_directories(test_postproc PRIVATE include src)
target_link_libraries(test_postproc PRIVATE mrz_ocr)
target_compile_options(test_postproc PRIVATE -O1)
add_test(NAME mrz_postproc COMMAND test_postproc)

# In-process golden test: both backends must reproduce a rendered
# ICAO-valid record exactly, no external corpus files required.
add_executable(test_golden tests/test_golden.c)
target_link_libraries(test_golden PRIVATE mrz_ocr)
add_test(NAME mrz_golden COMMAND test_golden)
