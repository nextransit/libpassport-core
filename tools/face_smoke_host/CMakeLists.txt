cmake_minimum_required(VERSION 3.18)
project(face_smoke_host LANGUAGES CXX)

# Standalone host smoke for the vendored 3_1_seetaface2 sources.
# - Never copies third-party sources; consumes the existing tree.
# - BUILD_EXAMPLE=OFF so we do not pull in Windows vcxproj trees.
# - The smoke driver fails loudly if the .dat model files are missing.

set(SEETA2_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../3_1_seetaface2" CACHE PATH
    "Path to the vendored 3_1_seetaface2 source tree")

if(NOT EXISTS "${SEETA2_ROOT}/CMakeLists.txt")
    message(FATAL_ERROR
        "face_smoke_host: cannot find 3_1_seetaface2 at ${SEETA2_ROOT}. "
        "Re-run with -DSEETA2_ROOT=/path/to/3_1_seetaface2")
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libs (host smoke)")
set(BUILD_EXAMPLE OFF CACHE BOOL "")
set(BUILD_DETECOTOR ON CACHE BOOL "")
set(BUILD_LANDMARKER ON CACHE BOOL "")
set(BUILD_RECOGNIZER ON CACHE BOOL "")

# Vendored SeetaNet enables -msse2/-mfma whenever PLATFORM is x86/x64/
# x86_64. Those flags are invalid on Apple Silicon (arm64), so we pin
# PLATFORM to a non-x86 sentinel value before add_subdirectory runs. The
# sentinel only affects string-based conditional logic in the vendor
# tree; actual target architecture is controlled by the host toolchain.
if(NOT DEFINED PLATFORM OR PLATFORM STREQUAL "")
    set(PLATFORM "arm64-apple-macos" CACHE STRING "host smoke: skip x86 flags" FORCE)
endif()

add_subdirectory(${SEETA2_ROOT} seetaface2_build)

# The vendored component CMakeLists declare target_include_directories
# with PRIVATE for their own include/ trees, and reference
# ${CMAKE_SOURCE_DIR}/SeetaNet/include which resolves to OUR project
# root, not to the vendored tree. Patch the include path on the
# produced targets AFTER add_subdirectory, so the dependency chain is
# preserved. We use BEFORE so the path is tried first; the original
# PRIVATE entries from the vendor remain in place.
foreach(seeta_target SeetaFaceDetector SeetaFaceLandmarker SeetaFaceRecognizer)
    if(TARGET ${seeta_target})
        target_include_directories(${seeta_target} BEFORE PRIVATE
            ${SEETA2_ROOT}/SeetaNet/include
        )
    endif()
endforeach()

add_executable(face_smoke_host
    src/main.cpp
)

target_include_directories(face_smoke_host PRIVATE
    ${SEETA2_ROOT}
    ${SEETA2_ROOT}/FaceDetector/include
    ${SEETA2_ROOT}/FaceLandmarker/include
    ${SEETA2_ROOT}/FaceRecognizer/include
    ${SEETA2_ROOT}/SeetaNet/include
)

target_link_libraries(face_smoke_host
    PRIVATE
        SeetaFaceDetector
        SeetaFaceLandmarker
        SeetaFaceRecognizer
)

# Make sure the executable is scheduled only after the three archives
# are produced. Without this, the top-level Makefile may attempt to
# link before the subdirectory archives exist.
add_dependencies(face_smoke_host
    SeetaFaceDetector
    SeetaFaceLandmarker
    SeetaFaceRecognizer
)

set_target_properties(face_smoke_host PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
