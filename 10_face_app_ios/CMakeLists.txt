# Build the vendored SeetaFace2 components as static archives (.a) for
# the iOS target. These archives are then linked by the hand-written
# FaceApp.xcodeproj that wraps the Swift / Objective-C++ source tree.
#
# This file intentionally avoids being an iOS app target itself: CMake's
# Xcode generator does not yet reliably emit iOS app projects on
# Xcode 15+ (issues around Swift runtime auto-linking, simulator SDK
# selection, and FRAMEWORK_SEARCH_PATHS), so we generate the static
# libs here and use a checked-in .xcodeproj for the app shell.

cmake_minimum_required(VERSION 3.18)
project(faceapp_ios_prereqs LANGUAGES C CXX OBJC OBJCXX)

set(SEETA2_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../3_1_seetaface2" CACHE PATH
    "Path to the vendored 3_1_seetaface2 source tree")
if(NOT EXISTS "${SEETA2_ROOT}/CMakeLists.txt")
    message(FATAL_ERROR "faceapp_ios: cannot find 3_1_seetaface2 at ${SEETA2_ROOT}")
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "")
set(BUILD_EXAMPLE OFF CACHE BOOL "")
set(BUILD_DETECOTOR ON CACHE BOOL "")
set(BUILD_LANDMARKER ON CACHE BOOL "")
set(BUILD_RECOGNIZER ON CACHE BOOL "")

# Vendor enables -msse2/-mfma only on x86/x64 PLATFORM strings. iOS
# is arm64, so pin a sentinel to keep the x86 flags off.
if(NOT DEFINED PLATFORM OR PLATFORM STREQUAL "")
    set(PLATFORM "ios-arm64" CACHE STRING "" FORCE)
endif()

add_subdirectory(${SEETA2_ROOT} seetaface2_build)

# Inject SeetaNet/include so internal headers resolve. Also force
# both spellings of __ARM_NEON to dodge the vendor typo:
#   #if defined(__ARM_NEON__) || defined(__ARM_NEON)   <-- second is a typo
foreach(_seeta_target SeetaFaceDetector SeetaFaceLandmarker SeetaFaceRecognizer SeetaNet)
    if(NOT TARGET ${_seeta_target})
        continue()
    endif()
    target_include_directories(${_seeta_target} BEFORE PRIVATE
        ${SEETA2_ROOT}/SeetaNet/include)
    target_compile_definitions(${_seeta_target} PRIVATE
        __ARM_NEON=1
        __ARM_NEON__=1)
endforeach()
