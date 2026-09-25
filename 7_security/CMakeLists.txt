cmake_minimum_required(VERSION 3.13)
project(passport_security C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Wpedantic)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../1_mrz_decode
                 ${CMAKE_CURRENT_BINARY_DIR}/mrz_build)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../3_face_recognition
                 ${CMAKE_CURRENT_BINARY_DIR}/face_build)

find_library(M_LIB m)
add_library(security STATIC src/sha1.c src/des.c src/aes.c src/security.c)
target_include_directories(security PUBLIC include)
# DES/AES are provided by Apple's CommonCrypto on macOS.
if (APPLE)
    target_link_libraries(security PUBLIC mrz face ${M_LIB})
else()
    # On other platforms, link against OpenSSL.
    find_package(OpenSSL REQUIRED)
    target_link_libraries(security PUBLIC mrz face OpenSSL::Crypto ${M_LIB})
endif()

add_executable(security_tool src/main.c)
target_link_libraries(security_tool PRIVATE security mrz face)

enable_testing()
add_executable(test_security tests/test_security.c)
# Static libs must come in dependency order: dependents FIRST so their
# unresolved symbols can be satisfied by dependents. We list mrz
# *before* security in the static chain.
target_link_libraries(test_security PRIVATE security mrz face)
add_test(NAME security_unit COMMAND test_security)
