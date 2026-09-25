cmake_minimum_required(VERSION 3.13)
project(face_recognition C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Wpedantic)

add_library(face STATIC src/image.c src/process.c)
target_include_directories(face PUBLIC include)
find_library(M_LIB m)
target_link_libraries(face PUBLIC ${M_LIB})

add_executable(face_tool src/main.c)
target_link_libraries(face_tool PRIVATE face)

add_executable(gen_sample tools/gen_sample.c)
target_link_libraries(gen_sample PRIVATE face)

enable_testing()
add_executable(test_face tests/test_face.c)
target_link_libraries(test_face PRIVATE face)
add_test(NAME face_unit COMMAND test_face)
