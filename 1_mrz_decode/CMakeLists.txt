cmake_minimum_required(VERSION 3.13)
project(mrz_decode C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Wpedantic)

add_library(mrz STATIC src/mrz.c)
target_include_directories(mrz PUBLIC include)

add_executable(mrz_tool src/main.c)
target_link_libraries(mrz_tool PRIVATE mrz)

enable_testing()
add_executable(test_mrz tests/test_mrz.c)
target_link_libraries(test_mrz PRIVATE mrz)
add_test(NAME mrz_unit COMMAND test_mrz)
