cmake_minimum_required(VERSION 3.13)
project(anticounterfeit C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Wpedantic)

# Depend on the MRZ library shipped in ../1_mrz_decode
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../1_mrz_decode
                 ${CMAKE_CURRENT_BINARY_DIR}/mrz_build)

add_library(ac STATIC src/ac.c)
target_include_directories(ac PUBLIC include)
target_link_libraries(ac PUBLIC mrz)

add_executable(ac_tool src/main.c)
target_link_libraries(ac_tool PRIVATE ac)

enable_testing()
add_executable(test_ac tests/test_ac.c)
target_link_libraries(test_ac PRIVATE ac)
add_test(NAME ac_unit COMMAND test_ac)
