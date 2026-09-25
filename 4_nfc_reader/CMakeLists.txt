cmake_minimum_required(VERSION 3.13)
project(nfc_reader C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Wpedantic)

add_library(nfc STATIC src/nfc.c src/mock_json.c)
target_include_directories(nfc PUBLIC include)

# Force libc linkage so unresolved fopen in libnfc is resolved.
add_executable(nfc_tool src/main.c)
target_link_libraries(nfc_tool PRIVATE nfc)
# Pull in ALL object files from libnfc.a, even those that aren't
# referenced from main.c. We do this because mock_json.o uses fopen
# which main.c does not, so without --whole-archive the linker
# drops the whole library because no required symbols are missing.
if (APPLE)
    # Force-load every .o inside libnfc.a so unused symbols (fopen
    # referenced only by mock_json.c) are still pulled in. */
    set_target_properties(nfc_tool PROPERTIES
        LINK_FLAGS "-Wl,-force_load,libnfc.a")
else()
    target_link_libraries(nfc_tool PRIVATE -Wl,--whole-archive nfc -Wl,--no-whole-archive)
endif()
target_link_libraries(nfc_tool PRIVATE nfc)

enable_testing()
add_test(NAME nfc_smoke COMMAND ${CMAKE_COMMAND} -E echo "NFC mock backend; run ./run.sh for end-to-end test")
