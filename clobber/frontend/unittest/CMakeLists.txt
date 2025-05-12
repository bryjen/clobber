cmake_minimum_required(VERSION 3.11)
project(clobber_tests CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(clobber_tests)

target_precompile_headers(clobber_tests PRIVATE 
    "${CMAKE_CURRENT_SOURCE_DIR}/helpers/pch.hpp"
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/helpers
)

target_sources(clobber_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/helpers/helpers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/helpers/expr_tostring.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/helpers/type_tostring.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/parser_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/semantics_tests.cpp
)


find_package(spdlog CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)
find_package(magic_enum CONFIG REQUIRED)

target_link_libraries(clobber_tests PRIVATE 
    GTest::gtest GTest::gtest_main     
    spdlog::spdlog_header_only
    magic_enum::magic_enum
    clobber clobber_common
)

include(GoogleTest)
add_test(NAME all_clobber_tests COMMAND clobber_tests)

target_compile_definitions(clobber_tests PRIVATE
    ENABLE_TOKENIZER_TESTS
    # ENABLE_PARSER_TESTS
    ENABLE_SEMANTICS_TESTS
)

# copy test files into output executable directory
add_custom_command(
    TARGET clobber_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "[frontend/COPY_TEST_FILES] Copying test .clj files from \"${CMAKE_CURRENT_SOURCE_DIR}/test_files\" to \"$<TARGET_FILE_DIR:clobber_tests>/test_files\""
)

file(GLOB TEST_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test_files/*.clj")

list(LENGTH TEST_FILES test_file_count)
add_custom_command(
    TARGET clobber_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "[frontend/COPY_TEST_FILES] Found ${test_file_count} test .clj file(s)"
)

foreach(test_file ${TEST_FILES})
    get_filename_component(test_file_name ${test_file} NAME)
    add_custom_command(
        TARGET clobber_tests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy "${test_file}" "$<TARGET_FILE_DIR:clobber_tests>/test_files/${test_file_name}"
        COMMAND ${CMAKE_COMMAND} -E echo "[frontend/COPY_TEST_FILES] Copied \"${test_file}\" to \"$<TARGET_FILE_DIR:clobber_tests>/test_files/${test_file_name}\""
    )
endforeach()