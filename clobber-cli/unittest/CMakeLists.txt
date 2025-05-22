cmake_minimum_required(VERSION 3.11)
project(clobber_cli_tests CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CLI11 CONFIG REQUIRED)

add_executable(clobber_cli_tests)

include_directories(
    # manually include parent proj. headers
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_sources(clobber_cli_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/args_parsing_tests.cpp

    # manually add parent proj. source files 
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/args.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/actions/build_run.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/actions/new.cpp
)


find_package(GTest CONFIG REQUIRED)
target_link_libraries(clobber_cli_tests PRIVATE 
    clobber
    clobber_mlir_backend

    GTest::gtest GTest::gtest_main     
    CLI11::CLI11
)

include(GoogleTest)
add_test(NAME all_clobber_cli_tests COMMAND clobber_cli_tests)