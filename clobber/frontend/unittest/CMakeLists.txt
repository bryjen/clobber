cmake_minimum_required(VERSION 3.11)
project(clobber_tests CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/parser_tests.cpp
)


find_package(GTest CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE 
    GTest::gtest GTest::gtest_main     
    clobber
)

include(GoogleTest)
# gtest_discover_tests(${PROJECT_NAME})
add_test(NAME all_clobber_tests COMMAND ${PROJECT_NAME})