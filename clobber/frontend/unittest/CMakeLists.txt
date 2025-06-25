project(clobber_tests C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(clobber_tests)

target_precompile_headers(clobber_tests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/helpers/pch.hpp"
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/
    ${CMAKE_CURRENT_SOURCE_DIR}/helpers
)

target_sources(clobber_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/helpers/helpers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/helpers/tostring.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/helpers/syntax_factory.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/test_cases.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/parser_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/semantics_tests.cpp
)

find_package(GTest CONFIG REQUIRED)
find_package(magic_enum CONFIG REQUIRED)

target_link_libraries(clobber_tests PRIVATE
    GTest::gtest GTest::gtest_main
    magic_enum::magic_enum
    clobber clobber_common
)

set_target_properties(clobber_tests PROPERTIES UNITY_BUILD ON)

include(GoogleTest)
add_test(NAME all_clobber_tests COMMAND clobber_tests)

target_compile_definitions(clobber_tests PRIVATE
    # ENABLE_TOKENIZER_TESTS
    ENABLE_PARSER_TESTS
    # ENABLE_SEMANTICS_TESTS
)
