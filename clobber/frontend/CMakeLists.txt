project(clobber CXX)

add_library(clobber
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/tokenizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/parser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/semantics.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/macro_expand.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/lib/ast/ast_accel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/ast/ast_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/ast/ast_macros.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/ast/ast_traversal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/ast/ast_span.cpp

#    ${CMAKE_CURRENT_SOURCE_DIR}/lib/clobber/internal/diagnostic_factory.cpp
)

target_precompile_headers(clobber PRIVATE 
    "${CMAKE_CURRENT_SOURCE_DIR}/include/clobber/pch.hpp"
)

target_include_directories(clobber 
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

find_package(magic_enum CONFIG REQUIRED)
target_link_libraries(clobber
    PUBLIC  clobber_common
    PRIVATE magic_enum::magic_enum
)

set_target_properties(clobber PROPERTIES UNITY_BUILD ON)


enable_testing()
add_subdirectory(unittest)