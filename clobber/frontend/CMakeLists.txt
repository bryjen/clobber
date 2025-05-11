project(clobber CXX)

add_library(clobber
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/ast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/tokenizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/parser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/semantics.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/base_parser_error.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/lib/clobber/internal/parser_error_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/clobber/internal/diagnostic_factory.cpp
)

target_include_directories(clobber 
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

target_link_libraries(clobber
    PUBLIC clobber_common
)


enable_testing()
add_subdirectory(unittest)