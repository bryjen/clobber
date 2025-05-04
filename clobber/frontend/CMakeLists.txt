project(clobber CXX)

add_library(clobber
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/ast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/tokenizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/parser.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/lib/clobber/internal/parser_errors.cpp
)

target_include_directories(clobber 
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

enable_testing()
add_subdirectory(unittest)