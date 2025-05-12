project(clobber_common CXX)

add_library(clobber_common 
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/utils.cpp
)

target_precompile_headers(clobber_common PRIVATE 
    "${CMAKE_CURRENT_SOURCE_DIR}/include/clobber/common/pch.hpp"
)

target_include_directories(clobber_common
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(CLOBBER_USE_CRT)
    message(STATUS "[clobber_common] Windows CRT debugging enabled for non-release Windows builds.")
    target_compile_definitions(clobber_common PUBLIC CLOBBER_USE_CRT)
endif()