project(clobber_common CXX)

add_library(clobber_common INTERFACE
)

target_include_directories(clobber_common
    INTERFACE  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(CLOBBER_USE_CRT)
    message(STATUS "[clobber_common] Windows CRT debugging enabled for non-release Windows builds.")
    target_compile_definitions(clobber_common INTERFACE CLOBBER_USE_CRT)
endif()