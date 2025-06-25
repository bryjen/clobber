cmake_minimum_required(VERSION 3.12)
project(clobber C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS_LINK_COMPILE_COMMANDS ON)

add_custom_target(link_compile_commands ALL
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_BINARY_DIR}/compile_commands.json
        ${CMAKE_SOURCE_DIR}/compile_commands.json
    DEPENDS ${CMAKE_BINARY_DIR}/compile_commands.json
    COMMENT "Creating symlink to compile_commands.json"
)

# global build options (propagated to sub-projects)
if (CLOBBER_STRICT)
    message(STATUS "[root] strict (pedantic) build enabled")
    if(MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message(STATUS "[root] strictest possible warning configuration for clang (+ clang-cl) set")
        add_compile_options(
            /W4                         # high MSVC warning level
            /WX                         # warnings as errors
            /permissive-                # strict standard conformance (MSVC)
            /Zc:__cplusplus             # use real c++ standard value
            /clang:-Wall                # enable Clang warnings
            /clang:-Wextra              # more Clang warnings
            /clang:-Wpedantic           # iso conformance warnings
            /clang:-Wconversion         # implicit conversions
            /clang:-Wsign-conversion    # signed/unsigned conversions
            /clang:-Wshadow             # variable shadowing
            /clang:-Wnull-dereference   # null pointer use
            /clang:-Wdouble-promotion   # float to double silently
            /clang:-Wunused-parameter   # unused function args
        )
    elseif(MSVC)
        message(STATUS "[root] strictest possible warning configuration for MSVC (cl) set")
        target_compile_options(your_target PRIVATE
            /W4
            /WX
            /permissive-
            /Zc:__cplusplus
        )
    endif()
endif()


# Project is tested to work on woa64>
# The project might not compile properly on any other platform/architecture combination.
if(WIN32 AND CMAKE_SYSTEM_PROCESSOR STREQUAL "ARM64")
    message(STATUS "Detected Windows ARM64 (woa64).")
else()
    message(WARNING "Detected ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}, the project is only tested on Windows/ARM64 - building on this platform may not work.")
endif()

# vcpkg setup
if(DEFINED ENV{VCPKG_ROOT})
    message(STATUS "[vcpkg] vcpkg found.")
    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/deps/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
    message(STATUS "[vcpkg] \"CMAKE_TOOLCHAIN_FILE\" set to ${CMAKE_TOOLCHAIN_FILE}")
endif()


enable_testing()

add_subdirectory(clobber/frontend)
add_subdirectory(clobber/common)
add_subdirectory(clobber/backend-mlir)

if (MSVC)
    # global compiler flags (shoutout chatgpt)
    # enable /MP (parallel compile)
    add_compile_options($<$<CXX_COMPILER_ID:MSVC>:/MP>)

    # ennable /Bt+ (profile per-file compile times)
    add_compile_options($<$<CXX_COMPILER_ID:MSVC>:/Bt+>)

    # enable /debug:fastlink (faster linking in Debug)
    add_link_options($<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:MSVC>>:/debug:fastlink>)
endif()
