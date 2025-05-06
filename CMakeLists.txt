cmake_minimum_required(VERSION 3.11)
project(clobber CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


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
add_subdirectory(clobber/backend-mlir)
add_subdirectory(clobber/cli)