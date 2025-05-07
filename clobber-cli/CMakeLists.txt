project(clobber_cli CXX)

find_package(CLI11 CONFIG REQUIRED)

add_executable(clobber_cli
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/args.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/actions/build_run.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/actions/new.cpp
)

target_include_directories(clobber_cli PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(clobber_cli PRIVATE 
    clobber
    clobber_mlir_backend

    CLI11::CLI11
)

enable_testing()
add_subdirectory(unittest)