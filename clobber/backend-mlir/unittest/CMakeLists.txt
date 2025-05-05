cmake_minimum_required(VERSION 3.11)
project(clobber_mlir_backend_tests CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(clobber_mlir_backend_tests)

target_sources(clobber_mlir_backend_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/emitter_tests.cpp
)

# TODO: figure out how to properly propagate (or not) llvm/mlir dependencies

file(GLOB MLIR_LIBS "$ENV{LLVM_MLIR_LIB_DIR}/*.lib")
target_link_libraries(clobber_mlir_backend_tests PRIVATE 
    ${MLIR_LIBS}
    ntdll.lib
)

target_include_directories(clobber_mlir_backend_tests
  PUBLIC  
    ${CMAKE_CURRENT_SOURCE_DIR}/helpers

    $ENV{LLVM_SRC_DIR}/llvm/include  # llvm source headers
    $ENV{LLVM_SRC_DIR}/mlir/include  # mlir source headers

    $ENV{LLVM_BUILD_DIR}/include  # llvm/mlir generated headers
    $ENV{LLVM_BUILD_DIR}/tools/mlir/include

  PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

# include_directories(
#     ${CMAKE_CURRENT_SOURCE_DIR}/helpers
# )



find_package(GTest CONFIG REQUIRED)
target_link_libraries(clobber_mlir_backend_tests PRIVATE 
    GTest::gtest GTest::gtest_main     
    # clobber
    # clobber_mlir_backend
)

include(GoogleTest)
add_test(NAME all_clobber_tests COMMAND clobber_mlir_backend_tests)


# copy test files into output executable directory
add_custom_command(
    TARGET clobber_mlir_backend_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "[backend-mlir/COPY_TEST_FILES] Copying test .mlir files from \"${CMAKE_CURRENT_SOURCE_DIR}/test_files\" to \"$<TARGET_FILE_DIR:clobber_mlir_backend_tests>/test_files\""
)

# file(GLOB TEST_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test_files/*.mlir")
file(GLOB_RECURSE TEST_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test_files/**/*.mlir")

list(LENGTH TEST_FILES test_file_count)
add_custom_command(
    TARGET clobber_mlir_backend_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "[backend-mlir/COPY_TEST_FILES] Found ${test_file_count} test .mlir file(s)"
)

foreach(test_file ${TEST_FILES})
    get_filename_component(test_file_name ${test_file} NAME)
    add_custom_command(
        TARGET clobber_mlir_backend_tests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy "${test_file}" "$<TARGET_FILE_DIR:clobber_mlir_backend_tests>/test_files/${test_file_name}"
        COMMAND ${CMAKE_COMMAND} -E echo "[backend-mlir/COPY_TEST_FILES] Copied \"${test_file}\" to \"$<TARGET_FILE_DIR:clobber_mlir_backend_tests>/test_files/${test_file_name}\""
    )
endforeach()