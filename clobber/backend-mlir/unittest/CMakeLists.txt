project(clobber_mlir_backend_tests C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(clobber_mlir_backend_tests)

target_sources(clobber_mlir_backend_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/emitter_tests.cpp
)

target_include_directories(clobber_mlir_backend_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/helpers
    ${CMAKE_CURRENT_SOURCE_DIR}/lib
    ${MLIR_INCLUDE_DIRS}
)

find_package(MLIR CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)
find_package(magic_enum CONFIG REQUIRED)

target_link_libraries(clobber_mlir_backend_tests PRIVATE
    GTest::gtest GTest::gtest_main
    magic_enum::magic_enum
    clobber clobber_mlir_backend clobber_common
)

target_compile_definitions(clobber_mlir_backend_tests PRIVATE
    ${MLIR_DEFINITIONS}
)

if (MSVC)
    target_compile_options(clobber_mlir_backend_tests PRIVATE /MP)
endif()

include(GoogleTest)
add_test(NAME all_clobber_tests COMMAND clobber_mlir_backend_tests)

# copy test files into output executable directory
add_custom_command(
    TARGET clobber_mlir_backend_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "[backend-mlir/COPY_TEST_FILES] Copying test .mlir files from \"${CMAKE_CURRENT_SOURCE_DIR}/test_files\" to \"$<TARGET_FILE_DIR:clobber_mlir_backend_tests>/test_files\""
)

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
