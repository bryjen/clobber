cmake_minimum_required(VERSION 3.11)
project(clobber_tests CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer_tests.cpp
)


find_package(GTest CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE 
    GTest::gtest GTest::gtest_main     
    clobber
)

include(GoogleTest)
add_test(NAME all_clobber_tests COMMAND ${PROJECT_NAME})


# copy test files into output executable directory
add_custom_command(
    TARGET clobber_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "[frontend/COPY_TEST_FILES] Copying test .clj files from \"${CMAKE_CURRENT_SOURCE_DIR}/test_files\" to \"$<TARGET_FILE_DIR:clobber_tests>/test_files\""
)

file(GLOB TEST_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test_files/*.clj")

list(LENGTH TEST_FILES shader_count)
add_custom_command(
    TARGET clobber_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "[frontend/COPY_TEST_FILES] Found ${shader_count} test .clj file(s)"
)

foreach(test_file ${TEST_FILES})
    get_filename_component(test_file_name ${test_file} NAME)
    add_custom_command(
        TARGET clobber_tests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy "${test_file}" "$<TARGET_FILE_DIR:clobber_tests>/test_files/${test_file_name}"
        COMMAND ${CMAKE_COMMAND} -E echo "[frontend/COPY_TEST_FILES] Copied \"${test_file}\" to \"$<TARGET_FILE_DIR:clobber_tests>/test_files/${test_file_name}\""
    )
endforeach()