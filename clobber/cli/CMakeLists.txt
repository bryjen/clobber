project(clobber_cli CXX)

add_executable(clobber_cli
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

target_link_libraries(clobber_cli PRIVATE clobber)