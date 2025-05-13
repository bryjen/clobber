project(clobber_mlir_backend CXX)

add_library(clobber_mlir_backend
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/emitter.cpp
)

target_precompile_headers(clobber_common PRIVATE 
    "${CMAKE_CURRENT_SOURCE_DIR}/include/clobber/mlir-backend/pch.hpp"
)


# llvm/mlir linking
if(NOT DEFINED ENV{LLVM_SRC_DIR})
  message(FATAL_ERROR "Environment variable LLVM_SRC_DIR is not set. Set it to the base directory of the LLVM repo.")
endif()
if(NOT DEFINED ENV{LLVM_BUILD_DIR})
  message(FATAL_ERROR "Environment variable LLVM_BUILD_DIR is not set. Set it to the base directory LLVM build output.")
endif()

set(LLVM_MLIR_LIB_DIR "$ENV{LLVM_BUILD_DIR}\\${LLVM_CONFIG}\\lib")

message(STATUS "[LLVM_CONFIG]:           ${LLVM_CONFIG}")
message(STATUS "[ENV:LLVM_SRC_DIR]:      $ENV{LLVM_SRC_DIR}")
message(STATUS "[ENV:LLVM_BUILD_DIR]:    $ENV{LLVM_BUILD_DIR}")
message(STATUS "[LLVM_MLIR_LIB_DIR]:     ${LLVM_MLIR_LIB_DIR}")

# load and link all .lib files
file(GLOB MLIR_LIBS "${LLVM_MLIR_LIB_DIR}/*.lib")
# message(STATUS ${MLIR_LIBS})

target_link_libraries(clobber_mlir_backend 
  PUBLIC
    # llvm/mlir 
    ${MLIR_LIBS}
    ntdll.lib

    # clobber frontend
    clobber
)

target_include_directories(clobber_mlir_backend
  PUBLIC  
    ${CMAKE_CURRENT_SOURCE_DIR}/include

    $ENV{LLVM_SRC_DIR}/llvm/include  # llvm source headers
    $ENV{LLVM_SRC_DIR}/mlir/include  # mlir source headers

    $ENV{LLVM_BUILD_DIR}/include  # llvm/mlir generated headers
    $ENV{LLVM_BUILD_DIR}/tools/mlir/include

  # PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

# configure tests
enable_testing()
add_subdirectory(unittest)