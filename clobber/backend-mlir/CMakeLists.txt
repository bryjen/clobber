project(clobber_mlir_backend CXX)

add_library(clobber_mlir_backend
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/emitter.cpp
)




# llvm/mlir linking
if(NOT DEFINED ENV{LLVM_SRC_DIR})
  message(FATAL_ERROR "Environment variable LLVM_SRC_DIR is not set. Set it to the base directory of the LLVM repo.")
endif()
if(NOT DEFINED ENV{LLVM_BUILD_DIR})
  message(FATAL_ERROR "Environment variable LLVM_BUILD_DIR is not set. Set it to the base directory LLVM build output.")
endif()
if(NOT DEFINED ENV{LLVM_MLIR_LIB_DIR})
  message(FATAL_ERROR "Environment variable LLVM_MLIR_LIB_DIR is not set. Set it to the directory containing .lib files.")
endif()
message(STATUS "[ENV:LLVM_SRC_DIR]:      $ENV{LLVM_SRC_DIR}")
message(STATUS "[ENV:LLVM_BUILD_DIR]:    $ENV{LLVM_BUILD_DIR}")
message(STATUS "[ENV:LLVM_MLIR_LIB_DIR]: $ENV{LLVM_MLIR_LIB_DIR}")

# credits: chatgpt
macro(find_and_link_llvm_mlir_libraries target liblist)
  foreach(lib ${liblist})
    find_library(${lib}
      NAMES ${lib}
      PATHS $ENV{LLVM_BUILD_DIR}/Release/lib
      NO_DEFAULT_PATH
    )
    if(NOT ${lib})
      message(FATAL_ERROR "Library ${lib} not found in $ENV{LLVM_BUILD_DIR}/Release/lib")
    endif()
    target_link_libraries(${target} PRIVATE ${${lib}})
  endforeach()
endmacro()

find_and_link_llvm_mlir_libraries(clobber_mlir_backend
  MLIRIR
  MLIRFuncDialect
  MLIRTosaDialect
  MLIRSupport
  MLIRParser
  MLIRPass
  MLIRTransforms

  LLVMCore
  LLVMSupport
  LLVMTarget
)

target_include_directories(clobber_mlir_backend
  PUBLIC  
    ${CMAKE_CURRENT_SOURCE_DIR}/include

    $ENV{LLVM_SRC_DIR}/llvm/include  # llvm source headers
    $ENV{LLVM_SRC_DIR}/mlir/include  # mlir source headers

    $ENV{LLVM_BUILD_DIR}/include  # llvm/mlir generated headers
    $ENV{LLVM_BUILD_DIR}/tools/mlir/include

  PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

# configure tests
enable_testing()
add_subdirectory(unittest)