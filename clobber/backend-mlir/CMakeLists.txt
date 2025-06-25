project(clobber_mlir_backend C CXX)

add_library(clobber_mlir_backend
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/emitter.cpp
)

target_precompile_headers(clobber_common PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include/clobber/mlir-backend/pch.hpp"
)



# load and link all .lib files
message(STATUS "[CMAKE_PREFIX_PATH]:  ${CMAKE_PREFIX_PATH}")

find_package(MLIR REQUIRED CONFIG)
find_package(magic_enum CONFIG REQUIRED)

target_link_libraries(clobber_mlir_backend
  PRIVATE
    MLIRSupport
    MLIRIR
    MLIRParser
    MLIRAnalysis
    MLIRTransforms
    MLIRPass

    # Core dialects
    MLIRAffineDialect
    MLIRFuncDialect
    MLIRSCFDialect
    MLIRMemRefDialect
    MLIRTensorDialect
    MLIRDLTIDialect

    # Optional bytecode + execution
    MLIRBytecodeReader
    MLIRBytecodeWriter
    MLIRExecutionEngine

    # Extra dialects
    MLIRTosaDialect
    MLIRSPIRVDialect
    MLIRSPIRVTransforms

    # LLVM lowering
    MLIRLLVMDialect
    MLIRLLVMIRTransforms
    MLIRArithToLLVM
    MLIRFuncToLLVM
    MLIRMemRefToLLVM
    MLIRIndexToLLVM
    MLIRControlFlowToLLVM
    MLIRLLVMCommonConversion

    # Translation/export
    # MLIRTargetLLVMIR  // should work but not working


    clobber
    clobber_common

    magic_enum::magic_enum
)

target_include_directories(clobber_mlir_backend
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include

  PRIVATE
    ${MLIR_INCLUDE_DIRS}
)

target_compile_definitions(clobber_mlir_backend
  PRIVATE
    ${MLIR_DEFINITIONS}
)

# configure tests
enable_testing()
add_subdirectory(unittest)
