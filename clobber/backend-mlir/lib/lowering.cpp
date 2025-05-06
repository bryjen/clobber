#pragma warning(push)
#pragma warning(disable : 4267 4244 4996)
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>

#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>

#pragma warning(pop)

#include "clobber/ast.hpp"
#include "clobber/mlir-backend/emit_error.hpp"
#include "clobber/mlir-backend/lowering.hpp"

mlir::ModuleOp
Lowering::lower_to_spirv(mlir::MLIRContext &context, const mlir::ModuleOp &tosa_module,
                         std::vector<EmitError> &emit_errors) {
    throw 0;
}

mlir::ModuleOp
Lowering::lower_to_llvm(mlir::MLIRContext &context, const mlir::ModuleOp &tosa_module,
                        std::vector<EmitError> &emit_errors) {
    throw 0;
}