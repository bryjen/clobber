#pragma once

#pragma warning(push)
#pragma warning(disable : 4267 4244 4996)
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#pragma warning(pop)

// #include "pch.hpp"
#include "clobber/mlir-backend/pch.hpp"

struct SemanticModel; // clobber/semantics.hpp

struct EmitError {
public:
    EmitError() = default;
    EmitError(int span_start, int span_len, const std::string &general_err_msg, const std::string &err_msg) { throw 0; }
    ~EmitError() = default;

    std::string
    GetFormattedErrorMsg(const std::string &file, const std::string &source_text) {
        throw 0;
    }

protected:
    int span_start;
    int span_len;
    std::string general_err_msg;
    std::string err_msg;
};

namespace clobber {
    mlir::ModuleOp emit(mlir::MLIRContext &context, const SemanticModel &);
}; // namespace clobber