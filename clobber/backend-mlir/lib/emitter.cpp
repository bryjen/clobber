#pragma warning(push)
#pragma warning(disable : 4267 4244 4996)
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#pragma warning(pop)

#include "emitter.hpp"

void
testTosaMLIR() {
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::tosa::TosaDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();

    mlir::OpBuilder builder(&context);
    auto module = mlir::ModuleOp::create(builder.getUnknownLoc());

    // tensor<i32> scalar type
    auto type     = mlir::RankedTensorType::get({}, builder.getIntegerType(32));
    auto funcType = builder.getFunctionType({type, type}, {});

    auto func   = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), "main", funcType);
    auto &entry = *func.addEntryBlock();
    builder.setInsertionPointToStart(&entry);

    auto sum =
        builder.create<mlir::tosa::AddOp>(builder.getUnknownLoc(), type, entry.getArgument(0), entry.getArgument(1));

    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());

    module.push_back(func);

    if (mlir::failed(mlir::verify(module))) {
        llvm::errs() << "TOSA MLIR verification failed\n";
    } else {
        llvm::outs() << "TOSA MLIR module:\n";
        module.dump();
    }
}