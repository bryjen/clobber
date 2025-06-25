#include <array>
#include <format>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// #define NOMINMAX
// #include <windows.h>

#pragma warning(push)
#pragma warning(disable : 4267 4244 4996)
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>

#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>

#include <mlir/Dialect/LLVMIR/BasicPtxBuilderInterface.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMInterfaces.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#pragma warning(pop)

#include <clobber/common/diagnostic.hpp>

#include <clobber/ast.hpp>
#include <clobber/parser.hpp>
#include <clobber/semantics.hpp>

#include "clobber/mlir-backend/emitter.hpp"

// Simple logging helper
namespace logging {
    inline void
    info(const std::string &message) {
        std::cout << message << std::endl;
    }

    inline void
    warn(const std::string &message) {
        std::cout << "[WARN] " << message << std::endl;
    }

    inline void
    set_pattern(const std::string &pattern) {
        // No-op for std::cout, but kept for compatibility
    }
} // namespace logging

// clang-format off
const std::vector<std::string> test_source_contents = {
R"(1)",

R"((+ 1 2))",

R"((+ 1 2)
(* 3 4))",

R"((+ 1.0f 2.0f))",

R"((+ 1 2 3 4))",

R"(
(let [x 10
      y 5]
  (+ x y))
)",

R"("hello, world!")",
};
// clang-format on

class EmitterTests : public ::testing::TestWithParam<size_t> {};

TEST_P(EmitterTests, EmitterTestsCore) {
#ifdef CRT_ENABLED
    INIT_CRT_DEBUG();
    ::testing::GTEST_FLAG(output) = "none";
#endif
    logging::set_pattern("%v");

    size_t test_case_idx;
    std::string file_path;
    std::string source_text;
    std::vector<clobber::Token> tokens;

    std::unique_ptr<clobber::CompilationUnit> compilation_unit;
    std::unique_ptr<clobber::SemanticModel> semantic_model;
    std::vector<std::string> inferred_type_strs;

    test_case_idx = GetParam();
    file_path     = std::format("./test_files/{}.clj", test_case_idx);
    source_text   = test_source_contents[test_case_idx];

    logging::info(std::format("source:\n```\n{}\n```", source_text));

    std::vector<clobber::Diagnostic> diagnostics;

    mlir::ModuleOp _module;
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::tosa::TosaDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::spirv::SPIRVDialect>();
    context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
    context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();

    tokens           = clobber::tokenize(source_text);
    compilation_unit = clobber::parse(source_text, tokens, diagnostics);
    semantic_model   = clobber::get_semantic_model(std::move(compilation_unit), diagnostics);
    _module          = clobber::emit(context, *semantic_model, diagnostics);

    logging::info(std::format("\nMLIR:\n```"));
    _module->dump();
    logging::info(std::format("```"));

    try {
        if (mlir::failed(mlir::verify(_module))) {
            llvm::errs() << "MLIR module verification FAILED\n";
        } else {
            llvm::outs() << "MLIR module OK:\n";
        }
    } catch (...) {
        llvm::errs() << "MLIR module verification FAILED (WITH ERRORS)\n";
    }

#ifdef CRT_ENABLED
    if (_CrtDumpMemoryLeaks()) {
        logging::warn("^ Okay (empty if alright)\nv Memory leaks (not aight)\n");
    }
#endif
}

TEST(SanityChecks, llvm_sanity_check_1) {
    mlir::MLIRContext context;
    context.loadDialect<mlir::func::FuncDialect, mlir::arith::ArithDialect>();

    mlir::OpBuilder builder(&context);
    mlir::Location loc = builder.getUnknownLoc();

    // Create a new MLIR module
    auto module = mlir::ModuleOp::create(loc);

    // Create function type: () -> i32
    auto i32    = builder.getI32Type();
    auto fnType = builder.getFunctionType({}, i32);

    // Create @main function
    auto func          = builder.create<mlir::func::FuncOp>(loc, "main", fnType);
    mlir::Block *entry = func.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    // Emit constants
    auto c1 = builder.create<mlir::arith::ConstantOp>(loc, builder.getI32IntegerAttr(42));
    auto c2 = builder.create<mlir::arith::ConstantOp>(loc, builder.getI32IntegerAttr(58));

    // Add them
    // auto sum = builder.create<mlir::arith::AddIOp>(loc, c1, c2);
    auto sum = builder.create<mlir::arith::AddIOp>(loc, c1, c2).getResult();
    builder.create<mlir::func::ReturnOp>(loc, sum);

    // Add function to module
    module.push_back(func);

    if (mlir::failed(mlir::verify(module))) {
        llvm::errs() << "MLIR verification failed\n";
        module.dump();
    } else {
        llvm::outs() << "MLIR module:\n";
        module.dump();
    }

    EXPECT_TRUE(true);
}

TEST(SanityChecks, llvm_sanity_check_2) {
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::tosa::TosaDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::spirv::SPIRVDialect>();
    context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();

    mlir::OpBuilder builder(&context);
    auto i32 = builder.getI32Type();
    auto loc = builder.getUnknownLoc();

    // Create a new MLIR module
    auto module = mlir::ModuleOp::create(loc);

    auto funcType      = builder.getFunctionType({}, i32);
    auto func          = builder.create<mlir::func::FuncOp>(loc, "main", funcType);
    mlir::Region &body = func.getBody();

    // Create all blocks up front
    mlir::Block *entry = builder.createBlock(&body);
    mlir::Block *sub   = builder.createBlock(&body);
    mlir::Block *merge = builder.createBlock(&body);
    merge->addArgument(i32, loc);

    // Emit into entry block
    builder.setInsertionPointToStart(entry);
    auto a = builder.create<mlir::arith::ConstantOp>(loc, builder.getI32IntegerAttr(10));
    builder.create<mlir::cf::BranchOp>(loc, sub); // ✅ now definitely in entry block

    // Emit into sub block
    builder.setInsertionPointToStart(sub);
    auto x = builder.create<mlir::arith::AddIOp>(loc, a, a);
    builder.create<mlir::cf::BranchOp>(loc, merge, mlir::ValueRange{x.getResult()});

    // Emit into merge block
    builder.setInsertionPointToStart(merge);
    auto z = merge->getArgument(0);
    auto y = builder.create<mlir::arith::AddIOp>(loc, z, a);
    builder.create<mlir::func::ReturnOp>(loc, y.getResult());

    // Add function to module
    module.push_back(func);

    if (mlir::failed(mlir::verify(module))) {
        llvm::errs() << "MLIR verification failed\n";
        module.dump();
    } else {
        llvm::outs() << "MLIR module:\n";
        module.dump();
    }

    EXPECT_TRUE(true);
}

// INSTANTIATE_TEST_SUITE_P(EmitterTestsCore, EmitterTests, ::testing::Values(0, 1, 2, 3));
// INSTANTIATE_TEST_SUITE_P(EmitterTestsCore, EmitterTests, ::testing::Values(3));
// INSTANTIATE_TEST_SUITE_P(EmitterTestsCore, EmitterTests, ::testing::Values(5));
INSTANTIATE_TEST_SUITE_P(EmitterTestsCore, EmitterTests, ::testing::Values(6));