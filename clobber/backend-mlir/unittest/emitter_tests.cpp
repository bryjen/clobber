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

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#pragma warning(pop)

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <clobber/ast.hpp>
#include <clobber/parser.hpp>
#include <clobber/semantics.hpp>

#include "clobber/mlir-backend/emitter.hpp"

// clang-format off
const std::vector<std::string> test_source_contents = {
R"(1)",

R"((+ 1 2))",

R"((+ 1 2)
(* 3 4))",

R"((+ 1.0f 2.0f))",

R"((+ 1 2 3 4))",
};
// clang-format on

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

class EmitterTests : public ::testing::TestWithParam<size_t> {};

TEST_P(EmitterTests, EmitterTestsCore) {
#ifdef CRT_ENABLED
    INIT_CRT_DEBUG();
    ::testing::GTEST_FLAG(output) = "none";
#endif
    spdlog::set_pattern("%v");

    size_t test_case_idx;
    std::string file_path;
    std::string source_text;
    std::vector<ClobberToken> tokens;

    std::unique_ptr<CompilationUnit> compilation_unit;
    std::unique_ptr<SemanticModel> semantic_model;
    std::vector<std::string> inferred_type_strs;

    test_case_idx = GetParam();
    file_path     = std::format("./test_files/{}.clj", test_case_idx);
    source_text   = test_source_contents[test_case_idx];

    spdlog::info(std::format("source:\n```\n{}\n```", source_text));

    // mlir::OwningOpRef<mlir::ModuleOp> _module;
    mlir::ModuleOp _module;
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::tosa::TosaDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::spirv::SPIRVDialect>();

    tokens           = clobber::tokenize(source_text);
    compilation_unit = clobber::parse(source_text, tokens);
    semantic_model   = clobber::get_semantic_model(std::move(compilation_unit));

    _module = clobber::emit(context, *semantic_model);

    spdlog::info(std::format("\nMLIR:\n```"));
    _module->dump();
    spdlog::info(std::format("```"));

    /*
    try {
        if (mlir::failed(mlir::verify(module))) {
            llvm::errs() << "MLIR module verification FAILED\n";
        } else {
            llvm::outs() << "MLIR module OK:\n";
        }
    } catch (...) {
        llvm::errs() << "MLIR module verification FAILED (WITH ERRORS)\n";
    }
    */

#ifdef CRT_ENABLED
    if (_CrtDumpMemoryLeaks()) {
        spdlog::warn("^ Okay (empty if alright)\nv Memory leaks (not aight)\n");
    }
#endif
}

// INSTANTIATE_TEST_SUITE_P(EmitterTestsCore, EmitterTests, ::testing::Values(0, 1, 2, 3));
INSTANTIATE_TEST_SUITE_P(EmitterTestsCore, EmitterTests, ::testing::Values(3));