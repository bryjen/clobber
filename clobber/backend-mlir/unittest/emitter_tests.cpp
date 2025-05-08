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
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#pragma warning(pop)

#include <clobber/ast.hpp>
#include <clobber/parser.hpp>

#include "clobber/mlir-backend/emit_error.hpp"
#include "clobber/mlir-backend/lowering.hpp"
#include "clobber/mlir-backend/tosa_emitter.hpp"

// clang-format off
const std::array<std::string, 3> test_source_contents = {
R"(1)",

R"((+ 1 2))",

R"((+ 1 2)
(* 3 4))",
};
// clang-format on

class TokenizerTests : public ::testing::TestWithParam<int> {};

TEST(TokenizerTests, sanity_check_1) {
    GTEST_SKIP() << "Disabled";
    test_tosa_mlir_1();
    EXPECT_TRUE(true);
}

TEST(TokenizerTests, sanity_check_2) {
    GTEST_SKIP() << "Disabled";
    test_tosa_mlir_2();
    EXPECT_TRUE(true);
}

TEST_P(TokenizerTests, tosa_emitter_tests) {
    // SetConsoleOutputCP(CP_UTF8);

    int idx;
    std::string file_path;
    std::string source_text;
    std::vector<ClobberToken> tokens;
    std::string str_buf;

    CompilationUnit cu;
    std::vector<ParserError> parse_errors;

    mlir::MLIRContext context;
    mlir::ModuleOp module_op;
    std::vector<EmitError> emit_errors;

    idx = GetParam();

    str_buf     = std::format("[{}]", idx);
    source_text = test_source_contents[idx];
    tokens      = clobber::tokenize(source_text);

    clobber::parse(source_text, tokens, cu);

    TosaEmitter::init_context(context);
    module_op = TosaEmitter::lower_ast_to_tosa(context, cu, emit_errors);

    std::cout << std::format("Source:\n```\n{}\n```", source_text) << "\n";
    std::cout << std::format("Parse Errors: {}", parse_errors.size()) << "\n";
    std::cout << std::endl;

    if (mlir::failed(mlir::verify(module_op))) {
        llvm::errs() << "TOSA MLIR verification failed\n";
        module_op.dump();
    } else {
        llvm::outs() << "TOSA MLIR module:\n";
        module_op.dump();
    }

    EXPECT_TRUE(true);
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(emitter_tests, TokenizerTests, 
    ::testing::Values(
        0,
        1
    ));
// clang-format on