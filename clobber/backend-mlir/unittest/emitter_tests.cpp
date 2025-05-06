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
//1.
R"((+ 1 2)
(* 3 4))",

//2.
R"(
(let [x 10
      y 5]
  (+ x y))
)",

//3.
R"(
(fn [x] (* x x))
((fn [x] (* x x)) 5)
)"
};
// clang-format on

class EmitterTests : public ::testing::TestWithParam<int> {};

TEST_P(EmitterTests, SanityCheck) {
    testTosaMLIR();
    // mlir_test();
    EXPECT_TRUE(true);
}

TEST_P(EmitterTests, TOSA_Emitter_Tests) {
    // SetConsoleOutputCP(CP_UTF8);

    int idx;
    std::string file_path;
    std::string source_text;
    std::vector<Token> tokens;
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
    cu          = clobber::parse(source_text, tokens, parse_errors);

    TosaEmitter::init_context(context);
    module_op = TosaEmitter::lower_ast_to_tosa(context, cu, emit_errors);

    if (mlir::failed(mlir::verify(module_op))) {
        llvm::errs() << "TOSA MLIR verification failed\n";
        module_op.dump();
    } else {
        llvm::outs() << "TOSA MLIR module:\n";
        module_op.dump();
    }

    EXPECT_TRUE(true);
}

INSTANTIATE_TEST_SUITE_P(TOSA_Emitter_Tests, EmitterTests, ::testing::Values(0));