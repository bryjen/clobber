
#include "pch.hpp"

#include "helpers/helpers.hpp"
#include "helpers/syntax_factory.hpp"
#include "helpers/tostring.hpp"

#include <clobber/common/utils.hpp>

#include <clobber/ast/ast.hpp>
#include <clobber/parser.hpp>
#include <clobber/semantics.hpp>

#include "test_cases.hpp"

using namespace SemanticTestsHelpers;

class SemanticsTests : public ::testing::TestWithParam<size_t> {
protected:
    void
    SetUp() override {
        auto *info             = ::testing::UnitTest::GetInstance()->current_test_info();
        std::string suite_name = info->test_suite_name();
        std::string test_name  = info->name();

        std::replace(suite_name.begin(), suite_name.end(), '/', '_');
        std::replace(test_name.begin(), test_name.end(), '/', '_');

        const std::string output_file_path = std::format("./logs/{}_{}_{}.txt", suite_name, test_name, GetParam());
        const std::string logger_name      = std::format("logger_{}", GetParam());
        Logging::init_logger(logger_name, output_file_path);
    }

    void
    TearDown() override {
        const std::string logger_name = std::format("logger_{}", GetParam());
        Logging::dispose_logger(logger_name);
    }
};

#ifndef ENABLE_SEMANTICS_TESTS
TEST_P(SemanticsTests, DISABLED_SemanticsTests) {
#else
TEST_P(SemanticsTests, SemanticsTests) {
#endif
#ifdef CRT_ENABLED
    INIT_CRT_DEBUG();
    ::testing::GTEST_FLAG(output) = "none";
#endif
    std::vector<clobber::Diagnostic> diagnostics;

    size_t test_case_idx    = GetParam();
    std::string source_text = test_cases::parser::sources[test_case_idx];

    spdlog::info(std::format("src:\n```\n{}\n```", source_text));

    std::vector<clobber::Token> tokens                     = clobber::tokenize(source_text);
    std::unique_ptr<clobber::CompilationUnit> cu           = clobber::parse(source_text, tokens, diagnostics);
    std::unique_ptr<clobber::SemanticModel> semantic_model = clobber::get_semantic_model(std::move(cu), diagnostics);

    std::vector<std::string> inferred_type_strs = get_expr_inferred_type_strs(*semantic_model);
    spdlog::info(str_utils::join("\n", inferred_type_strs));

#ifdef CRT_ENABLED
    if (_CrtDumpMemoryLeaks()) {
        spdlog::warn("^ Okay (empty if alright)\nv Memory leaks (not aight)\n");
    }
#endif

    EXPECT_TRUE(true);
}

// INSTANTIATE_TEST_SUITE_P(SemanticsTests, SemanticsTests, ::testing::Values(0, 1, 2, 3));
INSTANTIATE_TEST_SUITE_P(SemanticsTests, SemanticsTests, ::testing::Values(4));