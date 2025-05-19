#include "pch.hpp"

#include "helpers/helpers.hpp"
#include "helpers/syntax_factory.hpp"
#include "helpers/tostring.hpp"
#include "test_cases.hpp"

#include <clobber/common/diagnostic.hpp>
#include <clobber/common/utils.hpp>

#include <clobber/ast.hpp>
#include <clobber/parser.hpp>

using namespace ParserTestsHelpers;

namespace {
    std::vector<clobber::Expr *>
    get_raw_ptrs(std::vector<std::shared_ptr<clobber::Expr>> sptrs) {
        std::vector<clobber::Expr *> exprs;
        for (const auto &sptr : sptrs) {
            exprs.push_back(sptr.get());
        }
        return exprs;
    }

    void
    print_tree_vis(const std::string &source_text, std::vector<clobber::Expr *> exprs) {
        std::vector<std::string> tree_strs;
        for (auto &expr : exprs) {
            tree_strs.push_back(expr_visualize_tree(source_text, *expr));
        }
        spdlog::info(std::format("```\n{}\n```", str_utils::join("\n", tree_strs)));
    }
}; // namespace

class ParserTests : public ::testing::TestWithParam<size_t> {
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

#ifndef ENABLE_PARSER_TESTS
TEST_P(ParserTests, DISABLED_ParserTests) {
#else
TEST_P(ParserTests, ParserTests) {
#endif
#ifdef CRT_ENABLED
    INIT_CRT_DEBUG();
    ::testing::GTEST_FLAG(output) = "none";
#endif

    std::vector<clobber::Diagnostic> diagnostics;

    size_t test_case_idx     = GetParam();
    std::string source_text  = test_cases::parser::sources[test_case_idx];
    auto expected_expr_sptrs = test_cases::parser::expected_exprs[test_case_idx];
    auto expected_exprs      = get_raw_ptrs(expected_expr_sptrs);

    spdlog::info(std::format("source text:\n```\n{}\n```", source_text));
    spdlog::info("\nexpected tree:");
    print_tree_vis(source_text, expected_exprs);

    std::vector<clobber::Token> tokens           = clobber::tokenize(source_text);
    std::unique_ptr<clobber::CompilationUnit> cu = clobber::parse(source_text, tokens, diagnostics);

    if (cu->diagnostics.size() > 0) {
        std::string file                  = "C:/USER/Documents/clobber_proj/main.clj";
        std::vector<std::string> err_msgs = get_error_msgs(file, source_text, cu->diagnostics);
        for (size_t i = 0; i < err_msgs.size(); i++) {
            std::cout << err_msgs[i] << "\n";
        }

        std::cout << std::endl;
        EXPECT_TRUE(false);
        return;
    }

    std::vector<std::string> expr_strs;
    std::vector<std::string> tree_strs;
    for (auto &expr_base : cu->exprs) {
        // expr_strs.push_back(expr_tostring(source_text, *expr_base));
        tree_strs.push_back(expr_visualize_tree(source_text, *expr_base));
    }

    /*
    spdlog::info("");
    spdlog::info(std::format("reconstructed (new):\n```\n{}\n```", str_utils::join("", expr_strs)));
    */

    spdlog::info("");
    spdlog::info(std::format("tree:\n```\n{}\n```", str_utils::join("", tree_strs)));

#ifdef CRT_ENABLED
    if (_CrtDumpMemoryLeaks()) {
        spdlog::warn("^ Okay (empty if alright)\nv Memory leaks (not aight)\n");
    }
#endif

    EXPECT_TRUE(true);
}

INSTANTIATE_TEST_SUITE_P(EvenValues, ParserTests, ::testing::Values(0, 1, 2, 4));
// INSTANTIATE_TEST_SUITE_P(EvenValues, ParserTests, ::testing::Values(1));
// INSTANTIATE_TEST_SUITE_P(EvenValues, ParserTests, ::testing::Values(0));