#include <filesystem>

#include <spdlog/spdlog.h>

#include "helpers/expr_tostring.hpp"
#include "helpers/helpers.hpp"
#include "helpers/syntax_factory.hpp"

#include <clobber/ast.hpp>
#include <clobber/common/utils.hpp>
#include <clobber/parser.hpp>

using namespace ParserTestsHelpers;

class ParserTests : public ::testing::TestWithParam<int> {
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
    int test_case_idx;
    std::string file_path;
    std::string source_text;
    std::vector<ClobberToken> tokens;

    std::unique_ptr<CompilationUnit> cu;

    test_case_idx = GetParam();
    file_path     = std::format("./test_files/{}.clj", test_case_idx);
    source_text   = read_all_text(file_path);

    spdlog::info(std::format("source:\n```\n{}\n```", source_text));

    tokens = clobber::tokenize(source_text);
    cu     = clobber::parse(source_text, tokens);

    if (cu->parse_errors.size() > 0) {
        std::string file                  = "C:/USER/Documents/clobber_proj/main.clj";
        std::vector<std::string> err_msgs = get_error_msgs(file, source_text, cu->parse_errors);
        for (size_t i = 0; i < err_msgs.size(); i++) {
            std::cout << err_msgs[i] << "\n";
        }

        std::cout << std::endl;
        EXPECT_TRUE(false);
        return;
    }

    std::vector<std::string> expr_strs;
    std::vector<std::reference_wrapper<const ExprBase>> expr_views = ptr_utils::get_expr_views(cu->exprs);
    for (const auto &expr_base : expr_views) {
        expr_strs.push_back(expr2str::expr_base(source_text, expr_base.get()));
    }

    spdlog::info("");
    spdlog::info(std::format("reconstructed:\n```\n{}\n```", str_utils::join("", expr_strs)));

    for (const auto &expr_base : expr_views) {
        spdlog::info(expr_base.get().id);
    }

#ifdef CRT_ENABLED
    if (_CrtDumpMemoryLeaks()) {
        spdlog::warn("^ Okay (empty if alright)\nv Memory leaks (not aight)\n");
    }
#endif

    EXPECT_TRUE(true);
}

// INSTANTIATE_TEST_SUITE_P(EvenValues, ParserTests, ::testing::Values(0, 1, 2));
// INSTANTIATE_TEST_SUITE_P(EvenValues, ParserTests, ::testing::Values(4));
INSTANTIATE_TEST_SUITE_P(EvenValues, ParserTests, ::testing::Values(0));