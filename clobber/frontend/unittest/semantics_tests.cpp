
#include <filesystem>

#include <spdlog/spdlog.h>

#include "helpers/expr_tostring.hpp"
#include "helpers/helpers.hpp"
#include "helpers/syntax_factory.hpp"

#include <clobber/common/utils.hpp>

#include <clobber/ast.hpp>
#include <clobber/parser.hpp>
#include <clobber/semantics.hpp>

using namespace ParserTestsHelpers;

class SemanticsTests : public ::testing::TestWithParam<int> {
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

#ifdef CLOBBER_TESTS_DISABLE_SEMANTICS_TESTS
TEST_P(SemanticsTests, DISABLED_SemanticsTests) {
#else
TEST_P(SemanticsTests, SemanticsTests) {
#endif
#ifdef CRT_ENABLED
    INIT_CRT_DEBUG();
    ::testing::GTEST_FLAG(output) = "none";
#endif
    int test_case_idx;
    std::string file_path;
    std::string source_text;
    std::vector<ClobberToken> tokens;

    std::unique_ptr<CompilationUnit> compilation_unit;
    std::unique_ptr<SemanticModel> semantic_model;

    test_case_idx = GetParam();
    file_path     = std::format("./test_files/{}.clj", test_case_idx);
    source_text   = read_all_text(file_path);

    spdlog::info(std::format("source:\n```\n{}\n```", source_text));

    tokens           = clobber::tokenize(source_text);
    compilation_unit = clobber::parse(source_text, tokens);
    semantic_model   = clobber::get_semantic_model(source_text, std::move(compilation_unit));

#ifdef CRT_ENABLED
    if (_CrtDumpMemoryLeaks()) {
        spdlog::warn("^ Okay (empty if alright)\nv Memory leaks (not aight)\n");
    }
#endif

    EXPECT_TRUE(true);
}

// INSTANTIATE_TEST_SUITE_P(EvenValues, ParserTests, ::testing::Values(0, 1, 2));
// INSTANTIATE_TEST_SUITE_P(EvenValues, ParserTests, ::testing::Values(4));
INSTANTIATE_TEST_SUITE_P(SemanticsTests, SemanticsTests, ::testing::Values(0));