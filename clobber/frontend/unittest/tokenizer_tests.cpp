#include "pch.hpp"
#include "test_cases.hpp"

#include "helpers/helpers.hpp"
#include "helpers/syntax_factory.hpp"

#include <clobber/ast/ast.hpp>
#include <clobber/parser.hpp>

using path = std::filesystem::path;
using namespace SyntaxFactory;
using namespace TokenizerTestsHelpers;

class TokenizerTests : public ::testing::TestWithParam<size_t> {
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

#ifndef ENABLE_TOKENIZER_TESTS
TEST_P(TokenizerTests, DISABLED_tokenizer_tests) {
#else
TEST_P(TokenizerTests, tokenizer_tests) {
#endif
#ifdef CRT_ENABLED
    INIT_CRT_DEBUG();
    ::testing::GTEST_FLAG(output) = "none";
#endif
    std::vector<clobber::Token> actual_tokens;
    std::vector<clobber::Token> expected_tokens;

    size_t test_case_idx          = GetParam();
    const std::string source_text = test_cases::tokenizer::sources[test_case_idx];
    expected_tokens               = test_cases::tokenizer::expected_tokens[test_case_idx];

    actual_tokens = clobber::tokenize(source_text);

    print_tokens(source_text, expected_tokens, actual_tokens);

    ASSERT_TRUE(are_num_tokens_equal(expected_tokens, actual_tokens));
    EXPECT_TRUE(are_tokens_vec_equal(source_text, expected_tokens, actual_tokens));
    EXPECT_TRUE(is_roundtrippable(source_text, actual_tokens));

#ifdef CRT_ENABLED
    if (_CrtDumpMemoryLeaks()) {
        logging::warn("^ Okay (empty if alright)\nv Memory leaks (not aight)\n");
    }
#endif
}

// INSTANTIATE_TEST_SUITE_P(tokenizer_tests, TokenizerTests, ::testing::Values(0, 1, 2, 3));
INSTANTIATE_TEST_SUITE_P(tokenizer_tests, TokenizerTests, ::testing::Values(4, 5, 6, 7, 8, 9, 10));