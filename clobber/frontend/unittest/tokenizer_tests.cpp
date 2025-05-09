#include <filesystem>
#include <spdlog/spdlog.h>

#include "helpers/helpers.hpp"
#include "helpers/syntax_factory.hpp"

#include <clobber/ast.hpp>
#include <clobber/parser.hpp>

using path = std::filesystem::path;
using namespace SyntaxFactory;
using namespace TokenizerTestsHelpers;

// clang-format off
std::vector<std::vector<ClobberToken>> expected_cases = {
    { 
        OpenParen(),
        Plus(),
        NumericLiteral(1),
        NumericLiteral(2),
        CloseParen(),
        OpenParen(),
        Asterisk(),
        NumericLiteral(3),
        NumericLiteral(4),
        CloseParen(),
        Eof()
    },
    { 
        OpenParen(),
        Identifier("let"),
        OpenBracket(),
        Identifier("x"),
        NumericLiteral(10),
        Identifier("y"),
        NumericLiteral(5),
        CloseBracket(),
        OpenParen(),
        Plus(),
        Identifier("x"),
        Identifier("y"),
        CloseParen(),
        CloseParen(),
        Eof()
    },
    {
        OpenParen(),
        Identifier("fn"),
        OpenBracket(),
        Identifier("x"),
        CloseBracket(),
        OpenParen(),
        Asterisk(),
        Identifier("x"),
        Identifier("x"),
        CloseParen(),
        CloseParen(),

        OpenParen(),
        OpenParen(),
        Identifier("fn"),
        OpenBracket(),
        Identifier("x"),
        CloseBracket(),
        OpenParen(),
        Asterisk(),
        Identifier("x"),
        Identifier("x"),
        CloseParen(),
        CloseParen(),
        NumericLiteral(5),
        CloseParen(),
        Eof()
    },
    {
        OpenParen(),
        Identifier("def"),
        Identifier("main"),
        OpenBracket(),
        Identifier("x"),
        Identifier("y"),
        CloseBracket(),

        OpenParen(),
        Identifier("let"),
        Asterisk(),  // TODO: see if we separate the two or just have them as one 
        OpenParen(),

        OpenParen(),
        Identifier("z"),
        OpenParen(),
        Plus(),
        Identifier("x"),
        Identifier("y"),
        CloseParen(),
        CloseParen(),

        OpenParen(),
        Identifier("z2"),
        OpenParen(),
        Identifier("relu"),
        Identifier("z"),
        CloseParen(),
        CloseParen(),

        OpenParen(),
        Identifier("out"),
        OpenParen(),
        Identifier("matmul"),
        Identifier("z2"),
        Identifier("weights"),
        CloseParen(),
        CloseParen(),
        CloseParen(),

        Identifier("out"),
        CloseParen(),
        CloseParen(),
        Eof()
    }
};
// clang-format on

class TokenizerTests : public ::testing::TestWithParam<int> {
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

#ifdef CLOBBER_TESTS_DISABLE_TOKENIZER_TESTS
TEST_P(TokenizerTests, DISABLED_IsEven) {
#else
TEST_P(TokenizerTests, IsEven) {
#endif
#ifdef CRT_ENABLED
    INIT_CRT_DEBUG();
    ::testing::GTEST_FLAG(output) = "none";
#endif
    int test_case_idx;
    std::string file_path;
    std::string source_text;
    std::vector<ClobberToken> actual_tokens;
    std::vector<ClobberToken> expected_tokens;

    test_case_idx   = GetParam();
    file_path       = std::format("./test_files/{}.clj", test_case_idx);
    source_text     = read_all_text(file_path);
    expected_tokens = expected_cases[test_case_idx];

    actual_tokens = clobber::tokenize(source_text);

    print_tokens(source_text, expected_tokens, actual_tokens);

    ASSERT_TRUE(are_num_tokens_equal(expected_tokens, actual_tokens));
    EXPECT_TRUE(are_tokens_equal(expected_tokens, actual_tokens));
    EXPECT_TRUE(is_roundtrippable(source_text, actual_tokens));

#ifdef CRT_ENABLED
    if (_CrtDumpMemoryLeaks()) {
        spdlog::warn("^ Okay (empty if alright)\nv Memory leaks (not aight)\n");
    }
#endif
}

// INSTANTIATE_TEST_SUITE_P(EvenValues, TokenizerTests, ::testing::Values(0));
INSTANTIATE_TEST_SUITE_P(EvenValues, TokenizerTests, ::testing::Values(0, 1, 2, 3));