#include <array>
#include <filesystem>
#include <fmt/core.h>
#include <format>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <gtest/gtest.h>
#include <magic_enum/magic_enum.hpp>

#include <clobber/common/debug.hpp> // common debug header

#include "helpers/syntax_factory.hpp"
#include <clobber/ast.hpp>
#include <clobber/parser.hpp>

using path = std::filesystem::path;
using namespace SyntaxFactory;

std::string
get_executable_directory() {
    char buffer[MAX_PATH];
    GetModuleFileNameA(nullptr, buffer, MAX_PATH);
    std::filesystem::path exe_path(buffer);
    return exe_path.parent_path().string();
}

void
init_logger(const std::string &logger_name, const std::string &out_log_path) {
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto file_sink    = std::make_shared<spdlog::sinks::basic_file_sink_mt>(out_log_path, true);

    std::vector<spdlog::sink_ptr> sinks{console_sink, file_sink};
    auto logger = std::make_shared<spdlog::logger>(logger_name, sinks.begin(), sinks.end());
    spdlog::set_default_logger(logger);

    // spdlog::set_pattern("[%H:%M:%S.%e; %^%l%$]: %v");
    spdlog::set_pattern("%v");
}

void
dispose_logger(const std::string &logger_name) {
    spdlog::get(logger_name)->flush();
    spdlog::drop(logger_name);
}

// clang-format off
std::vector<std::vector<ClobberToken>> expected_cases = {
    { 
        OpenParen(),
        Plus(),
        NumericLiteral(1),
        NumericLiteral(2),
        CloseParen(),
        OpenParen(),
        Plus(),
        NumericLiteral(3),
        NumericLiteral(4),
        CloseParen()
    },
    { 
        OpenParen(),
        Plus(),
        NumericLiteral(1),
        NumericLiteral(2),
        CloseParen(),
        OpenParen(),
        Plus(),
        NumericLiteral(3),
        NumericLiteral(4),
        CloseParen()
    },
    { 
        OpenParen(),
        Plus(),
        NumericLiteral(1),
        NumericLiteral(2),
        CloseParen(),
        OpenParen(),
        Plus(),
        NumericLiteral(3),
        NumericLiteral(4),
        CloseParen()
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
        init_logger(logger_name, output_file_path);
    }

    void
    TearDown() override {
        const std::string logger_name = std::format("logger_{}", GetParam());
        dispose_logger(logger_name);
    }
};

std::string
read_all_text(const std::string &path) {
    std::ifstream file(path, std::ios::in | std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(file)), {});
}

std::string
to_string_any(const std::any &a) {
    if (a.type() == typeid(int))
        return std::to_string(std::any_cast<int>(a));
    if (a.type() == typeid(std::string))
        return std::any_cast<std::string>(a);
    if (a.type() == typeid(bool))
        return std::any_cast<bool>(a) ? "true" : "false";
    // add more types as needed
    return "<unsupported>";
}

std::string
clobber_token_tostring(const ClobberToken &token, bool use_alignment = false) {
    std::string value_str      = to_string_any(token.value);
    std::string token_type_str = std::string(magic_enum::enum_name(token.token_type));
    if (use_alignment) { // cannot reduce to conditional due to `std::format` constexpr constraint
        return std::format("(tt: {:>20.20} (val: `{}`)", token_type_str, value_str);
    } else {
        return std::format("(tt: {} (val: `{}`)", token_type_str, value_str);
    }
}

std::string
reconstruct_source_text_from_tokens(const std::string &source_text, const std::vector<ClobberToken> &tokens) {
    std::ostringstream builder;
    for (size_t i = 0; i < tokens.size(); i++) {
        ClobberToken token = tokens[i];
        builder << token.ExtractFullText(source_text);
    }

    return builder.str();
}

void
print_tokens(const std::string &source_text, const std::vector<ClobberToken> &expected_tokens,
             const std::vector<ClobberToken> &actual_tokens, bool use_alignment = false) {
#ifndef CRT_ENABLED
    spdlog::info(std::format("[Expected; n={}]", expected_tokens.size()));
    spdlog::info("---------------------------------------------------------");
    for (size_t i = 0; i < expected_tokens.size(); i++) {
        ClobberToken token = expected_tokens[i];
        spdlog::info(std::format("[{:>2}] {}", i, clobber_token_tostring(token, true)));
    }
    spdlog::info(std::format("Source text:\n```\n{}\n```\n", source_text));

    spdlog::info("");
    spdlog::info(std::format("[Actual; n={}]", actual_tokens.size()));
    spdlog::info("---------------------------------------------------------");
    for (size_t i = 0; i < actual_tokens.size(); i++) {
        ClobberToken token = actual_tokens[i];
        spdlog::info(std::format("[{:>2}] {}", i, clobber_token_tostring(token, true)));
    }
    const std::string reconstructed = reconstruct_source_text_from_tokens(source_text, actual_tokens);
    spdlog::info(std::format("Reconstructed text:\n```\n{}\n```\n", reconstructed));
#endif
}

::testing::AssertionResult
are_num_tokens_equal(const std::vector<ClobberToken> &expected, const std::vector<ClobberToken> &actual) {
    size_t actual_num_tokens   = actual.size();
    size_t expected_num_tokens = expected.size();
    if (actual_num_tokens == expected_num_tokens) {
        return ::testing::AssertionSuccess();
    } else {
        return ::testing::AssertionFailure() << std::format("Expected {} tokens, but received {}", expected_num_tokens, actual_num_tokens);
    }
}

::testing::AssertionResult
are_tokens_equal(const std::vector<ClobberToken> &expected_tokens, const std::vector<ClobberToken> &actual_tokens) {
    // we're assumed to have equal number of tokens, asserted by "assert_equal_number_tokens"
    size_t num_tokens;

    num_tokens = expected_tokens.size();
    for (size_t i = 0; i < num_tokens; i++) {
        ClobberToken expected;
        ClobberToken actual;

        expected = expected_tokens[i];
        actual   = actual_tokens[i];
        if (!ClobberToken::AreEquivalent(expected, actual)) {
            return ::testing::AssertionFailure() << std::format("Tokens at {} are not equal; expected: {}; actual: {}", i,
                                                                clobber_token_tostring(expected), clobber_token_tostring(actual));
        }
    }

    return ::testing::AssertionSuccess();
}

::testing::AssertionResult
is_roundtrippable(const std::string &source_text, const std::vector<ClobberToken> &actual_tokens) {
    const std::string reconstructed = reconstruct_source_text_from_tokens(source_text, actual_tokens);
    if (source_text == reconstructed) {
        return ::testing::AssertionSuccess();
    } else {
        return ::testing::AssertionFailure() << "Actual tokens don't reconstruct the original text.";
    }
}

TEST_P(TokenizerTests, IsEven) {
#ifdef CRT_ENABLED
    INIT_CRT_DEBUG();
    ::testing::GTEST_FLAG(output) = "none";
#endif
    // int *leak = new int(42);

    int test_case_idx;
    std::string file_path;
    std::string source_text;
    std::vector<ClobberToken> actual_tokens;
    std::vector<ClobberToken> expected_tokens;

    test_case_idx   = GetParam();
    file_path       = std::format("./test_files/{}.clj", test_case_idx);
    source_text     = read_all_text(file_path);
    expected_tokens = expected_cases[0];

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

// INSTANTIATE_TEST_SUITE_P(EvenValues, TokenizerTests, ::testing::Values(0, 1, 2));
INSTANTIATE_TEST_SUITE_P(EvenValues, TokenizerTests, ::testing::Values(0));