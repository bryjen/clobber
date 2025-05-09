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

#include <magic_enum/magic_enum.hpp>

#include "helpers.hpp"

#include <clobber/ast.hpp>
#include <clobber/parser.hpp>

template <typename T>
bool
assert_vectors_same_size(const std::vector<T> &actual, const std::vector<T> &expected, std::string *out_err_msg) {
    size_t actual_len   = actual.size();
    size_t expected_len = expected.size();

    if (actual_len == expected_len) {
        return true;
    }

    *out_err_msg = std::format("Mismatched actual & expected lengths, expected {} but got {}.", expected_len, actual_len);
    return false;
}

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
clobber_token_tostring(const ClobberToken &token, bool use_alignment) {
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
             const std::vector<ClobberToken> &actual_tokens, bool use_alignment) {
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

std::string
get_executable_directory() {
    char buffer[MAX_PATH];
    GetModuleFileNameA(nullptr, buffer, MAX_PATH);
    std::filesystem::path exe_path(buffer);
    return exe_path.parent_path().string();
}

void
Logging::init_logger(const std::string &logger_name, const std::string &out_log_path) {
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto file_sink    = std::make_shared<spdlog::sinks::basic_file_sink_mt>(out_log_path, true);

    std::vector<spdlog::sink_ptr> sinks{console_sink, file_sink};
    auto logger = std::make_shared<spdlog::logger>(logger_name, sinks.begin(), sinks.end());
    spdlog::set_default_logger(logger);

    // spdlog::set_pattern("[%H:%M:%S.%e; %^%l%$]: %v");
    spdlog::set_pattern("%v");
}

void
Logging::dispose_logger(const std::string &logger_name) {
    spdlog::get(logger_name)->flush();
    spdlog::drop(logger_name);
}

::testing::AssertionResult
TokenizerTestsHelpers::are_num_tokens_equal(const std::vector<ClobberToken> &expected, const std::vector<ClobberToken> &actual) {
    size_t actual_num_tokens   = actual.size();
    size_t expected_num_tokens = expected.size();
    if (actual_num_tokens == expected_num_tokens) {
        return ::testing::AssertionSuccess();
    } else {
        return ::testing::AssertionFailure() << std::format("Expected {} tokens, but received {}", expected_num_tokens, actual_num_tokens);
    }
}

::testing::AssertionResult
TokenizerTestsHelpers::are_tokens_equal(const std::vector<ClobberToken> &expected_tokens, const std::vector<ClobberToken> &actual_tokens) {
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
TokenizerTestsHelpers::is_roundtrippable(const std::string &source_text, const std::vector<ClobberToken> &actual_tokens) {
    const std::string reconstructed = reconstruct_source_text_from_tokens(source_text, actual_tokens);
    if (source_text == reconstructed) {
        return ::testing::AssertionSuccess();
    } else {
        return ::testing::AssertionFailure() << "Actual tokens don't reconstruct the original text.";
    }
}

::testing::AssertionResult
are_compilation_units_equivalent(const CompilationUnit &, const CompilationUnit &) {
    throw 0;
}

std::vector<std::string>
ParserTestsHelpers::get_error_msgs(const std::string &file, const std::string &source_text, const std::vector<ParserError> &parse_errors) {
    std::vector<std::string> errs;

    size_t i;
    for (i = 0; i < parse_errors.size(); i++) {
        ParserError parse_err = parse_errors[i];
        errs.push_back(parse_err.GetFormattedErrorMsg(file, source_text));
    }

    return errs;
}