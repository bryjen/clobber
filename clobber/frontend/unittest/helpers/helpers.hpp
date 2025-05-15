#pragma once

#include "pch.hpp"

namespace clobber {
    struct Diagnostic;      // clobber/common/diagnostic.hpp
    struct Token;           // clobber/ast.hpp
    struct CompilationUnit; // clobber/ast.hpp
    struct SemanticModel;   // clobber/semantics.hpp
}; // namespace clobber

std::string get_executable_directory();
std::string to_string_any(const std::any &a); // helper function for converting a `std::any` to a string
std::string read_all_text(const std::string &);
std::string clobber_token_tostring(const clobber::Token &token, bool use_alignment = false);
std::string reconstruct_source_text_from_tokens(const std::string &, const std::vector<clobber::Token> &);

void print_tokens(const std::string &, const std::vector<clobber::Token> &, const std::vector<clobber::Token> &);

namespace Logging {
    void init_logger(const std::string &logger_name, const std::string &out_log_path);
    void dispose_logger(const std::string &logger_name);
}; // namespace Logging

namespace TokenizerTestsHelpers {
    ::testing::AssertionResult are_num_tokens_equal(const std::vector<clobber::Token> &, const std::vector<clobber::Token> &);
    ::testing::AssertionResult are_tokens_equal(const std::vector<clobber::Token> &, const std::vector<clobber::Token> &);
    ::testing::AssertionResult is_roundtrippable(const std::string &, const std::vector<clobber::Token> &);
}; // namespace TokenizerTestsHelpers

namespace ParserTestsHelpers {
    /* @brief Converts a list of parse errors equivalent to a list of pretty formatted error messages. */
    std::vector<std::string> get_error_msgs(const std::string &, const std::string &, const std::vector<clobber::Diagnostic> &);

    ::testing::AssertionResult are_compilation_units_equivalent(const clobber::CompilationUnit &, const clobber::CompilationUnit &);
} // namespace ParserTestsHelpers

namespace SemanticTestsHelpers {
    std::vector<std::string> get_expr_inferred_type_strs(const clobber::SemanticModel &);
} // namespace SemanticTestsHelpers