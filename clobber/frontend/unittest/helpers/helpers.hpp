#pragma once

#include <any>
#include <string>
#include <vector>

#include <gtest/gtest.h>

struct ClobberToken;    // clobber/ast.hpp
struct CompilationUnit; // clobber/ast.hpp
struct ParserError;     // clobber/parser.hpp
struct SemanticModel;   // clobber/semantics.hpp

/* @brief
 * @remark to remove?
 */
template <typename T> bool assert_vectors_same_size(const std::vector<T> &, const std::vector<T> &, std::string *);

std::string get_executable_directory();
std::string to_string_any(const std::any &a); // helper function for converting a `std::any` to a string
std::string read_all_text(const std::string &);
std::string clobber_token_tostring(const ClobberToken &token, bool use_alignment = false);
std::string reconstruct_source_text_from_tokens(const std::string &, const std::vector<ClobberToken> &);

void print_tokens(const std::string &, const std::vector<ClobberToken> &, const std::vector<ClobberToken> &, bool use_alignment = false);

namespace Logging {
void init_logger(const std::string &logger_name, const std::string &out_log_path);
void dispose_logger(const std::string &logger_name);
}; // namespace Logging

namespace TokenizerTestsHelpers {
::testing::AssertionResult are_num_tokens_equal(const std::vector<ClobberToken> &, const std::vector<ClobberToken> &);
::testing::AssertionResult are_tokens_equal(const std::vector<ClobberToken> &, const std::vector<ClobberToken> &);
::testing::AssertionResult is_roundtrippable(const std::string &, const std::vector<ClobberToken> &);
}; // namespace TokenizerTestsHelpers

namespace ParserTestsHelpers {
/* @brief Converts a list of parse errors equivalent to a list of pretty formatted error messages. */
std::vector<std::string> get_error_msgs(const std::string &, const std::string &, const std::vector<ParserError> &);

::testing::AssertionResult are_compilation_units_equivalent(const CompilationUnit &, const CompilationUnit &);
} // namespace ParserTestsHelpers

namespace SemanticTestsHelpers {
std::vector<std::string> get_expr_inferred_type_strs(const SemanticModel &);
} // namespace SemanticTestsHelpers