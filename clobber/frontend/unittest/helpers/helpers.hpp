#ifndef HELPERS_HPP
#define HELPERS_HPP

#include <any>
#include <string>
#include <vector>

#include <gtest/gtest.h>

struct ClobberToken; // clobber/ast.hpp
struct ParserError;  // clobber/parser.hpp

/* @brief
 * @remark to remove?
 */
template <typename T> bool assert_vectors_same_size(const std::vector<T> &, const std::vector<T> &, std::string *);

/* @brief
 * @remark to remove?
 */
std::vector<std::string> get_error_msgs(const std::string &, const std::string &, const std::vector<ParserError> &);

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

namespace TokenizerAssertions {
::testing::AssertionResult are_num_tokens_equal(const std::vector<ClobberToken> &, const std::vector<ClobberToken> &);
::testing::AssertionResult are_tokens_equal(const std::vector<ClobberToken> &, const std::vector<ClobberToken> &);
::testing::AssertionResult is_roundtrippable(const std::string &, const std::vector<ClobberToken> &);
}; // namespace TokenizerAssertions

#endif // HELPERS_HPP