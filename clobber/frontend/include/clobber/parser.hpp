#pragma once

#include <clobber/common/debug.hpp> // common debug header

#include "pch.hpp"

struct ClobberToken;    // ast.hpp
struct CompilationUnit; // ast.hpp

struct ParserError {
public:
    ParserError();
    ParserError(size_t span_start, size_t span_len, const std::string &general_err_msg, const std::string &err_msg);
    ~ParserError();

    std::string GetFormattedErrorMsg(const std::string &file, const std::string &source_text);

protected:
    size_t span_start;
    size_t span_len;
    std::string general_err_msg;
    std::string err_msg;
};

namespace clobber {
/* @brief
 */
std::vector<ClobberToken> tokenize(const std::string &);

/* @brief
 */
std::unique_ptr<CompilationUnit> parse(const std::string &source_text, const std::vector<ClobberToken> &tokens);
}; // namespace clobber
