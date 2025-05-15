#pragma once

#include <clobber/common/debug.hpp> // common debug header

#include "pch.hpp"

namespace clobber { // clobber/ast.hpp
    struct ClobberToken;
    struct CompilationUnit;
} // namespace clobber

namespace clobber {
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

    /* @brief
     */
    std::vector<clobber::ClobberToken> tokenize(const std::string &);

    /* @brief
     */
    std::unique_ptr<clobber::CompilationUnit> parse(const std::string &source_text, const std::vector<clobber::ClobberToken> &tokens);
}; // namespace clobber
