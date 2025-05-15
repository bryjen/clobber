#pragma once

#include <clobber/common/debug.hpp> // common debug header

#include "pch.hpp"

namespace clobber { // clobber/ast.hpp
    struct Token;
    struct CompilationUnit;
} // namespace clobber

namespace clobber {
    /* @brief
     */
    std::vector<clobber::Token> tokenize(const std::string &);

    /* @brief
     */
    std::unique_ptr<clobber::CompilationUnit> parse(const std::string &source_text, const std::vector<clobber::Token> &tokens,
                                                    std::vector<clobber::Diagnostic> &diagnostics);
}; // namespace clobber
