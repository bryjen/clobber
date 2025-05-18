#pragma once

#include "helpers/pch.hpp"

namespace clobber {
    struct Token;         // clobber/ast.hpp
    struct Expr;          // clobber/ast.hpp
    struct SemanticModel; // clobber/semantics.hpp
}; // namespace clobber

namespace test_cases {
    namespace tokenizer {
        extern const std::vector<std::string> sources;

        extern const std::vector<std::vector<clobber::Token>> expected_tokens;
    }; // namespace tokenizer

    namespace tokenizer {
        extern const std::vector<std::string> sources;

        extern const std::vector<std::vector<clobber::Expr>> expected_exprs;
    }; // namespace tokenizer
} // namespace test_cases