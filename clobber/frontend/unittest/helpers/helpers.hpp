#pragma once

#include "pch.hpp"

namespace clobber {
    // clobber/common/diagnostic.hpp
    struct Diagnostic;

    // clobber/ast.hpp
    struct Token;
    struct CompilationUnit;

    // clobber/ast.hpp
    struct Expr;
    struct NumLiteralExpr;
    struct CallExpr;
    struct IdentifierExpr;
    struct BindingVectorExpr;
    struct LetExpr;
    struct ParameterVectorExpr;
    struct FnExpr;
    struct DefExpr;
    struct DoExpr;
    struct StringLiteralExpr;
    struct CharLiteralExpr;

    namespace accel {
        struct AccelExpr;
        struct MatMulExpr;
        struct RelUExpr;
    }; // namespace accel

    struct SemanticModel; // clobber/semantics.hpp
}; // namespace clobber

namespace clobber {}; // namespace clobber

namespace {
    std::string to_string_any(const std::any &a); // helper function for converting a `std::any` to a string
    std::string read_all_text(const std::string &);
    std::string reconstruct_source_text_from_tokens(const std::string &, const std::vector<clobber::Token> &);
} // namespace

namespace Logging {
    void init_logger(const std::string &logger_name, const std::string &out_log_path);
    void dispose_logger(const std::string &logger_name);
}; // namespace Logging

namespace TokenizerTestsHelpers {
    void print_tokens(const std::string &, const std::vector<clobber::Token> &, const std::vector<clobber::Token> &);
    bool are_tokens_equal(const clobber::Token &, const clobber::Token &);

    ::testing::AssertionResult are_num_tokens_equal(const std::vector<clobber::Token> &, const std::vector<clobber::Token> &);
    ::testing::AssertionResult are_tokens_vec_equal(const std::string &, const std::vector<clobber::Token> &,
                                                    const std::vector<clobber::Token> &);
    ::testing::AssertionResult is_roundtrippable(const std::string &, const std::vector<clobber::Token> &);
}; // namespace TokenizerTestsHelpers

namespace ParserTestsHelpers {
    /* @brief Converts a list of parse errors equivalent to a list of pretty formatted error messages. */
    std::vector<std::string> get_error_msgs(const std::string &, const std::string &, const std::vector<clobber::Diagnostic> &);

    ::testing::AssertionResult are_compilation_units_equivalent(const std::string &source_text, std::vector<clobber::Expr *> expected,
                                                                std::vector<clobber::Expr *> actual, bool print);
} // namespace ParserTestsHelpers

namespace SemanticTestsHelpers {
    std::vector<std::string> get_expr_inferred_type_strs(const clobber::SemanticModel &);
} // namespace SemanticTestsHelpers