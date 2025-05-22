// header containing functionality for pretty-printing AST nodes in clobber.
// not in main library to reduce compilation times (?); maybe measure if theres a noticeable performance difference.

#pragma once

#include "pch.hpp"

namespace clobber {
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

    // clobber/semantics.hpp
    struct Type;
}; // namespace clobber

/* @brief Gets the text encompassed by the passed expr as represented in the source text. */
std::string expr_tostring(const std::string &source_text, const clobber::Expr &expr);

/* @brief Gets the string representation of a clobber type. */
std::string type_tostring(const clobber::Type &);

/* @brief Visualizes an AST starting from the passed expr. */
std::string expr_visualize_tree(const std::string &source_text, const clobber::Expr &expr);