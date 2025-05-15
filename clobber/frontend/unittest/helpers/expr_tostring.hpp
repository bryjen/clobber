// header containing functionality for pretty-printing AST nodes in clobber.
// not in main library to reduce compilation times (?); maybe measure if theres a noticeable performance difference.

#pragma once

#include "pch.hpp"

// clobber/ast.hpp
namespace clobber {
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
}; // namespace clobber

namespace ExprToString {
    std::string expr_base(const std::string &source_text, const clobber::Expr &expr);
    std::string num_lit_expr(const std::string &source_text, const clobber::NumLiteralExpr &expr);
    std::string str_lit_expr(const std::string &source_text, const clobber::StringLiteralExpr &expr);
    std::string char_lit_expr(const std::string &source_text, const clobber::CharLiteralExpr &expr);
    std::string call_expr(const std::string &source_text, const clobber::CallExpr &expr);
    std::string iden_expr(const std::string &source_text, const clobber::IdentifierExpr &expr);
    std::string binding_vector_expr(const std::string &source_text, const clobber::BindingVectorExpr &expr);
    std::string let_expr(const std::string &source_text, const clobber::LetExpr &expr);
    std::string parameter_vector_expr(const std::string &source_text, const clobber::ParameterVectorExpr &expr);
    std::string fn_expr(const std::string &source_text, const clobber::FnExpr &expr);
    std::string def_expr(const std::string &source_text, const clobber::DefExpr &expr);
    std::string do_expr(const std::string &source_text, const clobber::DoExpr &expr);
    std::string accel_expr(const std::string &source_text, const clobber::accel::AccelExpr &expr);
    std::string mat_mul_expr(const std::string &source_text, const clobber::accel::MatMulExpr &expr);
    std::string relu_expr(const std::string &source_text, const clobber::accel::RelUExpr &expr);
}; // namespace ExprToString

std::string tree_visualization(const std::string &source_text, const clobber::Expr &expr);

namespace expr2str = ExprToString; // provide flatcase alias for usage consistency in the consumer