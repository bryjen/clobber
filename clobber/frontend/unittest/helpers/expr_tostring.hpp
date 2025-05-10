// header containing functionality for pretty-printing AST nodes in clobber.
// not in main library to reduce compilation times (?); maybe measure if theres a noticeable performance difference.

#ifndef EXPR_TOSTRING_HPP
#define EXPR_TOSTRING_HPP

// clobber/ast.hpp
struct ExprBase;
struct NumLiteralExpr;
struct CallExpr;
struct IdentifierExpr;
struct BindingVectorExpr;
struct LetExpr;
struct ParameterVectorExpr;
struct FnExpr;
struct DefExpr;
struct DoExpr;
struct DoExpr;

struct AccelExpr;
struct MatMulExpr;
struct RelUExpr;

#include <string>

namespace ExprToString {
std::string expr_base(const std::string &source_text, const ExprBase &expr);
std::string num_lit_expr(const std::string &source_text, const NumLiteralExpr &expr);
std::string call_expr(const std::string &source_text, const CallExpr &expr);
std::string iden_expr(const std::string &source_text, const IdentifierExpr &expr);
std::string binding_vector_expr(const std::string &source_text, const BindingVectorExpr &expr);
std::string let_expr(const std::string &source_text, const LetExpr &expr);
std::string parameter_vector_expr(const std::string &source_text, const ParameterVectorExpr &expr);
std::string fn_expr(const std::string &source_text, const FnExpr &expr);
std::string def_expr(const std::string &source_text, const DefExpr &expr);
std::string do_expr(const std::string &source_text, const DoExpr &expr);

std::string accel_expr(const std::string &source_text, const AccelExpr &expr);
std::string mat_mul_expr(const std::string &source_text, const MatMulExpr &expr);
std::string relu_expr(const std::string &source_text, const RelUExpr &expr);
}; // namespace ExprToString

std::string tree_visualization(const std::string &source_text, const ExprBase &expr);

namespace expr2str = ExprToString; // provide flatcase alias for usage consistency in the consumer

#endif // EXPR_TOSTRING_HPP