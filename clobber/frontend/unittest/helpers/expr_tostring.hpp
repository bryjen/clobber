// header containing functionality for pretty-printing AST nodes in clobber.
// not in main library to reduce compilation times (?); maybe measure if theres a noticeable performance difference.

#ifndef EXPR_TOSTRING_HPP
#define EXPR_TOSTRING_HPP

struct ExprBase;       // clobber/ast.hpp
struct NumLiteralExpr; // clobber/ast.hpp
struct CallExpr;       // clobber/ast.hpp
struct IdentifierExpr; // clobber/ast.hpp

#include <string>

namespace ExprToString {
std::string expr_base(const std::string &source_text, const ExprBase &expr);
std::string num_lit_expr(const std::string &source_text, const NumLiteralExpr &expr);
std::string call_expr(const std::string &source_text, const CallExpr &expr);
std::string iden_expr(const std::string &source_text, const IdentifierExpr &expr);
}; // namespace ExprToString

std::string tree_visualization(const std::string &source_text, const ExprBase &expr);

namespace expr2str = ExprToString; // provide flatcase alias for usage consistency in the consumer

#endif // EXPR_TOSTRING_HPP