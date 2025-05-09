#include <format>
#include <sstream>
#include <string>
#include <vector>

#include <magic_enum/magic_enum.hpp>

#include <clobber/ast.hpp>
#include <clobber/common/utils.hpp>

#include "expr_tostring.hpp"

std::string
indent(size_t indentation, const std::string &str) {
    return std::format("{}{}", str_utils::spaces(indentation), str);
}

inline std::string
expr_type_tostring(ClobberExprType expr_type) {
    return std::string(magic_enum::enum_name(expr_type));
}

inline std::string
normalize(const std::string &str) {
    throw 0;
}

std::string
ExprToString::expr_base(const std::string &source_text, const ExprBase &expr) {
    switch (expr.expr_type) {
    case ClobberExprType::NumericLiteralExpr: {
        const NumLiteralExpr &num_lit_expr = static_cast<const NumLiteralExpr &>(expr);
        return expr2str::num_lit_expr(source_text, num_lit_expr);
    }
    case ClobberExprType::CallExpr: {
        const CallExpr &call_expr = static_cast<const CallExpr &>(expr);
        return expr2str::call_expr(source_text, call_expr);
    }
    case ClobberExprType::IdentifierExpr: {
        const IdentifierExpr &iden_expr = static_cast<const IdentifierExpr &>(expr);
        return expr2str::iden_expr(source_text, iden_expr);
    }
    default:
        throw 0;
    }
}

std::string
token_tostring(const std::string &source_text, const ClobberToken &token) {
    return source_text.substr(token.full_start, token.full_length);
}

std::string
ExprToString::num_lit_expr(const std::string &source_text, const NumLiteralExpr &num_lit_expr) {
    return token_tostring(source_text, num_lit_expr.token);
}

std::string
ExprToString::call_expr(const std::string &source_text, const CallExpr &call_expr) {
    std::vector<std::string> lines;
    lines.push_back(token_tostring(source_text, call_expr.open_paren_token));
    lines.push_back(token_tostring(source_text, call_expr.operator_token));

    std::vector<std::reference_wrapper<const ExprBase>> view_arguments = ptr_utils::get_expr_views(call_expr.arguments);
    for (const auto &arg_expr : view_arguments) {
        lines.push_back(ExprToString::expr_base(source_text, arg_expr.get()));
    }

    lines.push_back(token_tostring(source_text, call_expr.close_paren_token));

    return str_utils::join("", lines);
}

std::string
ExprToString::iden_expr(const std::string &source_text, const IdentifierExpr &iden_expr) {
    return token_tostring(source_text, iden_expr.token);
}

std::string
tree_visualization(const std::string &source_text, const ExprBase &expr) {
    throw 0;
}