#include "expr_tostring.hpp"
#include "pch.hpp"

#include <magic_enum/magic_enum.hpp>

#include <clobber/ast.hpp>
#include <clobber/parser.hpp>
#include <clobber/semantics.hpp>

#include <clobber/common/utils.hpp>

std::string
indent(size_t indentation, const std::string &str) {
    return std::format("{}{}", str_utils::spaces(indentation), str);
}

inline std::string
expr_type_tostring(clobber::Expr::Type expr_type) {
    return std::string(magic_enum::enum_name(expr_type));
}

std::string
ExprToString::expr_base(const std::string &source_text, const clobber::Expr &expr) {
    switch (expr.type) {
    case clobber::Expr::Type::NumericLiteralExpr: {
        const clobber::NumLiteralExpr &num_lit_expr = static_cast<const clobber::NumLiteralExpr &>(expr);
        return expr2str::num_lit_expr(source_text, num_lit_expr);
    }
    case clobber::Expr::Type::StringLiteralExpr: {
        const clobber::StringLiteralExpr &str_lit_expr = static_cast<const clobber::StringLiteralExpr &>(expr);
        return expr2str::str_lit_expr(source_text, str_lit_expr);
    }
    case clobber::Expr::Type::CharLiteralExpr: {
        const clobber::CharLiteralExpr &str_lit_expr = static_cast<const clobber::CharLiteralExpr &>(expr);
        return expr2str::char_lit_expr(source_text, str_lit_expr);
    }

    case clobber::Expr::Type::CallExpr: {
        const clobber::CallExpr &call_expr = static_cast<const clobber::CallExpr &>(expr);
        return expr2str::call_expr(source_text, call_expr);
    }
    case clobber::Expr::Type::IdentifierExpr: {
        const clobber::IdentifierExpr &iden_expr = static_cast<const clobber::IdentifierExpr &>(expr);
        return expr2str::iden_expr(source_text, iden_expr);
    }
    case clobber::Expr::Type::LetExpr: {
        const clobber::LetExpr &let_expr = static_cast<const clobber::LetExpr &>(expr);
        return expr2str::let_expr(source_text, let_expr);
    }

        /*
        case clobber::Expr::Type::BindingVectorExpr: {
            const clobber::BindingVectorExpr &binding_vector_expr = static_cast<const clobber::BindingVectorExpr &>(expr);
            return expr2str::binding_vector_expr(source_text, binding_vector_expr);
        }
        case clobber::Expr::Type::ParameterVectorExpr: {
            const clobber::ParameterVectorExpr &parameter_vector_expr = static_cast<const clobber::ParameterVectorExpr &>(expr);
            return expr2str::parameter_vector_expr(source_text, parameter_vector_expr);
        }
        */

    case clobber::Expr::Type::FnExpr: {
        const clobber::FnExpr &fn_expr = static_cast<const clobber::FnExpr &>(expr);
        return expr2str::fn_expr(source_text, fn_expr);
    }
    case clobber::Expr::Type::DefExpr: {
        const clobber::DefExpr &def_expr = static_cast<const clobber::DefExpr &>(expr);
        return expr2str::def_expr(source_text, def_expr);
    }
    case clobber::Expr::Type::DoExpr: {
        const clobber::DoExpr &do_expr = static_cast<const clobber::DoExpr &>(expr);
        return expr2str::do_expr(source_text, do_expr);
    }

    case clobber::Expr::Type::AccelExpr: {
        const clobber::accel::AccelExpr &accel_expr = static_cast<const clobber::accel::AccelExpr &>(expr);
        return expr2str::accel_expr(source_text, accel_expr);
    }
    case clobber::Expr::Type::MatMulExpr: {
        const clobber::accel::MatMulExpr &mat_mul_expr = static_cast<const clobber::accel::MatMulExpr &>(expr);
        return expr2str::mat_mul_expr(source_text, mat_mul_expr);
    }
    case clobber::Expr::Type::RelUExpr: {
        const clobber::accel::RelUExpr &relu_expr = static_cast<const clobber::accel::RelUExpr &>(expr);
        return expr2str::relu_expr(source_text, relu_expr);
    }
    default: {
        std::string expr_type_str = std::string(magic_enum::enum_name(expr.type));
        std::cerr << std::format("No 'tostring' function for expr type \"{}\"", expr_type_str) << std::endl;
        throw 0;
    }
    }
}

std::string
token_tostring(const std::string &source_text, const clobber::ClobberToken &token) {
    return source_text.substr(token.full_start, token.full_length);
}

std::string
ExprToString::str_lit_expr(const std::string &source_text, const clobber::StringLiteralExpr &sle) {
    return token_tostring(source_text, sle.token);
}

std::string
ExprToString::char_lit_expr(const std::string &source_text, const clobber::CharLiteralExpr &cle) {
    return token_tostring(source_text, cle.token);
}

std::string
ExprToString::binding_vector_expr(const std::string &source_text, const clobber::BindingVectorExpr &bve) {
    std::vector<std::string> strs;
    strs.push_back(token_tostring(source_text, bve.open_bracket_token));

    for (size_t i = 0; i < bve.num_bindings; i++) {
        std::reference_wrapper<const clobber::IdentifierExpr> identifier = std::cref(*bve.identifiers[i]);
        std::reference_wrapper<const clobber::Expr> value                = std::cref(*bve.exprs[i]);

        strs.push_back(ExprToString::iden_expr(source_text, identifier));
        strs.push_back(ExprToString::expr_base(source_text, value));
    }

    strs.push_back(token_tostring(source_text, bve.close_bracket_token));
    return str_utils::join("", strs);
}

std::string
ExprToString::let_expr(const std::string &source_text, const clobber::LetExpr &let_expr) {
    std::vector<std::string> strs;
    strs.push_back(token_tostring(source_text, let_expr.open_paren_token));
    strs.push_back(token_tostring(source_text, let_expr.let_token));
    strs.push_back(ExprToString::binding_vector_expr(source_text, std::cref(*let_expr.binding_vector_expr)));

    for (const auto &body_expr : let_expr.body_exprs) {
        strs.push_back(ExprToString::expr_base(source_text, std::cref(*body_expr)));
    }

    strs.push_back(token_tostring(source_text, let_expr.close_paren_token));
    return str_utils::join("", strs);
}

std::string
ExprToString::parameter_vector_expr(const std::string &source_text, const clobber::ParameterVectorExpr &pve) {
    std::vector<std::string> strs;
    strs.push_back(token_tostring(source_text, pve.open_bracket_token));

    for (size_t i = 0; i < pve.identifiers.size(); i++) {
        std::reference_wrapper<const clobber::IdentifierExpr> identifier = std::cref(*pve.identifiers[i]);
        strs.push_back(ExprToString::iden_expr(source_text, identifier));
    }

    strs.push_back(token_tostring(source_text, pve.close_bracket_token));
    return str_utils::join("", strs);
}

std::string
ExprToString::fn_expr(const std::string &source_text, const clobber::FnExpr &fn_expr) {
    std::vector<std::string> strs;
    strs.push_back(token_tostring(source_text, fn_expr.open_paren_token));
    strs.push_back(token_tostring(source_text, fn_expr.fn_token));
    strs.push_back(ExprToString::parameter_vector_expr(source_text, std::cref(*fn_expr.parameter_vector_expr)));

    for (const auto &body_expr : fn_expr.body_exprs) {
        strs.push_back(ExprToString::expr_base(source_text, std::cref(*body_expr)));
    }

    strs.push_back(token_tostring(source_text, fn_expr.close_paren_token));
    return str_utils::join("", strs);
}

std::string
ExprToString::def_expr(const std::string &source_text, const clobber::DefExpr &def_expr) {
    std::vector<std::string> strs;
    strs.push_back(token_tostring(source_text, def_expr.open_paren_token));
    strs.push_back(token_tostring(source_text, def_expr.def_token));
    strs.push_back(ExprToString::iden_expr(source_text, std::cref(*def_expr.identifier)));
    strs.push_back(ExprToString::expr_base(source_text, std::cref(*def_expr.value)));
    strs.push_back(token_tostring(source_text, def_expr.close_paren_token));
    return str_utils::join("", strs);
}

std::string
ExprToString::do_expr(const std::string &source_text, const clobber::DoExpr &do_expr) {
    std::vector<std::string> strs;
    strs.push_back(token_tostring(source_text, do_expr.open_paren_token));
    strs.push_back(token_tostring(source_text, do_expr.do_token));

    for (const auto &body_expr : do_expr.body_exprs) {
        strs.push_back(ExprToString::expr_base(source_text, std::cref(*body_expr)));
    }

    strs.push_back(token_tostring(source_text, do_expr.close_paren_token));
    return str_utils::join("", strs);
}

std::string
ExprToString::num_lit_expr(const std::string &source_text, const clobber::NumLiteralExpr &num_lit_expr) {
    return token_tostring(source_text, num_lit_expr.token);
}

std::string
ExprToString::call_expr(const std::string &source_text, const clobber::CallExpr &call_expr) {
    std::vector<std::string> strs;
    strs.push_back(token_tostring(source_text, call_expr.open_paren_token));
    strs.push_back(token_tostring(source_text, call_expr.operator_token));

    std::vector<std::reference_wrapper<const clobber::Expr>> view_arguments = ptr_utils::get_expr_views(call_expr.arguments);
    for (const auto &arg_expr : view_arguments) {
        strs.push_back(ExprToString::expr_base(source_text, arg_expr.get()));
    }

    strs.push_back(token_tostring(source_text, call_expr.close_paren_token));
    return str_utils::join("", strs);
}

std::string
ExprToString::accel_expr(const std::string &source_text, const clobber::accel::AccelExpr &accel_expr) {
    std::vector<std::string> strs;
    strs.push_back(token_tostring(source_text, accel_expr.open_paren_token));
    strs.push_back(token_tostring(source_text, accel_expr.accel_token));
    strs.push_back(ExprToString::binding_vector_expr(source_text, std::cref(*accel_expr.binding_vector_expr)));

    for (const auto &body_expr : accel_expr.body_exprs) {
        strs.push_back(ExprToString::expr_base(source_text, std::cref(*body_expr)));
    }

    strs.push_back(token_tostring(source_text, accel_expr.close_paren_token));
    return str_utils::join("", strs);
}

std::string
ExprToString::mat_mul_expr(const std::string &source_text, const clobber::accel::MatMulExpr &mat_mul_expr) {
    std::vector<std::string> strs;
    strs.push_back(token_tostring(source_text, mat_mul_expr.open_paren_token));
    strs.push_back(token_tostring(source_text, mat_mul_expr.mat_mul_token));
    strs.push_back(ExprToString::expr_base(source_text, std::cref(*mat_mul_expr.fst_operand)));
    strs.push_back(ExprToString::expr_base(source_text, std::cref(*mat_mul_expr.snd_operand)));
    strs.push_back(token_tostring(source_text, mat_mul_expr.close_paren_token));
    return str_utils::join("", strs);
}

std::string
ExprToString::relu_expr(const std::string &source_text, const clobber::accel::RelUExpr &relu_expr) {
    std::vector<std::string> strs;
    strs.push_back(token_tostring(source_text, relu_expr.open_paren_token));
    strs.push_back(token_tostring(source_text, relu_expr.relu_token));
    strs.push_back(ExprToString::expr_base(source_text, std::cref(*relu_expr.operand)));
    strs.push_back(token_tostring(source_text, relu_expr.close_paren_token));
    return str_utils::join("", strs);
}

std::string
ExprToString::iden_expr(const std::string &source_text, const clobber::IdentifierExpr &iden_expr) {
    return token_tostring(source_text, iden_expr.token);
}

std::string
tree_visualization(const std::string &, const clobber::Expr &) {
    throw 0;
}