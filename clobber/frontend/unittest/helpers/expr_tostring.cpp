#include "pch.hpp"

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
    case ClobberExprType::StringLiteralExpr: {
        const StringLiteralExpr &str_lit_expr = static_cast<const StringLiteralExpr &>(expr);
        return expr2str::str_lit_expr(source_text, str_lit_expr);
    }
    case ClobberExprType::CharLiteralExpr: {
        const CharLiteralExpr &str_lit_expr = static_cast<const CharLiteralExpr &>(expr);
        return expr2str::char_lit_expr(source_text, str_lit_expr);
    }

    case ClobberExprType::CallExpr: {
        const CallExpr &call_expr = static_cast<const CallExpr &>(expr);
        return expr2str::call_expr(source_text, call_expr);
    }
    case ClobberExprType::IdentifierExpr: {
        const IdentifierExpr &iden_expr = static_cast<const IdentifierExpr &>(expr);
        return expr2str::iden_expr(source_text, iden_expr);
    }
    case ClobberExprType::LetExpr: {
        const LetExpr &let_expr = static_cast<const LetExpr &>(expr);
        return expr2str::let_expr(source_text, let_expr);
    }
    case ClobberExprType::BindingVectorExpr: {
        const BindingVectorExpr &binding_vector_expr = static_cast<const BindingVectorExpr &>(expr);
        return expr2str::binding_vector_expr(source_text, binding_vector_expr);
    }
    case ClobberExprType::ParameterVectorExpr: {
        const ParameterVectorExpr &parameter_vector_expr = static_cast<const ParameterVectorExpr &>(expr);
        return expr2str::parameter_vector_expr(source_text, parameter_vector_expr);
    }
    case ClobberExprType::FnExpr: {
        const FnExpr &fn_expr = static_cast<const FnExpr &>(expr);
        return expr2str::fn_expr(source_text, fn_expr);
    }
    case ClobberExprType::DefExpr: {
        const DefExpr &def_expr = static_cast<const DefExpr &>(expr);
        return expr2str::def_expr(source_text, def_expr);
    }
    case ClobberExprType::DoExpr: {
        const DoExpr &do_expr = static_cast<const DoExpr &>(expr);
        return expr2str::do_expr(source_text, do_expr);
    }

    case ClobberExprType::AccelExpr: {
        const AccelExpr &accel_expr = static_cast<const AccelExpr &>(expr);
        return expr2str::accel_expr(source_text, accel_expr);
    }
    case ClobberExprType::MatMulExpr: {
        const MatMulExpr &mat_mul_expr = static_cast<const MatMulExpr &>(expr);
        return expr2str::mat_mul_expr(source_text, mat_mul_expr);
    }
    case ClobberExprType::RelUExpr: {
        const RelUExpr &relu_expr = static_cast<const RelUExpr &>(expr);
        return expr2str::relu_expr(source_text, relu_expr);
    }
    default: {
        std::string expr_type_str = std::string(magic_enum::enum_name(expr.expr_type));
        std::cerr << std::format("No 'tostring' function for expr type \"{}\"", expr_type_str) << std::endl;
        throw 0;
    }
    }
}

std::string
token_tostring(const std::string &source_text, const ClobberToken &token) {
    return source_text.substr(token.full_start, token.full_length);
}

std::string
ExprToString::str_lit_expr(const std::string &source_text, const StringLiteralExpr &sle) {
    return token_tostring(source_text, sle.token);
}

std::string
ExprToString::char_lit_expr(const std::string &source_text, const CharLiteralExpr &cle) {
    return token_tostring(source_text, cle.token);
}

std::string
ExprToString::binding_vector_expr(const std::string &source_text, const BindingVectorExpr &bve) {
    std::vector<std::string> strs;
    strs.push_back(token_tostring(source_text, bve.open_bracket_token));

    for (size_t i = 0; i < bve.num_bindings; i++) {
        std::reference_wrapper<const IdentifierExpr> identifier = std::cref(*bve.identifiers[i]);
        std::reference_wrapper<const ExprBase> value            = std::cref(*bve.exprs[i]);

        strs.push_back(ExprToString::iden_expr(source_text, identifier));
        strs.push_back(ExprToString::expr_base(source_text, value));
    }

    strs.push_back(token_tostring(source_text, bve.close_bracket_token));
    return str_utils::join("", strs);
}

std::string
ExprToString::let_expr(const std::string &source_text, const LetExpr &let_expr) {
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
ExprToString::parameter_vector_expr(const std::string &source_text, const ParameterVectorExpr &pve) {
    std::vector<std::string> strs;
    strs.push_back(token_tostring(source_text, pve.open_bracket_token));

    for (size_t i = 0; i < pve.identifiers.size(); i++) {
        std::reference_wrapper<const IdentifierExpr> identifier = std::cref(*pve.identifiers[i]);
        strs.push_back(ExprToString::iden_expr(source_text, identifier));
    }

    strs.push_back(token_tostring(source_text, pve.close_bracket_token));
    return str_utils::join("", strs);
}

std::string
ExprToString::fn_expr(const std::string &source_text, const FnExpr &fn_expr) {
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
ExprToString::def_expr(const std::string &source_text, const DefExpr &def_expr) {
    std::vector<std::string> strs;
    strs.push_back(token_tostring(source_text, def_expr.open_paren_token));
    strs.push_back(token_tostring(source_text, def_expr.def_token));
    strs.push_back(ExprToString::iden_expr(source_text, std::cref(*def_expr.identifier)));
    strs.push_back(ExprToString::expr_base(source_text, std::cref(*def_expr.value)));
    strs.push_back(token_tostring(source_text, def_expr.close_paren_token));
    return str_utils::join("", strs);
}

std::string
ExprToString::do_expr(const std::string &source_text, const DoExpr &do_expr) {
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
ExprToString::num_lit_expr(const std::string &source_text, const NumLiteralExpr &num_lit_expr) {
    return token_tostring(source_text, num_lit_expr.token);
}

std::string
ExprToString::call_expr(const std::string &source_text, const CallExpr &call_expr) {
    std::vector<std::string> strs;
    strs.push_back(token_tostring(source_text, call_expr.open_paren_token));
    strs.push_back(token_tostring(source_text, call_expr.operator_token));

    std::vector<std::reference_wrapper<const ExprBase>> view_arguments = ptr_utils::get_expr_views(call_expr.arguments);
    for (const auto &arg_expr : view_arguments) {
        strs.push_back(ExprToString::expr_base(source_text, arg_expr.get()));
    }

    strs.push_back(token_tostring(source_text, call_expr.close_paren_token));
    return str_utils::join("", strs);
}

std::string
ExprToString::accel_expr(const std::string &source_text, const AccelExpr &accel_expr) {
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
ExprToString::mat_mul_expr(const std::string &source_text, const MatMulExpr &mat_mul_expr) {
    std::vector<std::string> strs;
    strs.push_back(token_tostring(source_text, mat_mul_expr.open_paren_token));
    strs.push_back(token_tostring(source_text, mat_mul_expr.mat_mul_token));
    strs.push_back(ExprToString::expr_base(source_text, std::cref(*mat_mul_expr.fst_operand)));
    strs.push_back(ExprToString::expr_base(source_text, std::cref(*mat_mul_expr.snd_operand)));
    strs.push_back(token_tostring(source_text, mat_mul_expr.close_paren_token));
    return str_utils::join("", strs);
}

std::string
ExprToString::relu_expr(const std::string &source_text, const RelUExpr &relu_expr) {
    std::vector<std::string> strs;
    strs.push_back(token_tostring(source_text, relu_expr.open_paren_token));
    strs.push_back(token_tostring(source_text, relu_expr.relu_token));
    strs.push_back(ExprToString::expr_base(source_text, std::cref(*relu_expr.operand)));
    strs.push_back(token_tostring(source_text, relu_expr.close_paren_token));
    return str_utils::join("", strs);
}

std::string
ExprToString::iden_expr(const std::string &source_text, const IdentifierExpr &iden_expr) {
    return token_tostring(source_text, iden_expr.token);
}

std::string
tree_visualization(const std::string &source_text, const ExprBase &expr) {
    throw 0;
}