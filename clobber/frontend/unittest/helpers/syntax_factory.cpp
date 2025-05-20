#include "syntax_factory.hpp"
#include "pch.hpp"

using namespace SyntaxFactory;

namespace SyntaxFactory {
    clobber::ParameterVectorExpr *
    ParameterVectorExpr(std::vector<clobber::IdentifierExpr *> identifiers) {
        std::vector<std::unique_ptr<clobber::IdentifierExpr>> identifier_uptrs;
        for (const auto &identifier : identifiers) {
            identifier_uptrs.push_back(std::unique_ptr<clobber::IdentifierExpr>(identifier));
        }

        return new clobber::ParameterVectorExpr(OpenBracket(), std::move(identifier_uptrs), CloseBracket());
    }

    clobber::BindingVectorExpr *
    BindingVectorExpr(std::vector<clobber::IdentifierExpr *> identifiers, std::vector<clobber::Expr *> exprs) {
        std::vector<std::unique_ptr<clobber::IdentifierExpr>> identifier_uptrs;
        for (const auto &identifier : identifiers) {
            identifier_uptrs.push_back(std::unique_ptr<clobber::IdentifierExpr>(identifier));
        }

        std::vector<std::unique_ptr<clobber::Expr>> expr_uptrs;
        for (const auto &expr : exprs) {
            expr_uptrs.push_back(std::unique_ptr<clobber::Expr>(expr));
        }

        if (identifiers.size() != exprs.size()) {
            std::cerr << std::format("Error creating `BindingVectorExpr`, identifiers and values counts do not match.");
            throw -1;
        }

        size_t nb = identifiers.size();
        return new clobber::BindingVectorExpr(OpenBracket(), std::move(identifier_uptrs), std::move(expr_uptrs), CloseBracket(), nb);
    }

    clobber::BuiltinTypeExpr *
    BuiltinTypeExpr(const clobber::Token &type_token) {
        return new clobber::BuiltinTypeExpr(Caret(), type_token);
    }

    clobber::UserDefinedTypeExpr *
    UserDefinedTypeExpr(const clobber::Token &identifier_token) {
        return new clobber::UserDefinedTypeExpr(Caret(), identifier_token);
    }

    clobber::ParameterizedTypeExpr *
    ParameterizedTypeExpr(clobber::TypeExpr *type_expr, std::vector<clobber::Expr *> param_values) {
        std::vector<std::unique_ptr<clobber::Expr>> param_value_uptrs;
        for (const auto &param_value : param_values) {
            param_value_uptrs.push_back(std::unique_ptr<clobber::Expr>(param_value));
        }

        std::vector<clobber::Token> commas;
        for (size_t i = 0; i < param_values.size() - 1; i++) {
            commas.push_back(Comma());
        }

        auto type_expr_uptr = std::unique_ptr<clobber::TypeExpr>(type_expr);
        return new clobber::ParameterizedTypeExpr(std::move(type_expr_uptr), LessThan(), std::move(param_value_uptrs), commas,
                                                  GreaterThan());
    }

    clobber::IdentifierExpr *
    IdentifierExpr(const std::string &name) {
        return new clobber::IdentifierExpr(name, Identifier(name));
    }

    clobber::NumLiteralExpr *
    NumLiteralExpr(const std::string &num_lit) {
        return new clobber::NumLiteralExpr(NumericLiteral(num_lit));
    }

    clobber::StringLiteralExpr *
    StringLiteralExpr(const std::string &value) {
        return new clobber::StringLiteralExpr(value, StringLiteralInsertDoubleQuot(value));
    }

    clobber::CharLiteralExpr *
    CharLiteralExpr(const std::string &char_lit) {
        return new clobber::CharLiteralExpr(char_lit, CharLiteral(char_lit));
    }

    clobber::LetExpr *
    LetExpr(clobber::BindingVectorExpr *bve, std::vector<clobber::Expr *> body_exprs) {
        auto bve_uptr = std::unique_ptr<clobber::BindingVectorExpr>(bve);

        std::vector<std::unique_ptr<clobber::Expr>> body_expr_uptrs;
        for (const auto &body_expr : body_exprs) {
            body_expr_uptrs.push_back(std::unique_ptr<clobber::Expr>(body_expr));
        }

        return new clobber::LetExpr(OpenParen(), LetKeyword(), std::move(bve_uptr), std::move(body_expr_uptrs), CloseParen());
    }

    clobber::FnExpr *
    FnExpr(clobber::ParameterVectorExpr *pve, std::vector<clobber::Expr *> body_exprs) {
        auto pve_uptr = std::unique_ptr<clobber::ParameterVectorExpr>(pve);

        std::vector<std::unique_ptr<clobber::Expr>> body_expr_uptrs;
        for (const auto &body_expr : body_exprs) {
            body_expr_uptrs.push_back(std::unique_ptr<clobber::Expr>(body_expr));
        }

        return new clobber::FnExpr(OpenParen(), LetKeyword(), std::move(pve_uptr), std::move(body_expr_uptrs), CloseParen());
    }

    clobber::DefExpr *
    DefExpr(clobber::IdentifierExpr *identifier, clobber::Expr *value) {
        auto identifier_uptr = std::unique_ptr<clobber::IdentifierExpr>(identifier);
        auto value_uptr      = std::unique_ptr<clobber::Expr>(value);
        return new clobber::DefExpr(OpenParen(), DefKeyword(), std::move(identifier_uptr), std::move(value_uptr), CloseParen());
    }

    clobber::DoExpr *
    DoExpr(std::vector<clobber::Expr *> body_exprs) {
        std::vector<std::unique_ptr<clobber::Expr>> body_expr_uptrs;
        for (const auto &body_expr : body_exprs) {
            body_expr_uptrs.push_back(std::unique_ptr<clobber::Expr>(body_expr));
        }

        return new clobber::DoExpr(OpenParen(), DoKeyword(), std::move(body_expr_uptrs), CloseParen());
    }

    clobber::CallExpr *
    CallExpr(clobber::Expr *operator_expr, std::vector<clobber::Expr *> arguments) {
        auto operator_expr_uptr = std::unique_ptr<clobber::Expr>(operator_expr);

        std::vector<std::unique_ptr<clobber::Expr>> argument_uptrs;
        for (const auto &argument : arguments) {
            argument_uptrs.push_back(std::unique_ptr<clobber::Expr>(argument));
        }

        return new clobber::CallExpr(OpenParen(), std::move(operator_expr_uptr), std::move(argument_uptrs), CloseParen());
    }

    clobber::CallExpr *
    CallExpr(const std::string &fn_name, std::vector<clobber::Expr *> arguments) {
        return CallExpr(IdentifierExpr(fn_name), arguments);
    }

    clobber::accel::AccelExpr *
    AccelExpr(clobber::BindingVectorExpr *bve, std::vector<clobber::Expr *> body_exprs) {
        auto bve_uptr = std::unique_ptr<clobber::BindingVectorExpr>(bve);

        std::vector<std::unique_ptr<clobber::Expr>> body_expr_uptrs;
        for (const auto &body_expr : body_exprs) {
            body_expr_uptrs.push_back(std::unique_ptr<clobber::Expr>(body_expr));
        }

        return new clobber::accel::AccelExpr(OpenParen(), AccelKeyword(), std::move(bve_uptr), std::move(body_expr_uptrs), CloseParen());
    }

    clobber::accel::MatMulExpr *
    MatMulExpr(clobber::Expr *fst_operand, clobber::Expr *snd_operand) {
        auto fst_operand_uptr = std::unique_ptr<clobber::Expr>(fst_operand);
        auto snd_operand_uptr = std::unique_ptr<clobber::Expr>(snd_operand);

        return new clobber::accel::MatMulExpr(OpenParen(), MatmulKeyword(), std::move(fst_operand_uptr), std::move(snd_operand_uptr),
                                              CloseParen());
    }

    clobber::accel::RelUExpr *
    RelUExpr(clobber::Expr *operand) {
        auto operand_uptr = std::unique_ptr<clobber::Expr>(operand);

        return new clobber::accel::RelUExpr(OpenParen(), ReluKeyword(), std::move(operand_uptr), CloseParen());
    }

} // namespace SyntaxFactory