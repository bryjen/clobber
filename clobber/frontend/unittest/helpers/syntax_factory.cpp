#include "syntax_factory.hpp"

#include <magic_enum/magic_enum.hpp>

#include <clobber/common/utils.hpp>

#include "pch.hpp"

using namespace SyntaxFactory;

namespace {
    std::string
    get_repr(const clobber::BindingVectorExpr *bve) {
        if (!bve->metadata.contains(default_str_metadata_tag)) {
            std::string expr_type_str = "BindingVectorExpr";
            std::string msg = std::format("Couldn't find \"{}\" in the metadata for \"\"", default_str_metadata_tag, expr_type_str);
            std::cerr << msg << std::endl;
            throw std::runtime_error(msg);
        }

        return std::any_cast<std::string>(bve->metadata.at(default_str_metadata_tag)); // guaranteed to be a string
    }

    std::string
    get_repr(const clobber::ParameterVectorExpr *pve) {
        if (!pve->metadata.contains(default_str_metadata_tag)) {
            std::string expr_type_str = "ParameterVectorExpr";
            std::string msg = std::format("Couldn't find \"{}\" in the metadata for \"\"", default_str_metadata_tag, expr_type_str);
            std::cerr << msg << std::endl;
            throw std::runtime_error(msg);
        }

        return std::any_cast<std::string>(pve->metadata.at(default_str_metadata_tag)); // guaranteed to be a string
    }

    std::string
    get_repr(const clobber::Expr *expr) {
        if (!expr->metadata.contains(default_str_metadata_tag)) {
            std::string expr_type_str = std::string(magic_enum::enum_name(expr->type));
            std::string msg = std::format("Couldn't find \"{}\" in the metadata for \"\"", default_str_metadata_tag, expr_type_str);
            std::cerr << msg << std::endl;
            throw std::runtime_error(msg);
        }

        return std::any_cast<std::string>(expr->metadata.at(default_str_metadata_tag)); // guaranteed to be a string
    }

    std::string
    get_repr(const clobber::Token *token) {
        if (!token->metadata.contains(default_str_metadata_tag)) {
            std::string expr_type_str = std::string(magic_enum::enum_name(token->type));
            std::string msg = std::format("Couldn't find \"{}\" in the metadata for \"\"", default_str_metadata_tag, expr_type_str);
            std::cerr << msg << std::endl;
            throw std::runtime_error(msg);
        }

        return std::any_cast<std::string>(token->metadata.at(default_str_metadata_tag)); // guaranteed to be a string
    }
}; // namespace

namespace SyntaxFactory {
    clobber::ParameterVectorExpr *
    ParameterVectorExpr(std::vector<clobber::IdentifierExpr *> identifiers) {
        std::vector<std::string> strs;
        strs.push_back("[");

        std::vector<std::unique_ptr<clobber::IdentifierExpr>> identifier_uptrs;
        for (size_t i = 0; i < identifiers.size(); i++) {
            auto identifier = identifiers[i];
            strs.push_back(get_repr(identifier));
            identifier_uptrs.push_back(std::unique_ptr<clobber::IdentifierExpr>(identifier));

            if (i < identifiers.size() - 1) {
                strs.push_back(" ");
            }
        }

        strs.push_back("]");

        clobber::ParameterVectorExpr *pve = new clobber::ParameterVectorExpr(OpenBracket(), std::move(identifier_uptrs), CloseBracket());
        pve->metadata[default_str_metadata_tag] = std::string(str_utils::join("", strs));
        return pve;
    }

    clobber::BindingVectorExpr *
    BindingVectorExpr(std::vector<clobber::IdentifierExpr *> identifiers, std::vector<clobber::Expr *> exprs) {
        if (identifiers.size() != exprs.size()) {
            std::cerr << std::format("Error creating `BindingVectorExpr`, identifiers and values counts do not match.");
            throw -1;
        }

        std::vector<std::string> strs;
        strs.push_back("[");

        std::vector<std::unique_ptr<clobber::IdentifierExpr>> identifier_uptrs;
        std::vector<std::unique_ptr<clobber::Expr>> expr_uptrs;

        for (size_t i = 0; i < identifiers.size(); i++) {
            auto identifier = identifiers[i];
            strs.push_back(get_repr(identifier));
            identifier_uptrs.push_back(std::unique_ptr<clobber::IdentifierExpr>(identifier));
            strs.push_back(" ");

            auto expr = exprs[i];
            strs.push_back(get_repr(expr));
            expr_uptrs.push_back(std::unique_ptr<clobber::Expr>(expr));

            if (i < identifiers.size() - 1) {
                strs.push_back(" ");
            }
        }

        strs.push_back("]");

        size_t nb = identifiers.size();
        auto bve  = new clobber::BindingVectorExpr(OpenBracket(), std::move(identifier_uptrs), std::move(expr_uptrs), CloseBracket(), nb);
        bve->metadata[default_str_metadata_tag] = std::string(str_utils::join("", strs));
        return bve;
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
        std::vector<std::string> strs;

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
        auto ie                                = new clobber::IdentifierExpr(name, Identifier(name));
        ie->metadata[default_str_metadata_tag] = std::string(name);
        return ie;
    }

    clobber::NumLiteralExpr *
    NumLiteralExpr(const std::string &num_lit) {
        auto nle                                = new clobber::NumLiteralExpr(NumericLiteral(num_lit));
        nle->metadata[default_str_metadata_tag] = std::string(num_lit);
        return nle;
    }

    clobber::StringLiteralExpr *
    StringLiteralExpr(const std::string &value) {
        auto sle                                = new clobber::StringLiteralExpr(value, StringLiteralInsertDoubleQuot(value));
        sle->metadata[default_str_metadata_tag] = std::string(std::format("\"{}\"", value));
        return sle;
    }

    clobber::CharLiteralExpr *
    CharLiteralExpr(const std::string &char_lit) {
        auto cle                                = new clobber::CharLiteralExpr(char_lit, CharLiteral(char_lit));
        cle->metadata[default_str_metadata_tag] = std::string(char_lit);
        return cle;
    }

    clobber::LetExpr *
    LetExpr(clobber::BindingVectorExpr *bve, std::vector<clobber::Expr *> body_exprs) {
        std::vector<std::string> strs;
        strs.push_back("(");
        strs.push_back("let ");

        auto bve_uptr = std::unique_ptr<clobber::BindingVectorExpr>(bve);
        strs.push_back(std::format("{} ", get_repr(bve_uptr.get())));

        std::vector<std::unique_ptr<clobber::Expr>> body_expr_uptrs;
        for (size_t i = 0; i < body_exprs.size(); i++) {
            auto body_expr = body_exprs[i];
            strs.push_back(get_repr(body_expr));
            body_expr_uptrs.push_back(std::unique_ptr<clobber::Expr>(body_expr));

            if (i < body_exprs.size() - 1) {
                strs.push_back(" ");
            }
        }

        strs.push_back(")");

        auto le = new clobber::LetExpr(OpenParen(), LetKeyword(), std::move(bve_uptr), std::move(body_expr_uptrs), CloseParen());
        le->metadata[default_str_metadata_tag] = std::string(str_utils::join("", strs));
        return le;
    }

    clobber::FnExpr *
    FnExpr(clobber::ParameterVectorExpr *pve, std::vector<clobber::Expr *> body_exprs) {
        std::vector<std::string> strs;
        strs.push_back("(");
        strs.push_back("fn ");

        auto pve_uptr = std::unique_ptr<clobber::ParameterVectorExpr>(pve);
        strs.push_back(get_repr(pve_uptr.get()));
        strs.push_back(" ");

        std::vector<std::unique_ptr<clobber::Expr>> body_expr_uptrs;
        for (size_t i = 0; i < body_exprs.size(); i++) {
            auto body_expr = body_exprs[i];
            strs.push_back(get_repr(body_expr));
            body_expr_uptrs.push_back(std::unique_ptr<clobber::Expr>(body_expr));

            if (i < body_exprs.size() - 1) {
                strs.push_back(" ");
            }
        }

        strs.push_back(")");
        auto fe = new clobber::FnExpr(OpenParen(), FnKeyword(), std::move(pve_uptr), std::move(body_expr_uptrs), CloseParen());
        fe->metadata[default_str_metadata_tag] = std::string(str_utils::join("", strs));
        return fe;
    }

    clobber::DefExpr *
    DefExpr(clobber::IdentifierExpr *identifier, clobber::Expr *value) {
        std::vector<std::string> strs;
        strs.push_back("(");
        strs.push_back("def ");

        auto identifier_uptr = std::unique_ptr<clobber::IdentifierExpr>(identifier);
        strs.push_back(get_repr(identifier_uptr.get()));
        strs.push_back(" ");

        auto value_uptr = std::unique_ptr<clobber::Expr>(value);
        strs.push_back(get_repr(value_uptr.get()));

        strs.push_back(")");

        auto de = new clobber::DefExpr(OpenParen(), DefKeyword(), std::move(identifier_uptr), std::move(value_uptr), CloseParen());
        de->metadata[default_str_metadata_tag] = std::string(str_utils::join("", strs));
        return de;
    }

    clobber::DoExpr *
    DoExpr(std::vector<clobber::Expr *> body_exprs) {
        std::vector<std::string> strs;
        strs.push_back("(");
        strs.push_back("do ");

        std::vector<std::unique_ptr<clobber::Expr>> body_expr_uptrs;
        for (size_t i = 0; i < body_exprs.size(); i++) {
            auto body_expr = body_exprs[i];
            strs.push_back(get_repr(body_expr));
            body_expr_uptrs.push_back(std::unique_ptr<clobber::Expr>(body_expr));

            if (i < body_exprs.size() - 1) {
                strs.push_back(" ");
            }
        }

        strs.push_back(")");

        auto de                                = new clobber::DoExpr(OpenParen(), DoKeyword(), std::move(body_expr_uptrs), CloseParen());
        de->metadata[default_str_metadata_tag] = std::string(str_utils::join("", strs));
        return de;
    }

    clobber::CallExpr *
    CallExpr(clobber::Expr *operator_expr, std::vector<clobber::Expr *> arguments) {
        std::vector<std::string> strs;
        strs.push_back("(");

        auto operator_expr_uptr = std::unique_ptr<clobber::Expr>(operator_expr);
        strs.push_back(get_repr(operator_expr_uptr.get()));
        strs.push_back(" ");

        std::vector<std::unique_ptr<clobber::Expr>> argument_uptrs;
        for (size_t i = 0; i < arguments.size(); i++) {
            auto argument = arguments[i];
            strs.push_back(get_repr(argument));
            argument_uptrs.push_back(std::unique_ptr<clobber::Expr>(argument));

            if (i < arguments.size() - 1) {
                strs.push_back(" ");
            }
        }

        strs.push_back(")");

        auto ce = new clobber::CallExpr(OpenParen(), std::move(operator_expr_uptr), std::move(argument_uptrs), CloseParen());
        ce->metadata[default_str_metadata_tag] = std::string(str_utils::join("", strs));
        return ce;
    }

    clobber::CallExpr *
    CallExpr(const std::string &fn_name, std::vector<clobber::Expr *> arguments) {
        return CallExpr(IdentifierExpr(fn_name), arguments);
    }

    clobber::accel::AccelExpr *
    AccelExpr(clobber::BindingVectorExpr *bve, std::vector<clobber::Expr *> body_exprs) {
        std::vector<std::string> strs;
        strs.push_back("(");
        strs.push_back("accel");

        auto bve_uptr = std::unique_ptr<clobber::BindingVectorExpr>(bve);
        strs.push_back(get_repr(bve_uptr.get()));

        std::vector<std::unique_ptr<clobber::Expr>> body_expr_uptrs;
        for (const auto &body_expr : body_exprs) {
            body_expr_uptrs.push_back(std::unique_ptr<clobber::Expr>(body_expr));
        }

        strs.push_back(")");

        auto ae = new clobber::accel::AccelExpr(OpenParen(), AccelKeyword(), std::move(bve_uptr), std::move(body_expr_uptrs), CloseParen());
        ae->metadata[default_str_metadata_tag] = std::string(str_utils::join("", strs));
        return ae;
    }

    clobber::accel::MatMulExpr *
    MatMulExpr(clobber::Expr *fst_operand, clobber::Expr *snd_operand) {
        std::vector<std::string> strs;
        strs.push_back("(");
        strs.push_back("matmul");

        auto fst_operand_uptr = std::unique_ptr<clobber::Expr>(fst_operand);
        auto snd_operand_uptr = std::unique_ptr<clobber::Expr>(snd_operand);

        strs.push_back(")");

        auto mme = new clobber::accel::MatMulExpr(OpenParen(), MatmulKeyword(), std::move(fst_operand_uptr), std::move(snd_operand_uptr),
                                                  CloseParen());
        mme->metadata[default_str_metadata_tag] = std::string(str_utils::join("", strs));
        return mme;
    }

    clobber::accel::RelUExpr *
    RelUExpr(clobber::Expr *operand) {
        std::vector<std::string> strs;
        strs.push_back("(");
        strs.push_back("relu");

        auto operand_uptr = std::unique_ptr<clobber::Expr>(operand);

        strs.push_back(")");

        auto re = new clobber::accel::RelUExpr(OpenParen(), ReluKeyword(), std::move(operand_uptr), CloseParen());
        re->metadata[default_str_metadata_tag] = std::string(str_utils::join("", strs));
        return re;
    }

} // namespace SyntaxFactory