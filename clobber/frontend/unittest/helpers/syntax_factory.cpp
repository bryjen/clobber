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

#define PTR_GET_STR_METADATA(UPTR) std::any_cast<std::string>(UPTR->metadata.at(default_str_metadata_tag))
#define GET_STR_METADATA(OBJ) std::any_cast<std::string>(OBJ.metadata.at(default_str_metadata_tag))

namespace SyntaxFactory {
    clobber::Parameter *
    Parameter(clobber::IdentifierExpr *identifier, clobber::TypeExpr *type_annot) {
        auto identifier_uptr = std::unique_ptr<clobber::IdentifierExpr>(identifier);
        auto type_annot_uptr = std::unique_ptr<clobber::TypeExpr>(type_annot);

        const std::string metadata_str =
            std::string(std::format("{} {}", PTR_GET_STR_METADATA(identifier_uptr), PTR_GET_STR_METADATA(type_annot_uptr)));
        auto parameter                                = new clobber::Parameter(std::move(identifier_uptr), std::move(type_annot_uptr));
        parameter->metadata[default_str_metadata_tag] = metadata_str;
        return parameter;
    }
    clobber::Binding *
    Binding(clobber::IdentifierExpr *identifier, clobber::TypeExpr *type_annot, clobber::Expr *value) {
        auto identifier_uptr = std::unique_ptr<clobber::IdentifierExpr>(identifier);
        auto type_annot_uptr = std::unique_ptr<clobber::TypeExpr>(type_annot);
        auto value_uptr      = std::unique_ptr<clobber::Expr>(value);

        const std::string metadata_str = std::string(std::format("{} {} {}", PTR_GET_STR_METADATA(identifier_uptr),
                                                                 PTR_GET_STR_METADATA(type_annot_uptr), PTR_GET_STR_METADATA(value_uptr)));
        auto parameter = new clobber::Binding(std::move(identifier_uptr), std::move(type_annot_uptr), std::move(value_uptr));
        parameter->metadata[default_str_metadata_tag] = metadata_str;
        return parameter;
    }

    clobber::ParameterVectorExpr *
    ParameterVector(std::vector<clobber::Parameter *> parameters) {
        std::vector<std::string> param_strs;
        std::vector<std::unique_ptr<clobber::Parameter>> parameter_uptrs;
        for (const auto &param : parameters) {
            param_strs.push_back(std::any_cast<std::string>(param->metadata.at(default_str_metadata_tag)));
            parameter_uptrs.push_back(std::unique_ptr<clobber::Parameter>(param));
        }

        clobber::ParameterVectorExpr *pve = new clobber::ParameterVectorExpr(OpenBracket(), std::move(parameter_uptrs), CloseBracket());
        pve->metadata[default_str_metadata_tag] = std::string(std::format("[{}]", str_utils::join(" ", param_strs)));
        return pve;
    }

    clobber::BindingVectorExpr *
    BindingVector(std::vector<clobber::Binding *> bindings) {
        std::vector<std::string> binding_strs;
        std::vector<std::unique_ptr<clobber::Binding>> binding_uptrs;
        for (const auto &binding : bindings) {
            binding_strs.push_back(std::any_cast<std::string>(binding->metadata.at(default_str_metadata_tag)));
            binding_uptrs.push_back(std::unique_ptr<clobber::Binding>(binding));
        }

        clobber::BindingVectorExpr *bve         = new clobber::BindingVectorExpr(OpenBracket(), std::move(binding_uptrs), CloseBracket());
        bve->metadata[default_str_metadata_tag] = std::string(std::format("[{}]", str_utils::join(" ", binding_strs)));
        return bve;
    }

    clobber::BuiltinTypeExpr *
    BuiltinType(const clobber::Token &type_token) {
        auto bte                                = new clobber::BuiltinTypeExpr(Caret(), type_token);
        bte->metadata[default_str_metadata_tag] = std::string(std::format("^{}", GET_STR_METADATA(type_token)));
        return bte;
    }

    clobber::UserDefinedTypeExpr *
    UserDefinedType(const clobber::Token &identifier_token) {
        auto udt                                = new clobber::UserDefinedTypeExpr(Caret(), identifier_token);
        udt->metadata[default_str_metadata_tag] = std::string(std::format("^{}", GET_STR_METADATA(identifier_token)));
        return udt;
    }

    clobber::ParameterizedTypeExpr *
    ParameterizedType(clobber::TypeExpr *type_expr, std::vector<clobber::Expr *> param_values) {
        std::vector<std::string> strs;

        std::vector<std::string> param_value_strs;
        std::vector<std::unique_ptr<clobber::Expr>> param_value_uptrs;
        for (const auto &param_value : param_values) {
            param_value_strs.push_back(PTR_GET_STR_METADATA(param_value));
            param_value_uptrs.push_back(std::unique_ptr<clobber::Expr>(param_value));
        }

        std::vector<clobber::Token> commas;
        for (size_t i = 0; i < param_values.size() - 1; i++) {
            commas.push_back(Comma());
        }

        auto type_expr_uptr = std::unique_ptr<clobber::TypeExpr>(type_expr);
        auto pte =
            new clobber::ParameterizedTypeExpr(std::move(type_expr_uptr), LessThan(), std::move(param_value_uptrs), commas, GreaterThan());
        pte->metadata[default_str_metadata_tag] =
            std::format("{}<{}>", PTR_GET_STR_METADATA(pte->type_expr), str_utils::join(",", param_value_strs));
        return pte;
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
        auto bve_uptr = std::unique_ptr<clobber::BindingVectorExpr>(bve);

        std::vector<std::string> value_strs;
        std::vector<std::unique_ptr<clobber::Expr>> body_expr_uptrs;
        for (const auto &body_expr : body_exprs) {
            body_expr_uptrs.push_back(std::unique_ptr<clobber::Expr>(body_expr));
            value_strs.push_back(get_repr(body_expr));
        }

        auto ae = new clobber::accel::AccelExpr(OpenParen(), AccelKeyword(), std::move(bve_uptr), std::move(body_expr_uptrs), CloseParen());
        ae->metadata[default_str_metadata_tag] = std::string(std::format("(accel {} {})", get_repr(bve), str_utils::join(" ", value_strs)));
        return ae;
    }

    clobber::KeywordLiteralExpr *
    KeywordLiteralExpr(const std::string &name) {
        auto ie                                = new clobber::KeywordLiteralExpr(name, KeywordLiteralInsertColon(name));
        ie->metadata[default_str_metadata_tag] = std::string(":" + name);
        return ie;
    }

    clobber::VectorExpr *
    VectorExpr(std::vector<clobber::Expr *> values) {
        std::vector<std::string> strs;
        std::vector<std::unique_ptr<clobber::Expr>> value_uptrs;
        for (const auto &value : values) {
            strs.push_back(get_repr(value));
            value_uptrs.push_back(std::unique_ptr<clobber::Expr>(value));
        }

        std::vector<clobber::Token> commas;

        auto ve = new clobber::VectorExpr(OpenBracket(), std::move(value_uptrs), std::move(commas), CloseBrace());
        ve->metadata[default_str_metadata_tag] = std::string(std::format("[{}]", str_utils::join(" ", strs)));
        return ve;
    }

    clobber::accel::TOSAOpExpr *
    TosaOpExpr(const clobber::Token op_token, std::vector<clobber::Expr *> values) {
        std::vector<std::string> value_strs;
        std::vector<std::unique_ptr<clobber::Expr>> value_uptrs;
        for (const auto &value : values) {
            value_uptrs.push_back(std::unique_ptr<clobber::Expr>(value));
            value_strs.push_back(get_repr(value));
        }

        auto ce = new clobber::accel::TOSAOpExpr(OpenParen(), op_token, std::move(value_uptrs), CloseParen());
        ce->metadata[default_str_metadata_tag] = std::string(std::format("({} {})", get_repr(&op_token), str_utils::join(" ", value_strs)));
        return ce;
    }

    clobber::accel::TensorExpr *
    TensorExpr(std::vector<clobber::Expr *> values) {
        std::vector<std::string> value_strs;
        std::vector<std::unique_ptr<clobber::Expr>> value_uptrs;
        for (const auto &value : values) {
            value_uptrs.push_back(std::unique_ptr<clobber::Expr>(value));
            value_strs.push_back(get_repr(value));
        }

        auto ce = new clobber::accel::TensorExpr(OpenParen(), TensorKeyword(), std::move(value_uptrs), CloseParen());
        ce->metadata[default_str_metadata_tag] = std::string(std::format("(tensor {})", str_utils::join(" ", value_strs)));
        return ce;
    }
} // namespace SyntaxFactory