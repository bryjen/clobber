#include <cstddef>
#include <typeindex>

#include "clobber/ast.hpp"

namespace utils {
    std::vector<std::unique_ptr<clobber::IdentifierExpr>>
    deepcopy_identifiers(const std::vector<std::unique_ptr<clobber::IdentifierExpr>> &identifiers) {
        std::vector<std::unique_ptr<clobber::IdentifierExpr>> copies;
        for (const auto &identifier : identifiers) {
            copies.push_back(identifier->clone_nowrap());
        }
        return copies;
    }

    std::vector<std::unique_ptr<clobber::Expr>>
    deepcopy_exprs(const std::vector<std::unique_ptr<clobber::Expr>> &exprs) {
        std::vector<std::unique_ptr<clobber::Expr>> copies;
        for (const auto &expr : exprs) {
            copies.push_back(expr->clone());
        }
        return copies;
    }

    size_t
    combine_hashes(const std::vector<std::size_t> &hashes) {
        std::size_t seed = 0;
        for (std::size_t h : hashes) {
            seed ^= h + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
} // namespace utils

size_t
clobber::Token::hash() const {
    std::vector<size_t> hashes;
    hashes.push_back(std::hash<std::type_index>{}(std::type_index(typeid(*this))));
    hashes.push_back(std::hash<Token::Type>{}(this->type));
    hashes.push_back(std::hash<size_t>{}(this->span.start));
    hashes.push_back(std::hash<size_t>{}(this->span.length));
    hashes.push_back(std::hash<size_t>{}(this->full_span.start));
    hashes.push_back(std::hash<size_t>{}(this->full_span.length));
    return utils::combine_hashes(hashes);
}

std::string
clobber::Token::ExtractText(const std::string &source_text) const {
    return source_text.substr(this->span.start, this->span.length);
}

std::string
clobber::Token::ExtractFullText(const std::string &source_text) const {
    return source_text.substr(this->full_span.start, this->full_span.length);
}

// OBSOLETE/DEPRECATED ?
bool
clobber::Token::AreEquivalent(const Token &token1, const Token &token2) {
    if (token1.type != token2.type) {
        return false;
    }

    return false;
}

clobber::Expr::Expr(Expr::Type type)
    : type(type) {}

clobber::TypeExpr::TypeExpr(TypeExpr::Type type)
    : Expr(Expr::Type::TypeExpr)
    , type_kind(type) {}

clobber::TypeExpr::TypeExpr(const TypeExpr &other)
    : Expr(other.type)
    , type_kind(other.type_kind) {}

clobber::BuiltinTypeExpr::BuiltinTypeExpr(const Token &caret_token, const Token &type_token)
    : TypeExpr(TypeExpr::Type::BuiltinType)
    , caret_token(caret_token)
    , type_keyword_token(type_token) {}

clobber::BuiltinTypeExpr::BuiltinTypeExpr(const BuiltinTypeExpr &other)
    : TypeExpr(other.type_kind)
    , caret_token(other.caret_token)
    , type_keyword_token(other.type_keyword_token) {}

clobber::Span
clobber::BuiltinTypeExpr::span() const {
    size_t start = this->caret_token.span.start;
    size_t len   = this->caret_token.span.length;
    len += this->type_keyword_token.span.length;
    return Span{start, len};
}

size_t
clobber::BuiltinTypeExpr::hash() const {
    std::vector<size_t> hashes;
    hashes.push_back(std::hash<std::type_index>{}(std::type_index(typeid(*this))));
    hashes.push_back(std::hash<clobber::TypeExpr::Type>{}(this->type_kind));

    hashes.push_back(this->type_keyword_token.hash());
    return utils::combine_hashes(hashes);
}

std::unique_ptr<clobber::Expr>
clobber::BuiltinTypeExpr::clone() const {
    throw 0;
}

std::unique_ptr<clobber::TypeExpr>
clobber::BuiltinTypeExpr::clone_nowrap() const {
    throw 0;
}

clobber::UserDefinedTypeExpr::UserDefinedTypeExpr(const Token &caret_token, const Token &identifier_token)
    : TypeExpr(TypeExpr::Type::UserDefinedType)
    , caret_token(caret_token)
    , identifier_token(identifier_token) {}

clobber::UserDefinedTypeExpr::UserDefinedTypeExpr(const UserDefinedTypeExpr &other)
    : TypeExpr(other.type_kind)
    , caret_token(other.caret_token)
    , identifier_token(other.identifier_token) {}

clobber::Span
clobber::UserDefinedTypeExpr::span() const {
    size_t start = this->caret_token.span.start;
    size_t len   = this->caret_token.span.length;
    len += this->identifier_token.span.length;
    return Span{start, len};
}

size_t
clobber::UserDefinedTypeExpr::hash() const {
    std::vector<size_t> hashes;
    hashes.push_back(std::hash<std::type_index>{}(std::type_index(typeid(*this))));
    hashes.push_back(std::hash<clobber::TypeExpr::Type>{}(this->type_kind));

    hashes.push_back(this->identifier_token.hash());
    return utils::combine_hashes(hashes);
}

std::unique_ptr<clobber::Expr>
clobber::UserDefinedTypeExpr::clone() const {
    throw 0;
}

std::unique_ptr<clobber::TypeExpr>
clobber::UserDefinedTypeExpr::clone_nowrap() const {
    throw 0;
}

clobber::ParameterizedTypeExpr::ParameterizedTypeExpr(std::unique_ptr<TypeExpr> type_expr, const Token &less_than_token,
                                                      std::vector<std::unique_ptr<Expr>> &&param_values, std::vector<Token> commas,
                                                      const Token &greater_than_token)
    : TypeExpr(TypeExpr::Type::ParameterizedType)
    , type_expr(std::move(type_expr))
    , less_than_token(less_than_token)
    , param_values(std::move(param_values))
    , commas(commas)
    , greater_than_token(greater_than_token) {}

clobber::ParameterizedTypeExpr::ParameterizedTypeExpr(const ParameterizedTypeExpr &other)
    : TypeExpr(other.type_kind)
    , type_expr(other.type_expr->clone_nowrap())
    , less_than_token(other.less_than_token)
    , param_values(utils::deepcopy_exprs(other.param_values))
    , commas(other.commas)
    , greater_than_token(other.greater_than_token) {}

clobber::Span
clobber::ParameterizedTypeExpr::span() const {
    Span type_expr_span = this->type_expr->span();
    size_t start        = type_expr_span.start;
    size_t len          = type_expr_span.length + this->less_than_token.span.length;

    for (const auto &param_value : this->param_values) {
        len += param_value->span().length;
    }

    for (const auto &comma : this->commas) {
        len += comma.span.length;
    }

    len += this->greater_than_token.span.length;
    return Span{start, len};
}

size_t
clobber::ParameterizedTypeExpr::hash() const {
    std::vector<size_t> hashes;
    hashes.push_back(std::hash<std::type_index>{}(std::type_index(typeid(*this))));
    hashes.push_back(std::hash<clobber::TypeExpr::Type>{}(this->type_kind));

    hashes.push_back(this->less_than_token.hash());
    hashes.push_back(this->greater_than_token.hash());
    hashes.push_back(this->type_expr->hash());

    for (const auto &param_expr : param_values) {
        hashes.push_back(param_expr->hash());
    }

    return utils::combine_hashes(hashes);
}

std::unique_ptr<clobber::Expr>
clobber::ParameterizedTypeExpr::clone() const {
    throw 0;
}

std::unique_ptr<clobber::TypeExpr>
clobber::ParameterizedTypeExpr::clone_nowrap() const {
    throw 0;
}

clobber::ParenthesizedExpr::ParenthesizedExpr(Expr::Type type, const Token &open_paren_token, const Token &close_paren_token)
    : Expr(type)
    , open_paren_token(open_paren_token)
    , close_paren_token(close_paren_token) {}

clobber::ParenthesizedExpr::ParenthesizedExpr(const ParenthesizedExpr &other)
    : Expr(other.type)
    , open_paren_token(other.open_paren_token)
    , close_paren_token(other.close_paren_token) {}

clobber::NumLiteralExpr::NumLiteralExpr(const Token &token)
    : Expr(Expr::Type::NumericLiteralExpr)
    , token(token) {}

clobber::NumLiteralExpr::NumLiteralExpr(const NumLiteralExpr &other)
    : Expr(other.type)
    , token(other.token) {}

clobber::Span
clobber::NumLiteralExpr::span() const {
    return this->token.span();
}

size_t
clobber::NumLiteralExpr::hash() const {
    std::vector<size_t> hashes;
    hashes.push_back(std::hash<std::type_index>{}(std::type_index(typeid(*this))));
    hashes.push_back(this->token.hash());
    return utils::combine_hashes(hashes);
}

std::unique_ptr<clobber::Expr>
clobber::NumLiteralExpr::clone() const {
    throw 0;
}

clobber::StringLiteralExpr::StringLiteralExpr(const std::string &value, const Token &token)
    : Expr(Expr::Type::StringLiteralExpr)
    , value(value)
    , token(token) {}

clobber::StringLiteralExpr::StringLiteralExpr(const StringLiteralExpr &other)
    : Expr(other.type)
    , value(other.value)
    , token(other.token) {}

size_t
clobber::StringLiteralExpr::hash() const {
    std::vector<size_t> hashes;
    hashes.push_back(std::hash<std::type_index>{}(std::type_index(typeid(*this))));
    hashes.push_back(std::hash<std::string>{}(this->value));
    hashes.push_back(this->token.hash());
    return utils::combine_hashes(hashes);
}

clobber::Span
clobber::StringLiteralExpr::span() const {
    throw 0;
}

std::unique_ptr<clobber::Expr>
clobber::StringLiteralExpr::clone() const {
    throw 0;
}

clobber::CharLiteralExpr::CharLiteralExpr(const std::string &value, const Token &token)
    : Expr(Expr::Type::CharLiteralExpr)
    , value(value)
    , token(token) {}

clobber::CharLiteralExpr::CharLiteralExpr(const CharLiteralExpr &other)
    : Expr(other.type)
    , value(other.value)
    , token(other.token) {}

clobber::Span
clobber::CharLiteralExpr::span() const {
    throw 0;
}

size_t
clobber::CharLiteralExpr::hash() const {
    std::vector<size_t> hashes;
    hashes.push_back(std::hash<std::type_index>{}(std::type_index(typeid(*this))));
    hashes.push_back(std::hash<std::string>{}(this->value));
    hashes.push_back(this->token.hash());
    return utils::combine_hashes(hashes);
}

std::unique_ptr<clobber::Expr>
clobber::CharLiteralExpr::clone() const {
    throw 0;
}

clobber::IdentifierExpr::IdentifierExpr(const std::string &name, const Token &token)
    : Expr(Expr::Type::IdentifierExpr)
    , name(name)
    , token(token) {}

clobber::IdentifierExpr::IdentifierExpr(const IdentifierExpr &other)
    : Expr(other.type)
    , name(other.name)
    , token(other.token) {}

clobber::Span
clobber::IdentifierExpr::span() const {
    throw 0;
}

size_t
clobber::IdentifierExpr::hash() const {
    std::vector<size_t> hashes;
    hashes.push_back(std::hash<std::type_index>{}(std::type_index(typeid(*this))));
    hashes.push_back(std::hash<std::string>{}(this->name));
    hashes.push_back(this->token.hash());
    return utils::combine_hashes(hashes);
}

std::unique_ptr<clobber::Expr>
clobber::IdentifierExpr::clone() const {
    throw 0;
}

std::unique_ptr<clobber::IdentifierExpr>
clobber::IdentifierExpr::clone_nowrap() {
    return std::make_unique<IdentifierExpr>(this->name, this->token);
}

clobber::BindingVectorExpr::BindingVectorExpr(const Token &open_bracket_token, std::vector<std::unique_ptr<IdentifierExpr>> &&identifiers,
                                              std::vector<std::unique_ptr<Expr>> &&exprs, const Token &close_bracket_token,
                                              size_t num_bindings)
    : open_bracket_token(open_bracket_token)
    , identifiers(std::move(identifiers))
    , exprs(std::move(exprs))
    , close_bracket_token(close_bracket_token)
    , num_bindings(num_bindings) {}

clobber::BindingVectorExpr::BindingVectorExpr(const BindingVectorExpr &other)
    : open_bracket_token(other.open_bracket_token)
    , identifiers(utils::deepcopy_identifiers(other.identifiers))
    , exprs(utils::deepcopy_exprs(other.exprs))
    , close_bracket_token(other.close_bracket_token)
    , num_bindings(other.num_bindings) {}

clobber::Span
clobber::BindingVectorExpr::span() const {
    throw 0;
}

std::unique_ptr<clobber::BindingVectorExpr>
clobber::BindingVectorExpr::clone_nowrap() const {
    throw 0;
}

clobber::ParameterVectorExpr::ParameterVectorExpr(const Token &open_bracket_token,
                                                  std::vector<std::unique_ptr<IdentifierExpr>> &&identifiers,
                                                  const Token &close_bracket_token)
    : open_bracket_token(open_bracket_token)
    , identifiers(std::move(identifiers))
    , close_bracket_token(close_bracket_token) {}

clobber::ParameterVectorExpr::ParameterVectorExpr(const ParameterVectorExpr &other)
    : open_bracket_token(other.open_bracket_token)
    , identifiers(utils::deepcopy_identifiers(other.identifiers))
    , close_bracket_token(other.close_bracket_token) {}

clobber::Span
clobber::ParameterVectorExpr::span() const {
    throw 0;
}

std::unique_ptr<clobber::ParameterVectorExpr>
clobber::ParameterVectorExpr::clone_nowrap() const {
    throw 0;
}

clobber::LetExpr::LetExpr(const Token &open_paren_token, const Token &let_token, std::unique_ptr<BindingVectorExpr> binding_vector_expr,
                          std::vector<std::unique_ptr<Expr>> &&body_exprs, const Token &close_paren_token)
    : ParenthesizedExpr(Expr::Type::LetExpr, open_paren_token, close_paren_token)
    , let_token(let_token)
    , binding_vector_expr(std::move(binding_vector_expr))
    , body_exprs(std::move(body_exprs)) {}

clobber::LetExpr::LetExpr(const LetExpr &other)
    : ParenthesizedExpr(other.type, other.open_paren_token, other.close_paren_token)
    , let_token(other.let_token)
    , binding_vector_expr(std::move(other.binding_vector_expr->clone_nowrap()))
    , body_exprs(utils::deepcopy_exprs(other.body_exprs)) {}

clobber::Span
clobber::LetExpr::span() const {
    throw 0;
}

size_t
clobber::LetExpr::hash() const {
    std::vector<size_t> hashes;
    hashes.push_back(std::hash<std::type_index>{}(std::type_index(typeid(*this))));
    hashes.push_back(this->let_token.hash());
    hashes.push_back(this->open_paren_token.hash());
    hashes.push_back(this->close_paren_token.hash());

    for (const auto &body_epxr : this->body_exprs) {
        hashes.push_back(body_epxr->hash());
    }

    return utils::combine_hashes(hashes);
}

std::unique_ptr<clobber::Expr>
clobber::LetExpr::clone() const {
    throw 0;
}

clobber::FnExpr::FnExpr(const Token &open_paren_token, const Token &fn_token, std::unique_ptr<ParameterVectorExpr> parameter_vector_expr,
                        std::vector<std::unique_ptr<Expr>> &&body_exprs, const Token &close_paren_token)
    : ParenthesizedExpr(Expr::Type::FnExpr, open_paren_token, close_paren_token)
    , fn_token(fn_token)
    , parameter_vector_expr(std::move(parameter_vector_expr))
    , body_exprs(std::move(body_exprs)) {}

clobber::FnExpr::FnExpr(const FnExpr &other)
    : ParenthesizedExpr(other.type, other.open_paren_token, other.close_paren_token)
    , fn_token(other.fn_token)
    , parameter_vector_expr(std::move(other.parameter_vector_expr->clone_nowrap()))
    , body_exprs(utils::deepcopy_exprs(other.body_exprs)) {}

size_t
clobber::FnExpr::hash() const {
    std::vector<size_t> hashes;
    hashes.push_back(std::hash<std::type_index>{}(std::type_index(typeid(*this))));
    hashes.push_back(this->fn_token.hash());
    hashes.push_back(this->open_paren_token.hash());
    hashes.push_back(this->close_paren_token.hash());

    for (const auto &body_epxr : this->body_exprs) {
        hashes.push_back(body_epxr->hash());
    }

    return utils::combine_hashes(hashes);
}

clobber::Span
clobber::FnExpr::span() const {
    throw 0;
}

std::unique_ptr<clobber::Expr>
clobber::FnExpr::clone() const {
    throw 0;
}

clobber::DefExpr::DefExpr(const Token &open_paren_token, const Token &def_token, std::unique_ptr<IdentifierExpr> identifier,
                          std::unique_ptr<Expr> value, const Token &close_paren_token)
    : ParenthesizedExpr(Expr::Type::DefExpr, open_paren_token, close_paren_token)
    , def_token(def_token)
    , identifier(std::move(identifier))
    , value(std::move(value)) {}

clobber::DefExpr::DefExpr(const DefExpr &other)
    : ParenthesizedExpr(other.type, other.open_paren_token, other.close_paren_token)
    , def_token(other.def_token)
    , identifier(other.identifier->clone_nowrap())
    , value(other.value->clone()) {}

clobber::Span
clobber::DefExpr::span() const {
    throw 0;
}

size_t
clobber::DefExpr::hash() const {
    std::vector<size_t> hashes;
    hashes.push_back(std::hash<std::type_index>{}(std::type_index(typeid(*this))));
    hashes.push_back(this->def_token.hash());
    hashes.push_back(this->open_paren_token.hash());
    hashes.push_back(this->close_paren_token.hash());
    hashes.push_back(this->identifier->hash());
    hashes.push_back(this->value->hash());
    return utils::combine_hashes(hashes);
}

std::unique_ptr<clobber::Expr>
clobber::DefExpr::clone() const {
    throw 0;
}

clobber::DoExpr::DoExpr(const Token &open_paren_token, const Token &do_token, std::vector<std::unique_ptr<Expr>> &&body_exprs,
                        const Token &close_paren_token)
    : ParenthesizedExpr(Expr::Type::DoExpr, open_paren_token, close_paren_token)
    , do_token(do_token)
    , body_exprs(std::move(body_exprs)) {}

clobber::DoExpr::DoExpr(const DoExpr &other)
    : ParenthesizedExpr(other.type, other.open_paren_token, other.close_paren_token)
    , do_token(other.do_token)
    , body_exprs(utils::deepcopy_exprs(other.body_exprs)) {}

clobber::Span
clobber::DoExpr::span() const {
    throw 0;
}

size_t
clobber::DoExpr::hash() const {
    std::vector<size_t> hashes;
    hashes.push_back(std::hash<std::type_index>{}(std::type_index(typeid(*this))));
    hashes.push_back(this->do_token.hash());
    hashes.push_back(this->open_paren_token.hash());
    hashes.push_back(this->close_paren_token.hash());

    for (const auto &body_epxr : this->body_exprs) {
        hashes.push_back(body_epxr->hash());
    }

    return utils::combine_hashes(hashes);
}

std::unique_ptr<clobber::Expr>
clobber::DoExpr::clone() const {
    throw 0;
}

clobber::CallExpr::CallExpr(const Token &open_paren_token, std::unique_ptr<clobber::Expr> operator_expr,
                            std::vector<std::unique_ptr<Expr>> &&arguments, const Token &close_paren_token)
    : ParenthesizedExpr(Expr::Type::CallExpr, open_paren_token, close_paren_token)
    , operator_expr(std::move(operator_expr))
    , arguments(std::move(arguments)) {}

clobber::CallExpr::CallExpr(const CallExpr &other)
    : ParenthesizedExpr(other.type, other.open_paren_token, other.close_paren_token)
    , operator_expr(other.operator_expr->clone())
    , arguments(utils::deepcopy_exprs(other.arguments)) {}

clobber::Span
clobber::CallExpr::span() const {
    throw 0;
}

size_t
clobber::CallExpr::hash() const {
    std::vector<size_t> hashes;
    hashes.push_back(std::hash<std::type_index>{}(std::type_index(typeid(*this))));
    hashes.push_back(this->operator_expr->hash());
    hashes.push_back(this->open_paren_token.hash());
    hashes.push_back(this->close_paren_token.hash());

    for (const auto &arg_expr : this->arguments) {
        hashes.push_back(arg_expr->hash());
    }

    return utils::combine_hashes(hashes);
}

std::unique_ptr<clobber::Expr>
clobber::CallExpr::clone() const {
    throw 0;
}

clobber::accel::AccelExpr::AccelExpr(const Token &open_paren_token, const Token &accel_token,
                                     std::unique_ptr<BindingVectorExpr> binding_vector_expr,
                                     std::vector<std::unique_ptr<Expr>> &&body_exprs, const Token &close_paren_token)
    : ParenthesizedExpr(Expr::Type::AccelExpr, open_paren_token, close_paren_token)
    , accel_token(accel_token)
    , binding_vector_expr(std::move(binding_vector_expr))
    , body_exprs(std::move(body_exprs)) {}

clobber::accel::AccelExpr::AccelExpr(const AccelExpr &other)
    : ParenthesizedExpr(other.type, other.open_paren_token, other.close_paren_token)
    , accel_token(other.accel_token)
    , binding_vector_expr(std::move(other.binding_vector_expr->clone_nowrap()))
    , body_exprs(utils::deepcopy_exprs(other.body_exprs)) {}

clobber::Span
clobber::accel::AccelExpr::span() const {
    throw 0;
}

size_t
clobber::accel::AccelExpr::hash() const {
    std::vector<size_t> hashes;
    hashes.push_back(std::hash<std::type_index>{}(std::type_index(typeid(*this))));
    hashes.push_back(this->accel_token.hash());
    hashes.push_back(this->open_paren_token.hash());
    hashes.push_back(this->close_paren_token.hash());

    for (const auto &body_expr : this->body_exprs) {
        hashes.push_back(body_expr->hash());
    }

    return utils::combine_hashes(hashes);
}

std::unique_ptr<clobber::Expr>
clobber::accel::AccelExpr::clone() const {
    throw 0;
}

clobber::accel::MatMulExpr::MatMulExpr(const Token &open_paren_token, const Token &mat_mul_token, std::unique_ptr<Expr> fst_operand,
                                       std::unique_ptr<Expr> snd_operand, const Token &close_paren_token)
    : ParenthesizedExpr(Expr::Type::MatMulExpr, open_paren_token, close_paren_token)
    , mat_mul_token(mat_mul_token)
    , fst_operand(std::move(fst_operand))
    , snd_operand(std::move(snd_operand)) {}

clobber::accel::MatMulExpr::MatMulExpr(const MatMulExpr &other)
    : ParenthesizedExpr(other.type, other.open_paren_token, other.close_paren_token)
    , mat_mul_token(other.mat_mul_token)
    , fst_operand(other.fst_operand->clone())
    , snd_operand(other.snd_operand->clone()) {}

size_t
clobber::accel::MatMulExpr::hash() const {
    std::vector<size_t> hashes;
    hashes.push_back(std::hash<std::type_index>{}(std::type_index(typeid(*this))));
    hashes.push_back(this->mat_mul_token.hash());
    hashes.push_back(this->open_paren_token.hash());
    hashes.push_back(this->close_paren_token.hash());
    hashes.push_back(this->fst_operand->hash());
    hashes.push_back(this->snd_operand->hash());
    return utils::combine_hashes(hashes);
}

clobber::Span
clobber::accel::MatMulExpr::span() const {
    throw 0;
}

std::unique_ptr<clobber::Expr>
clobber::accel::MatMulExpr::clone() const {
    throw 0;
}

clobber::accel::RelUExpr::RelUExpr(const Token &open_paren_token, const Token &relu_token, std::unique_ptr<Expr> operand,
                                   const Token &close_paren_token)
    : ParenthesizedExpr(Expr::Type::RelUExpr, open_paren_token, close_paren_token)
    , relu_token(relu_token)
    , operand(std::move(operand)) {}

clobber::accel::RelUExpr::RelUExpr(const RelUExpr &other)
    : ParenthesizedExpr(other.type, other.open_paren_token, other.close_paren_token)
    , relu_token(other.relu_token)
    , operand(other.operand->clone()) {}

clobber::Span
clobber::accel::RelUExpr::span() const {
    throw 0;
}

size_t
clobber::accel::RelUExpr::hash() const {
    std::vector<size_t> hashes;
    hashes.push_back(std::hash<std::type_index>{}(std::type_index(typeid(*this))));
    hashes.push_back(this->relu_token.hash());
    hashes.push_back(this->open_paren_token.hash());
    hashes.push_back(this->close_paren_token.hash());
    hashes.push_back(this->operand->hash());
    return utils::combine_hashes(hashes);
}

std::unique_ptr<clobber::Expr>
clobber::accel::RelUExpr::clone() const {
    throw 0;
}

clobber::CompilationUnit::CompilationUnit(const std::string &source_text, std::vector<std::unique_ptr<Expr>> &&exprs,
                                          const std::vector<clobber::Diagnostic> &diagnostics)
    : source_text(source_text)
    , exprs(std::move(exprs))
    , diagnostics(diagnostics) {}

clobber::CompilationUnit::CompilationUnit(const CompilationUnit &other)
    : source_text(other.source_text)
    , exprs(utils::deepcopy_exprs(other.exprs))
    , diagnostics(other.diagnostics) {}

clobber::Expr *
clobber::AstWalker::on_expr(clobber::Expr *e) {
    switch (e->type) {
    case clobber::Expr::Type::LetExpr: {
        ParenthesizedExpr *pe = static_cast<ParenthesizedExpr *>(e);
        return on_paren_expr(pe);
    }
    case clobber::Expr::Type::NumericLiteralExpr: {
        NumLiteralExpr *nle = static_cast<NumLiteralExpr *>(e);
        return on_num_literal_expr(nle);
    }
    case clobber::Expr::Type::StringLiteralExpr: {
        StringLiteralExpr *sle = static_cast<StringLiteralExpr *>(e);
        return on_string_literal_expr(sle);
    }
    case clobber::Expr::Type::CharLiteralExpr: {
        CharLiteralExpr *cle = static_cast<CharLiteralExpr *>(e);
        return on_char_literal_expr(cle);
    }
    case clobber::Expr::Type::IdentifierExpr: {
        IdentifierExpr *ie = static_cast<IdentifierExpr *>(e);
        return on_identifier_expr(ie);
    }
    case clobber::Expr::Type::FnExpr: {
        FnExpr *fe = static_cast<FnExpr *>(e);
        return on_fn_expr(fe);
    }
    case clobber::Expr::Type::DefExpr: {
        DefExpr *de = static_cast<DefExpr *>(e);
        return on_def_expr(de);
    }
    case clobber::Expr::Type::DoExpr: {
        DoExpr *de = static_cast<DoExpr *>(e);
        return on_do_expr(de);
    }
    case clobber::Expr::Type::CallExpr: {
        CallExpr *ce = static_cast<CallExpr *>(e);
        return on_call_expr(ce);
    }
    case clobber::Expr::Type::AccelExpr: {
        accel::AccelExpr *ae = static_cast<accel::AccelExpr *>(e);
        return on_accel_expr(ae);
    }
    case clobber::Expr::Type::MatMulExpr: {
        accel::MatMulExpr *mme = static_cast<accel::MatMulExpr *>(e);
        return on_mat_mul_expr(mme);
    }
    case clobber::Expr::Type::RelUExpr: {
        accel::RelUExpr *re = static_cast<accel::RelUExpr *>(e);
        return on_relu_expr(re);
    }
    default: {
        return nullptr;
    }
    }
}

clobber::NumLiteralExpr *
clobber::AstWalker::on_num_literal_expr(clobber::NumLiteralExpr *nle) {
    return nle;
}

clobber::StringLiteralExpr *
clobber::AstWalker::on_string_literal_expr(clobber::StringLiteralExpr *sle) {
    return sle;
}

clobber::CharLiteralExpr *
clobber::AstWalker::on_char_literal_expr(clobber::CharLiteralExpr *cle) {
    return cle;
}

clobber::IdentifierExpr *
clobber::AstWalker::on_identifier_expr(clobber::IdentifierExpr *ie) {
    return ie;
}

clobber::ParenthesizedExpr *
clobber::AstWalker::on_paren_expr(clobber::ParenthesizedExpr *pe) {
    switch (pe->type) {
    case clobber::Expr::Type::LetExpr: {
        LetExpr *le = static_cast<LetExpr *>(pe);
        return on_let_expr(le);
    }
    case clobber::Expr::Type::FnExpr: {
        FnExpr *fe = static_cast<FnExpr *>(pe);
        return on_fn_expr(fe);
    }
    case clobber::Expr::Type::DefExpr: {
        DefExpr *de = static_cast<DefExpr *>(pe);
        return on_def_expr(de);
    }
    case clobber::Expr::Type::DoExpr: {
        DoExpr *de = static_cast<DoExpr *>(pe);
        return on_do_expr(de);
    }
    case clobber::Expr::Type::CallExpr: {
        CallExpr *ce = static_cast<CallExpr *>(pe);
        return on_call_expr(ce);
    }
    case clobber::Expr::Type::AccelExpr: {
        accel::AccelExpr *ae = static_cast<accel::AccelExpr *>(pe);
        return on_accel_expr(ae);
    }
    case clobber::Expr::Type::MatMulExpr: {
        accel::MatMulExpr *mme = static_cast<accel::MatMulExpr *>(pe);
        return on_mat_mul_expr(mme);
    }
    case clobber::Expr::Type::RelUExpr: {
        accel::RelUExpr *re = static_cast<accel::RelUExpr *>(pe);
        return on_relu_expr(re);
    }
    default: {
        return nullptr;
    }
    }
}

clobber::BindingVectorExpr *
clobber::AstWalker::on_binding_vector_expr(clobber::BindingVectorExpr *bve) {
    for (size_t i = 0; i < bve->num_bindings; i++) {
        {
            auto old_ptr = bve->identifiers[i].get();
            auto new_ptr = on_identifier_expr(old_ptr);
            if (old_ptr != new_ptr) {
                bve->identifiers[i].reset(new_ptr);
            }
        }

        {
            auto old_ptr = bve->exprs[i].get();
            auto new_ptr = on_expr(old_ptr);
            if (old_ptr != new_ptr) {
                bve->exprs[i].reset(new_ptr);
            }
        }
    }

    return bve;
}

clobber::ParameterVectorExpr *
clobber::AstWalker::on_parameter_vector_expr(clobber::ParameterVectorExpr *pe) {
    for (std::unique_ptr<clobber::IdentifierExpr> &identifier_uptr : pe->identifiers) {
        auto old_ptr = identifier_uptr.get();
        auto new_ptr = on_identifier_expr(old_ptr);
        if (old_ptr != new_ptr) {
            identifier_uptr.reset(new_ptr);
        }
    }

    return pe;
}

clobber::LetExpr *
clobber::AstWalker::on_let_expr(clobber::LetExpr *le) {
    { // local scope so I can use variables 'old_ptr' and 'new_ptr' names, too lazy
        auto old_ptr = le->binding_vector_expr.get();
        auto new_ptr = on_binding_vector_expr(old_ptr);
        if (old_ptr != new_ptr) {
            le->binding_vector_expr.reset(new_ptr);
        }
    }

    for (std::unique_ptr<clobber::Expr> &expr_uptr : le->body_exprs) {
        auto old_ptr = expr_uptr.get();
        auto new_ptr = on_expr(old_ptr);
        if (old_ptr != new_ptr) {
            expr_uptr.reset(new_ptr);
        }
    }

    return le;
}

clobber::FnExpr *
clobber::AstWalker::on_fn_expr(clobber::FnExpr *fe) {
    {
        auto old_ptr = fe->parameter_vector_expr.get();
        auto new_ptr = on_parameter_vector_expr(old_ptr);
        if (old_ptr != new_ptr) {
            fe->parameter_vector_expr.reset(new_ptr);
        }
    }

    for (std::unique_ptr<clobber::Expr> &expr_uptr : fe->body_exprs) {
        auto old_ptr = expr_uptr.get();
        auto new_ptr = on_expr(old_ptr);
        if (old_ptr != new_ptr) {
            expr_uptr.reset(new_ptr);
        }
    }

    return fe;
}

clobber::DefExpr *
clobber::AstWalker::on_def_expr(clobber::DefExpr *de) {
    {
        auto old_ptr = de->identifier.get();
        auto new_ptr = on_identifier_expr(old_ptr);
        if (old_ptr != new_ptr) {
            de->identifier.reset(new_ptr);
        }
    }

    {
        auto old_ptr = de->value.get();
        auto new_ptr = on_expr(old_ptr);
        if (old_ptr != new_ptr) {
            de->value.reset(new_ptr);
        }
    }

    return de;
}

clobber::DoExpr *
clobber::AstWalker::on_do_expr(clobber::DoExpr *de) {
    for (std::unique_ptr<clobber::Expr> &expr_uptr : de->body_exprs) {
        auto old_ptr = expr_uptr.get();
        auto new_ptr = on_expr(old_ptr);
        if (old_ptr != new_ptr) {
            expr_uptr.reset(new_ptr);
        }
    }

    return de;
}

clobber::CallExpr *
clobber::AstWalker::on_call_expr(clobber::CallExpr *ce) {
    for (std::unique_ptr<clobber::Expr> &expr_uptr : ce->arguments) {
        auto old_ptr = expr_uptr.get();
        auto new_ptr = on_expr(old_ptr);
        if (old_ptr != new_ptr) {
            expr_uptr.reset(new_ptr);
        }
    }

    return ce;
}

clobber::accel::AccelExpr *
clobber::AstWalker::on_accel_expr(clobber::accel::AccelExpr *ae) {
    {
        auto old_ptr = ae->binding_vector_expr.get();
        auto new_ptr = on_binding_vector_expr(old_ptr);
        if (old_ptr != new_ptr) {
            ae->binding_vector_expr.reset(new_ptr);
        }
    }

    for (std::unique_ptr<clobber::Expr> &expr_uptr : ae->body_exprs) {
        auto old_ptr = expr_uptr.get();
        auto new_ptr = on_expr(old_ptr);
        if (old_ptr != new_ptr) {
            expr_uptr.reset(new_ptr);
        }
    }
    return ae;
}

clobber::accel::MatMulExpr *
clobber::AstWalker::on_mat_mul_expr(clobber::accel::MatMulExpr *mme) {
    {
        auto old_ptr = mme->fst_operand.get();
        auto new_ptr = on_expr(old_ptr);
        if (old_ptr != new_ptr) {
            mme->fst_operand.reset(new_ptr);
        }
    }

    {
        auto old_ptr = mme->snd_operand.get();
        auto new_ptr = on_expr(old_ptr);
        if (old_ptr != new_ptr) {
            mme->snd_operand.reset(new_ptr);
        }
    }

    return mme;
}

clobber::accel::RelUExpr *
clobber::AstWalker::on_relu_expr(clobber::accel::RelUExpr *re) {
    {
        auto old_ptr = re->operand.get();
        auto new_ptr = on_expr(old_ptr);
        if (old_ptr != new_ptr) {
            re->operand.reset(new_ptr);
        }
    }

    return re;
}
