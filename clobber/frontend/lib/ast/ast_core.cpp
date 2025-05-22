#include <cstddef>
#include <typeindex>

#include "clobber/ast/ast.hpp"
#include "clobber/internal/utils.hpp"

size_t
clobber::Token::hash() const {
    std::vector<size_t> hashes;
    hashes.push_back(std::hash<std::type_index>{}(std::type_index(typeid(*this))));
    hashes.push_back(std::hash<Token::Type>{}(this->type));
    hashes.push_back(std::hash<size_t>{}(this->span.start));
    hashes.push_back(std::hash<size_t>{}(this->span.length));
    hashes.push_back(std::hash<size_t>{}(this->full_span.start));
    hashes.push_back(std::hash<size_t>{}(this->full_span.length));
    return combine_hashes(hashes);
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

size_t
clobber::BuiltinTypeExpr::hash() const {
    std::vector<size_t> hashes;
    hashes.push_back(std::hash<std::type_index>{}(std::type_index(typeid(*this))));
    hashes.push_back(std::hash<clobber::TypeExpr::Type>{}(this->type_kind));

    hashes.push_back(this->type_keyword_token.hash());
    return combine_hashes(hashes);
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

size_t
clobber::UserDefinedTypeExpr::hash() const {
    std::vector<size_t> hashes;
    hashes.push_back(std::hash<std::type_index>{}(std::type_index(typeid(*this))));
    hashes.push_back(std::hash<clobber::TypeExpr::Type>{}(this->type_kind));

    hashes.push_back(this->identifier_token.hash());
    return combine_hashes(hashes);
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
    , param_values(deepcopy_exprs(other.param_values))
    , commas(other.commas)
    , greater_than_token(other.greater_than_token) {}

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

    return combine_hashes(hashes);
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

size_t
clobber::NumLiteralExpr::hash() const {
    std::vector<size_t> hashes;
    hashes.push_back(std::hash<std::type_index>{}(std::type_index(typeid(*this))));
    hashes.push_back(this->token.hash());
    return combine_hashes(hashes);
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
    return combine_hashes(hashes);
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

size_t
clobber::CharLiteralExpr::hash() const {
    std::vector<size_t> hashes;
    hashes.push_back(std::hash<std::type_index>{}(std::type_index(typeid(*this))));
    hashes.push_back(std::hash<std::string>{}(this->value));
    hashes.push_back(this->token.hash());
    return combine_hashes(hashes);
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

size_t
clobber::IdentifierExpr::hash() const {
    std::vector<size_t> hashes;
    hashes.push_back(std::hash<std::type_index>{}(std::type_index(typeid(*this))));
    hashes.push_back(std::hash<std::string>{}(this->name));
    hashes.push_back(this->token.hash());
    return combine_hashes(hashes);
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
    , identifiers(deepcopy_identifiers(other.identifiers))
    , exprs(deepcopy_exprs(other.exprs))
    , close_bracket_token(other.close_bracket_token)
    , num_bindings(other.num_bindings) {}

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
    , identifiers(deepcopy_identifiers(other.identifiers))
    , close_bracket_token(other.close_bracket_token) {}

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
    , body_exprs(deepcopy_exprs(other.body_exprs)) {}

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

    return combine_hashes(hashes);
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
    , body_exprs(deepcopy_exprs(other.body_exprs)) {}

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

    return combine_hashes(hashes);
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

size_t
clobber::DefExpr::hash() const {
    std::vector<size_t> hashes;
    hashes.push_back(std::hash<std::type_index>{}(std::type_index(typeid(*this))));
    hashes.push_back(this->def_token.hash());
    hashes.push_back(this->open_paren_token.hash());
    hashes.push_back(this->close_paren_token.hash());
    hashes.push_back(this->identifier->hash());
    hashes.push_back(this->value->hash());
    return combine_hashes(hashes);
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
    , body_exprs(deepcopy_exprs(other.body_exprs)) {}

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

    return combine_hashes(hashes);
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
    , arguments(deepcopy_exprs(other.arguments)) {}

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

    return combine_hashes(hashes);
}

std::unique_ptr<clobber::Expr>
clobber::CallExpr::clone() const {
    throw 0;
}

clobber::CompilationUnit::CompilationUnit(const std::string &source_text, std::vector<std::unique_ptr<Expr>> &&exprs,
                                          const std::vector<clobber::Diagnostic> &diagnostics)
    : source_text(source_text)
    , exprs(std::move(exprs))
    , diagnostics(diagnostics) {}

clobber::CompilationUnit::CompilationUnit(const CompilationUnit &other)
    : source_text(other.source_text)
    , exprs(deepcopy_exprs(other.exprs))
    , diagnostics(other.diagnostics) {}
