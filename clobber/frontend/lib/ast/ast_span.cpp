// separate .cpp file specifically for the `span` method of ast node.

#include <cstddef>
#include <typeindex>

#include "clobber/ast/ast.hpp"
#include "clobber/internal/utils.hpp"

clobber::Span
clobber::BuiltinTypeExpr::span() const {
    size_t start = this->caret_token.span.start;
    size_t len   = this->caret_token.span.length;
    len += this->type_keyword_token.span.length;
    return Span{start, len};
}

clobber::Span
clobber::UserDefinedTypeExpr::span() const {
    size_t start = this->caret_token.span.start;
    size_t len   = this->caret_token.span.length;
    len += this->identifier_token.span.length;
    return Span{start, len};
}

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

clobber::Span
clobber::NumLiteralExpr::span() const {
    return this->token.full_span;
}

clobber::Span
clobber::StringLiteralExpr::span() const {
    return this->token.full_span;
}

clobber::Span
clobber::CharLiteralExpr::span() const {
    return this->token.full_span;
}

clobber::Span
clobber::IdentifierExpr::span() const {
    return this->token.full_span;
}

clobber::Span
clobber::BindingVectorExpr::span() const {
    size_t start = this->open_bracket_token.full_span.start;
    size_t len   = this->open_bracket_token.full_span.length;

    for (size_t i = 0; i < this->num_bindings; i++) {
        len += this->identifiers[i]->span().length;
        len += this->exprs[i]->span().length;
    }

    len += this->close_bracket_token.span.length;
    return Span{start, len};
}

clobber::Span
clobber::ParameterVectorExpr::span() const {
    size_t start = this->open_bracket_token.full_span.start;
    size_t len   = this->open_bracket_token.full_span.length;

    for (const auto &identifier : this->identifiers) {
        len += identifier->span().length;
    }

    len += this->close_bracket_token.span.length;
    return Span{start, len};
}

clobber::Span
clobber::LetExpr::span() const {
    size_t start = this->open_paren_token.full_span.start;
    size_t len   = this->open_paren_token.full_span.length;

    len += this->let_token.span.length;
    len += this->binding_vector_expr->span().length;
    for (const auto &expr : this->body_exprs) {
        len += expr->span().length;
    }

    len += this->close_paren_token.span.length;
    return Span{start, len};
}

clobber::Span
clobber::FnExpr::span() const {
    size_t start = this->open_paren_token.full_span.start;
    size_t len   = this->open_paren_token.full_span.length;

    len += this->fn_token.span.length;
    len += this->parameter_vector_expr->span().length;
    for (const auto &expr : this->body_exprs) {
        len += expr->span().length;
    }

    len += this->close_paren_token.span.length;
    return Span{start, len};
}

clobber::Span
clobber::DefExpr::span() const {
    size_t start = this->open_paren_token.full_span.start;
    size_t len   = this->open_paren_token.full_span.length;
    len += this->def_token.span.length;
    len += this->identifier->span().length;
    len += this->value->span().length;
    len += this->close_paren_token.span.length;
    return Span{start, len};
}

clobber::Span
clobber::DoExpr::span() const {
    size_t start = this->open_paren_token.full_span.start;
    size_t len   = this->open_paren_token.full_span.length;

    len += this->do_token.span.length;
    for (const auto &expr : this->body_exprs) {
        len += expr->span().length;
    }

    len += this->close_paren_token.span.length;
    return Span{start, len};
}

clobber::Span
clobber::CallExpr::span() const {
    size_t start = this->open_paren_token.full_span.start;
    size_t len   = this->open_paren_token.full_span.length;

    len += this->operator_expr->span().length;
    for (const auto &argument : this->arguments) {
        len += argument->span().length;
    }

    len += this->close_paren_token.span.length;
    return Span{start, len};
}
