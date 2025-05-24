// separate .cpp file specifically for the `span` method of ast node.

#include <cstddef>
#include <typeindex>

#include "clobber/ast/ast.hpp"
#include "clobber/internal/utils.hpp"

clobber::Span
clobber::VectorExpr::span() const {
    size_t start = this->open_bracket_token.span.start;
    size_t end   = this->close_bracket_token.span.start + this->close_bracket_token.span.length;
    return Span{start, end - start};
}

clobber::Span
clobber::BuiltinTypeExpr::span() const {
    size_t start = this->caret_token.span.start;
    size_t end   = this->type_keyword_token.span.start + this->type_keyword_token.span.length;
    return Span{start, end - start};
}

clobber::Span
clobber::UserDefinedTypeExpr::span() const {
    size_t start = this->caret_token.span.start;
    size_t end   = this->identifier_token.span.start + this->identifier_token.span.length;
    return Span{start, end - start};
}

clobber::Span
clobber::ParameterizedTypeExpr::span() const {
    size_t start = this->type_expr->span().start;
    size_t end   = this->greater_than_token.span.start + this->greater_than_token.span.length;
    return Span{start, end - start};
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
clobber::KeywordLiteralExpr::span() const {
    return this->token.full_span;
}

clobber::Span
clobber::BindingVectorExpr::span() const {
    size_t start = this->open_bracket_token.full_span.start;
    size_t end   = this->close_bracket_token.span.start + this->close_bracket_token.span.length;
    return Span{start, end - start};
}

clobber::Span
clobber::ParameterVectorExpr::span() const {
    size_t start = this->open_bracket_token.full_span.start;
    size_t end   = this->close_bracket_token.span.start + this->close_bracket_token.span.length;
    return Span{start, end - start};
}

clobber::Span
clobber::LetExpr::span() const {
    size_t start = this->open_paren_token.full_span.start;
    size_t end   = this->close_paren_token.span.start + this->close_paren_token.span.length;
    return Span{start, end - start};
}

clobber::Span
clobber::FnExpr::span() const {
    size_t start = this->open_paren_token.full_span.start;
    size_t end   = this->close_paren_token.span.start + this->close_paren_token.span.length;
    return Span{start, end - start};
}

clobber::Span
clobber::DefExpr::span() const {
    size_t start = this->open_paren_token.full_span.start;
    size_t end   = this->close_paren_token.span.start + this->close_paren_token.span.length;
    return Span{start, end - start};
}

clobber::Span
clobber::DoExpr::span() const {
    size_t start = this->open_paren_token.full_span.start;
    size_t end   = this->close_paren_token.span.start + this->close_paren_token.span.length;
    return Span{start, end - start};
}

clobber::Span
clobber::CallExpr::span() const {
    size_t start = this->open_paren_token.full_span.start;
    size_t end   = this->close_paren_token.span.start + this->close_paren_token.span.length;
    return Span{start, end - start};
}

clobber::Span
clobber::accel::AccelExpr::span() const {
    size_t start = this->open_paren_token.span.start;
    size_t end   = this->close_paren_token.span.start + this->close_paren_token.span.length;
    return Span{start, end - start};
}

clobber::Span
clobber::accel::TOSAOpExpr::span() const {
    size_t start = this->open_paren_token.span.start;
    size_t end   = this->close_paren_token.span.start + this->close_paren_token.span.length;
    return Span{start, end - start};
}

clobber::Span
clobber::accel::TensorExpr::span() const {
    size_t start = this->open_paren_token.span.start;
    size_t end   = this->close_paren_token.span.start + this->close_paren_token.span.length;
    return Span{start, end - start};
}