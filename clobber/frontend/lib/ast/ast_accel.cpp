#include <cstddef>
#include <typeindex>

#include "clobber/ast/ast.hpp"
#include "clobber/internal/utils.hpp"

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
    , body_exprs(deepcopy_exprs(other.body_exprs)) {}

clobber::Span
clobber::accel::AccelExpr::span() const {
    size_t start = this->open_paren_token.span.start;
    size_t len   = this->open_paren_token.span.length;

    len += this->accel_token.span.length;
    len += this->binding_vector_expr->span().length;
    for (const auto &expr : this->body_exprs) {
        len += expr->span().length;
    }

    len += this->close_paren_token.span.length;
    return Span{start, len};
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

    return combine_hashes(hashes);
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
    return combine_hashes(hashes);
}

clobber::Span
clobber::accel::MatMulExpr::span() const {
    size_t start = this->open_paren_token.span.start;
    size_t len   = this->open_paren_token.span.length;
    len += this->mat_mul_token.span.length;
    len += this->fst_operand->span().length;
    len += this->snd_operand->span().length;
    len += this->close_paren_token.span.length;
    return Span{start, len};
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
    size_t start = this->open_paren_token.span.start;
    size_t len   = this->open_paren_token.span.length;
    len += this->relu_token.span.length;
    len += this->operand->span().length;
    len += this->close_paren_token.span.length;
    return Span{start, len};
}

size_t
clobber::accel::RelUExpr::hash() const {
    std::vector<size_t> hashes;
    hashes.push_back(std::hash<std::type_index>{}(std::type_index(typeid(*this))));
    hashes.push_back(this->relu_token.hash());
    hashes.push_back(this->open_paren_token.hash());
    hashes.push_back(this->close_paren_token.hash());
    hashes.push_back(this->operand->hash());
    return combine_hashes(hashes);
}

std::unique_ptr<clobber::Expr>
clobber::accel::RelUExpr::clone() const {
    throw 0;
}