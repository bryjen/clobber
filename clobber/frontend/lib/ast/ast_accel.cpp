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

clobber::accel::TOSAOpExpr::TOSAOpExpr(const Token &open_paren_token, const Token &op_token, std::vector<std::unique_ptr<Expr>> &&arguments,
                                       const Token &close_paren_token)
    : ParenthesizedExpr(Expr::Type::TosaOpExpr, open_paren_token, close_paren_token)
    , op_token(op_token)
    , arguments(std::move(arguments)) {}

size_t
clobber::accel::TOSAOpExpr::hash() const {
    std::vector<size_t> hashes;
    hashes.push_back(std::hash<std::type_index>{}(std::type_index(typeid(*this))));
    hashes.push_back(this->op_token.hash());
    hashes.push_back(this->open_paren_token.hash());
    hashes.push_back(this->close_paren_token.hash());

    for (const auto &argument : this->arguments) {
        hashes.push_back(argument->hash());
    }

    return combine_hashes(hashes);
}

clobber::accel::TensorExpr::TensorExpr(const Token &open_paren_token, const Token &tensor_token,
                                       std::vector<std::unique_ptr<Expr>> &&arguments, const Token &close_paren_token)
    : ParenthesizedExpr(Expr::Type::TensorExpr, open_paren_token, close_paren_token)
    , tensor_token(tensor_token)
    , arguments(std::move(arguments)) {}

size_t
clobber::accel::TensorExpr::hash() const {
    std::vector<size_t> hashes;
    hashes.push_back(std::hash<std::type_index>{}(std::type_index(typeid(*this))));
    hashes.push_back(this->tensor_token.hash());
    hashes.push_back(this->open_paren_token.hash());
    hashes.push_back(this->close_paren_token.hash());

    for (const auto &argument : this->arguments) {
        hashes.push_back(argument->hash());
    }

    return combine_hashes(hashes);
}