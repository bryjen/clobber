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

    /*
    template <typename... Rest>
    inline std::size_t
    hash_combine(std::size_t first, std::size_t second, Rest... rest) {
        std::size_t combined = hash_combine(first, second);
        if constexpr (sizeof...(rest) == 0) {
            return combined;
        } else {
            return hash_combine(combined, rest...);
        }
    }
    */

    std::size_t
    combine_hashes(const std::vector<std::size_t> &hashes) {
        std::size_t seed = 0;
        for (std::size_t h : hashes) {
            seed ^= h + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
} // namespace utils

size_t
clobber::ClobberToken::hash() const {
    std::vector<size_t> hashes;
    hashes.push_back(std::hash<std::type_index>{}(std::type_index(typeid(*this))));
    hashes.push_back(std::hash<ClobberTokenType>{}(this->token_type));
    hashes.push_back(std::hash<size_t>{}(this->start));
    hashes.push_back(std::hash<size_t>{}(this->length));
    hashes.push_back(std::hash<size_t>{}(this->full_start));
    hashes.push_back(std::hash<size_t>{}(this->full_length));
    return utils::combine_hashes(hashes);
}

std::string
clobber::ClobberToken::ExtractText(const std::string &source_text) {
    return source_text.substr(this->start, this->length);
}

std::string
clobber::ClobberToken::ExtractFullText(const std::string &source_text) {
    return source_text.substr(this->full_start, this->full_length);
}

// OBSOLETE/DEPRECATED ?
bool
clobber::ClobberToken::AreEquivalent(const ClobberToken &token1, const ClobberToken &token2) {
    if (token1.token_type != token2.token_type) {
        return false;
    }

    return false;
}

clobber::Expr::Expr(Expr::Type type)
    : type(type) {}

clobber::ParenthesizedExpr::ParenthesizedExpr(Expr::Type type, const ClobberToken &open_paren_token, const ClobberToken &close_paren_token)
    : Expr(type)
    , open_paren_token(open_paren_token)
    , close_paren_token(close_paren_token) {}

clobber::ParenthesizedExpr::ParenthesizedExpr(const ParenthesizedExpr &other)
    : Expr(other.type)
    , open_paren_token(other.open_paren_token)
    , close_paren_token(other.close_paren_token) {}

clobber::NumLiteralExpr::NumLiteralExpr(const ClobberToken &token)
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
    return utils::combine_hashes(hashes);
}

std::unique_ptr<clobber::Expr>
clobber::NumLiteralExpr::clone() const {
    throw 0;
}

clobber::StringLiteralExpr::StringLiteralExpr(const std::string &value, const ClobberToken &token)
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

std::unique_ptr<clobber::Expr>
clobber::StringLiteralExpr::clone() const {
    throw 0;
}

clobber::CharLiteralExpr::CharLiteralExpr(const std::string &value, const ClobberToken &token)
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
    return utils::combine_hashes(hashes);
}

std::unique_ptr<clobber::Expr>
clobber::CharLiteralExpr::clone() const {
    throw 0;
}

clobber::IdentifierExpr::IdentifierExpr(const std::string &name, const ClobberToken &token)
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

clobber::BindingVectorExpr::BindingVectorExpr(const ClobberToken &open_bracket_token,
                                              std::vector<std::unique_ptr<IdentifierExpr>> &&identifiers,
                                              std::vector<std::unique_ptr<Expr>> &&exprs, const ClobberToken &close_bracket_token,
                                              size_t num_bindings)
    : open_bracket_token(open_bracket_token)
    , identifiers(std::move(identifiers))
    , exprs(std::move(exprs))
    , close_bracket_token(close_bracket_token)
    , num_bindings(num_bindings) {}

clobber::BindingVectorExpr::BindingVectorExpr(const BindingVectorExpr &other)
    : open_bracket_token(other.open_bracket_token)
    , identifiers(std::move(utils::deepcopy_identifiers(other.identifiers)))
    , exprs(std::move(utils::deepcopy_exprs(other.exprs)))
    , close_bracket_token(other.close_bracket_token)
    , num_bindings(other.num_bindings) {}

std::unique_ptr<clobber::BindingVectorExpr>
clobber::BindingVectorExpr::clone_nowrap() const {
    throw 0;
}

clobber::ParameterVectorExpr::ParameterVectorExpr(const ClobberToken &open_bracket_token,
                                                  std::vector<std::unique_ptr<IdentifierExpr>> &&identifiers,
                                                  const ClobberToken &close_bracket_token)
    : open_bracket_token(open_bracket_token)
    , identifiers(std::move(identifiers))
    , close_bracket_token(close_bracket_token) {}

clobber::ParameterVectorExpr::ParameterVectorExpr(const ParameterVectorExpr &other)
    : open_bracket_token(other.open_bracket_token)
    , identifiers(std::move(utils::deepcopy_identifiers(other.identifiers)))
    , close_bracket_token(other.close_bracket_token) {}

std::unique_ptr<clobber::ParameterVectorExpr>
clobber::ParameterVectorExpr::clone_nowrap() const {
    throw 0;
}

clobber::LetExpr::LetExpr(const ClobberToken &open_paren_token, const ClobberToken &let_token,
                          std::unique_ptr<BindingVectorExpr> binding_vector_expr, std::vector<std::unique_ptr<Expr>> &&body_exprs,
                          const ClobberToken &close_paren_token)
    : ParenthesizedExpr(Expr::Type::LetExpr, open_paren_token, close_paren_token)
    , let_token(let_token)
    , binding_vector_expr(std::move(binding_vector_expr))
    , body_exprs(std::move(body_exprs)) {}

clobber::LetExpr::LetExpr(const LetExpr &other)
    : ParenthesizedExpr(other.type, other.open_paren_token, other.close_paren_token)
    , let_token(other.let_token)
    , binding_vector_expr(std::move(other.binding_vector_expr->clone_nowrap()))
    , body_exprs(std::move(utils::deepcopy_exprs(other.body_exprs))) {}

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

clobber::FnExpr::FnExpr(const ClobberToken &open_paren_token, const ClobberToken &fn_token,
                        std::unique_ptr<ParameterVectorExpr> parameter_vector_expr, std::vector<std::unique_ptr<Expr>> &&body_exprs,
                        const ClobberToken &close_paren_token)
    : ParenthesizedExpr(Expr::Type::FnExpr, open_paren_token, close_paren_token)
    , fn_token(fn_token)
    , parameter_vector_expr(std::move(parameter_vector_expr))
    , body_exprs(std::move(body_exprs)) {}

clobber::FnExpr::FnExpr(const FnExpr &other)
    : ParenthesizedExpr(other.type, other.open_paren_token, other.close_paren_token)
    , fn_token(other.fn_token)
    , parameter_vector_expr(std::move(other.parameter_vector_expr->clone_nowrap()))
    , body_exprs(std::move(utils::deepcopy_exprs(other.body_exprs))) {}

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

std::unique_ptr<clobber::Expr>
clobber::FnExpr::clone() const {
    throw 0;
}

clobber::DefExpr::DefExpr(const ClobberToken &open_paren_token, const ClobberToken &def_token, std::unique_ptr<IdentifierExpr> identifier,
                          std::unique_ptr<Expr> value, const ClobberToken &close_paren_token)
    : ParenthesizedExpr(Expr::Type::DefExpr, open_paren_token, close_paren_token)
    , def_token(def_token)
    , identifier(std::move(identifier))
    , value(std::move(value)) {}

clobber::DefExpr::DefExpr(const DefExpr &other)
    : ParenthesizedExpr(other.type, other.open_paren_token, other.close_paren_token)
    , def_token(other.def_token)
    , identifier(std::move(other.identifier->clone_nowrap()))
    , value(std::move(other.value->clone())) {}

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

clobber::DoExpr::DoExpr(const ClobberToken &open_paren_token, const ClobberToken &do_token, std::vector<std::unique_ptr<Expr>> &&body_exprs,
                        const ClobberToken &close_paren_token)
    : ParenthesizedExpr(Expr::Type::DoExpr, open_paren_token, close_paren_token)
    , do_token(do_token)
    , body_exprs(std::move(body_exprs)) {}

clobber::DoExpr::DoExpr(const DoExpr &other)
    : ParenthesizedExpr(other.type, other.open_paren_token, other.close_paren_token)
    , do_token(other.do_token)
    , body_exprs(std::move(utils::deepcopy_exprs(other.body_exprs))) {}

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

clobber::CallExpr::CallExpr(CallExprOperatorExprType operator_expr_type, const ClobberToken &open_paren_token,
                            const ClobberToken &operator_token, const ClobberToken &close_paren_token,
                            std::vector<std::unique_ptr<Expr>> &&arguments)
    : ParenthesizedExpr(Expr::Type::CallExpr, open_paren_token, close_paren_token)
    , operator_expr_type(operator_expr_type)
    , operator_token(operator_token)
    , arguments(std::move(arguments)) {}

clobber::CallExpr::CallExpr(const CallExpr &other)
    : ParenthesizedExpr(other.type, other.open_paren_token, other.close_paren_token)
    , operator_expr_type(other.operator_expr_type)
    , operator_token(other.operator_token)
    , arguments(std::move(utils::deepcopy_exprs(other.arguments))) {}

size_t
clobber::CallExpr::hash() const {
    std::vector<size_t> hashes;
    hashes.push_back(std::hash<std::type_index>{}(std::type_index(typeid(*this))));
    hashes.push_back(std::hash<CallExprOperatorExprType>{}(this->operator_expr_type));
    hashes.push_back(this->operator_token.hash());
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

clobber::accel::AccelExpr::AccelExpr(const ClobberToken &open_paren_token, const ClobberToken &accel_token,
                                     std::unique_ptr<BindingVectorExpr> binding_vector_expr,
                                     std::vector<std::unique_ptr<Expr>> &&body_exprs, const ClobberToken &close_paren_token)
    : ParenthesizedExpr(Expr::Type::AccelExpr, open_paren_token, close_paren_token)
    , accel_token(accel_token)
    , binding_vector_expr(std::move(binding_vector_expr))
    , body_exprs(std::move(body_exprs)) {}

clobber::accel::AccelExpr::AccelExpr(const AccelExpr &other)
    : ParenthesizedExpr(other.type, other.open_paren_token, other.close_paren_token)
    , accel_token(other.accel_token)
    , binding_vector_expr(std::move(other.binding_vector_expr->clone_nowrap()))
    , body_exprs(std::move(utils::deepcopy_exprs(other.body_exprs))) {}

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

clobber::accel::MatMulExpr::MatMulExpr(const ClobberToken &open_paren_token, const ClobberToken &mat_mul_token,
                                       std::unique_ptr<Expr> fst_operand, std::unique_ptr<Expr> snd_operand,
                                       const ClobberToken &close_paren_token)
    : ParenthesizedExpr(Expr::Type::MatMulExpr, open_paren_token, close_paren_token)
    , mat_mul_token(mat_mul_token)
    , fst_operand(std::move(fst_operand))
    , snd_operand(std::move(snd_operand)) {}

clobber::accel::MatMulExpr::MatMulExpr(const MatMulExpr &other)
    : ParenthesizedExpr(other.type, other.open_paren_token, other.close_paren_token)
    , mat_mul_token(other.mat_mul_token)
    , fst_operand(std::move(other.fst_operand->clone()))
    , snd_operand(std::move(other.snd_operand->clone())) {}

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

std::unique_ptr<clobber::Expr>
clobber::accel::MatMulExpr::clone() const {
    throw 0;
}

clobber::accel::RelUExpr::RelUExpr(const ClobberToken &open_paren_token, const ClobberToken &relu_token, std::unique_ptr<Expr> operand,
                                   const ClobberToken &close_paren_token)
    : ParenthesizedExpr(Expr::Type::RelUExpr, open_paren_token, close_paren_token)
    , relu_token(relu_token)
    , operand(std::move(operand)) {}

clobber::accel::RelUExpr::RelUExpr(const RelUExpr &other)
    : ParenthesizedExpr(other.type, other.open_paren_token, other.close_paren_token)
    , relu_token(other.relu_token)
    , operand(std::move(other.operand->clone())) {}

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

clobber::CompilationUnit::CompilationUnit(const std::string &source_text, std::vector<std::unique_ptr<Expr>> &&exprs)
    : source_text(source_text)
    , exprs(std::move(exprs))
// , parse_errors(std::move(parse_errors))
{}

clobber::CompilationUnit::CompilationUnit(const CompilationUnit &other)
    : source_text(other.source_text)
    , exprs(std::move(utils::deepcopy_exprs(other.exprs)))
// , parse_errors(other.parse_errors) // parse_errors can be copied via value semantics
{}