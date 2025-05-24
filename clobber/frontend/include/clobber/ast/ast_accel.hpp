#pragma once

#include <clobber/common/debug.hpp>

#include "clobber/ast/ast_core.hpp"
#include "clobber/pch.hpp"

namespace clobber {
    namespace accel {
        // --- Accel specific AST nodes
        // We don't define a separate AST for hardware accelerated syntax for ease on the parser side. This check is instead offloaded
        // during semantic analysis.

        /* @brief Represents a hardware accelerated code block. */
        struct AccelExpr final : ParenthesizedExpr {
            Token accel_token;
            std::unique_ptr<BindingVectorExpr> binding_vector_expr;
            std::vector<std::unique_ptr<Expr>> body_exprs;

        public:
            AccelExpr(const Token &, const Token &, std::unique_ptr<BindingVectorExpr>, std::vector<std::unique_ptr<Expr>> &&,
                      const Token &);
            AccelExpr(const AccelExpr &);

            Span span() const override;
            size_t hash() const override;
            std::unique_ptr<Expr> clone() const override;
        };

        // rationale:
        // MLIR TOSA ops can be represented as basically a function call.
        // We parse them as regular call expressions and validate them later during semantic analysis.
        // Use our own type for it to separate it from regular function calls.
        /* @brief Represents a MLIR TOSA operation. */
        struct TOSAOpExpr final : ParenthesizedExpr {
            Token op_token;
            std::vector<std::unique_ptr<Expr>> arguments;

            TOSAOpExpr(const Token &, const Token &, std::vector<std::unique_ptr<Expr>> &&, const Token &);
            TOSAOpExpr(const TOSAOpExpr &);

        public:
            Span span() const override;
            size_t hash() const override;
            std::unique_ptr<Expr> clone() const override;
        };

        // rationale:
        // direct mapping to a tosa op, but we would like to have multiple ways to declare a tensor (ex. zero init by passing i32
        // dimensions, or by passing a vector of values and infer the shape from there)
        /* @brief Represents a tensor declaration. */
        struct TensorExpr final : ParenthesizedExpr {
            Token tensor_token;
            std::vector<std::unique_ptr<Expr>> arguments;

            TensorExpr(const Token &, const Token &, std::vector<std::unique_ptr<Expr>> &&, const Token &);
            TensorExpr(const TensorExpr &);

        public:
            Span span() const override;
            size_t hash() const override;
            std::unique_ptr<Expr> clone() const override;
        };
    }; // namespace accel
}; // namespace clobber