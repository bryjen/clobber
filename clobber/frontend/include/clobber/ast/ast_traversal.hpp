#pragma once

#include <clobber/common/debug.hpp>

#include "clobber/ast/ast_accel.hpp"
#include "clobber/ast/ast_core.hpp"
#include "clobber/pch.hpp"

namespace clobber {
    class AstWalker {
    public:
        // functions for actually traversing the ast top to bottom
        // protected functions above are pretty much callbacks in these traversal functions
        void walk(const Expr &);

    protected:
        // notes:
        // no callback for binding/parameterized vectors, inheritor

        void
        on_expr(const Expr &expr) { // for naming consistency in subclass implementations
            walk(expr);
        }

        void on_type_expr(const TypeExpr &expr);

        virtual void on_num_literal_expr(const NumLiteralExpr &)         = 0;
        virtual void on_string_literal_expr(const StringLiteralExpr &)   = 0;
        virtual void on_char_literal_expr(const CharLiteralExpr &)       = 0;
        virtual void on_identifier_expr(const IdentifierExpr &)          = 0;
        virtual void on_keyword_literal_expr(const KeywordLiteralExpr &) = 0;
        virtual void on_vector_expr(const VectorExpr &)                  = 0;

        virtual void on_builtin_type_expr(const BuiltinTypeExpr &)             = 0;
        virtual void on_user_defined_type_expr(const UserDefinedTypeExpr &)    = 0;
        virtual void on_parameterized_type_expr(const ParameterizedTypeExpr &) = 0;

        virtual void on_let_expr(const LetExpr &)   = 0;
        virtual void on_fn_expr(const FnExpr &)     = 0;
        virtual void on_def_expr(const DefExpr &)   = 0;
        virtual void on_do_expr(const DoExpr &)     = 0;
        virtual void on_call_expr(const CallExpr &) = 0;

        virtual void on_accel_expr(const accel::AccelExpr &)    = 0;
        virtual void on_tosa_op_expr(const accel::TOSAOpExpr &) = 0;
        virtual void on_tensor_expr(const accel::TensorExpr &)  = 0;

        /* @brief Callback when descending to a child note. */
        virtual void on_descent_callback() = 0;

        /* @brief Callback when ascending upwards to a parent note. */
        virtual void on_ascent_callback() = 0;
    };
}; // namespace clobber