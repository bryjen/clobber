#pragma once

#include <clobber/common/debug.hpp>

#include "clobber/ast/ast_accel.hpp"
#include "clobber/ast/ast_core.hpp"
#include "clobber/pch.hpp"

namespace clobber {
    class AstRewriter {
    protected:
        virtual Expr *on_expr(Expr *);

        // 'leaf' exprs
        virtual NumLiteralExpr *on_num_literal_expr(NumLiteralExpr *);
        virtual StringLiteralExpr *on_string_literal_expr(StringLiteralExpr *);
        virtual CharLiteralExpr *on_char_literal_expr(CharLiteralExpr *);

        virtual IdentifierExpr *on_identifier_expr(IdentifierExpr *);
        virtual KeywordLiteralExpr *on_keyword_literal_expr(KeywordLiteralExpr *);

        virtual VectorExpr *on_vector_expr(VectorExpr **);

        // 'node' exprs
        virtual ParenthesizedExpr *on_paren_expr(ParenthesizedExpr *);
        virtual BindingVectorExpr *on_binding_vector_expr(BindingVectorExpr *);
        virtual ParameterVectorExpr *on_parameter_vector_expr(ParameterVectorExpr *);
        virtual LetExpr *on_let_expr(LetExpr *);
        virtual FnExpr *on_fn_expr(FnExpr *);
        virtual DefExpr *on_def_expr(DefExpr *);
        virtual DoExpr *on_do_expr(DoExpr *);
        virtual CallExpr *on_call_expr(CallExpr *);

        // hardware acceleration exprs
        virtual accel::AccelExpr *on_accel_expr(accel::AccelExpr *);
    };

    class AstWalker {
    public:
        // functions for actually traversing the ast top to bottom
        // protected functions above are pretty much callbacks in these traversal functions
        void walk(const Expr &);

    protected:
        // notes:
        // no callback for binding/parameterized vectors, inheritor

        virtual void on_num_literal_expr(const NumLiteralExpr &)         = 0;
        virtual void on_string_literal_expr(const StringLiteralExpr &)   = 0;
        virtual void on_char_literal_expr(const CharLiteralExpr &)       = 0;
        virtual void on_identifier_expr(const IdentifierExpr &)          = 0;
        virtual void on_keyword_literal_expr(const KeywordLiteralExpr &) = 0;
        virtual void on_vector_expr(const VectorExpr &)                  = 0;

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