
#include <cstddef>
#include <typeindex>

#include "clobber/ast/ast.hpp"

using namespace clobber;
using namespace clobber::accel;

void
clobber::AstWalker::walk(const Expr &e) {
    switch (e.type) {

        // leaf types
    case Expr::Type::NumericLiteralExpr: {
        const NumLiteralExpr &casted = static_cast<const NumLiteralExpr &>(e);
        on_num_literal_expr(casted);
        break;
    }
    case Expr::Type::StringLiteralExpr: {
        const StringLiteralExpr &casted = static_cast<const StringLiteralExpr &>(e);
        on_string_literal_expr(casted);
        break;
    }
    case Expr::Type::CharLiteralExpr: {
        const CharLiteralExpr &casted = static_cast<const CharLiteralExpr &>(e);
        on_char_literal_expr(casted);
        break;
    }
    case Expr::Type::IdentifierExpr: {
        const IdentifierExpr &casted = static_cast<const IdentifierExpr &>(e);
        on_identifier_expr(casted);
        break;
    }
    case Expr::Type::KeywordLiteralExpr: {
        const KeywordLiteralExpr &casted = static_cast<const KeywordLiteralExpr &>(e);
        on_keyword_literal_expr(casted);
        break;
    }
    case Expr::Type::VectorExpr: {
        const VectorExpr &ve = static_cast<const VectorExpr &>(e);
        on_vector_expr(ve);

        on_descent_callback();
        for (const auto &value : ve.values) {
            walk(*value);
        }
        on_ascent_callback();
        break;
    }

    case Expr::Type::TypeExpr: {
        const TypeExpr &te = static_cast<const TypeExpr &>(e);
        on_type_expr(te);
        break;
    }

        // node/parent node types needing traversal
    case Expr::Type::CallExpr: {
        const CallExpr &ce = static_cast<const CallExpr &>(e);
        on_call_expr(ce);

        on_descent_callback();
        walk(*ce.operator_expr);
        for (const auto &arg : ce.arguments) {
            walk(*arg);
        }
        on_ascent_callback();
        break;
    }
    case Expr::Type::LetExpr: {
        const LetExpr &casted = static_cast<const LetExpr &>(e);
        on_let_expr(casted);

        on_descent_callback();
        for (const auto &body_expr : casted.body_exprs) {
            walk(*body_expr);
        }
        on_ascent_callback();
        break;
    }
    case Expr::Type::FnExpr: {
        const FnExpr &casted = static_cast<const FnExpr &>(e);
        on_fn_expr(casted);

        on_descent_callback();
        for (const auto &body_expr : casted.body_exprs) {
            walk(*body_expr);
        }
        on_ascent_callback();
        break;
    }
    case Expr::Type::DefExpr: {
        const DefExpr &casted = static_cast<const DefExpr &>(e);
        on_def_expr(casted);

        on_descent_callback();
        walk(*casted.identifier);
        walk(*casted.value);
        on_ascent_callback();
        break;
    }
    case Expr::Type::DoExpr: {
        const DoExpr &casted = static_cast<const DoExpr &>(e);
        on_do_expr(casted);

        on_descent_callback();
        for (const auto &body_expr : casted.body_exprs) {
            walk(*body_expr);
        }
        on_ascent_callback();
        break;
    }

    // accel nodes
    case Expr::Type::AccelExpr: {
        const AccelExpr &casted = static_cast<const AccelExpr &>(e);
        on_accel_expr(casted);

        on_descent_callback();
        for (const auto &body_expr : casted.body_exprs) {
            walk(*body_expr);
        }
        on_ascent_callback();
        break;
    }

    case Expr::Type::TosaOpExpr: {
        const TOSAOpExpr &toe = static_cast<const TOSAOpExpr &>(e);
        on_tosa_op_expr(toe);

        on_descent_callback();
        for (const auto &arg : toe.arguments) {
            walk(*arg);
        }
        on_ascent_callback();
        break;
    }
    case Expr::Type::TensorExpr: {
        const TensorExpr &te = static_cast<const TensorExpr &>(e);
        on_tensor_expr(te);

        on_descent_callback();
        for (const auto &arg : te.arguments) {
            walk(*arg);
        }
        on_ascent_callback();
        break;
    }
    }
}

void
clobber::AstWalker::on_type_expr(const TypeExpr &expr) {
    switch (expr.type_kind) {
    case TypeExpr::Type::BuiltinType: {
        const clobber::BuiltinTypeExpr &casted = static_cast<const clobber::BuiltinTypeExpr &>(expr);
        on_builtin_type_expr(casted);
        break;
    }
    case TypeExpr::Type::UserDefinedType: {
        const clobber::UserDefinedTypeExpr &casted = static_cast<const clobber::UserDefinedTypeExpr &>(expr);
        on_user_defined_type_expr(casted);
        break;
    }
    case TypeExpr::Type::ParameterizedType: {
        const clobber::ParameterizedTypeExpr &casted = static_cast<const clobber::ParameterizedTypeExpr &>(expr);
        on_parameterized_type_expr(casted);

        on_descent_callback();
        for (const auto &values : casted.param_values) {
            on_expr(*values);
        }
        on_ascent_callback();
        break;
    }
    default: {
        break;
    }
    }
}