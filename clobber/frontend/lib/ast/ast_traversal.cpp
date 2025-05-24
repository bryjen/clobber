
#include <cstddef>
#include <typeindex>

#include "clobber/ast/ast.hpp"
#include "clobber/internal/utils.hpp"

using namespace clobber;
using namespace clobber::accel;

clobber::Expr *
clobber::AstRewriter::on_expr(clobber::Expr *e) {
    switch (e->type) {
    case clobber::Expr::Type::LetExpr: {
        ParenthesizedExpr *pe = static_cast<ParenthesizedExpr *>(e);
        return on_paren_expr(pe);
    }
    case clobber::Expr::Type::NumericLiteralExpr: {
        NumLiteralExpr *nle = static_cast<NumLiteralExpr *>(e);
        return on_num_literal_expr(nle);
    }
    case clobber::Expr::Type::StringLiteralExpr: {
        StringLiteralExpr *sle = static_cast<StringLiteralExpr *>(e);
        return on_string_literal_expr(sle);
    }
    case clobber::Expr::Type::CharLiteralExpr: {
        CharLiteralExpr *cle = static_cast<CharLiteralExpr *>(e);
        return on_char_literal_expr(cle);
    }
    case clobber::Expr::Type::IdentifierExpr: {
        IdentifierExpr *ie = static_cast<IdentifierExpr *>(e);
        return on_identifier_expr(ie);
    }
    case clobber::Expr::Type::FnExpr: {
        FnExpr *fe = static_cast<FnExpr *>(e);
        return on_fn_expr(fe);
    }
    case clobber::Expr::Type::DefExpr: {
        DefExpr *de = static_cast<DefExpr *>(e);
        return on_def_expr(de);
    }
    case clobber::Expr::Type::DoExpr: {
        DoExpr *de = static_cast<DoExpr *>(e);
        return on_do_expr(de);
    }
    case clobber::Expr::Type::CallExpr: {
        CallExpr *ce = static_cast<CallExpr *>(e);
        return on_call_expr(ce);
    }
    case clobber::Expr::Type::AccelExpr: {
        accel::AccelExpr *ae = static_cast<accel::AccelExpr *>(e);
        return on_accel_expr(ae);
    }
    default: {
        return nullptr;
    }
    }
}

clobber::NumLiteralExpr *
clobber::AstRewriter::on_num_literal_expr(clobber::NumLiteralExpr *nle) {
    return nle;
}

clobber::StringLiteralExpr *
clobber::AstRewriter::on_string_literal_expr(clobber::StringLiteralExpr *sle) {
    return sle;
}

clobber::CharLiteralExpr *
clobber::AstRewriter::on_char_literal_expr(clobber::CharLiteralExpr *cle) {
    return cle;
}

clobber::IdentifierExpr *
clobber::AstRewriter::on_identifier_expr(clobber::IdentifierExpr *ie) {
    return ie;
}

clobber::ParenthesizedExpr *
clobber::AstRewriter::on_paren_expr(clobber::ParenthesizedExpr *pe) {
    switch (pe->type) {
    case clobber::Expr::Type::LetExpr: {
        LetExpr *le = static_cast<LetExpr *>(pe);
        return on_let_expr(le);
    }
    case clobber::Expr::Type::FnExpr: {
        FnExpr *fe = static_cast<FnExpr *>(pe);
        return on_fn_expr(fe);
    }
    case clobber::Expr::Type::DefExpr: {
        DefExpr *de = static_cast<DefExpr *>(pe);
        return on_def_expr(de);
    }
    case clobber::Expr::Type::DoExpr: {
        DoExpr *de = static_cast<DoExpr *>(pe);
        return on_do_expr(de);
    }
    case clobber::Expr::Type::CallExpr: {
        CallExpr *ce = static_cast<CallExpr *>(pe);
        return on_call_expr(ce);
    }
    case clobber::Expr::Type::AccelExpr: {
        accel::AccelExpr *ae = static_cast<accel::AccelExpr *>(pe);
        return on_accel_expr(ae);
    }
    default: {
        return nullptr;
    }
    }
}

clobber::BindingVectorExpr *
clobber::AstRewriter::on_binding_vector_expr(clobber::BindingVectorExpr *bve) {
    for (size_t i = 0; i < bve->num_bindings; i++) {
        {
            auto old_ptr = bve->identifiers[i].get();
            auto new_ptr = on_identifier_expr(old_ptr);
            if (old_ptr != new_ptr) {
                bve->identifiers[i].reset(new_ptr);
            }
        }

        {
            auto old_ptr = bve->exprs[i].get();
            auto new_ptr = on_expr(old_ptr);
            if (old_ptr != new_ptr) {
                bve->exprs[i].reset(new_ptr);
            }
        }
    }

    return bve;
}

clobber::ParameterVectorExpr *
clobber::AstRewriter::on_parameter_vector_expr(clobber::ParameterVectorExpr *pe) {
    for (std::unique_ptr<clobber::IdentifierExpr> &identifier_uptr : pe->identifiers) {
        auto old_ptr = identifier_uptr.get();
        auto new_ptr = on_identifier_expr(old_ptr);
        if (old_ptr != new_ptr) {
            identifier_uptr.reset(new_ptr);
        }
    }

    return pe;
}

clobber::LetExpr *
clobber::AstRewriter::on_let_expr(clobber::LetExpr *le) {
    { // local scope so I can use variables 'old_ptr' and 'new_ptr' names, too lazy
        auto old_ptr = le->binding_vector_expr.get();
        auto new_ptr = on_binding_vector_expr(old_ptr);
        if (old_ptr != new_ptr) {
            le->binding_vector_expr.reset(new_ptr);
        }
    }

    for (std::unique_ptr<clobber::Expr> &expr_uptr : le->body_exprs) {
        auto old_ptr = expr_uptr.get();
        auto new_ptr = on_expr(old_ptr);
        if (old_ptr != new_ptr) {
            expr_uptr.reset(new_ptr);
        }
    }

    return le;
}

clobber::FnExpr *
clobber::AstRewriter::on_fn_expr(clobber::FnExpr *fe) {
    {
        auto old_ptr = fe->parameter_vector_expr.get();
        auto new_ptr = on_parameter_vector_expr(old_ptr);
        if (old_ptr != new_ptr) {
            fe->parameter_vector_expr.reset(new_ptr);
        }
    }

    for (std::unique_ptr<clobber::Expr> &expr_uptr : fe->body_exprs) {
        auto old_ptr = expr_uptr.get();
        auto new_ptr = on_expr(old_ptr);
        if (old_ptr != new_ptr) {
            expr_uptr.reset(new_ptr);
        }
    }

    return fe;
}

clobber::DefExpr *
clobber::AstRewriter::on_def_expr(clobber::DefExpr *de) {
    {
        auto old_ptr = de->identifier.get();
        auto new_ptr = on_identifier_expr(old_ptr);
        if (old_ptr != new_ptr) {
            de->identifier.reset(new_ptr);
        }
    }

    {
        auto old_ptr = de->value.get();
        auto new_ptr = on_expr(old_ptr);
        if (old_ptr != new_ptr) {
            de->value.reset(new_ptr);
        }
    }

    return de;
}

clobber::DoExpr *
clobber::AstRewriter::on_do_expr(clobber::DoExpr *de) {
    for (std::unique_ptr<clobber::Expr> &expr_uptr : de->body_exprs) {
        auto old_ptr = expr_uptr.get();
        auto new_ptr = on_expr(old_ptr);
        if (old_ptr != new_ptr) {
            expr_uptr.reset(new_ptr);
        }
    }

    return de;
}

clobber::CallExpr *
clobber::AstRewriter::on_call_expr(clobber::CallExpr *ce) {
    for (std::unique_ptr<clobber::Expr> &expr_uptr : ce->arguments) {
        auto old_ptr = expr_uptr.get();
        auto new_ptr = on_expr(old_ptr);
        if (old_ptr != new_ptr) {
            expr_uptr.reset(new_ptr);
        }
    }

    return ce;
}

clobber::accel::AccelExpr *
clobber::AstRewriter::on_accel_expr(clobber::accel::AccelExpr *ae) {
    {
        auto old_ptr = ae->binding_vector_expr.get();
        auto new_ptr = on_binding_vector_expr(old_ptr);
        if (old_ptr != new_ptr) {
            ae->binding_vector_expr.reset(new_ptr);
        }
    }

    for (std::unique_ptr<clobber::Expr> &expr_uptr : ae->body_exprs) {
        auto old_ptr = expr_uptr.get();
        auto new_ptr = on_expr(old_ptr);
        if (old_ptr != new_ptr) {
            expr_uptr.reset(new_ptr);
        }
    }
    return ae;
}

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

        // TODO: Add case for type exprs nd shi

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