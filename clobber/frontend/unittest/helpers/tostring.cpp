#include "tostring.hpp"
#include "pch.hpp"

#include <magic_enum/magic_enum.hpp>

#include <clobber/ast/ast.hpp>
#include <clobber/parser.hpp>
#include <clobber/semantics.hpp>

#include <clobber/common/utils.hpp>

using namespace str_utils;

namespace {
    std::string
    indent(size_t indentation, const std::string &str) {
        return std::format("{}{}", str_utils::spaces(indentation), str);
    }

    inline std::string
    expr_type_tostring(clobber::Expr::Type expr_type) {
        return std::string(magic_enum::enum_name(expr_type));
    }

    inline std::string
    tt_tostring(clobber::Token::Type token_type) {
        return std::string(magic_enum::enum_name(token_type));
    }

    std::string
    format_string_metadata(const std::unordered_map<std::string, std::any> &metadata) {
        // two separate iterations cause why not

        std::vector<std::tuple<std::string, std::string>> kvps;
        for (const auto &[key, value] : metadata) {
            auto name = value.type().name();
            if (value.type() == typeid(std::string)) {
                const std::string &str = std::any_cast<const std::string &>(value);
                kvps.emplace_back(key, str);
            }
        }

        std::vector<std::string> strs;
        for (const auto &[k, v] : kvps) {
            strs.push_back(std::format("\"{}\" : `{}`", k, v));
        }

        return strs.size() > 0 ? std::format("{{ {} }}", str_utils::join("; ", strs)) : "";
    }
}; // namespace

class ImmutableTreeVisualizer final : public clobber::AstWalker {
public:
    static std::string
    visualize(const std::string &source_text, clobber::Expr &expr) {
        ImmutableTreeVisualizer itv(source_text);
        itv.walk(expr);
        return str_utils::join("\n", itv.lines);
    }

private:
    const bool print_tokens = true;
    const std::string &source_text;
    size_t current_indentation;
    std::vector<std::string> lines;

    ImmutableTreeVisualizer(const std::string &source_text)
        : print_tokens(true)
        , source_text(source_text)
        , current_indentation(0)
        , lines({}) {}

protected:
    void
    on_parameter_vector_expr(const clobber::ParameterVectorExpr &pve) {
        clobber::Span span       = pve.span();
        std::string full_text    = norm(source_text.substr(span.start, span.length));
        std::string str_metadata = format_string_metadata(pve.metadata);
        std::string str          = indent(current_indentation, std::format("[ParameterVectorExpr] `{}` {}", full_text, str_metadata));
        lines.push_back(str);
    }

    void
    on_binding_vector_expr(const clobber::BindingVectorExpr &bve) {
        clobber::Span span       = bve.span();
        std::string full_text    = norm(source_text.substr(span.start, span.length));
        std::string str_metadata = format_string_metadata(bve.metadata);
        std::string str          = indent(current_indentation, std::format("[BindingVectorExpr] `{}` {}", full_text, str_metadata));
        lines.push_back(str);
    }

    void
    on_vector_expr(const clobber::VectorExpr &ve) override {
        std::string full_text    = norm(expr_tostring(source_text, ve));
        std::string str_metadata = format_string_metadata(ve.metadata);
        std::string str = indent(current_indentation, std::format("[{}] `{}` {}", expr_type_tostring(ve.type), full_text, str_metadata));
        lines.push_back(str);
    }

    void
    on_keyword_literal_expr(const clobber::KeywordLiteralExpr &kle) override {
        std::string full_text    = norm(expr_tostring(source_text, kle));
        std::string str_metadata = format_string_metadata(kle.metadata);
        std::string str = indent(current_indentation, std::format("[{}] `{}` {}", expr_type_tostring(kle.type), full_text, str_metadata));
        lines.push_back(str);
    }

    void
    on_tosa_op_expr(const clobber::accel::TOSAOpExpr &toe) override {
        std::string full_text    = norm(expr_tostring(source_text, toe));
        std::string str_metadata = format_string_metadata(toe.metadata);
        std::string str = indent(current_indentation, std::format("[{}] `{}` {}", expr_type_tostring(toe.type), full_text, str_metadata));
        lines.push_back(str);

        IndentationGuard _(*this);
        on_token(toe.op_token);
    }

    void
    on_num_literal_expr(const clobber::NumLiteralExpr &nle) override {
        std::string full_text    = norm(expr_tostring(source_text, nle));
        std::string str_metadata = format_string_metadata(nle.metadata);
        std::string str = indent(current_indentation, std::format("[{}] `{}` {}", expr_type_tostring(nle.type), full_text, str_metadata));
        lines.push_back(str);
    }

    void
    on_string_literal_expr(const clobber::StringLiteralExpr &sle) override {
        std::string full_text    = norm(expr_tostring(source_text, sle));
        std::string str_metadata = format_string_metadata(sle.metadata);
        std::string str = indent(current_indentation, std::format("[{}] `{}` {}", expr_type_tostring(sle.type), full_text, str_metadata));
        lines.push_back(str);
    }

    void
    on_char_literal_expr(const clobber::CharLiteralExpr &cle) override {
        std::string full_text    = norm(expr_tostring(source_text, cle));
        std::string str_metadata = format_string_metadata(cle.metadata);
        std::string str = indent(current_indentation, std::format("[{}] `{}` {}", expr_type_tostring(cle.type), full_text, str_metadata));
        lines.push_back(str);
    }

    void
    on_identifier_expr(const clobber::IdentifierExpr &ie) override {
        std::string full_text    = norm(expr_tostring(source_text, ie));
        std::string str_metadata = format_string_metadata(ie.metadata);
        std::string str = indent(current_indentation, std::format("[{}] `{}` {}", expr_type_tostring(ie.type), full_text, str_metadata));
        lines.push_back(str);
    }

    void
    on_let_expr(const clobber::LetExpr &le) override {
        std::string full_text    = norm(expr_tostring(source_text, le));
        std::string str_metadata = format_string_metadata(le.metadata);
        std::string str = indent(current_indentation, std::format("[{}] `{}` {}", expr_type_tostring(le.type), full_text, str_metadata));
        lines.push_back(str);

        IndentationGuard _(*this);
        on_token(le.let_token);
        on_binding_vector_expr(*le.binding_vector_expr);
    }

    void
    on_fn_expr(const clobber::FnExpr &fe) override {
        std::string full_text    = norm(expr_tostring(source_text, fe));
        std::string str_metadata = format_string_metadata(fe.metadata);
        std::string str = indent(current_indentation, std::format("[{}] `{}` {}", expr_type_tostring(fe.type), full_text, str_metadata));
        lines.push_back(str);

        IndentationGuard _(*this);
        on_token(fe.fn_token);
        on_parameter_vector_expr(*fe.parameter_vector_expr);
    }

    void
    on_def_expr(const clobber::DefExpr &de) override {
        std::string full_text    = norm(expr_tostring(source_text, de));
        std::string str_metadata = format_string_metadata(de.metadata);
        std::string str = indent(current_indentation, std::format("[{}] `{}` {}", expr_type_tostring(de.type), full_text, str_metadata));
        lines.push_back(str);

        IndentationGuard _(*this);
        on_token(de.def_token);
    }

    void
    on_do_expr(const clobber::DoExpr &de) override {
        std::string full_text    = norm(expr_tostring(source_text, de));
        std::string str_metadata = format_string_metadata(de.metadata);
        std::string str = indent(current_indentation, std::format("[{}] `{}` {}", expr_type_tostring(de.type), full_text, str_metadata));
        lines.push_back(str);

        IndentationGuard _(*this);
        on_token(de.do_token);
    }

    void
    on_call_expr(const clobber::CallExpr &ce) override {
        std::string full_text    = norm(expr_tostring(source_text, ce));
        std::string str_metadata = format_string_metadata(ce.metadata);
        std::string str = indent(current_indentation, std::format("[{}] `{}` {}", expr_type_tostring(ce.type), full_text, str_metadata));
        lines.push_back(str);
    }

    void
    on_accel_expr(const clobber::accel::AccelExpr &ae) override {
        std::string full_text    = norm(expr_tostring(source_text, ae));
        std::string str_metadata = format_string_metadata(ae.metadata);
        std::string str = indent(current_indentation, std::format("[{}] `{}` {}", expr_type_tostring(ae.type), full_text, str_metadata));
        lines.push_back(str);

        IndentationGuard _(*this);
        on_token(ae.accel_token);
        on_binding_vector_expr(*ae.binding_vector_expr);
    }

    void
    on_descent_callback() override {
        inc_indentation();
    }

    virtual void
    on_ascent_callback() override {
        dec_indentation();
    }

    void
    inc_indentation() {
        current_indentation = std::max((size_t)0, current_indentation + 4);
    }

    void
    dec_indentation() {
        current_indentation = std::max((size_t)0, current_indentation - 4);
    }

    /* @brief Syntactic sugar that increments indentation, and decrements when it goes out of scope. */
    struct IndentationGuard {
        ImmutableTreeVisualizer &itv;
        IndentationGuard(ImmutableTreeVisualizer &itv)
            : itv(itv) {
            itv.inc_indentation();
        }

        ~IndentationGuard() { itv.dec_indentation(); }
    };

    void
    on_token(const clobber::Token &token) {
        std::string full_text    = norm(token.ExtractFullText(source_text));
        std::string str_metadata = format_string_metadata(token.metadata);
        std::string str = indent(current_indentation, std::format("[{}] `{}` {}", tt_tostring(token.type), full_text, str_metadata));
        lines.push_back(str);
    }
};

#pragma region deprecated
class [[deprecated]] ToString final : public clobber::AstRewriter {
public:
    std::string
    walk(const std::string &source_text, clobber::Expr &expr) {
        this->source_text = &source_text;
        lines             = {};

        this->on_expr(&expr);
        std::string str = str_utils::join("", lines);

        this->source_text = nullptr;
        return str;
    }

    std::string
    walk_bve(const std::string &source_text, clobber::BindingVectorExpr &bve) {
        this->source_text = &source_text;
        lines             = {};

        this->on_binding_vector_expr(&bve);
        std::string str   = str_utils::join("", lines);
        this->source_text = nullptr;
        return str;
    }

    std::string
    walk_pve(const std::string &source_text, clobber::ParameterVectorExpr &pve) {
        this->source_text = &source_text;
        lines             = {};

        this->on_parameter_vector_expr(&pve);
        std::string str   = str_utils::join("", lines);
        this->source_text = nullptr;
        return str;
    }

protected:
    void
    on_token(const clobber::Token &token) {
        lines.push_back(token.ExtractFullText(*this->source_text));
    }

    clobber::NumLiteralExpr *
    on_num_literal_expr(clobber::NumLiteralExpr *nle) override {
        this->on_token(nle->token);
        return nle;
    }

    clobber::StringLiteralExpr *
    on_string_literal_expr(clobber::StringLiteralExpr *sle) override {
        this->on_token(sle->token);
        return sle;
    }

    clobber::CharLiteralExpr *
    on_char_literal_expr(clobber::CharLiteralExpr *cle) override {
        this->on_token(cle->token);
        return cle;
    }

    clobber::IdentifierExpr *
    on_identifier_expr(clobber::IdentifierExpr *ie) override {
        this->on_token(ie->token);
        return ie;
    }

    clobber::BindingVectorExpr *
    on_binding_vector_expr(clobber::BindingVectorExpr *bve) override {
        this->on_token(bve->open_bracket_token);
        auto _ = clobber::AstRewriter::on_binding_vector_expr(bve);
        this->on_token(bve->close_bracket_token);
        return bve;
    }

    clobber::ParameterVectorExpr *
    on_parameter_vector_expr(clobber::ParameterVectorExpr *pve) override {
        this->on_token(pve->open_bracket_token);
        auto _ = clobber::AstRewriter::on_parameter_vector_expr(pve);
        this->on_token(pve->close_bracket_token);
        return pve;
    }

    clobber::LetExpr *
    on_let_expr(clobber::LetExpr *le) override {
        this->on_token(le->open_paren_token);
        this->on_token(le->let_token);
        auto _ = clobber::AstRewriter::on_let_expr(le);
        this->on_token(le->close_paren_token);
        return le;
    }

    clobber::FnExpr *
    on_fn_expr(clobber::FnExpr *fe) override {
        this->on_token(fe->open_paren_token);
        this->on_token(fe->fn_token);
        auto _ = clobber::AstRewriter::on_fn_expr(fe);
        this->on_token(fe->close_paren_token);
        return fe;
    }

    clobber::DefExpr *
    on_def_expr(clobber::DefExpr *de) override {
        this->on_token(de->open_paren_token);
        this->on_token(de->def_token);
        auto _ = clobber::AstRewriter::on_def_expr(de);
        this->on_token(de->close_paren_token);
        return de;
    }

    clobber::DoExpr *
    on_do_expr(clobber::DoExpr *de) override {
        this->on_token(de->open_paren_token);
        this->on_token(de->do_token);
        auto _ = clobber::AstRewriter::on_do_expr(de);
        this->on_token(de->close_paren_token);
        return de;
    }

    clobber::CallExpr *
    on_call_expr(clobber::CallExpr *ce) override {
        this->on_token(ce->open_paren_token);
        this->on_expr(ce->operator_expr.get());
        auto _ = clobber::AstRewriter::on_call_expr(ce);
        this->on_token(ce->close_paren_token);
        return ce;
    }

    clobber::accel::AccelExpr *
    on_accel_expr(clobber::accel::AccelExpr *ae) override {
        this->on_token(ae->open_paren_token);
        this->on_token(ae->accel_token);
        auto _ = clobber::AstRewriter::on_accel_expr(ae);
        this->on_token(ae->close_paren_token);
        return ae;
    }

private:
    const std::string *source_text;
    std::vector<std::string> lines;
};

class [[deprecated]] TreeVisualizer final : public clobber::AstRewriter {
public:
    std::string
    walk(const std::string &source_text, clobber::Expr &expr) {
        std::string str;

        this->source_text   = &source_text;
        current_indentation = 0;
        lines               = {};

        this->on_expr(&expr);
        str = str_utils::join("\n", lines);

        this->source_text = nullptr;
        return str;
    }

protected:
    void
    on_token(const clobber::Token &token) {
        std::string full_text    = norm(token.ExtractFullText(*source_text));
        std::string str_metadata = format_string_metadata(token.metadata);
        std::string str = indent(current_indentation, std::format("[{}] `{}` {}", tt_tostring(token.type), full_text, str_metadata));
        lines.push_back(str);
    }

    clobber::NumLiteralExpr *
    on_num_literal_expr(clobber::NumLiteralExpr *nle) override {
        ToString ts{};
        std::string full_text    = norm(ts.walk(*source_text, *nle));
        std::string str_metadata = format_string_metadata(nle->metadata);
        std::string str = indent(current_indentation, std::format("[{}] `{}` {}", expr_type_tostring(nle->type), full_text, str_metadata));
        lines.push_back(str);
        return nle;
    }

    clobber::StringLiteralExpr *
    on_string_literal_expr(clobber::StringLiteralExpr *sle) override {
        ToString ts{};
        std::string full_text    = norm(ts.walk(*source_text, *sle));
        std::string str_metadata = format_string_metadata(sle->metadata);
        std::string str = indent(current_indentation, std::format("[{}] `{}` {}", expr_type_tostring(sle->type), full_text, str_metadata));
        lines.push_back(str);
        return sle;
    }

    clobber::CharLiteralExpr *
    on_char_literal_expr(clobber::CharLiteralExpr *cle) override {
        ToString ts{};
        std::string full_text    = norm(ts.walk(*source_text, *cle));
        std::string str_metadata = format_string_metadata(cle->metadata);
        std::string str = indent(current_indentation, std::format("[{}] `{}` {}", expr_type_tostring(cle->type), full_text, str_metadata));
        lines.push_back(str);
        return cle;
    }

    clobber::IdentifierExpr *
    on_identifier_expr(clobber::IdentifierExpr *ie) override {
        ToString ts{};
        std::string full_text    = norm(ts.walk(*source_text, *ie));
        std::string str_metadata = format_string_metadata(ie->metadata);
        std::string str = indent(current_indentation, std::format("[{}] `{}` {}", expr_type_tostring(ie->type), full_text, str_metadata));
        lines.push_back(str);
        return ie;
    }

    clobber::BindingVectorExpr *
    on_binding_vector_expr(clobber::BindingVectorExpr *bve) override {
        ToString ts{};

        // TODO: see what you can do about this:
        std::string full_text    = norm(ts.walk_bve(*source_text, *bve));
        std::string str_metadata = format_string_metadata(bve->metadata);
        std::string str          = indent(current_indentation, std::format("[BindingVector] `{}` {}", full_text, str_metadata));
        lines.push_back(str);

        inc_indentation();
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
        dec_indentation();

        return bve;
    }

    clobber::ParameterVectorExpr *
    on_parameter_vector_expr(clobber::ParameterVectorExpr *pve) override {
        ToString ts{};

        // TODO: see what you can do about this:
        std::string full_text    = norm(ts.walk_pve(*source_text, *pve));
        std::string str_metadata = format_string_metadata(pve->metadata);
        std::string str          = indent(current_indentation, std::format("[ParameterVector] `{}` {}", full_text, str_metadata));
        lines.push_back(str);

        inc_indentation();
        for (auto &identifier : pve->identifiers) {
            auto old_ptr = identifier.get();
            auto new_ptr = on_identifier_expr(old_ptr);
            if (old_ptr != new_ptr) {
                identifier.reset(new_ptr);
            }
        }
        dec_indentation();

        return pve;
    }

    clobber::LetExpr *
    on_let_expr(clobber::LetExpr *le) override {
        ToString ts{};

        std::string full_text    = norm(ts.walk(*source_text, *le));
        std::string str_metadata = format_string_metadata(le->metadata);
        std::string str = indent(current_indentation, std::format("[{}] `{}` {}", expr_type_tostring(le->type), full_text, str_metadata));
        lines.push_back(str);

        inc_indentation();
        if (print_tokens) {
            on_token(le->open_paren_token);
            on_token(le->let_token);
        }

        on_binding_vector_expr(le->binding_vector_expr.get());

        for (auto &body_expr : le->body_exprs) {
            auto old_ptr = body_expr.get();
            auto new_ptr = on_expr(old_ptr);
            if (old_ptr != new_ptr) {
                body_expr.reset(new_ptr);
            }
        }

        if (print_tokens) {
            on_token(le->close_paren_token);
        }
        dec_indentation();

        return le;
    }

    clobber::FnExpr *
    on_fn_expr(clobber::FnExpr *fe) override {
        ToString ts{};

        std::string full_text    = norm(ts.walk(*source_text, *fe));
        std::string str_metadata = format_string_metadata(fe->metadata);
        std::string str = indent(current_indentation, std::format("[{}] `{}` {}", expr_type_tostring(fe->type), full_text, str_metadata));
        lines.push_back(str);

        inc_indentation();
        if (print_tokens) {
            on_token(fe->open_paren_token);
            on_token(fe->fn_token);
        }

        on_parameter_vector_expr(fe->parameter_vector_expr.get());

        for (auto &body_expr : fe->body_exprs) {
            auto old_ptr = body_expr.get();
            auto new_ptr = on_expr(old_ptr);
            if (old_ptr != new_ptr) {
                body_expr.reset(new_ptr);
            }
        }

        if (print_tokens) {
            on_token(fe->close_paren_token);
        }
        dec_indentation();

        return fe;
    }

    clobber::DefExpr *
    on_def_expr(clobber::DefExpr *de) override {
        ToString ts{};

        std::string full_text    = norm(ts.walk(*source_text, *de));
        std::string str_metadata = format_string_metadata(de->metadata);
        std::string str = indent(current_indentation, std::format("[{}] `{}` {}", expr_type_tostring(de->type), full_text, str_metadata));
        lines.push_back(str);

        inc_indentation();
        if (print_tokens) {
            on_token(de->open_paren_token);
            on_token(de->def_token);
        }

        on_identifier_expr(de->identifier.get());
        on_expr(de->value.get());

        if (print_tokens) {
            on_token(de->close_paren_token);
        }
        dec_indentation();

        return de;
    }

    clobber::DoExpr *
    on_do_expr(clobber::DoExpr *de) override {
        ToString ts{};

        std::string full_text    = norm(ts.walk(*source_text, *de));
        std::string str_metadata = format_string_metadata(de->metadata);
        std::string str = indent(current_indentation, std::format("[{}] `{}` {}", expr_type_tostring(de->type), full_text, str_metadata));
        lines.push_back(str);

        inc_indentation();
        if (print_tokens) {
            on_token(de->open_paren_token);
            on_token(de->do_token);
        }

        for (auto &body_expr : de->body_exprs) {
            auto old_ptr = body_expr.get();
            auto new_ptr = on_expr(old_ptr);
            if (old_ptr != new_ptr) {
                body_expr.reset(new_ptr);
            }
        }

        if (print_tokens) {
            on_token(de->close_paren_token);
        }
        dec_indentation();

        return de;
    }

    clobber::CallExpr *
    on_call_expr(clobber::CallExpr *ce) override {
        ToString ts{};

        std::string full_text    = norm(ts.walk(*source_text, *ce));
        std::string str_metadata = format_string_metadata(ce->metadata);
        std::string str = indent(current_indentation, std::format("[{}] `{}` {}", expr_type_tostring(ce->type), full_text, str_metadata));
        lines.push_back(str);

        inc_indentation();
        if (print_tokens) {
            on_token(ce->open_paren_token);
        }

        on_expr(ce->operator_expr.get());
        for (auto &argument : ce->arguments) {
            auto old_ptr = argument.get();
            auto new_ptr = on_expr(old_ptr);
            if (old_ptr != new_ptr) {
                argument.reset(new_ptr);
            }
        }

        if (print_tokens) {
            on_token(ce->close_paren_token);
        }
        dec_indentation();

        return ce;
    }

    clobber::accel::AccelExpr *
    on_accel_expr(clobber::accel::AccelExpr *ae) override {
        ToString ts{};

        std::string full_text    = norm(ts.walk(*source_text, *ae));
        std::string str_metadata = format_string_metadata(ae->metadata);
        std::string str = indent(current_indentation, std::format("[{}] `{}` {}", expr_type_tostring(ae->type), full_text, str_metadata));
        lines.push_back(str);

        inc_indentation();
        for (auto &body_expr : ae->body_exprs) {
            auto old_ptr = body_expr.get();
            auto new_ptr = on_expr(old_ptr);
            if (old_ptr != new_ptr) {
                body_expr.reset(new_ptr);
            }
        }
        dec_indentation();

        return ae;
    }

private:
    void
    inc_indentation() {
        current_indentation = std::max((size_t)0, current_indentation + 4);
    }

    void
    dec_indentation() {
        current_indentation = std::max((size_t)0, current_indentation - 4);
    }

    const bool print_tokens = true;
    const std::string *source_text;
    size_t current_indentation;
    std::vector<std::string> lines;
};

#pragma endregion

std::string
expr_tostring(const std::string &source_text, const clobber::Expr &expr) {
    clobber::Span span = expr.span();
    return source_text.substr(span.start, span.length);
}

std::string
expr_visualize_tree(const std::string &source_text, clobber::Expr &expr) {
    return ImmutableTreeVisualizer::visualize(source_text, expr);
}

std::string
type_tostring(const clobber::Type &type) {
    std::string repr;
    switch (type.kind) {
    case clobber::Type::Int: {
        repr = "int";
        break;
    }
    case clobber::Type::Float: {
        repr = "float";
        break;
    }
    case clobber::Type::Double: {
        repr = "double";
        break;
    }
    case clobber::Type::String: {
        repr = "string";
        break;
    }
    case clobber::Type::Char: {
        repr = "string";
        break;
    }
    case clobber::Type::Bool: {
        repr = "bool";
        break;
    }
    case clobber::Type::Func: {
        std::vector<std::string> type_strs;
        /*
        for (const auto &type_param : type.params) {
        }
        */
        repr = std::format("({})", str_utils::join(" -> ", type_strs));
        break;
    }
    }

    return repr;
}