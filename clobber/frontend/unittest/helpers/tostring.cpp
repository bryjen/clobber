#include "tostring.hpp"
#include "pch.hpp"
#include <concepts>

#include <magic_enum/magic_enum.hpp>

#include <clobber/ast/ast.hpp>
#include <clobber/parser.hpp>
#include <clobber/semantics.hpp>

#include <clobber/common/utils.hpp>

using namespace str_utils;

namespace {
    constexpr size_t indentation_width = 4;

    std::string
    indent(size_t indentation, const std::string &str) {
        return std::format("{}{}", str_utils::spaces(indentation), str);
    }

    template <typename T>
    concept ParserEnumTypes = std::same_as<T, clobber::Expr::Type> || std::same_as<T, clobber::TypeExpr::Type>;

    template <ParserEnumTypes E>
    inline std::string
    type_tostring(E e) {
        return std::string(magic_enum::enum_name(e));
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
        std::vector<std::string> lines;
        ImmutableTreeVisualizer itv(source_text, lines);
        itv.walk(expr);
        return str_utils::join("\n", itv.lines);
    }

private:
    const bool print_tokens = true;
    const std::string &source_text;
    size_t current_indentation;
    std::vector<std::string> &lines;

    ImmutableTreeVisualizer(const std::string &source_text, std::vector<std::string> &lines)
        : print_tokens(true)
        , source_text(source_text)
        , current_indentation(0)
        , lines(lines) {}

    ImmutableTreeVisualizer(const std::string &source_text, std::vector<std::string> &lines, int indentation)
        : print_tokens(true)
        , source_text(source_text)
        , current_indentation(indentation)
        , lines(lines) {}

protected:
    void
    on_parameter(const clobber::Parameter &param) {
        clobber::Span span       = param.span();
        std::string full_text    = norm(source_text.substr(span.start, span.length));
        std::string str_metadata = format_string_metadata(param.metadata);
        std::string str          = indent(current_indentation, std::format("[Parameter] `{}` {}", full_text, str_metadata));
        lines.push_back(str);

        IndentationGuard _(*this);
        on_identifier_expr(*param.identifier);
        if (param.type_annot) {
            on_type_expr(*param.type_annot);
        }
    }

    void
    on_binding(const clobber::Binding &binding) {
        clobber::Span span       = binding.span();
        std::string full_text    = norm(source_text.substr(span.start, span.length));
        std::string str_metadata = format_string_metadata(binding.metadata);
        std::string str          = indent(current_indentation, std::format("[Binding] `{}` {}", full_text, str_metadata));
        lines.push_back(str);

        IndentationGuard _(*this);
        on_identifier_expr(*binding.identifier);
        if (binding.type_annot) {
            on_type_expr(*binding.type_annot);
        }
        on_expr(*binding.value);
    }

    void
    on_builtin_type_expr(const clobber::BuiltinTypeExpr &bte) {
        clobber::Span span       = bte.span();
        std::string full_text    = norm(source_text.substr(span.start, span.length));
        std::string str_metadata = format_string_metadata(bte.metadata);
        std::string str = indent(current_indentation, std::format("[{}] `{}` {}", type_tostring(bte.type_kind), full_text, str_metadata));
        lines.push_back(str);
    }

    void
    on_user_defined_type_expr(const clobber::UserDefinedTypeExpr &udte) {
        clobber::Span span       = udte.span();
        std::string full_text    = norm(source_text.substr(span.start, span.length));
        std::string str_metadata = format_string_metadata(udte.metadata);
        std::string str = indent(current_indentation, std::format("[{}] `{}` {}", type_tostring(udte.type_kind), full_text, str_metadata));
        lines.push_back(str);
    }

    void
    on_parameterized_type_expr(const clobber::ParameterizedTypeExpr &pte) {
        clobber::Span span       = pte.span();
        std::string full_text    = norm(source_text.substr(span.start, span.length));
        std::string str_metadata = format_string_metadata(pte.metadata);
        std::string str = indent(current_indentation, std::format("[{}] `{}` {}", type_tostring(pte.type_kind), full_text, str_metadata));
        lines.push_back(str);
    }

    void
    on_parameter_vector_expr(const clobber::ParameterVectorExpr &pve) {
        clobber::Span span       = pve.span();
        std::string full_text    = norm(source_text.substr(span.start, span.length));
        std::string str_metadata = format_string_metadata(pve.metadata);
        std::string str          = indent(current_indentation, std::format("[ParameterVectorExpr] `{}` {}", full_text, str_metadata));
        lines.push_back(str);

        ImmutableTreeVisualizer itv(source_text, lines, current_indentation + indentation_width); // sub itv for walking bve
        for (const auto &parameter : pve.parameters) {
            itv.on_parameter(*parameter);
        }
    }

    void
    on_binding_vector_expr(const clobber::BindingVectorExpr &bve) {
        clobber::Span span       = bve.span();
        std::string full_text    = norm(source_text.substr(span.start, span.length));
        std::string str_metadata = format_string_metadata(bve.metadata);
        std::string str          = indent(current_indentation, std::format("[BindingVectorExpr] `{}` {}", full_text, str_metadata));
        lines.push_back(str);

        ImmutableTreeVisualizer itv(source_text, lines, current_indentation + indentation_width); // sub itv for walking bve
        for (const auto &binding : bve.bindings) {
            itv.on_binding(*binding);
        }
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
    on_tensor_expr(const clobber::accel::TensorExpr &te) override {
        std::string full_text    = norm(expr_tostring(source_text, te));
        std::string str_metadata = format_string_metadata(te.metadata);
        std::string str = indent(current_indentation, std::format("[{}] `{}` {}", expr_type_tostring(te.type), full_text, str_metadata));
        lines.push_back(str);

        IndentationGuard _(*this);
        on_token(te.tensor_token);
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
        current_indentation = std::max((size_t)0, current_indentation + indentation_width);
    }

    void
    dec_indentation() {
        current_indentation = std::max((size_t)0, current_indentation - indentation_width);
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