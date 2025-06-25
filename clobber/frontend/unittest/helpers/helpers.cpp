
#include <any>
#include <format>

#include "helpers.hpp"
#include "syntax_factory.hpp"
#include "tostring.hpp"

#include <clobber/common/diagnostic.hpp>
#include <clobber/common/utils.hpp>

#include <clobber/ast/ast.hpp>
#include <clobber/parser.hpp>
#include <clobber/semantics.hpp>

using namespace str_utils;

namespace {
    std::string
    read_all_text(const std::string &path) {
        std::ifstream file(path, std::ios::in | std::ios::binary);
        return std::string((std::istreambuf_iterator<char>(file)), {});
    }

    std::string
    to_string_any(const std::any &a) {
        if (a.type() == typeid(int))
            return std::to_string(std::any_cast<int>(a));
        if (a.type() == typeid(std::string))
            return std::any_cast<std::string>(a);
        if (a.type() == typeid(bool))
            return std::any_cast<bool>(a) ? "true" : "false";
        if (a.type() == typeid(char))
            return std::to_string(std::any_cast<char>(a));

        return "<unsupported>";
    }

    std::string
    clobber_token_tostring(const std::string &source_text, const clobber::Token &token, bool use_alignment = true) {
        std::string value_str      = norm(token.ExtractFullText(source_text));
        std::string token_type_str = std::string(magic_enum::enum_name(token.type));
        if (use_alignment) { // cannot reduce to conditional due to `std::format` constexpr constraint
            return std::format("(tt: {:>20.20} (val: `{}`)", token_type_str, value_str);
        } else {
            return std::format("(tt: {} (val: `{}`)", token_type_str, value_str);
        }
    }

    std::string
    reconstruct_source_text_from_tokens(const std::string &source_text, const std::vector<clobber::Token> &tokens) {
        std::ostringstream builder;
        for (size_t i = 0; i < tokens.size(); i++) {
            clobber::Token token = tokens[i];
            builder << token.ExtractFullText(source_text);
        }

        return builder.str();
    }
} // namespace

// ast flattening
namespace {
    std::vector<clobber::Expr *>
    flatten_expr(clobber::Expr *expr) {
        throw 0;
    }
}; // namespace

namespace Logging {
    void
    init_logger(const std::string &logger_name, const std::string &out_log_path) {
        // No-op for std::cout based logging
        logging::set_pattern("%v");
    }

    void
    dispose_logger(const std::string &logger_name) {
        logging::flush();
        logging::drop(logger_name);
    }
}; // namespace Logging

namespace TokenizerTestsHelpers {
    void
    print_tokens(const std::string &source_text, const std::vector<clobber::Token> &expected_tokens,
                 const std::vector<clobber::Token> &actual_tokens) {
#ifndef CRT_ENABLED
        logging::info(std::format("[Expected; n={}]", expected_tokens.size()));
        logging::info("---------------------------------------------------------");
        for (size_t i = 0; i < expected_tokens.size(); i++) {
            clobber::Token token = expected_tokens[i];
            logging::info(std::format("[{:>2}] {}", i, clobber_token_tostring(source_text, token, true)));
        }
        logging::info(std::format("Source text:\n```\n{}\n```\n", source_text));

        logging::info("");
        logging::info(std::format("[Actual; n={}]", actual_tokens.size()));
        logging::info("---------------------------------------------------------");
        for (size_t i = 0; i < actual_tokens.size(); i++) {
            clobber::Token token = actual_tokens[i];
            logging::info(std::format("[{:>2}] {}", i, clobber_token_tostring(source_text, token, true)));
        }
        const std::string reconstructed = reconstruct_source_text_from_tokens(source_text, actual_tokens);
        logging::info(std::format("Reconstructed text:\n```\n{}\n```\n", reconstructed));
#endif
    }

    bool
    are_tokens_equal(const clobber::Token &t1, const clobber::Token &t2) {
        return t1.type == t2.type;
    }

    ::testing::AssertionResult
    are_num_tokens_equal(const std::vector<clobber::Token> &expected, const std::vector<clobber::Token> &actual) {
        size_t actual_num_tokens   = actual.size();
        size_t expected_num_tokens = expected.size();
        if (actual_num_tokens == expected_num_tokens) {
            return ::testing::AssertionSuccess();
        } else {
            return ::testing::AssertionFailure()
                   << std::format("Expected {} tokens, but received {}", expected_num_tokens, actual_num_tokens);
        }
    }

    ::testing::AssertionResult
    are_tokens_vec_equal(const std::string &source_text, const std::vector<clobber::Token> &expected_tokens,
                         const std::vector<clobber::Token> &actual_tokens) {
        // we're assumed to have equal number of tokens, asserted by "assert_equal_number_tokens"
        size_t num_tokens;

        num_tokens = expected_tokens.size();
        for (size_t i = 0; i < num_tokens; i++) {
            clobber::Token expected;
            clobber::Token actual;

            expected = expected_tokens[i];
            actual   = actual_tokens[i];
            if (!TokenizerTestsHelpers::are_tokens_equal(expected, actual)) {
                std::string assertion_msg =
                    std::format("Tokens at {} are not equal; expected: {}; actual: {}", i, clobber_token_tostring(source_text, expected),
                                clobber_token_tostring(source_text, actual));
                return ::testing::AssertionFailure() << assertion_msg;
            }
        }

        return ::testing::AssertionSuccess();
    }

    ::testing::AssertionResult
    is_roundtrippable(const std::string &source_text, const std::vector<clobber::Token> &actual_tokens) {
        const std::string source_text_norm   = norm(source_text);
        const std::string reconstructed      = reconstruct_source_text_from_tokens(source_text, actual_tokens);
        const std::string reconstructed_norm = norm(reconstructed);
        if (source_text_norm == reconstructed_norm) {
            return ::testing::AssertionSuccess();
        } else {
            return ::testing::AssertionFailure() << "Actual tokens don't reconstruct the original text.";
        }
    }
}; // namespace TokenizerTestsHelpers

::testing::AssertionResult
are_compilation_units_equivalent(const clobber::CompilationUnit &, const clobber::CompilationUnit &) {
    throw 0;
}

namespace ParserTestsHelpers {

    class TreeReprFlattener final : public clobber::AstWalker {
    public:
        static std::vector<std::string>
        flatten_expr(const std::string &source_text, std::function<std::string(const clobber::Token &)> token_get_repr,
                     std::function<std::string(const clobber::Expr &)> expr_get_repr, const clobber::Expr &expr) {
            auto pa_expr_get_repr  = [&expr_get_repr](const clobber::Expr &expr) { return norm(expr_get_repr(expr)); };
            auto pa_token_get_repr = [&token_get_repr](const clobber::Token &token) { return norm(token_get_repr(token)); };

            TreeReprFlattener trf(source_text, pa_expr_get_repr, pa_token_get_repr);
            trf.walk(expr);
            return trf.strs;
        }

    private:
        const std::string &source_text;
        const std::function<std::string(const clobber::Expr &)> expr_get_repr   = 0;
        const std::function<std::string(const clobber::Token &)> token_get_repr = 0;
        std::vector<std::string> strs;

        TreeReprFlattener(const std::string &source_text, std::function<std::string(const clobber::Expr &)> expr_get_repr,
                          std::function<std::string(const clobber::Token &)> token_get_repr)
            : source_text(source_text)
            , expr_get_repr(expr_get_repr)
            , token_get_repr(token_get_repr)
            , strs({}) {}

    protected:
        void
        on_token(const clobber::Token &token) {
            strs.push_back(token_get_repr(token));
        }

        void
        on_parameter_vector_expr(const clobber::ParameterVectorExpr &pve) {
            clobber::Span span = pve.span();
            strs.push_back(norm(source_text.substr(span.start, span.length)));

            on_token(pve.open_bracket_token);
            for (const auto &id : pve.identifiers) {
                on_identifier_expr(*id);
            }
            on_token(pve.close_bracket_token);
        }

        void
        on_binding_vector_expr(const clobber::BindingVectorExpr &bve) {
            clobber::Span span = bve.span();
            strs.push_back(norm(source_text.substr(span.start, span.length)));

            on_token(bve.open_bracket_token);
            for (size_t i = 0; i < bve.num_bindings; i++) {
                on_identifier_expr(*bve.identifiers[i]);
                walk(*bve.identifiers[i]);
            }
            on_token(bve.close_bracket_token);
        }

        void
        on_num_literal_expr(const clobber::NumLiteralExpr &nle) override {
            strs.push_back(expr_get_repr(nle));
            on_token(nle.token);
        }

        void
        on_string_literal_expr(const clobber::StringLiteralExpr &sle) override {
            strs.push_back(expr_get_repr(sle));
            on_token(sle.token);
        }

        void
        on_char_literal_expr(const clobber::CharLiteralExpr &cle) override {
            strs.push_back(expr_get_repr(cle));
            on_token(cle.token);
        }

        void
        on_identifier_expr(const clobber::IdentifierExpr &ie) override {
            strs.push_back(expr_get_repr(ie));
            on_token(ie.token);
        }

        void
        on_let_expr(const clobber::LetExpr &le) override {
            strs.push_back(expr_get_repr(le));

            on_token(le.open_paren_token);
            on_token(le.let_token);
            on_binding_vector_expr(*le.binding_vector_expr);
            for (const auto &body_expr : le.body_exprs) {
                walk(*body_expr);
            }
            on_token(le.close_paren_token);
        }

        void
        on_fn_expr(const clobber::FnExpr &fe) override {
            strs.push_back(expr_get_repr(fe));

            on_token(fe.open_paren_token);
            on_token(fe.fn_token);
            on_parameter_vector_expr(*fe.parameter_vector_expr);
            for (const auto &body_expr : fe.body_exprs) {
                walk(*body_expr);
            }
            on_token(fe.close_paren_token);
        }

        void
        on_def_expr(const clobber::DefExpr &de) override {
            strs.push_back(expr_get_repr(de));

            on_token(de.open_paren_token);
            on_token(de.def_token);
            on_identifier_expr(*de.identifier);
            walk(*de.value);
            on_token(de.close_paren_token);
        }

        void
        on_do_expr(const clobber::DoExpr &de) override {
            strs.push_back(expr_get_repr(de));

            on_token(de.open_paren_token);
            on_token(de.do_token);
            for (const auto &body_expr : de.body_exprs) {
                walk(*body_expr);
            }
            on_token(de.close_paren_token);
        }

        void
        on_call_expr(const clobber::CallExpr &ce) override {
            strs.push_back(expr_get_repr(ce));

            on_token(ce.open_paren_token);
            walk(*ce.operator_expr);
            for (const auto &arg : ce.arguments) {
                walk(*arg);
            }
            on_token(ce.close_paren_token);
        }

        void
        on_accel_expr(const clobber::accel::AccelExpr &ae) override {
            strs.push_back(expr_get_repr(ae));

            on_token(ae.open_paren_token);
            on_token(ae.accel_token);
            on_binding_vector_expr(*ae.binding_vector_expr);
            for (const auto &body_expr : ae.body_exprs) {
                walk(*body_expr);
            }
            on_token(ae.close_paren_token);
        }

        void
        on_mat_mul_expr(const clobber::accel::MatMulExpr &mme) override {
            strs.push_back(expr_get_repr(mme));

            on_token(mme.open_paren_token);
            on_token(mme.mat_mul_token);
            walk(*mme.fst_operand);
            walk(*mme.snd_operand);
            on_token(mme.close_paren_token);
        }

        void
        on_relu_expr(const clobber::accel::RelUExpr &re) override {
            strs.push_back(expr_get_repr(re));

            on_token(re.open_paren_token);
            on_token(re.relu_token);
            walk(*re.operand);
            on_token(re.close_paren_token);
        }

        void
        on_descent_callback() override {
            return;
        }

        virtual void
        on_ascent_callback() override {
            return;
        }
    };

    void
    print_flattened_asts(const std::vector<std::string> expected_flattened, const std::vector<std::string> actual_flattened) {
        auto eft = expected_flattened | std::views::transform([](auto &s) { return std::format("`{}`", s); });
        std::vector<std::string> expected_modified(eft.begin(), eft.end());
        const std::string expected_str = std::format("[ {} ]", str_utils::join(", ", expected_modified));

        auto aft = actual_flattened | std::views::transform([](auto &s) { return std::format("`{}`", s); });
        std::vector<std::string> actual_modified(eft.begin(), eft.end());
        const std::string actual_str = std::format("[ {} ]", str_utils::join(", ", actual_modified));

        logging::info(std::format("Expected:\n{}", expected_str));
        logging::info(std::format("Actual:\n{}", actual_str));
    }

    std::vector<std::string>
    get_error_msgs(const std::string &file, const std::string &source_text, const std::vector<clobber::Diagnostic> &diagnostics) {
        std::vector<std::string> errs;

        size_t i;
        for (i = 0; i < diagnostics.size(); i++) {
            clobber::Diagnostic diagnostic = diagnostics[i];
            errs.push_back(diagnostic.GetFormattedErrorMsg(file, source_text));
        }

        return errs;
    }

    ::testing::AssertionResult
    are_compilation_units_equivalent(const std::string &source_text, std::vector<clobber::Expr *> expected,
                                     std::vector<clobber::Expr *> actual, bool print) {
        // we compare by getting the string representations of each node and flattening them to an array.
        // compare each value in the array to assert the compilation units are equivalent with regards to their asts.

        // remember: asts from the parsers have spans that refer directly to the source text.
        // asts generated from the syntax factory have expected strings as metadata.
        // we pass callbacks to the flattener to handle the above case.

        // expected callbacks
        std::function<std::string(const clobber::Expr &)> expected_expr_get_repr = [](const clobber::Expr &expr) {
            return std::any_cast<std::string>(expr.metadata.at(default_str_metadata_tag));
        };

        std::function<std::string(const clobber::Token &)> expected_token_get_repr = [](const clobber::Token &token) {
            return std::any_cast<std::string>(token.metadata.at(default_str_metadata_tag));
        };

        // actual callbacks
        std::function<std::string(const clobber::Expr &)> actual_expr_get_repr = [&source_text](const clobber::Expr &expr) {
            clobber::Span span = expr.span();
            return source_text.substr(span.start, span.length);
        };

        std::function<std::string(const clobber::Token &)> actual_token_get_repr = [&source_text](const clobber::Token &token) {
            clobber::Span span = token.full_span;
            return source_text.substr(span.start, span.length);
        };

        std::vector<std::string> expected_flattened;
        for (const auto &expr : expected) {
            std::vector<std::string> flattened =
                TreeReprFlattener::flatten_expr(source_text, expected_token_get_repr, expected_expr_get_repr, *expr);
            expected_flattened.insert(expected_flattened.end(), flattened.begin(), flattened.end());
        }

        std::vector<std::string> actual_flattened;
        for (const auto &expr : actual) {
            std::vector<std::string> flattened =
                TreeReprFlattener::flatten_expr(source_text, actual_token_get_repr, actual_expr_get_repr, *expr);
            actual_flattened.insert(actual_flattened.end(), flattened.begin(), flattened.end());
        }

        if (print) {
            print_flattened_asts(expected_flattened, actual_flattened);
        }

        return ::testing::AssertionSuccess();
    }
}; // namespace ParserTestsHelpers

namespace SemanticTestsHelpers {

// source text shorthand for semantic models
#define SRC_TXT semantic_model.compilation_unit.get()->source_text

    std::string
    fmt_hash(size_t hash) {
        std::string as_str = std::to_string(hash); // so that we can easily truncate it
        return std::format("[{:>10.10}]", as_str); // unlikely that the first x digits collide, better to have spaced hashes when printing
    }

    std::vector<std::string> get_expr_inferred_type_strs_core(const clobber::SemanticModel &, clobber::Expr &);

    std::vector<std::string>
    num_lit_expr_inferred_strs(const clobber::SemanticModel &semantic_model, clobber::Expr &expr) {
        std::vector<std::string> strs{};
        clobber::NumLiteralExpr &nle_expr = static_cast<clobber::NumLiteralExpr &>(expr);

        size_t hash          = nle_expr.hash();
        auto it              = semantic_model.type_map->find(hash);
        std::string type_str = it != semantic_model.type_map->end() ? type_tostring(*it->second) : "<NOTYPE>";
        strs.push_back(std::format("{}: {} `{}`", fmt_hash(hash), type_str, norm(expr_tostring(SRC_TXT, nle_expr))));
        return strs;
    }

    std::vector<std::string>
    str_lit_expr_inferred_strs(const clobber::SemanticModel &semantic_model, clobber::Expr &expr) {
        std::vector<std::string> strs{};
        clobber::StringLiteralExpr &sle = static_cast<clobber::StringLiteralExpr &>(expr);

        auto it              = semantic_model.type_map->find(sle.hash());
        std::string type_str = it != semantic_model.type_map->end() ? type_tostring(*it->second) : "<NOTYPE>";
        strs.push_back(std::format("{}: {} `{}`", fmt_hash(sle.hash()), type_str, norm(expr_tostring(SRC_TXT, sle))));
        return strs;
    }

    std::vector<std::string>
    char_lit_expr_inferred_strs(const clobber::SemanticModel &semantic_model, clobber::Expr &expr) {
        std::vector<std::string> strs{};
        clobber::CharLiteralExpr &cle = static_cast<clobber::CharLiteralExpr &>(expr);

        auto it              = semantic_model.type_map->find(cle.hash());
        std::string type_str = it != semantic_model.type_map->end() ? type_tostring(*it->second) : "<NOTYPE>";
        strs.push_back(std::format("{}: {} `{}`", fmt_hash(cle.hash()), type_str, norm(expr_tostring(SRC_TXT, cle))));
        return strs;
    }

    std::vector<std::string>
    ident_expr_inferred_strs(const clobber::SemanticModel &semantic_model, clobber::Expr &expr) {
        std::vector<std::string> strs{};
        clobber::IdentifierExpr &iden_expr = static_cast<clobber::IdentifierExpr &>(expr);

        auto it              = semantic_model.type_map->find(iden_expr.hash());
        std::string type_str = it != semantic_model.type_map->end() ? type_tostring(*it->second) : "<NOTYPE>";
        strs.push_back(std::format("{}: {} `{}`", fmt_hash(iden_expr.hash()), type_str, norm(expr_tostring(SRC_TXT, iden_expr))));
        return strs;
    }

    std::vector<std::string>
    let_expr_inferred_strs(const clobber::SemanticModel &semantic_model, clobber::Expr &expr) {
        std::vector<std::string> strs{};
        clobber::LetExpr &let_expr = static_cast<clobber::LetExpr &>(expr);

        auto it              = semantic_model.type_map->find(let_expr.hash());
        std::string type_str = it != semantic_model.type_map->end() ? type_tostring(*it->second) : "<NOTYPE>";
        strs.push_back(std::format("{}: {} `{}`", fmt_hash(let_expr.hash()), type_str, norm(expr_tostring(SRC_TXT, let_expr))));

        // auto body_expr_views = ptr_utils::get_expr_views(let_expr.body_exprs);
        for (const auto &body_expr_view : let_expr.body_exprs) {
            auto sub_strs = get_expr_inferred_type_strs_core(semantic_model, *body_expr_view);
            strs.insert(strs.end(), sub_strs.begin(), sub_strs.end());
        }

        return strs;
    }

    std::vector<std::string>
    fn_expr_inferred_strs(const clobber::SemanticModel &semantic_model, clobber::Expr &expr) {
        throw 0;
    }

    std::vector<std::string>
    def_expr_inferred_strs(const clobber::SemanticModel &semantic_model, clobber::Expr &expr) {
        throw 0;
    }

    std::vector<std::string>
    do_expr_inferred_strs(const clobber::SemanticModel &semantic_model, clobber::Expr &expr) {
        throw 0;
    }

    std::vector<std::string>
    call_expr_inferred_strs(const clobber::SemanticModel &semantic_model, clobber::Expr &expr) {
        std::vector<std::string> strs{};
        clobber::CallExpr &call_expr = static_cast<clobber::CallExpr &>(expr);

        auto it              = semantic_model.type_map->find(call_expr.hash());
        std::string type_str = it != semantic_model.type_map->end() ? type_tostring(*it->second) : "<NOTYPE>";
        strs.push_back(std::format("{}: {} `{}`", fmt_hash(call_expr.hash()), type_str, norm(expr_tostring(SRC_TXT, call_expr))));

        // auto argument_expr_views = ptr_utils::get_expr_views(call_expr.arguments);
        for (const auto &argument_expr_view : call_expr.arguments) {
            auto sub_strs = get_expr_inferred_type_strs_core(semantic_model, *argument_expr_view);
            strs.insert(strs.end(), sub_strs.begin(), sub_strs.end());
        }

        return strs;
    }

    std::vector<std::string>
    get_expr_inferred_type_strs_core(const clobber::SemanticModel &semantic_model, clobber::Expr &expr) {
        using InferredTypeStrsDelegate = std::vector<std::string> (*)(const clobber::SemanticModel &, clobber::Expr &);

        // clang-format off
        const std::unordered_map<clobber::Expr::Type, InferredTypeStrsDelegate> delegates = {
            {clobber::Expr::Type::NumericLiteralExpr, num_lit_expr_inferred_strs},
            {clobber::Expr::Type::StringLiteralExpr, str_lit_expr_inferred_strs},
            {clobber::Expr::Type::CharLiteralExpr, char_lit_expr_inferred_strs},
            {clobber::Expr::Type::IdentifierExpr, ident_expr_inferred_strs},
            {clobber::Expr::Type::LetExpr, let_expr_inferred_strs},
            {clobber::Expr::Type::CallExpr, call_expr_inferred_strs},
        };
        // clang-format on

        auto it = delegates.find(expr.type);
        return it != delegates.end() ? it->second(semantic_model, expr) : std::vector<std::string>{};
    }

    std::vector<std::string>
    get_expr_inferred_type_strs(const clobber::SemanticModel &semantic_model) {
        std::vector<std::string> strs;
        // auto expr_views = ptr_utils::get_expr_views(semantic_model.compilation_unit->exprs);
        for (auto &expr_view : semantic_model.compilation_unit->exprs) {
            auto sub_strs = get_expr_inferred_type_strs_core(semantic_model, *expr_view);
            strs.insert(strs.end(), sub_strs.begin(), sub_strs.end());
        }
        return strs;
    }
}; // namespace SemanticTestsHelpers
