#include "pch.hpp"

#include "helpers.hpp"
#include "tostring.hpp"

#include <clobber/common/diagnostic.hpp>
#include <clobber/common/utils.hpp>

#include <clobber/ast.hpp>
#include <clobber/parser.hpp>
#include <clobber/semantics.hpp>

std::string
normalize(const std::string &str) {
    return str_utils::normalize_whitespace(str_utils::remove_newlines(str_utils::trim(str)));
}

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
    std::string value_str      = normalize(token.ExtractFullText(source_text));
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

std::string
get_executable_directory() {
    char buffer[MAX_PATH];
    GetModuleFileNameA(nullptr, buffer, MAX_PATH);
    std::filesystem::path exe_path(buffer);
    return exe_path.parent_path().string();
}

namespace Logging {
    void
    init_logger(const std::string &logger_name, const std::string &out_log_path) {
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        auto file_sink    = std::make_shared<spdlog::sinks::basic_file_sink_mt>(out_log_path, true);

        std::vector<spdlog::sink_ptr> sinks{console_sink, file_sink};
        auto logger = std::make_shared<spdlog::logger>(logger_name, sinks.begin(), sinks.end());
        spdlog::set_default_logger(logger);

        // spdlog::set_pattern("[%H:%M:%S.%e; %^%l%$]: %v");
        spdlog::set_pattern("%v");
    }

    void
    dispose_logger(const std::string &logger_name) {
        spdlog::get(logger_name)->flush();
        spdlog::drop(logger_name);
    }
}; // namespace Logging

namespace TokenizerTestsHelpers {
    void
    print_tokens(const std::string &source_text, const std::vector<clobber::Token> &expected_tokens,
                 const std::vector<clobber::Token> &actual_tokens) {
#ifndef CRT_ENABLED
        spdlog::info(std::format("[Expected; n={}]", expected_tokens.size()));
        spdlog::info("---------------------------------------------------------");
        for (size_t i = 0; i < expected_tokens.size(); i++) {
            clobber::Token token = expected_tokens[i];
            spdlog::info(std::format("[{:>2}] {}", i, clobber_token_tostring(source_text, token, true)));
        }
        spdlog::info(std::format("Source text:\n```\n{}\n```\n", source_text));

        spdlog::info("");
        spdlog::info(std::format("[Actual; n={}]", actual_tokens.size()));
        spdlog::info("---------------------------------------------------------");
        for (size_t i = 0; i < actual_tokens.size(); i++) {
            clobber::Token token = actual_tokens[i];
            spdlog::info(std::format("[{:>2}] {}", i, clobber_token_tostring(source_text, token, true)));
        }
        const std::string reconstructed = reconstruct_source_text_from_tokens(source_text, actual_tokens);
        spdlog::info(std::format("Reconstructed text:\n```\n{}\n```\n", reconstructed));
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
        const std::string source_text_norm   = normalize(source_text);
        const std::string reconstructed      = reconstruct_source_text_from_tokens(source_text, actual_tokens);
        const std::string reconstructed_norm = normalize(reconstructed);
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

std::vector<std::string>
ParserTestsHelpers::get_error_msgs(const std::string &file, const std::string &source_text,
                                   const std::vector<clobber::Diagnostic> &diagnostics) {
    std::vector<std::string> errs;

    size_t i;
    for (i = 0; i < diagnostics.size(); i++) {
        clobber::Diagnostic diagnostic = diagnostics[i];
        errs.push_back(diagnostic.GetFormattedErrorMsg(file, source_text));
    }

    return errs;
}

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
        strs.push_back(std::format("{}: {} `{}`", fmt_hash(hash), type_str, normalize(expr_tostring(SRC_TXT, nle_expr))));
        return strs;
    }

    std::vector<std::string>
    str_lit_expr_inferred_strs(const clobber::SemanticModel &semantic_model, clobber::Expr &expr) {
        std::vector<std::string> strs{};
        clobber::StringLiteralExpr &sle = static_cast<clobber::StringLiteralExpr &>(expr);

        auto it              = semantic_model.type_map->find(sle.hash());
        std::string type_str = it != semantic_model.type_map->end() ? type_tostring(*it->second) : "<NOTYPE>";
        strs.push_back(std::format("{}: {} `{}`", fmt_hash(sle.hash()), type_str, normalize(expr_tostring(SRC_TXT, sle))));
        return strs;
    }

    std::vector<std::string>
    char_lit_expr_inferred_strs(const clobber::SemanticModel &semantic_model, clobber::Expr &expr) {
        std::vector<std::string> strs{};
        clobber::CharLiteralExpr &cle = static_cast<clobber::CharLiteralExpr &>(expr);

        auto it              = semantic_model.type_map->find(cle.hash());
        std::string type_str = it != semantic_model.type_map->end() ? type_tostring(*it->second) : "<NOTYPE>";
        strs.push_back(std::format("{}: {} `{}`", fmt_hash(cle.hash()), type_str, normalize(expr_tostring(SRC_TXT, cle))));
        return strs;
    }

    std::vector<std::string>
    ident_expr_inferred_strs(const clobber::SemanticModel &semantic_model, clobber::Expr &expr) {
        std::vector<std::string> strs{};
        clobber::IdentifierExpr &iden_expr = static_cast<clobber::IdentifierExpr &>(expr);

        auto it              = semantic_model.type_map->find(iden_expr.hash());
        std::string type_str = it != semantic_model.type_map->end() ? type_tostring(*it->second) : "<NOTYPE>";
        strs.push_back(std::format("{}: {} `{}`", fmt_hash(iden_expr.hash()), type_str, normalize(expr_tostring(SRC_TXT, iden_expr))));
        return strs;
    }

    std::vector<std::string>
    let_expr_inferred_strs(const clobber::SemanticModel &semantic_model, clobber::Expr &expr) {
        std::vector<std::string> strs{};
        clobber::LetExpr &let_expr = static_cast<clobber::LetExpr &>(expr);

        auto it              = semantic_model.type_map->find(let_expr.hash());
        std::string type_str = it != semantic_model.type_map->end() ? type_tostring(*it->second) : "<NOTYPE>";
        strs.push_back(std::format("{}: {} `{}`", fmt_hash(let_expr.hash()), type_str, normalize(expr_tostring(SRC_TXT, let_expr))));

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
        strs.push_back(std::format("{}: {} `{}`", fmt_hash(call_expr.hash()), type_str, normalize(expr_tostring(SRC_TXT, call_expr))));

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
