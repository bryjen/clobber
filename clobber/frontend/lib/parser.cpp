#include <expected>

#include <magic_enum/magic_enum.hpp>

#include "clobber/pch.hpp"

#include <clobber/common/diagnostic.hpp>
#include <clobber/common/utils.hpp>

#include "clobber/ast/ast.hpp"
#include "clobber/parser.hpp"

#include "clobber/internal/diagnostic_factory.hpp"

struct ParseContext {
    const std::string &source_text;
    const std::vector<clobber::Token> &tokens;

    size_t current_idx;
    std::vector<clobber::Diagnostic> &diagnostics;
};

namespace error_recovery {
    enum class Recovery {
        SeekCloseParen,
        SeekCloseBrace,
    };

    void
    recover(Recovery recovery_action, ParseContext &ctx) {
        // lambda of 'try_get_token' because intellisense erroring and being a bitch
        auto token = [](const std::vector<clobber::Token> &tokens, size_t idx) {
            size_t tokens_len = tokens.size();
            return (idx >= 0 && idx < tokens_len) ? std::make_optional(tokens[idx]) : std::nullopt;
        };

        size_t stack_size = 0; // account for open parens
        const clobber::Token::Type stopping_token_type =
            recovery_action == Recovery::SeekCloseParen ? clobber::Token::Type::CloseParenToken : clobber::Token::Type::CloseBracketToken;
        const clobber::Token::Type inverse_token_Type =
            recovery_action == Recovery::SeekCloseParen ? clobber::Token::Type::OpenParenToken : clobber::Token::Type::OpenBracketToken;
        Option<clobber::Token> current_token = token(ctx.tokens, ctx.current_idx);
        while (current_token) {
            if (current_token.value().type == stopping_token_type && stack_size-- <= 0) {
                break;
            } else if (current_token.value().type == inverse_token_Type) {
                stack_size++;
            }

            current_token = token(ctx.tokens, ctx.current_idx++);
        }
    }
}; // namespace error_recovery

using namespace error_recovery;

template <typename T> using Option = std::optional<T>;
using clobber::Diagnostic;
using ParseResult   = std::expected<std::unique_ptr<clobber::Expr>, Diagnostic>;
using ParseDelegate = ParseResult (*)(ParseContext &);
#define PARSE_DELEGATE_FN(FN_NAME) ParseResult FN_NAME(ParseContext &);

PARSE_DELEGATE_FN(try_parse)
PARSE_DELEGATE_FN(try_parse_call_expr)
PARSE_DELEGATE_FN(try_parse_call_expr_or_special_form)
PARSE_DELEGATE_FN(try_parse_let_expr)
PARSE_DELEGATE_FN(try_parse_fn_expr)
PARSE_DELEGATE_FN(try_parse_def_expr)
PARSE_DELEGATE_FN(try_parse_do_expr)

PARSE_DELEGATE_FN(try_parse_identifier)
PARSE_DELEGATE_FN(try_parse_keyword_literal_expr)

PARSE_DELEGATE_FN(try_parse_numeric_literal_expr)
PARSE_DELEGATE_FN(try_parse_string_literal_expr)
PARSE_DELEGATE_FN(try_parse_char_literal_expr)
PARSE_DELEGATE_FN(try_parse_vector_expr)

// accel specific syntax
PARSE_DELEGATE_FN(try_parse_accel_expr)
PARSE_DELEGATE_FN(try_parse_tosa_op_expr)
PARSE_DELEGATE_FN(try_parse_tensor_expr)

Option<clobber::Token>
try_get_token(const std::vector<clobber::Token> &tokens, size_t idx) {
    size_t tokens_len = tokens.size();
    return (idx >= 0 && idx < tokens_len) ? std::make_optional(tokens[idx]) : std::nullopt;
}

Option<ParseDelegate>
try_get_parse_fun(clobber::Token::Type token_type) {
    // clang-format off
    auto is_operator_token_type = [](clobber::Token::Type tt) { 
        const std::unordered_set<clobber::Token::Type> valid_token_types = {
            clobber::Token::Type::PlusToken,
            clobber::Token::Type::MinusToken,
            clobber::Token::Type::AsteriskToken,
            clobber::Token::Type::SlashToken
        };
        return valid_token_types.contains(tt); 
    };
    // clang-format on

    // keep using if-statements instead of a map/switch to support predicates over different token types
    if (token_type == clobber::Token::Type::NumericLiteralToken) {
        return std::make_optional(try_parse_numeric_literal_expr);

    } else if (token_type == clobber::Token::Type::OpenParenToken) {
        return std::make_optional(try_parse_call_expr_or_special_form);

    } else if (token_type == clobber::Token::Type::IdentifierToken || is_operator_token_type(token_type)) {
        return std::make_optional(try_parse_identifier);

    } else if (token_type == clobber::Token::Type::StringLiteralToken) {
        return std::make_optional(try_parse_string_literal_expr);

    } else if (token_type == clobber::Token::Type::CharLiteralToken) {
        return std::make_optional(try_parse_char_literal_expr);

    } else if (token_type == clobber::Token::Type::OpenBracketToken) {
        return std::make_optional(try_parse_vector_expr);

    } else if (token_type == clobber::Token::Type::KeywordLiteralToken) {
        return std::make_optional(try_parse_keyword_literal_expr);

    } else {
        return std::nullopt;
    }
}

std::expected<std::unique_ptr<clobber::TypeExpr>, Diagnostic> try_parse_builtin_type_expr(const clobber::Token &, ParseContext &);
std::expected<std::unique_ptr<clobber::TypeExpr>, Diagnostic> try_parse_user_defined_type_expr(const clobber::Token &, ParseContext &);
std::expected<std::unique_ptr<clobber::TypeExpr>, Diagnostic> try_parse_parameterized_type_expr(std::unique_ptr<clobber::TypeExpr>,
                                                                                                ParseContext &);

// remarks:
// assumes that the current char is the caret token.
std::expected<std::unique_ptr<clobber::TypeExpr>, Diagnostic>
try_parse_type_expr(ParseContext &ctx) {
    auto is_type_keyword_token = [](const clobber::Token::Type &type) {
        // clang-format off
        std::unordered_set<clobber::Token::Type> token_types = {
            clobber::Token::Type::CharKeywordToken,
            clobber::Token::Type::StringKeywordToken,
            clobber::Token::Type::VectorKeywordToken,
            clobber::Token::Type::I8KeywordToken,
            clobber::Token::Type::I16KeywordToken,
            clobber::Token::Type::I32KeywordToken,
            clobber::Token::Type::I64KeywordToken,
            clobber::Token::Type::F32KeywordToken,
            clobber::Token::Type::F64KeywordToken
        };
        // clang-format on

        auto it = token_types.find(type);
        return it != token_types.end();
    };

    clobber::Token caret_token   = ctx.tokens[ctx.current_idx++];
    clobber::Token current_token = ctx.tokens[ctx.current_idx];

    std::expected<std::unique_ptr<clobber::TypeExpr>, Diagnostic> type_expr;
    if (is_type_keyword_token(current_token.type)) {
        type_expr = try_parse_builtin_type_expr(caret_token, ctx);
    } else if (current_token.type == clobber::Token::Type::IdentifierToken) {
        type_expr = try_parse_builtin_type_expr(caret_token, ctx);
    } else {
        auto err = diag::parser::not_valid_type_token(current_token.span);
        return std::unexpected(err);
    }

    if (!type_expr) {
        return type_expr; // return error type
    }

    current_token = ctx.tokens[ctx.current_idx];
    return ctx.tokens[ctx.current_idx].type == clobber::Token::Type::LessThanToken
               ? try_parse_parameterized_type_expr(std::move(type_expr.value()), ctx)
               : std::move(type_expr);
}

std::expected<std::unique_ptr<clobber::TypeExpr>, Diagnostic>
try_parse_builtin_type_expr(const clobber::Token &caret_token, ParseContext &ctx) {
    clobber::Token type_keyword_token = ctx.tokens[ctx.current_idx++]; // asserted by caller
    return std::make_unique<clobber::BuiltinTypeExpr>(caret_token, type_keyword_token);
}

std::expected<std::unique_ptr<clobber::TypeExpr>, Diagnostic>
try_parse_user_defined_type_expr(const clobber::Token &caret_token, ParseContext &ctx) {
    clobber::Token identifier_token = ctx.tokens[ctx.current_idx++]; // asserted by caller
    return std::make_unique<clobber::UserDefinedTypeExpr>(caret_token, identifier_token);
}

std::expected<std::unique_ptr<clobber::TypeExpr>, Diagnostic>
try_parse_parameterized_type_expr(std::unique_ptr<clobber::TypeExpr> type_expr, ParseContext &ctx) {
    clobber::Token lt_token = ctx.tokens[ctx.current_idx++];

    std::vector<std::unique_ptr<clobber::Expr>> pvalues;
    std::vector<clobber::Token> commas;
    while (ctx.current_idx < ctx.tokens.size()) {
        ParseResult param_value = try_parse(ctx);
        if (!param_value) {
            return std::unexpected(param_value.error());
        }

        pvalues.push_back(std::move(param_value.value()));

        clobber::Token current_token = ctx.tokens[ctx.current_idx];
        if (current_token.type == clobber::Token::Type::GreaterThanToken) {
            clobber::Token gt_token = ctx.tokens[ctx.current_idx++];
            return std::make_unique<clobber::ParameterizedTypeExpr>(std::move(type_expr), lt_token, std::move(pvalues), commas, gt_token);
        } else if (current_token.type == clobber::Token::Type::CommaToken) {
            commas.push_back(ctx.tokens[ctx.current_idx++]);
        }
    }
}

ParseResult
try_parse_numeric_literal_expr(ParseContext &ctx) {
    return std::make_unique<clobber::NumLiteralExpr>(
        ctx.tokens[ctx.current_idx++]); // no bounds check, current token exists, asserted by caller
}

ParseResult
try_parse_string_literal_expr(ParseContext &ctx) {
    clobber::Token token = ctx.tokens[ctx.current_idx];
    std::string str      = token.ExtractText(ctx.source_text);
    if (str.size() > 2) {
        str = str.substr(1, str.size() - 2);
    }

    return std::make_unique<clobber::StringLiteralExpr>(str, ctx.tokens[ctx.current_idx++]);
}

ParseResult
try_parse_char_literal_expr(ParseContext &ctx) {
    clobber::Token token = ctx.tokens[ctx.current_idx];
    std::string str      = token.ExtractText(ctx.source_text);
    if (str.size() > 2) {
        str = str.substr(1, str.size() - 2);
    }

    return std::make_unique<clobber::CharLiteralExpr>(
        str, ctx.tokens[ctx.current_idx++]); // no bounds check, current token exists, asserted by caller
}

ParseResult
try_parse_identifier(ParseContext &ctx) {
    clobber::Token token   = ctx.tokens[ctx.current_idx]; // guaranteed to exist by caller
    const std::string name = ctx.source_text.substr(token.span.start, token.span.length);
    return std::make_unique<clobber::IdentifierExpr>(name, ctx.tokens[ctx.current_idx++]);
}

std::expected<std::unique_ptr<clobber::BindingVectorExpr>, Diagnostic>
try_parse_binding_vector_expr(ParseContext &ctx) {
    clobber::Token open_bracket_token;
    std::vector<std::unique_ptr<clobber::IdentifierExpr>> identifiers;
    std::vector<std::unique_ptr<clobber::Expr>> exprs;
    clobber::Token close_bracket_token;
    Option<clobber::Token> current_token;
    size_t num_bindings = 0;

    open_bracket_token = ctx.tokens[ctx.current_idx++];

    while (true) {
        current_token = try_get_token(ctx.tokens, ctx.current_idx);
        if (current_token && current_token.value().type == clobber::Token::Type::CloseBracketToken) {
            break;
        }

        ParseResult ident_expr_base_result = try_parse_identifier(ctx);
        if (ident_expr_base_result) {
            std::unique_ptr<clobber::Expr> ident_expr_base = std::move(ident_expr_base_result.value());
            if (auto *raw = dynamic_cast<clobber::IdentifierExpr *>(ident_expr_base.get())) {
                std::unique_ptr<clobber::IdentifierExpr> ident_expr(static_cast<clobber::IdentifierExpr *>(ident_expr_base.release()));
                identifiers.push_back(std::move(ident_expr));
            }
        }

        ParseResult value_expr = try_parse(ctx);
        if (value_expr) {
            exprs.push_back(std::move(value_expr.value()));
        }

        num_bindings++;
    }

    close_bracket_token = ctx.tokens[ctx.current_idx++];
    return std::make_unique<clobber::BindingVectorExpr>(open_bracket_token, std::move(identifiers), std::move(exprs), close_bracket_token,
                                                        num_bindings);
}

std::expected<std::unique_ptr<clobber::ParameterVectorExpr>, Diagnostic>
try_parse_parameter_vector_expr(ParseContext &ctx) {
    clobber::Token open_bracket_token;
    std::vector<std::unique_ptr<clobber::IdentifierExpr>> identifiers;
    clobber::Token close_bracket_token;
    Option<clobber::Token> current_token;

    open_bracket_token = ctx.tokens[ctx.current_idx++];

    while (true) {
        current_token = try_get_token(ctx.tokens, ctx.current_idx);
        if (current_token && current_token.value().type == clobber::Token::Type::CloseBracketToken) {
            break;
        }

        ParseResult ident_expr_base_result = try_parse_identifier(ctx);
        if (ident_expr_base_result) {
            std::unique_ptr<clobber::Expr> ident_expr_base = std::move(ident_expr_base_result.value());
            if (auto *raw = dynamic_cast<clobber::IdentifierExpr *>(ident_expr_base.get())) {
                std::unique_ptr<clobber::IdentifierExpr> ident_expr(static_cast<clobber::IdentifierExpr *>(ident_expr_base.release()));
                identifiers.push_back(std::move(ident_expr));
            }
        }
    }

    close_bracket_token = ctx.tokens[ctx.current_idx++];
    return std::make_unique<clobber::ParameterVectorExpr>(open_bracket_token, std::move(identifiers), close_bracket_token);
}

ParseResult
try_parse_call_expr_or_special_form(ParseContext &ctx) {
    // clang-format off
    const std::unordered_map<clobber::Token::Type, ParseDelegate> special_form_parse_fns = {
        { clobber::Token::Type::LetKeywordToken, try_parse_let_expr},
        { clobber::Token::Type::FnKeywordToken, try_parse_fn_expr},
        { clobber::Token::Type::DefKeywordToken, try_parse_def_expr},
        { clobber::Token::Type::DoKeywordToken, try_parse_do_expr},
        { clobber::Token::Type::AccelKeywordToken, try_parse_accel_expr},

        // { clobber::Token::Type::MatmulKeywordToken, try_parse_matmul_expr},
        // { clobber::Token::Type::ReluKeywordToken, try_parse_relu_expr},

        { clobber::Token::Type::TensorKeywordToken, try_parse_tensor_expr },

        { clobber::Token::Type::ReshapeKeywordToken, try_parse_tosa_op_expr },
        { clobber::Token::Type::TransposeKeywordToken, try_parse_tosa_op_expr },
        { clobber::Token::Type::TileKeywordToken, try_parse_tosa_op_expr },
        { clobber::Token::Type::SliceKeywordToken, try_parse_tosa_op_expr },
        { clobber::Token::Type::ConcatKeywordToken, try_parse_tosa_op_expr },
        { clobber::Token::Type::IdentityKeywordToken, try_parse_tosa_op_expr },
        { clobber::Token::Type::CastKeywordToken, try_parse_tosa_op_expr },
        { clobber::Token::Type::Conv2dKeywordToken, try_parse_tosa_op_expr },
        { clobber::Token::Type::DepthwiseConv2dKeywordToken, try_parse_tosa_op_expr },
        { clobber::Token::Type::MatmulKeywordToken, try_parse_tosa_op_expr },
        { clobber::Token::Type::FullyConnectedKeywordToken, try_parse_tosa_op_expr },
        { clobber::Token::Type::AvgPool2dKeywordToken, try_parse_tosa_op_expr },
        { clobber::Token::Type::MaxPool2dKeywordToken, try_parse_tosa_op_expr },
        { clobber::Token::Type::PadKeywordToken, try_parse_tosa_op_expr },
        { clobber::Token::Type::ReluKeywordToken, try_parse_tosa_op_expr },
        { clobber::Token::Type::SigmoidKeywordToken, try_parse_tosa_op_expr },
        { clobber::Token::Type::TanhKeywordToken, try_parse_tosa_op_expr },
        { clobber::Token::Type::SoftmaxKeywordToken, try_parse_tosa_op_expr },
    };
    // clang-format on

    // assuming the current token is an open paren token, we peek forward to see if its a keyword
    ParseDelegate parse_fn       = try_parse_call_expr;
    Option<clobber::Token> token = try_get_token(ctx.tokens, ctx.current_idx + 1);
    if (token) {
        const clobber::Token::Type token_type = token.value().type;
        auto it                               = special_form_parse_fns.find(token_type);
        parse_fn                              = it != special_form_parse_fns.end() ? it->second : try_parse_call_expr;
    }

    return parse_fn(ctx);
}

ParseResult
try_parse_let_expr(ParseContext &ctx) {
    Option<clobber::Token> current_token;

    clobber::Token open_paren_token = ctx.tokens[ctx.current_idx++];
    clobber::Token let_token        = ctx.tokens[ctx.current_idx++]; // asserted by caller

    std::expected<std::unique_ptr<clobber::BindingVectorExpr>, Diagnostic> binding_vector_expr_result = try_parse_binding_vector_expr(ctx);
    if (!binding_vector_expr_result) {
        return std::unexpected(binding_vector_expr_result.error());
    }

    std::vector<std::unique_ptr<clobber::Expr>> body_exprs;
    current_token = try_get_token(ctx.tokens, ctx.current_idx);
    while (current_token && current_token.value().type != clobber::Token::Type::CloseParenToken) {
        ParseResult expr = try_parse(ctx);
        if (expr) {
            body_exprs.push_back(std::move(expr.value()));
        }

        current_token = try_get_token(ctx.tokens, ctx.current_idx);
    }

    clobber::Token close_paren_token = ctx.tokens[ctx.current_idx++];
    return std::make_unique<clobber::LetExpr>(open_paren_token, let_token, std::move(binding_vector_expr_result.value()),
                                              std::move(body_exprs), close_paren_token);
}

ParseResult
try_parse_fn_expr(ParseContext &ctx) {
    Option<clobber::Token> current_token;

    clobber::Token open_paren_token = ctx.tokens[ctx.current_idx++];
    clobber::Token fn_token         = ctx.tokens[ctx.current_idx++]; // asserted by caller

    std::expected<std::unique_ptr<clobber::ParameterVectorExpr>, Diagnostic> parameter_vector_expr_result =
        try_parse_parameter_vector_expr(ctx);
    if (!parameter_vector_expr_result) {
        return std::unexpected(parameter_vector_expr_result.error());
    }

    std::vector<std::unique_ptr<clobber::Expr>> body_exprs;
    current_token = try_get_token(ctx.tokens, ctx.current_idx);
    while (current_token && current_token.value().type != clobber::Token::Type::CloseParenToken) {
        ParseResult expr_result = try_parse(ctx);
        if (!expr_result) {
            return expr_result;
        }

        body_exprs.push_back(std::move(expr_result.value()));
        current_token = try_get_token(ctx.tokens, ctx.current_idx);
    }

    clobber::Token close_paren_token = ctx.tokens[ctx.current_idx++];
    return std::make_unique<clobber::FnExpr>(open_paren_token, fn_token, std::move(parameter_vector_expr_result.value()),
                                             std::move(body_exprs), close_paren_token);
}

ParseResult
try_parse_def_expr(ParseContext &ctx) {
    Option<clobber::Token> current_token;

    clobber::Token open_paren_token = ctx.tokens[ctx.current_idx++];
    clobber::Token def_token        = ctx.tokens[ctx.current_idx++]; // asserted by caller

    std::unique_ptr<clobber::IdentifierExpr> identifier;
    ParseResult ident_expr_base_result = try_parse_identifier(ctx);
    if (ident_expr_base_result) {
        std::unique_ptr<clobber::Expr> ident_expr_base = std::move(ident_expr_base_result.value());
        if (auto *raw = dynamic_cast<clobber::IdentifierExpr *>(ident_expr_base.get())) {
            std::unique_ptr<clobber::IdentifierExpr> temp(static_cast<clobber::IdentifierExpr *>(ident_expr_base.release()));
            identifier = std::move(temp);
        }
    }

    ParseResult value_result = try_parse(ctx);
    if (!value_result) {
        return value_result;
    }

    clobber::Token close_paren_token = ctx.tokens[ctx.current_idx++];
    return std::make_unique<clobber::DefExpr>(open_paren_token, def_token, std::move(identifier), std::move(value_result.value()),
                                              close_paren_token);
}

ParseResult
try_parse_do_expr(ParseContext &ctx) {
    Option<clobber::Token> current_token;

    clobber::Token open_paren_token = ctx.tokens[ctx.current_idx++];
    clobber::Token do_token         = ctx.tokens[ctx.current_idx++]; // asserted by caller

    std::vector<std::unique_ptr<clobber::Expr>> body_exprs;
    current_token = try_get_token(ctx.tokens, ctx.current_idx);
    while (current_token && current_token.value().type != clobber::Token::Type::CloseParenToken) {
        ParseResult expr = try_parse(ctx);
        if (expr) {
            body_exprs.push_back(std::move(expr.value()));
        }

        current_token = try_get_token(ctx.tokens, ctx.current_idx);
    }

    clobber::Token close_paren_token = ctx.tokens[ctx.current_idx++];
    return std::make_unique<clobber::DoExpr>(open_paren_token, do_token, std::move(body_exprs), close_paren_token);
}

ParseResult
try_parse_accel_expr(ParseContext &ctx) {
    Option<clobber::Token> current_token;

    const clobber::Token open_paren_token = ctx.tokens[ctx.current_idx++];
    const clobber::Token accel_token      = ctx.tokens[ctx.current_idx++]; // asserted by caller

    std::expected<std::unique_ptr<clobber::BindingVectorExpr>, Diagnostic> binding_vector_expr_result = try_parse_binding_vector_expr(ctx);
    if (!binding_vector_expr_result) {
        return std::unexpected(binding_vector_expr_result.error());
    }

    std::vector<std::unique_ptr<clobber::Expr>> body_exprs;
    current_token = try_get_token(ctx.tokens, ctx.current_idx);
    while (current_token && current_token.value().type != clobber::Token::Type::CloseParenToken) {
        ParseResult expr = try_parse(ctx);
        if (expr) {
            body_exprs.push_back(std::move(expr.value()));
        }

        current_token = try_get_token(ctx.tokens, ctx.current_idx);
    }

    const clobber::Token close_paren_token = current_token.value();
    ctx.current_idx++;
    return std::make_unique<clobber::accel::AccelExpr>(open_paren_token, accel_token, std::move(binding_vector_expr_result.value()),
                                                       std::move(body_exprs), close_paren_token);
}

ParseResult
try_parse_tensor_expr(ParseContext &ctx) {
    clobber::Token open_paren_token = ctx.tokens[ctx.current_idx++];
    clobber::Token tensor_token     = ctx.tokens[ctx.current_idx++];

    Option<clobber::Token> current_token = try_get_token(ctx.tokens, ctx.current_idx);

    std::vector<std::unique_ptr<clobber::Expr>> arguments;
    while (current_token && current_token.value().type != clobber::Token::Type::CloseParenToken) {
        ParseResult arg_expr = try_parse(ctx);
        if (arg_expr) {
            arguments.push_back(std::move(arg_expr.value()));
        }

        current_token = try_get_token(ctx.tokens, ctx.current_idx);
    }

    clobber::Token close_paren_token = current_token.value();

    ctx.current_idx++;
    return std::make_unique<clobber::accel::TensorToken>(open_paren_token, tensor_token, std::move(arguments), close_paren_token);
}

ParseResult
try_parse_tosa_op_expr(ParseContext &ctx) {
    clobber::Token open_paren_token = ctx.tokens[ctx.current_idx++];
    clobber::Token op_token         = ctx.tokens[ctx.current_idx++];

    Option<clobber::Token> current_token = try_get_token(ctx.tokens, ctx.current_idx);

    std::vector<std::unique_ptr<clobber::Expr>> arguments;
    while (current_token && current_token.value().type != clobber::Token::Type::CloseParenToken) {
        ParseResult arg_expr = try_parse(ctx);
        if (arg_expr) {
            arguments.push_back(std::move(arg_expr.value()));
        }

        current_token = try_get_token(ctx.tokens, ctx.current_idx);
    }

    clobber::Token close_paren_token = current_token.value();

    ctx.current_idx++;
    return std::make_unique<clobber::accel::TOSAOpExpr>(open_paren_token, op_token, std::move(arguments), close_paren_token);
}

ParseResult
try_parse_vector_expr(ParseContext &ctx) {
    clobber::Token open_bracket_token = ctx.tokens[ctx.current_idx++];

    std::vector<std::unique_ptr<clobber::Expr>> values;
    std::vector<clobber::Token> commas;

    Option<clobber::Token> current_token = try_get_token(ctx.tokens, ctx.current_idx);
    while (current_token && current_token.value().type != clobber::Token::Type::CloseBracketToken) {
        ParseResult value_result = try_parse(ctx);
        if (value_result) {
            values.push_back(std::move(value_result.value()));
        }

        current_token = try_get_token(ctx.tokens, ctx.current_idx);
        // TODO: Add support for commas here
    }

    clobber::Token close_bracket_token = current_token.value();

    ctx.current_idx++;
    return std::make_unique<clobber::VectorExpr>(open_bracket_token, std::move(values), std::move(commas), close_bracket_token);
}

ParseResult
try_parse_keyword_literal_expr(ParseContext &ctx) {
    clobber::Token token   = ctx.tokens[ctx.current_idx];                                     // guaranteed to exist by caller
    const std::string name = ctx.source_text.substr(token.span.start + 1, token.span.length); // +1 to ignore the colon
    return std::make_unique<clobber::KeywordLiteralExpr>(name, ctx.tokens[ctx.current_idx++]);
}

ParseResult
try_parse_call_expr(ParseContext &ctx) {
    clobber::Token close_paren_token;
    clobber::Token open_paren_token = ctx.tokens[ctx.current_idx++];

    ParseResult operator_expr = try_parse(ctx);
    if (!operator_expr) {
        return operator_expr;
    }

    Option<clobber::Token> current_token = try_get_token(ctx.tokens, ctx.current_idx);

    std::vector<std::unique_ptr<clobber::Expr>> arguments;
    while (current_token && current_token.value().type != clobber::Token::Type::CloseParenToken) {
        ParseResult arg_expr = try_parse(ctx);
        if (arg_expr) {
            arguments.push_back(std::move(arg_expr.value()));
        }

        current_token = try_get_token(ctx.tokens, ctx.current_idx);
    }

    if (current_token && current_token.value().type == clobber::Token::Type::CloseParenToken) {
        close_paren_token = current_token.value();
    }

    ctx.current_idx++;
    return std::make_unique<clobber::CallExpr>(open_paren_token, std::move(operator_expr.value()), std::move(arguments), close_paren_token);
}

ParseResult
try_parse(ParseContext &ctx) {
    clobber::Token current_token;
    Option<clobber::Token> token_opt;
    ParseDelegate parse_fn;
    Option<ParseDelegate> parse_fn_opt;

    token_opt = try_get_token(ctx.tokens, ctx.current_idx);
    if (!token_opt) {
        return nullptr; // silently error
    }
    current_token = token_opt.value();

    parse_fn_opt = try_get_parse_fun(current_token.type);
    if (!parse_fn_opt) {
        recover(Recovery::SeekCloseParen, ctx);
        const std::string token_str = std::string(magic_enum::enum_name(current_token.type));
        const std::string err_msg   = std::format("Could not find a parse function for the token type `{}`", token_str);
        return std::unexpected(diag::parser::internal_err(current_token.span, err_msg));
    }
    parse_fn = parse_fn_opt.value();
    return parse_fn(ctx);
}

std::unique_ptr<clobber::CompilationUnit>
clobber::parse(const std::string &source_text, const std::vector<clobber::Token> &tokens, std::vector<clobber::Diagnostic> &diagnostics) {
    // clang-format off
    ParseContext ctx{
        .source_text = source_text,
        .tokens = tokens,
        .current_idx = 0,
        .diagnostics = diagnostics
    }; // clang-format on

    std::vector<std::unique_ptr<clobber::Expr>> exprs;
    std::vector<clobber::Diagnostic> parse_errors;
    while (ctx.current_idx < tokens.size()) {
        if (tokens[ctx.current_idx].type == clobber::Token::Type::EofToken) {
            break;
        }

        // 'current_idx' passed by reference, implicitly modified
        ParseResult parsed_expr = try_parse(ctx);
        if (parsed_expr) {
            exprs.push_back(std::move(parsed_expr.value()));
        }
    }

    return std::make_unique<clobber::CompilationUnit>(source_text, std::move(exprs), diagnostics);
}