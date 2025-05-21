#include <magic_enum/magic_enum.hpp>

#include "clobber/pch.hpp"

#include <clobber/common/diagnostic.hpp>
#include <clobber/common/utils.hpp>

#include "clobber/ast.hpp"
#include "clobber/parser.hpp"

#include "clobber/internal/diagnostic_factory.hpp"

struct ParseContext {
    const std::string &source_text;
    const std::vector<clobber::Token> &tokens;

    size_t current_idx;
    std::vector<clobber::Diagnostic> &diagnostics;
};

template <typename T> using Option = std::optional<T>;

// We deal with raw pointers because we need to perform post processing on returned expr types.
// For example, we need to cast 'Expr' to some subtype to fit the properties of the expr that we're trying to parse.
// Dealing with raw pointers makes this infinitely easier - as long as the caller always wraps it in a unique ptr, which isn't that hard to
// assert.
using ParseDelegate = clobber::Expr *(*)(ParseContext &);

// macro to help defining forward declarations
#define PARSE_DELEGATE_FN(FN_NAME) clobber::Expr *FN_NAME(ParseContext &);

PARSE_DELEGATE_FN(try_parse)
PARSE_DELEGATE_FN(try_parse_call_expr)
PARSE_DELEGATE_FN(try_parse_identifier)
PARSE_DELEGATE_FN(try_parse_call_expr_or_special_form)
PARSE_DELEGATE_FN(try_parse_let_expr)
PARSE_DELEGATE_FN(try_parse_fn_expr)
PARSE_DELEGATE_FN(try_parse_def_expr)
PARSE_DELEGATE_FN(try_parse_do_expr)

PARSE_DELEGATE_FN(try_parse_numeric_literal_expr)
PARSE_DELEGATE_FN(try_parse_string_literal_expr)
PARSE_DELEGATE_FN(try_parse_char_literal_expr)

// accel specific syntax
PARSE_DELEGATE_FN(try_parse_accel_expr)
PARSE_DELEGATE_FN(try_parse_matmul_expr)
PARSE_DELEGATE_FN(try_parse_relu_expr)

Option<clobber::Token>
try_get_token(const std::vector<clobber::Token> &tokens, size_t idx) {
    size_t tokens_len = tokens.size();
    return (idx >= 0 && idx < tokens_len) ? std::make_optional(tokens[idx]) : std::nullopt;
}

void
vector_recover(const std::vector<clobber::Token> &tokens, size_t &idx) {
    Option<clobber::Token> current_token = try_get_token(tokens, idx);
    while (current_token && current_token.value().type != clobber::Token::Type::CloseBracketToken) {
        current_token = try_get_token(tokens, idx++);
    }
}

void
recover(const std::vector<clobber::Token> &, size_t &idx) {
    idx = idx + 1;
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

    } else {
        return std::nullopt;
    }
}

clobber::TypeExpr *try_parse_builtin_type_expr(const clobber::Token &, ParseContext &);
clobber::TypeExpr *try_parse_user_defined_type_expr(const clobber::Token &, ParseContext &);
clobber::TypeExpr *try_parse_parameterized_type_expr(clobber::TypeExpr *, ParseContext &);

// remarks:
// assumes that the current char is the caret token.
clobber::TypeExpr *
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

    clobber::TypeExpr *type_expr;
    if (is_type_keyword_token(current_token.type)) {
        type_expr = try_parse_builtin_type_expr(caret_token, ctx);
    } else if (current_token.type == clobber::Token::Type::IdentifierToken) {
        type_expr = try_parse_builtin_type_expr(caret_token, ctx);
    } else {
        // TODO: throw some bullshit here
        return nullptr;
    }

    current_token = ctx.tokens[ctx.current_idx];
    return ctx.tokens[ctx.current_idx].type == clobber::Token::Type::LessThanToken ? try_parse_parameterized_type_expr(type_expr, ctx)
                                                                                   : type_expr;
}

clobber::TypeExpr *
try_parse_builtin_type_expr(const clobber::Token &caret_token, ParseContext &ctx) {
    clobber::Token type_keyword_token = ctx.tokens[ctx.current_idx++]; // asserted by caller
    return new clobber::BuiltinTypeExpr(caret_token, type_keyword_token);
}

clobber::TypeExpr *
try_parse_user_defined_type_expr(const clobber::Token &caret_token, ParseContext &ctx) {
    clobber::Token identifier_token = ctx.tokens[ctx.current_idx++]; // asserted by caller
    return new clobber::UserDefinedTypeExpr(caret_token, identifier_token);
}

clobber::TypeExpr *
try_parse_parameterized_type_expr(clobber::TypeExpr *type_expr, ParseContext &ctx) {
    clobber::Token lt_token = ctx.tokens[ctx.current_idx++];

    std::vector<std::unique_ptr<clobber::Expr>> pvalues;
    std::vector<clobber::Token> commas;
    while (ctx.current_idx < ctx.tokens.size()) {
        auto param_value = try_parse(ctx);
        if (!param_value) {
            return nullptr;
        }

        pvalues.push_back(std::unique_ptr<clobber::Expr>(param_value));

        clobber::Token current_token = ctx.tokens[ctx.current_idx];
        if (current_token.type == clobber::Token::Type::GreaterThanToken) {
            clobber::Token gt_token = ctx.tokens[ctx.current_idx++];
            auto type_expr_uptr     = std::unique_ptr<clobber::TypeExpr>(type_expr);
            return new clobber::ParameterizedTypeExpr(std::move(type_expr_uptr), lt_token, std::move(pvalues), commas, gt_token);

        } else if (current_token.type == clobber::Token::Type::CommaToken) {
            commas.push_back(ctx.tokens[ctx.current_idx++]);

        } else {
            // TODO: throw some bullshit here
            return nullptr;
        }
    }
}

clobber::Expr *
try_parse_numeric_literal_expr(ParseContext &ctx) {
    return new clobber::NumLiteralExpr(ctx.tokens[ctx.current_idx++]); // no bounds check, current token exists, asserted by caller
}

clobber::Expr *
try_parse_string_literal_expr(ParseContext &ctx) {
    clobber::Token token = ctx.tokens[ctx.current_idx];
    std::string str      = token.ExtractText(ctx.source_text);
    if (str.size() > 2) {
        str = str.substr(1, str.size() - 2);
    }

    return new clobber::StringLiteralExpr(str, ctx.tokens[ctx.current_idx++]);
}

clobber::Expr *
try_parse_char_literal_expr(ParseContext &ctx) {
    clobber::Token token = ctx.tokens[ctx.current_idx];
    std::string str      = token.ExtractText(ctx.source_text);
    if (str.size() > 2) {
        str = str.substr(1, str.size() - 2);
    }

    return new clobber::CharLiteralExpr(str, ctx.tokens[ctx.current_idx++]); // no bounds check, current token exists, asserted by caller
}

clobber::Expr *
try_parse_identifier(ParseContext &ctx) {
    clobber::Token token                = ctx.tokens[ctx.current_idx]; // guaranteed to exist by caller
    const std::string name              = ctx.source_text.substr(token.span.start, token.span.length);
    clobber::IdentifierExpr *ident_expr = new clobber::IdentifierExpr(name, ctx.tokens[ctx.current_idx]);
    ctx.current_idx++;
    return ident_expr;
}

clobber::BindingVectorExpr *
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

        clobber::Expr *ident_expr_base = try_parse_identifier(ctx);
        if (ident_expr_base) {
            clobber::IdentifierExpr *ident_expr = dynamic_cast<clobber::IdentifierExpr *>(ident_expr_base);
            identifiers.push_back(std::unique_ptr<clobber::IdentifierExpr>(ident_expr));
        }

        clobber::Expr *value_expr = try_parse(ctx);
        if (value_expr) {
            exprs.push_back(std::unique_ptr<clobber::Expr>(value_expr));
        }

        num_bindings++;
    }

    close_bracket_token = ctx.tokens[ctx.current_idx++];
    return new clobber::BindingVectorExpr(open_bracket_token, std::move(identifiers), std::move(exprs), close_bracket_token, num_bindings);
}

clobber::ParameterVectorExpr *
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

        clobber::Expr *ident_expr_base = try_parse_identifier(ctx);
        if (ident_expr_base) {
            clobber::IdentifierExpr *ident_expr = dynamic_cast<clobber::IdentifierExpr *>(ident_expr_base);
            identifiers.push_back(std::unique_ptr<clobber::IdentifierExpr>(ident_expr));
        }
    }

    close_bracket_token = ctx.tokens[ctx.current_idx++];
    return new clobber::ParameterVectorExpr(open_bracket_token, std::move(identifiers), close_bracket_token);
}

clobber::Expr *
try_parse_call_expr_or_special_form(ParseContext &ctx) {
    // clang-format off
    const std::unordered_map<clobber::Token::Type, ParseDelegate> special_form_parse_fns = {
        {clobber::Token::Type::LetKeywordToken, try_parse_let_expr},
        {clobber::Token::Type::FnKeywordToken, try_parse_fn_expr},
        {clobber::Token::Type::DefKeywordToken, try_parse_def_expr},
        {clobber::Token::Type::DoKeywordToken, try_parse_do_expr},
        {clobber::Token::Type::AccelKeywordToken, try_parse_accel_expr},

        {clobber::Token::Type::MatmulKeywordToken, try_parse_matmul_expr},
        {clobber::Token::Type::ReluKeywordToken, try_parse_relu_expr},
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

clobber::Expr *
try_parse_let_expr(ParseContext &ctx) {
    clobber::Token open_paren_token;
    clobber::Token close_paren_token;
    clobber::Token let_token;
    std::unique_ptr<clobber::BindingVectorExpr> binding_vector_expr;
    std::vector<std::unique_ptr<clobber::Expr>> body_exprs;
    Option<clobber::Token> current_token;

    open_paren_token    = ctx.tokens[ctx.current_idx++];
    let_token           = ctx.tokens[ctx.current_idx++]; // asserted by caller
    binding_vector_expr = std::unique_ptr<clobber::BindingVectorExpr>(try_parse_binding_vector_expr(ctx));

    current_token = try_get_token(ctx.tokens, ctx.current_idx);
    while (current_token && current_token.value().type != clobber::Token::Type::CloseParenToken) {
        clobber::Expr *expr = try_parse(ctx);
        if (expr) {
            body_exprs.push_back(std::unique_ptr<clobber::Expr>(expr));
        }

        current_token = try_get_token(ctx.tokens, ctx.current_idx);
    }

    close_paren_token = ctx.tokens[ctx.current_idx++];
    return new clobber::LetExpr(open_paren_token, let_token, std::move(binding_vector_expr), std::move(body_exprs), close_paren_token);
}

clobber::Expr *
try_parse_fn_expr(ParseContext &ctx) {
    clobber::Token open_paren_token;
    clobber::Token close_paren_token;
    clobber::Token fn_token;
    std::unique_ptr<clobber::ParameterVectorExpr> parameter_vector_expr;
    std::vector<std::unique_ptr<clobber::Expr>> body_exprs;
    Option<clobber::Token> current_token;

    open_paren_token = ctx.tokens[ctx.current_idx++];
    fn_token         = ctx.tokens[ctx.current_idx++]; // asserted by caller

    parameter_vector_expr = std::unique_ptr<clobber::ParameterVectorExpr>(try_parse_parameter_vector_expr(ctx));

    current_token = try_get_token(ctx.tokens, ctx.current_idx);
    while (current_token && current_token.value().type != clobber::Token::Type::CloseParenToken) {
        clobber::Expr *expr = try_parse(ctx);
        if (!expr) {
            return nullptr;
        }

        body_exprs.push_back(std::unique_ptr<clobber::Expr>(expr));
        current_token = try_get_token(ctx.tokens, ctx.current_idx);
    }

    close_paren_token = ctx.tokens[ctx.current_idx++];
    return new clobber::FnExpr(open_paren_token, fn_token, std::move(parameter_vector_expr), std::move(body_exprs), close_paren_token);
}

clobber::Expr *
try_parse_def_expr(ParseContext &ctx) {
    clobber::Token open_paren_token;
    clobber::Token close_paren_token;
    clobber::Token def_token;
    std::unique_ptr<clobber::IdentifierExpr> identifier;
    std::unique_ptr<clobber::Expr> value;
    Option<clobber::Token> current_token;

    open_paren_token = ctx.tokens[ctx.current_idx++];
    def_token        = ctx.tokens[ctx.current_idx++]; // asserted by caller

    clobber::Expr *ident_expr_base = try_parse_identifier(ctx);
    if (ident_expr_base) {
        clobber::IdentifierExpr *ident_expr = dynamic_cast<clobber::IdentifierExpr *>(ident_expr_base);
        identifier                          = std::unique_ptr<clobber::IdentifierExpr>(ident_expr);
    }

    clobber::Expr *value_expr = try_parse(ctx);
    if (value_expr) {
        value = std::unique_ptr<clobber::Expr>(value_expr);
    }

    close_paren_token = ctx.tokens[ctx.current_idx++];
    return new clobber::DefExpr(open_paren_token, def_token, std::move(identifier), std::move(value), close_paren_token);
}

clobber::Expr *
try_parse_do_expr(ParseContext &ctx) {
    clobber::Token open_paren_token;
    clobber::Token close_paren_token;
    clobber::Token do_token;
    std::vector<std::unique_ptr<clobber::Expr>> body_exprs;
    Option<clobber::Token> current_token;

    open_paren_token = ctx.tokens[ctx.current_idx++];
    do_token         = ctx.tokens[ctx.current_idx++]; // asserted by caller

    current_token = try_get_token(ctx.tokens, ctx.current_idx);
    while (current_token && current_token.value().type != clobber::Token::Type::CloseParenToken) {
        clobber::Expr *expr = try_parse(ctx);
        if (expr) {
            body_exprs.push_back(std::unique_ptr<clobber::Expr>(expr));
        }

        current_token = try_get_token(ctx.tokens, ctx.current_idx);
    }

    close_paren_token = ctx.tokens[ctx.current_idx++];
    return new clobber::DoExpr(open_paren_token, do_token, std::move(body_exprs), close_paren_token);
}

clobber::Expr *
try_parse_accel_expr(ParseContext &ctx) {
    clobber::Token open_paren_token;
    clobber::Token close_paren_token;
    clobber::Token accel_token;
    std::unique_ptr<clobber::BindingVectorExpr> binding_vector_expr;
    std::vector<std::unique_ptr<clobber::Expr>> body_exprs;
    Option<clobber::Token> current_token;

    open_paren_token = ctx.tokens[ctx.current_idx++];
    accel_token      = ctx.tokens[ctx.current_idx++]; // asserted by caller

    clobber::BindingVectorExpr *raw_binding_vector_expr = try_parse_binding_vector_expr(ctx);
    if (binding_vector_expr) {
        binding_vector_expr = std::unique_ptr<clobber::BindingVectorExpr>(raw_binding_vector_expr);
    }

    current_token = try_get_token(ctx.tokens, ctx.current_idx);
    while (current_token && current_token.value().type != clobber::Token::Type::CloseParenToken) {
        clobber::Expr *expr = try_parse(ctx);
        if (expr) {
            body_exprs.push_back(std::unique_ptr<clobber::Expr>(expr));
        }

        current_token = try_get_token(ctx.tokens, ctx.current_idx);
    }

    close_paren_token = ctx.tokens[ctx.current_idx++];
    return new clobber::accel::AccelExpr(open_paren_token, accel_token, std::move(binding_vector_expr), std::move(body_exprs),
                                         close_paren_token);
}

clobber::Expr *
try_parse_matmul_expr(ParseContext &ctx) {
    clobber::Token open_paren_token;
    clobber::Token close_paren_token;
    clobber::Token mat_mul_token;
    std::unique_ptr<clobber::Expr> fst_operand;
    std::unique_ptr<clobber::Expr> snd_operand;
    Option<clobber::Token> current_token;

    open_paren_token = ctx.tokens[ctx.current_idx++];
    mat_mul_token    = ctx.tokens[ctx.current_idx++]; // asserted by caller

    clobber::Expr *fst_operand_raw = try_parse(ctx);
    if (fst_operand) {
        fst_operand = std::unique_ptr<clobber::Expr>(fst_operand_raw);
    }

    clobber::Expr *snd_operand_raw = try_parse(ctx);
    if (snd_operand) {
        snd_operand = std::unique_ptr<clobber::Expr>(snd_operand_raw);
    }

    close_paren_token = ctx.tokens[ctx.current_idx++];
    return new clobber::accel::MatMulExpr(open_paren_token, mat_mul_token, std::move(fst_operand), std::move(snd_operand),
                                          close_paren_token);
}

clobber::Expr *
try_parse_relu_expr(ParseContext &ctx) {
    clobber::Token open_paren_token;
    clobber::Token close_paren_token;
    clobber::Token relu_token;
    std::unique_ptr<clobber::Expr> operand;
    Option<clobber::Token> current_token;

    open_paren_token = ctx.tokens[ctx.current_idx++];
    relu_token       = ctx.tokens[ctx.current_idx++]; // asserted by caller

    clobber::Expr *operand_raw = try_parse(ctx);
    if (operand) {
        operand = std::unique_ptr<clobber::Expr>(operand_raw);
    }

    if (ctx.tokens[ctx.current_idx + 1].type != clobber::Token::Type::CloseParenToken) {

        recover(ctx.tokens, ctx.current_idx);
        return nullptr;
    }

    close_paren_token = ctx.tokens[ctx.current_idx++];
    return new clobber::accel::RelUExpr(open_paren_token, relu_token, std::move(operand), close_paren_token);
}

clobber::Expr *
try_parse_call_expr(ParseContext &ctx) {
    clobber::Token open_paren_token;
    clobber::Token close_paren_token;
    std::vector<std::unique_ptr<clobber::Expr>> arguments;
    Option<clobber::Token> current_token;

    open_paren_token = ctx.tokens[ctx.current_idx++];

    clobber::Expr *operator_expr = try_parse(ctx);
    if (!operator_expr) {
        // TODO: throw some bullshit here
        return nullptr;
    }

    current_token = try_get_token(ctx.tokens, ctx.current_idx);

    while (current_token && current_token.value().type != clobber::Token::Type::CloseParenToken) {
        clobber::Expr *arg_expr = try_parse(ctx);
        if (arg_expr) {
            arguments.push_back(std::unique_ptr<clobber::Expr>(arg_expr));
        }

        current_token = try_get_token(ctx.tokens, ctx.current_idx);
    }

    if (current_token && current_token.value().type == clobber::Token::Type::CloseParenToken) {
        close_paren_token = current_token.value();
    }

    ctx.current_idx++;
    return new clobber::CallExpr(open_paren_token, std::unique_ptr<clobber::Expr>(operator_expr), std::move(arguments), close_paren_token);
}

clobber::Expr *
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
        const std::string token_str = std::string(magic_enum::enum_name(current_token.type));
        const std::string err_msg   = std::format("Could not find a parse function for the token type `{}`", token_str);
        auto err                    = diag::parser::internal_err(current_token.span.start, current_token.span.length, err_msg);
        ctx.diagnostics.push_back(err);
        recover(ctx.tokens, ctx.current_idx);
        return nullptr;
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
        clobber::Expr *parsed_expr = try_parse(ctx);
        if (parsed_expr) {
            exprs.push_back(std::unique_ptr<clobber::Expr>(parsed_expr));
        }
    }

    return std::make_unique<clobber::CompilationUnit>(source_text, std::move(exprs), diagnostics);
}