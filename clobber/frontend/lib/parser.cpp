#include "clobber/pch.hpp"

#include <clobber/common/diagnostic.hpp>
#include <clobber/common/utils.hpp>

#include "clobber/ast.hpp"
#include "clobber/parser.hpp"

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
    // keep using if-statements instead of a map/switch to support predicates over different token types
    const std::unordered_set<clobber::Token::Type> valid_token_types = {clobber::Token::Type::NumericLiteralToken};
    auto is_reserved_symbol_identifier = [&valid_token_types](clobber::Token::Type tt) { return valid_token_types.contains(tt); };

    if (token_type == clobber::Token::Type::NumericLiteralToken) {
        return std::make_optional(try_parse_numeric_literal_expr);

    } else if (token_type == clobber::Token::Type::OpenParenToken) {
        return std::make_optional(try_parse_call_expr_or_special_form);

    } else if (token_type == clobber::Token::Type::IdentifierToken || is_reserved_symbol_identifier(token_type)) {
        return std::make_optional(try_parse_identifier);

    } else if (token_type == clobber::Token::Type::StringLiteralToken) {
        return std::make_optional(try_parse_string_literal_expr);

    } else if (token_type == clobber::Token::Type::CharLiteralToken) {
        return std::make_optional(try_parse_char_literal_expr);

    } else {
        return std::nullopt;
    }
}

clobber::Expr *
try_parse_numeric_literal_expr(ParseContext &ctx) {
    clobber::NumLiteralExpr *nle;
    clobber::Token current_token;

    current_token = ctx.tokens[ctx.current_idx]; // no bounds check, current token exists, asserted by caller

    // nle        = new NumLiteralExpr();
    nle->token = current_token;
    nle->type  = clobber::Expr::Type::NumericLiteralExpr;

    ctx.current_idx++;
    return nle;
}

clobber::Expr *
try_parse_string_literal_expr(ParseContext &ctx) {
    clobber::StringLiteralExpr *sle;
    // sle = new StringLiteralExpr();

    clobber::Token token = ctx.tokens[ctx.current_idx];
    std::string str      = token.ExtractText(ctx.source_text);
    if (str.size() > 2) {
        str = str.substr(1, str.size() - 2);
    }

    sle->value = str;
    sle->token = ctx.tokens[ctx.current_idx]; // no bounds check, current token exists, asserted by caller
    sle->type  = clobber::Expr::Type::StringLiteralExpr;

    ctx.current_idx++;
    return sle;
}

clobber::Expr *
try_parse_char_literal_expr(ParseContext &ctx) {
    clobber::CharLiteralExpr *cle;
    // cle = new CharLiteralExpr();

    clobber::Token token = ctx.tokens[ctx.current_idx];
    std::string str      = token.ExtractText(ctx.source_text);
    if (str.size() > 2) {
        str = str.substr(1, str.size() - 2);
    }

    cle->value = str;
    cle->token = ctx.tokens[ctx.current_idx]; // no bounds check, current token exists, asserted by caller
    cle->type  = clobber::Expr::Type::CharLiteralExpr;

    ctx.current_idx++;
    return cle;
}

clobber::Expr *
try_parse_identifier(ParseContext &ctx) {
    clobber::IdentifierExpr *iden_expr;
    clobber::Token token = ctx.tokens[ctx.current_idx]; // guaranteed to exist by caller

    // iden_expr            = new IdentifierExpr();
    iden_expr->name  = ctx.source_text.substr(token.start, token.length);
    iden_expr->token = token;
    iden_expr->type  = clobber::Expr::Type::IdentifierExpr;

    ctx.current_idx++;
    return iden_expr;
}

clobber::BindingVectorExpr *
try_parse_binding_vector_expr(ParseContext &ctx) {
    clobber::BindingVectorExpr *binding_vector_expr;
    // binding_vector_expr = new BindingVectorExpr();
    Option<clobber::Token> current_token;

    // binding_vector_expr->type               = clobber::Expr::Type::BindingVectorExpr;
    binding_vector_expr->open_bracket_token = ctx.tokens[ctx.current_idx++];

    while (true) {
        current_token = try_get_token(ctx.tokens, ctx.current_idx);
        if (current_token && current_token.value().type == clobber::Token::Type::CloseBracketToken) {
            break;
        }

        clobber::Expr *ident_expr_base = try_parse_identifier(ctx);
        if (ident_expr_base) {
            clobber::IdentifierExpr *ident_expr = dynamic_cast<clobber::IdentifierExpr *>(ident_expr_base);
            binding_vector_expr->identifiers.push_back(std::unique_ptr<clobber::IdentifierExpr>(ident_expr));
        }

        clobber::Expr *value_expr = try_parse(ctx);
        if (value_expr) {
            binding_vector_expr->exprs.push_back(std::unique_ptr<clobber::Expr>(value_expr));
        }

        binding_vector_expr->num_bindings++;
    }

    binding_vector_expr->close_bracket_token = ctx.tokens[ctx.current_idx++];
    return binding_vector_expr;
}

clobber::ParameterVectorExpr *
try_parse_parameter_vector_expr(ParseContext &ctx) {
    clobber::ParameterVectorExpr *parameter_vector_expr;
    // parameter_vector_expr = new ParameterVectorExpr();
    Option<clobber::Token> current_token;

    // parameter_vector_expr->expr_type          = ClobberExprType::ParameterVectorExpr;
    parameter_vector_expr->open_bracket_token = ctx.tokens[ctx.current_idx++];

    while (true) {
        current_token = try_get_token(ctx.tokens, ctx.current_idx);
        if (current_token && current_token.value().type == clobber::Token::Type::CloseBracketToken) {
            break;
        }

        clobber::Expr *ident_expr_base = try_parse_identifier(ctx);
        if (ident_expr_base) {
            clobber::IdentifierExpr *ident_expr = dynamic_cast<clobber::IdentifierExpr *>(ident_expr_base);
            parameter_vector_expr->identifiers.push_back(std::unique_ptr<clobber::IdentifierExpr>(ident_expr));
        }
    }

    parameter_vector_expr->close_bracket_token = ctx.tokens[ctx.current_idx++];
    return parameter_vector_expr;
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
        {clobber::Token::Type::TosaMatmulKeywordToken, try_parse_matmul_expr},
        {clobber::Token::Type::TosaReluKeywordToken, try_parse_relu_expr},
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
    clobber::LetExpr *let_expr;
    // let_expr = new LetExpr();
    clobber::BindingVectorExpr *binding_vector_expr;
    Option<clobber::Token> current_token;

    let_expr->type             = clobber::Expr::Type::LetExpr;
    let_expr->open_paren_token = ctx.tokens[ctx.current_idx++];
    let_expr->let_token        = ctx.tokens[ctx.current_idx++]; // asserted by caller

    binding_vector_expr           = try_parse_binding_vector_expr(ctx);
    let_expr->binding_vector_expr = std::unique_ptr<clobber::BindingVectorExpr>(binding_vector_expr);

    current_token = try_get_token(ctx.tokens, ctx.current_idx);
    while (current_token && current_token.value().type != clobber::Token::Type::CloseParenToken) {
        clobber::Expr *expr = try_parse(ctx);
        if (expr) {
            let_expr->body_exprs.push_back(std::unique_ptr<clobber::Expr>(expr));
        }

        current_token = try_get_token(ctx.tokens, ctx.current_idx);
    }

    let_expr->close_paren_token = ctx.tokens[ctx.current_idx++];
    return let_expr;
}

clobber::Expr *
try_parse_fn_expr(ParseContext &ctx) {
    clobber::FnExpr *fn_expr;
    // fn_expr         = new FnExpr();
    clobber::ParameterVectorExpr *parameter_vector_expr;
    Option<clobber::Token> current_token;

    fn_expr->type             = clobber::Expr::Type::FnExpr;
    fn_expr->open_paren_token = ctx.tokens[ctx.current_idx++];
    fn_expr->fn_token         = ctx.tokens[ctx.current_idx++]; // asserted by caller

    parameter_vector_expr          = try_parse_parameter_vector_expr(ctx);
    fn_expr->parameter_vector_expr = std::unique_ptr<clobber::ParameterVectorExpr>(parameter_vector_expr);

    current_token = try_get_token(ctx.tokens, ctx.current_idx);
    while (current_token && current_token.value().type != clobber::Token::Type::CloseParenToken) {
        clobber::Expr *expr = try_parse(ctx);
        if (expr) {
            fn_expr->body_exprs.push_back(std::unique_ptr<clobber::Expr>(expr));
        }

        current_token = try_get_token(ctx.tokens, ctx.current_idx);
    }

    fn_expr->close_paren_token = ctx.tokens[ctx.current_idx++];
    return fn_expr;
}

clobber::Expr *
try_parse_def_expr(ParseContext &ctx) {
    clobber::DefExpr *def_expr;
    // def_expr = new DefExpr();
    Option<clobber::Token> current_token;

    def_expr->type             = clobber::Expr::Type::DefExpr;
    def_expr->open_paren_token = ctx.tokens[ctx.current_idx++];
    def_expr->def_token        = ctx.tokens[ctx.current_idx++]; // asserted by caller

    clobber::Expr *ident_expr_base = try_parse_identifier(ctx);
    if (ident_expr_base) {
        clobber::IdentifierExpr *ident_expr = dynamic_cast<clobber::IdentifierExpr *>(ident_expr_base);
        def_expr->identifier                = std::unique_ptr<clobber::IdentifierExpr>(ident_expr);
    }

    clobber::Expr *value_expr = try_parse(ctx);
    if (value_expr) {
        def_expr->value = std::unique_ptr<clobber::Expr>(value_expr);
    }

    def_expr->close_paren_token = ctx.tokens[ctx.current_idx++];
    return def_expr;
}

clobber::Expr *
try_parse_do_expr(ParseContext &ctx) {
    clobber::DoExpr *do_expr;
    // do_expr = new DoExpr();
    Option<clobber::Token> current_token;

    do_expr->type             = clobber::Expr::Type::DoExpr;
    do_expr->open_paren_token = ctx.tokens[ctx.current_idx++];
    do_expr->do_token         = ctx.tokens[ctx.current_idx++]; // asserted by caller

    current_token = try_get_token(ctx.tokens, ctx.current_idx);
    while (current_token && current_token.value().type != clobber::Token::Type::CloseParenToken) {
        clobber::Expr *expr = try_parse(ctx);
        if (expr) {
            do_expr->body_exprs.push_back(std::unique_ptr<clobber::Expr>(expr));
        }

        current_token = try_get_token(ctx.tokens, ctx.current_idx);
    }

    do_expr->close_paren_token = ctx.tokens[ctx.current_idx++];
    return do_expr;
}

clobber::Expr *
try_parse_accel_expr(ParseContext &ctx) {
    clobber::accel::AccelExpr *accel_expr;
    // accel_expr = new AccelExpr();
    Option<clobber::Token> current_token;

    accel_expr->type             = clobber::Expr::Type::AccelExpr;
    accel_expr->open_paren_token = ctx.tokens[ctx.current_idx++];
    accel_expr->accel_token      = ctx.tokens[ctx.current_idx++]; // asserted by caller

    clobber::BindingVectorExpr *binding_vector_expr = try_parse_binding_vector_expr(ctx);
    if (binding_vector_expr) {
        accel_expr->binding_vector_expr = std::unique_ptr<clobber::BindingVectorExpr>(binding_vector_expr);
    }

    current_token = try_get_token(ctx.tokens, ctx.current_idx);
    while (current_token && current_token.value().type != clobber::Token::Type::CloseParenToken) {
        clobber::Expr *expr = try_parse(ctx);
        if (expr) {
            accel_expr->body_exprs.push_back(std::unique_ptr<clobber::Expr>(expr));
        }

        current_token = try_get_token(ctx.tokens, ctx.current_idx);
    }

    accel_expr->close_paren_token = ctx.tokens[ctx.current_idx++];
    return accel_expr;
}

clobber::Expr *
try_parse_matmul_expr(ParseContext &ctx) {
    clobber::accel::MatMulExpr *matmul_expr;
    // matmul_expr = new MatMulExpr();
    Option<clobber::Token> current_token;

    matmul_expr->type             = clobber::Expr::Type::MatMulExpr;
    matmul_expr->open_paren_token = ctx.tokens[ctx.current_idx++];
    matmul_expr->mat_mul_token    = ctx.tokens[ctx.current_idx++]; // asserted by caller

    clobber::Expr *fst_operand = try_parse(ctx);
    if (fst_operand) {
        matmul_expr->fst_operand = std::unique_ptr<clobber::Expr>(fst_operand);
    }

    clobber::Expr *snd_operand = try_parse(ctx);
    if (snd_operand) {
        matmul_expr->snd_operand = std::unique_ptr<clobber::Expr>(snd_operand);
    }

    matmul_expr->close_paren_token = ctx.tokens[ctx.current_idx++];
    return matmul_expr;
}

clobber::Expr *
try_parse_relu_expr(ParseContext &ctx) {
    clobber::accel::RelUExpr *relu_expr;
    // relu_expr = new RelUExpr();
    Option<clobber::Token> current_token;

    relu_expr->type             = clobber::Expr::Type::RelUExpr;
    relu_expr->open_paren_token = ctx.tokens[ctx.current_idx++];
    relu_expr->relu_token       = ctx.tokens[ctx.current_idx++]; // asserted by caller

    clobber::Expr *operand = try_parse(ctx);
    if (operand) {
        relu_expr->operand = std::unique_ptr<clobber::Expr>(operand);
    }

    relu_expr->close_paren_token = ctx.tokens[ctx.current_idx++];
    return relu_expr;
}

clobber::Expr *
try_parse_call_expr(ParseContext &ctx) {
    clobber::CallExpr *ce;
    Option<clobber::Token> current_token;
    std::vector<clobber::Expr> arg_exprs;

    // ce            = new CallExpr();
    ce->arguments = std::vector<std::unique_ptr<clobber::Expr>>{};
    ce->type      = clobber::Expr::Type::CallExpr;

    ce->open_paren_token = ctx.tokens[ctx.current_idx++];
    ce->operator_token   = ctx.tokens[ctx.current_idx++];
    current_token        = try_get_token(ctx.tokens, ctx.current_idx);

    while (current_token && current_token.value().type != clobber::Token::Type::CloseParenToken) {
        clobber::Expr *arg_expr = try_parse(ctx);
        if (arg_expr) {
            ce->arguments.push_back(std::unique_ptr<clobber::Expr>(arg_expr));
        }

        current_token = try_get_token(ctx.tokens, ctx.current_idx);
    }

    if (current_token && current_token.value().type == clobber::Token::Type::CloseParenToken) {
        ce->close_paren_token = current_token.value();
    }

    ctx.current_idx++;
    return ce;
}

clobber::Expr *
try_parse(ParseContext &ctx) {
    clobber::Token current_token;
    Option<clobber::Token> token_opt;
    ParseDelegate parse_fn;
    Option<ParseDelegate> parse_fn_opt;

    token_opt = try_get_token(ctx.tokens, ctx.current_idx);
    if (!token_opt) {
        return nullptr;
    }
    current_token = token_opt.value();

    parse_fn_opt = try_get_parse_fun(current_token.type);
    if (!parse_fn_opt) {
        // clobber::ParserError err = err::InternalErr(1, current_token.start, current_token.length);
        // parse_errors.push_back(err);
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