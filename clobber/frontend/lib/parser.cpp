#include "clobber/pch.hpp"

#include <clobber/common/utils.hpp>

#include "clobber/ast.hpp"
#include "clobber/parser.hpp"

#include "clobber/internal/parser_error_factory.hpp"

// using namespace clobber;

template <typename T> using Option = std::optional<T>;

// We deal with raw pointers because we need to perform post processing on returned expr types.
// For example, we need to cast 'Expr' to some subtype to fit the properties of the expr that we're trying to parse.
// Dealing with raw pointers makes this infinitely easier - as long as the caller always wraps it in a unique ptr, which isn't that hard to
// assert.
using ParseDelegate = clobber::Expr *(*)(const std::string &, const std::vector<clobber::ClobberToken> &,
                                         std::vector<clobber::ParserError> &, size_t &);

// macro to help defining forward declarations
#define PARSE_DELEGATE_FN(FN_NAME)                                                                                                         \
    clobber::Expr *FN_NAME(const std::string &, const std::vector<clobber::ClobberToken> &, std::vector<clobber::ParserError> &, size_t &);

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

Option<clobber::ClobberToken>
try_get_token(const std::vector<clobber::ClobberToken> &tokens, size_t idx) {
    size_t tokens_len = tokens.size();
    return (idx >= 0 && idx < tokens_len) ? std::make_optional(tokens[idx]) : std::nullopt;
}

void
vector_recover(const std::vector<clobber::ClobberToken> &tokens, size_t &idx) {
    Option<clobber::ClobberToken> current_token = try_get_token(tokens, idx);
    while (current_token && current_token.value().token_type != clobber::ClobberTokenType::CloseBracketToken) {
        current_token = try_get_token(tokens, idx++);
    }
}

void
recover(const std::vector<clobber::ClobberToken> &, size_t &idx) {
    idx = idx + 1;
}

Option<ParseDelegate>
try_get_parse_fun(clobber::ClobberTokenType token_type) {
    // keep using if-statements instead of a map/switch to support predicates over different token types
    const std::unordered_set<clobber::ClobberTokenType> valid_token_types = {clobber::ClobberTokenType::NumericLiteralToken};
    auto is_reserved_symbol_identifier = [&valid_token_types](clobber::ClobberTokenType tt) { return valid_token_types.contains(tt); };

    if (token_type == clobber::ClobberTokenType::NumericLiteralToken) {
        return std::make_optional(try_parse_numeric_literal_expr);

    } else if (token_type == clobber::ClobberTokenType::OpenParenToken) {
        return std::make_optional(try_parse_call_expr_or_special_form);

    } else if (token_type == clobber::ClobberTokenType::IdentifierToken || is_reserved_symbol_identifier(token_type)) {
        return std::make_optional(try_parse_identifier);

    } else if (token_type == clobber::ClobberTokenType::StringLiteralToken) {
        return std::make_optional(try_parse_string_literal_expr);

    } else if (token_type == clobber::ClobberTokenType::CharLiteralToken) {
        return std::make_optional(try_parse_char_literal_expr);

    } else {
        return std::nullopt;
    }
}

clobber::Expr *
try_parse_numeric_literal_expr(const std::string &, const std::vector<clobber::ClobberToken> &tokens, std::vector<clobber::ParserError> &,
                               size_t &idx) {
    clobber::NumLiteralExpr *nle;
    clobber::ClobberToken current_token;

    current_token = tokens[idx]; // no bounds check, current token exists, asserted by caller

    // nle        = new NumLiteralExpr();
    nle->token = current_token;
    nle->type  = clobber::Expr::Type::NumericLiteralExpr;

    idx++;
    return nle;
}

clobber::Expr *
try_parse_string_literal_expr(const std::string &source_text, const std::vector<clobber::ClobberToken> &tokens,
                              std::vector<clobber::ParserError> &parse_errors, size_t &idx) {
    clobber::StringLiteralExpr *sle;
    // sle = new StringLiteralExpr();

    clobber::ClobberToken token = tokens[idx];
    std::string str             = token.ExtractText(source_text);
    if (str.size() > 2) {
        str = str.substr(1, str.size() - 2);
    }

    sle->value = str;
    sle->token = tokens[idx]; // no bounds check, current token exists, asserted by caller
    sle->type  = clobber::Expr::Type::StringLiteralExpr;

    idx++;
    return sle;
}

clobber::Expr *
try_parse_char_literal_expr(const std::string &source_text, const std::vector<clobber::ClobberToken> &tokens,
                            std::vector<clobber::ParserError> &parse_errors, size_t &idx) {
    clobber::CharLiteralExpr *cle;
    // cle = new CharLiteralExpr();

    clobber::ClobberToken token = tokens[idx];
    std::string str             = token.ExtractText(source_text);
    if (str.size() > 2) {
        str = str.substr(1, str.size() - 2);
    }

    cle->value = str;
    cle->token = tokens[idx]; // no bounds check, current token exists, asserted by caller
    cle->type  = clobber::Expr::Type::CharLiteralExpr;

    idx++;
    return cle;
}

clobber::Expr *
try_parse_identifier(const std::string &source_text, const std::vector<clobber::ClobberToken> &tokens,
                     std::vector<clobber::ParserError> &parse_errors, size_t &idx) {
    clobber::IdentifierExpr *iden_expr;
    clobber::ClobberToken token = tokens[idx]; // guaranteed to exist by caller

    // iden_expr            = new IdentifierExpr();
    iden_expr->name  = source_text.substr(token.start, token.length);
    iden_expr->token = token;
    iden_expr->type  = clobber::Expr::Type::IdentifierExpr;

    idx++;
    return iden_expr;
}

clobber::BindingVectorExpr *
try_parse_binding_vector_expr(const std::string &source_text, const std::vector<clobber::ClobberToken> &tokens,
                              std::vector<clobber::ParserError> &parse_errors, size_t &idx) {
    clobber::BindingVectorExpr *binding_vector_expr;
    // binding_vector_expr = new BindingVectorExpr();
    Option<clobber::ClobberToken> current_token;

    // binding_vector_expr->type               = clobber::Expr::Type::BindingVectorExpr;
    binding_vector_expr->open_bracket_token = tokens[idx++];

    while (true) {
        current_token = try_get_token(tokens, idx);
        if (current_token && current_token.value().token_type == clobber::ClobberTokenType::CloseBracketToken) {
            break;
        }

        clobber::Expr *ident_expr_base = try_parse_identifier(source_text, tokens, parse_errors, idx);
        if (ident_expr_base) {
            clobber::IdentifierExpr *ident_expr = dynamic_cast<clobber::IdentifierExpr *>(ident_expr_base);
            binding_vector_expr->identifiers.push_back(std::unique_ptr<clobber::IdentifierExpr>(ident_expr));
        }

        clobber::Expr *value_expr = try_parse(source_text, tokens, parse_errors, idx);
        if (value_expr) {
            binding_vector_expr->exprs.push_back(std::unique_ptr<clobber::Expr>(value_expr));
        }

        binding_vector_expr->num_bindings++;
    }

    binding_vector_expr->close_bracket_token = tokens[idx++];
    return binding_vector_expr;
}

clobber::ParameterVectorExpr *
try_parse_parameter_vector_expr(const std::string &source_text, const std::vector<clobber::ClobberToken> &tokens,
                                std::vector<clobber::ParserError> &parse_errors, size_t &idx) {
    clobber::ParameterVectorExpr *parameter_vector_expr;
    // parameter_vector_expr = new ParameterVectorExpr();
    Option<clobber::ClobberToken> current_token;

    // parameter_vector_expr->expr_type          = ClobberExprType::ParameterVectorExpr;
    parameter_vector_expr->open_bracket_token = tokens[idx++];

    while (true) {
        current_token = try_get_token(tokens, idx);
        if (current_token && current_token.value().token_type == clobber::ClobberTokenType::CloseBracketToken) {
            break;
        }

        clobber::Expr *ident_expr_base = try_parse_identifier(source_text, tokens, parse_errors, idx);
        if (ident_expr_base) {
            clobber::IdentifierExpr *ident_expr = dynamic_cast<clobber::IdentifierExpr *>(ident_expr_base);
            parameter_vector_expr->identifiers.push_back(std::unique_ptr<clobber::IdentifierExpr>(ident_expr));
        }
    }

    parameter_vector_expr->close_bracket_token = tokens[idx++];
    return parameter_vector_expr;
}

clobber::Expr *
try_parse_call_expr_or_special_form(const std::string &source_text, const std::vector<clobber::ClobberToken> &tokens,
                                    std::vector<clobber::ParserError> &parse_errors, size_t &idx) {
    // clang-format off
    const std::unordered_map<clobber::ClobberTokenType, ParseDelegate> special_form_parse_fns = {
        {clobber::ClobberTokenType::LetKeywordToken, try_parse_let_expr},
        {clobber::ClobberTokenType::FnKeywordToken, try_parse_fn_expr},
        {clobber::ClobberTokenType::DefKeywordToken, try_parse_def_expr},
        {clobber::ClobberTokenType::DoKeywordToken, try_parse_do_expr},
        {clobber::ClobberTokenType::AccelKeywordToken, try_parse_accel_expr},
        {clobber::ClobberTokenType::TosaMatmulKeywordToken, try_parse_matmul_expr},
        {clobber::ClobberTokenType::TosaReluKeywordToken, try_parse_relu_expr},
    };
    // clang-format on

    // assuming the current token is an open paren token, we peek forward to see if its a keyword
    ParseDelegate parse_fn              = try_parse_call_expr;
    Option<clobber::ClobberToken> token = try_get_token(tokens, idx + 1);
    if (token) {
        const clobber::ClobberTokenType token_type = token.value().token_type;
        auto it                                    = special_form_parse_fns.find(token_type);
        parse_fn                                   = it != special_form_parse_fns.end() ? it->second : try_parse_call_expr;
    }

    return parse_fn(source_text, tokens, parse_errors, idx);
}

clobber::Expr *
try_parse_let_expr(const std::string &source_text, const std::vector<clobber::ClobberToken> &tokens,
                   std::vector<clobber::ParserError> &parse_errors, size_t &idx) {
    clobber::LetExpr *let_expr;
    // let_expr = new LetExpr();
    clobber::BindingVectorExpr *binding_vector_expr;
    Option<clobber::ClobberToken> current_token;

    let_expr->type             = clobber::Expr::Type::LetExpr;
    let_expr->open_paren_token = tokens[idx++];
    let_expr->let_token        = tokens[idx++]; // asserted by caller

    binding_vector_expr           = try_parse_binding_vector_expr(source_text, tokens, parse_errors, idx);
    let_expr->binding_vector_expr = std::unique_ptr<clobber::BindingVectorExpr>(binding_vector_expr);

    current_token = try_get_token(tokens, idx);
    while (current_token && current_token.value().token_type != clobber::ClobberTokenType::CloseParenToken) {
        clobber::Expr *expr = try_parse(source_text, tokens, parse_errors, idx);
        if (expr) {
            let_expr->body_exprs.push_back(std::unique_ptr<clobber::Expr>(expr));
        }

        current_token = try_get_token(tokens, idx);
    }

    let_expr->close_paren_token = tokens[idx++];
    return let_expr;
}

clobber::Expr *
try_parse_fn_expr(const std::string &source_text, const std::vector<clobber::ClobberToken> &tokens,
                  std::vector<clobber::ParserError> &parse_errors, size_t &idx) {
    clobber::FnExpr *fn_expr;
    // fn_expr         = new FnExpr();
    clobber::ParameterVectorExpr *parameter_vector_expr;
    Option<clobber::ClobberToken> current_token;

    fn_expr->type             = clobber::Expr::Type::FnExpr;
    fn_expr->open_paren_token = tokens[idx++];
    fn_expr->fn_token         = tokens[idx++]; // asserted by caller

    parameter_vector_expr          = try_parse_parameter_vector_expr(source_text, tokens, parse_errors, idx);
    fn_expr->parameter_vector_expr = std::unique_ptr<clobber::ParameterVectorExpr>(parameter_vector_expr);

    current_token = try_get_token(tokens, idx);
    while (current_token && current_token.value().token_type != clobber::ClobberTokenType::CloseParenToken) {
        clobber::Expr *expr = try_parse(source_text, tokens, parse_errors, idx);
        if (expr) {
            fn_expr->body_exprs.push_back(std::unique_ptr<clobber::Expr>(expr));
        }

        current_token = try_get_token(tokens, idx);
    }

    fn_expr->close_paren_token = tokens[idx++];
    return fn_expr;
}

clobber::Expr *
try_parse_def_expr(const std::string &source_text, const std::vector<clobber::ClobberToken> &tokens,
                   std::vector<clobber::ParserError> &parse_errors, size_t &idx) {
    clobber::DefExpr *def_expr;
    // def_expr = new DefExpr();
    Option<clobber::ClobberToken> current_token;

    def_expr->type             = clobber::Expr::Type::DefExpr;
    def_expr->open_paren_token = tokens[idx++];
    def_expr->def_token        = tokens[idx++]; // asserted by caller

    clobber::Expr *ident_expr_base = try_parse_identifier(source_text, tokens, parse_errors, idx);
    if (ident_expr_base) {
        clobber::IdentifierExpr *ident_expr = dynamic_cast<clobber::IdentifierExpr *>(ident_expr_base);
        def_expr->identifier                = std::unique_ptr<clobber::IdentifierExpr>(ident_expr);
    }

    clobber::Expr *value_expr = try_parse(source_text, tokens, parse_errors, idx);
    if (value_expr) {
        def_expr->value = std::unique_ptr<clobber::Expr>(value_expr);
    }

    def_expr->close_paren_token = tokens[idx++];
    return def_expr;
}

clobber::Expr *
try_parse_do_expr(const std::string &source_text, const std::vector<clobber::ClobberToken> &tokens,
                  std::vector<clobber::ParserError> &parse_errors, size_t &idx) {
    clobber::DoExpr *do_expr;
    // do_expr = new DoExpr();
    Option<clobber::ClobberToken> current_token;

    do_expr->type             = clobber::Expr::Type::DoExpr;
    do_expr->open_paren_token = tokens[idx++];
    do_expr->do_token         = tokens[idx++]; // asserted by caller

    current_token = try_get_token(tokens, idx);
    while (current_token && current_token.value().token_type != clobber::ClobberTokenType::CloseParenToken) {
        clobber::Expr *expr = try_parse(source_text, tokens, parse_errors, idx);
        if (expr) {
            do_expr->body_exprs.push_back(std::unique_ptr<clobber::Expr>(expr));
        }

        current_token = try_get_token(tokens, idx);
    }

    do_expr->close_paren_token = tokens[idx++];
    return do_expr;
}

clobber::Expr *
try_parse_accel_expr(const std::string &source_text, const std::vector<clobber::ClobberToken> &tokens,
                     std::vector<clobber::ParserError> &parse_errors, size_t &idx) {
    clobber::accel::AccelExpr *accel_expr;
    // accel_expr = new AccelExpr();
    Option<clobber::ClobberToken> current_token;

    accel_expr->type             = clobber::Expr::Type::AccelExpr;
    accel_expr->open_paren_token = tokens[idx++];
    accel_expr->accel_token      = tokens[idx++]; // asserted by caller

    clobber::BindingVectorExpr *binding_vector_expr = try_parse_binding_vector_expr(source_text, tokens, parse_errors, idx);
    if (binding_vector_expr) {
        accel_expr->binding_vector_expr = std::unique_ptr<clobber::BindingVectorExpr>(binding_vector_expr);
    }

    current_token = try_get_token(tokens, idx);
    while (current_token && current_token.value().token_type != clobber::ClobberTokenType::CloseParenToken) {
        clobber::Expr *expr = try_parse(source_text, tokens, parse_errors, idx);
        if (expr) {
            accel_expr->body_exprs.push_back(std::unique_ptr<clobber::Expr>(expr));
        }

        current_token = try_get_token(tokens, idx);
    }

    accel_expr->close_paren_token = tokens[idx++];
    return accel_expr;
}

clobber::Expr *
try_parse_matmul_expr(const std::string &source_text, const std::vector<clobber::ClobberToken> &tokens,
                      std::vector<clobber::ParserError> &parse_errors, size_t &idx) {
    clobber::accel::MatMulExpr *matmul_expr;
    // matmul_expr = new MatMulExpr();
    Option<clobber::ClobberToken> current_token;

    matmul_expr->type             = clobber::Expr::Type::MatMulExpr;
    matmul_expr->open_paren_token = tokens[idx++];
    matmul_expr->mat_mul_token    = tokens[idx++]; // asserted by caller

    clobber::Expr *fst_operand = try_parse(source_text, tokens, parse_errors, idx);
    if (fst_operand) {
        matmul_expr->fst_operand = std::unique_ptr<clobber::Expr>(fst_operand);
    }

    clobber::Expr *snd_operand = try_parse(source_text, tokens, parse_errors, idx);
    if (snd_operand) {
        matmul_expr->snd_operand = std::unique_ptr<clobber::Expr>(snd_operand);
    }

    matmul_expr->close_paren_token = tokens[idx++];
    return matmul_expr;
}

clobber::Expr *
try_parse_relu_expr(const std::string &source_text, const std::vector<clobber::ClobberToken> &tokens,
                    std::vector<clobber::ParserError> &parse_errors, size_t &idx) {
    clobber::accel::RelUExpr *relu_expr;
    // relu_expr = new RelUExpr();
    Option<clobber::ClobberToken> current_token;

    relu_expr->type             = clobber::Expr::Type::RelUExpr;
    relu_expr->open_paren_token = tokens[idx++];
    relu_expr->relu_token       = tokens[idx++]; // asserted by caller

    clobber::Expr *operand = try_parse(source_text, tokens, parse_errors, idx);
    if (operand) {
        relu_expr->operand = std::unique_ptr<clobber::Expr>(operand);
    }

    relu_expr->close_paren_token = tokens[idx++];
    return relu_expr;
}

clobber::Expr *
try_parse_call_expr(const std::string &source_text, const std::vector<clobber::ClobberToken> &tokens,
                    std::vector<clobber::ParserError> &parse_errors, size_t &idx) {
    clobber::CallExpr *ce;
    Option<clobber::ClobberToken> current_token;
    std::vector<clobber::Expr> arg_exprs;

    // ce            = new CallExpr();
    ce->arguments = std::vector<std::unique_ptr<clobber::Expr>>{};
    ce->type      = clobber::Expr::Type::CallExpr;

    ce->open_paren_token = tokens[idx++];
    ce->operator_token   = tokens[idx++];
    current_token        = try_get_token(tokens, idx);

    while (current_token && current_token.value().token_type != clobber::ClobberTokenType::CloseParenToken) {
        clobber::Expr *arg_expr = try_parse(source_text, tokens, parse_errors, idx);
        if (arg_expr) {
            ce->arguments.push_back(std::unique_ptr<clobber::Expr>(arg_expr));
        }

        current_token = try_get_token(tokens, idx);
    }

    if (current_token && current_token.value().token_type == clobber::ClobberTokenType::CloseParenToken) {
        ce->close_paren_token = current_token.value();
    }

    idx++;
    return ce;
}

clobber::Expr *
try_parse(const std::string &source_text, const std::vector<clobber::ClobberToken> &tokens, std::vector<clobber::ParserError> &parse_errors,
          size_t &idx) {
    clobber::ClobberToken current_token;
    Option<clobber::ClobberToken> token_opt;
    ParseDelegate parse_fn;
    Option<ParseDelegate> parse_fn_opt;

    token_opt = try_get_token(tokens, idx);
    if (!token_opt) {
        return nullptr;
    }
    current_token = token_opt.value();

    parse_fn_opt = try_get_parse_fun(current_token.token_type);
    if (!parse_fn_opt) {
        // clobber::ParserError err = err::InternalErr(1, current_token.start, current_token.length);
        // parse_errors.push_back(err);
        recover(tokens, idx);
        return nullptr;
    }
    parse_fn = parse_fn_opt.value();

    return parse_fn(source_text, tokens, parse_errors, idx);
}

std::unique_ptr<clobber::CompilationUnit>
clobber::parse(const std::string &source_text, const std::vector<clobber::ClobberToken> &tokens) {

    std::vector<std::unique_ptr<clobber::Expr>> exprs;
    std::vector<clobber::ParserError> parse_errors;

    size_t current_idx;
    size_t tokens_len;

    current_idx = 0;
    tokens_len  = tokens.size();

    while (current_idx < tokens_len) {
        if (tokens[current_idx].token_type == clobber::ClobberTokenType::EofToken) {
            break;
        }

        // 'current_idx' passed by reference, implicitly modified
        clobber::Expr *parsed_expr = try_parse(source_text, tokens, parse_errors, current_idx);
        if (parsed_expr) {
            exprs.push_back(std::unique_ptr<clobber::Expr>(parsed_expr));
        }
    }

    return std::make_unique<clobber::CompilationUnit>(source_text, std::move(exprs));
}