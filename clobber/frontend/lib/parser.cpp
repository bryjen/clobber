#include "clobber/pch.hpp"

#include <clobber/common/utils.hpp>

#include "clobber/ast.hpp"
#include "clobber/parser.hpp"

#include "clobber/internal/parser_error_factory.hpp"

template <typename T> using Option = std::optional<T>;

// We deal with raw pointers because we need to perform post processing on returned expr types.
// For example, we need to cast 'ExprBase' to some subtype to fit the properties of the expr that we're trying to parse.
// Dealing with raw pointers makes this infinitely easier - as long as the caller always wraps it in a unique ptr, which isn't that hard to
// assert.
using ParseDelegate = ExprBase *(*)(const std::string &, const std::vector<ClobberToken> &, std::vector<ParserError> &, size_t &);

// macro to help defining forward declarations
#define PARSE_DELEGATE_FN(FN_NAME)                                                                                                         \
    ExprBase *FN_NAME(const std::string &, const std::vector<ClobberToken> &, std::vector<ParserError> &, size_t &);

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

Option<ClobberToken>
try_get_token(const std::vector<ClobberToken> &tokens, size_t idx) {
    size_t tokens_len = tokens.size();
    return (idx >= 0 && idx < tokens_len) ? std::make_optional(tokens[idx]) : std::nullopt;
}

void
vector_recover(const std::vector<ClobberToken> &tokens, size_t &idx) {
    Option<ClobberToken> current_token = try_get_token(tokens, idx);
    while (current_token && current_token.value().token_type != ClobberTokenType::CloseBracketToken) {
        current_token = try_get_token(tokens, idx++);
    }
}

void
recover(const std::vector<ClobberToken> &, size_t &idx) {
    idx = idx + 1;
}

Option<ParseDelegate>
try_get_parse_fun(ClobberTokenType token_type) {
    // keep using if-statements instead of a map/switch to support predicates over different token types
    const std::unordered_set<ClobberTokenType> valid_token_types = {ClobberTokenType::NumericLiteralToken};
    auto is_reserved_symbol_identifier = [&valid_token_types](ClobberTokenType tt) { return valid_token_types.contains(tt); };

    if (token_type == ClobberTokenType::NumericLiteralToken) {
        return std::make_optional(try_parse_numeric_literal_expr);

    } else if (token_type == ClobberTokenType::OpenParenToken) {
        return std::make_optional(try_parse_call_expr_or_special_form);

    } else if (token_type == ClobberTokenType::IdentifierToken || is_reserved_symbol_identifier(token_type)) {
        return std::make_optional(try_parse_identifier);

    } else if (token_type == ClobberTokenType::StringLiteralToken) {
        return std::make_optional(try_parse_string_literal_expr);

    } else if (token_type == ClobberTokenType::CharLiteralToken) {
        return std::make_optional(try_parse_char_literal_expr);

    } else {
        return std::nullopt;
    }
}

ExprBase *
try_parse_numeric_literal_expr(const std::string &, const std::vector<ClobberToken> &tokens, std::vector<ParserError> &, size_t &idx) {
    NumLiteralExpr *nle;
    ClobberToken current_token;

    current_token = tokens[idx]; // no bounds check, current token exists, asserted by caller

    nle            = new NumLiteralExpr();
    nle->token     = current_token;
    nle->expr_type = ClobberExprType::NumericLiteralExpr;

    idx++;
    return nle;
}

ExprBase *
try_parse_string_literal_expr(const std::string &, const std::vector<ClobberToken> &tokens, std::vector<ParserError> &, size_t &idx) {
    StringLiteralExpr *sle = new StringLiteralExpr();

    std::string str = any_cast<std::string>(tokens[idx].value);
    if (str.size() > 2) {
        str = str.substr(1, str.size() - 2);
    }

    sle->value     = str;
    sle->token     = tokens[idx]; // no bounds check, current token exists, asserted by caller
    sle->expr_type = ClobberExprType::StringLiteralExpr;

    idx++;
    return sle;
}

ExprBase *
try_parse_char_literal_expr(const std::string &, const std::vector<ClobberToken> &tokens, std::vector<ParserError> &, size_t &idx) {
    CharLiteralExpr *cle = new CharLiteralExpr();

    std::string str = any_cast<std::string>(tokens[idx].value);
    if (str.size() > 2) {
        str = str.substr(1, str.size() - 2);
    }

    cle->value     = str;
    cle->token     = tokens[idx]; // no bounds check, current token exists, asserted by caller
    cle->expr_type = ClobberExprType::CharLiteralExpr;
    idx++;
    return cle;
}

ExprBase *
try_parse_identifier(const std::string &source_text, const std::vector<ClobberToken> &tokens, std::vector<ParserError> &, size_t &idx) {
    IdentifierExpr *iden_expr;
    ClobberToken token;

    token                = tokens[idx]; // guaranteed to exist by caller
    iden_expr            = new IdentifierExpr();
    iden_expr->name      = source_text.substr(token.start, token.length);
    iden_expr->token     = token;
    iden_expr->expr_type = ClobberExprType::IdentifierExpr;

    idx++;
    return iden_expr;
}

BindingVectorExpr *
try_parse_binding_vector_expr(const std::string &source_text, const std::vector<ClobberToken> &tokens,
                              std::vector<ParserError> &parser_errors, size_t &idx) {
    BindingVectorExpr *binding_vector_expr = new BindingVectorExpr();
    Option<ClobberToken> current_token;

    binding_vector_expr->expr_type          = ClobberExprType::BindingVectorExpr;
    binding_vector_expr->open_bracket_token = tokens[idx++];

    while (true) {
        current_token = try_get_token(tokens, idx);
        if (current_token && current_token.value().token_type == ClobberTokenType::CloseBracketToken) {
            break;
        }

        ExprBase *ident_expr_base = try_parse_identifier(source_text, tokens, parser_errors, idx);
        if (ident_expr_base) {
            IdentifierExpr *ident_expr = dynamic_cast<IdentifierExpr *>(ident_expr_base);
            binding_vector_expr->identifiers.push_back(std::unique_ptr<IdentifierExpr>(ident_expr));
        }

        ExprBase *value_expr = try_parse(source_text, tokens, parser_errors, idx);
        if (value_expr) {
            binding_vector_expr->exprs.push_back(std::unique_ptr<ExprBase>(value_expr));
        }

        binding_vector_expr->num_bindings++;
    }

    binding_vector_expr->close_bracket_token = tokens[idx++];
    return binding_vector_expr;
}

ParameterVectorExpr *
try_parse_parameter_vector_expr(const std::string &source_text, const std::vector<ClobberToken> &tokens,
                                std::vector<ParserError> &parser_errors, size_t &idx) {
    ParameterVectorExpr *parameter_vector_expr = new ParameterVectorExpr();
    Option<ClobberToken> current_token;

    parameter_vector_expr->expr_type          = ClobberExprType::ParameterVectorExpr;
    parameter_vector_expr->open_bracket_token = tokens[idx++];

    while (true) {
        current_token = try_get_token(tokens, idx);
        if (current_token && current_token.value().token_type == ClobberTokenType::CloseBracketToken) {
            break;
        }

        ExprBase *ident_expr_base = try_parse_identifier(source_text, tokens, parser_errors, idx);
        if (ident_expr_base) {
            IdentifierExpr *ident_expr = dynamic_cast<IdentifierExpr *>(ident_expr_base);
            parameter_vector_expr->identifiers.push_back(std::unique_ptr<IdentifierExpr>(ident_expr));
        }
    }

    parameter_vector_expr->close_bracket_token = tokens[idx++];
    return parameter_vector_expr;
}

ExprBase *
try_parse_call_expr_or_special_form(const std::string &source_text, const std::vector<ClobberToken> &tokens,
                                    std::vector<ParserError> &parser_errors, size_t &idx) {
    // clang-format off
    const std::unordered_map<ClobberTokenType, ParseDelegate> special_form_parse_fns = {
        {ClobberTokenType::LetKeywordToken, try_parse_let_expr},
        {ClobberTokenType::FnKeywordToken, try_parse_fn_expr},
        {ClobberTokenType::DefKeywordToken, try_parse_def_expr},
        {ClobberTokenType::DoKeywordToken, try_parse_do_expr},
        {ClobberTokenType::AccelKeywordToken, try_parse_accel_expr},
        {ClobberTokenType::TosaMatmulKeywordToken, try_parse_matmul_expr},
        {ClobberTokenType::TosaReluKeywordToken, try_parse_relu_expr},
    };
    // clang-format on

    // assuming the current token is an open paren token, we peek forward to see if its a keyword
    ParseDelegate parse_fn     = try_parse_call_expr;
    Option<ClobberToken> token = try_get_token(tokens, idx + 1);
    if (token) {
        const ClobberTokenType token_type = token.value().token_type;
        auto it                           = special_form_parse_fns.find(token_type);
        parse_fn                          = it != special_form_parse_fns.end() ? it->second : try_parse_call_expr;
    }

    return parse_fn(source_text, tokens, parser_errors, idx);
}

ExprBase *
try_parse_let_expr(const std::string &source_text, const std::vector<ClobberToken> &tokens, std::vector<ParserError> &parser_errors,
                   size_t &idx) {
    LetExpr *let_expr = new LetExpr();
    BindingVectorExpr *binding_vector_expr;
    Option<ClobberToken> current_token;

    let_expr->expr_type        = ClobberExprType::LetExpr;
    let_expr->open_paren_token = tokens[idx++];
    let_expr->let_token        = tokens[idx++]; // asserted by caller

    binding_vector_expr           = try_parse_binding_vector_expr(source_text, tokens, parser_errors, idx);
    let_expr->binding_vector_expr = std::unique_ptr<BindingVectorExpr>(binding_vector_expr);

    current_token = try_get_token(tokens, idx);
    while (current_token && current_token.value().token_type != ClobberTokenType::CloseParenToken) {
        ExprBase *expr = try_parse(source_text, tokens, parser_errors, idx);
        if (expr) {
            let_expr->body_exprs.push_back(std::unique_ptr<ExprBase>(expr));
        }

        current_token = try_get_token(tokens, idx);
    }

    let_expr->close_paren_token = tokens[idx++];
    return let_expr;
}

ExprBase *
try_parse_fn_expr(const std::string &source_text, const std::vector<ClobberToken> &tokens, std::vector<ParserError> &parser_errors,
                  size_t &idx) {
    FnExpr *fn_expr = new FnExpr();
    ParameterVectorExpr *parameter_vector_expr;
    Option<ClobberToken> current_token;

    fn_expr->expr_type        = ClobberExprType::FnExpr;
    fn_expr->open_paren_token = tokens[idx++];
    fn_expr->fn_token         = tokens[idx++]; // asserted by caller

    parameter_vector_expr          = try_parse_parameter_vector_expr(source_text, tokens, parser_errors, idx);
    fn_expr->parameter_vector_expr = std::unique_ptr<ParameterVectorExpr>(parameter_vector_expr);

    current_token = try_get_token(tokens, idx);
    while (current_token && current_token.value().token_type != ClobberTokenType::CloseParenToken) {
        ExprBase *expr = try_parse(source_text, tokens, parser_errors, idx);
        if (expr) {
            fn_expr->body_exprs.push_back(std::unique_ptr<ExprBase>(expr));
        }

        current_token = try_get_token(tokens, idx);
    }

    fn_expr->close_paren_token = tokens[idx++];
    return fn_expr;
}

ExprBase *
try_parse_def_expr(const std::string &source_text, const std::vector<ClobberToken> &tokens, std::vector<ParserError> &parser_errors,
                   size_t &idx) {
    DefExpr *def_expr = new DefExpr();
    Option<ClobberToken> current_token;

    def_expr->expr_type        = ClobberExprType::DefExpr;
    def_expr->open_paren_token = tokens[idx++];
    def_expr->def_token        = tokens[idx++]; // asserted by caller

    ExprBase *ident_expr_base = try_parse_identifier(source_text, tokens, parser_errors, idx);
    if (ident_expr_base) {
        IdentifierExpr *ident_expr = dynamic_cast<IdentifierExpr *>(ident_expr_base);
        def_expr->identifier       = std::unique_ptr<IdentifierExpr>(ident_expr);
    }

    ExprBase *value_expr = try_parse(source_text, tokens, parser_errors, idx);
    if (value_expr) {
        def_expr->value = std::unique_ptr<ExprBase>(value_expr);
    }

    def_expr->close_paren_token = tokens[idx++];
    return def_expr;
}

ExprBase *
try_parse_do_expr(const std::string &source_text, const std::vector<ClobberToken> &tokens, std::vector<ParserError> &parser_errors,
                  size_t &idx) {
    DoExpr *do_expr = new DoExpr();
    Option<ClobberToken> current_token;

    do_expr->expr_type        = ClobberExprType::DoExpr;
    do_expr->open_paren_token = tokens[idx++];
    do_expr->do_token         = tokens[idx++]; // asserted by caller

    current_token = try_get_token(tokens, idx);
    while (current_token && current_token.value().token_type != ClobberTokenType::CloseParenToken) {
        ExprBase *expr = try_parse(source_text, tokens, parser_errors, idx);
        if (expr) {
            do_expr->body_exprs.push_back(std::unique_ptr<ExprBase>(expr));
        }

        current_token = try_get_token(tokens, idx);
    }

    do_expr->close_paren_token = tokens[idx++];
    return do_expr;
}

ExprBase *
try_parse_accel_expr(const std::string &source_text, const std::vector<ClobberToken> &tokens, std::vector<ParserError> &parser_errors,
                     size_t &idx) {
    AccelExpr *accel_expr = new AccelExpr();
    Option<ClobberToken> current_token;

    accel_expr->expr_type        = ClobberExprType::AccelExpr;
    accel_expr->open_paren_token = tokens[idx++];
    accel_expr->accel_token      = tokens[idx++]; // asserted by caller

    BindingVectorExpr *binding_vector_expr = try_parse_binding_vector_expr(source_text, tokens, parser_errors, idx);
    if (binding_vector_expr) {
        accel_expr->binding_vector_expr = std::unique_ptr<BindingVectorExpr>(binding_vector_expr);
    }

    current_token = try_get_token(tokens, idx);
    while (current_token && current_token.value().token_type != ClobberTokenType::CloseParenToken) {
        ExprBase *expr = try_parse(source_text, tokens, parser_errors, idx);
        if (expr) {
            accel_expr->body_exprs.push_back(std::unique_ptr<ExprBase>(expr));
        }

        current_token = try_get_token(tokens, idx);
    }

    accel_expr->close_paren_token = tokens[idx++];
    return accel_expr;
}

ExprBase *
try_parse_matmul_expr(const std::string &source_text, const std::vector<ClobberToken> &tokens, std::vector<ParserError> &parser_errors,
                      size_t &idx) {
    MatMulExpr *matmul_expr = new MatMulExpr();
    Option<ClobberToken> current_token;

    matmul_expr->expr_type        = ClobberExprType::MatMulExpr;
    matmul_expr->open_paren_token = tokens[idx++];
    matmul_expr->mat_mul_token    = tokens[idx++]; // asserted by caller

    ExprBase *fst_operand = try_parse(source_text, tokens, parser_errors, idx);
    if (fst_operand) {
        matmul_expr->fst_operand = std::unique_ptr<ExprBase>(fst_operand);
    }

    ExprBase *snd_operand = try_parse(source_text, tokens, parser_errors, idx);
    if (snd_operand) {
        matmul_expr->snd_operand = std::unique_ptr<ExprBase>(snd_operand);
    }

    matmul_expr->close_paren_token = tokens[idx++];
    return matmul_expr;
}

ExprBase *
try_parse_relu_expr(const std::string &source_text, const std::vector<ClobberToken> &tokens, std::vector<ParserError> &parser_errors,
                    size_t &idx) {
    RelUExpr *relu_expr = new RelUExpr();
    Option<ClobberToken> current_token;

    relu_expr->expr_type        = ClobberExprType::RelUExpr;
    relu_expr->open_paren_token = tokens[idx++];
    relu_expr->relu_token       = tokens[idx++]; // asserted by caller

    ExprBase *operand = try_parse(source_text, tokens, parser_errors, idx);
    if (operand) {
        relu_expr->operand = std::unique_ptr<ExprBase>(operand);
    }

    relu_expr->close_paren_token = tokens[idx++];
    return relu_expr;
}

ExprBase *
try_parse_call_expr(const std::string &source_text, const std::vector<ClobberToken> &tokens, std::vector<ParserError> &parser_errors,
                    size_t &idx) {
    CallExpr *ce;
    Option<ClobberToken> current_token;
    std::vector<ExprBase> arg_exprs;

    ce            = new CallExpr();
    ce->arguments = std::vector<std::unique_ptr<ExprBase>>{};
    ce->expr_type = ClobberExprType::CallExpr;

    ce->open_paren_token = tokens[idx++];
    ce->operator_token   = tokens[idx++];
    current_token        = try_get_token(tokens, idx);

    while (current_token && current_token.value().token_type != ClobberTokenType::CloseParenToken) {
        ExprBase *arg_expr = try_parse(source_text, tokens, parser_errors, idx);
        if (arg_expr) {
            ce->arguments.push_back(std::unique_ptr<ExprBase>(arg_expr));
        }

        current_token = try_get_token(tokens, idx);
    }

    if (current_token && current_token.value().token_type == ClobberTokenType::CloseParenToken) {
        ce->close_paren_token = current_token.value();
    }

    idx++;
    return ce;
}

ExprBase *
try_parse(const std::string &source_text, const std::vector<ClobberToken> &tokens, std::vector<ParserError> &parser_errors, size_t &idx) {
    ClobberToken current_token;
    Option<ClobberToken> token_opt;
    ParseDelegate parse_fn;
    Option<ParseDelegate> parse_fn_opt;

    token_opt = try_get_token(tokens, idx);
    if (!token_opt) {
        return nullptr;
    }
    current_token = token_opt.value();

    parse_fn_opt = try_get_parse_fun(current_token.token_type);
    if (!parse_fn_opt) {
        ParserError err = err::InternalErr(1, current_token.start, current_token.length);
        parser_errors.push_back(err);
        recover(tokens, idx);
        return nullptr;
    }
    parse_fn = parse_fn_opt.value();

    return parse_fn(source_text, tokens, parser_errors, idx);
}

std::unique_ptr<CompilationUnit>
clobber::parse(const std::string &source_text, const std::vector<ClobberToken> &tokens) {
    // clang-format off
    std::unique_ptr<CompilationUnit> cu = std::make_unique<CompilationUnit>(CompilationUnit{
        source_text,
        {},
        {}
    });
    // clang-format on
    size_t current_idx;
    size_t tokens_len;

    current_idx = 0;
    tokens_len  = tokens.size();

    while (current_idx < tokens_len) {
        if (tokens[current_idx].token_type == ClobberTokenType::EofToken) {
            break;
        }

        // 'current_idx' passed by reference, implicitly modified
        ExprBase *parsed_expr = try_parse(source_text, tokens, cu->parse_errors, current_idx);
        if (parsed_expr) {
            cu->exprs.push_back(std::unique_ptr<ExprBase>(parsed_expr));
        }
    }

    return cu;
}