#include <memory>
#include <optional>
#include <unordered_set>

#include <clobber/common/utils.hpp>

#include "clobber/ast.hpp"
#include "clobber/parser.hpp"

#include "clobber/internal/parser_error_factory.hpp"

template <typename T> using Option = std::optional<T>;
using ParseDelegate = ExprBase *(*)(const std::string &, const std::vector<ClobberToken> &, std::vector<ParserError> &, size_t &);

// macro to help defining forward declarations
#define PARSE_DELEGATE_FN(FN_NAME)                                                                                                         \
    ExprBase *FN_NAME(const std::string &, const std::vector<ClobberToken> &, std::vector<ParserError> &, size_t &);

PARSE_DELEGATE_FN(try_parse)
PARSE_DELEGATE_FN(try_parse_call_expr)
PARSE_DELEGATE_FN(try_parse_numeric_literal_expr)
PARSE_DELEGATE_FN(try_parse_identifier)
PARSE_DELEGATE_FN(try_parse_call_expr_or_special_form)
PARSE_DELEGATE_FN(try_parse_let_expr)
PARSE_DELEGATE_FN(try_parse_fn_expr)

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
recover(const std::vector<ClobberToken> &tokens, size_t &idx) {
    idx = idx + 1;
}

Option<ParseDelegate>
try_get_parse_fun(ClobberTokenType token_type) {
    const std::unordered_set<ClobberTokenType> valid_token_types = {ClobberTokenType::NumericLiteralToken};
    auto is_reserved_symbol_identifier = [&valid_token_types](ClobberTokenType tt) { return valid_token_types.contains(tt); };

    if (token_type == ClobberTokenType::NumericLiteralToken) {
        return std::make_optional(try_parse_numeric_literal_expr);

    } else if (token_type == ClobberTokenType::OpenParenToken) {
        // return std::make_optional(try_parse_call_expr);
        return std::make_optional(try_parse_call_expr_or_special_form);

    } else if (token_type == ClobberTokenType::IdentifierToken || is_reserved_symbol_identifier(token_type)) {
        return std::make_optional(try_parse_identifier);

    } else {
        return std::nullopt;
    }
}

ExprBase *
try_parse_numeric_literal_expr(const std::string &source_text, const std::vector<ClobberToken> &tokens,
                               std::vector<ParserError> &parser_errors, size_t &idx) {
    NumLiteralExpr *nle;

    ClobberToken current_token;
    std::string num_as_str;
    Option<int> stoi_results;

    current_token = tokens[idx]; // no bounds check, current token exists, asserted by caller
    num_as_str    = current_token.ExtractText(source_text);
    stoi_results  = str_utils::try_stoi(num_as_str);

    if (!stoi_results) {
        ParserError err = err::InternalErr(0, current_token.start, current_token.length);
        parser_errors.push_back(err);
        recover(tokens, idx);
        return nullptr;
    }

    nle            = new NumLiteralExpr();
    nle->token     = current_token;
    nle->value     = stoi_results.value();
    nle->expr_type = ClobberExprType::NumericLiteralExpr;

    idx++;
    return nle;
}

ExprBase *
try_parse_identifier(const std::string &source_text, const std::vector<ClobberToken> &tokens, std::vector<ParserError> &parser_errors,
                     size_t &idx) {
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

ExprBase *
try_parse_call_expr_or_special_form(const std::string &source_text, const std::vector<ClobberToken> &tokens,
                                    std::vector<ParserError> &parser_errors, size_t &idx) {
    // assuming the current token is an open paren token, we peek forward to see if its a keyword
    ParseDelegate parse_fn;
    Option<ClobberToken> token = try_get_token(tokens, idx + 1);

    if (token) {
        const ClobberTokenType token_type = token.value().token_type;
        switch (token_type) {
        case ClobberTokenType::LetKeywordToken:
            parse_fn = try_parse_let_expr;
            break;
        case ClobberTokenType::FnKeywordToken:
            parse_fn = try_parse_fn_expr;
            break;
        default:
            // if the token does not match a valid keyword, then we just treat it as a function call
            parse_fn = try_parse_call_expr;
            break;
        }
    }

    return parse_fn(source_text, tokens, parser_errors, idx);
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
try_parse_let_expr(const std::string &source_text, const std::vector<ClobberToken> &tokens, std::vector<ParserError> &parser_errors,
                   size_t &idx) {
    LetExpr *let_expr = new LetExpr();
    BindingVectorExpr *binding_vector_expr;
    Option<ClobberToken> current_token;

    let_expr->expr_type        = ClobberExprType::LetExpr;
    let_expr->open_paren_token = tokens[idx++];
    let_expr->let_keyword      = tokens[idx++]; // asserted by caller

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
    fn_expr->fn_keyword       = tokens[idx++]; // asserted by caller

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

void
clobber::parse(const std::string &source_text, const std::vector<ClobberToken> &tokens, CompilationUnit &out_compilation_unit) {
    CompilationUnit cu;
    size_t current_idx;
    size_t tokens_len;

    out_compilation_unit.parse_errors = std::vector<ParserError>{};
    out_compilation_unit.exprs        = std::vector<std::unique_ptr<ExprBase>>{};

    current_idx = 0;
    tokens_len  = tokens.size();

    while (current_idx < tokens_len) {
        if (tokens[current_idx].token_type == ClobberTokenType::EofToken) {
            break;
        }

        // 'current_idx' passed by reference, implicitly modified
        ExprBase *parsed_expr = try_parse(source_text, tokens, out_compilation_unit.parse_errors, current_idx);
        if (parsed_expr) {
            out_compilation_unit.exprs.push_back(std::unique_ptr<ExprBase>(parsed_expr));
        }
    }
}