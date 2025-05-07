#include <memory>
#include <optional>
#include <unordered_map>

#include "clobber/ast.hpp"
#include "clobber/parser.hpp"

#include "clobber/internal/parser_error_factory.hpp"

// clang-format off
template <typename T> 
using Option = std::optional<T>;

using ParseDelegate = ExprBase* (*)(const std::string &, const std::vector<Token> &, std::vector<ParserError> &, size_t &);

ExprBase* try_parse(const std::string &, const std::vector<Token> &, std::vector<ParserError> &, size_t &);
ExprBase* try_parse_numeric_literal_expr(const std::string &, const std::vector<Token> &, std::vector<ParserError> &, size_t &);
ExprBase* try_parse_call_expr(const std::string &, const std::vector<Token> &, std::vector<ParserError> &, size_t &);
// clang-format on

void
recover(size_t &idx) {
    idx = idx + 1;
}

Option<Token>
try_get_token(const std::vector<Token> &tokens, size_t idx) {
    size_t tokens_len = tokens.size();
    return (idx >= 0 && idx < tokens_len) ? std::make_optional(tokens[idx]) : std::nullopt;
}

Option<ParseDelegate>
try_get_parse_fun(TokenType token_type) {
    // clang-format off
    static std::unordered_map<TokenType, ParseDelegate> parse_functions = {
        { TokenType::NumericLiteralToken, try_parse_numeric_literal_expr },
        { TokenType::OpenParen, try_parse_call_expr },
    };
    // clang-format on

    auto it = parse_functions.find(token_type);
    return (it != parse_functions.end()) ? std::make_optional(it->second) : std::nullopt;
}

Option<int>
try_stoi(const std::string &str) {
    Option<int> opt = std::nullopt;
    try {
        opt = std::make_optional(std::stoi(str));
    } catch (...) {
        // ignored
    }
    return opt;
}

ExprBase *
try_parse_numeric_literal_expr(const std::string &source_text, const std::vector<Token> &tokens, std::vector<ParserError> &parser_errors,
                               size_t &idx) {
    NumLiteralExpr *nle;

    Token current_token;
    std::string num_as_str;
    Option<int> stoi_results;

    current_token = tokens[idx]; // no bounds check, current token exists, asserted by caller
    num_as_str    = current_token.ExtractText(source_text);
    stoi_results  = try_stoi(num_as_str);

    if (!stoi_results) {
        ParserError err = err::InternalErr(current_token.start, current_token.length);
        parser_errors.push_back(err);
        recover(idx);
        return nullptr;
    }

    nle            = new NumLiteralExpr();
    nle->value     = stoi_results.value();
    nle->expr_type = ExprType::NumericLiteralExpr;

    idx++;
    return nle;
}

ExprBase *
try_parse_call_expr(const std::string &source_text, const std::vector<Token> &tokens, std::vector<ParserError> &parser_errors,
                    size_t &idx) {
    CallExpr *ce;

    Token operator_token;
    Option<Token> current_token;
    std::vector<ExprBase> arg_exprs;

    operator_token = tokens[++idx];
    current_token  = try_get_token(tokens, ++idx);

    ce                 = new CallExpr();
    ce->arguments      = std::vector<std::unique_ptr<ExprBase>>{};
    ce->operator_token = operator_token;
    ce->expr_type      = ExprType::CallExpr;

    while (current_token && current_token.value().token_type != TokenType::CloseParen) {
        ExprBase *arg_expr = try_parse(source_text, tokens, parser_errors, idx);
        if (arg_expr) {
            ce->arguments.push_back(std::unique_ptr<ExprBase>(arg_expr));
        }

        current_token = try_get_token(tokens, idx);
    }

    idx++;
    return ce;
}

ExprBase *
try_parse(const std::string &source_text, const std::vector<Token> &tokens, std::vector<ParserError> &parser_errors, size_t &idx) {
    Token current_token;
    Option<Token> token_opt;
    ParseDelegate parse_fn;
    Option<ParseDelegate> parse_fn_opt;

    token_opt = try_get_token(tokens, idx);
    if (!token_opt) {
        return nullptr;
    }
    current_token = token_opt.value();

    parse_fn_opt = try_get_parse_fun(current_token.token_type);
    if (!parse_fn_opt) {
        ParserError err = err::InternalErr(current_token.start, current_token.length);
        parser_errors.push_back(err);
        recover(idx);
        return nullptr;
    }
    parse_fn = parse_fn_opt.value();

    return parse_fn(source_text, tokens, parser_errors, idx);
}

void
clobber::parse(const std::string &source_text, const std::vector<Token> &tokens, CompilationUnit &out_compilation_unit) {
    CompilationUnit cu;
    size_t current_idx;
    size_t tokens_len;

    out_compilation_unit.parse_errors = std::vector<ParserError>{};
    out_compilation_unit.exprs        = std::vector<std::unique_ptr<ExprBase>>{};

    current_idx = 0;
    tokens_len  = tokens.size();

    while (current_idx < tokens_len) {
        // 'current_idx' passed by reference, implicitly modified
        ExprBase *parsed_expr = try_parse(source_text, tokens, out_compilation_unit.parse_errors, current_idx);
        if (parsed_expr) {
            out_compilation_unit.exprs.push_back(std::unique_ptr<ExprBase>(parsed_expr));
        }

        current_idx++;
    }
}