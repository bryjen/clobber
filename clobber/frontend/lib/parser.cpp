#include <optional>
#include <unordered_map>

#include "clobber/ast.hpp"
#include "clobber/parser.hpp"

#include "clobber/internal/parser_errors.hpp"

ParserError::ParserError() {}

ParserError::~ParserError() {}

ParserError::ParserError(int span_start, int span_len, const std::string &general_err_msg, const std::string &err_msg) {
    this->span_start      = span_start;
    this->span_len        = span_len;
    this->general_err_msg = general_err_msg;
    this->err_msg         = err_msg;
}

template <typename T> using Option = std::optional<T>;
using ParseDelegate = Option<ExprBase> (*)(const std::string &, const std::vector<Token> &, std::vector<ParserError> &,
                                           size_t &);

// clang-format off
Option<ExprBase> try_parse(const std::string &, const std::vector<Token> &, std::vector<ParserError> &, size_t &);
Option<ExprBase> try_parse_numeric_literal_expr(const std::string &, const std::vector<Token> &, std::vector<ParserError> &, size_t &);
Option<ExprBase> try_parse_call_expr(const std::string &, const std::vector<Token> &, std::vector<ParserError> &, size_t &);
// clang-format on

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

Option<ExprBase>
try_parse_numeric_literal_expr(const std::string &source_text, const std::vector<Token> &tokens,
                               std::vector<ParserError> &parser_errors, size_t &idx) {
    NumLiteralExpr nle;
    Token current_token;
    std::string num_as_str;
    Option<int> stoi_results;

    current_token = tokens[idx]; // no bounds check, current token exists, asserted by caller
    num_as_str    = current_token.ExtractText(source_text);
    stoi_results  = try_stoi(num_as_str);

    if (!stoi_results) {
        // TODO: Add error
        return std::nullopt;
    }

    idx++;
    nle.value = stoi_results.value();
    return std::make_optional(nle);
}

Option<ExprBase>
try_parse_call_expr(const std::string &source_text, const std::vector<Token> &tokens,
                    std::vector<ParserError> &parser_errors, size_t &idx) {
    CallExpr ce;
    Token operator_token;
    Option<Token> current_token;
    std::vector<ExprBase> arg_exprs;

    operator_token = tokens[++idx];
    current_token  = try_get_token(tokens, ++idx);

    while (current_token && current_token.value().token_type != TokenType::CloseParen) {
        Option<ExprBase> arg_expr_opt = try_parse(source_text, tokens, parser_errors, idx);
        if (arg_expr_opt) {
            arg_exprs.push_back(arg_expr_opt.value());
        }

        current_token = try_get_token(tokens, idx);
    }

    ce.operator_token = operator_token;
    ce.arguments      = arg_exprs;
    return std::make_optional(ce);
}

Option<ExprBase>
try_parse(const std::string &source_text, const std::vector<Token> &tokens, std::vector<ParserError> &parser_errors,
          size_t &idx) {
    Token current_token;
    Option<Token> token_opt;
    ParseDelegate parse_fn;
    Option<ParseDelegate> parse_fn_opt;
    Option<ExprBase> expr_opt;

    expr_opt = std::nullopt;

    token_opt = try_get_token(tokens, idx);
    if (!token_opt) {
        return expr_opt;
    }
    current_token = token_opt.value();

    parse_fn_opt = try_get_parse_fun(current_token.token_type);
    if (!parse_fn_opt) {
        // TODO: error here
        return expr_opt;
    }
    parse_fn = parse_fn_opt.value();

    expr_opt = parse_fn(source_text, tokens, parser_errors, idx);
    return expr_opt;
}

CompilationUnit
clobber::parse(const std::string &source_text, const std::vector<Token> &tokens,
               std::vector<ParserError> &out_parser_errors) {
    CompilationUnit cu;
    size_t current_idx;
    size_t tokens_len;
    std::vector<ExprBase> exprs;

    current_idx = 0;
    tokens_len  = tokens.size();
    out_parser_errors.clear();

    while (current_idx < tokens_len) {
        // 'current_idx' passed by reference, implicitly modified
        Option<ExprBase> parsed_expr_opt = try_parse(source_text, tokens, out_parser_errors, current_idx);
        if (parsed_expr_opt) {
            exprs.push_back(parsed_expr_opt.value());
        }

        current_idx++;
    }

    cu.exprs = exprs;
    return cu;
}