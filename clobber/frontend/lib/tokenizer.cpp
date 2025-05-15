#include "clobber/pch.hpp"

#include <clobber/common/utils.hpp>

#include "clobber/ast.hpp"
#include "clobber/parser.hpp"

template <typename T> using Option = std::optional<T>;
const char eof_char                = std::char_traits<char>::eof();

// fwd declarations
bool is_alpha(char);
bool is_numeric(char);
bool is_alphanumeric(char);
bool is_space(char);
bool is_symbol(char);

bool number_pred(char);
bool identifier_pred(char);
bool string_pred(char);

char try_get_char(const std::string &, size_t);

/* @return Returns the number of characters that fulfills the predicate starting from 'start_idx' inclusive. */
size_t read_char_sequence(bool (*)(char), const std::string &, size_t);

/* @return Returns the number of space characters starting from 'start_idx' inclusive. */
size_t consume_space_characters(const std::string &, size_t);

/* @return Returns the number of characters taken by the symbol, token type is set as an out parameter. */
size_t try_parse_symbol(const std::string &, size_t, clobber::Token::Type &);

Option<clobber::Token::Type> get_type_if_keyword_str(const std::string &value);

std::vector<clobber::Token>
clobber::tokenize(const std::string &source_text) {
    std::vector<clobber::Token> tokens;

    size_t current_idx = 0;
    size_t st_len      = source_text.length();

    while (current_idx < st_len) {
        size_t full_start_idx = 0; // token start including spaces
        size_t start_idx      = 0; // token start EXCLUDING spaces
        size_t spaces_len     = 0; // length of space characters starting from the full start idx
        size_t token_len      = 0; // length of the actual token characters from the start idx

        clobber::Token::Type token_type;
        clobber::Token token;

        std::any value;
        std::string value_str; // string representation of the value

        spaces_len     = consume_space_characters(source_text, current_idx);
        full_start_idx = current_idx;
        start_idx      = full_start_idx + spaces_len;
        current_idx    = start_idx;

        char peek_char = try_get_char(source_text, current_idx);
        if (peek_char == eof_char) {
            break;
        } else if (is_numeric(peek_char)) {
            // tokenize as number
            token_len  = read_char_sequence(number_pred, source_text, current_idx);
            token_type = clobber::Token::Type::NumericLiteralToken;
            value      = source_text.substr(start_idx, token_len);
        } else if (isalpha(peek_char)) {
            // tokenize as identifier
            Option<clobber::Token::Type> token_type_opt;

            token_len = read_char_sequence(identifier_pred, source_text, current_idx);
            value_str = source_text.substr(start_idx, token_len);
            value     = value_str;

            token_type_opt = get_type_if_keyword_str(value_str);
            token_type     = token_type_opt ? token_type_opt.value() : clobber::Token::Type::IdentifierToken;
        } else if (peek_char == '"') {
            // tokenize as string

            token_len = read_char_sequence(string_pred, source_text, current_idx + 1); // +1 to skip the double quot for now
            token_len += 2;                                                            // +1 to include the ending double quot
            value_str = source_text.substr(start_idx, token_len);
            value     = value_str;

            token_type = clobber::Token::Type::StringLiteralToken;
        } else if (peek_char == '\'') {
            // tokenize as char
            // the char token can contain more than one char, but this is intended to be asserted in further stages in the pipeline

            token_len = read_char_sequence(string_pred, source_text, current_idx + 1); // +1 to skip the quot for now
            token_len += 2;                                                            // +1 to include the quot
            value_str = source_text.substr(start_idx, token_len);
            value     = value_str;

            token_type = clobber::Token::Type::CharLiteralToken;
        } else {
            // tokenize as symbol
            token_len = try_parse_symbol(source_text, current_idx, token_type);
            value     = source_text.substr(start_idx, token_len);
        }

        current_idx += token_len;

        token.start       = start_idx;
        token.length      = token_len;
        token.full_start  = full_start_idx;
        token.full_length = spaces_len + token_len;
        token.type        = token_type;
        // token.value       = value;

        tokens.push_back(token);
    }

    clobber::Token eof_token{};
    eof_token.full_start  = st_len;
    eof_token.start       = st_len;
    eof_token.full_length = 0;
    eof_token.length      = 0;
    eof_token.type        = clobber::Token::Type::EofToken;
    // eof_token.value       = std::string{std::char_traits<char>::eof()};

    tokens.push_back(eof_token);

    return tokens;
}

bool
is_alpha(char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

bool
is_numeric(char c) {
    return (c >= '0' && c <= '9');
}

bool
is_alphanumeric(char c) {
    return is_alpha(c) || is_numeric(c);
}

bool
is_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

bool
is_symbol(char c) {
    static constexpr std::string_view extras = "!$%&*+-,./:<=>?@^_~";
    return std::isalnum(static_cast<unsigned char>(c)) || extras.find(c) != std::string_view::npos;
}

// supports both integer and floating point types
// as well as optional floating point specifiers
bool
number_pred(char c) {
    return (c == '.') || is_alphanumeric(c);
}

bool
identifier_pred(char c) {
    return (c == '-') || is_alphanumeric(c);
}

bool
string_pred(char c) {
    return is_symbol(c) || is_alphanumeric(c) || is_space(c);
}

char
try_get_char(const std::string &source_text, size_t idx) {
    size_t total_chars = source_text.size();
    return (idx >= 0 && idx < total_chars) ? source_text[idx] : eof_char;
}

size_t
read_char_sequence(bool (*predicate)(char), const std::string &source_text, size_t start_idx) {
    size_t current_idx = start_idx;

    while (predicate(source_text[current_idx])) {
        current_idx++;
    }

    return current_idx - start_idx;
}

size_t
consume_space_characters(const std::string &source_text, size_t start_idx) {
    return read_char_sequence(is_space, source_text, start_idx);
}

size_t
try_parse_symbol(const std::string &source_text, size_t start_index, clobber::Token::Type &out_token_type) {
    char fst_char = try_get_char(source_text, start_index);

    // we don't need to parse two character operators for now

    // clang-format off
    std::unordered_map<char, clobber::Token::Type> token_map = {
        { '(', clobber::Token::Type::OpenParenToken },
        { ')', clobber::Token::Type::CloseParenToken },
        { '[', clobber::Token::Type::OpenBracketToken },
        { ']', clobber::Token::Type::CloseBracketToken },
        { '{', clobber::Token::Type::OpenBraceToken },
        { '}', clobber::Token::Type::CloseBraceToken },
        { '+', clobber::Token::Type::PlusToken },
        { '-', clobber::Token::Type::MinusToken },
        { '*', clobber::Token::Type::AsteriskToken },
        { '/', clobber::Token::Type::SlashToken },
        { '\\', clobber::Token::Type::BackslashToken },
        { '=', clobber::Token::Type::EqualsToken },
        { '<', clobber::Token::Type::LessThanToken },
        { '>', clobber::Token::Type::GreaterThanToken },
    };
    // clang-format on

    auto it        = token_map.find(fst_char);
    out_token_type = (it != token_map.end()) ? it->second : clobber::Token::Type::BadToken;
    return 1;
}

Option<clobber::Token::Type>
get_type_if_keyword_str(const std::string &value) {
    // clang-format off
    const std::unordered_map<std::string, clobber::Token::Type> keyword_tokentype_map {
        {"let", clobber::Token::Type::LetKeywordToken },
        {"fn", clobber::Token::Type::FnKeywordToken },
        {"def", clobber::Token::Type::DefKeywordToken },
        {"do", clobber::Token::Type::DoKeywordToken },
        {"accel", clobber::Token::Type::AccelKeywordToken },
        {"tosa-reshape", clobber::Token::Type::TosaReshapeKeywordToken },
        {"tosa-transpose", clobber::Token::Type::TosaTransposeKeywordToken },
        {"tosa-tile", clobber::Token::Type::TosaTileKeywordToken },
        {"tosa-slice", clobber::Token::Type::TosaSliceKeywordToken },
        {"tosa-concat", clobber::Token::Type::TosaConcatKeywordToken },
        {"tosa-identity", clobber::Token::Type::TosaIdentityKeywordToken },
        {"tosa-cast", clobber::Token::Type::TosaCastKeywordToken },
        {"tosa-conv2d", clobber::Token::Type::TosaConv2dKeywordToken },
        {"tosa-depthwise-conv2d", clobber::Token::Type::TosaDepthwiseConv2dKeywordToken },
        {"tosa-matmul", clobber::Token::Type::TosaMatmulKeywordToken },
        {"tosa-fully-connected", clobber::Token::Type::TosaFullyConnectedKeywordToken },
        {"tosa-avgpool2d", clobber::Token::Type::TosaAvgPool2dKeywordToken },
        {"tosa-maxpool2d", clobber::Token::Type::TosaMaxPool2dKeywordToken },
        {"tosa-pad", clobber::Token::Type::TosaPadKeywordToken },
        {"tosa-relu", clobber::Token::Type::TosaReluKeywordToken },
        {"tosa-sigmoid", clobber::Token::Type::TosaSigmoidKeywordToken },
        {"tosa-tanh", clobber::Token::Type::TosaTanhKeywordToken },
        {"tosa-softmax", clobber::Token::Type::TosaSoftmaxKeywordToken },
    };
    // clang-format on

    auto it = keyword_tokentype_map.find(value);
    return it != keyword_tokentype_map.end() ? std::make_optional(it->second) : std::nullopt;
}