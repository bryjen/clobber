#include <any>
#include <optional>
#include <unordered_map>
#include <vector>

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

char try_get_char(const std::string &, int);

/* @return Returns the number of characters that fulfills the predicate starting from 'start_idx' inclusive. */
int read_char_sequence(bool (*)(char), const std::string &, int);

/* @return Returns the number of space characters starting from 'start_idx' inclusive. */
int consume_space_characters(const std::string &, int);

/* @return Returns the number of characters taken by the symbol, token type is set as an out parameter. */
int try_parse_symbol(const std::string &, int, ClobberTokenType &);

Option<ClobberTokenType> get_type_if_keyword_str(const std::string &value);

std::vector<ClobberToken>
clobber::tokenize(const std::string &source_text) {
    std::vector<ClobberToken> tokens;

    int current_idx = 0;
    int st_len      = (int)source_text.length();

    while (current_idx < st_len) {
        int full_start_idx = 0; // token start including spaces
        int start_idx      = 0; // token start EXCLUDING spaces
        int spaces_len     = 0; // length of space characters starting from the full start idx
        int token_len      = 0; // length of the actual token characters from the start idx

        ClobberTokenType token_type;
        ClobberToken token;

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
            token_type = ClobberTokenType::NumericLiteralToken;
            value      = source_text.substr(start_idx, token_len);
        } else if (isalpha(peek_char)) {
            // tokenize as identifier
            std::string value_str;
            Option<ClobberTokenType> token_type_opt;

            token_len = read_char_sequence(identifier_pred, source_text, current_idx);
            value_str = source_text.substr(start_idx, token_len);
            value     = value_str;

            token_type_opt = get_type_if_keyword_str(value_str);
            token_type     = token_type_opt ? token_type_opt.value() : ClobberTokenType::IdentifierToken;
        } else if (peek_char == '"') {
            // tokenize as string
            std::string value_str;
            Option<ClobberTokenType> token_type_opt;

            token_len = read_char_sequence(string_pred, source_text, current_idx + 1); // +1 to skip the double quot for now
            token_len += 2;                                                            // +1 to include the ending double quot
            value_str = source_text.substr(start_idx, token_len);
            value     = value_str;

            token_type = ClobberTokenType::StringLiteralToken;
        } else if (peek_char == '\'') {
            // tokenize as char
            // the char token can contain more than one char, but this is intended to be asserted in further stages in the pipeline
            std::string value_str;
            Option<ClobberTokenType> token_type_opt;

            token_len = read_char_sequence(string_pred, source_text, current_idx + 1); // +1 to skip the quot for now
            token_len += 2;                                                            // +1 to include the quot
            value_str = source_text.substr(start_idx, token_len);
            value     = value_str;

            token_type = ClobberTokenType::CharLiteralToken;
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
        token.token_type  = token_type;
        token.value       = value;

        tokens.push_back(token);
    }

    ClobberToken eof_token{};
    eof_token.full_start  = st_len;
    eof_token.start       = st_len;
    eof_token.full_length = 0;
    eof_token.length      = 0;
    eof_token.token_type  = ClobberTokenType::EofToken;
    eof_token.value       = std::string{std::char_traits<char>::eof()};

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
try_get_char(const std::string &source_text, int idx) {
    size_t total_chars = source_text.size();
    return (idx >= 0 && idx < total_chars) ? source_text[idx] : eof_char;
}

int
read_char_sequence(bool (*predicate)(char), const std::string &source_text, int start_idx) {
    int current_idx = start_idx;

    while (predicate(source_text[current_idx])) {
        current_idx++;
    }

    return current_idx - start_idx;
}

int
consume_space_characters(const std::string &source_text, int start_idx) {
    return read_char_sequence(is_space, source_text, start_idx);
}

int
try_parse_symbol(const std::string &source_text, int start_index, ClobberTokenType &out_token_type) {
    char fst_char = try_get_char(source_text, start_index);

    // we don't need to parse two character operators for now

    // clang-format off
    std::unordered_map<char, ClobberTokenType> token_map = {
        { '(', ClobberTokenType::OpenParenToken },
        { ')', ClobberTokenType::CloseParenToken },
        { '[', ClobberTokenType::OpenBracketToken },
        { ']', ClobberTokenType::CloseBracketToken },
        { '{', ClobberTokenType::OpenBraceToken },
        { '}', ClobberTokenType::CloseBraceToken },

        { '+', ClobberTokenType::PlusToken },
        { '-', ClobberTokenType::MinusToken },
        { '*', ClobberTokenType::AsteriskToken },
        { '/', ClobberTokenType::SlashToken },
        { '\\', ClobberTokenType::BackslashToken },
        { '=', ClobberTokenType::EqualsToken },
        { '<', ClobberTokenType::LessThanToken },
        { '>', ClobberTokenType::GreaterThanToken },
    };
    // clang-format on

    auto it        = token_map.find(fst_char);
    out_token_type = (it != token_map.end()) ? it->second : ClobberTokenType::BadToken;
    return 1;
}

Option<ClobberTokenType>
get_type_if_keyword_str(const std::string &value) {

    // clang-format off
    const std::unordered_map<std::string, ClobberTokenType> keyword_tokentype_map {
        {"let", ClobberTokenType::LetKeywordToken },
        {"fn", ClobberTokenType::FnKeywordToken },
        {"def", ClobberTokenType::DefKeywordToken },
        {"do", ClobberTokenType::DoKeywordToken },
        {"accel", ClobberTokenType::AccelKeywordToken },
        {"tosa-reshape", ClobberTokenType::TosaReshapeKeywordToken },
        {"tosa-transpose", ClobberTokenType::TosaTransposeKeywordToken },
        {"tosa-tile", ClobberTokenType::TosaTileKeywordToken },
        {"tosa-slice", ClobberTokenType::TosaSliceKeywordToken },
        {"tosa-concat", ClobberTokenType::TosaConcatKeywordToken },
        {"tosa-identity", ClobberTokenType::TosaIdentityKeywordToken },
        {"tosa-cast", ClobberTokenType::TosaCastKeywordToken },
        {"tosa-conv2d", ClobberTokenType::TosaConv2dKeywordToken },
        {"tosa-depthwise-conv2d", ClobberTokenType::TosaDepthwiseConv2dKeywordToken },
        {"tosa-matmul", ClobberTokenType::TosaMatmulKeywordToken },
        {"tosa-fully-connected", ClobberTokenType::TosaFullyConnectedKeywordToken },
        {"tosa-avgpool2d", ClobberTokenType::TosaAvgPool2dKeywordToken },
        {"tosa-maxpool2d", ClobberTokenType::TosaMaxPool2dKeywordToken },
        {"tosa-pad", ClobberTokenType::TosaPadKeywordToken },
        {"tosa-relu", ClobberTokenType::TosaReluKeywordToken },
        {"tosa-sigmoid", ClobberTokenType::TosaSigmoidKeywordToken },
        {"tosa-tanh", ClobberTokenType::TosaTanhKeywordToken },
        {"tosa-softmax", ClobberTokenType::TosaSoftmaxKeywordToken },
    };
    // clang-format on

    auto it = keyword_tokentype_map.find(value);
    return it != keyword_tokentype_map.end() ? std::make_optional(it->second) : std::nullopt;
}