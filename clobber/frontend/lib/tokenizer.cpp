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
char try_get_char(const std::string &, int);

constexpr bool (*identifier_pred)(char) = [](char c) -> bool { return (c == '-') || is_alphanumeric(c); };

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
            token_len                 = read_char_sequence(is_numeric, source_text, current_idx);
            token_type                = ClobberTokenType::NumericLiteralToken;
            value_str                 = source_text.substr(start_idx, token_len);
            Option<int> int_value_opt = str_utils::try_stoi(value_str);
            if (int_value_opt) {
                value = int_value_opt.value();
            } else {
                // TODO: Throw an error here
            }
        } else if (isalpha(peek_char)) {
            // tokenize as identifier or string literal
            std::string value_str;
            Option<ClobberTokenType> token_type_opt;

            token_len = read_char_sequence(identifier_pred, source_text, current_idx);
            value_str = source_text.substr(start_idx, token_len);
            value     = value_str;

            token_type_opt = get_type_if_keyword_str(value_str);
            token_type     = token_type_opt ? token_type_opt.value() : ClobberTokenType::IdentifierToken;
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
    return c >= '0' && c <= '9';
}

bool
is_alphanumeric(char c) {
    return is_alpha(c) || is_numeric(c);
}

bool
is_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
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
    if (value == "let") {
        return std::make_optional(ClobberTokenType::LetKeywordToken);

    } else if (value == "fn") {
        return std::make_optional(ClobberTokenType::FnKeywordToken);

    } else if (value == "def") {
        return std::make_optional(ClobberTokenType::DefKeywordToken);

    } else if (value == "do") {
        return std::make_optional(ClobberTokenType::DoKeywordToken);

    } else if (value == "accel") {
        return std::make_optional(ClobberTokenType::AccelKeywordToken);

    } else if (value == "tosa-reshape") {
        return std::make_optional(ClobberTokenType::TosaReshapeKeywordToken);

    } else if (value == "tosa-transpose") {
        return std::make_optional(ClobberTokenType::TosaTransposeKeywordToken);

    } else if (value == "tosa-tile") {
        return std::make_optional(ClobberTokenType::TosaTileKeywordToken);

    } else if (value == "tosa-slice") {
        return std::make_optional(ClobberTokenType::TosaSliceKeywordToken);

    } else if (value == "tosa-concat") {
        return std::make_optional(ClobberTokenType::TosaConcatKeywordToken);

    } else if (value == "tosa-identity") {
        return std::make_optional(ClobberTokenType::TosaIdentityKeywordToken);

    } else if (value == "tosa-cast") {
        return std::make_optional(ClobberTokenType::TosaCastKeywordToken);

    } else if (value == "tosa-conv2d") {
        return std::make_optional(ClobberTokenType::TosaConv2dKeywordToken);

    } else if (value == "tosa-depthwise-conv2d") {
        return std::make_optional(ClobberTokenType::TosaDepthwiseConv2dKeywordToken);

    } else if (value == "tosa-matmul") {
        return std::make_optional(ClobberTokenType::TosaMatmulKeywordToken);

    } else if (value == "tosa-fully-connected") {
        return std::make_optional(ClobberTokenType::TosaFullyConnectedKeywordToken);

    } else if (value == "tosa-avgpool2d") {
        return std::make_optional(ClobberTokenType::TosaAvgPool2dKeywordToken);

    } else if (value == "tosa-maxpool2d") {
        return std::make_optional(ClobberTokenType::TosaMaxPool2dKeywordToken);

    } else if (value == "tosa-pad") {
        return std::make_optional(ClobberTokenType::TosaPadKeywordToken);

    } else if (value == "tosa-relu") {
        return std::make_optional(ClobberTokenType::TosaReluKeywordToken);

    } else if (value == "tosa-sigmoid") {
        return std::make_optional(ClobberTokenType::TosaSigmoidKeywordToken);

    } else if (value == "tosa-tanh") {
        return std::make_optional(ClobberTokenType::TosaTanhKeywordToken);

    } else if (value == "tosa-softmax") {
        return std::make_optional(ClobberTokenType::TosaSoftmaxKeywordToken);
    }

    return std::nullopt;
}