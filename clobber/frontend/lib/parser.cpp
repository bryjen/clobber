#include "clobber/parser.hpp"
#include "clobber/ast.hpp"
#include <any>
#include <optional>
#include <unordered_map>
#include <vector>

template <typename T> using Option = std::optional<T>;

const char eof_char = std::char_traits<char>::eof();

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
try_get_char(int idx) {
    return eof_char;
}

/*
 * \brief
 * \return Returns the number of characters that fulfills the predicate starting from 'start_idx' inclusive.
 */
int
read_char_sequence(bool (*predicate)(char), const std::string &source_text, int start_idx) {
    int current_idx = start_idx;

    while (predicate(source_text[current_idx])) {
        current_idx++;
    }

    return current_idx - start_idx;
}

/*
 * \brief
 * \return Returns the number of space characters starting from 'start_idx' inclusive.
 */
int
consume_space_characters(const std::string &source_text, int start_idx) {
    return read_char_sequence(is_space, source_text, start_idx);
}

int
try_parse_token_type(const std::string &source_text, int start_index, TokenType &out_token_type) {
    char fst_char = try_get_char(start_index);

    // we don't need to parse two character operators for now

    // clang-format off
    std::unordered_map<char, TokenType> token_map = {
        { '(', TokenType::OpenParen },
        { ')', TokenType::CloseParen },
        { '+', TokenType::PlusToken },
        { '=', TokenType::EqualsToken },
    };
    // clang-format on

    auto it        = token_map.find(fst_char);
    out_token_type = (it != token_map.end()) ? it->second : TokenType::BadToken;
    return 1;
}

std::vector<Token>
Tokenize(const std::string &source_text) {
    std::vector<Token> tokens;

    int current_idx = 0;
    int st_len      = source_text.length();

    while (current_idx < st_len) {
        int full_start_idx = 0; // token start including spaces
        int start_idx      = 0; // token start EXCLUDING spaces
        int spaces_len     = 0; // length of space characters starting from the full start idx
        int token_len      = 0; // length of the actual token characters from the start idx
        TokenType token_type;
        std::any value;
        Token token;

        spaces_len     = consume_space_characters(source_text, current_idx);
        full_start_idx = current_idx;
        start_idx      = full_start_idx + spaces_len;
        current_idx    = start_idx;

        char peek_char = try_get_char(current_idx);
        if (peek_char == eof_char) {
            break;
        } else if (is_numeric(peek_char)) {
            // tokenize as number
            int token_len = read_char_sequence(is_numeric, source_text, current_idx);
            token_type    = TokenType::NumericLiteralToken;
        } else if (isalpha(peek_char)) {
            // tokenize as identifier or string literal
            int token_len = read_char_sequence(is_alphanumeric, source_text, current_idx);
            token_type    = TokenType::IdentifierToken;
        } else {
            // tokenize as symbol
            int token_len = try_parse_token_type(source_text, current_idx, token_type);
        }

        token.start       = start_idx;
        token.length      = token_len;
        token.full_start  = full_start_idx;
        token.full_length = spaces_len + token_len;
        token.token_type  = token_type;
        token.value       = value;
        tokens.push_back(token);
    }

    return tokens;
}