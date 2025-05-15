#pragma once

#include "clobber/ast.hpp"

/*
#define DEFINE_TOKEN_FUNC(NAME, STR)                                                                                                       \
    inline clobber::ClobberToken NAME() {                                                                                                  \
        ClobberToken token{};                                                                                                              \
        token.token_type = clobber::ClobberTokenType::NAME##Token;                                                                         \
        token.value      = std::string(STR);                                                                                               \
        return token;                                                                                                                      \
    }
*/

#define DEFINE_TOKEN_FUNC(NAME, STR)                                                                                                       \
    inline clobber::Token NAME() {                                                                                                         \
        clobber::Token token{};                                                                                                            \
        token.type = clobber::Token::Type::NAME##Token;                                                                                    \
        return token;                                                                                                                      \
    }

namespace SyntaxFactory {

    DEFINE_TOKEN_FUNC(OpenParen, "(")
    DEFINE_TOKEN_FUNC(CloseParen, ")")
    DEFINE_TOKEN_FUNC(OpenBracket, "[")
    DEFINE_TOKEN_FUNC(CloseBracket, "]")
    DEFINE_TOKEN_FUNC(OpenBrace, "{")
    DEFINE_TOKEN_FUNC(CloseBrace, "}")

    DEFINE_TOKEN_FUNC(Plus, "+")
    DEFINE_TOKEN_FUNC(Minus, "-")
    DEFINE_TOKEN_FUNC(Asterisk, "*")
    DEFINE_TOKEN_FUNC(Slash, "/")
    DEFINE_TOKEN_FUNC(Backslash, "\\")
    DEFINE_TOKEN_FUNC(Equals, "\\")
    DEFINE_TOKEN_FUNC(LessThan, "<")
    DEFINE_TOKEN_FUNC(GreaterThan, ">")
    DEFINE_TOKEN_FUNC(Eof, std::string{std::char_traits<char>::eof()})

    DEFINE_TOKEN_FUNC(LetKeyword, "let")
    DEFINE_TOKEN_FUNC(DefKeyword, "def")
    DEFINE_TOKEN_FUNC(FnKeyword, "fn")

    /* @brief Constructs a string literal token, inserts the double quotes into the value provided. */
    inline clobber::Token
    StringLiteralInsertDoubleQuot(std::string value) {
        clobber::Token token{};
        token.type = clobber::Token::Type::StringLiteralToken;
        // token.value      = std::format("\"{}\"", value);
        return token;
    }

    inline clobber::Token
    CharLiteral(char c) {
        clobber::Token token{};
        token.type = clobber::Token::Type::CharLiteralToken;
        // token.value      = std::format("'{}'", c);
        return token;
    }

    inline clobber::Token
    Identifier(std::string name) {
        clobber::Token token{};
        token.type = clobber::Token::Type::IdentifierToken;
        // token.value      = name;
        return token;
    }

    inline clobber::Token
    NumericLiteral(int value) {
        clobber::Token token{};
        token.type = clobber::Token::Type::NumericLiteralToken;
        // token.value      = std::to_string(value);
        return token;
    }

    inline clobber::Token
    NumericLiteral(float value, int decimal_places = 2) {
        clobber::Token token{};
        token.type = clobber::Token::Type::NumericLiteralToken;
        // token.value      = std::format("{:.{}f}f", value, decimal_places);
        return token;
    }

    inline clobber::Token
    NumericLiteral(double value, int decimal_places = 2, bool postfix_d = false) {
        clobber::Token token{};
        token.type = clobber::Token::Type::NumericLiteralToken;
        // token.value      = postfix_d ? std::format("{:.{}f}d", value, decimal_places) : std::format("{:.{}f}", value, decimal_places);
        return token;
    }

    inline clobber::Token
    BadToken() {
        clobber::Token token{};
        token.type = clobber::Token::Type::BadToken;
        // return token;
    }

}; // namespace SyntaxFactory