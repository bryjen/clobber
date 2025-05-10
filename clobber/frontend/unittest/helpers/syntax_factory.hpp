#ifndef SYNTAX_FACTORY_HPP
#define SYNTAX_FACTORY_HPP

#include "clobber/ast.hpp"

#define DEFINE_TOKEN_FUNC(NAME, STR)                                                                                                       \
    inline ClobberToken NAME() {                                                                                                           \
        ClobberToken token{};                                                                                                              \
        token.token_type = ClobberTokenType::NAME##Token;                                                                                  \
        token.value      = std::string(STR);                                                                                               \
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

inline ClobberToken
Identifier(std::string name) {
    ClobberToken token{};
    token.token_type = ClobberTokenType::IdentifierToken;
    token.value      = name;
    return token;
}

inline ClobberToken
NumericLiteral(int value) {
    ClobberToken token{};
    token.token_type = ClobberTokenType::NumericLiteralToken;
    token.value      = value;
    return token;
}

inline ClobberToken
BadToken() {
    ClobberToken token{};
    token.token_type = ClobberTokenType::BadToken;
    return token;
}

}; // namespace SyntaxFactory

#endif // SYNTAX_FACTORY_HPP