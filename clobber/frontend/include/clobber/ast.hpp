#ifndef AST_HPP
#define AST_HPP

#include <any>
#include <vector>

enum class TokenType {
    OpenParen,
    CloseParen,
    PlusToken,
    EqualsToken,
    BadToken,

    IdentifierToken,
    NumericLiteralToken,
};

struct Token final {
    int start;
    int length;
    int full_start;  // includes trivia
    int full_length; // includes trivia

    TokenType token_type;
    std::any value;
};

enum class ExprType {
    CallExpr,
    NumericLiteralExpr,
};

struct ExprBase {
    ExprType expr_type;
};

struct CallExpr final : ExprBase {
    Token operator_token;
    std::vector<ExprBase> arguments;
};

#endif // AST_HPP