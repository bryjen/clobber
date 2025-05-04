#ifndef AST_HPP
#define AST_HPP

#include <any>
#include <string>
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

public:
    std::string ExtractText(const std::string &);
    std::string ExtractFullText(const std::string &);
};

/*
enum class ExprType {
    CallExpr,
    NumericLiteralExpr,
};
*/

struct ExprBase {
    // ExprType expr_type;
};

struct NumLiteralExpr final : ExprBase {
    int value;
    // possible support for floating point stuff goes here
};

struct CallExpr final : ExprBase {
    Token operator_token;
    std::vector<ExprBase> arguments;
};

struct CompilationUnit {
    std::vector<ExprBase> exprs;
};

#endif // AST_HPP