#ifndef AST_HPP
#define AST_HPP

#include <any>
#include <memory>
#include <string>
#include <vector>

struct ParserError; // clobber/parser.hpp

/* @brief Represents the type of a token.
 */
enum class TokenType {
    OpenParen,
    CloseParen,
    PlusToken,
    EqualsToken,
    BadToken,

    IdentifierToken,
    NumericLiteralToken,
};

/* @brief Represents a token.
 */
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

/* @brief Enum representing the expression type.
 */
enum class ExprType {
    CallExpr,
    NumericLiteralExpr,
};

/* @brief Represents a clobber expression. Base class for all expression types.
 */
struct ExprBase {
    ExprType expr_type;
    virtual ~ExprBase() = default;
};

/* @brief Represents a numerical literal expression.
 */
struct NumLiteralExpr final : ExprBase {
    int value;
};

/* @brief Represents a call expression.
 */
struct CallExpr final : ExprBase {
    Token operator_token;
    std::vector<std::unique_ptr<ExprBase>> arguments;
};

/* @brief Represents a clobber compilation unit. Usually contains all the contents of a source file.
 */
struct CompilationUnit {
    std::vector<std::unique_ptr<ExprBase>> exprs;
    std::vector<ParserError> parse_errors;
};

#endif // AST_HPP