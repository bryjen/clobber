#ifndef AST_HPP
#define AST_HPP

#include <clobber/common/debug.hpp> // common debug header

#include <any>
#include <memory>
#include <string>
#include <vector>

struct ParserError; // clobber/parser.hpp

/* @brief Represents the type of a token.
 */
enum class ClobberTokenType {
    OpenParenToken,
    CloseParenToken,
    OpenBracketToken,
    CloseBracketToken,
    OpenBraceToken,
    CloseBraceToken,

    PlusToken,
    MinusToken,
    AsteriskToken,
    SlashToken,     // '/'
    BackslashToken, // '\'
    EqualsToken,
    LessThanToken,    // '<'
    GreaterThanToken, // '>'

    IdentifierToken,
    NumericLiteralToken,

    BadToken,
    EofToken,
};

/* @brief Represents a token.
 */
struct ClobberToken final {
    int start;
    int length;
    int full_start;  // includes trivia
    int full_length; // includes trivia

    ClobberTokenType token_type;
    std::any value;

public:
    std::string ExtractText(const std::string &);
    std::string ExtractFullText(const std::string &);

    static bool AreEquivalent(const ClobberToken &, const ClobberToken &);
};

/* @brief Enum representing the expression type.
 */
enum class ClobberExprType {
    CallExpr,
    NumericLiteralExpr,
    IdentifierExpr,
};

/* @brief Represents a clobber expression. Base class for all expression types.
 */
struct ExprBase {
    ClobberExprType expr_type;
    virtual ~ExprBase() = default;
};

/* @brief Represents a numerical literal expression.
 */
struct NumLiteralExpr final : ExprBase {
    int value;
    ClobberToken token;
};

/* @brief Represents a call expression.
 */
struct CallExpr final : ExprBase {
    ClobberToken open_paren_token;
    ClobberToken operator_token;
    ClobberToken close_paren_token;
    std::vector<std::unique_ptr<ExprBase>> arguments;
};

/* @brief Represents an identifier.
 */
struct IdentifierExpr final : ExprBase {
    std::string name;
    ClobberToken token;
};

/* @brief Represents a clobber compilation unit. Usually contains all the contents of a source file.
 */
struct CompilationUnit {
    std::vector<std::unique_ptr<ExprBase>> exprs;
    std::vector<ParserError> parse_errors;
};

#endif // AST_HPP