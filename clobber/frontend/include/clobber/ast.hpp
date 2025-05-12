#pragma once

#include <clobber/common/debug.hpp> // common debug header

#include "pch.hpp"

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
    StringLiteralToken,
    CharLiteralToken,

    LetKeywordToken,
    FnKeywordToken,
    DefKeywordToken,
    DoKeywordToken,

    // hardware acceleration tokens
    AccelKeywordToken, // `accel`

    TosaReshapeKeywordToken,   // `tosa-reshape`
    TosaTransposeKeywordToken, // `tosa-transpose`
    TosaTileKeywordToken,      // `tosa-tile`
    TosaSliceKeywordToken,     // `tosa-slice`
    TosaConcatKeywordToken,    // `tosa-concat`
    TosaIdentityKeywordToken,  // `tosa-identity`
    TosaCastKeywordToken,      // `tosa-cast`

    TosaConv2dKeywordToken,          // `tosa-conv2d`
    TosaDepthwiseConv2dKeywordToken, // `tosa-depthwise-conv2d`
    TosaMatmulKeywordToken,          // `tosa-matmul`
    TosaFullyConnectedKeywordToken,  // `tosa-fully-connected`
    TosaAvgPool2dKeywordToken,       // `tosa-avgpool2d`
    TosaMaxPool2dKeywordToken,       // `tosa-maxpool2d`
    TosaPadKeywordToken,             // `tosa-pad`

    TosaReluKeywordToken,    // `tosa-relu`
    TosaSigmoidKeywordToken, // `tosa-sigmoid`
    TosaTanhKeywordToken,    // `tosa-tanh`
    TosaSoftmaxKeywordToken, // `tosa-softmax`

    BadToken,
    EofToken,
};

/* @brief Represents a token.
 */
struct ClobberToken final {
    size_t start;
    size_t length;
    size_t full_start;  // includes trivia
    size_t full_length; // includes trivia

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
    IdentifierExpr,
    ParameterVectorExpr,
    BindingVectorExpr,

    NumericLiteralExpr,
    StringLiteralExpr,
    CharLiteralExpr,

    LetExpr,
    FnExpr,
    DefExpr,
    DoExpr,

    AccelExpr,
    MatMulExpr,
    RelUExpr,
};

// TODO:
// binding and parameterized vectors are not expressions, they do not have the same properties as expressions (Ex. they don't have a return
// type).

/* @brief Represents a clobber expression. Base class for all expression types.
 */
struct ExprBase {
    // used for hashing down the line
    size_t id = nextId++;

    ClobberExprType expr_type;

    virtual ~ExprBase() = default;

private:
    static inline size_t nextId = 0;
};

/* @brief Represents a numerical literal expression.
 */
struct NumLiteralExpr final : ExprBase {
    ClobberToken token;
};

/* @brief Represents a string literal expression.
 */
struct StringLiteralExpr final : ExprBase {
    std::string value;
    ClobberToken token;
};

/* @brief Represents a char literal expression.
 * @remarks In the tokenizer, '...' can contain multiple characters. Although this can be asserted in the parser, we delegate this
 * responsibility to the semantic analyzer. The value does not contain the single quotes.
 */
struct CharLiteralExpr final : ExprBase {
    std::string value;
    ClobberToken token;
};

/* @brief Represents an identifier.
 */
struct IdentifierExpr final : ExprBase {
    std::string name;
    ClobberToken token;
};

/* @brief Represents a variable binding list. Ex. `[x 1 y 2]`
 */
struct BindingVectorExpr final : ExprBase {
    ClobberToken open_bracket_token;
    std::vector<std::unique_ptr<IdentifierExpr>> identifiers;
    std::vector<std::unique_ptr<ExprBase>> exprs;
    ClobberToken close_bracket_token;
    size_t num_bindings;
};

/* @brief Represents a parameter vector list. Ex. `[x y]`
 */
struct ParameterVectorExpr final : ExprBase {
    ClobberToken open_bracket_token;
    std::vector<std::unique_ptr<IdentifierExpr>> identifiers;
    ClobberToken close_bracket_token;
};

/* @brief Represents a `let` expression. Ex. `(let [x 1 y 2] (x))`
 */
struct LetExpr final : ExprBase {
    ClobberToken open_paren_token;
    ClobberToken let_token;
    std::unique_ptr<BindingVectorExpr> binding_vector_expr;
    std::vector<std::unique_ptr<ExprBase>> body_exprs;
    ClobberToken close_paren_token;
};

/* @brief Represents a `fn` expression. Ex. `(fn [x y] (+ x y))`
 */
struct FnExpr final : ExprBase {
    ClobberToken open_paren_token;
    ClobberToken fn_token;
    std::unique_ptr<ParameterVectorExpr> parameter_vector_expr;
    std::vector<std::unique_ptr<ExprBase>> body_exprs;
    ClobberToken close_paren_token;
};

/* @brief Represents a `def` expression. Ex. `(def x 2)`
 */
struct DefExpr final : ExprBase {
    ClobberToken open_paren_token;
    ClobberToken def_token;
    std::unique_ptr<IdentifierExpr> identifier;
    std::unique_ptr<ExprBase> value;
    ClobberToken close_paren_token;
};

/* @brief Represents a `def` expression. Ex. `(do (def x 2)(+ x 2))`
 */
struct DoExpr final : ExprBase {
    ClobberToken open_paren_token;
    ClobberToken do_token;
    std::vector<std::unique_ptr<ExprBase>> body_exprs;
    ClobberToken close_paren_token;
};

/* @brief Enum class representing the type of operation/function being called in a `CallExpr`.
 */
enum class CallExprOperatorExprType {
    IdentifierExpr,
    AnonymousFunctionExpr,
};

/* @brief Represents a call expression.
 */
struct CallExpr final : ExprBase {
    CallExprOperatorExprType operator_expr_type;

    ClobberToken open_paren_token;
    ClobberToken operator_token;
    ClobberToken close_paren_token;
    std::vector<std::unique_ptr<ExprBase>> arguments;
};

// --- Accel specific AST nodes
// We don't define a separate AST for hardware accelerated syntax for ease on the parser side. This check is instead offloaded during
// semantic analysis.

/* @brief Represents a hardware accelerated code block.
 */
struct AccelExpr final : ExprBase {
    ClobberToken open_paren_token;
    ClobberToken accel_token;
    std::unique_ptr<BindingVectorExpr> binding_vector_expr;
    std::vector<std::unique_ptr<ExprBase>> body_exprs;
    ClobberToken close_paren_token;
};

/* @brief Represents a matrix multiply expression.
 */
struct MatMulExpr final : ExprBase {
    ClobberToken open_paren_token;
    ClobberToken mat_mul_token;
    std::unique_ptr<ExprBase> fst_operand;
    std::unique_ptr<ExprBase> snd_operand;
    ClobberToken close_paren_token;
};

/* @brief Represents a RelU expression.
 */
struct RelUExpr final : ExprBase {
    ClobberToken open_paren_token;
    ClobberToken relu_token;
    std::unique_ptr<ExprBase> operand;
    ClobberToken close_paren_token;
};

/* @brief Represents a clobber compilation unit. Usually contains all the contents of a source file.
 */
struct CompilationUnit {
    const std::string &source_text;
    std::vector<std::unique_ptr<ExprBase>> exprs;
    std::vector<ParserError> parse_errors;
};