#pragma once

#include <clobber/common/debug.hpp> // common debug header

#include "pch.hpp"

namespace clobber {

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

    // remarks: cheap to copy
    /* @brief Represents a token. */
    struct ClobberToken final {
        size_t start;
        size_t length;
        size_t full_start;  // includes trivia
        size_t full_length; // includes trivia
        ClobberTokenType token_type;

    public:
        std::string ExtractText(const std::string &);
        std::string ExtractFullText(const std::string &);

        size_t hash() const;
        static bool AreEquivalent(const ClobberToken &, const ClobberToken &);
    };

    /* @brief Represents a clobber expression. Base class for all expression types. */
    struct Expr {
        enum class Type {
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

        Type type;

    public:
        Expr(Expr::Type);

        virtual size_t hash() const                 = 0;
        virtual std::unique_ptr<Expr> clone() const = 0;
    };

    struct IdentifierExpr; // fwd dec

    // Helps us to deal with the redundancies of declaring open and close parentheses for types.
    /* @brief Represents a parenthesized clobber expression. */
    struct ParenthesizedExpr : Expr {
        ClobberToken open_paren_token;
        ClobberToken close_paren_token;

    public:
        ParenthesizedExpr(Expr::Type, const ClobberToken &, const ClobberToken &);
        ParenthesizedExpr(const ParenthesizedExpr &);

        virtual size_t hash() const                 = 0;
        virtual std::unique_ptr<Expr> clone() const = 0;
        // virtual ~ParenthesizedExpr()                             = default;
    };

    /* @brief Represents a variable binding list. Ex. `[x 1 y 2]` */
    struct BindingVectorExpr final {
        ClobberToken open_bracket_token;
        std::vector<std::unique_ptr<IdentifierExpr>> identifiers;
        std::vector<std::unique_ptr<Expr>> exprs;
        ClobberToken close_bracket_token;
        size_t num_bindings;

    public:
        BindingVectorExpr(const ClobberToken &, std::vector<std::unique_ptr<IdentifierExpr>> &&, std::vector<std::unique_ptr<Expr>> &&,
                          const ClobberToken &, size_t);
        BindingVectorExpr(const BindingVectorExpr &);

        std::unique_ptr<BindingVectorExpr> clone_nowrap() const;
    };

    /* @brief Represents a parameter vector list. Ex. `[x y]` */
    struct ParameterVectorExpr final {
        ClobberToken open_bracket_token;
        std::vector<std::unique_ptr<IdentifierExpr>> identifiers;
        ClobberToken close_bracket_token;

    public:
        ParameterVectorExpr(const ClobberToken &, std::vector<std::unique_ptr<IdentifierExpr>> &&, const ClobberToken &);
        ParameterVectorExpr(const ParameterVectorExpr &);

        std::unique_ptr<ParameterVectorExpr> clone_nowrap() const;
    };

    /* @brief Represents a numerical literal expression. */
    struct NumLiteralExpr final : Expr {
        ClobberToken token;

    public:
        NumLiteralExpr(const ClobberToken &);
        NumLiteralExpr(const NumLiteralExpr &);

        size_t hash() const override;
        std::unique_ptr<Expr> clone() const override;
    };

    /* @brief Represents a string literal expression. */
    struct StringLiteralExpr final : Expr {
        std::string value;
        ClobberToken token;

    public:
        StringLiteralExpr(const std::string &, const ClobberToken &);
        StringLiteralExpr(const StringLiteralExpr &);

        size_t hash() const override;
        std::unique_ptr<Expr> clone() const override;
    };

    /* @brief Represents a char literal expression.
     * @remarks In the tokenizer, '...' can contain multiple characters. Although this can be asserted in the parser, we delegate this
     * responsibility to the semantic analyzer. The value does not contain the single quotes.
     */
    struct CharLiteralExpr final : Expr {
        std::string value;
        ClobberToken token;

    public:
        CharLiteralExpr(const std::string &, const ClobberToken &);
        CharLiteralExpr(const CharLiteralExpr &);

        size_t hash() const override;
        std::unique_ptr<Expr> clone() const override;
    };

    /* @brief Represents an identifier. */
    struct IdentifierExpr final : Expr {
        std::string name;
        ClobberToken token;

    public:
        IdentifierExpr(const std::string &, const ClobberToken &);
        IdentifierExpr(const IdentifierExpr &);

        std::unique_ptr<IdentifierExpr> clone_nowrap();

        size_t hash() const override;
        std::unique_ptr<Expr> clone() const override;
    };

    /* @brief Represents a `let` expression. Ex. `(let [x 1 y 2] (x))` */
    struct LetExpr final : ParenthesizedExpr {
        ClobberToken let_token;
        std::unique_ptr<BindingVectorExpr> binding_vector_expr;
        std::vector<std::unique_ptr<Expr>> body_exprs;

    public:
        LetExpr(const ClobberToken &, const ClobberToken &, std::unique_ptr<BindingVectorExpr>, std::vector<std::unique_ptr<Expr>> &&,
                const ClobberToken &);
        LetExpr(const LetExpr &);

        size_t hash() const override;
        std::unique_ptr<Expr> clone() const override;
    };

    /* @brief Represents a `fn` expression. Ex. `(fn [x y] (+ x y))` */
    struct FnExpr final : ParenthesizedExpr {
        ClobberToken fn_token;
        std::unique_ptr<ParameterVectorExpr> parameter_vector_expr;
        std::vector<std::unique_ptr<Expr>> body_exprs;

    public:
        FnExpr(const ClobberToken &, const ClobberToken &, std::unique_ptr<ParameterVectorExpr>, std::vector<std::unique_ptr<Expr>> &&,
               const ClobberToken &);
        FnExpr(const FnExpr &);

        size_t hash() const override;
        std::unique_ptr<Expr> clone() const override;
    };

    /* @brief Represents a `def` expression. Ex. `(def x 2)` */
    struct DefExpr final : ParenthesizedExpr {
        ClobberToken def_token;
        std::unique_ptr<IdentifierExpr> identifier;
        std::unique_ptr<Expr> value;

    public:
        DefExpr(const ClobberToken &, const ClobberToken &, std::unique_ptr<IdentifierExpr>, std::unique_ptr<Expr>, const ClobberToken &);
        DefExpr(const DefExpr &);

        size_t hash() const override;
        std::unique_ptr<Expr> clone() const override;
    };

    /* @brief Represents a `def` expression. Ex. `(do (def x 2)(+ x 2))` */
    struct DoExpr final : ParenthesizedExpr {
        ClobberToken do_token;
        std::vector<std::unique_ptr<Expr>> body_exprs;

    public:
        DoExpr(const ClobberToken &, const ClobberToken &, std::vector<std::unique_ptr<Expr>> &&, const ClobberToken &);
        DoExpr(const DoExpr &);

        size_t hash() const override;
        std::unique_ptr<Expr> clone() const override;
    };

    /* @brief Enum class representing the type of operation/function being called in a `CallExpr`. */
    enum class CallExprOperatorExprType {
        IdentifierExpr,
        AnonymousFunctionExpr,
    };

    /* @brief Represents a call expression. */
    struct CallExpr final : ParenthesizedExpr {
        CallExprOperatorExprType operator_expr_type;

        ClobberToken operator_token;
        std::vector<std::unique_ptr<Expr>> arguments;

    public:
        CallExpr(CallExprOperatorExprType, const ClobberToken &, const ClobberToken &, const ClobberToken &,
                 std::vector<std::unique_ptr<Expr>> &&);
        CallExpr(const CallExpr &);

        size_t hash() const override;
        std::unique_ptr<Expr> clone() const override;
    };

    namespace accel {
        // --- Accel specific AST nodes
        // We don't define a separate AST for hardware accelerated syntax for ease on the parser side. This check is instead offloaded
        // during semantic analysis.

        /* @brief Represents a hardware accelerated code block. */
        struct AccelExpr final : ParenthesizedExpr {
            ClobberToken accel_token;
            std::unique_ptr<BindingVectorExpr> binding_vector_expr;
            std::vector<std::unique_ptr<Expr>> body_exprs;

        public:
            AccelExpr(const ClobberToken &, const ClobberToken &, std::unique_ptr<BindingVectorExpr>, std::vector<std::unique_ptr<Expr>> &&,
                      const ClobberToken &);
            AccelExpr(const AccelExpr &);

            size_t hash() const override;
            std::unique_ptr<Expr> clone() const override;
        };

        /* @brief Represents a matrix multiply expression. */
        struct MatMulExpr final : ParenthesizedExpr {
            ClobberToken mat_mul_token;
            std::unique_ptr<Expr> fst_operand;
            std::unique_ptr<Expr> snd_operand;

        public:
            MatMulExpr(const ClobberToken &, const ClobberToken &, std::unique_ptr<Expr>, std::unique_ptr<Expr>, const ClobberToken &);
            MatMulExpr(const MatMulExpr &);

            size_t hash() const override;
            std::unique_ptr<Expr> clone() const override;
        };

        /* @brief Represents a RelU expression. */
        struct RelUExpr final : ParenthesizedExpr {
            ClobberToken relu_token;
            std::unique_ptr<Expr> operand;

        public:
            RelUExpr(const ClobberToken &, const ClobberToken &, std::unique_ptr<Expr>, const ClobberToken &);
            RelUExpr(const RelUExpr &);

            size_t hash() const override;
            std::unique_ptr<Expr> clone() const override;
        };
    }; // namespace accel

    /* @brief Represents a clobber compilation unit. Usually contains all the contents of a source file. */
    struct CompilationUnit {
        std::string source_text;
        std::vector<std::unique_ptr<Expr>> exprs;
        // std::vector<clobber::ParserError> parse_errors;

    public:
        CompilationUnit(const std::string &, std::vector<std::unique_ptr<Expr>> &&);
        CompilationUnit(const CompilationUnit &);

        size_t hash() const;
        std::unique_ptr<CompilationUnit> clone() const;
    };

}; // namespace clobber