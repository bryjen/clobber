#pragma once

#include <clobber/common/debug.hpp> // common debug header

#include "pch.hpp"

namespace clobber {
    struct Diagnostic; // clobber/common/diagnostic.hpp
};

namespace clobber {

    // remarks: cheap to copy
    /* @brief Represents a token. */
    struct Token final {
        enum class Type {
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

        size_t start;
        size_t length;
        size_t full_start;  // includes trivia
        size_t full_length; // includes trivia
        Token::Type type;

    public:
        std::string ExtractText(const std::string &) const;
        std::string ExtractFullText(const std::string &) const;

        size_t hash() const;
        static bool AreEquivalent(const Token &, const Token &);
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

        Expr::Type type;

    public:
        Expr(Expr::Type);

        virtual size_t hash() const                 = 0;
        virtual std::unique_ptr<Expr> clone() const = 0;
    };

    struct IdentifierExpr; // fwd dec

    // Helps us to deal with the redundancies of declaring open and close parentheses for types.
    /* @brief Represents a parenthesized clobber expression. */
    struct ParenthesizedExpr : Expr {
        Token open_paren_token;
        Token close_paren_token;

    public:
        ParenthesizedExpr(Expr::Type, const Token &, const Token &);
        ParenthesizedExpr(const ParenthesizedExpr &);

        virtual size_t hash() const                 = 0;
        virtual std::unique_ptr<Expr> clone() const = 0;
        // virtual ~ParenthesizedExpr()                             = default;
    };

    /* @brief Represents a variable binding list. Ex. `[x 1 y 2]` */
    struct BindingVectorExpr final {
        Token open_bracket_token;
        std::vector<std::unique_ptr<IdentifierExpr>> identifiers;
        std::vector<std::unique_ptr<Expr>> exprs;
        Token close_bracket_token;
        size_t num_bindings;

    public:
        BindingVectorExpr(const Token &, std::vector<std::unique_ptr<IdentifierExpr>> &&, std::vector<std::unique_ptr<Expr>> &&,
                          const Token &, size_t);
        BindingVectorExpr(const BindingVectorExpr &);

        std::unique_ptr<BindingVectorExpr> clone_nowrap() const;
    };

    /* @brief Represents a parameter vector list. Ex. `[x y]` */
    struct ParameterVectorExpr final {
        Token open_bracket_token;
        std::vector<std::unique_ptr<IdentifierExpr>> identifiers;
        Token close_bracket_token;

    public:
        ParameterVectorExpr(const Token &, std::vector<std::unique_ptr<IdentifierExpr>> &&, const Token &);
        ParameterVectorExpr(const ParameterVectorExpr &);

        std::unique_ptr<ParameterVectorExpr> clone_nowrap() const;
    };

    /* @brief Represents a numerical literal expression. */
    struct NumLiteralExpr final : Expr {
        Token token;

    public:
        NumLiteralExpr(const Token &);
        NumLiteralExpr(const NumLiteralExpr &);

        size_t hash() const override;
        std::unique_ptr<Expr> clone() const override;
    };

    /* @brief Represents a string literal expression. */
    struct StringLiteralExpr final : Expr {
        std::string value;
        Token token;

    public:
        StringLiteralExpr(const std::string &, const Token &);
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
        Token token;

    public:
        CharLiteralExpr(const std::string &, const Token &);
        CharLiteralExpr(const CharLiteralExpr &);

        size_t hash() const override;
        std::unique_ptr<Expr> clone() const override;
    };

    /* @brief Represents an identifier. */
    struct IdentifierExpr final : Expr {
        std::string name;
        Token token;

    public:
        IdentifierExpr(const std::string &, const Token &);
        IdentifierExpr(const IdentifierExpr &);

        std::unique_ptr<IdentifierExpr> clone_nowrap();

        size_t hash() const override;
        std::unique_ptr<Expr> clone() const override;
    };

    /* @brief Represents a `let` expression. Ex. `(let [x 1 y 2] (x))` */
    struct LetExpr final : ParenthesizedExpr {
        Token let_token;
        std::unique_ptr<BindingVectorExpr> binding_vector_expr;
        std::vector<std::unique_ptr<Expr>> body_exprs;

    public:
        LetExpr(const Token &, const Token &, std::unique_ptr<BindingVectorExpr>, std::vector<std::unique_ptr<Expr>> &&, const Token &);
        LetExpr(const LetExpr &);

        size_t hash() const override;
        std::unique_ptr<Expr> clone() const override;
    };

    /* @brief Represents a `fn` expression. Ex. `(fn [x y] (+ x y))` */
    struct FnExpr final : ParenthesizedExpr {
        Token fn_token;
        std::unique_ptr<ParameterVectorExpr> parameter_vector_expr;
        std::vector<std::unique_ptr<Expr>> body_exprs;

    public:
        FnExpr(const Token &, const Token &, std::unique_ptr<ParameterVectorExpr>, std::vector<std::unique_ptr<Expr>> &&, const Token &);
        FnExpr(const FnExpr &);

        size_t hash() const override;
        std::unique_ptr<Expr> clone() const override;
    };

    /* @brief Represents a `def` expression. Ex. `(def x 2)` */
    struct DefExpr final : ParenthesizedExpr {
        Token def_token;
        std::unique_ptr<IdentifierExpr> identifier;
        std::unique_ptr<Expr> value;

    public:
        DefExpr(const Token &, const Token &, std::unique_ptr<IdentifierExpr>, std::unique_ptr<Expr>, const Token &);
        DefExpr(const DefExpr &);

        size_t hash() const override;
        std::unique_ptr<Expr> clone() const override;
    };

    /* @brief Represents a `def` expression. Ex. `(do (def x 2)(+ x 2))` */
    struct DoExpr final : ParenthesizedExpr {
        Token do_token;
        std::vector<std::unique_ptr<Expr>> body_exprs;

    public:
        DoExpr(const Token &, const Token &, std::vector<std::unique_ptr<Expr>> &&, const Token &);
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

        Token operator_token;
        std::vector<std::unique_ptr<Expr>> arguments;

    public:
        CallExpr(CallExprOperatorExprType, const Token &, const Token &, const Token &, std::vector<std::unique_ptr<Expr>> &&);
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
            Token accel_token;
            std::unique_ptr<BindingVectorExpr> binding_vector_expr;
            std::vector<std::unique_ptr<Expr>> body_exprs;

        public:
            AccelExpr(const Token &, const Token &, std::unique_ptr<BindingVectorExpr>, std::vector<std::unique_ptr<Expr>> &&,
                      const Token &);
            AccelExpr(const AccelExpr &);

            size_t hash() const override;
            std::unique_ptr<Expr> clone() const override;
        };

        /* @brief Represents a matrix multiply expression. */
        struct MatMulExpr final : ParenthesizedExpr {
            Token mat_mul_token;
            std::unique_ptr<Expr> fst_operand;
            std::unique_ptr<Expr> snd_operand;

        public:
            MatMulExpr(const Token &, const Token &, std::unique_ptr<Expr>, std::unique_ptr<Expr>, const Token &);
            MatMulExpr(const MatMulExpr &);

            size_t hash() const override;
            std::unique_ptr<Expr> clone() const override;
        };

        /* @brief Represents a RelU expression. */
        struct RelUExpr final : ParenthesizedExpr {
            Token relu_token;
            std::unique_ptr<Expr> operand;

        public:
            RelUExpr(const Token &, const Token &, std::unique_ptr<Expr>, const Token &);
            RelUExpr(const RelUExpr &);

            size_t hash() const override;
            std::unique_ptr<Expr> clone() const override;
        };
    }; // namespace accel

    /* @brief Represents a clobber compilation unit. Usually contains all the contents of a source file. */
    struct CompilationUnit {
        const std::string &source_text;
        const std::vector<std::unique_ptr<Expr>> exprs;
        const std::vector<clobber::Diagnostic> &diagnostics;

    public:
        CompilationUnit(const std::string &, std::vector<std::unique_ptr<Expr>> &&, const std::vector<clobber::Diagnostic> &);
        CompilationUnit(const CompilationUnit &);

        size_t hash() const;
        std::unique_ptr<CompilationUnit> clone() const;
    };

    class AstWalker {
    protected:
        virtual Expr *on_expr(Expr *);

        // 'leaf' exprs
        virtual NumLiteralExpr *on_num_literal_expr(NumLiteralExpr *);
        virtual StringLiteralExpr *on_string_literal_expr(StringLiteralExpr *);
        virtual CharLiteralExpr *on_char_literal_expr(CharLiteralExpr *);
        virtual IdentifierExpr *on_identifier_expr(IdentifierExpr *);

        // 'node' exprs
        virtual ParenthesizedExpr *on_paren_expr(ParenthesizedExpr *);
        virtual BindingVectorExpr *on_binding_vector_expr(BindingVectorExpr *);
        virtual ParameterVectorExpr *on_parameter_vector_expr(ParameterVectorExpr *);
        virtual LetExpr *on_let_expr(LetExpr *);
        virtual FnExpr *on_fn_expr(FnExpr *);
        virtual DefExpr *on_def_expr(DefExpr *);
        virtual DoExpr *on_do_expr(DoExpr *);
        virtual CallExpr *on_call_expr(CallExpr *);

        // hardware acceleration exprs
        virtual accel::AccelExpr *on_accel_expr(accel::AccelExpr *);
        virtual accel::MatMulExpr *on_mat_mul_expr(accel::MatMulExpr *);
        virtual accel::RelUExpr *on_relu_expr(accel::RelUExpr *);
    };

}; // namespace clobber