#pragma once

#include <clobber/common/debug.hpp>

#include "clobber/pch.hpp"

// fwd decs
namespace clobber {
    struct Diagnostic; // clobber/common/diagnostic.hpp
};

namespace clobber {
    struct Span final {
        size_t start  = 0;
        size_t length = 0;
    };

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
            SlashToken,     // /
            BackslashToken, // '\'
            EqualsToken,
            LessThanToken,       // <
            GreaterThanToken,    // >
            CaretToken,          // '^'
            KeywordLiteralToken, // :<LITERAL>

            // macro stuff
            QuoteToken,        // '
            BacktickToken,     // `
            TildeToken,        // ~
            TildeSpliceToken,  // ~@
            DispatchHashToken, // #
            AtToken,           // @
            AmpersandToken,    // &
            CommaToken,

            IdentifierToken,
            NumericLiteralToken,
            StringLiteralToken,
            CharLiteralToken,

            NsKeywordToken,
            IfKeywordToken,
            LetKeywordToken,
            FnKeywordToken,
            DefKeywordToken,
            DefMacroKeywordToken,
            DoKeywordToken,

            // type keywords
            CharKeywordToken,   // "string"
            StringKeywordToken, // "string"
            VectorKeywordToken, // "vector"
            I8KeywordToken,     // "i8"
            I16KeywordToken,    // "i16"
            I32KeywordToken,    // "i32"
            I64KeywordToken,    // "i64"
            F32KeywordToken,    // "f32"
            F64KeywordToken,    // "f64"

            // hardware acceleration tokens
            AccelKeywordToken,  // `accel`
            TensorKeywordToken, // `tensor`

            ReshapeKeywordToken,   // `reshape`
            TransposeKeywordToken, // `transpose`
            TileKeywordToken,      // `tile`
            SliceKeywordToken,     // `slice`
            ConcatKeywordToken,    // `concat`
            IdentityKeywordToken,  // `identity`
            CastKeywordToken,      // `cast`

            Conv2dKeywordToken,          // `conv2d`
            DepthwiseConv2dKeywordToken, // `depthwise-conv2d`
            MatmulKeywordToken,          // `matmul`
            FullyConnectedKeywordToken,  // `fully-connected`
            AvgPool2dKeywordToken,       // `avgpool2d`
            MaxPool2dKeywordToken,       // `maxpool2d`
            PadKeywordToken,             // `pad`

            ReluKeywordToken,    // `relu`
            SigmoidKeywordToken, // `sigmoid`
            TanhKeywordToken,    // `tanh`
            SoftmaxKeywordToken, // `softmax`

            BadToken,
            EofToken,
        };

        Span span;
        Span full_span;

        Token::Type type;
        std::unordered_map<std::string, std::any> metadata;

    public:
        std::string ExtractText(const std::string &) const;
        std::string ExtractFullText(const std::string &) const;

        size_t hash() const;
        static bool AreEquivalent(const Token &, const Token &);
    };

    /* @brief Represents a clobber expression. Base class for all expression types. */
    struct Expr {
        enum class Type {
            NumericLiteralExpr,
            StringLiteralExpr,
            CharLiteralExpr,
            VectorExpr,
            TensorExpr,

            KeywordLiteralExpr,
            IdentifierExpr,

            TypeExpr,

            CallExpr,
            LetExpr,
            FnExpr,
            DefExpr,
            DoExpr,

            AccelExpr,
            TosaOpExpr,
        };

        Expr::Type type;
        std::unordered_map<std::string, std::any> metadata;

    public:
        Expr(Expr::Type);

        virtual Span span() const                   = 0;
        virtual size_t hash() const                 = 0;
        virtual std::unique_ptr<Expr> clone() const = 0;
    };

    /* @brief */
    struct TypeExpr : Expr {
        enum class Type {
            BuiltinType,
            UserDefinedType,
            ParameterizedType
        };

        TypeExpr::Type type_kind;

    public:
        TypeExpr(TypeExpr::Type);
        TypeExpr(const TypeExpr &);

        virtual Span span() const                              = 0;
        virtual size_t hash() const                            = 0;
        virtual std::unique_ptr<Expr> clone() const            = 0;
        virtual std::unique_ptr<TypeExpr> clone_nowrap() const = 0;
    };

    /* @brief */
    struct BuiltinTypeExpr final : TypeExpr {
        Token caret_token;
        Token type_keyword_token;

    public:
        BuiltinTypeExpr(const Token &, const Token &);
        BuiltinTypeExpr(const BuiltinTypeExpr &);

        Span span() const override;
        size_t hash() const override;
        std::unique_ptr<Expr> clone() const override;
        std::unique_ptr<TypeExpr> clone_nowrap() const override;
    };

    /* @brief */
    struct UserDefinedTypeExpr final : TypeExpr {
        Token caret_token;
        Token identifier_token;

    public:
        UserDefinedTypeExpr(const Token &, const Token &);
        UserDefinedTypeExpr(const UserDefinedTypeExpr &);

        Span span() const override;
        size_t hash() const override;
        std::unique_ptr<Expr> clone() const override;
        std::unique_ptr<TypeExpr> clone_nowrap() const override;
    };

    /* @brief */
    struct ParameterizedTypeExpr final : TypeExpr { // 'composite' type, so it shouldn't include caret token
        std::unique_ptr<TypeExpr> type_expr;
        Token less_than_token;
        std::vector<std::unique_ptr<Expr>> param_values;
        std::vector<Token> commas;
        Token greater_than_token;

    public:
        ParameterizedTypeExpr(std::unique_ptr<TypeExpr>, const Token &, std::vector<std::unique_ptr<Expr>> &&, std::vector<Token>,
                              const Token &);
        ParameterizedTypeExpr(const ParameterizedTypeExpr &);

        Span span() const override;
        size_t hash() const override;
        std::unique_ptr<Expr> clone() const override;
        std::unique_ptr<TypeExpr> clone_nowrap() const override;
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

        virtual Span span() const                   = 0;
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

        std::unordered_map<std::string, std::any> metadata;

    public:
        BindingVectorExpr(const Token &, std::vector<std::unique_ptr<IdentifierExpr>> &&, std::vector<std::unique_ptr<Expr>> &&,
                          const Token &, size_t);
        BindingVectorExpr(const BindingVectorExpr &);

        Span span() const;
        std::unique_ptr<BindingVectorExpr> clone_nowrap() const;
    };

    /* @brief Represents a parameter vector list. Ex. `[x y]` */
    struct ParameterVectorExpr final {
        Token open_bracket_token;
        std::vector<std::unique_ptr<IdentifierExpr>> identifiers;
        Token close_bracket_token;

        std::unordered_map<std::string, std::any> metadata;

    public:
        ParameterVectorExpr(const Token &, std::vector<std::unique_ptr<IdentifierExpr>> &&, const Token &);
        ParameterVectorExpr(const ParameterVectorExpr &);

        Span span() const;
        std::unique_ptr<ParameterVectorExpr> clone_nowrap() const;
    };

    struct VectorExpr final : Expr {
        Token open_bracket_token;
        std::vector<std::unique_ptr<Expr>> values;
        std::vector<Token> commas; // optional
        Token close_bracket_token;

    public:
        VectorExpr(const Token &, std::vector<std::unique_ptr<Expr>> &&, std::vector<Token> &&, const Token &);
        VectorExpr(const VectorExpr &);

        Span span() const override;
        size_t hash() const override;
        std::unique_ptr<Expr> clone() const override;
    };

    /* @brief Represents a numerical literal expression. */
    struct NumLiteralExpr final : Expr {
        Token token;

    public:
        NumLiteralExpr(const Token &);
        NumLiteralExpr(const NumLiteralExpr &);

        Span span() const override;
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

        Span span() const override;
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

        Span span() const override;
        size_t hash() const override;
        std::unique_ptr<Expr> clone() const override;
    };

    /* @brief Represents a keyword literal, for example `:param_name`. */
    struct KeywordLiteralExpr final : Expr {
        Token token;
        std::string name; // without colon

    public:
        KeywordLiteralExpr(const std::string &, const Token &);
        KeywordLiteralExpr(const KeywordLiteralExpr &);

        std::unique_ptr<KeywordLiteralExpr> clone_nowrap();

        Span span() const override;
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

        Span span() const override;
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

        Span span() const override;
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

        Span span() const override;
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

        Span span() const override;
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

        Span span() const override;
        size_t hash() const override;
        std::unique_ptr<Expr> clone() const override;
    };

    /* @brief Represents a call expression. */
    struct CallExpr final : ParenthesizedExpr {
        std::unique_ptr<Expr> operator_expr;
        std::vector<std::unique_ptr<Expr>> arguments;

    public:
        CallExpr(const Token &, std::unique_ptr<Expr> operator_expr, std::vector<std::unique_ptr<Expr>> &&, const Token &);
        CallExpr(const CallExpr &);

        Span span() const override;
        size_t hash() const override;
        std::unique_ptr<Expr> clone() const override;
    };

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
}; // namespace clobber