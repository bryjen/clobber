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
            SlashToken,
            BackslashToken,
            EqualsToken,
            LessThanToken,
            GreaterThanToken,
            CaretToken,
            KeywordLiteralToken,
            QuoteToken,
            BacktickToken,
            TildeToken,
            TildeSpliceToken,
            DispatchHashToken,
            AtToken,
            AmpersandToken,
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
            CharKeywordToken,
            StringKeywordToken,
            VectorKeywordToken,
            I8KeywordToken,
            I16KeywordToken,
            I32KeywordToken,
            I64KeywordToken,
            F32KeywordToken,
            F64KeywordToken,
            AccelKeywordToken,
            TensorKeywordToken,
            ReshapeKeywordToken,
            TransposeKeywordToken,
            TileKeywordToken,
            SliceKeywordToken,
            ConcatKeywordToken,
            IdentityKeywordToken,
            CastKeywordToken,
            Conv2dKeywordToken,
            DepthwiseConv2dKeywordToken,
            MatmulKeywordToken,
            FullyConnectedKeywordToken,
            AvgPool2dKeywordToken,
            MaxPool2dKeywordToken,
            PadKeywordToken,
            ReluKeywordToken,
            SigmoidKeywordToken,
            TanhKeywordToken,
            SoftmaxKeywordToken,
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
        virtual Span span() const   = 0;
        virtual size_t hash() const = 0;
    };

    struct TypeExpr : Expr {
        enum class Type {
            BuiltinType,
            UserDefinedType,
            ParameterizedType
        };

        TypeExpr::Type type_kind;

    public:
        TypeExpr(TypeExpr::Type);
    };

    struct BuiltinTypeExpr final : TypeExpr {
        Token caret_token;
        Token type_keyword_token;

    public:
        BuiltinTypeExpr(const Token &, const Token &);
        Span span() const override;
        size_t hash() const override;
    };

    struct UserDefinedTypeExpr final : TypeExpr {
        Token caret_token;
        Token identifier_token;

    public:
        UserDefinedTypeExpr(const Token &, const Token &);
        Span span() const override;
        size_t hash() const override;
    };

    struct ParameterizedTypeExpr final : TypeExpr {
        std::unique_ptr<TypeExpr> type_expr;
        Token less_than_token;
        std::vector<std::unique_ptr<Expr>> param_values;
        std::vector<Token> commas;
        Token greater_than_token;

    public:
        ParameterizedTypeExpr(std::unique_ptr<TypeExpr>, const Token &, std::vector<std::unique_ptr<Expr>> &&, std::vector<Token>,
                              const Token &);
        Span span() const override;
        size_t hash() const override;
    };

    struct IdentifierExpr;

    struct ParenthesizedExpr : Expr {
        Token open_paren_token;
        Token close_paren_token;

    public:
        ParenthesizedExpr(Expr::Type, const Token &, const Token &);
        virtual Span span() const   = 0;
        virtual size_t hash() const = 0;
    };

    struct Binding final {
        std::unique_ptr<IdentifierExpr> identifier;
        std::unique_ptr<TypeExpr> type_annot;
        std::unique_ptr<Expr> value;

        std::unordered_map<std::string, std::any> metadata;

    public:
        Binding(std::unique_ptr<IdentifierExpr>, std::unique_ptr<TypeExpr>, std::unique_ptr<Expr>);
        Span span() const;
    };

    struct BindingVectorExpr final {
        Token open_bracket_token;
        std::vector<std::unique_ptr<Binding>> bindings;
        Token close_bracket_token;

        std::unordered_map<std::string, std::any> metadata;

    public:
        BindingVectorExpr(const Token &, std::vector<std::unique_ptr<Binding>> &&, const Token &);
        Span span() const;
    };

    struct Parameter final {
        std::unique_ptr<IdentifierExpr> identifier;
        std::unique_ptr<TypeExpr> type_annot;

        std::unordered_map<std::string, std::any> metadata;

    public:
        Parameter(std::unique_ptr<IdentifierExpr>, std::unique_ptr<TypeExpr>);
        Span span() const;
    };

    struct ParameterVectorExpr final {
        Token open_bracket_token;
        std::vector<std::unique_ptr<Parameter>> parameters;
        Token close_bracket_token;

        std::unordered_map<std::string, std::any> metadata;

    public:
        ParameterVectorExpr(const Token &, std::vector<std::unique_ptr<Parameter>> &&, const Token &);
        Span span() const;
    };

    struct VectorExpr final : Expr {
        Token open_bracket_token;
        std::vector<std::unique_ptr<Expr>> values;
        std::vector<Token> commas;
        Token close_bracket_token;

    public:
        VectorExpr(const Token &, std::vector<std::unique_ptr<Expr>> &&, std::vector<Token> &&, const Token &);
        Span span() const override;
        size_t hash() const override;
    };

    struct NumLiteralExpr final : Expr {
        Token token;

    public:
        NumLiteralExpr(const Token &);
        Span span() const override;
        size_t hash() const override;
    };

    struct StringLiteralExpr final : Expr {
        std::string value;
        Token token;

    public:
        StringLiteralExpr(const std::string &, const Token &);
        Span span() const override;
        size_t hash() const override;
    };

    struct CharLiteralExpr final : Expr {
        std::string value;
        Token token;

    public:
        CharLiteralExpr(const std::string &, const Token &);
        Span span() const override;
        size_t hash() const override;
    };

    struct KeywordLiteralExpr final : Expr {
        Token token;
        std::string name;

    public:
        KeywordLiteralExpr(const std::string &, const Token &);
        Span span() const override;
        size_t hash() const override;
    };

    struct IdentifierExpr final : Expr {
        std::string name;
        Token token;

    public:
        IdentifierExpr(const std::string &, const Token &);
        Span span() const override;
        size_t hash() const override;
    };

    struct LetExpr final : ParenthesizedExpr {
        Token let_token;
        std::unique_ptr<BindingVectorExpr> binding_vector_expr;
        std::vector<std::unique_ptr<Expr>> body_exprs;

    public:
        LetExpr(const Token &, const Token &, std::unique_ptr<BindingVectorExpr>, std::vector<std::unique_ptr<Expr>> &&, const Token &);
        Span span() const override;
        size_t hash() const override;
    };

    struct FnExpr final : ParenthesizedExpr {
        Token fn_token;
        std::unique_ptr<ParameterVectorExpr> parameter_vector_expr;
        std::vector<std::unique_ptr<Expr>> body_exprs;

    public:
        FnExpr(const Token &, const Token &, std::unique_ptr<ParameterVectorExpr>, std::vector<std::unique_ptr<Expr>> &&, const Token &);
        Span span() const override;
        size_t hash() const override;
    };

    struct DefExpr final : ParenthesizedExpr {
        Token def_token;
        std::unique_ptr<IdentifierExpr> identifier;
        std::unique_ptr<Expr> value;

    public:
        DefExpr(const Token &, const Token &, std::unique_ptr<IdentifierExpr>, std::unique_ptr<Expr>, const Token &);
        Span span() const override;
        size_t hash() const override;
    };

    struct DoExpr final : ParenthesizedExpr {
        Token do_token;
        std::vector<std::unique_ptr<Expr>> body_exprs;

    public:
        DoExpr(const Token &, const Token &, std::vector<std::unique_ptr<Expr>> &&, const Token &);
        Span span() const override;
        size_t hash() const override;
    };

    struct CallExpr final : ParenthesizedExpr {
        std::unique_ptr<Expr> operator_expr;
        std::vector<std::unique_ptr<Expr>> arguments;

    public:
        CallExpr(const Token &, std::unique_ptr<Expr> operator_expr, std::vector<std::unique_ptr<Expr>> &&, const Token &);
        Span span() const override;
        size_t hash() const override;
    };

    struct CompilationUnit {
        const std::string &source_text;
        const std::vector<std::unique_ptr<Expr>> exprs;
        const std::vector<clobber::Diagnostic> &diagnostics;

    public:
        CompilationUnit(const std::string &, std::vector<std::unique_ptr<Expr>> &&, const std::vector<clobber::Diagnostic> &);
        size_t hash() const;
    };
}; // namespace clobber