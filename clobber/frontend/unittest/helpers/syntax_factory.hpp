#pragma once

#include "clobber/ast/ast.hpp"

// defining annotations to help remember which syntax factory functions are meant to be used where.
// using custom annotations instead of comments because they stand out more 🔥.
#if defined(__clang__) || defined(__GNUC__)
#define USAGE_NOTE(msg) __attribute__((annotate(msg)))
#else
#define USAGE_NOTE(msg)
#endif

const std::string default_str_metadata_tag = "expected_repr";

#define DEFINE_TOKEN_FUNC(NAME, VALUE)                                                                                                     \
    inline clobber::Token NAME() {                                                                                                         \
        clobber::Token token{};                                                                                                            \
        token.type                               = clobber::Token::Type::NAME##Token;                                                      \
        token.metadata[default_str_metadata_tag] = std::string(VALUE);                                                                     \
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
    DEFINE_TOKEN_FUNC(Equals, "=")
    DEFINE_TOKEN_FUNC(LessThan, "<")
    DEFINE_TOKEN_FUNC(GreaterThan, ">")

    DEFINE_TOKEN_FUNC(Quote, "'")
    DEFINE_TOKEN_FUNC(Backtick, "`")
    DEFINE_TOKEN_FUNC(Tilde, "~")
    DEFINE_TOKEN_FUNC(TildeSplice, "~@")
    DEFINE_TOKEN_FUNC(DispatchHash, "#")
    DEFINE_TOKEN_FUNC(At, "@")
    DEFINE_TOKEN_FUNC(Ampersand, "&")
    DEFINE_TOKEN_FUNC(Caret, "^")

    DEFINE_TOKEN_FUNC(Comma, ",")
    DEFINE_TOKEN_FUNC(Eof, "")

    DEFINE_TOKEN_FUNC(NsKeyword, "ns")
    DEFINE_TOKEN_FUNC(IfKeyword, "if")
    DEFINE_TOKEN_FUNC(LetKeyword, "let")
    DEFINE_TOKEN_FUNC(FnKeyword, "fn")
    DEFINE_TOKEN_FUNC(DefKeyword, "def")
    DEFINE_TOKEN_FUNC(DefMacroKeyword, "defmacro")
    DEFINE_TOKEN_FUNC(DoKeyword, "do")

    // type keywords
    DEFINE_TOKEN_FUNC(CharKeyword, "char")
    DEFINE_TOKEN_FUNC(StringKeyword, "string")
    DEFINE_TOKEN_FUNC(VectorKeyword, "vector")
    DEFINE_TOKEN_FUNC(I8Keyword, "i8")
    DEFINE_TOKEN_FUNC(I16Keyword, "i16")
    DEFINE_TOKEN_FUNC(I32Keyword, "i32")
    DEFINE_TOKEN_FUNC(I64Keyword, "i64")
    DEFINE_TOKEN_FUNC(F32Keyword, "f32")
    DEFINE_TOKEN_FUNC(F64Keyword, "f64")

    // hardware acceleration tokens
    DEFINE_TOKEN_FUNC(AccelKeyword, "accel")
    DEFINE_TOKEN_FUNC(TensorKeyword, "tensor")
    DEFINE_TOKEN_FUNC(ReshapeKeyword, "reshape")
    DEFINE_TOKEN_FUNC(TransposeKeyword, "transpose")
    DEFINE_TOKEN_FUNC(TileKeyword, "tile")
    DEFINE_TOKEN_FUNC(SliceKeyword, "slice")
    DEFINE_TOKEN_FUNC(ConcatKeyword, "concat")
    DEFINE_TOKEN_FUNC(IdentityKeyword, "identity")
    DEFINE_TOKEN_FUNC(CastKeyword, "cast")
    DEFINE_TOKEN_FUNC(Conv2dKeyword, "conv2d")
    DEFINE_TOKEN_FUNC(DepthwiseConv2dKeyword, "depthwise-conv2d")
    DEFINE_TOKEN_FUNC(MatmulKeyword, "matmul")
    DEFINE_TOKEN_FUNC(FullyConnectedKeyword, "fully-connected")
    DEFINE_TOKEN_FUNC(AvgPool2dKeyword, "avg-pool2d")
    DEFINE_TOKEN_FUNC(MaxPool2dKeyword, "max-pool2d")
    DEFINE_TOKEN_FUNC(PadKeyword, "pad")
    DEFINE_TOKEN_FUNC(ReluKeyword, "relu")
    DEFINE_TOKEN_FUNC(SigmoidKeyword, "sigmoid")
    DEFINE_TOKEN_FUNC(TanhKeyword, "tanh")
    DEFINE_TOKEN_FUNC(SoftmaxKeyword, "softmax")

#pragma region tokenizer_legacy_only
    USAGE_NOTE("tokenizer only")
    inline clobber::Token
    CharLiteral(char c) {
        clobber::Token token{};
        token.type = clobber::Token::Type::CharLiteralToken;
        return token;
    }

    USAGE_NOTE("tokenizer only")
    inline clobber::Token
    NumericLiteral(int value) {
        clobber::Token token{};
        token.type = clobber::Token::Type::NumericLiteralToken;
        return token;
    }

    USAGE_NOTE("tokenizer only")
    inline clobber::Token
    NumericLiteral(float value, int decimal_places = 2) {
        clobber::Token token{};
        token.type = clobber::Token::Type::NumericLiteralToken;
        return token;
    }

    USAGE_NOTE("tokenizer only")
    inline clobber::Token
    NumericLiteral(double value, int decimal_places = 2, bool postfix_d = false) {
        clobber::Token token{};
        token.type = clobber::Token::Type::NumericLiteralToken;
        return token;
    }
#pragma endregion

    /* @brief Constructs a string literal token, inserts the double quotes into the value provided. */
    inline clobber::Token
    StringLiteralInsertDoubleQuot(std::string value) {
        clobber::Token token{};
        token.type                               = clobber::Token::Type::StringLiteralToken;
        token.metadata[default_str_metadata_tag] = std::string(std::format("\"{}\"", value));
        return token;
    }

    inline clobber::Token
    CharLiteral(const std::string &src) {
        clobber::Token token{};
        token.type                               = clobber::Token::Type::CharLiteralToken;
        token.metadata[default_str_metadata_tag] = std::string(src);
        return token;
    }

    inline clobber::Token
    Identifier(const std::string &name) {
        clobber::Token token{};
        token.type                               = clobber::Token::Type::IdentifierToken;
        token.metadata[default_str_metadata_tag] = std::string(name);
        return token;
    }

    inline clobber::Token
    KeywordLiteralInsertColon(const std::string &name) {
        clobber::Token token{};
        token.type                               = clobber::Token::Type::KeywordLiteralToken;
        token.metadata[default_str_metadata_tag] = std::string(name);
        return token;
    }

    /* @remarks String parameter represents what a numeric literal expression references in source code.
                Useful for avoiding inaccuracies when comparing floating point numbers. */
    inline clobber::Token
    NumericLiteral(const std::string &string_repr) {
        clobber::Token token{};
        token.type                               = clobber::Token::Type::NumericLiteralToken;
        token.metadata[default_str_metadata_tag] = std::string(string_repr);
        return token;
    }

    inline clobber::Token
    BadToken() {
        clobber::Token token{};
        token.type = clobber::Token::Type::BadToken;
        return token;
    }

    inline clobber::Token
    KeywordLiteral(const std::string &literal) {
        clobber::Token token{};
        token.type = clobber::Token::Type::KeywordLiteralToken;
        return token;
    }

    clobber::ParameterVectorExpr *ParameterVectorExpr(std::vector<clobber::IdentifierExpr *> identifiers);
    clobber::BindingVectorExpr *BindingVectorExpr(std::vector<clobber::IdentifierExpr *> identifiers, std::vector<clobber::Expr *> values);

    clobber::BuiltinTypeExpr *BuiltinTypeExpr(const clobber::Token &type_token);
    clobber::UserDefinedTypeExpr *UserDefinedTypeExpr(const clobber::Token &identifier_token);
    clobber::ParameterizedTypeExpr *ParameterizedTypeExpr(clobber::TypeExpr *type_expr, std::vector<clobber::Expr *> param_values);

    clobber::IdentifierExpr *IdentifierExpr(const std::string &name);
    clobber::NumLiteralExpr *NumLiteralExpr(const std::string &);
    clobber::NumLiteralExpr *NumLiteralExpr(int value);
    clobber::NumLiteralExpr *NumLiteralExpr(float value);
    clobber::NumLiteralExpr *NumLiteralExpr(double value);
    clobber::StringLiteralExpr *StringLiteralExpr(const std::string &value);
    clobber::CharLiteralExpr *CharLiteralExpr(char value);
    clobber::KeywordLiteralExpr *KeywordLiteralExpr(const std::string &value);
    clobber::VectorExpr *VectorExpr(std::vector<clobber::Expr *> values);

    clobber::LetExpr *LetExpr(clobber::BindingVectorExpr *bve, std::vector<clobber::Expr *> body_exprs);
    clobber::FnExpr *FnExpr(clobber::ParameterVectorExpr *pve, std::vector<clobber::Expr *> body_exprs);
    clobber::DefExpr *DefExpr(clobber::IdentifierExpr *identifier, clobber::Expr *value);
    clobber::DoExpr *DoExpr(std::vector<clobber::Expr *> body_exprs);

    clobber::CallExpr *CallExpr(clobber::Expr *operator_expr, std::vector<clobber::Expr *> arguments);
    clobber::CallExpr *CallExpr(const std::string &fn_name, std::vector<clobber::Expr *> arguments);

    clobber::accel::AccelExpr *AccelExpr(clobber::BindingVectorExpr *bve, std::vector<clobber::Expr *> body_exprs);
    clobber::accel::TOSAOpExpr *TosaOpExpr(const clobber::Token op_token, std::vector<clobber::Expr *> values);

}; // namespace SyntaxFactory