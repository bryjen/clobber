#pragma once

#include "clobber/ast.hpp"

#define DEFINE_TOKEN_FUNC(NAME)                                                                                                            \
    inline clobber::Token NAME() {                                                                                                         \
        clobber::Token token{};                                                                                                            \
        token.type = clobber::Token::Type::NAME##Token;                                                                                    \
        return token;                                                                                                                      \
    }

namespace SyntaxFactory {
    DEFINE_TOKEN_FUNC(OpenParen)
    DEFINE_TOKEN_FUNC(CloseParen)
    DEFINE_TOKEN_FUNC(OpenBracket)
    DEFINE_TOKEN_FUNC(CloseBracket)
    DEFINE_TOKEN_FUNC(OpenBrace)
    DEFINE_TOKEN_FUNC(CloseBrace)
    DEFINE_TOKEN_FUNC(Plus)
    DEFINE_TOKEN_FUNC(Minus)
    DEFINE_TOKEN_FUNC(Asterisk)
    DEFINE_TOKEN_FUNC(Slash)
    DEFINE_TOKEN_FUNC(Backslash)
    DEFINE_TOKEN_FUNC(Equals)
    DEFINE_TOKEN_FUNC(LessThan)
    DEFINE_TOKEN_FUNC(GreaterThan)

    DEFINE_TOKEN_FUNC(Quote)
    DEFINE_TOKEN_FUNC(Backtick)
    DEFINE_TOKEN_FUNC(Tilde)
    DEFINE_TOKEN_FUNC(TildeSplice)
    DEFINE_TOKEN_FUNC(DispatchHash)
    DEFINE_TOKEN_FUNC(At)
    DEFINE_TOKEN_FUNC(Ampersand)

    DEFINE_TOKEN_FUNC(Comma)
    DEFINE_TOKEN_FUNC(Caret)
    DEFINE_TOKEN_FUNC(Eof)

    DEFINE_TOKEN_FUNC(NsKeyword)
    DEFINE_TOKEN_FUNC(IfKeyword)
    DEFINE_TOKEN_FUNC(LetKeyword)
    DEFINE_TOKEN_FUNC(FnKeyword)
    DEFINE_TOKEN_FUNC(DefKeyword)
    DEFINE_TOKEN_FUNC(DefMacroKeyword)
    DEFINE_TOKEN_FUNC(DoKeyword)

    // type keywords
    DEFINE_TOKEN_FUNC(CharKeyword)
    DEFINE_TOKEN_FUNC(StringKeyword)
    DEFINE_TOKEN_FUNC(VectorKeyword)
    DEFINE_TOKEN_FUNC(I8Keyword)
    DEFINE_TOKEN_FUNC(I16Keyword)
    DEFINE_TOKEN_FUNC(I32Keyword)
    DEFINE_TOKEN_FUNC(I64Keyword)
    DEFINE_TOKEN_FUNC(F32Keyword)
    DEFINE_TOKEN_FUNC(F64Keyword)

    // hardware acceleration tokens
    DEFINE_TOKEN_FUNC(AccelKeyword)
    DEFINE_TOKEN_FUNC(TensorKeyword)
    DEFINE_TOKEN_FUNC(ReshapeKeyword)
    DEFINE_TOKEN_FUNC(TransposeKeyword)
    DEFINE_TOKEN_FUNC(TileKeyword)
    DEFINE_TOKEN_FUNC(SliceKeyword)
    DEFINE_TOKEN_FUNC(ConcatKeyword)
    DEFINE_TOKEN_FUNC(IdentityKeyword)
    DEFINE_TOKEN_FUNC(CastKeyword)
    DEFINE_TOKEN_FUNC(Conv2dKeyword)
    DEFINE_TOKEN_FUNC(DepthwiseConv2dKeyword)
    DEFINE_TOKEN_FUNC(MatmulKeyword)
    DEFINE_TOKEN_FUNC(FullyConnectedKeyword)
    DEFINE_TOKEN_FUNC(AvgPool2dKeyword)
    DEFINE_TOKEN_FUNC(MaxPool2dKeyword)
    DEFINE_TOKEN_FUNC(PadKeyword)
    DEFINE_TOKEN_FUNC(ReluKeyword)
    DEFINE_TOKEN_FUNC(SigmoidKeyword)
    DEFINE_TOKEN_FUNC(TanhKeyword)
    DEFINE_TOKEN_FUNC(SoftmaxKeyword)

    /* @brief Constructs a string literal token, inserts the double quotes into the value provided. */
    inline clobber::Token
    StringLiteralInsertDoubleQuot(std::string value) {
        clobber::Token token{};
        token.type = clobber::Token::Type::StringLiteralToken;
        return token;
    }

    inline clobber::Token
    CharLiteral(char c) {
        clobber::Token token{};
        token.type = clobber::Token::Type::CharLiteralToken;
        return token;
    }

    inline clobber::Token
    Identifier(std::string name) {
        clobber::Token token{};
        token.type = clobber::Token::Type::IdentifierToken;
        return token;
    }

    inline clobber::Token
    NumericLiteral(int value) {
        clobber::Token token{};
        token.type = clobber::Token::Type::NumericLiteralToken;
        return token;
    }

    inline clobber::Token
    NumericLiteral(float value, int decimal_places = 2) {
        clobber::Token token{};
        token.type = clobber::Token::Type::NumericLiteralToken;
        return token;
    }

    inline clobber::Token
    NumericLiteral(double value, int decimal_places = 2, bool postfix_d = false) {
        clobber::Token token{};
        token.type = clobber::Token::Type::NumericLiteralToken;
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
    clobber::NumLiteralExpr *NumLiteralExpr(int value);
    clobber::NumLiteralExpr *NumLiteralExpr(float value);
    clobber::NumLiteralExpr *NumLiteralExpr(double value);
    clobber::StringLiteralExpr *StringLiteralExpr(const std::string &value);
    clobber::CharLiteralExpr *CharLiteralExpr(char value);

    clobber::LetExpr *LetExpr(clobber::BindingVectorExpr *bve, std::vector<clobber::Expr *> body_exprs);
    clobber::FnExpr *FnExpr(clobber::ParameterVectorExpr *pve, std::vector<clobber::Expr *> body_exprs);
    clobber::DefExpr *DefExpr(clobber::IdentifierExpr *identifier, clobber::Expr *value);
    clobber::DoExpr *DoExpr(std::vector<clobber::Expr *> body_exprs);

    clobber::CallExpr *CallExpr(clobber::Expr *operator_expr, std::vector<clobber::Expr *> arguments);
    clobber::CallExpr *CallExpr(const std::string &fn_name, std::vector<clobber::Expr *> arguments);

    clobber::accel::AccelExpr *AccelExpr(clobber::BindingVectorExpr *bve, std::vector<clobber::Expr *> body_exprs);
    clobber::accel::MatMulExpr *MatMulExpr(clobber::Expr *fst_operand, clobber::Expr *snd_operand);
    clobber::accel::RelUExpr *RelUExpr(clobber::Expr *operand);

}; // namespace SyntaxFactory