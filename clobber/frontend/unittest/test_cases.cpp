#include "test_cases.hpp"

#include "helpers/syntax_factory.hpp"

using namespace SyntaxFactory;

// note that for tokens, the test cases were made before the the api was finalized.
// the following test cases just asserts that the tokenizer can parse the keywords properly.

const std::vector<std::string> test_cases::tokenizer::sources = {
    // 0
    R"((+ 1 2)
(* 3 4)
(+ "Hello, " "world!")
(+ 'a' 'b')
(+ 10.0f 0.2f)
(+ 10.0 0.2)
(+ 10.0d 0.2d))",

    // 1
    R"((let [x 10
      y 5]
  (+ x y)))",

    // 2
    R"((fn [x] (* x x))
((fn [x] (* x x)) 5)
)",

    // 3
    R"((do
  (def x 10)
  (def y 5)
  (+ x y)))",

    // 4
    R"((accel 
    (conv2d        input weights :stride 1 :pad 0) 
    (relu) 
    (avgpool2d     :kernel 2)))",

    // 5
    R"(
    (let [x (reshape tensor [1 224 224 3])]
        (transpose   x [0 3 1 2]) 
        (slice       x [0 0 0 0] [1 112 112 3]) 
        (concat      [x y] 3))
    )",

    // 6
    R"(
    (def fc-softmax 
        (fn [input weights bias] 
            (softmax (fully-connected input weights bias))))
    )",

    // 7
    R"((defn blend 
        [^tensor<f32,3,224,224> a 
         ^tensor<f32,3,224,224> b 
         ^f32                   alpha] 
            (add (mul a alpha) (mul b (sub 1.0 alpha))))
    )",

    // macros:
    // 8
    R"(
(defmacro defn [name params & body]
  `(def ~name
     (fn ~name
       (~params ~@body))))
    )",

    // 9
    R"(
(defmacro my-macro
  [^tensor<f32,3> x & args]
  `(let [~'val ~x
         ~@(map (fn [#'arg] `(~'arg ~arg)) args)]
     ~@args))
    )",

    // 10
    R"(
(ns my.math.core)

(defn classify-char [^char c]
  (if (= c \newline)
    "line-break"
    "other"))

(defn add-i32 [^i32 a ^i32 b]
  (+ a b))

(defn scale-f32 [^f32 x ^f32 factor]
  (* x factor))

(defn promote-i16-to-i64 [^i16 x]
  (i64-cast x))

(defn normalize-vec [^vector<f32,3> v]
  (let [sum (reduce + v)]
    (map (fn [^f32 x] (/ x sum)) v)))
    )",

    R"(
    )",
};

const std::vector<std::vector<clobber::Token>> test_cases::tokenizer::expected_tokens = {
    // clang-format off
    { // 0
        OpenParen(),
        Plus(),
        NumericLiteral(1),
        NumericLiteral(2),
        CloseParen(),

        OpenParen(),
        Asterisk(),
        NumericLiteral(3),
        NumericLiteral(4),
        CloseParen(),

        OpenParen(),
        Plus(),
        StringLiteralInsertDoubleQuot("Hello, "),
        StringLiteralInsertDoubleQuot("world!"),
        CloseParen(),

        OpenParen(),
        Plus(),
        CharLiteral('a'),
        CharLiteral('b'),
        CloseParen(),

        OpenParen(),
        Plus(),
        NumericLiteral(10.0f, 1),
        NumericLiteral(0.2f, 1),
        CloseParen(),

        OpenParen(),
        Plus(),
        NumericLiteral(10.0, 1, false),
        NumericLiteral(0.2, 1, false),
        CloseParen(),

        OpenParen(),
        Plus(),
        NumericLiteral(10.0, 1, true),
        NumericLiteral(0.2, 1, true),
        CloseParen(),

        Eof()
    },
    { // 1
        OpenParen(),
        LetKeyword(),
        OpenBracket(),
        Identifier("x"),
        NumericLiteral(10),
        Identifier("y"),
        NumericLiteral(5),
        CloseBracket(),
        OpenParen(),
        Plus(),
        Identifier("x"),
        Identifier("y"),
        CloseParen(),
        CloseParen(),
        Eof()
    },
    { // 2
        OpenParen(),
        FnKeyword(),
        OpenBracket(),
        Identifier("x"),
        CloseBracket(),
        OpenParen(),
        Asterisk(),
        Identifier("x"),
        Identifier("x"),
        CloseParen(),
        CloseParen(),

        OpenParen(),
        OpenParen(),
        FnKeyword(),
        OpenBracket(),
        Identifier("x"),
        CloseBracket(),
        OpenParen(),
        Asterisk(),
        Identifier("x"),
        Identifier("x"),
        CloseParen(),
        CloseParen(),
        NumericLiteral(5),
        CloseParen(),
        Eof()
    },
    { // 3
        OpenParen(),
        DoKeyword(),
        OpenParen(),
        DefKeyword(),
        Identifier("x"),
        NumericLiteral(10),
        CloseParen(),
        OpenParen(),
        DefKeyword(),
        Identifier("y"),
        NumericLiteral(5),
        CloseParen(),
        OpenParen(),
        Plus(),
        Identifier("x"),
        Identifier("y"),
        CloseParen(),
        CloseParen(),
        Eof()
    },
    { // 4
        OpenParen(),
        AccelKeyword(),
        OpenParen(),
        Conv2dKeyword(),
        Identifier("input"),
        Identifier("weights"),
        KeywordLiteral(":stride"),
        NumericLiteral(1),
        KeywordLiteral(":pad"),
        NumericLiteral(0),
        CloseParen(),
        OpenParen(),
        ReluKeyword(),
        CloseParen(),
        OpenParen(),
        AvgPool2dKeyword(),
        KeywordLiteral(":kernel"),
        NumericLiteral(2),
        CloseParen(),
        CloseParen(),
        Eof()
    },
    { // 5
        OpenParen(),
        LetKeyword(),
        OpenBracket(),
        Identifier("x"),
        OpenParen(),
        ReshapeKeyword(),
        TensorKeyword(),
        OpenBracket(),
        NumericLiteral(1),
        NumericLiteral(224),
        NumericLiteral(224),
        NumericLiteral(3),
        CloseBracket(),
        CloseParen(),
        CloseBracket(),

        OpenParen(),
        TransposeKeyword(),
        Identifier("x"),
        OpenBracket(),
        NumericLiteral(0),
        NumericLiteral(3),
        NumericLiteral(1),
        NumericLiteral(2),
        CloseBracket(),
        CloseParen(),
        
        OpenParen(),
        SliceKeyword(),
        Identifier("x"),
        OpenBracket(),
        NumericLiteral(0),
        NumericLiteral(0),
        NumericLiteral(0),
        NumericLiteral(0),
        CloseBracket(),
        OpenBracket(),
        NumericLiteral(1),
        NumericLiteral(112),
        NumericLiteral(112),
        NumericLiteral(3),
        CloseBracket(),
        CloseParen(),

        OpenParen(),
        ConcatKeyword(),
        OpenBracket(),
        Identifier("x"),
        Identifier("y"),
        CloseBracket(),
        NumericLiteral(3),
        CloseParen(),

        CloseParen(),
        Eof()
    },
    { // 6
        OpenParen(),
        DefKeyword(),
        Identifier("fc-softmax"),
        OpenParen(),
        FnKeyword(),
        OpenBracket(),
        Identifier("input"),
        Identifier("weights"),
        Identifier("bias"),
        CloseBracket(),
        OpenParen(),
        SoftmaxKeyword(),
        OpenParen(),
        FullyConnectedKeyword(),
        Identifier("input"),
        Identifier("weights"),
        Identifier("bias"),
        CloseParen(),
        CloseParen(),
        CloseParen(),
        CloseParen(),
        Eof()
    },
    { // 7
        OpenParen(),
        Identifier("defn"),
        Identifier("blend"),
        OpenBracket(),
        Caret(),
        TensorKeyword(),
        LessThan(),
        F32Keyword(),
        Comma(),
        NumericLiteral(3),
        Comma(),
        NumericLiteral(224),
        Comma(),
        NumericLiteral(224),
        GreaterThan(),
        Identifier("a"),
        Caret(),
        TensorKeyword(),
        LessThan(),
        F32Keyword(),
        Comma(),
        NumericLiteral(3),
        Comma(),
        NumericLiteral(224),
        Comma(),
        NumericLiteral(224),
        GreaterThan(),
        Identifier("a"),
        Caret(),
        F32Keyword(),
        Identifier("alpha"),
        CloseBracket(),
        OpenParen(),
        Identifier("add"),
        OpenParen(),
        Identifier("mul"),
        Identifier("a"),
        Identifier("alpha"),
        CloseParen(),
        OpenParen(),
        Identifier("mul"),
        Identifier("b"),
        OpenParen(),
        Identifier("sub"),
        NumericLiteral(1.0),
        Identifier("alpha"),
        CloseParen(),
        CloseParen(),
        CloseParen(),
        CloseParen(),
        Eof()
    },
    { // 8
        OpenParen(),
        DefMacroKeyword(),
        Identifier("defn"),
        OpenBracket(),
        Identifier("name"),
        Identifier("params"),
        Ampersand(),
        Identifier("body"),
        CloseBracket(),
        Backtick(),
        OpenParen(),
        DefKeyword(),
        Tilde(),
        Identifier("name"),
        OpenParen(),
        FnKeyword(),
        Tilde(),
        Identifier("name"),
        OpenParen(),
        Tilde(),
        Identifier("params"),
        TildeSplice(),
        Identifier("body"),
        CloseParen(),
        CloseParen(),
        CloseParen(),
        CloseParen(),
        Eof()
    },
    { // 9
        OpenParen(),
        DefMacroKeyword(),
        Identifier("my-macro"),
        OpenBracket(),
        Caret(),
        TensorKeyword(),
        LessThan(),
        F32Keyword(),
        Comma(),
        NumericLiteral(3),
        GreaterThan(),
        Identifier("x"),
        Ampersand(),
        Identifier("args"),
        CloseBracket(),
        Backtick(),
        OpenParen(),
        LetKeyword(),
        OpenBracket(),
        Tilde(),
        Quote(),
        Identifier("val"),
        Tilde(),
        Identifier("x"),
        TildeSplice(),
        OpenParen(),
        Identifier("map"),
        OpenParen(),
        FnKeyword(),
        OpenBracket(),
        DispatchHash(),
        Quote(),
        Identifier("arg"),
        CloseBracket(),
        Backtick(),
        OpenParen(),
        Tilde(),
        Quote(),
        Identifier("arg"),
        Tilde(),
        Identifier("arg"),
        CloseParen(),
        CloseParen(),
        Identifier("arg"),
        CloseParen(),
        CloseBracket(),
        TildeSplice(),
        Identifier("arg"),
        CloseParen(),
        CloseParen(),
        Eof()
    }, 
    { // 10
        OpenParen(),
        NsKeyword(),
        Identifier("my.math.core"),
        CloseParen(),

        OpenParen(),
        Identifier("defn"),
        Identifier("classifier-char"),
        OpenBracket(),
        Caret(),
        CharKeyword(),
        Identifier("c"),
        CloseBracket(),
        OpenParen(),
        IfKeyword(),
        OpenParen(),
        Equals(),
        Identifier("c"),
        CharLiteral('\n'),
        CloseParen(),
        StringLiteralInsertDoubleQuot("line-break"),
        StringLiteralInsertDoubleQuot("other"),
        CloseParen(),
        CloseParen(),

        OpenParen(),
        Identifier("defn"),
        Identifier("add-i32"),
        OpenBracket(),
        Caret(),
        I32Keyword(),
        Identifier("a"),
        Caret(),
        I32Keyword(),
        Identifier("b"),
        CloseBracket(),
        OpenParen(),
        Plus(),
        Identifier("a"),
        Identifier("b"),
        CloseParen(),
        CloseParen(),

        OpenParen(),
        Identifier("defn"),
        Identifier("scale-f32"),
        OpenBracket(),
        Caret(),
        F32Keyword(),
        Identifier("x"),
        Caret(),
        F32Keyword(),
        Identifier("factor"),
        CloseBracket(),
        OpenParen(),
        Asterisk(),
        Identifier("x"),
        Identifier("factor"),
        CloseParen(),
        CloseParen(),

        OpenParen(),
        Identifier("defn"),
        Identifier("promote-i16-to-i64"),
        OpenBracket(),
        Caret(),
        I16Keyword(),
        Identifier("x"),
        CloseBracket(),
        OpenParen(),
        Identifier("i64-cast"),
        Identifier("x"),
        CloseParen(),
        CloseParen(),

        OpenParen(),
        Identifier("defn"),
        Identifier("normalize-vec"),
        OpenBracket(),
        Caret(),
        VectorKeyword(),
        LessThan(),
        F32Keyword(),
        Comma(),
        NumericLiteral(3),
        GreaterThan(),
        Identifier("v"),
        CloseBracket(),
        OpenParen(),
        LetKeyword(),
        OpenBracket(),
        Identifier("sum"),
        OpenParen(),
        Identifier("reduce"),
        Plus(),
        Identifier("v"),
        CloseParen(),
        CloseBracket(),
        OpenParen(),
        Identifier("map"),
        OpenParen(),
        FnKeyword(),
        OpenBracket(),
        Caret(),
        F32Keyword(),
        Identifier("x"),
        CloseBracket(),
        OpenParen(),
        Slash(),
        Identifier("x"),
        Identifier("sum"),
        CloseParen(),
        CloseParen(),
        Identifier("v"),
        CloseParen(),
        CloseParen(),
        CloseParen(),

        Eof()
    }
    // clang-format on
};

const std::vector<std::string> test_cases::parser::sources = {
    // 0
    R"((+ 1 2)
(* 3 4)
(+ "Hello, " "world!")
(+ \a \b)
(+ 10.0f 0.2f)
(+ 10.0 0.2)
(+ 10.0d 0.2d))",

    // 1
    R"((let [x 10
      y 5]
  (+ x y)))",

    // 2
    R"((fn [x] (* x x))
((fn [x] (* x x)) 5))",

    // 3
    R"((do
  (def x 10)
  (def y 5)
  (+ x y)))",

    // 4
    R"(
(accel [x 10 y 10] 
    (relu (matmul x y :shape [3 224 224])))
    )",

    R"(
(accel [x 10 y 10] 
    (relu (matmul x y :shape [ 10 ])))
    )",

    R"(
(accel [x 10 y 10] 
    (relu (matmul (reshape x :shape [5 10]) y)))
    )",
};

const std::vector<std::vector<std::shared_ptr<clobber::Expr>>> test_cases::parser::expected_exprs = {
    // clang-format off
    { // 0
        {
            std::shared_ptr<clobber::Expr>(
                CallExpr("+", { 
                    NumLiteralExpr("1"), 
                    NumLiteralExpr("2") 
                    }
                )
            ),
            std::shared_ptr<clobber::Expr>(
                CallExpr("*", { 
                    NumLiteralExpr("3"), 
                    NumLiteralExpr("4") 
                    }
                )
            ),
            std::shared_ptr<clobber::Expr>(
                CallExpr("+", { 
                    StringLiteralExpr("Hello, "), 
                    StringLiteralExpr("world!") 
                    }
                )
            ),
            std::shared_ptr<clobber::Expr>(
                CallExpr("+", { 
                    CharLiteralExpr("\\a"), 
                    CharLiteralExpr("\\b") 
                    }
                )
            ),
            std::shared_ptr<clobber::Expr>(
                CallExpr("+", { 
                    NumLiteralExpr("10.0f"), 
                    NumLiteralExpr("0.2f") 
                    }
                )
            ),
            std::shared_ptr<clobber::Expr>(
                CallExpr("+", { 
                    NumLiteralExpr("10.0"), 
                    NumLiteralExpr("0.2") 
                    }
                )
            ),
            std::shared_ptr<clobber::Expr>(
                CallExpr("+", { 
                    NumLiteralExpr("10.0d"), 
                    NumLiteralExpr("0.2d") 
                    }
                )
            ),
        }
    },
    { // 1
        {
            std::shared_ptr<clobber::Expr>(
                LetExpr(
                    BindingVectorExpr(
                        {
                            IdentifierExpr("x"), 
                            IdentifierExpr("y")
                        },
                        {
                            NumLiteralExpr("10"), 
                            NumLiteralExpr("5")
                        }
                    ),
                    {
                        CallExpr("+", { 
                            IdentifierExpr("x"), 
                            IdentifierExpr("y")
                            }
                        )
                    }
                )
            ),
        }
    },
    { // 2
        {
            std::shared_ptr<clobber::Expr>(
                FnExpr(
                    ParameterVectorExpr({ IdentifierExpr("x") }),
                    { 
                        CallExpr("*", {
                            IdentifierExpr("x"), 
                            IdentifierExpr("x")
                            }
                        )
                    }
                )
            ),
            std::shared_ptr<clobber::Expr>(
                CallExpr(
                    FnExpr(
                        ParameterVectorExpr({ IdentifierExpr("x") }),
                        { 
                            CallExpr("*", {
                                IdentifierExpr("x"), 
                                IdentifierExpr("x")
                                }
                            )
                        }
                    ),
                    {
                        NumLiteralExpr("5")
                    }
                )
            )
        }
    },
    { // 3
        {
            std::shared_ptr<clobber::Expr>(
                DoExpr({
                    DefExpr(IdentifierExpr("x"), NumLiteralExpr("10")),
                    DefExpr(IdentifierExpr("y"), NumLiteralExpr("5")),
                    CallExpr("+", {
                        IdentifierExpr("x"), 
                        IdentifierExpr("y")
                        }
                    )
                })
            )
        }
    },
    { // 4
        {
            std::shared_ptr<clobber::Expr>(
                AccelExpr(
                    BindingVectorExpr(
                        {
                            IdentifierExpr("x"), 
                            IdentifierExpr("y")
                        },
                        {
                            NumLiteralExpr("10"), 
                            NumLiteralExpr("10")
                        }
                    ),
                    {
                        TosaOpExpr(
                            ReluKeyword(),
                            {
                                TosaOpExpr(
                                    MatmulKeyword(),
                                    {
                                        IdentifierExpr("x"),
                                        IdentifierExpr("y"),
                                        KeywordLiteralExpr("shape"),
                                        VectorExpr(
                                            {
                                                NumLiteralExpr("3"),
                                                NumLiteralExpr("224"),
                                                NumLiteralExpr("224")
                                            }
                                        )
                                    }
                                )
                            }
                        )
                    }
                )
            )
        }
    }
    // clang-format on
};