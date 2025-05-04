
#include <array>
#include <clobber/ast.hpp>
#include <clobber/parser.hpp>
#include <format>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define CONSOLE_LOG

// clang-format off
const std::array<std::string, 3> test_source_contents = {
//1.
R"(
(+ 1 2)
(* 3 4)
)",

//2.
R"(
(let [x 10
      y 5]
  (+ x y))
)",

//3.
R"(
(fn [x] (* x x))
((fn [x] (* x x)) 5)
)"
};
// clang-format on

class ParserTests : public ::testing::TestWithParam<int> {};

TEST_P(ParserTests, IsEven) {
    int idx;
    std::string file_path;
    std::string source_text;
    std::vector<Token> tokens;
    std::string str_buf;

    CompilationUnit cu;
    std::vector<ParserError> parse_errors;

    idx = GetParam();

    str_buf     = std::format("[{}]", idx);
    source_text = test_source_contents[idx];
    tokens      = clobber::tokenize(source_text);

    cu = clobber::parse(source_text, tokens, parse_errors);

    std::cout << std::format("[{}] exprs: {}", idx, cu.exprs.size()) << std::endl;

    EXPECT_TRUE(true);
}

// Define test data
INSTANTIATE_TEST_SUITE_P(EvenValues, ParserTests, ::testing::Values(0));