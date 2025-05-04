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
#define USE_HARD_CODED_SOURCE

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

class TokenizerTests : public ::testing::TestWithParam<int> {};

std::string
read_all_text(const std::string &path) {
    std::ifstream file(path, std::ios::in | std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(file)), {});
}

TEST_P(TokenizerTests, IsEven) {
    int idx;
    std::string file_path;
    std::string source_text;
    std::vector<Token> tokens;
    std::string str_buf;

    idx = GetParam();

#ifdef USE_HARD_CODED_SOURCE
    str_buf     = std::format("[{}]", idx);
    source_text = test_source_contents[idx];
#else
    file_path   = std::format("./test_files/{}.clj", idx);
    str_buf     = std::format("[{}] file \"{}\"", idx, file_path);
    source_text = read_all_text(file_path);
#endif

    tokens = clobber::tokenize(source_text);

#ifdef CONSOLE_LOG
    std::cout << str_buf << "\n";

    std::cout << source_text << "\n";
    std::cout << std::endl;
#endif

    std::cout << tokens.size() << std::endl;

    int i;
    for (i = 0; i < tokens.size(); i++) {
        Token token = tokens[i];
        std::cout << "\t\"" << token.ExtractFullText(source_text) << "\"\n";
    }

    std::cout << "`";

    std::ostringstream builder;
    for (i = 0; i < tokens.size(); i++) {
        Token token = tokens[i];
        builder << token.ExtractFullText(source_text);
    }

    std::cout << builder.str() << "\n`" << std::endl;

    EXPECT_TRUE(true);
}

// Define test data
INSTANTIATE_TEST_SUITE_P(EvenValues, TokenizerTests, ::testing::Values(0, 1, 2));