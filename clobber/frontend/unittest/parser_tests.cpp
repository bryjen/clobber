
#include "helpers/helpers.hpp"
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
#include <windows.h>

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

#ifdef DISABLE_PARSER_TESTS
TEST_P(ParserTests, DISABLED_ParserTests) {
#else
TEST_P(ParserTests, ParserTests) {
#endif
    // GTEST_SKIP() << "Disabled";
    SetConsoleOutputCP(CP_UTF8);

    int idx;
    std::string file_path;
    std::string source_text;
    std::vector<ClobberToken> tokens;
    std::string str_buf;

    CompilationUnit cu;
    std::vector<ParserError> parse_errors;

    idx = GetParam();

    str_buf     = std::format("[{}]", idx);
    source_text = test_source_contents[idx];
    tokens      = clobber::tokenize(source_text);

    clobber::parse(source_text, tokens, cu);

    std::cout << std::format("[{}] exprs: {}", idx, cu.exprs.size()) << std::endl;
    std::cout << std::format("[{}] errs:  {}", idx, parse_errors.size()) << std::endl;

    if (parse_errors.size() > 0) {
        std::string file                  = "C:/USER/Documents/clobber_proj/main.clj";
        std::vector<std::string> err_msgs = get_error_msgs(file, source_text, parse_errors);
        for (size_t i = 0; i < err_msgs.size(); i++) {
            std::cout << err_msgs[i] << "\n";
        }

        std::cout << std::endl;
    }

    EXPECT_TRUE(true);
}

// Define test data
INSTANTIATE_TEST_SUITE_P(EvenValues, ParserTests, ::testing::Values(0));