#include <clobber/ast.hpp>
#include <clobber/parser.hpp>
#include <format>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <vector>

class TokenizerTests : public ::testing::TestWithParam<int> {};

std::string
read_all_text(const std::string &path) {
    std::ifstream file(path, std::ios::in | std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(file)), {});
}

TEST_P(TokenizerTests, IsEven) {
    int idx;
    std::string file_path;
    std::string file_contents;
    // std::vector<Token> tokens;
    std::string str_buf;

    idx           = GetParam();
    file_path     = std::format("test_files/{}.clj", idx);
    file_contents = read_all_text(file_path);
    // tokens        = Tokenize(file_contents);
    Tokenize(file_contents);

    str_buf = std::format("[{}] file \"{}\"\n", idx, file_path);
    std::cout << str_buf;

    EXPECT_TRUE(true);
}

// Define test data
INSTANTIATE_TEST_SUITE_P(EvenValues, TokenizerTests, ::testing::Values(1, 2, 3));