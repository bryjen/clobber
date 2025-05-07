#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "clobber/cli/args.hpp"

class MyParamTest : public ::testing::TestWithParam<std::vector<std::string>> {};

TEST_P(MyParamTest, HandlesParam) {
    int argc;
    char **argv;
    std::vector<std::string> cli_args; // cli args as string vector, passed via test
    Args args{};                       // the parsed arguments
    int parse_results;

    cli_args = GetParam();

    std::vector<char *> argv_vector;
    for (auto &s : cli_args) {
        argv_vector.push_back(s.data());
    }

    argc = static_cast<int>(argv_vector.size());
    argv = argv_vector.data();

    parse_results = cli11_parse_args(argc, argv, args);
    if (parse_results != 0) {
        FAIL() << "Failed to parse the input cli arguments.";
    }

    EXPECT_TRUE(true);
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    StringVecTests, 
    MyParamTest, 
    ::testing::Values(
        std::vector<std::string>{"a", "b"}, 
        std::vector<std::string>{"x", "y", "z"}
    )
);
// clang-format on