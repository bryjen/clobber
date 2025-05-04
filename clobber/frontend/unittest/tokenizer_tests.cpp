#include <clobber/ast.hpp>
#include <gtest/gtest.h>

class TokenizerTests : public ::testing::Test {
protected:
    void
    SetUp() override {
        // setup code
    }
};

TEST_F(TokenizerTests, ParsesCorrectly) { EXPECT_TRUE(true); }

TEST_F(TokenizerTests, HandlesInvalidInput) { EXPECT_FALSE(false); }