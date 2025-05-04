#include <clobber/ast.hpp>
#include <gtest/gtest.h>

class ParserTest : public ::testing::Test {
protected:
    void
    SetUp() override {
        // setup code
    }
};

TEST_F(ParserTest, ParsesCorrectly) { EXPECT_TRUE(true); }

TEST_F(ParserTest, HandlesInvalidInput) { EXPECT_FALSE(false); }