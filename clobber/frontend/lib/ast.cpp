#include "clobber/ast.hpp"

std::string
ClobberToken::ExtractText(const std::string &source_text) {
    return source_text.substr(this->start, this->length);
}

std::string
ClobberToken::ExtractFullText(const std::string &source_text) {
    return source_text.substr(this->full_start, this->full_length);
}

bool
ClobberToken::AreEquivalent(const ClobberToken &token1, const ClobberToken &token2) {
    if (token1.token_type != token2.token_type) {
        return false;
    }

    if (token1.value.type() != token2.value.type()) {
        return false;
    }

    if (token1.value.type() == typeid(int) && token2.value.type() == typeid(int)) {
        if (std::any_cast<int>(token1.value) == std::any_cast<int>(token2.value)) {
            return true;
        }
    }

    if (token1.value.type() == typeid(std::string) && token2.value.type() == typeid(std::string)) {
        if (std::any_cast<std::string>(token1.value) == std::any_cast<std::string>(token2.value)) {
            return true;
        }
    }

    // add further support for known types if necessary

    return false;
}