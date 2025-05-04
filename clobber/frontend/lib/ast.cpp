#include "clobber/ast.hpp"

std::string
Token::ExtractText(const std::string &source_text) {
    return source_text.substr(this->start, this->length);
}

std::string
Token::ExtractFullText(const std::string &source_text) {
    return source_text.substr(this->full_start, this->full_length);
}