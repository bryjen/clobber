#ifndef PARSER_HPP
#define PARSER_HPP

#include <string>
#include <vector>

struct Token; // ast.hpp

std::vector<Token> Tokenize(std::string &source_text);

#endif // PARSER_HPP