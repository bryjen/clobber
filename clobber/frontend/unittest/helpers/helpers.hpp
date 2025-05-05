#ifndef HELPERS_HPP
#define HELPERS_HPP

#include <string>
#include <vector>

struct ParserError; // clobber/parser.hpp

template <typename T> bool assert_vectors_same_size(const std::vector<T> &, const std::vector<T> &, std::string *);

std::vector<std::string> get_error_msgs(const std::string &, const std::string &, const std::vector<ParserError> &);

#endif // HELPERS_HPP