#ifndef PARSER_HPP
#define PARSER_HPP

#include <string>
#include <vector>

struct Token;           // ast.hpp
struct CompilationUnit; // ast.hpp

class ParserError {
public:
    ParserError();
    ParserError(int span_start, int span_len, const std::string &general_err_msg, const std::string &err_msg);
    ~ParserError();

private:
    int span_start;
    int span_len;
    std::string general_err_msg;
    std::string err_msg;
};

namespace clobber {
/*
 * \brief
 */
std::vector<Token> tokenize(const std::string &);

/*
 * \brief
 */
CompilationUnit parse(const std::string &source_text, const std::vector<Token> &tokens,
                      std::vector<ParserError> &out_parser_errors);
} // namespace clobber

#endif // PARSER_HPP