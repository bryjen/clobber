#ifndef PARSER_ERROR_HPP
#define PARSER_ERROR_HPP

struct ParserError; // clobber/parser.hpp

namespace ParserErrorFactory {
ParserError InternalErr(int span_start, int span_len);
};

namespace err = ParserErrorFactory; // alias for convenience

#endif // PARSER_ERROR_HPP