#pragma once

struct ParserError; // clobber/parser.hpp

namespace ParserErrorFactory {
ParserError InternalErr(int span_start, int span_len);
ParserError InternalErr(int error_code, int span_start, int span_len);
}; // namespace ParserErrorFactory

namespace err = ParserErrorFactory; // alias for convenience