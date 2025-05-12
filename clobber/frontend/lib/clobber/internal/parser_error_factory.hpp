#pragma once

struct ParserError; // clobber/parser.hpp

namespace ParserErrorFactory {
ParserError InternalErr(size_t span_start, size_t span_len);
ParserError InternalErr(int error_code, size_t span_start, size_t span_len);
}; // namespace ParserErrorFactory

namespace err = ParserErrorFactory; // alias for convenience