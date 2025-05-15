#pragma once

namespace clobber {
    struct ParserError; // clobber/parser.hpp
};

namespace ParserErrorFactory {
    clobber::ParserError InternalErr(size_t span_start, size_t span_len);
    clobber::ParserError InternalErr(int error_code, size_t span_start, size_t span_len);
}; // namespace ParserErrorFactory

namespace err = ParserErrorFactory; // alias for convenience