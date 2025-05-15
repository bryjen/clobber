#include "clobber/pch.hpp"

#include "clobber/ast.hpp"
#include "clobber/parser.hpp"

#include "clobber/internal/parser_error_factory.hpp"

namespace InternalErr {
    const std::string default_general_err_msg = "An internal error occurred.";
    const std::string default_err_msg         = "An unexpected error occurred.";
} // namespace InternalErr

clobber::ParserError
ParserErrorFactory::InternalErr(size_t span_start, size_t span_len) {
    return clobber::ParserError(span_start, span_len, InternalErr::default_general_err_msg, InternalErr::default_err_msg);
}

clobber::ParserError
ParserErrorFactory::InternalErr(int err_code, size_t span_start, size_t span_len) {
    std::string general_err_msg = std::format("{} ({})", InternalErr::default_general_err_msg, err_code);
    return clobber::ParserError(span_start, span_len, general_err_msg, InternalErr::default_err_msg);
}
